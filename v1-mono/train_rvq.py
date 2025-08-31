#!/usr/bin/env python3
import os, time, argparse, datetime, glob, warnings, logging, math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from npq_dataset import list_npq, NPQWindowDataset, read_npq
from rvq_transformer import RVQTransformer

import torchaudio, soundfile as sf
from transformers import DacModel


# ---------------- helpers ----------------

def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def map_model_name(name: str) -> str:
    return {
        "16khz":"descript/dac_16khz",
        "24khz":"descript/dac_24khz",
        "44khz":"descript/dac_44khz",
        "44.1khz":"descript/dac_44khz",
    }.get(name.lower(), name)

def infer_global_vocab(files: List[str], K_target: int) -> List[int]:
    """Pick conservative per-codebook vocab sizes (min across a sample of files)."""
    vocab_min = None
    for fp in files[:min(32, len(files))]:
        _, hdr = read_npq(fp)
        vs = hdr["vocab_sizes"][:K_target]
        vocab_min = vs if vocab_min is None else [min(a, b) for a, b in zip(vocab_min, vs)]
    return vocab_min

def loss_with_breakdown(logits_list, targets, vocab_sizes):
    """
    Returns: mean_loss, [per_codebook_ce...]
    """
    B, T, K = targets.shape
    per = []
    total = 0.0
    for k in range(K):
        v = vocab_sizes[k]
        logits = logits_list[k].reshape(B * T, v)
        tgt = targets[..., k].reshape(B * T).clamp(min=0)
        ce = nn.functional.cross_entropy(logits, tgt, reduction="mean")
        per.append(ce)
        total += ce
    return total / K, per

def save_ckpt(path: str, model, opt, scaler, scheduler, epoch: int,
              best_val: float, cfg: dict, vocab_sizes: list, global_step: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val": float(best_val),
        "cfg": cfg,
        "vocab_sizes": vocab_sizes,
        "global_step": int(global_step),
        "saved_at": now_ts(),
    }
    torch.save(ckpt, path)
    return path

def find_autoresume(save_dir: str, prefix: str) -> Optional[str]:
    latest = os.path.join(save_dir, f"{prefix}_latest.pt")
    if os.path.isfile(latest):
        return latest
    cands = sorted(glob.glob(os.path.join(save_dir, f"{prefix}_latest.pt")))
    return cands[-1] if cands else None

def configure_quiet_mode(quiet: bool):
    if not quiet:
        return
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers.utils.logging import set_verbosity_error
        set_verbosity_error()
    except Exception:
        pass
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torchaudio").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

def banner(msg: str):
    print("=" * 8, msg, "=" * 8)

@torch.no_grad()
def generate_tokens_sliding(model: RVQTransformer, K: int, vocab_sizes, frames: int, ctx: int,
                            temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, device="cuda"):
    device = torch.device(device); model.eval().to(device)
    B = 1; prev = torch.full((B,1,K), -1, dtype=torch.long, device=device)
    out = []
    for _ in range(frames):
        if prev.shape[1] > ctx: prev = prev[:, -ctx:, :]
        logits_list, _ = model(prev)
        toks=[]
        for k in range(K):
            v = vocab_sizes[k]
            logits = logits_list[k][:,-1,:v] / max(1e-6, temperature)
            if top_k > 0:
                vals, idx = torch.topk(logits, k=min(top_k,v), dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, idx, vals); logits = mask
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cdf = torch.cumsum(probs, dim=-1)
                cutoff = (cdf > top_p).float().argmax(dim=-1, keepdim=True)
                thresh = sorted_logits.gather(1, cutoff)
                logits = torch.where(logits >= thresh, logits, torch.full_like(logits, float("-inf")))
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1)
            toks.append(idx)
        toks = torch.cat(toks, dim=1)
        out.append(toks)
        prev = torch.cat([prev, toks.unsqueeze(1)], dim=1)
    return torch.cat(out, dim=0).cpu().numpy()  # [T,K]

@torch.no_grad()
def decode_with_dac(tokens_TK, repo: str = "44khz", sr_out: int = 44100, device="cuda"):
    model = DacModel.from_pretrained(map_model_name(repo))
    device = torch.device(device); model.to(device).eval()
    codes = torch.from_numpy(tokens_TK.T).long().unsqueeze(0).to(device)  # [1,K,T]
    wav = model.decode(audio_codes=codes).audio_values.squeeze().cpu()
    model_sr = int(getattr(getattr(model,"config",None),"sampling_rate",44100))
    if model_sr != sr_out:
        wav = torchaudio.transforms.Resample(model_sr, sr_out)(wav.unsqueeze(0)).squeeze(0)
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav.numpy(), sr_out


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Train an RVQ Transformer on NPQ tokens (auto-arch resume + cosine LR + eval audio + timestamped ckpts)."
    )
    ap.add_argument("--data", nargs="+", required=True, help='NPQ files or globs, e.g. "npq_out/*.npq"')
    ap.add_argument("--ctx", type=int, default=1024, help="context frames (~11.9s at 44.1k)")
    ap.add_argument("--step", type=int, default=None, help="window stride (default: ctx//2)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50, help="train until this epoch number")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--codebooks", type=int, default=9, help="use first K codebooks")
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_dir", default="rvq_ckpts")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--amp", action="store_true", help="mixed precision")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="optional tag in checkpoint names")

    # resume/verbosity
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from (overrides auto-resume)")
    ap.add_argument("--no-resume", action="store_true", help="disable auto-resume scan")
    ap.add_argument("--ignore-arch", action="store_true", help="do NOT restore architecture from checkpoint")
    ap.add_argument("--quiet", action="store_true", help="suppress most warnings/logs")

    # cosine scheduler knobs
    ap.add_argument("--warmup-ratio", type=float, default=0.03, help="fraction of total steps for linear warmup")
    ap.add_argument("--eta-min-ratio", type=float, default=0.05, help="final LR is eta_min_ratio * base LR")
    ap.add_argument("--log-lr-steps", type=int, default=100, help="log learning rate every N steps")

    # eval-audio knobs
    ap.add_argument("--eval-audio-every", type=int, default=5, help="emit short eval WAV every N epochs (0=off)")
    ap.add_argument("--eval-frames", type=int, default=430, help="~5s @ 86 fps")
    ap.add_argument("--eval-temperature", type=float, default=1.0)
    ap.add_argument("--eval-top-k", type=int, default=0)
    ap.add_argument("--eval-top-p", type=float, default=0.0)
    ap.add_argument("--eval-outdir", default="rvq_eval")
    ap.add_argument("--eval-dac-repo", default="44khz")

    args = ap.parse_args()
    configure_quiet_mode(args.quiet)
    torch.manual_seed(args.seed)

    files = list_npq(args.data)
    if not files:
        raise SystemExit("No NPQ files matched.")

    os.makedirs(args.save_dir, exist_ok=True)

    # Naming
    prefix = f"rvq_{args.codebooks}cb"
    if args.tag:
        prefix += f"_{args.tag}"
    latest_ptr = os.path.join(args.save_dir, f"{prefix}_latest.pt")
    best_ptr   = os.path.join(args.save_dir, f"{prefix}_best.pt")

    # ---- Detect resume first, so we can restore architecture ----
    resume_path = None
    if args.resume:
        resume_path = args.resume
    elif not args.no_resume:
        resume_path = find_autoresume(args.save_dir, prefix)

    ckpt_cfg = None
    ckpt_vocab_sizes = None
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_cfg = ckpt.get("cfg", {})
        ckpt_vocab_sizes = ckpt.get("vocab_sizes", None)
        if not args.ignore_arch and ckpt_cfg:
            # Restore model-defining args from checkpoint
            old = (args.d_model, args.layers, args.heads, args.ctx, args.codebooks)
            args.d_model  = ckpt_cfg.get("d_model", args.d_model)
            args.layers   = ckpt_cfg.get("layers",  args.layers)
            args.heads    = ckpt_cfg.get("heads",   args.heads)
            args.ctx      = ckpt_cfg.get("ctx",     args.ctx)
            args.codebooks= ckpt_cfg.get("codebooks", args.codebooks)
            new = (args.d_model, args.layers, args.heads, args.ctx, args.codebooks)
            banner("ARCH FROM CHECKPOINT")
            print(f"Restored architecture: d_model={args.d_model}, layers={args.layers}, heads={args.heads}, ctx={args.ctx}, K={args.codebooks}")
            if old != new:
                print(f"(CLI arch values overridden by checkpoint; use --ignore-arch to keep CLI values.)")

    # Data (vocab sizes)
    if ckpt_vocab_sizes is not None and not args.ignore_arch:
        vocab_sizes = ckpt_vocab_sizes[:args.codebooks]
        banner("DATASET")
        print("Using vocab sizes from checkpoint:", vocab_sizes)
    else:
        vocab_sizes = infer_global_vocab(files, args.codebooks)
        banner("DATASET")
        print("Using inferred vocab sizes:", vocab_sizes)

    full = NPQWindowDataset(files, ctx_frames=args.ctx, step_frames=args.step, min_K=args.codebooks)
    n_total = len(full)
    n_val = max(1, int(n_total * args.val_split))
    n_train = max(1, n_total - n_val)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    print(f"windows: train={len(train_ds)} val={len(val_ds)} (ctx={args.ctx})")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    steps_per_epoch = max(1, len(train_dl))
    total_steps = steps_per_epoch * max(1, args.epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    cosine_steps = max(1, total_steps - warmup_steps)
    eta_min = args.lr * args.eta_min_ratio

    banner("RUNTIME")
    print(f"Device         : {args.device}")
    print(f"Auto-resume    : {'ON' if not args.no_resume and not args.resume else 'OFF' if args.no_resume else 'manual (--resume)'}")
    print(f"Save dir       : {args.save_dir}")
    print(f"Prefix         : {prefix}")
    print(f"Steps/epoch    : {steps_per_epoch}")
    print(f"Total steps    : {total_steps}  (epochs={args.epochs})")
    print(f"Warmup steps   : {warmup_steps}  ({args.warmup_ratio:.1%})")
    print(f"Cosine steps   : {cosine_steps}")
    print(f"LR base / min  : {args.lr:.6g} / {eta_min:.6g}")
    print(f"AMP mixed prec.: {'ON' if args.amp else 'OFF'}")
    print(f"Quiet mode     : {'ON' if args.quiet else 'OFF'}")
    if args.eval_audio_every > 0:
        print(f"Eval audio     : every {args.eval_audio_every} epoch(s) -> {args.eval_outdir}")

    # Device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available; using CPU.")
        device = torch.device("cpu")
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("WARNING: MPS not available; using CPU.")
        device = torch.device("cpu")

    # Model / optim
    model = RVQTransformer(vocab_sizes=vocab_sizes, d_model=args.d_model,
                           n_layer=args.layers, n_head=args.heads, max_ctx=args.ctx + 8).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = GradScaler(enabled=args.amp)

    # Scheduler
    warmup = LinearLR(opt, start_factor=0.10, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=eta_min)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # ---- Resume (load states) ----
    start_epoch = 1
    best_val = float("inf")
    global_step = 0

    if resume_path and os.path.isfile(resume_path):
        banner("RESUME")
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        try: opt.load_state_dict(ckpt["opt"])
        except Exception as e: print("  (opt state not loaded)", e)
        if args.amp and ckpt.get("scaler") is not None:
            try: scaler.load_state_dict(ckpt["scaler"])
            except Exception as e: print("  (scaler state not loaded)", e)
        try:
            if ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print("  (scheduler state not loaded)", e)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        global_step = int(ckpt.get("global_step", 0))
        print(f"  -> starting at epoch {start_epoch}, best_val={best_val:.6f}, global_step={global_step}")
    else:
        banner("STARTING NEW RUN")
        print("No resume checkpoint found; training from scratch.")

    # ---- Train ----
    banner("TRAIN")
    latest_ptr = os.path.join(args.save_dir, f"rvq_{args.codebooks}cb" + (f"_{args.tag}" if args.tag else "") + "_latest.pt")
    best_ptr   = os.path.join(args.save_dir, f"rvq_{args.codebooks}cb" + (f"_{args.tag}" if args.tag else "") + "_best.pt")
    os.makedirs(args.eval_outdir, exist_ok=True) if args.eval_audio_every > 0 else None

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            total = 0.0
            steps = 0

            for batch in train_dl:
                inputs  = batch["inputs"][:, :, :args.codebooks].to(device, non_blocking=True)
                targets = batch["targets"][:, :, :args.codebooks].to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with autocast(enabled=args.amp):
                    logits_list, _ = model(inputs)
                    loss, _ = loss_with_breakdown(logits_list, targets, vocab_sizes)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()  # advance LR every batch

                total += loss.item()
                steps += 1
                global_step += 1

                if args.log_lr_steps and (global_step % args.log_lr_steps == 0):
                    cur_lr = opt.param_groups[0]["lr"]
                    print(f"  step {global_step:>7} | lr {cur_lr:.6g} | loss {loss.item():.4f}")

            train_loss = total / max(1, steps)

            # Validate (with per-codebook CE)
            model.eval()
            vtotal = 0.0
            vsteps = 0
            percb_sum = None
            with torch.no_grad():
                for batch in val_dl:
                    inputs  = batch["inputs"][:, :, :args.codebooks].to(device, non_blocking=True)
                    targets = batch["targets"][:, :, :args.codebooks].to(device, non_blocking=True)
                    with autocast(enabled=args.amp):
                        logits_list, _ = model(inputs)
                        vloss, percb = loss_with_breakdown(logits_list, targets, vocab_sizes)
                    vtotal += vloss.item()
                    vsteps += 1
                    per_vals = [p.detach().item() for p in percb]
                    if percb_sum is None:
                        percb_sum = per_vals
                    else:
                        percb_sum = [a+b for a,b in zip(percb_sum, per_vals)]

            val_loss = vtotal / max(1, vsteps)
            ppl = math.exp(max(1e-9, val_loss))
            dt = time.time() - t0
            cur_lr = opt.param_groups[0]["lr"]
            print(f"epoch {epoch:04d} | train {train_loss:.4f} | val {val_loss:.4f} (ppl {ppl:.2f}) | {dt:.1f}s | step {global_step} | lr {cur_lr:.6g}")

            if percb_sum is not None:
                percb_avg = [s / vsteps for s in percb_sum]
                pcs = " ".join([f"cb{k}={v:.2f}" for k, v in enumerate(percb_avg)])
                print("  val CE per codebook:", pcs)

            # Save latest (always)
            _ = save_ckpt(
                path=latest_ptr,
                model=model, opt=opt, scaler=scaler if args.amp else None, scheduler=scheduler,
                epoch=epoch, best_val=best_val, cfg=vars(args),
                vocab_sizes=vocab_sizes, global_step=global_step
            )

            # Improve best?
            if val_loss < best_val:
                best_val = val_loss
                _ = save_ckpt(
                    path=best_ptr,
                    model=model, opt=opt, scaler=scaler if args.amp else None, scheduler=scheduler,
                    epoch=epoch, best_val=best_val, cfg=vars(args),
                    vocab_sizes=vocab_sizes, global_step=global_step
                )
                print("  new BEST ->", best_ptr)
                archival = os.path.join(args.save_dir, f"rvq_{args.codebooks}cb" + (f"_{args.tag}" if args.tag else "") + f"_best_e{epoch:04d}__{now_ts()}.pt")
                _ = save_ckpt(
                    path=archival,
                    model=model, opt=opt, scaler=scaler if args.amp else None, scheduler=scheduler,
                    epoch=epoch, best_val=best_val, cfg=vars(args),
                    vocab_sizes=vocab_sizes, global_step=global_step
                )
                print("  archived:", archival)

            # Emit eval audio
            if args.eval_audio_every > 0 and (epoch % args.eval_audio_every == 0):
                try:
                    print("  [eval audio] generating...")
                    K = len(vocab_sizes)
                    tokens_TK = generate_tokens_sliding(
                        model, K, vocab_sizes, frames=args.eval_frames, ctx=args.ctx,
                        temperature=args.eval_temperature, top_k=args.eval_top_k,
                        top_p=args.eval_top_p, device=args.device
                    )
                    wav, sr = decode_with_dac(tokens_TK, repo=args.eval_dac_repo, sr_out=44100, device=args.device)
                    out_path = os.path.join(args.eval_outdir, f"rvq_{args.codebooks}cb" + (f"_{args.tag}" if args.tag else "") + f"_e{epoch:04d}__{now_ts()}.wav")
                    sf.write(out_path, (wav * 32767).astype("int16"), sr, subtype="PCM_16")
                    print("  [eval audio] wrote:", out_path)
                except Exception as e:
                    print("  [eval audio] ERROR:", e)

        print("Done.")
        print("  latest:", latest_ptr)
        print("  best  :", best_ptr)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: saving latest before exit...")
        _ = save_ckpt(
            path=latest_ptr,
            model=model, opt=opt, scaler=scaler if args.amp else None, scheduler=scheduler,
            epoch=max(1, epoch if 'epoch' in locals() else 1),
            best_val=best_val, cfg=vars(args),
            vocab_sizes=vocab_sizes, global_step=global_step
        )
        print("Saved:", latest_ptr)
        raise


if __name__ == "__main__":
    main()
