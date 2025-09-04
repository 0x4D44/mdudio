#!/usr/bin/env python3
import os, time, argparse, datetime, glob, warnings, logging, math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch import amp

from npq_dataset import list_npq, NPQWindowDataset, read_npq_header, validate_npq_files
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
    vocab_min = None
    for fp in files[:min(64, len(files))]:
        try:
            hdr, _ = read_npq_header(fp)
        except Exception:
            continue
        vs = hdr["vocab_sizes"][:K_target]
        vocab_min = vs if vocab_min is None else [min(a, b) for a, b in zip(vocab_min, vs)]
    if vocab_min is None:
        raise RuntimeError("Could not infer vocab sizes: no valid NPQ headers.")
    return vocab_min

def parse_cb_weights(spec: str, K: int, alpha: float) -> List[float]:
    if spec.lower() == "auto":
        return [1.0 + alpha * (k / max(1, K - 1)) for k in range(K)]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if len(vals) != K:
        raise ValueError(f"--cb-weights expects {K} values, got {len(vals)}")
    return vals

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

def save_ckpt(path: str, model, opt, scaler, scheduler, epoch: int,
              best_val: float, cfg: dict, vocab_sizes: list, global_step: int,
              ema_state_dict=None):
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
        "ema": ema_state_dict,
    }
    torch.save(ckpt, path)
    return path

def find_autoresume(save_dir: str, prefix: str) -> Optional[str]:
    latest = os.path.join(save_dir, f"{prefix}_latest.pt")
    if os.path.isfile(latest):
        return latest
    cands = sorted(glob.glob(os.path.join(save_dir, f"{prefix}_latest.pt")))
    return cands[-1] if cands else None

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.params = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.data.is_floating_point():
                    self.params[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.params and p.requires_grad and p.data.is_floating_point():
                self.params[n].mul_(d).add_(p.data, alpha=1.0 - d)

    def to_state_dict(self):
        return {k: v.cpu() for k, v in self.params.items()}

    @torch.no_grad()
    def load_state_dict(self, sd: dict, device):
        self.params = {k: v.to(device) for k, v in sd.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.params and p.data.is_floating_point():
                p.data.copy_(self.params[n].to(p.device))


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
                mask = torch.full_like(logits, float("-inf")); mask.scatter_(1, idx, vals); logits = mask
            if 0.0 < top_p < 1.0:
                sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
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
        description="Train RVQ Transformer on NPQ tokens (SDPA, EMA, smoothing, CB weights, grad-accum, checkpointing)."
    )
    ap.add_argument("--data", nargs="+", required=True, help='NPQ globs, e.g. "npq_out/*.npq"')
    ap.add_argument("--val-list", default=None, help="Optional text file with NPQ paths for validation set (one per line)")

    # Model/sequence
    ap.add_argument("--ctx", type=int, default=1024, help="context frames")
    ap.add_argument("--step", type=int, default=None, help="window stride (default: ctx//2)")
    ap.add_argument("--codebooks", type=int, default=9)

    # Capacity
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--batch", type=int, default=12, help="micro-batch size per step")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu","mps"])
    ap.add_argument("--amp", action="store_true", help="mixed precision")
    ap.add_argument("--grad-checkpoint", action="store_true", help="enable gradient checkpointing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default="rvq_ckpts_big")
    ap.add_argument("--tag", default="", help="name tag")

    # Resume/arch
    ap.add_argument("--resume", default=None, help="checkpoint to resume (overrides auto-resume)")
    ap.add_argument("--no-resume", action="store_true", help="disable auto-resume scan")
    ap.add_argument("--ignore-arch", dest="ignore_arch", action="store_true",
                    help="do NOT restore architecture from checkpoint")
    ap.add_argument("--quiet", action="store_true")

    # Cosine scheduler
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--eta-min-ratio", type=float, default=0.05)
    ap.add_argument("--log-lr-steps", type=int, default=100)

    # EMA + smoothing + CB weights
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 to disable)")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--cb-weights", type=str, default="auto", help='"auto" or comma list of K floats')
    ap.add_argument("--cb-weights-auto-alpha", type=float, default=0.5, help="alpha for auto ramp (coarse->fine)")

    # Stability guards
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--outlier-skip-factor", type=float, default=5.0)
    ap.add_argument("--outlier-ema-beta", type=float, default=0.98)
    ap.add_argument("--panic-lr-mul", type=float, default=0.5)
    ap.add_argument("--panic-snapshot", action="store_true")

    # Eval audio
    ap.add_argument("--eval-audio-every", type=int, default=5)
    ap.add_argument("--eval-frames", type=int, default=1290)  # ~15 s
    ap.add_argument("--eval-temperature", type=float, default=1.1)
    ap.add_argument("--eval-top-k", type=int, default=0)
    ap.add_argument("--eval-top-p", type=float, default=0.0)
    ap.add_argument("--eval-outdir", default="rvq_eval")
    ap.add_argument("--eval-dac-repo", default="44khz")
    ap.add_argument("--eval-on-cpu", action="store_true")

    # Archival
    ap.add_argument("--archive-every", type=int, default=10)
    ap.add_argument("--keep-archived", type=int, default=20)

    args = ap.parse_args()
    configure_quiet_mode(args.quiet)
    torch.manual_seed(args.seed)

    # Enable TF32 for speed/stability on CUDA (Ada/4070Ti etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    files = list_npq(args.data)
    if not files:
        raise SystemExit("No NPQ files matched.")

    # Validate and filter NPQs
    good, bad = validate_npq_files(files, expect_K=args.codebooks)
    if bad:
        print(f"WARNING: {len(bad)} NPQ file(s) excluded due to errors:")
        for p, reason in bad[:10]:
            print(f"  - {p}: {reason}")
        if len(bad) > 10:
            print(f"  ... and {len(bad)-10} more.")
    files = good
    if not files:
        raise SystemExit("No valid NPQ files remain after validation.")

    os.makedirs(args.save_dir, exist_ok=True)

    # Pointers
    prefix = f"rvq_{args.codebooks}cb" + (f"_{args.tag}" if args.tag else "")
    latest_ptr = os.path.join(args.save_dir, f"{prefix}_latest.pt")
    best_ptr   = os.path.join(args.save_dir, f"{prefix}_best.pt")

    # Resume
    resume_path = args.resume if args.resume else (find_autoresume(args.save_dir, prefix) if not args.no_resume else None)
    ckpt_cfg = None; ckpt_vocab = None; ckpt_ema_state = None
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_cfg = ckpt.get("cfg", {})
        ckpt_vocab = ckpt.get("vocab_sizes", None)
        ckpt_ema_state = ckpt.get("ema", None)
        if not args.ignore_arch and ckpt_cfg:
            args.d_model   = ckpt_cfg.get("d_model",  args.d_model)
            args.layers    = ckpt_cfg.get("layers",   args.layers)
            args.heads     = ckpt_cfg.get("heads",    args.heads)
            args.ctx       = ckpt_cfg.get("ctx",      args.ctx)
            args.codebooks = ckpt_cfg.get("codebooks", args.codebooks)
            args.dropout   = ckpt_cfg.get("dropout",  args.dropout)
            print("======== ARCH FROM CHECKPOINT ========")
            print(f"d_model={args.d_model}, layers={args.layers}, heads={args.heads}, ctx={args.ctx}, K={args.codebooks}, dropout={args.dropout}")

    # Data / vocab sizes
    vocab_sizes = (ckpt_vocab[:args.codebooks] if (ckpt_vocab and not args.ignore_arch) else infer_global_vocab(files, args.codebooks))
    print("======== DATASET ========")
    print("Vocab sizes:", vocab_sizes)

    # Build datasets
    if args.val_list and os.path.isfile(args.val_list):
        with open(args.val_list, "r") as fp:
            val_set_paths = [ln.strip() for ln in fp if ln.strip()]
        val_set_paths = [p for p in val_set_paths if os.path.exists(p)]
        train_files = [p for p in files if p not in set(val_set_paths)]
        val_files = val_set_paths
        train_ds = NPQWindowDataset(train_files, ctx_frames=args.ctx, step_frames=args.step, min_K=args.codebooks)
        val_ds   = NPQWindowDataset(val_files,   ctx_frames=args.ctx, step_frames=args.step, min_K=args.codebooks)
    else:
        full_ds = NPQWindowDataset(files, ctx_frames=args.ctx, step_frames=args.step, min_K=args.codebooks)
        n_total = len(full_ds)
        n_val = max(1, int(n_total * args.val_split))
        n_train = max(1, n_total - n_val)
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    print(f"windows: train={len(train_ds)} val={len(val_ds)} (ctx={args.ctx}, step={args.step or args.ctx//2})")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    steps_per_epoch = max(1, len(train_dl) // max(1, args.accum))  # optimizer steps per epoch
    total_steps = steps_per_epoch * max(1, args.epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    cosine_steps = max(1, total_steps - warmup_steps)
    eta_min = args.lr * args.eta_min_ratio

    print("======== RUNTIME ========")
    print(f"Device       : {args.device}")
    print(f"Save dir     : {args.save_dir}")
    print(f"Prefix       : {prefix}")
    print(f"Steps/epoch  : {steps_per_epoch} (optimizer steps; accum={args.accum}, micro-batch={args.batch})")
    print(f"Total steps  : {total_steps}  (epochs={args.epochs})")
    print(f"Warmup steps : {warmup_steps}  ({args.warmup_ratio:.1%})")
    print(f"Cosine steps : {cosine_steps}")
    print(f"LR base/min  : {args.lr:.6g} / {eta_min:.6g}")
    print(f"EMA          : {args.ema if args.ema>0 else 'OFF'}")
    print(f"Label smooth : {args.label_smoothing}")
    print(f"CB weights   : {args.cb_weights} (alpha={args.cb_weights_auto_alpha})")
    print(f"Grad clip    : {args.grad_clip}")
    if args.grad_checkpoint:
        print("Gradient checkpointing: ON")

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
                           n_layer=args.layers, n_head=args.heads,
                           max_ctx=args.ctx + 8, dropout=args.dropout,
                           use_checkpoint=args.grad_checkpoint).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # EMA model
    ema = None
    if args.ema > 0.0:
        ema = EMA(model, decay=args.ema)
        if resume_path and os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location="cpu")
            if ckpt.get("ema") is not None:
                ema.load_state_dict(ckpt["ema"], device=device)

    # Scheduler
    warmup = LinearLR(opt, start_factor=0.10, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=eta_min)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # Resume states
    start_epoch = 1
    best_val = float("inf")
    global_step = 0
    if resume_path and os.path.isfile(resume_path):
        print("======== RESUME ========")
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
        print("======== STARTING NEW RUN ========")
        print("Training from scratch.")

    # Loss helpers
    K = len(vocab_sizes)
    cb_weights = parse_cb_weights(args.cb_weights, K, args.cb_weights_auto_alpha)
    cb_weights_t = torch.tensor(cb_weights, device=device, dtype=torch.float32)

    def loss_with_breakdown(logits_list, targets):
        B, T, K_ = targets.shape
        assert K_ == K
        per = []
        for k in range(K):
            v = vocab_sizes[k]
            logits = logits_list[k].reshape(B * T, v)
            tgt = targets[..., k].reshape(B * T).clamp_min(0)
            ce = nn.functional.cross_entropy(
                logits, tgt, reduction="mean", label_smoothing=args.label_smoothing
            )
            per.append(ce)
        per_stacked = torch.stack(per)  # [K]
        weighted = (per_stacked * cb_weights_t).sum() / cb_weights_t.sum()
        return weighted, per_stacked

    # ---- Train ----
    print("======== TRAIN ========")
    os.makedirs(args.eval_outdir, exist_ok=True) if args.eval_audio_every > 0 else None
    ema_loss = None

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            total_raw = 0.0  # sum of unscaled (pre-accum) losses
            eff_steps = 0    # optimizer steps in this epoch

            opt.zero_grad(set_to_none=True)
            micro_idx = 0

            for batch in train_dl:
                inputs  = batch["inputs"][:, :, :K].to(device, non_blocking=True)
                targets = batch["targets"][:, :, :K].to(device, non_blocking=True)

                with amp.autocast(device_type="cuda", enabled=args.amp):
                    logits_list, _ = model(inputs)
                    loss_raw, _ = loss_with_breakdown(logits_list, targets)  # unscaled
                    loss = loss_raw / max(1, args.accum)

                # Outlier skip based on raw loss
                if args.outlier_skip_factor > 0:
                    cur = float(loss_raw.detach().item())
                    ema_loss = cur if ema_loss is None else (args.outlier_ema_beta * ema_loss + (1 - args.outlier_ema_beta) * cur)
                    if ema_loss is not None and cur > args.outlier_skip_factor * max(1e-6, ema_loss):
                        print(f"  [SKIP] outlier batch: loss {cur:.3f} > {args.outlier_skip_factor:.1f}× EMA {ema_loss:.3f}")
                        continue

                if not torch.isfinite(loss_raw):
                    print(f"  [PANIC] non-finite loss at step {global_step}. Rolling back.")
                    # Reload last good (best if exists else latest)
                    reload_from = best_ptr if os.path.exists(best_ptr) else latest_ptr
                    if os.path.exists(reload_from):
                        ckpt = torch.load(reload_from, map_location="cpu")
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
                        # LR haircut
                        for g in opt.param_groups:
                            g["lr"] = max(g["lr"] * args.panic_lr_mul, args.lr * args.eta_min_ratio)
                        print("  [PANIC] reloaded from:", reload_from, " new lr:", opt.param_groups[0]["lr"])
                    continue

                scaler.scale(loss).backward()
                micro_idx += 1
                total_raw += float(loss_raw.detach().item())

                if (micro_idx % max(1, args.accum)) == 0:
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    if ema: ema.update(model)
                    scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                    eff_steps += 1

                    if args.log_lr_steps and (global_step % args.log_lr_steps == 0):
                        cur_lr = opt.param_groups[0]["lr"]
                        print(f"  step {global_step:>7} | lr {cur_lr:.6g} | loss {loss_raw.detach().item():.4f}")

            # Flush leftover micro-batches
            if (micro_idx % max(1, args.accum)) != 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(opt)
                scaler.update()
                if ema: ema.update(model)
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                eff_steps += 1

            train_loss = total_raw / max(1, eff_steps)

            # ---- Validation (EMA weights if present) ----
            eval_model = model
            backup_sd = None
            if ema:
                backup_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
                eval_model = model
                eval_model.eval()

            vtotal = 0.0
            vsteps = 0
            percb_sum = None
            with torch.no_grad():
                for batch in val_dl:
                    inputs  = batch["inputs"][:, :, :K].to(device, non_blocking=True)
                    targets = batch["targets"][:, :, :K].to(device, non_blocking=True)
                    with amp.autocast(device_type="cuda", enabled=args.amp):
                        logits_list, _ = eval_model(inputs)
                        vloss, percb = loss_with_breakdown(logits_list, targets)
                    vtotal += float(vloss.detach().item())
                    vsteps += 1
                    per_vals = [float(p.detach().item()) for p in percb]
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

            # Restore training weights if EMA temporarily applied
            if ema and backup_sd is not None:
                model.load_state_dict(backup_sd)

            # ---- Save latest ----
            _ = save_ckpt(
                path=latest_ptr,
                model=model, opt=opt, scaler=scaler if args.amp else None, scheduler=scheduler,
                epoch=epoch, best_val=best_val, cfg=vars(args),
                vocab_sizes=vocab_sizes, global_step=global_step,
                ema_state_dict=(ema.to_state_dict() if ema else None)
            )

            # Periodic archival
            if args.archive_every > 0 and (epoch % args.archive_every == 0):
                archival = os.path.join(args.save_dir, f"{prefix}_epoch_e{epoch:04d}__{now_ts()}.pt")
                _ = save_ckpt(
                    path=archival,
                    model=model, opt=opt, scaler=scaler if args.amp else None, scheduler=scheduler,
                    epoch=epoch, best_val=best_val, cfg=vars(args),
                    vocab_sizes=vocab_sizes, global_step=global_step,
                    ema_state_dict=(ema.to_state_dict() if ema else None)
                )
                # prune old
                pat = os.path.join(args.save_dir, f"{prefix}_epoch_e*.pt")
                files_arch = sorted(glob.glob(pat))
                if len(files_arch) > args.keep_archived:
                    for pth in files_arch[: len(files_arch) - args.keep_archived]:
                        try: os.remove(pth)
                        except OSError: pass
                print("  archived epoch:", archival)

            # Improve best? (compare on EMA val)
            if val_loss < best_val:
                best_val = val_loss
                # Save BEST using EMA weights if available
                if ema:
                    backup_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    ema.copy_to(model)
                _ = save_ckpt(
                    path=best_ptr,
                    model=model, opt=opt, scaler=scaler if args.amp else None, scheduler=scheduler,
                    epoch=epoch, best_val=best_val, cfg=vars(args),
                    vocab_sizes=vocab_sizes, global_step=global_step,
                    ema_state_dict=(ema.to_state_dict() if ema else None)
                )
                if ema and backup_sd is not None:
                    model.load_state_dict(backup_sd)
                print("  new BEST ->", best_ptr)
                archival = os.path.join(args.save_dir, f"{prefix}_best_e{epoch:04d}__{now_ts()}.pt")
                _ = save_ckpt(
                    path=archival,
                    model=model, opt=opt, scaler=scaler if args.amp else None, scheduler=scheduler,
                    epoch=epoch, best_val=best_val, cfg=vars(args),
                    vocab_sizes=vocab_sizes, global_step=global_step,
                    ema_state_dict=(ema.to_state_dict() if ema else None)
                )
                print("  archived best:", archival)

            # ---- Eval audio (EMA) ----
            if args.eval_audio_every > 0 and (epoch % args.eval_audio_every == 0):
                try:
                    print("  [eval audio] generating...")
                    if ema:
                        backup_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        ema.copy_to(model)
                        model.eval()
                    tokens_TK = generate_tokens_sliding(
                        model, K, vocab_sizes, frames=args.eval_frames, ctx=args.ctx,
                        temperature=args.eval_temperature, top_k=args.eval_top_k,
                        top_p=args.eval_top_p, device=args.device
                    )
                    dac_dev = "cpu" if args.eval_on_cpu else args.device
                    wav, sr = decode_with_dac(tokens_TK, repo=args.eval_dac_repo, sr_out=44100, device=dac_dev)
                    out_path = os.path.join(args.eval_outdir, f"{prefix}_e{epoch:04d}__{now_ts()}.wav")
                    sf.write(out_path, (wav * 32767).astype("int16"), sr, subtype="PCM_16")
                    print("  [eval audio] wrote:", out_path)
                    if ema and backup_sd is not None:
                        model.load_state_dict(backup_sd)
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
            vocab_sizes=vocab_sizes, global_step=global_step,
            ema_state_dict=(ema.to_state_dict() if ema else None)
        )
        print("Saved:", latest_ptr)
        raise


if __name__ == "__main__":
    main()
