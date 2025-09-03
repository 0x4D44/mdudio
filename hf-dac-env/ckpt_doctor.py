#!/usr/bin/env python3
import os, re, glob, sys, argparse, json, math, csv
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np

# Optional imports for activation probe
try:
    from rvq_transformer import RVQTransformer
    from npq_dataset import read_npq, NPQWindowDataset, list_npq
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False


def human_size(n):
    for unit in ["","K","M","B"]:
        if abs(n) < 1000: return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"

def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    meta = {
        "path": path,
        "epoch": int(ckpt.get("epoch", -1)),
        "global_step": int(ckpt.get("global_step", -1)),
        "best_val": float(ckpt.get("best_val", float("nan"))),
        "saved_at": ckpt.get("saved_at", ""),
        "cfg": ckpt.get("cfg", {}),
        "vocab_sizes": ckpt.get("vocab_sizes", []),
    }
    opt_sd = ckpt.get("opt", None)
    lrs = []
    if opt_sd and "param_groups" in opt_sd:
        for pg in opt_sd["param_groups"]:
            lrs.append(pg.get("lr", None))
    meta["lrs"] = lrs
    return ckpt, meta

def tensor_stats(t: torch.Tensor):
    t = t.detach()
    if not t.is_floating_point():
        return None
    numel = t.numel()
    finite = torch.isfinite(t)
    n_finite = finite.sum().item()
    n_nan = torch.isnan(t).sum().item()
    n_inf = (~torch.isfinite(t) & ~torch.isnan(t)).sum().item()
    absmax = t.abs().max().item() if n_finite > 0 else float("nan")
    l2 = torch.linalg.vector_norm(t.float()).item() if n_finite > 0 else float("nan")
    zero_frac = (t == 0).to(torch.float32).mean().item() if numel > 0 else 0.0
    return dict(numel=numel, finite=n_finite, n_nan=n_nan, n_inf=n_inf, absmax=absmax, l2=l2, zero_frac=zero_frac)

def weights_report(sd):
    total_l2 = 0.0
    total_params = 0
    nonfinite_tensors = []
    per_tensor = []
    for name, t in sd.items():
        if not torch.is_tensor(t): continue
        s = tensor_stats(t)
        if s is None: continue
        total_l2 += 0.0 if math.isnan(s["l2"]) else s["l2"]**2
        total_params += s["numel"]
        if s["n_nan"] or s["n_inf"]:
            nonfinite_tensors.append((name, s["n_nan"], s["n_inf"]))
        per_tensor.append((name, s))
    total_l2 = math.sqrt(total_l2)
    return dict(total_params=total_params, total_l2=total_l2,
                nonfinite=nonfinite_tensors, per_tensor=per_tensor)

def adam_state_report(opt_sd):
    out = dict(n_states=0, exp_avg_sq_max=0.0, exp_avg_sq_mean=0.0, exp_avg_max=0.0, exp_avg_mean=0.0, any_nonfinite=False)
    if not opt_sd or "state" not in opt_sd: return out
    n = 0; sum_v = 0.0; max_v = 0.0; sum_m = 0.0; max_m = 0.0
    any_nonfinite = False
    for st in opt_sd["state"].values():
        v = st.get("exp_avg_sq", None)
        m = st.get("exp_avg", None)
        if v is not None:
            v = v.detach()
            if not torch.isfinite(v).all(): any_nonfinite = True
            max_v = max(max_v, float(v.max().item()))
            sum_v += float(v.mean().item()); n += 1
        if m is not None:
            m = m.detach()
            if not torch.isfinite(m).all(): any_nonfinite = True
            max_m = max(max_m, float(m.abs().max().item()))
            sum_m += float(m.abs().mean().item())
    out.update(
        n_states=n,
        exp_avg_sq_max=max_v,
        exp_avg_sq_mean=(sum_v / max(1,n)),
        exp_avg_max=max_m,
        exp_avg_mean=(sum_m / max(1,n)),
        any_nonfinite=any_nonfinite,
    )
    return out

def diff_norm(sd_a, sd_b):
    # Compute ||theta_a - theta_b||2 over shared float tensors
    keys = set(k for k in sd_a.keys() if torch.is_tensor(sd_a[k])) & set(k for k in sd_b.keys() if torch.is_tensor(sd_b[k]))
    sq = 0.0; sa = 0.0
    for k in keys:
        a = sd_a[k]; b = sd_b[k]
        if not (a.is_floating_point() and b.is_floating_point()): continue
        da = (a.detach().float() - b.detach().float()).view(-1)
        sq += torch.dot(da, da).item()
        sa += torch.dot(a.detach().float().view(-1), a.detach().float().view(-1)).item()
    return math.sqrt(max(0.0, sq)), math.sqrt(max(0.0, sa))

def short_path(p): return os.path.basename(p)

@torch.no_grad()
def activation_probe(ckpt, meta, data_glob, device="cpu", frames=256):
    if not HAS_MODEL:
        return {"note":"rvq_transformer/npq_dataset not found; skipping probe."}
    cfg = meta["cfg"]; vocab_sizes = meta["vocab_sizes"]
    if not vocab_sizes:
        return {"note":"no vocab_sizes in checkpoint; skipping probe."}
    K = len(vocab_sizes)
    # Build model and load weights
    model = RVQTransformer(vocab_sizes=vocab_sizes, d_model=cfg["d_model"],
                           n_layer=cfg["layers"], n_head=cfg["heads"], max_ctx=cfg["ctx"]+8)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)

    # Build a tiny batch from NPQ files
    paths = list_npq([data_glob])
    if not paths:
        return {"note":f"no NPQ matched {data_glob}; skipping probe."}
    ds = NPQWindowDataset(paths[:1], ctx_frames=min(frames, cfg["ctx"]), step_frames=None, min_K=K)
    item = ds[0]
    inputs = torch.from_numpy(item["inputs"][:frames, :K]).unsqueeze(0).to(device)  # [1,T,K]
    # Forward
    logits_list, _ = model(inputs)
    report = {"finite_logits_all": True, "heads": []}
    for k, logits in enumerate(logits_list):
        # logits: [B,T,Vk]
        L = logits.float().detach().cpu()
        finite = torch.isfinite(L).all().item()
        report["finite_logits_all"] &= bool(finite)
        mu = float(L.mean().item()); sd = float(L.std().item()); mx = float(L.abs().max().item())
        # softmax entropy
        V = L.shape[-1]
        probs = torch.softmax(L[0], dim=-1)  # [T,V]
        ent = (-probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean().item()  # in nats
        report["heads"].append({"k":k, "mean":mu, "std":sd, "absmax":mx, "entropy_nats":float(ent), "vocab":int(V)})
    return report

def main():
    ap = argparse.ArgumentParser(description="Inspect RVQ checkpoints for signs of divergence/instability.")
    ap.add_argument("--ckpts", required=True, help="Directory or glob of *.pt files")
    ap.add_argument("--probe-data", default=None, help="(Optional) NPQ glob for a tiny activation/logit probe")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    ap.add_argument("--csv", default="ckpt_doctor_report.csv", help="Output CSV of summary stats")
    args = ap.parse_args()

    # Collect files
    if os.path.isdir(args.ckpts):
        files = sorted(glob.glob(os.path.join(args.ckpts, "*.pt")))
    else:
        files = sorted(glob.glob(args.ckpts))
    if not files:
        print("No checkpoints found.")
        sys.exit(1)

    print(f"Found {len(files)} checkpoint(s).")

    rows = []
    prev_sd = None
    for i, f in enumerate(files):
        ckpt, meta = load_ckpt(f)
        sd = ckpt["model"]
        wrep = weights_report(sd)
        arep = adam_state_report(ckpt.get("opt"))

        delta_norm, prev_norm = (float("nan"), float("nan"))
        if prev_sd is not None:
            delta_norm, prev_norm = diff_norm(sd, prev_sd)

        lrs = meta["lrs"]
        lr_str = ",".join([f"{x:.6g}" for x in lrs]) if lrs else "n/a"

        print("\n=== ", short_path(f), "===")
        print(f"  epoch={meta['epoch']} step={meta['global_step']} best_val={meta['best_val']:.4f} saved_at={meta['saved_at']}")
        print(f"  lr(s)={lr_str}")
        print(f"  total_params={human_size(wrep['total_params'])} total_L2={wrep['total_l2']:.3e}")
        if not math.isnan(delta_norm):
            rel = delta_norm / (prev_norm + 1e-12)
            print(f"  Δθ L2 vs prev = {delta_norm:.3e} (relative {rel:.3e})")
        if wrep["nonfinite"]:
            print("  NON-FINITE tensors detected:")
            for name, n_nan, n_inf in wrep["nonfinite"]:
                print(f"    {name}: nan={n_nan} inf={n_inf}")
        print(f"  Adam exp_avg_sq: mean={arep['exp_avg_sq_mean']:.3e} max={arep['exp_avg_sq_max']:.3e} | exp_avg abs mean={arep['exp_avg_mean']:.3e} max={arep['exp_avg_max']:.3e} | nonfinite={arep['any_nonfinite']}")

        # Optional probe
        probe = {}
        if args.probe_data:
            dev = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
            try:
                probe = activation_probe(ckpt, meta, args.probe_data, device=dev, frames=min(256, meta["cfg"].get("ctx", 1024)))
                if "heads" in probe:
                    finite = probe.get("finite_logits_all", True)
                    ents = [h["entropy_nats"] for h in probe["heads"]]
                    print(f"  Probe logits finite={finite}; avg entropy per head: {[round(e,2) for e in ents]}")
                else:
                    print(f"  Probe: {probe.get('note','(no info)')}")
            except Exception as e:
                print(f"  Probe ERROR: {e}")

        # Summarize per-layer zero fractions briefly (top 3 biggest)
        zero_fracs = []
        for name, s in wrep["per_tensor"]:
            if s["numel"] >= 1024:  # ignore tiny tensors
                zero_fracs.append((name, s["zero_frac"]))
        zero_fracs.sort(key=lambda x: x[1], reverse=True)
        top_zeros = "; ".join([f"{n}:{zf*100:.1f}%" for n, zf in zero_fracs[:3]])

        rows.append({
            "path": short_path(f),
            "epoch": meta["epoch"],
            "step": meta["global_step"],
            "best_val": meta["best_val"],
            "saved_at": meta["saved_at"],
            "lr": lrs[0] if lrs else float("nan"),
            "total_params": wrep["total_params"],
            "total_L2": wrep["total_l2"],
            "delta_L2": delta_norm,
            "rel_delta": (delta_norm / (prev_norm + 1e-9)) if not math.isnan(delta_norm) else float("nan"),
            "adam_v_mean": arep["exp_avg_sq_mean"],
            "adam_v_max": arep["exp_avg_sq_max"],
            "adam_m_mean": arep["exp_avg_mean"],
            "adam_m_max": arep["exp_avg_max"],
            "adam_nonfinite": int(arep["any_nonfinite"]),
            "nonfinite_tensors": len(wrep["nonfinite"]),
            "top_zero_fracs": top_zeros,
            "probe_note": probe.get("note",""),
            "probe_finite_logits": probe.get("finite_logits_all", None),
        })

        prev_sd = sd

    # Write CSV
    csv_path = args.csv
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote summary CSV: {csv_path}")
    print("Tip: plot epoch vs best_val, delta_L2, and adam_v_max to spot spikes.")
    print("Heuristics: nonfinite_tensors>0, adam_v_max ↑↑, or huge rel_delta ⇒ likely divergence around that checkpoint.")
    

if __name__ == "__main__":
    main()
