# RVQ Training Runs — Summary & Notes

Last updated: 2025-09-04

## TL;DR

We've run three distinct training attempts:
1. Run #1 - small/base model (defaults), `ctx≈1280` (~15s): learned, then diverged hard around epoch ~355.
2. Run #2 - bigger model, more data, `ctx=1280`: trained multi-day, reached val≈4.19 then plateaued.
3. Run #3 - "fast" run, shorter `ctx=512`, same big model class + speedups: currently running; build/compile issues fixed; this is the speed-optimized path we’ll iterate on next.

Below are configs, commands, events, and lessons.
---
## Run #1 — Initial attempt (small/base)

> Goal: Prove pipeline works end-to-end on mono 44.1kHz / 9 codebooks.

### Approx model (defaults at the time):
- Codebooks: K=9 (vocab 1024 each)
- Transformer: **d_model≈512, layers≈12, heads≈8**
- Context: **ctx≈1280 frames (~15s @ ~85–90 fps)**
- Loss: CE per codebook, label smoothing (later)
- Optim: AdamW, cosine LR (later), no EMA initially

### Command (representative):
```python
python train_rvq.py --data "npq_out/*.npq" \
  --ctx 1280 --batch 8 --epochs 300 --amp \
  --save_dir rvq_ckpts \
  --eval-audio-every 5 --eval-frames 1290 --eval-temperature 1.1 \
  --eval-outdir rvq_eval --log-lr-steps 50 --quiet
```

### Key events:
- Early/ mid training looked sane (val ~4.5).
- Later diverged catastrophically (e.g., epoch ~355: val loss exploded; codebook CE spiking; ppl astronomically large).
- Root cause likely: combination of no EMA, no outlier skip, and occasional unstable batches with long context and a relatively small model.

> Outcome: Abandoned run. Introduced a checkpoint doctor script to analyze divergence and added stability guards (EMA, grad clip, outlier skip) for future runs.

### Lessons:
- Add EMA, grad clipping, outlier-batch skip, cosine LR with warmup, and periodic archiving from day one.
- Keep a small eval sample frequency and archive “best” checkpoints with timestamp for regression tracking.
---
## Run #2 — Bigger model + much more data (long multi-day)

> Goal: Improve capacity + data scale, still ctx=1280 (~15s), stabilize training.

### Model:
- Codebooks: K=9 (1024 each)
- Transformer: **d_model=768, layers=16, heads=12, dropout=0.1**
- Context: **ctx=1280, step=640**
- Optim: AdamW (lr=2e-4, wd=0.02), warmup+cosine, EMA=0.999, label_smoothing=0.05
- Stability: grad_clip=1.0, outlier skip guard, checkpoint doctor
- Data: started ~15h, then scaled up to ~75 hours of NPQ

### Representative command (finalized):
```python
python train_rvq.py --data "npq_out/*.npq" \
  --ctx 1280 --batch 4 --accum 6 \
  --epochs 350 --amp --device cuda \
  --d_model 768 --layers 16 --heads 12 --dropout 0.1 \
  --lr 2e-4 --weight-decay 0.02 \
  --ema 0.999 --label-smoothing 0.05 --cb-weights auto --cb-weights-auto-alpha 0.5 \
  --eval-audio-every 5 --eval-frames 1290 --eval-temperature 1.1 \
  --save_dir rvq_ckpts_big --log-lr-steps 50 --archive-every 10 --keep-archived 20
```

### Runtime snapshot (from logs):

- Dataset windows: train=34520, val=1816 (ctx=1280, step=640)
- Steps/epoch: 1438 (optimizer steps; accum=6, micro-batch=4)
- Epoch time: ~20–21 min (e.g., 1217.1s), total run multi-day
- Best val: **≈4.1932 @ epoch 120 (ppl~66)**. Plateau around val ≈4.19–4.20 thereafter.

### Key events:
- After fixes (NPQ header parsing, stability guards), training ran stably for ~2.5 days.
- Improvement slowed; no new best after epoch ~120 despite continued training (val CE per codebook stalled, especially higher CBs).

> Outcome: Plateaued. Good stability; quality mediocre. We decided to reduce context and optimize throughput next.

### Lessons:
- Long context hurts throughput and increases instability/variance per step.
- Bigger model helped but diminishing returns with ctx=1280; likely capacity is not the only bottleneck.
- Data scale helps, but optimizer schedule and faster iteration are key to exploring hyperparameters.
---
## Run #3 - "Fast path": shorter ctx + system speedups

> Goal: Speed up iteration while keeping capacity; reduce ctx to 512 (≈6s) & add runtime optimizations.

### Model & runtime:
- Codebooks: K=9 (1024 each)
- Transformer: **d_model=768, layers=16, heads=12, dropout=0.1**
- Context: **ctx=512, step=256 (50% overlap)**
- Optim: AdamW (lr=2e-4, wd=0.02), warmup+cosine, EMA=0.999, label_smoothing=0.05
- Stability: grad_clip=1.0, outlier skip
- Speedups: fused AdamW, `torch.compile`, AMP, TF32, persistent workers, prefetch, CUDA graphs step boundary, modern SDPA
- Eval: short sample generation every 5 epochs (DAC decode on CPU to avoid VRAM spikes)

### Command:
```python
python train_rvq.py --data "npq_out/*.npq" \
  --ctx 512 --step 256 \
  --batch 6 --accum 6 \
  --epochs 250 --amp --device cuda \
  --d_model 768 --layers 16 --heads 12 --dropout 0.1 \
  --lr 2e-4 --weight-decay 0.02 \
  --fused-optim --compile \
  --ema 0.999 --label-smoothing 0.05 \
  --cb-weights auto --cb-weights-auto-alpha 0.5 \
  --val-every 1 \
  --eval-audio-every 5 --eval-on-cpu \
  --save_dir rvq_ckpts_fast --tag base512 --log-lr-steps 50 \
  --persistent-workers --prefetch 6 \
  --quiet
```

### Dataset/runtime snapshot:
- Vocab sizes: `[1024 × 9]`
- Windows: train=62640, val=3296 (ctx=512, step=256)
- Steps/epoch: 1740 (optimizer steps; accum=6, micro-batch=6)
- LR schedule: warmup 1%, cosine to eta_min=1e-5

### Key events & fixes:
- OOM earlier → fixed via lower micro-batch, accum, AMP, TF32.
- `torch.compile` crash due to SDPA API mismatch → replaced with version-proof `sdpa_kernel` wrapper + `cudagraph_mark_step_begin()` before each forward.
- Deprecated AMP/SDPA warnings → migrated to `torch.amp.autocast` / `torch.amp.GradScaler` and quiet mode.
- Now training starts cleanly with speedups; this run is our current baseline for rapid iteration and ablations.

> Outcome: Running with improved throughput; will compare learning curves vs Run #2 with similar batch-equivalent.

### Lessons:
- Shorter context accelerates training & reduces variance.
- `torch.compile` can help, but be strict about SDPA and CUDA-graphs step boundaries.
- Keep eval decode on CPU to avoid VRAM spikes mid-epoch.

### What we changed in the codebase (high-impact)
- Stability: EMA(0.999), grad clip=1.0, outlier skip (ema-based), label smoothing=0.05
- Scheduler: Warmup + cosine (with printed LR every N steps)
- Checkpoints: auto-resume, latest/best + timestamped archives, safe “doctor” inspection
- Logging: per-epoch wall-clock & total elapsed, LR logs, optional quiet mode
- Speed: fused AdamW (CUDA), AMP/TF32, persistent workers, prefetch, optional `torch.compile` with cudagraph_mark_step_begin()
- Attention: modern SDPA kernel selection compatible with compile (and old API fallback)
- Eval: optional periodic audio samples (EMA weights), DAC decode on CPU
- Data: Robust NPQ header parsing/validation; global vocab inference

### Interpreting metrics (quick refresher)
- `train`: micro-batch averaged training loss this epoch.
- `val`: validation loss (typically with EMA weights).
- `ppl`: `exp(val_loss)` — a rough perplexity proxy; lower is better.
- `val CE per codebook`: cross-entropy per quantizer; low-index CBs should drop earlier; high-index CBs tend to be harder.

## Lessons learned (so far)
1. Divergence can happen late; guard with EMA, clip, outlier skip, and keep best checkpoints frequently archived.
2. Context length is a major throughput lever — shortening to 512 made iteration faster without discarding capacity.
3. Bigger model (768/16/12) helped vs tiny, but data scale and schedule matter as much; consider balanced CB weights (`auto`) to stabilize high CBs.
4. Tooling matters: a checkpoint doctor saved time diagnosing the blow-up.
5. `torch.compile` is useful but requires correct SDPA context and CUDA-graphs step begin hook.

### Recommendations for the next runs
- **Stick with Run #3 skeleton** (ctx=512) to iterate faster.
- **Try slightly larger capacity** once stable (e.g., 896/20/14) or keep the current model and:
- Increase **effective batch** (if VRAM allows, via accum).
- Tune **LR** (e.g., 1.5e-4 to 3e-4) and warmup (1–3%).
- Adjust **CB weights** (e.g., `auto` alpha 0.3–0.7) to help higher CBs.
- Continue to **grow data** (diversity helps reconstruction realism).
- Consider **curriculum**: start ctx=512, later fine-tune at ctx=768–1024 if you need longer structure.
- Keep **eval audio** every 5–10 epochs for qualitative tracking.
---
# Appendix — Commands (copy-paste)

## Run #2 (big model, long context; plateaued)
```python
python train_rvq.py --data "npq_out/*.npq" \
  --ctx 1280 --step 640 \
  --batch 4 --accum 6 \
  --epochs 350 --amp --device cuda \
  --d_model 768 --layers 16 --heads 12 --dropout 0.1 \
  --lr 2e-4 --weight-decay 0.02 \
  --ema 0.999 --label-smoothing 0.05 --cb-weights auto --cb-weights-auto-alpha 0.5 \
  --eval-audio-every 5 --eval-frames 1290 --eval-temperature 1.1 \
  --save_dir rvq_ckpts_big --log-lr-steps 50
```

## Run #3 (fast path; current baseline)
```python
python train_rvq.py --data "npq_out/*.npq" \
  --ctx 512 --step 256 \
  --batch 6 --accum 6 \
  --epochs 250 --amp --device cuda \
  --d_model 768 --layers 16 --heads 12 --dropout 0.1 \
  --lr 2e-4 --weight-decay 0.02 \
  --fused-optim --compile \
  --ema 0.999 --label-smoothing 0.05 \
  --cb-weights auto --cb-weights-auto-alpha 0.5 \
  --val-every 1 \
  --eval-audio-every 5 --eval-on-cpu \
  --save_dir rvq_ckpts_fast --tag base512 --log-lr-steps 50 \
  --persistent-workers --prefetch 6 --quiet
```

---
If resuming, you can omit most arch flags; the script will auto-resume from *_latest.pt and restore the architecture unless you pass --ignore-arch.