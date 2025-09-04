# RVQ Token Modelling — Project Summary

This repo builds an end-to-end pipeline to:
1) **Encode** audio to **NPQ** (discrete RVQ tokens from Descript DAC),
2) **Train** an autoregressive Transformer on those tokens,
3) **Decode/Generate** tokens back to audio (via DAC).

It’s designed for **44.1 kHz**, **9 codebooks**, and long training runs on a **12 GB GPU** (e.g., RTX 4070 Ti) using memory-friendly tricks (Flash/SDPA attention, optional checkpointing, gradient accumulation).

---

## Files in this repo

- `dac_tokens_export.py` — Encode audio (`*.wav/*.mp3`) to **.npq** files (binary RVQ token format).
- `npq_to_wav.py` — Decode **.npq → .wav** (44.1 kHz, 16-bit PCM).
- `npq_dataset.py` — Robust **.npq** reader + sliding-window Dataset (tolerant header parser, validation).
- `rvq_transformer.py` — Token model (Transformer with SDPA/Flash attention; optional gradient checkpointing).
- `train_rvq.py` — Training script (cosine LR, warmup, EMA, label smoothing, CB weighting, accumulation, auto-resume & archiving, optional eval-audio).

> Optional utilities (not strictly required): a “checkpoint doctor” to summarize checkpoints (total params, EMA stats, spikes) can be added later.

---

## NPQ file format
**Header (little-endian, versioned):**
- magic 4s = b"NPQ1"
- version u16 = 1
- num\codebooks u16 = K
- token\rate f32 = frames per second (≈ 86.13 for DAC 44.1 kHz)
-  orig\bitrate f32 = kbps estimate from input file
-  seq\len u32 = T (frames)
-  vocab\sizes u32[K]
-  dtype\code u8 = 0: uint8 | 1: uint16 | 2: uint32

**Payload:**

> The reader in `npq\_dataset.py` is tolerant and also recognizes a few legacy layouts.

---

## Quick start
### 1) Encode audio → NPQ

```bash

# Example: encode a folder of mp3s to NPQ (44.1 kHz DAC, 9 codebooks, 10 s chunks)
python dac_tokens_export.py "data/**/*.mp3" \
  --model 44khz \
  --codebooks 9 \
  --chunk-seconds 10 \
  --outdir npq_out \
  --device cuda
```

You’ll see a per-file summary (model repo, SR, tokens fps, total frames, header bytes, output size, compression ratio).

### 2) Sanity-check NPQs

```bash

# Show parsed headers for a few files
python - <<'PY'
from npq_dataset import list_npq, read_npq_header
for p in list_npq(["npq_out/*.npq"])[:5]:
    hdr, off = read_npq_header(p)
    print(p, "K=", hdr["codebooks"], "T=", hdr["T"], "fps≈", round(hdr["fps"],3), "dtype=", hdr["dtype_code"], "payload_off=", off)
PY
```

### 3) Train the model

Safe baseline for a 12 GB GPU:
```bash

python train_rvq.py --data "npq_out/*.npq" \
  --ctx 1280 --batch 4 --accum 6 \
  --epochs 350 --amp --device cuda \
  --d_model 768 --layers 16 --heads 12 --dropout 0.1 \
  --lr 2e-4 --weight-decay 0.02 \
  --ema 0.999 --label-smoothing 0.05 \
  --cb-weights auto --cb-weights-auto-alpha 0.5 \
  --eval-audio-every 5 --eval-on-cpu \
  --save_dir rvq_ckpts_big --log-lr-steps 50
```

- Auto-resume: by default the script looks for rvq\_9cb\_latest.pt and resumes.
- Checkpoints:
  - rvq_9cb_latest.pt — rolling latest,
  - rvq_9cb_best.pt — best (lowest val),
  - archived copies: rvq_9cb_best_e####__YYYYMMDD-HHMMSS.pt and periodic epoch archives.

If you see a CUDA OOM with your settings, either lower --batch, enable --grad-checkpoint, or reduce --ctx to 1024. If you have spare VRAM, turn off checkpointing for ~20–30% faster steps.

### 4) Decode NPQ → WAV

```bash

python npq_to_wav.py npq_out/track_01.npq \
  --model 44khz --out reconstructed.wav
```

The tool prints header stats (K, fps, seq\_len, vocab sizes, dtype) and decodes to 44.1 kHz / 16-bit PCM.

## Training notes

### Dataset sizing & windows
- DAC 44.1k fps: ≈ 86.13 frames/s.
- With --ctx 1280 → each window is ~14.9 s.
- Default stride is ctx//2: --step 640 → ~7.45 s new audio per window (50% overlap).
- Unique train hours ≈ train_windows × step / fps / 3600.

Example from logs:

```kotlin
windows: train=34520 val=1816 (ctx=1280, step=640)
≈ 34,520 × 640 / 86.13 ≈ 71.3 hours train
≈ 1,816  × 640 / 86.13 ≈ 3.8  hours val
```

### Throughput & time per epoch
- Optimizer steps/epoch are printed (e.g., 1438 with batch=4, accum=6).
- Epoch time ≈ steps/epoch × time/step + validation time.
- Expect eval-audio overhead to be negligible; validation adds forward-only batches.

### LR schedule
- Warmup: --warmup-ratio 0.03 by default (e.g., ~10.5 epochs).
- Then cosine decay to --eta-min-ratio of base LR.
- LR is logged every --log-lr-steps.

### Stability & memory
- SDPA/Flash attention (in rvq\_transformer.py) avoids T×T masks, saves memory.
- Gradient accumulation (--accum) keeps VRAM low but preserves effective batch.
- Gradient checkpointing (--grad-checkpoint) saves memory at the cost of extra compute. Turn it off if you have headroom for faster steps.
- TF32 on CUDA is enabled for speed/throughput.

### Checkpointing & resume
- The trainer writes latest, best, and archived checkpoints (both periodic epoch snapshots and every new best).
- Auto-resume: if *_latest.pt exists in --save_dir, the run resumes unless you pass --no-resume.
- You can also specify --resume PATH explicitly.

### Metrics & how to read them
- Train/val loss = mean cross-entropy (nats) across codebooks (optionally weighted).
- Per-codebook CE: printed as cb0 … cb8; lower is better; coarse books (low index) typically improve first.
- Perplexity (ppl) is exp(loss) — useful as a rough secondary signal.
- Bits per token (BPT) ≈ loss / ln(2); for 1024-way (10 bits max), you’ll typically see BPT taper towards ~6–7 bits on fine books.

> Note: earlier versions displayed epoch | train 25.x due to averaging the sum of micro-batch losses over optimizer steps (inflated by --accum). The fix is to average by the micro-batch count instead.

One-line fix (optional, in train\_rvq.py):

```python
# track micro_count and use it to compute train_loss
total_raw = 0.0
micro_count = 0
...
total_raw += float(loss_raw.detach().item())
micro_count += 1
...
train_loss = total_raw / max(1, micro_count)
```

## Typical commands

### Encode a whole folder:

```python
python dac_tokens_export.py "music/*.mp3" --model 44khz --codebooks 9 --chunk-seconds 10 --outdir npq_out --device cuda
```

### Start a long training run (12 GB GPU, safe):
```python
python train_rvq.py --data "npq_out/*.npq" \
  --ctx 1280 --batch 4 --accum 6 \
  --epochs 350 --amp --device cuda \
  --d_model 768 --layers 16 --heads 12 --dropout 0.1 \
  --lr 2e-4 --weight-decay 0.02 \
  --ema 0.999 --label-smoothing 0.05 \
  --cb-weights auto --cb-weights-auto-alpha 0.5 \
  --eval-audio-every 5 --eval-on-cpu \
  --save_dir rvq_ckpts_big --log-lr-steps 50
```

### Resume (auto):
```python
python train_rvq.py --data "npq_out/*.npq" --ctx 1280 --batch 4 --accum 6 --amp --device cuda --save_dir rvq_ckpts_big
```

###  Decode back to audio:
```python
python npq_to_wav.py npq_out/song_01.npq --model 44khz --out out.wav
```

## Troubleshooting
- CUDA OOM on first step:
  - Lower --batch, enable --grad-checkpoint, or reduce --ctx to 1024.
  - Keep --accum to preserve the effective batch.
- Long pause after “======== TRAIN ========” + high “shared memory” usage:
  - Means attention is paging; reduce --batch/--ctx, ensure SDPA attention is used (already in rvq_transformer.py).
- All NPQs “bad header” but decoder works:
  - Use the tolerant reader in npq_dataset.py (supports the versioned header in dac_tokens_export.py).
- Divergence (val loss spikes to huge numbers):
  - Reload from *_best.pt or *_latest.pt (script does this if non-finite loss is detected),
  - Reduce LR (--lr) or lower --eta-min-ratio, consider more data.

## Model shape (intuition)
- Input at each frame: K=9 codebooks, each a token in [0, …, 1023]. We embed each codebook separately into d_model and sum them + positional embedding.
- Transformer (L layers, H heads) produces a hidden state per frame.
- We predict the next frame’s codebooks (multi-head classifier: one linear head per codebook).
- Training uses cross-entropy per codebook, optionally weighted (e.g., --cb-weights auto upweights finer books slightly).

## Performance tips
- If VRAM headroom exists (you’re at ~6–8 GB):
  - Disable checkpointing for ~20–30% faster steps.
  - Try increasing micro-batch (e.g., 4 → 5/6) while watching VRAM; don’t expect epoch time to halve (you still process the same windows).
- Shorter warmup (--warmup-ratio 0.01) brings the LR up sooner (watch stability).
- Context (--ctx) drives compute ~quadratically — 1024 is ~36% cheaper than 1280.
- Consider fused=True in AdamW (PyTorch 2.x on CUDA) and torch.compile(model, mode="reduce-overhead") for extra throughput (optional).

## License / model use

This pipeline uses Descript’s published DAC models from Hugging Face (descript/dac_44khz, etc.). Please check upstream licenses and terms for redistribution and use of the pretrained decoder/encoder.