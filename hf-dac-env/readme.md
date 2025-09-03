# NPQ: Neural Product Quantizer token files

`*.npq` files store **time-major RVQ tokens** for audio, suitable for training/generation with a Transformer and for round-trip decode via Descript’s DAC.

- **Container**: binary, little-endian  
- **Layout**: a compact header followed by a dense token matrix  
- **Tokens**: integer codebook indices, shape **[T, K]** (T = frames, K = codebooks)

> In our pipeline, `.npq` files are produced by `dac_tokens_export.py` and read by `train_rvq.py` / `npq_to_wav.py`.

---

## File format (NPQ v1)

**Magic**: `b'NPQ1'`  
All multi-byte fields are **little-endian**.

| Field            | Type     | Size  | Description |
|------------------|----------|------:|-------------|
| `magic`          | `4s`     | 4     | ASCII `"NPQ1"` |
| `version`        | `u16`    | 2     | Format version. Current: **1** |
| `num_codebooks`  | `u16`    | 2     | **K** (number of codebooks/quantizers) |
| `token_rate`     | `f32`    | 4     | Frames per second (**fps**) of tokens (≈ 86.13 for DAC 44.1 kHz) |
| `orig_bitrate`   | `f32`    | 4     | Source file bitrate in **kbps** (informational) |
| `seq_len`        | `u32`    | 4     | **T** (number of time frames) |
| `vocab_sizes[K]` | `u32[K]` | 4K    | Per-codebook vocabulary sizes (e.g. all **1024**) |
| `dtype_code`     | `u8`     | 1     | Payload integer width: **0=uint8**, **1=uint16**, **2=uint32** |

**Payload** (immediately after header):

- **Tokens** `[T, K]` stored **row-major** (time major), each entry is an unsigned int with width determined by `dtype_code`.
- Example: for `dtype_code=1`, payload is `T*K*2` bytes.

**Header size**: `21 + 4*K` bytes  
**Expected file size**: `21 + 4*K + elem*T*K` where `elem ∈ {1,2,4}` from `dtype_code`.

### Notes

- **Sampling rate** is *not* stored in v1. For DAC 44.1 kHz models we assume **44,100 Hz**; `token_rate` should be ≈ **44100 / 512 ≈ 86.1328125**.
- Tokens are **targets only** in the file (0…`vocab_size-1`). Training constructs inputs by shifting targets and using **BOS = −1** in the first row (BOS is **not** stored in `.npq`).
- If a model exposes fewer than K codebooks at inference, downstream tools may slice to the first `K_used`.

---

## Legacy variants (NPQ v0) we still accept

For backward compatibility, the loader also tolerates older/looser layouts that had no version and different field orderings (and no `dtype_code`, defaulting to `uint16`):

- `magic, K<i32>, fps<f32>, br<f32>, T<i32>, vocab<K*i32>` (payload `uint16`)
- `magic, K<i32>, fps<f32>, vocab<K*i32>, T<i32>` (payload `uint16`)

Some prototypes also included an `sr_hz<i32>` in the header; it’s ignored if present.

---

## Reference: header struct (v1)

```text
# Little-endian
# magic        : 4s  ('NPQ1')
# version      : H   (1)
# num_codebooks: H   (K)
# token_rate   : f   (fps)
# orig_bitrate : f   (kbps)
# seq_len      : I   (T)
# vocab_sizes  : K * I
# dtype_code   : B   (0/1/2)
