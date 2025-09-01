# NPQ Encoder/Decoder for DAC (encode-v2)

Tools to encode audio into DAC RVQ tokens (`.npq`) and decode back to WAV.
Default encoding uses stereo Mid/Side (M/S) with 9 codebooks for Mid and 4 for Side. Mono remains supported and backward‑compatible.

## Install
- Python 3.12 recommended (local venv supported by `pyvenv.cfg`).
- Install deps:
  - `pip install torch torchaudio transformers soundfile numpy`
  - For CUDA/MPS, install the matching PyTorch build from pytorch.org instructions.

## Project Structure
- `dac_tokens_export.py`: Encode audio → `.npq` tokens (default: stereo M/S).
- `npq_to_wav.py`: Decode `.npq` tokens → WAV.
- `npq_out/`: Outputs (generated `.npq` and `.wav`).
- Samples: `01.mp3`, `02.mp3` for quick trials.

## Usage
- Encode stereo (default M/S 9 mid + 4 side):
  - `python dac_tokens_export.py "*.mp3" --outdir npq_out --device cpu`
- Force mono (NPQ1) and choose K:
  - `python dac_tokens_export.py "*.wav" --mono --codebooks 9`
- Adjust codebooks (stereo):
  - `python dac_tokens_export.py song.wav --mid-codebooks 8 --side-codebooks 3`
- Choose model: `--model 16khz|24khz|44khz|<hf-repo>` (default `44khz`).
- Decode (auto-detects mono/stereo):
  - `python npq_to_wav.py npq_out/song.npq --out npq_out/song.wav`

## NPQ File Format
- NPQ1 (mono, existing):
  - Header (little-endian):
    - `magic:'NPQ1'`, `version:u16=1`, `num_codebooks:u16=K`, `token_rate:f32`, `orig_bitrate_kbps:f32`, `seq_len:u32=T`, `vocab_sizes:u32[K]`, `dtype_code:u8` (0=uint8,1=uint16,2=uint32)
  - Payload: tokens row-major `[T, K]`.
- NPQ2 (stereo Mid/Side):
  - Header: `magic:'NPQ2'`, `version:u16=1`, `channel_mode:u8=1`, `k_mid:u16`, `k_side:u16`, `token_rate:f32`, `orig_bitrate_kbps:f32`, `seq_len:u32=T`, `vocab_sizes:u32[k_mid+k_side]`, `dtype_code:u8`.
  - Payload: tokens `[T, k_mid+k_side]` with columns `[M... S...]`.
  - M/S mapping: encode with `M=(L+R)/2`, `S=(L-R)/2`; decode with `L=M+S`, `R=M-S`.

## Tips
- Devices: use `--device cpu|cuda|mps`. If unavailable, the tools fall back to CPU with a warning.
- Chunking: `--chunk-seconds` (default 10.0) controls encode chunk size and memory usage.
- Outputs: avoid committing large audio; keep `npq_out/` untracked if needed.

## Contributing
See `AGENTS.md` for coding style, testing guidance, and PR expectations.
