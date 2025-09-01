# Repository Guidelines

## Project Structure & Module Organization
- Source: `dac_tokens_export.py` (encode audio → `.npq`), `npq_to_wav.py` (decode `.npq` → WAV).
- Samples: `01.mp3`, `02.mp3` (for quick trials).
- Outputs: `npq_out/` (generated `.npq` and `.wav`).
- Local env: `bin/`, `lib/`, `pyvenv.cfg` indicate a Python 3.12 virtual environment in this folder.

## Build, Test, and Development Commands
- Run encoder: `python dac_tokens_export.py "*.mp3" --outdir npq_out --device cpu`
  - Produces `.npq` RVQ token files using the DAC 44.1 kHz model (default 9 codebooks).
- Run decoder: `python npq_to_wav.py npq_out/01.npq --out npq_out/01.wav`
  - Reconstructs 44.1 kHz, 16‑bit PCM WAV from tokens.
- Install deps (example): `pip install torch torchaudio transformers soundfile numpy`
  - Torch install may vary by OS/CUDA; consult PyTorch.org for the right command.

## Coding Style & Naming Conventions
- Python 3.12; follow PEP 8 with 4‑space indentation and type hints for public functions.
- Keep modules single‑purpose; CLIs use `argparse` with clear `--help` text.
- Prefer snake_case for variables/functions and UPPER_CASE for constants (e.g., `MAGIC`).
- Optional: format with Black and lint with Ruff; keep diffs minimal.

## Testing Guidelines
- Framework: `pytest` (recommended). Place tests in `tests/` with names like `test_npq_io.py`.
- Aim to cover: header parsing/serialization, chunking logic, and device fallbacks.
- Run: `pytest -q` (add `-k` to filter). Target ≥80% coverage where practical.

## Commit & Pull Request Guidelines
- Commits: use Conventional Commits (e.g., `feat: add MPS fallback`, `fix: correct dtype_code`).
- PRs: include a concise summary, reproduction steps, sample inputs/outputs from `npq_out/`, and any perf notes (CPU vs CUDA).
- Link related issues; add before/after CLI examples when changing arguments or defaults.

## Security & Configuration Tips
- Models download from Hugging Face on first run; cache is reused. Use `--device cpu|cuda|mps` explicitly.
- Large files: avoid committing audio outputs; keep `npq_out/` in `.gitignore` if versioning outside this folder.
