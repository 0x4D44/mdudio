#!/usr/bin/env python3
"""
Export DAC RVQ tokens to .npq (binary) from audio files.
Defaults: DAC 44.1 kHz model, 9 codebooks, chunked encoding.

Header (little-endian):
  magic        4s   'NPQ1'
  version      u16  1
  num_codebooks u16 K
  token_rate   f32  (frames per second)
  orig_bitrate f32  (kbps, avg from input file size & duration)
  seq_len      u32  (T frames)
  vocab_sizes  u32[K]
  dtype_code   u8   (0=uint8,1=uint16,2=uint32)
Payload:
  tokens       (T*K) unsigned ints, row-major [T, K]
"""
import argparse, glob, os, struct
from typing import Iterable, List, Optional, Tuple
import numpy as np
import torch, torchaudio
from transformers import DacModel, AutoProcessor

def map_model_name(name: str) -> str:
    name = name.strip()
    return {
        "16khz": "descript/dac_16khz",
        "24khz": "descript/dac_24khz",
        "44khz": "descript/dac_44khz",  # 44.1 kHz
        "44.1khz": "descript/dac_44khz",
    }.get(name.lower(), name)

def expand_inputs(patterns: Iterable[str]) -> List[str]:
    out = []
    for p in patterns:
        m = glob.glob(p, recursive=True)
        if m: out.extend(sorted(m))
        elif os.path.isfile(p): out.append(p)
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def pick_dtype(max_vocab_size: int) -> Tuple[np.dtype, int]:
    # dtype_code: 0=uint8, 1=uint16, 2=uint32
    if max_vocab_size <= 256: return (np.uint8, 0)
    if max_vocab_size <= 65536: return (np.uint16, 1)
    return (np.uint32, 2)

def infer_vocab_sizes(model, k: int) -> List[int]:
    cfg = getattr(model, "config", None)
    sizes = None
    if cfg is not None:
        for key in ["codebook_sizes","codebook_size","num_embeddings","n_bins"]:
            if hasattr(cfg, key):
                sizes = getattr(cfg, key); break
    if sizes is None: sizes = 1024
    if isinstance(sizes, int): return [int(sizes)] * k
    return [int(v) for v in list(sizes)[:k]]

def get_num_codebooks_from_config(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    if cfg is None: return None
    for key in ["num_codebooks","num_quantizers","n_q","n_codebooks"]:
        if hasattr(cfg, key):
            try: return int(getattr(cfg, key))
            except: pass
    return None

def file_bitrate_kbps(path: str, duration_sec: float) -> float:
    bits = os.path.getsize(path) * 8.0
    return (bits / max(duration_sec, 1e-9)) / 1000.0

def header_size_bytes(k: int) -> int:
    # 4 + 2 + 2 + 4 + 4 + 4 + 4k + 1
    return 21 + 4 * k

def write_npq(out_path: str, tokens_TK: np.ndarray, vocab_sizes: List[int],
              token_rate_fps: float, orig_bitrate_kbps: float) -> Tuple[int,int]:
    T, K = tokens_TK.shape
    max_vocab = max(int(v) for v in vocab_sizes[:K]) if K else 0
    np_dtype, dtype_code = pick_dtype(max_vocab)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4sH", b"NPQ1", 1))
        f.write(struct.pack("<H", K))
        f.write(struct.pack("<f", float(token_rate_fps)))
        f.write(struct.pack("<f", float(orig_bitrate_kbps)))
        f.write(struct.pack("<I", T))
        f.write(struct.pack(f"<{K}I", *[int(v) for v in vocab_sizes[:K]]))
        f.write(struct.pack("<B", dtype_code))
        tokens_TK.astype(np_dtype, copy=False).tofile(f)
    return header_size_bytes(K), os.path.getsize(out_path)

def orient_codes_2d(codes_2d: torch.Tensor, model_k: Optional[int]) -> Tuple[torch.Tensor, int]:
    """Return time-major [T,K], K_used. Accepts [T,K] or [K,T]."""
    if codes_2d.dim() != 2: raise ValueError(f"Expected 2D codes, got {tuple(codes_2d.shape)}")
    h, w = codes_2d.shape
    if model_k is not None:
        if h == model_k and w != model_k:  # [K,T]
            return codes_2d.T.contiguous(), model_k
        if w == model_k and h != model_k:  # [T,K]
            return codes_2d.contiguous(), model_k
    # fallback: smaller dim is K
    if h <= w: return codes_2d.T.contiguous(), h
    return codes_2d.contiguous(), w

def process_file(path: str, model: DacModel, processor: AutoProcessor, codebooks: Optional[int],
                 chunk_seconds: float, outdir: str, device: torch.device, verbose: bool=True) -> str:
    if verbose: print(f"\n==> {os.path.basename(path)}")
    # Load once (robust for MP3/VBR)
    wav, sr_in = torchaudio.load(path)           # [C, N]
    duration_sec = wav.shape[-1] / sr_in
    orig_bitrate_kbps = file_bitrate_kbps(path, duration_sec)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)  # mono
    # Resample to model SR
    target_sr = int(getattr(processor, "sampling_rate", 44100))
    if sr_in != target_sr:
        wav = torchaudio.transforms.Resample(sr_in, target_sr)(wav)
    sr = target_sr; samples = wav.shape[-1]
    # Chunking
    chunk_len = max(1, int(round(chunk_seconds * sr)))
    num_chunks = (samples + chunk_len - 1) // chunk_len
    if verbose:
        print(f"  input_dur: {duration_sec:.2f}s | sr_in:{sr_in} -> sr:{sr} | chunks:{num_chunks} (~{chunk_seconds:.1f}s)")
    # Encode
    model.eval(); model.to(device)
    token_rows: List[np.ndarray] = []
    model_k = get_num_codebooks_from_config(model)
    first_chunk_fps: Optional[float] = None
    used_k_final: Optional[int] = None

    for i in range(num_chunks):
        s, e = i * chunk_len, min((i + 1) * chunk_len, samples)
        if e <= s: continue
        chunk = wav[:, s:e].squeeze(0).numpy()   # 1D float
        inputs = processor(raw_audio=chunk, sampling_rate=sr, return_tensors="pt")
        x = inputs["input_values"].to(device)
        with torch.inference_mode():
            enc = model.encode(x)                # enc.audio_codes
        codes = enc.audio_codes
        # squeeze batch -> [?, ?]
        if codes.dim() == 3: codes2d = codes[0]
        elif codes.dim() == 2: codes2d = codes
        else: codes2d = codes.view(codes.shape[-2], codes.shape[-1])
        # Orient to [T,K]
        codes_TK, discovered_k = orient_codes_2d(codes2d, model_k)
        # Choose K
        target_k = 9 if codebooks is None else int(codebooks)
        cap_k = (model_k if model_k is not None else discovered_k)
        use_k = min(target_k, cap_k)
        sel = codes_TK[:, :use_k].cpu().numpy()  # [T_chunk, K_used]
        token_rows.append(sel)
        if first_chunk_fps is None:
            t_sec = max(1e-9, (e - s) / sr)
            first_chunk_fps = sel.shape[0] / t_sec
            used_k_final = use_k

    if not token_rows: raise RuntimeError(f"No tokens produced for {path}")
    tokens = np.concatenate(token_rows, axis=0)  # [T_total, K_used]
    T_total, K_used = tokens.shape
    token_rate_fps = float(T_total / max(duration_sec, 1e-9))
    vocab_sizes = infer_vocab_sizes(model, K_used)

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(outdir, f"{base}.npq")
    hdr_bytes, total_bytes = write_npq(out_path, tokens, vocab_sizes, token_rate_fps, orig_bitrate_kbps)

    compressed_kbps = (total_bytes * 8.0) / max(duration_sec, 1e-9) / 1000.0
    comp_ratio = (orig_bitrate_kbps / compressed_kbps) if compressed_kbps > 0 else float("inf")
    model_repo = getattr(model, "name_or_path", "unknown")

    print("  ── encode stats ─────────────────────────────────")
    print(f"  model_repo     : {model_repo}")
    print(f"  model_sr       : {sr} Hz")
    print(f"  codebooks_used : {K_used} (target {9 if codebooks is None else codebooks}, model cfg {model_k})")
    print(f"  audio_len      : {duration_sec:.3f} s")
    print(f"  tokens_frames  : {T_total}  (≈ {token_rate_fps:.3f} fps)")
    print(f"  tokens_total   : {T_total * K_used}")
    print(f"  vocab_sizes    : {vocab_sizes}")
    print(f"  header_bytes   : {hdr_bytes}")
    print(f"  file_out_bytes : {total_bytes}")
    print(f"  compressed     : {compressed_kbps:.3f} kbps  |  original: {orig_bitrate_kbps:.3f} kbps  |  ratio: {comp_ratio:.2f}x")
    if first_chunk_fps is not None and used_k_final is not None:
        print(f"  first_chunk_fps: {first_chunk_fps:.3f} fps (K={used_k_final})")
    print(f"  chunking       : {chunk_len} samples ~ {chunk_seconds:.3f} s per chunk")
    print(f"  wrote          : {out_path}")
    print("  ─────────────────────────────────────────────────")
    return out_path

def main():
    ap = argparse.ArgumentParser(
        description="Export DAC RVQ tokens (.npq) from audio (default: 44.1 kHz model, 9 codebooks)."
    )
    ap.add_argument("inputs", nargs="+", help='Files or globs, e.g. "*.mp3" "data/**/*.wav"')
    ap.add_argument("--model", default="44khz", help="16khz | 24khz | 44khz | <HF repo id> (default: 44khz)")
    ap.add_argument("--codebooks", type=int, default=9, help="Use first K codebooks (default: 9)")
    ap.add_argument("--chunk-seconds", type=float, default=10.0, help="Chunk length in seconds (default: 10.0)")
    ap.add_argument("--outdir", default="npq_out", help="Output dir for .npq files")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"], help="Compute device")
    args = ap.parse_args()

    files = expand_inputs(args.inputs)
    if not files: raise SystemExit("No input files matched.")

    repo = map_model_name(args.model)
    print(f"Loading model: {repo}")
    processor = AutoProcessor.from_pretrained(repo)
    model = DacModel.from_pretrained(repo)

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("WARNING: MPS requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")

    os.makedirs(args.outdir, exist_ok=True)
    for path in files:
        process_file(path, model, processor, args.codebooks, args.chunk_seconds, args.outdir, device, True)
    print(f"\nDone. Output dir: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
