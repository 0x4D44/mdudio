#!/usr/bin/env python3
import glob, os, struct
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

MAGIC = b"NPQ1"

# ----------------- file discovery -----------------
def list_npq(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        if os.path.isdir(p):
            out.extend(glob.glob(os.path.join(p, "*.npq")))
        else:
            out.extend(glob.glob(p))
    out = sorted(set(out))
    return [x for x in out if x.lower().endswith(".npq")]

# ----------------- tolerant header reader -----------------
def _unpack(f, fmt):
    sz = struct.calcsize(fmt)
    buf = f.read(sz)
    if len(buf) != sz:
        raise EOFError("Unexpected EOF")
    return struct.unpack(fmt, buf)

def read_npq_header(path: str) -> Tuple[Dict, int]:
    """
    Returns (hdr, payload_offset).
    Preferred (versioned) layout written by your encoder:
      magic[4], ver<u16>=1, K<u16>, fps<f32>, br<f32>, T<u32>, vocab<K*u32>, dtype<u8>
    Also tolerates legacy variants with no version / different order.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")

        # Try NPQv1
        try:
            pos0 = f.tell()
            ver, = _unpack(f, "<H")
            K16, = _unpack(f, "<H")
            if 1 <= ver <= 255 and 1 <= K16 <= 512:
                fps, = _unpack(f, "<f")
                br , = _unpack(f, "<f")
                T  , = _unpack(f, "<I")
                vocab = list(_unpack(f, f"<{K16}I"))
                dtype_code, = _unpack(f, "<B")
                elem = {0:1, 1:2, 2:4}.get(dtype_code, None)
                if elem is None:
                    raise ValueError("bad dtype_code")
                off = f.tell()
                needed = off + elem * T * K16
                if needed <= size and all(1 <= v <= (1<<32)-1 for v in vocab):
                    hdr = dict(
                        version=ver, codebooks=K16, fps=fps, sr_hz=44100,
                        bitrate_kbps=br, T=T, vocab_sizes=vocab, dtype_code=dtype_code
                    )
                    return hdr, off
            f.seek(pos0)
        except Exception:
            f.seek(4)

        # Legacy: K<i32>, fps<f32>, br<f32>, T<i32>, vocab<K*i32>
        try:
            K, = _unpack(f, "<i")
            fps, = _unpack(f, "<f")
            br , = _unpack(f, "<f")
            T  , = _unpack(f, "<i")
            vocab = list(_unpack(f, f"<{K}i"))
            dtype_code = 1  # default uint16 payload
            off = f.tell()
            needed = off + 2 * T * K
            if needed <= size and 1 <= K <= 512:
                hdr = dict(
                    version=0, codebooks=K, fps=fps, sr_hz=44100,
                    bitrate_kbps=br, T=T, vocab_sizes=[int(v) for v in vocab], dtype_code=dtype_code
                )
                return hdr, off
        except Exception:
            pass

        # Legacy fallback: K<i32>, fps<f32>, vocab<K*i32>, T<i32>
        try:
            f.seek(4)
            K, = _unpack(f, "<i")
            fps, = _unpack(f, "<f")
            vocab = list(_unpack(f, f"<{K}i"))
            T, = _unpack(f, "<i")
            dtype_code = 1
            off = f.tell()
            needed = off + 2 * T * K
            if needed <= size and 1 <= K <= 512:
                hdr = dict(
                    version=0, codebooks=K, fps=fps, sr_hz=44100,
                    bitrate_kbps=0.0, T=T, vocab_sizes=[int(v) for v in vocab], dtype_code=dtype_code
                )
                return hdr, off
        except Exception:
            pass

        raise ValueError(f"{path}: unrecognized NPQ header layout")

def validate_npq(path: str, expect_K: Optional[int] = None) -> Tuple[bool, str, Optional[Dict]]:
    try:
        hdr, off = read_npq_header(path)
    except Exception as e:
        return False, f"header error: {e}", None
    if expect_K is not None and hdr["codebooks"] < expect_K:
        return False, f"too few codebooks: {hdr['codebooks']} < {expect_K}", hdr
    elem = {0:1, 1:2, 2:4}[hdr.get("dtype_code", 1)]
    size = os.path.getsize(path)
    needed = off + elem * hdr["T"] * hdr["codebooks"]
    if size < needed:
        return False, f"truncated payload: size {size} < expected {needed}", hdr
    return True, "ok", hdr

def validate_npq_files(files: List[str], expect_K: Optional[int] = None):
    good, bad = [], []
    for p in files:
        ok, reason, _ = validate_npq(p, expect_K=expect_K)
        if ok: good.append(p)
        else:  bad.append((p, reason))
    return good, bad

# ----------------- full read -----------------
def read_npq(path: str) -> Tuple[np.ndarray, Dict]:
    hdr, off = read_npq_header(path)
    K, T = hdr["codebooks"], hdr["T"]
    dtype_code = hdr.get("dtype_code", 1)
    dtype = {0: np.uint8, 1: np.uint16, 2: np.uint32}[dtype_code]
    elem = {0:1, 1:2, 2:4}[dtype_code]
    with open(path, "rb") as f:
        f.seek(off)
        buf = f.read(T * K * elem)
        if len(buf) != T * K * elem:
            raise EOFError(f"{path}: truncated payload")
        tokens = np.frombuffer(buf, dtype=dtype).reshape(T, K)
    return tokens, hdr

# ----------------- dataset -----------------
class NPQWindowDataset(Dataset):
    """
    Sliding windows over NPQ tokens.
    Returns:
      inputs  [ctx,K] int64  (BOS=-1 at row 0)
      targets [ctx,K] int64
    """
    def __init__(self, files: List[str], ctx_frames: int = 1024, step_frames: Optional[int] = None, min_K: int = 9):
        super().__init__()
        self.files = files
        self.ctx = int(ctx_frames)
        self.step = int(step_frames) if step_frames is not None else max(1, self.ctx // 2)
        self.min_K = int(min_K)

        self._index = []  # (file_idx, start)
        for i, p in enumerate(files):
            try:
                hdr, _ = read_npq_header(p)
            except Exception:
                continue
            K = hdr["codebooks"]; T = hdr["T"]
            if K < self.min_K or T < self.ctx:
                continue
            for s in range(0, T - self.ctx + 1, self.step):
                self._index.append((i, s))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        file_i, s = self._index[idx]
        tokens, _ = read_npq(self.files[file_i])
        K_use = min(self.min_K, tokens.shape[1])
        target = tokens[s : s + self.ctx, :K_use].astype(np.int64)
        inputs = target.copy()
        inputs[1:] = target[:-1]
        inputs[0, :] = -1
        return {"inputs": torch.from_numpy(inputs), "targets": torch.from_numpy(target)}
