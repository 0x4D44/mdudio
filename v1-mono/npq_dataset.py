#!/usr/bin/env python3
import os, struct, glob, random
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

MAGIC = b"NPQ1"
DTYPE_MAP = {0: np.uint8, 1: np.uint16, 2: np.uint32}

def read_npq(path: str) -> Tuple[np.ndarray, Dict]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC: raise ValueError(f"{path}: bad magic {magic!r}")
        version, K, token_rate, orig_bitrate, T = struct.unpack("<HHffI", f.read(16))
        vocab_sizes = list(struct.unpack(f"<{K}I", f.read(4*K)))
        dtype_code = struct.unpack("<B", f.read(1))[0]
        if dtype_code not in DTYPE_MAP: raise ValueError(f"{path}: unknown dtype_code {dtype_code}")
        dtype = DTYPE_MAP[dtype_code]
        data = np.fromfile(f, dtype=dtype, count=T*K)
        if data.size != T*K: raise ValueError(f"{path}: truncated payload ({data.size}/{T*K})")
        tokens_TK = data.reshape(T, K)
    hdr = {"version":version,"K":K,"token_rate_fps":float(token_rate),"orig_bitrate_kbps":float(orig_bitrate),
           "T":int(T),"vocab_sizes":vocab_sizes,"dtype_code":int(dtype_code)}
    return tokens_TK, hdr

def list_npq(paths_or_globs: List[str]) -> List[str]:
    out=[]
    for p in paths_or_globs:
        m=glob.glob(p, recursive=True)
        if m: out.extend(sorted(m))
        elif os.path.isfile(p): out.append(p)
    seen, uniq = set(), []
    for x in out:
        if x not in seen: uniq.append(x); seen.add(x)
    return uniq

class NPQWindowDataset(Dataset):
    """
    Yields windows for teacher forcing:
      inputs:  [T_ctx,K] -> tokens from t-1 (BOS=-1 at t=0)
      targets: [T_ctx,K] -> tokens at t
    """
    def __init__(self, files: List[str], ctx_frames: int = 1024, step_frames: Optional[int] = None,
                 max_files: Optional[int] = None, max_windows_per_file: Optional[int] = None,
                 shuffle_within_file: bool = True, min_K: Optional[int] = None):
        self.files = files[:max_files] if max_files else files
        self.ctx = int(ctx_frames)
        self.step = int(step_frames) if step_frames else self.ctx // 2
        self.shuffle_within_file = shuffle_within_file
        self.min_K = min_K
        self.index=[]
        for i, fp in enumerate(self.files):
            try:
                tokens, hdr = read_npq(fp)
            except Exception as e:
                print(f"[WARN] skipping {fp}: {e}"); continue
            T, K = tokens.shape
            if self.min_K is not None and K < self.min_K: continue
            starts = list(range(0, max(0, T - self.ctx), self.step))
            if (T >= self.ctx) and ((T - self.ctx) % self.step != 0): starts.append(T - self.ctx)
            if T < self.ctx: starts=[0]
            if self.shuffle_within_file: random.shuffle(starts)
            if max_windows_per_file: starts = starts[:max_windows_per_file]
            for s in starts: self.index.append((i, s))
        random.shuffle(self.index)

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        file_idx, s = self.index[idx]
        fp = self.files[file_idx]
        tokens, hdr = read_npq(fp)  # small enough to load
        T, K = tokens.shape
        if T >= self.ctx:
            window = tokens[s:s+self.ctx]
        else:
            pad = -np.ones((self.ctx - T, K), dtype=np.int64)
            window = np.concatenate([pad, tokens.astype(np.int64)], axis=0)
        window = window.astype(np.int64)
        inputs = np.roll(window, shift=1, axis=0); inputs[0,:]=-1
        targets = window
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "K": K,
            "vocab_sizes": torch.tensor(hdr["vocab_sizes"][:K], dtype=torch.long),
            "token_rate_fps": torch.tensor(hdr["token_rate_fps"], dtype=torch.float32),
        }
