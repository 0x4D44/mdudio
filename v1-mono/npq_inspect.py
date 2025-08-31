#!/usr/bin/env python3
import sys, struct, numpy as np
MAGIC=b"NPQ1"; DTYPE={0:"u8",1:"u16",2:"u32"}
with open(sys.argv[1],"rb") as f:
    assert f.read(4)==MAGIC, "bad magic"
    v,K,fps,br,T=struct.unpack("<HHffI",f.read(16))
    vs=list(struct.unpack(f"<{K}I",f.read(4*K)))
    dt=DTYPE[struct.unpack("<B",f.read(1))[0]]
print(f"K={K} fps≈{fps:.2f} T={T} (~{T/max(fps,1e-9):.2f}s) vocab={vs} dtype={dt}")