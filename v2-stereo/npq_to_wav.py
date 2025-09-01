#!/usr/bin/env python3
"""Decode .npq (RVQ tokens) back to WAV (44.1 kHz, 16-bit PCM).

Supports:
- NPQ1 (mono): existing format
- NPQ2 (stereo Mid/Side): M tokens followed by S tokens in each frame
"""
import argparse, os, struct
from typing import Tuple, List
import numpy as np
import torch, torchaudio, soundfile as sf
from transformers import DacModel

MAGIC=b"NPQ1"; VERSION_SUPPORTED=1
DTYPE_MAP={0:np.uint8,1:np.uint16,2:np.uint32}; DTYPE_STR={0:"uint8",1:"uint16",2:"uint32"}

def map_model_name(name: str) -> str:
    name=name.strip().lower()
    return {"16khz":"descript/dac_16khz","24khz":"descript/dac_24khz","44khz":"descript/dac_44khz","44.1khz":"descript/dac_44khz"}.get(name,name)

def read_npq(path: str):
    with open(path,"rb") as f:
        magic=f.read(4)
        if magic==b"NPQ1":
            version,K,fps,br,T=struct.unpack("<HHffI",f.read(16))
            if version!=VERSION_SUPPORTED: raise ValueError(f"{path}: version {version} not supported")
            vocab=list(struct.unpack(f"<{K}I",f.read(4*K)))
            dtype_code=struct.unpack("<B",f.read(1))[0]
            if dtype_code not in DTYPE_MAP: raise ValueError(f"{path}: bad dtype_code {dtype_code}")
            dtype=DTYPE_MAP[dtype_code]
            data=np.fromfile(f,dtype=dtype,count=T*K)
            if data.size!=T*K: raise ValueError(f"{path}: truncated payload ({data.size}/{T*K})")
            tokens=data.reshape(T,K)
            hdr={"format":"NPQ1","version":version,"num_codebooks":K,
                 "token_rate_fps":float(fps),"orig_bitrate_kbps":float(br),
                 "seq_len":int(T),"vocab_sizes":vocab,"dtype_code":int(dtype_code),
                 "dtype_str":DTYPE_STR[dtype_code]}
            return tokens, hdr
        elif magic==b"NPQ2":
            version=struct.unpack("<H",f.read(2))[0]
            if version!=VERSION_SUPPORTED: raise ValueError(f"{path}: version {version} not supported")
            channel_mode=struct.unpack("<B",f.read(1))[0]
            k_mid,k_side=struct.unpack("<HH",f.read(4))
            fps,br=struct.unpack("<ff",f.read(8))
            T=struct.unpack("<I",f.read(4))[0]
            K_total=k_mid+k_side
            vocab=list(struct.unpack(f"<{K_total}I",f.read(4*K_total)))
            dtype_code=struct.unpack("<B",f.read(1))[0]
            if dtype_code not in DTYPE_MAP: raise ValueError(f"{path}: bad dtype_code {dtype_code}")
            dtype=DTYPE_MAP[dtype_code]
            data=np.fromfile(f,dtype=dtype,count=T*K_total)
            if data.size!=T*K_total: raise ValueError(f"{path}: truncated payload ({data.size}/{T*K_total})")
            tokens=data.reshape(T,K_total)
            hdr={"format":"NPQ2","version":version,"channel_mode":int(channel_mode),
                 "k_mid":int(k_mid),"k_side":int(k_side),"token_rate_fps":float(fps),
                 "orig_bitrate_kbps":float(br),"seq_len":int(T),"vocab_sizes":vocab,
                 "dtype_code":int(dtype_code),"dtype_str":DTYPE_STR[dtype_code]}
            return tokens, hdr
        else:
            raise ValueError(f"{path}: bad magic {magic!r}")

def decode_tokens(tokens_TK: np.ndarray, model: DacModel, device: torch.device, fps: float, seconds_per_chunk: float=10.0)->torch.Tensor:
    model.eval(); model.to(device)
    T, K = tokens_TK.shape
    tpc = max(1, int(round((fps if fps>0 else 75.0)*seconds_per_chunk)))
    out=[]
    for t0 in range(0,T,tpc):
        t1=min(T,t0+tpc); chunk=tokens_TK[t0:t1,:]
        codes=torch.from_numpy(chunk.T).long().unsqueeze(0).to(device) # [1,K,Tc]
        with torch.inference_mode():
            wav=model.decode(audio_codes=codes).audio_values
        out.append(wav.squeeze().detach().cpu())
    return torch.cat(out,dim=0)

def decode_tokens_ms(tokens_TK: np.ndarray, k_mid: int, k_side: int, model: DacModel, device: torch.device, fps: float, seconds_per_chunk: float=10.0) -> torch.Tensor:
    T,K_total=tokens_TK.shape
    assert k_mid + k_side == K_total, "Header K mismatch"
    M=tokens_TK[:, :k_mid]
    S=tokens_TK[:, k_mid:]
    wav_M=decode_tokens(M, model, device, fps, seconds_per_chunk)
    wav_S=decode_tokens(S, model, device, fps, seconds_per_chunk)
    # Reconstruct stereo LR from M/S
    L=wav_M + wav_S
    R=wav_M - wav_S
    return torch.stack([L, R], dim=0)  # [2, N]

def main():
    ap=argparse.ArgumentParser(description="Decode .npq to 44.1 kHz 16-bit WAV.")
    ap.add_argument("npq_path"); ap.add_argument("--model",default="44khz")
    ap.add_argument("--device",default="cpu",choices=["cpu","cuda","mps"])
    ap.add_argument("--seconds-per-chunk",type=float,default=10.0)
    ap.add_argument("--out",help="Output wav (default: alongside input)")
    args=ap.parse_args()

    tokens, hdr = read_npq(args.npq_path)
    fmt = hdr.get("format","NPQ1")
    print(f"File: {args.npq_path}")
    if fmt=="NPQ1":
        T=hdr["seq_len"]; K=hdr["num_codebooks"]; fps=hdr["token_rate_fps"]
        print(f"  magic: NPQ1 | version: {hdr['version']} | dtype: {hdr['dtype_str']}")
        print(f"  codebooks: {K} | token_rate: {fps:.3f} fps | seq_len: {T} (~{T/max(fps,1e-9):.2f}s)")
        print(f"  vocab_sizes: {hdr['vocab_sizes']} | orig_bitrate: {hdr['orig_bitrate_kbps']:.1f} kbps")
    else:
        T=hdr["seq_len"]; k_mid=hdr["k_mid"]; k_side=hdr["k_side"]; fps=hdr["token_rate_fps"]
        print(f"  magic: NPQ2 | version: {hdr['version']} | dtype: {hdr['dtype_str']}")
        print(f"  mode: stereo_ms | K_mid: {k_mid} | K_side: {k_side} | token_rate: {fps:.3f} fps | seq_len: {T} (~{T/max(fps,1e-9):.2f}s)")
        print(f"  vocab_sizes: {hdr['vocab_sizes']} | orig_bitrate: {hdr['orig_bitrate_kbps']:.1f} kbps")

    repo=map_model_name(args.model); print(f"\nLoading model: {repo}")
    model=DacModel.from_pretrained(repo)
    device=torch.device(args.device)
    if args.device=="cuda" and not torch.cuda.is_available(): print("WARNING: cuda not available; falling back to cpu"); device=torch.device("cpu")
    if args.device=="mps" and not torch.backends.mps.is_available(): print("WARNING: mps not available; falling back to cpu"); device=torch.device("cpu")

    mk=getattr(getattr(model,"config",None),"n_codebooks",None)
    if fmt=="NPQ1":
        K = hdr["num_codebooks"]
        if mk is not None and K>int(mk):
            print(f"WARNING: tokens have {K} codebooks but model has {mk}. Truncating."); tokens=tokens[:,:int(mk)]; K=int(mk)
    else:
        eff_k_mid = hdr["k_mid"]; eff_k_side=hdr["k_side"]
        if mk is not None and (eff_k_mid>int(mk) or eff_k_side>int(mk)):
            if eff_k_mid>int(mk):
                print(f"WARNING: mid has {eff_k_mid} codebooks but model has {mk}. Truncating mid."); eff_k_mid=int(mk)
            if eff_k_side>int(mk):
                print(f"WARNING: side has {eff_k_side} codebooks but model has {mk}. Truncating side."); eff_k_side=int(mk)
            tokens = np.concatenate([tokens[:, :eff_k_mid], tokens[:, hdr["k_mid"]:hdr["k_mid"]+eff_k_side]], axis=1)

    print("\nDecoding...")
    if fmt=="NPQ1":
        wav=decode_tokens(tokens, model, device, fps, args.seconds_per_chunk)
    else:
        k_mid_use = eff_k_mid if 'eff_k_mid' in locals() else hdr["k_mid"]
        k_side_use = eff_k_side if 'eff_k_side' in locals() else hdr["k_side"]
        wav_stereo=decode_tokens_ms(tokens, k_mid_use, k_side_use, model, device, fps, args.seconds_per_chunk)

    model_sr=int(getattr(getattr(model,"config",None),"sampling_rate",44100))
    out=args.out or (os.path.splitext(args.npq_path)[0]+".wav")
    if fmt=="NPQ1":
        if model_sr!=44100: wav=torchaudio.transforms.Resample(model_sr,44100)(wav.unsqueeze(0)).squeeze(0)
        wav=torch.clamp(wav,-1.0,1.0); wav_i16=(wav.numpy()*32767.0).astype(np.int16)
        sf.write(out, wav_i16, 44100, subtype="PCM_16")
    else:
        # wav_stereo: [2, N]
        if model_sr!=44100:
            wav_stereo=torchaudio.transforms.Resample(model_sr,44100)(wav_stereo)
        wav_stereo=torch.clamp(wav_stereo,-1.0,1.0)
        wav_i16=(wav_stereo.numpy().T*32767.0).astype(np.int16)  # shape [N,2]
        sf.write(out, wav_i16, 44100, subtype="PCM_16")
    print(f"\nWrote WAV: {out} @ 44100 Hz 16-bit")

if __name__=="__main__": main()
