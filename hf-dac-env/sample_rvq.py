#!/usr/bin/env python3
import argparse, os, glob
import numpy as np
import torch, torchaudio, soundfile as sf
from transformers import DacModel

from rvq_transformer import RVQTransformer

def map_model_name(name: str) -> str:
    return {"16khz":"descript/dac_16khz","24khz":"descript/dac_24khz","44khz":"descript/dac_44khz","44.1khz":"descript/dac_44khz"}.get(name.lower(), name)

def pick_ckpt(path_or_dir: str, which: str = "best") -> str:
    """If a directory is given, return its best or latest pointer; else return the file."""
    if os.path.isdir(path_or_dir):
        cand = os.path.join(path_or_dir, f"*_{which}.pt")
        matches = sorted(glob.glob(cand))
        if not matches:
            raise FileNotFoundError(f"No *_{which}.pt in {path_or_dir}")
        return matches[0]
    if os.path.isfile(path_or_dir):
        return path_or_dir
    raise FileNotFoundError(path_or_dir)

@torch.no_grad()
def generate_tokens_sliding(model: RVQTransformer, K: int, vocab_sizes, frames: int, ctx: int,
                            temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, device="cuda"):
    device=torch.device(device); model.eval().to(device)
    B=1; prev=torch.full((B,1,K),-1,dtype=torch.long,device=device)
    out=[]
    for _ in range(frames):
        if prev.shape[1] > ctx: prev = prev[:, -ctx:, :]
        logits_list,_=model(prev)
        toks=[]
        for k in range(K):
            v=vocab_sizes[k]
            logits=logits_list[k][:,-1,:v] / max(1e-6, temperature)
            if top_k>0:
                vals,idx=torch.topk(logits, k=min(top_k,v), dim=-1)
                mask=torch.full_like(logits, float("-inf")); mask.scatter_(1,idx,vals); logits=mask
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs=torch.softmax(sorted_logits,dim=-1)
                cdf=torch.cumsum(probs,dim=-1)
                cutoff=(cdf>top_p).float().argmax(dim=-1,keepdim=True)
                thresh=sorted_logits.gather(1,cutoff)
                logits=torch.where(logits>=thresh, logits, torch.full_like(logits, float("-inf")))
            probs=torch.softmax(logits,dim=-1)
            idx=torch.multinomial(probs,1)
            toks.append(idx)
        toks=torch.cat(toks,dim=1)  # [B,K]
        out.append(toks)
        prev=torch.cat([prev, toks.unsqueeze(1)], dim=1)
    return torch.cat(out, dim=0).cpu().numpy()  # [T,K]

def decode_with_dac(tokens_TK: np.ndarray, repo: str, sr_out: int = 44100, device="cuda"):
    model=DacModel.from_pretrained(map_model_name(repo))
    device=torch.device(device); model.to(device).eval()
    codes=torch.from_numpy(tokens_TK.T).long().unsqueeze(0).to(device)  # [1,K,T]
    with torch.inference_mode():
        wav=model.decode(audio_codes=codes).audio_values.squeeze().cpu()
    model_sr=int(getattr(getattr(model,"config",None),"sampling_rate",44100))
    if model_sr!=sr_out: wav=torchaudio.transforms.Resample(model_sr,sr_out)(wav.unsqueeze(0)).squeeze(0)
    wav=torch.clamp(wav,-1.0,1.0)
    return wav.numpy(), sr_out

def main():
    ap=argparse.ArgumentParser(description="Sample tokens from a trained RVQTransformer and decode with DAC.")
    ap.add_argument("--ckpt", required=True, help="checkpoint file OR directory containing *_best.pt / *_latest.pt")
    ap.add_argument("--which", default="best", choices=["best","latest"], help="when --ckpt is a directory")
    ap.add_argument("--frames", type=int, default=1290, help="~frames to generate (1290 ≈ 15s @ 86 fps)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--dac_repo", default="44khz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="gen.wav")
    args=ap.parse_args()

    ckpt_path = pick_ckpt(args.ckpt, args.which)
    print("loading", ckpt_path)
    ckpt=torch.load(ckpt_path, map_location="cpu")
    cfg=ckpt["cfg"]; vocab_sizes=ckpt["vocab_sizes"]; K=min(len(vocab_sizes), cfg["codebooks"])

    model=RVQTransformer(vocab_sizes=vocab_sizes[:K], d_model=cfg["d_model"],
                         n_layer=cfg["layers"], n_head=cfg["heads"], max_ctx=cfg["ctx"]+8).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()

    tokens_TK=generate_tokens_sliding(model,K,vocab_sizes[:K],frames=args.frames,ctx=cfg["ctx"],
                                      temperature=args.temperature,top_k=args.top_k,top_p=args.top_p,device=args.device)
    wav,sr=decode_with_dac(tokens_TK,repo=args.dac_repo,sr_out=44100,device=args.device)
    sf.write(args.out,(wav*32767).astype(np.int16),sr,subtype="PCM_16")
    print("wrote",args.out,"len≈",len(wav)/sr,"s")

if __name__=="__main__": main()
