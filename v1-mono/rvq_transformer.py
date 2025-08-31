#!/usr/bin/env python3
from typing import List
import torch, torch.nn as nn, torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, T: int, device):
        return self.emb(torch.arange(T, device=device))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model*mlp_ratio), d_model),
        )
    def forward(self, x, attn_mask=None):
        h=self.ln1(x)
        a,_=self.attn(h,h,h,attn_mask=attn_mask,need_weights=False)
        x=x+a
        x=x+self.mlp(self.ln2(x))
        return x

class RVQTransformer(nn.Module):
    """
    Predict K codebooks at time t from frames < t.
    prev_tokens: [B,T,K] int64, with -1 at t=0
    returns logits_list: list of [B,T,V_k] and packed [B,T,K,Vmax]
    """
    def __init__(self, vocab_sizes: List[int], d_model: int = 512, n_layer: int = 12, n_head: int = 8, max_ctx: int = 2048):
        super().__init__()
        self.K=len(vocab_sizes); self.vocab_sizes=vocab_sizes; self.d=d_model
        self.embeddings=nn.ModuleList([nn.Embedding(v,d_model) for v in vocab_sizes])
        for e in self.embeddings: nn.init.normal_(e.weight,std=0.02)
        self.bos=nn.Parameter(torch.zeros(self.K,d_model)); nn.init.normal_(self.bos,std=0.02)
        self.pos=PositionalEmbedding(d_model,max_ctx)
        self.blocks=nn.ModuleList([TransformerBlock(d_model,n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(d_model)
        self.heads=nn.ModuleList([nn.Linear(d_model,v) for v in vocab_sizes])
        for h in self.heads: nn.init.normal_(h.weight,std=0.02); nn.init.zeros_(h.bias)

    def forward(self, prev_tokens: torch.Tensor):
        B,T,K=prev_tokens.shape; device=prev_tokens.device
        x=torch.zeros(B,T,self.d,device=device)
        for k in range(K):
            ids=prev_tokens[...,k].clone()
            mask=(ids<0); ids=ids.clamp(min=0)
            e=self.embeddings[k](ids)
            if mask.any(): e=torch.where(mask.unsqueeze(-1), self.bos[k].view(1,1,-1), e)
            x=x+e
        x=x+self.pos(T,device).unsqueeze(0)
        # causal mask (float -inf above diagonal)
        m=torch.full((T,T), float("-inf"), device=device)
        m=torch.triu(m, diagonal=1)
        for blk in self.blocks:
            x=blk(x, attn_mask=m)
        x=self.ln_f(x)
        logits=[self.heads[k](x) for k in range(K)]  # [B,T,Vk]
        Vmax=max(self.vocab_sizes)
        packed=torch.full((B,T,K,Vmax), float("-inf"), device=device)
        for k in range(K):
            v=self.vocab_sizes[k]
            packed[:,:,k,:v]=logits[k]
        return logits, packed
