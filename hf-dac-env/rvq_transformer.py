#!/usr/bin/env python3
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt


class CausalSelfAttention(nn.Module):
    """
    Memory-efficient causal attention using PyTorch SDPA (Flash/Mem-efficient kernels on CUDA).
    No explicit T×T mask tensor is allocated.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B,T,3C]
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Flash/SDPA selection handled by PyTorch (enable kernels on CUDA)
        with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                            enable_mem_efficient=True,
                                            enable_math=True):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.drop_p if self.training else 0.0,
                is_causal=True,
            )  # [B,H,T,D]

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RVQTransformer(nn.Module):
    """
    Inputs: integer tokens [B, T, K] with -1 meaning BOS for that position.
    Per-codebook embeddings are summed per frame; learned positional embedding; causal Transformer.
    Outputs: list of K logits tensors, each [B, T, V_k].
    """
    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int = 512,
        n_layer: int = 12,
        n_head: int = 8,
        max_ctx: int = 1032,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.K = len(vocab_sizes)
        self.d_model = d_model
        self.max_ctx = max_ctx
        self.use_checkpoint = use_checkpoint

        # Per-codebook token embeddings + BOS vectors
        self.embeds = nn.ModuleList([nn.Embedding(v, d_model) for v in vocab_sizes])
        self.bos = nn.Parameter(torch.zeros(self.K, d_model))
        nn.init.normal_(self.bos, std=0.02)

        # Positional embedding (learned)
        self.pos = nn.Embedding(max_ctx, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

        # Per-codebook output heads
        self.heads = nn.ModuleList([nn.Linear(d_model, v) for v in vocab_sizes])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        tokens: [B, T, K] (int64) with -1 allowed (BOS).
        returns (logits_list[K], hidden[B,T,d])
        """
        B, T, K = tokens.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"
        if T > self.max_ctx:
            raise ValueError(f"Sequence length {T} exceeds max_ctx {self.max_ctx}")

        # Sum per-codebook embeddings, replacing -1 with BOS vectors
        x = 0.0
        for k in range(K):
            ids = tokens[..., k]  # [B,T]
            mask = ids.lt(0)
            ids_safe = ids.clamp_min(0)
            e = self.embeds[k](ids_safe)  # [B,T,d]
            if mask.any():
                e = torch.where(mask.unsqueeze(-1), self.bos[k].view(1, 1, -1), e)
            x = x + e

        # Add positions
        pos = self.pos.weight[:T].unsqueeze(0)  # [1,T,d]
        x = x + pos

        # Transformer (optional gradient checkpointing during training)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = ckpt(lambda inp, m=blk: m(inp), x, use_reentrant=False)
            else:
                x = blk(x)
        h = self.ln_f(x)

        # Heads
        logits_list = [head(h) for head in self.heads]
        return logits_list, h
