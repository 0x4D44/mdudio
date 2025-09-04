#!/usr/bin/env python3
# Basic checks: dataset shapes, model fwd/grad on tiny batch
from npq_dataset import list_npq, NPQWindowDataset, read_npq
from rvq_transformer import RVQTransformer
import torch, random

files=list_npq(["npq_out/*.npq"])
assert files, "No NPQ files in npq_out/"
# dataset
ds=NPQWindowDataset(files, ctx_frames=256, min_K=9, max_windows_per_file=2)
x=ds[0]; assert x["inputs"].shape==x["targets"].shape==(256, x["K"]); assert (x["inputs"][0]==-1).all()
print("dataset OK")
# model fwd/back
K=9; vocab=[1024]*K
m=RVQTransformer(vocab, d_model=256, n_layer=4, n_head=4, max_ctx=264).cuda()
B=2; batch=[ds[i] for i in range(min(len(ds),B))]
inp=torch.stack([b["inputs"][:,:K] for b in batch]).cuda()
tgt=torch.stack([b["targets"][:,:K] for b in batch]).cuda()
logits,_=m(inp)
loss=sum(torch.nn.functional.cross_entropy(logits[k].reshape(B*256,1024), tgt[...,k].reshape(B*256).clamp(min=0)) for k in range(K))/K
loss.backward()
print("forward/backward OK, loss=", float(loss))
