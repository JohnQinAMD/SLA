"""Per-kernel rocprof run for Stage 7 Tier 2 fwd+bwd.

Runs a minimal loop that executes the SLA fwd+bwd under whichever path is
selected via SLA_DISABLE_CK. Launch this under `rocprofv3 --kernel-trace`
and parse the resulting CSV with stage7_bwd_rocprof_agg.py.
"""
import os
import sys

sys.path.insert(0, "/work/aiter-amd")
sys.path.insert(0, "/work/SLA")

import torch
import sparse_linear_attention


B, H, S, D = 1, 24, 65536, 128  # Config C
BLKQ, BLKK = 128, 64
TOPK = 0.10

attn = sparse_linear_attention.SparseLinearAttention(
    head_dim=D, topk=TOPK, feature_map="softmax", BLKQ=BLKQ, BLKK=BLKK,
).cuda()

torch.manual_seed(0)
q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# Warm-up
for _ in range(5):
    q.grad = k.grad = v.grad = None
    out = attn(q, k, v)
    out.sum().backward()
torch.cuda.synchronize()

# Measured region
for _ in range(10):
    q.grad = k.grad = v.grad = None
    out = attn(q, k, v)
    out.sum().backward()
torch.cuda.synchronize()
