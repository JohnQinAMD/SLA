"""Stage 7 Tier 2: end-to-end fwd + bwd wall-clock bench CK vs Triton.

Runs `SparseLinearAttention(q,k,v).sum().backward()` end-to-end so we
exercise both the forward and backward via autograd — the exact path a
training step takes. Compares CK (ck_fwd + ck_bwd) against Triton.
"""
import os
import sys
import statistics
import time

sys.path.insert(0, "/work/aiter-amd")
sys.path.insert(0, "/work/SLA")

import torch
import sparse_linear_attention


torch.manual_seed(0)
# Config C from SLA/trition-autotune-benchmark-result.md (MI355X): published
# Triton autotune numbers are fwd 11.407 | bwd 46.714 ms.
B, H, S, D = 1, 24, 65536, 128
BLKQ, BLKK = 128, 64
TOPK = 0.10
print(f"workload: B={B} H={H} S={S} D={D} BLKQ={BLKQ} BLKK={BLKK} "
      f"topk={TOPK}", flush=True)

attn = sparse_linear_attention.SparseLinearAttention(
    head_dim=D, topk=TOPK, feature_map="softmax", BLKQ=BLKQ, BLKK=BLKK,
).cuda()


def make_inputs():
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    return q, k, v


def bench(label, use_ck: bool, iters: int = 30, warmup: int = 5):
    if use_ck:
        os.environ.pop("SLA_DISABLE_CK", None)
        os.environ.pop("SLA_DISABLE_CK_BWD", None)
    else:
        os.environ["SLA_DISABLE_CK"] = "1"
        os.environ["SLA_DISABLE_CK_BWD"] = "1"

    q, k, v = make_inputs()
    for _ in range(warmup):
        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(times)
    mn = min(times)
    print(f"  {label:16s}  median={med:7.3f} ms  min={mn:7.3f} ms  "
          f"({iters}-iter fwd+bwd)", flush=True)
    return med


print("\n-- wall-clock fwd+bwd end-to-end --")
tri = bench("triton", use_ck=False)
ck = bench("ck", use_ck=True)
print(f"\nspeedup CK vs Triton: {tri / ck:.2f}x  "
      f"(reduction: {(1 - ck / tri) * 100:.1f}%)")
