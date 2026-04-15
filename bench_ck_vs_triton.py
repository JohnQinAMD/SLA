"""End-to-end fwd + bwd wall-clock bench: CK vs Triton.

Runs `SparseLinearAttention(q,k,v).sum().backward()` end-to-end so we
exercise both the forward and backward via autograd — the exact path a
training step takes. Compares CK (ck_fwd + ck_bwd) against Triton for
both Config B (BLKQ=64) and Config C (BLKQ=128) from the Triton
autotune reference at triton-autotune-reference.md.
"""
import os
import statistics
import time

import torch
import sparse_linear_attention


torch.manual_seed(0)

# MI355X Triton autotune reference from triton-autotune-reference.md:
#   Config A  autotune=NO  BLKQ=64  BLKK=64   fwd 23.898  bwd 112.533
#   Config B  autotune=YES BLKQ=64  BLKK=64   fwd 13.287  bwd  47.560
#   Config C  autotune=YES BLKQ=128 BLKK=64   fwd 11.407  bwd  46.714
TRITON_REFERENCE = {
    "B": (13.287, 47.560),
    "C": (11.407, 46.714),
}

B, H, S, D = 1, 24, 65536, 128
TOPK = 0.10

CONFIGS = [
    ("B", 64, 64),
    ("C", 128, 64),
]


def make_inputs():
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    return q, k, v


def bench(attn, label, use_ck: bool, iters: int = 30, warmup: int = 5):
    if use_ck:
        os.environ.pop("SLA_DISABLE_CK", None)
        os.environ.pop("SLA_DISABLE_CK_BWD", None)
        os.environ["SLA_USE_CK_BWD"] = "1"
    else:
        os.environ["SLA_DISABLE_CK"] = "1"
        os.environ["SLA_DISABLE_CK_BWD"] = "1"
        os.environ.pop("SLA_USE_CK_BWD", None)

    q, k, v = make_inputs()
    for _ in range(warmup):
        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()

    # Separate fwd and bwd timing via cuda events; the perf_counter
    # total gives the end-to-end wall for the fwd+bwd pair.
    totals = []
    fwds = []
    bwds = []
    for _ in range(iters):
        q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        ef1 = torch.cuda.Event(True); ef2 = torch.cuda.Event(True)
        eb1 = torch.cuda.Event(True); eb2 = torch.cuda.Event(True)
        t0 = time.perf_counter()
        ef1.record()
        out = attn(q, k, v)
        ef2.record()
        loss = out.sum()
        eb1.record()
        loss.backward()
        eb2.record()
        torch.cuda.synchronize()
        totals.append((time.perf_counter() - t0) * 1000.0)
        fwds.append(ef1.elapsed_time(ef2))
        bwds.append(eb1.elapsed_time(eb2))

    fwd_med = statistics.median(fwds)
    bwd_med = statistics.median(bwds)
    total_med = statistics.median(totals)
    print(
        f"  {label:16s}  total={total_med:7.3f}  fwd={fwd_med:7.3f}  bwd={bwd_med:7.3f} ms "
        f"({iters}-iter)",
        flush=True,
    )
    return fwd_med, bwd_med, total_med


for name, BLKQ, BLKK in CONFIGS:
    ref_fwd, ref_bwd = TRITON_REFERENCE[name]
    print(f"\n=== Config {name}  "
          f"B={B} H={H} S={S} D={D} BLKQ={BLKQ} BLKK={BLKK} topk={TOPK} "
          f"(Triton autotune ref: fwd {ref_fwd} | bwd {ref_bwd}) ===",
          flush=True)
    attn = sparse_linear_attention.SparseLinearAttention(
        head_dim=D, topk=TOPK, feature_map="softmax", BLKQ=BLKQ, BLKK=BLKK,
    ).cuda()
    tri_fwd, tri_bwd, tri_total = bench(attn, "triton", use_ck=False)
    ck_fwd, ck_bwd, ck_total = bench(attn, "ck", use_ck=True)
    print(
        f"  speedup bwd:    CK {ck_bwd:.2f} vs Triton {tri_bwd:.2f} "
        f"= {tri_bwd / ck_bwd:.3f}x  "
        f"(vs Triton ref {ref_bwd}: {ref_bwd / ck_bwd:.3f}x)",
        flush=True,
    )
    print(
        f"  speedup total:  CK {ck_total:.2f} vs Triton {tri_total:.2f} "
        f"= {tri_total / ck_total:.3f}x",
        flush=True,
    )
