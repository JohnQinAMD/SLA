"""Forward-pass wall-clock bench: CK vs Triton.

Runs SparseLinearAttention forward under torch.no_grad -- the
inference path — and compares CK against Triton for Config B
(BLKQ=64) and Config C (BLKQ=128).

Enables SLA_ENABLE_TRITON_AUTOTUNING=1 so the Triton fallback is
measured at its autotuned best rather than the default heuristic
config.
"""
import os

os.environ.setdefault("SLA_ENABLE_TRITON_AUTOTUNING", "1")

import statistics

import torch
import sparse_linear_attention


torch.manual_seed(0)

# MI355X Triton autotune reference with BLKQ=64 and BLKQ=128:
#   Config B  autotune=YES BLKQ=64  BLKK=64   fwd 13.287
#   Config C  autotune=YES BLKQ=128 BLKK=64   fwd 11.407
TRITON_REFERENCE_FWD = {
    "B": 13.287,
    "C": 11.407,
}

B, H, S, D = 1, 24, 65536, 128
TOPK = 0.10

CONFIGS = [
    ("B", 64, 64),
    ("C", 128, 64),
]


def make_inputs():
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def bench(attn, label, use_ck: bool, iters: int = 30, warmup: int = 15):
    if use_ck:
        os.environ.pop("SLA_DISABLE_CK", None)
    else:
        os.environ["SLA_DISABLE_CK"] = "1"

    q, k, v = make_inputs()
    with torch.no_grad():
        for _ in range(warmup):
            attn(q, k, v)
        torch.cuda.synchronize()

        fwds = []
        for _ in range(iters):
            torch.cuda.synchronize()
            e1 = torch.cuda.Event(True); e2 = torch.cuda.Event(True)
            e1.record()
            attn(q, k, v)
            e2.record()
            torch.cuda.synchronize()
            fwds.append(e1.elapsed_time(e2))

    fwd_med = statistics.median(fwds)
    print(f"  {label:16s}  fwd={fwd_med:7.3f} ms  ({iters}-iter)", flush=True)
    return fwd_med


for name, BLKQ, BLKK in CONFIGS:
    ref_fwd = TRITON_REFERENCE_FWD[name]
    print(f"\n=== Config {name}  "
          f"B={B} H={H} S={S} D={D} BLKQ={BLKQ} BLKK={BLKK} topk={TOPK} "
          f"(Triton autotune ref: fwd {ref_fwd} ms) ===", flush=True)
    attn = sparse_linear_attention.SparseLinearAttention(
        head_dim=D, topk=TOPK, feature_map="softmax", BLKQ=BLKQ, BLKK=BLKK,
    ).cuda()
    tri_fwd = bench(attn, "triton", use_ck=False)
    ck_fwd = bench(attn, "ck",     use_ck=True)
    print(
        f"  speedup fwd: CK {ck_fwd:.2f} vs Triton {tri_fwd:.2f} "
        f"= {tri_fwd / ck_fwd:.3f}x  "
        f"(vs Triton ref {ref_fwd}: {ref_fwd / ck_fwd:.3f}x)",
        flush=True,
    )
