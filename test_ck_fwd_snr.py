"""CK sparse attention forward SNR test vs Triton reference.

Runs `_attention.apply(...)` twice on identical inputs — once via the
CK path (`SLA_DISABLE_CK=0`) and once via the Triton path
(`SLA_DISABLE_CK=1`) — then diffs the output tensors in dB SNR.

Gate: out SNR > 40 dB.
"""
import math
import os
import sys

import torch
import triton

from sparse_linear_attention.kernel import _attention, _attn_fwd
from sparse_linear_attention.utils import get_block_map


def compute_snr(x, y):
    x, y = x.float(), y.float()
    return (
        10 * torch.log10(x.norm().pow(2) / ((x - y).norm().pow(2) + 1e-12))
    ).item()


def _triton_fwd(q, k, v, lut_flat, topk, BLKQ, BLKK, qk_scale):
    """Run Triton _attn_fwd directly (bypassing the CK dispatcher)."""
    B, H, L, D = q.shape
    M_BLOCKS = triton.cdiv(L, BLKQ)
    out = torch.empty_like(v)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    _attn_fwd[(M_BLOCKS, B * H)](
        q, k, v, qk_scale, topk,
        lut_flat, lse, out,
        L, M_BLOCKS,
        D, BLKQ, BLKK,
    )
    return out


def run(B, H, S, D, topk_ratio=0.10, BLKQ=128, BLKK=64, seed=0):
    torch.manual_seed(seed)
    qk_scale = D ** -0.5

    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    sparse_map, lut, topk = get_block_map(
        q, k, topk_ratio=topk_ratio, BLKQ=BLKQ, BLKK=BLKK,
    )
    M_BLOCKS = (S + BLKQ - 1) // BLKQ
    lut_flat = lut.view(B * H, M_BLOCKS, topk).contiguous()

    # Triton reference
    tri = _triton_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(),
        lut_flat, topk, BLKQ, BLKK, qk_scale,
    )
    torch.cuda.synchronize()

    # CK path via the dispatcher
    os.environ.pop("SLA_DISABLE_CK", None)
    ck = _attention.apply(
        q.contiguous(), k.contiguous(), v.contiguous(),
        sparse_map, lut_flat, topk, BLKQ, BLKK, qk_scale,
    )
    torch.cuda.synchronize()

    snr = compute_snr(tri, ck)
    print(f"[{B},{H},{S},{D}] BLKQ={BLKQ} topk={topk}/{S // BLKK}  out SNR={snr:6.2f} dB")
    return snr


if __name__ == "__main__":
    SHAPES = [
        (1, 12, 16384, 128, 128, 0.10),
        (4, 12, 16384, 128, 128, 0.10),
        (1, 24, 16384, 128, 128, 0.10),
        (1, 24, 65536, 128, 128, 0.10),
        (1, 12, 16384, 128,  64, 0.10),
        (1, 24, 65536, 128,  64, 0.10),
    ]
    print("=== CK fwd vs Triton fwd SNR ===")
    worst = float("inf")
    for B, H, S, D, BLKQ, topk_ratio in SHAPES:
        snr = run(B, H, S, D, topk_ratio=topk_ratio, BLKQ=BLKQ, BLKK=64)
        worst = min(worst, snr)
    print(f"\nWorst SNR: {worst:.2f} dB")
    if worst < 40.0:
        print(f"FAIL: worst SNR below 40 dB")
        sys.exit(1)
    print("OK")
