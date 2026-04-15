"""CK sparse attention backward SNR test vs Triton autograd reference.

Runs a reference Triton backward (via SLA's autograd.Function) and
the CK VSA backward on identical inputs, then diffs dQ / dK / dV as
signal-to-noise ratio in dB.

Gate: dq/dk/dv SNR > 30 dB (bf16 accumulation headroom).
"""
import os
import sys

import torch

# Force the CK fwd dispatcher OFF so the Triton path runs end-to-end for
# the reference. We call aiter.ops.sla.sla_bwd directly for the CK path.
os.environ.setdefault("SLA_DISABLE_CK", "1")

from aiter.ops.sla import sla_bwd as ck_sla_bwd
from sparse_linear_attention.kernel import _attention
from sparse_linear_attention.utils import get_block_map


def compute_snr(x, y):
    x, y = x.float(), y.float()
    return 10 * torch.log10(
        torch.norm(x).pow(2) / (torch.norm(x - y).pow(2) + 1e-12)
    ).item()


def run(B, H, S, D, topk_ratio=0.10, BLKQ=128, BLKK=64, seed=0):
    torch.manual_seed(seed)
    qk_scale = D ** -0.5

    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    dout = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    # Build the block map once; Triton bwd needs both sparse_map and lut.
    # get_block_map only writes sparse_map when the q it's given has
    # requires_grad=True (optimization at utils.py:88), so pass q directly.
    sparse_map, lut, topk = get_block_map(q, k,
                                          topk_ratio=topk_ratio,
                                          BLKQ=BLKQ, BLKK=BLKK)
    M_BLOCKS = (S + BLKQ - 1) // BLKQ
    lut_flat = lut.view(B * H, M_BLOCKS, topk).contiguous()

    # ---- Triton reference: fwd + bwd via autograd ----
    out_tr = _attention.apply(q, k, v, sparse_map, lut_flat, topk, BLKQ, BLKK, qk_scale)
    dq_tr, dk_tr, dv_tr = torch.autograd.grad(
        outputs=(out_tr,),
        inputs=(q, k, v),
        grad_outputs=(dout,),
    )
    torch.cuda.synchronize()

    # ---- CK path: explicitly call sla_bwd on the same inputs ----
    # sla_bwd wants a sorted int32 LUT shaped [B, H, M_BLOCKS, topk].
    lut_bhq = lut.view(B, H, M_BLOCKS, topk)
    lut_sorted, _ = torch.sort(lut_bhq, dim=-1)
    lut_ck = lut_sorted.to(torch.int32).contiguous()

    # Get LSE for the CK bwd. For BLKQ=128 (Config C) we use the CK fwd
    # since it's available; for BLKQ=64 (Config A/B) the CK fwd isn't
    # instantiated at kM0=64, so we synthesize LSE via the SLA Triton
    # fwd kernel — both convention-match (log2-space) so the CK bwd
    # pipelines consume either one.
    if BLKQ == 128:
        from aiter.ops.sla import sla_fwd as ck_sla_fwd
        out_for_bwd, lse_for_bwd = ck_sla_fwd(
            q.detach().contiguous(),
            k.detach().contiguous(),
            v.detach().contiguous(),
            lut_ck,
            BLKQ,
            BLKK,
            float(qk_scale),
        )
    else:
        # BLKQ=64: cross-path route — Triton fwd produces log2-space LSE
        # (see kernel.py:82 `m_i + log2(l_i)`) which the CK bwd pipelines
        # consume directly (since P1.4 dropped the natural-log roundtrip).
        import triton
        from sparse_linear_attention.kernel import _attn_fwd
        out_for_bwd = torch.empty_like(v)
        lse_for_bwd = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        _attn_fwd[(M_BLOCKS, B * H)](
            q.detach().contiguous(), k.detach().contiguous(), v.detach().contiguous(),
            qk_scale, topk,
            lut_flat.contiguous(), lse_for_bwd, out_for_bwd,
            S, M_BLOCKS,
            D, BLKQ, BLKK,
        )

    dq_ck, dk_ck, dv_ck = ck_sla_bwd(
        dout.contiguous(),
        q.detach().contiguous(),
        k.detach().contiguous(),
        v.detach().contiguous(),
        out_for_bwd,
        lse_for_bwd,
        lut_ck,
        BLKQ,
        BLKK,
        float(qk_scale),
    )
    torch.cuda.synchronize()

    dq_snr = compute_snr(dq_tr, dq_ck)
    dk_snr = compute_snr(dk_tr, dk_ck)
    dv_snr = compute_snr(dv_tr, dv_ck)

    print(
        f"[{B},{H},{S},{D}] topk={topk}/{S//BLKK}  "
        f"dq={dq_snr:6.2f}dB  dk={dk_snr:6.2f}dB  dv={dv_snr:6.2f}dB"
    )
    print(f"   triton dq[0,0,0,:6]: {dq_tr[0,0,0,:6].float().tolist()}")
    print(f"   ck     dq[0,0,0,:6]: {dq_ck[0,0,0,:6].float().tolist()}")
    print(f"   triton dk[0,0,0,:6]: {dk_tr[0,0,0,:6].float().tolist()}")
    print(f"   ck     dk[0,0,0,:6]: {dk_ck[0,0,0,:6].float().tolist()}")


if __name__ == "__main__":
    # === BLKQ=128 sweep (Config C — full CK fwd + CK bwd) ===
    print("=== BLKQ=128 (Config C full-CK path) ===")
    run(1, 12, 16384, 128, topk_ratio=0.10, BLKQ=128, BLKK=64)
    run(4, 12, 16384, 128, topk_ratio=0.10, BLKQ=128, BLKK=64)
    run(1, 24, 16384, 128, topk_ratio=0.10, BLKQ=128, BLKK=64)
    run(1, 12, 16384, 128, topk_ratio=0.20, BLKQ=128, BLKK=64)
    run(1, 12, 16384, 128, topk_ratio=0.30, BLKQ=128, BLKK=64)
    run(1, 24, 65536, 128, topk_ratio=0.10, BLKQ=128, BLKK=64)

    # === BLKQ=64 sweep (Config A/B — Triton fwd + CK bwd cross-path) ===
    print("\n=== BLKQ=64 (Config A/B Triton-fwd + CK-bwd cross-path) ===")
    run(1, 12, 16384, 128, topk_ratio=0.10, BLKQ=64, BLKK=64)
    run(4, 12, 16384, 128, topk_ratio=0.10, BLKQ=64, BLKK=64)
    run(1, 24, 16384, 128, topk_ratio=0.10, BLKQ=64, BLKK=64)
    run(1, 12, 16384, 128, topk_ratio=0.20, BLKQ=64, BLKK=64)
    run(1, 12, 16384, 128, topk_ratio=0.30, BLKQ=64, BLKK=64)
    run(1, 24, 65536, 128, topk_ratio=0.10, BLKQ=64, BLKK=64)
