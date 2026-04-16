""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import torch
import triton
import triton.language as tl

from .tuner import Phase, get_triton_autotune_decorator

@triton.jit
def _attn_fwd(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, OS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)
    
    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L)


# Optional aiter CK-tile VSA forward path. `aiter.ops.sla.sla_fwd`
# is the CK sparse-attention kernel for BLKQ ∈ {64, 128}, BLKK == 64,
# D == 128, bf16. When unavailable (aiter not installed or the SLA op
# module missing) the gate below returns False and dispatch falls
# through to the Triton _attn_fwd kernel.
try:
    from aiter.ops.sla import sla_fwd as _aiter_sla_fwd
    _CK_SLA_FWD_AVAILABLE = True
except Exception:
    _aiter_sla_fwd = None
    _CK_SLA_FWD_AVAILABLE = False

_CK_SLA_AVAILABLE = _CK_SLA_FWD_AVAILABLE


def _sla_use_ck_fwd(q: torch.Tensor, D: int, BLOCK_M: int, BLOCK_N: int) -> bool:
    """Return True if the CK forward path is available for this shape.

    The CK fwd supports BLOCK_M ∈ {64, 128}, BLOCK_N == 64, D == 128, bf16.
    Disabled when SLA_DISABLE_CK=1.
    """
    import os
    if os.environ.get("SLA_DISABLE_CK", "0") == "1":
        return False
    return (
        _CK_SLA_FWD_AVAILABLE
        and q.dtype == torch.bfloat16
        and D == 128
        and BLOCK_M in (64, 128)
        and BLOCK_N == 64
        and q.is_contiguous()
    )


_sla_use_ck = _sla_use_ck_fwd  # legacy alias


def _attention_forward(q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
    """Inference-only SLA sparse attention forward.

    Dispatches to the CK sparse attention kernel when the shape is
    supported (BLOCK_M ∈ {64, 128}, BLOCK_N == 64, D == 128, bf16),
    otherwise falls back to the Triton _attn_fwd kernel. `k_block_id`
    is unused (retained for API parity with the training branch) and
    the returned tensor does not carry gradient state.
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert lut.is_contiguous()
    assert BLOCK_M == 64 or BLOCK_M == 128
    assert BLOCK_N == 64

    B, H, L, D = q.shape
    if qk_scale is None:
        qk_scale = D ** -0.5

    M_BLOCKS = triton.cdiv(L, BLOCK_M)

    if _sla_use_ck_fwd(q, D, BLOCK_M, BLOCK_N):
        lut_bhq = lut.view(B, H, M_BLOCKS, topk).to(torch.int32).contiguous()
        o_s, _lse = _aiter_sla_fwd(
            q, k, v, lut_bhq, BLOCK_M, BLOCK_N, float(qk_scale)
        )
        return o_s

    # Triton fallback.
    o_s = torch.empty_like(v)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    grid = (M_BLOCKS, B * H)
    fwd_kernel = get_triton_autotune_decorator(_attn_fwd, L, D, Phase.FORWARD)
    fwd_kernel[grid](
        q, k, v, qk_scale, topk,
        lut, lse, o_s,
        L, M_BLOCKS,
        D, BLOCK_M, BLOCK_N,
    )
    return o_s


class _attention:
    """Inference-only façade. Retains the `.apply(...)` call shape used
    by `SparseLinearAttention.forward` so the module API is unchanged."""

    @staticmethod
    def apply(q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        return _attention_forward(q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale)
