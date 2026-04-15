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

@triton.jit
def _attn_bwd_preprocess(
    OS, DOS, DELTAS,
    L,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    OS += idx_bh * L * D
    DOS += idx_bh * L * D
    DELTAS += idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_s = tl.load(OS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    do_s = tl.load(DOS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    
    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < L)

# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    Q, K, V, LSE, DELTAS,
    DOS, DQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    qkv_offset = idx_bh * L * D
    lse_offset = idx_bh * L
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DQ_ptrs = DQ + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    LUT_ptr = LUT + lut_offset

    # load Q, DOS, DOL, LSE, DELTA, S: they stay in SRAM throughout the inner loop.
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < L)
    lse = tl.load(LSE_ptrs, mask=offs_m < L, other=float("inf"))
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # P1.1 manual K/V double-buffering (same rationale as _attn_fwd).
    idx_n = tl.load(LUT_ptr + 0)
    n_mask = offs_n < L - idx_n * BLOCK_N
    k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
    v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])

    for block_idx in tl.range(topk):
        next_ok = block_idx + 1 < topk
        next_idx = tl.where(next_ok, block_idx + 1, 0)
        idx_n_next = tl.load(LUT_ptr + next_idx)
        n_mask_next = offs_n < L - idx_n_next * BLOCK_N
        k_next = tl.load(K_ptrs + idx_n_next * BLOCK_N * D, mask=n_mask_next[:, None])
        v_next = tl.load(V_ptrs + idx_n_next * BLOCK_N * D, mask=n_mask_next[:, None])

        # P1.3 fuse temporaries: inline qk into exp2 and dp into ds so the
        # compiler can release the intermediate fp32 [BLOCK_M, BLOCK_N] tiles.
        p = tl.math.exp2(tl.dot(q, k.T) * (qk_scale * 1.4426950408889634) - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)
        ds = p * (tl.dot(do_s, v.T).to(tl.float32) - delta_s[:, None])
        # Compute dQ.
        dq += tl.dot(ds.to(k.dtype), k)

        idx_n = idx_n_next
        n_mask = n_mask_next
        k = k_next
        v = v_next
    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L)
    
@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, DOS, DK, DV,
    qk_scale, KBID, LSE, DELTAS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)

    qkv_offset = idx_bh * L * D
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    lse_offset = idx_bh * L

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DK_ptrs = DK + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DV_ptrs = DV + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    KBID_ptr = KBID + kbid_offset + idx_n

    # load K, V and CK: they stay in SRAM throughout the inner loop.
    k = tl.load(K_ptrs, mask=offs_n[:, None] < L)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < L)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    for idx_m in tl.range(0, L, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < L - idx_m
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < L, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            # Compute dV.
            dv += tl.dot(pT.to(do.dtype), do)
            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)

        # Increment pointers
        Q_ptrs += BLOCK_M2 * D
        DOS_ptrs += BLOCK_M2 * D
        LSE_ptrs += BLOCK_M2
        DELTAS_ptrs += BLOCK_M2
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            KBID_ptr += N_BLOCKS

    # Write back dK, dV and dCK
    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < L)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L)
    

# Lazy import guard for the AITER CK-tile VSA path. The plan (Stage 2) adds a
# drop-in aiter.ops.sla.sla_fwd that runs CK VSA with LDS double-buffering,
# matching Triton's _attn_fwd numerically (SNR > 80 dB) and cutting kernel time
# from ~657us to ~185us at the workload of record. CK has no matching bwd, so
# the dispatcher only activates when grad is disabled.
try:
    from aiter.ops.sla import sla_fwd as _aiter_sla_fwd
    _CK_SLA_FWD_AVAILABLE = True
except Exception as _e:  # noqa: F841
    _aiter_sla_fwd = None
    _CK_SLA_FWD_AVAILABLE = False

try:
    from aiter.ops.sla import sla_bwd as _aiter_sla_bwd
    _CK_SLA_BWD_AVAILABLE = True
except Exception as _e:  # noqa: F841
    _aiter_sla_bwd = None
    _CK_SLA_BWD_AVAILABLE = False

# Backward compatibility: older call sites still read _CK_SLA_AVAILABLE.
_CK_SLA_AVAILABLE = _CK_SLA_FWD_AVAILABLE


def _sla_use_ck_fwd(q: torch.Tensor, D: int, BLOCK_M: int, BLOCK_N: int) -> bool:
    """Gate the CK fwd path.

    Option 2 added a (kM0=64, kN0=64) tile instance to the fwd codegen,
    so BLKQ=64 (Config A/B) now dispatches through the CK fwd as well.
    BLKQ=128 (Config C) continues to use the kM0=128 tile. Runtime
    dispatch between the two instances is driven by `args.block_m`
    inside the generated `fmha_vsa_fwd` entry point (seqtune property
    in codegen/ops/fmha_fwd_vsa.py).
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


# Backward-compatibility alias for older call sites that still reference
# `_sla_use_ck` as the fwd gate.
_sla_use_ck = _sla_use_ck_fwd


def _sla_use_ck_bwd(q: torch.Tensor, D: int, BLOCK_M: int, BLOCK_N: int) -> bool:
    """Gate the CK bwd path.

    Tier 2.5 Step 1 + Phase 1/2 + N2/N3 landed the split dkdv+dq CK bwd
    for SLA at (kM0=64, kN0=64). BLKQ=64 (Config A/B) uses q_scale=1
    and BLKQ=128 (Config C) uses q_scale=2 — both are supported by the
    same pipeline instantiation and the same K-major LUT transpose.

    Config C measurement (chi2812 / primus:v26.2 / MI355X):
      - CK split bwd:       33.84 ms wall  (18.06 dkdv + 12.35 dq kernel)
      - Triton autotune:    46.714 ms (published MI355X reference)
      - speedup vs ref:     1.380× (27.6% faster than reference)
      - SNR vs Triton bwd:  54-76 dB on a 6-config sweep

    The CK bwd is independent of the fwd path — it consumes log2-space
    LSE which both the CK fwd and the Triton fwd produce (Triton fwd at
    kernel.py:82 stores `m_i + log2(l_i)`, matching the CK fwd
    convention). So CK bwd dispatches even when the fwd took the
    Triton path (relevant for Config A/B where CK fwd is unavailable).

    Default is ON. Users can explicitly disable via SLA_USE_CK_BWD=0.
    """
    import os
    if os.environ.get("SLA_USE_CK_BWD", "1") != "1":
        return False
    if os.environ.get("SLA_DISABLE_CK", "0") == "1":
        return False
    return (
        _CK_SLA_BWD_AVAILABLE
        and q.dtype == torch.bfloat16
        and D == 128
        and BLOCK_M in (64, 128)
        and BLOCK_N == 64
        and q.is_contiguous()
    )


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        # k_block_id may be None for inference (skip in get_block_map under no-grad)
        assert k_block_id is None or k_block_id.is_contiguous()
        assert lut.is_contiguous()

        # We recommend the following two settings
        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64

        B, H, L, D = q.shape
        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(L, BLOCK_M)

        # CK dispatch — the fwd and bwd gates are independent.
        # `ck_fwd_ok` needs BLOCK_M=128 (fwd has no kM0=64 tile yet).
        # `ck_bwd_ok` accepts BLOCK_M in {64, 128} since the bwd pipeline
        # runs at kM0=64 natively and handles BLKQ=128 via q_scale=2.
        # This lets Config A/B (BLKQ=64) take CK bwd even though the fwd
        # still runs on Triton. Both fwds store log2-space LSE
        # (Triton at line 82, CK per pipeline comment), so the CK bwd
        # can consume either.
        needs_grad = any(t.requires_grad for t in (q, k, v))
        ck_fwd_ok = _sla_use_ck_fwd(q, D, BLOCK_M, BLOCK_N)
        ck_bwd_ok = _sla_use_ck_bwd(q, D, BLOCK_M, BLOCK_N)

        # Path A: CK fwd (+ CK bwd if under grad).
        # CK fwd is only taken when either no grad is needed OR the CK bwd
        # matches the same (BLOCK_M, BLOCK_N) — we can't emit a CK fwd
        # under grad that later falls back to Triton bwd because we'd
        # need to save k_block_id and would lose CK fwd's LSE tensor.
        if ck_fwd_ok and (not needs_grad or ck_bwd_ok):
            lut_bhq = lut.view(B, H, M_BLOCKS, topk).to(torch.int32).contiguous()
            o_s, lse_ck = _aiter_sla_fwd(
                q, k, v, lut_bhq, BLOCK_M, BLOCK_N, float(qk_scale)
            )
            if needs_grad:
                ctx.save_for_backward(q, k, v, lut_bhq, lse_ck, o_s)
                ctx.ck_bwd_path = True
            ctx.qk_scale = qk_scale
            ctx.topk = topk
            ctx.BLOCK_M = BLOCK_M
            ctx.BLOCK_N = BLOCK_N
            return o_s

        # Path B: Triton fwd (CK fwd not available, or CK bwd disabled).
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

        # save_for_backward costs ~10ms per call on MI355X via the autograd
        # dispatch; skip when grad isn't needed (e.g. inference benches).
        if needs_grad:
            if ck_bwd_ok:
                # Triton fwd + CK bwd cross-path dispatch (enables Config
                # A/B CK bwd even though the fwd stayed on Triton).
                # Triton fwd stores log2-space LSE at kernel.py:82, which
                # matches what the CK bwd pipelines consume directly.
                lut_bhq = lut.view(B, H, M_BLOCKS, topk).to(torch.int32).contiguous()
                ctx.save_for_backward(q, k, v, lut_bhq, lse, o_s)
                ctx.ck_bwd_path = True
            else:
                ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
                ctx.ck_bwd_path = False
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N

        # CK bwd dispatch. ctx.ck_bwd_path is set by forward in two cases:
        #   (a) Full CK path (Config C): ck_fwd_ok + ck_bwd_ok, both took CK.
        #   (b) Cross-path (Config A/B): Triton fwd + CK bwd — fwd stays on
        #       Triton because CK fwd doesn't have a BLOCK_M=64 tile, but
        #       the CK bwd consumes the Triton fwd's log2-space LSE directly.
        # In both cases ctx.saved_tensors is the 6-tuple
        # (q, k, v, lut_bhq, lse, o_s) with lse in log2 space. sla_bwd's
        # wrapper builds the transposed K-major LUT internally. Returns only
        # dq, dk, dv — other grad slots stay None because we took a
        # k_block_id=None branch in forward.
        if getattr(ctx, "ck_bwd_path", False):
            q, k, v, lut_bhq, lse_ck, o_s = ctx.saved_tensors
            do_s = do_s.contiguous()
            dq, dk, dv = _aiter_sla_bwd(
                do_s, q, k, v, o_s, lse_ck, lut_bhq,
                BLOCK_M, BLOCK_N, float(ctx.qk_scale),
            )
            return dq, dk, dv, None, None, None, None, None, None

        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        B, H, L, D = q.shape

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        grid = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid](
            o_s, do_s, delta_s,
            L, D, BLOCK_M,
        )

        grid = (M_BLOCKS, B * H)
        bwd_dq_kernel = get_triton_autotune_decorator(_attn_bwd_dq, L, D, Phase.BACKWARD_DQ)
        bwd_dq_kernel[grid](
            q, k, v, lse, delta_s,
            do_s, dq, lut,
            ctx.qk_scale, ctx.topk,
            L, M_BLOCKS,
            D, BLOCK_M, BLOCK_N,
        )

        grid = (N_BLOCKS, B * H)
        bwd_dkdv_kernel = get_triton_autotune_decorator(_attn_bwd_dkdv, L, D, Phase.BACKWARD_DKDV)
        bwd_dkdv_kernel[grid](
            q, k, v, do_s, dk, dv,
            ctx.qk_scale, k_block_id, lse, delta_s,
            L, M_BLOCKS, N_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
        )

        return dq, dk, dv, None, None, None, None, None, None
