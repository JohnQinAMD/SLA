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


@triton.jit
def compress_kernel(
    X, XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _smooth_k(k):
    """Smooth-K (SageAttention): subtract per-(B,H) mean from K.

    Wrapped in torch.compile so inductor can fuse the mean reduction with
    the subtract into one (or two) kernel(s) without a large intermediate
    that the eager `k - torch.mean(k, dim=-2, keepdim=True)` allocates.
    """
    return k - torch.mean(k, dim=-2, keepdim=True)


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    # smooth-k technique in SageAttention: subtract per-(B,H) mean from k
    # before pooling. Wrapped in torch.compile to avoid a large intermediate.
    # Set SLA_SKIP_SMOOTH_K=1 to bypass smooth-k
    # (saves preprocessing time at the cost of attention selection quality —
    # validate downstream before enabling in production).
    import os as _os
    if _os.environ.get("SLA_SKIP_SMOOTH_K", "0") == "1":
        arg_k = k
    else:
        arg_k = _smooth_k(k)
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    # `sparse_map` (`k_block_id`) is only consumed by the backward pass.
    # Skip the zeros_like + scatter_ when grad is not needed (inference).
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
        sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
        sparse_map.scatter_(-1, lut, 1)
    else:
        sparse_map = None
    return sparse_map, lut, topk
