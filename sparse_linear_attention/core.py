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
import torch.nn as nn
import torch.nn.functional as F

from .kernel import _attention
from .utils import get_block_map


# Fuse the linear-attention path via torch.compile. This brings inductor
# in to merge softmax + bmm + norm into ~3 kernels instead of ~6 dispatches and
# picks a better K-reduction GEMM for the [B*H, D, D] = K @ V shape.
# donated_buffer=False keeps the compiled backward compatible with
# retain_graph=True (used by the SLA bench and by typical training loops that
# do gradient accumulation).
import torch._functorch.config as _ft_config
_ft_config.donated_buffer = False


@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _calc_linear(qf, kf, v):
    kvsum = kf.transpose(-1, -2) @ v
    ksum = torch.sum(kf, dim=-2, keepdim=True)
    return (qf @ kvsum) / (1e-5 + (qf * ksum).sum(dim=-1, keepdim=True))


# Fuse the entire post-sparse non-sparse chain — feature_maps,
# linear-attention, proj_l, epilogue — under one torch.compile so inductor
# can pipeline across kernel boundaries. Inputs are already bf16 here.
@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _post_sparse_softmax(q_bf16, k_bf16, v_bf16, o_s, weight_bf16, bias_bf16, out_dtype):
    qf = torch.softmax(q_bf16, dim=-1)
    kf = torch.softmax(k_bf16, dim=-1)
    kvsum = kf.transpose(-1, -2) @ v_bf16
    ksum = torch.sum(kf, dim=-2, keepdim=True)
    o_l = (qf @ kvsum) / (1e-5 + (qf * ksum).sum(dim=-1, keepdim=True))
    o_l_proj = torch.matmul(o_l, weight_bf16.T) + bias_bf16
    return (o_s + o_l_proj).to(out_dtype)


# Fallback for non-softmax feature maps (elu, relu) — keeps the smaller
# fusions live but doesn't fuse across the feature_map boundary.
@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _proj_and_add(o_l, weight_bf16, bias_bf16, o_s, out_dtype):
    o_l_proj = torch.matmul(o_l, weight_bf16.T) + bias_bf16
    return (o_s + o_l_proj).to(out_dtype)


class SparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
        '''
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def _proj_l_bf16_views(self):
        """Cache bf16 views of the fp32 proj_l weights to avoid casting
        ~32 KB per forward call. Invalidated automatically if dtype changes
        (the cache key is the (dtype, device) tuple)."""
        key = (self.dtype, self.proj_l.weight.device)
        cache = getattr(self, "_proj_l_cache", None)
        if cache is None or cache[0] != key:
            with torch.no_grad():
                w = self.proj_l.weight.to(self.dtype)
                b = self.proj_l.bias.to(self.dtype)
            self._proj_l_cache = (key, w, b)
        return self._proj_l_cache[1], self._proj_l_cache[2]

    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
        '''
        dtype = q.dtype
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        o_s = _attention.apply(q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)

        weight_bf16, bias_bf16 = self._proj_l_bf16_views()

        # Fast path: when feature_map is softmax (the default), fuse the
        # entire post-sparse non-sparse chain (softmax × 2 + linear-attention
        # + proj_l + epilogue) into a single torch.compile region so inductor
        # can pipeline across the kernel boundaries.
        if self.feature_map_q is self.feature_map_k and \
                getattr(self.feature_map_q, "__name__", "") == "softmax_feature_map":
            o = _post_sparse_softmax(q, k, v, o_s, weight_bf16, bias_bf16, dtype)
        else:
            qf = self.feature_map_q(q)
            kf = self.feature_map_k(k)
            o_l = _calc_linear(qf, kf, v)
            o = _proj_and_add(o_l, weight_bf16, bias_bf16, o_s, dtype)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o
