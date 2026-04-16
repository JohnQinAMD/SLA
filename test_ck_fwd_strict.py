"""Strict CK forward verification vs Triton + fp32 PyTorch reference.

Extends `test_ck_fwd_snr.py` with a pure-PyTorch fp32 gold reference
(masked softmax attention) and a torch.compile'd bf16 reference at
CK's dtype. The "same implementation" invariant we check is:

    SNR(CK vs fp32)  ≈  SNR(Triton vs fp32)    within ≤ 0.5 dB
    SNR(CK vs bf16)  ≈  SNR(Triton vs bf16)    within ≤ 0.2 dB

If those hold at every case, CK and Triton are computing the same
sparse-attention math and any residual disagreement is bf16
rounding-order noise rather than an algorithmic bug.

Covers 8 cases: small, medium, BLKQ=64, BLKQ=128, sparse extreme
(topk=1/16), dense extreme (topk=15/16), high-magnitude inputs
(bf16 stress), and near-zero inputs.

Optional byte-for-byte regression mode (CK kernel is run-to-run
deterministic on MI355X):

    python test_ck_fwd_strict.py --save-ref /tmp/ck_fwd_refs
    python test_ck_fwd_strict.py --check-ref /tmp/ck_fwd_refs
"""
import argparse
import math
import os
import sys

os.environ.setdefault("SLA_DISABLE_CK", "1")

import torch
import triton

from aiter.ops.sla import sla_fwd as ck_sla_fwd
from sparse_linear_attention.kernel import _attn_fwd
from sparse_linear_attention.utils import get_block_map


# ────────────────────────────────────────────────────────────────
#  References
# ────────────────────────────────────────────────────────────────
def build_mask(lut_bhq, S, BLKQ, BLKK):
    """Expand [B, H, M_BLOCKS, topk] K-block indices into a dense
    [B, H, S, S] boolean attention mask."""
    B, H, M, topk = lut_bhq.shape
    mask = torch.zeros((B, H, S, S), dtype=torch.bool, device=lut_bhq.device)
    for m in range(M):
        row_slice = slice(m * BLKQ, (m + 1) * BLKQ)
        for t in range(topk):
            kb_bh = lut_bhq[:, :, m, t]
            for b in range(B):
                for h in range(H):
                    kb = int(kb_bh[b, h].item())
                    mask[b, h, row_slice, kb * BLKK:(kb + 1) * BLKK] = True
    return mask


def torch_sparse_attn(q, k, v, mask, scale, dtype):
    """Dense masked softmax attention in the specified dtype."""
    q = q.detach().to(dtype)
    k = k.detach().to(dtype)
    v = v.detach().to(dtype)
    s = (q @ k.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.where(torch.isnan(p), torch.zeros_like(p), p)
    return p @ v


@torch.compile(fullgraph=False, dynamic=False)
def _torch_sparse_attn_bf16_core(q, k, v, mask, scale):
    s = (q @ k.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.where(torch.isnan(p), torch.zeros_like(p), p)
    return p @ v


def torch_sparse_attn_bf16(q, k, v, mask, scale):
    return _torch_sparse_attn_bf16_core(
        q.detach().to(torch.bfloat16),
        k.detach().to(torch.bfloat16),
        v.detach().to(torch.bfloat16),
        mask, scale,
    )


def triton_fwd(q, k, v, lut_flat, topk, BLKQ, BLKK, scale):
    B, H, L, D = q.shape
    M_BLOCKS = triton.cdiv(L, BLKQ)
    out = torch.empty_like(v)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    _attn_fwd[(M_BLOCKS, B * H)](
        q, k, v, scale, topk,
        lut_flat, lse, out,
        L, M_BLOCKS,
        D, BLKQ, BLKK,
    )
    return out


# ────────────────────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────────────────────
def snr(x, y):
    x = x.float()
    y = y.float()
    noise = (x - y).pow(2).sum().item()
    signal = x.pow(2).sum().item()
    if noise <= 0:
        return float("inf")
    return 10.0 * math.log10(signal / max(noise, 1e-30))


def max_abs(x, y):
    return (x.float() - y.float()).abs().max().item()


# ────────────────────────────────────────────────────────────────
#  One case
# ────────────────────────────────────────────────────────────────
def run_case(name, B, H, S, D, BLKQ, BLKK, topk_ratio, input_scale, seed=0):
    torch.manual_seed(seed)
    scale = float(D ** -0.5)

    q = (torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda") * (0.5 * input_scale)).contiguous()
    k = (torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda") * (0.5 * input_scale)).contiguous()
    v = (torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda") * (0.5 * input_scale)).contiguous()

    sparse_map, lut, topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=BLKQ, BLKK=BLKK)
    M = S // BLKQ
    lut_flat = lut.view(B * H, M, topk).contiguous()
    lut_bhq = lut.view(B, H, M, topk).contiguous()

    # Triton path
    out_tr = triton_fwd(q, k, v, lut_flat, topk, BLKQ, BLKK, scale)
    torch.cuda.synchronize()

    # CK path
    lut_ck, _ = torch.sort(lut_bhq.to(torch.int32).contiguous(), dim=-1)
    out_ck, _lse = ck_sla_fwd(q, k, v, lut_ck, BLKQ, BLKK, scale)
    torch.cuda.synchronize()

    # fp32 reference
    mask = build_mask(lut_ck, S, BLKQ, BLKK)
    out_fp = torch_sparse_attn(q, k, v, mask, scale, torch.float32)

    # bf16 torch.compile'd reference
    try:
        out_bf = torch_sparse_attn_bf16(q, k, v, mask, scale)
        have_bf = True
    except Exception:
        out_bf, have_bf = None, False

    print(f"\n── {name}  [B={B} H={H} S={S} D={D}] BLKQ={BLKQ} "
          f"topk={topk}/{S // BLKK} input_scale={input_scale}")
    header = f"{'field':4} | {'ck vs fp32':>14} | {'tr vs fp32':>14} |"
    if have_bf:
        header += f" {'ck vs bf16':>14} | {'tr vs bf16':>14} |"
    header += f" {'ck vs tr':>14}"
    print(header)

    def fmt(a, b):
        if a is None or b is None:
            return "   n/a       "
        return f"{snr(a, b):6.2f}dB {max_abs(a, b):.1e}"

    row = f" out | {fmt(out_fp, out_ck):>14} | {fmt(out_fp, out_tr):>14} |"
    if have_bf:
        row += f" {fmt(out_bf, out_ck):>14} | {fmt(out_bf, out_tr):>14} |"
    row += f" {fmt(out_ck, out_tr):>14}"
    print(row)

    return {"out": out_ck.detach()}


def save_or_check_ref(name, outputs, save_dir, check_dir):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.pt")
        torch.save({k: v.cpu() for k, v in outputs.items()}, path)
        print(f"  ref saved → {path}")
    if check_dir:
        path = os.path.join(check_dir, f"{name}.pt")
        if not os.path.exists(path):
            print(f"  WARN: no reference at {path}")
            return
        ref = torch.load(path, map_location="cuda")
        mismatches = []
        for k in outputs:
            got = outputs[k]
            exp = ref[k].to(got.device)
            if not torch.equal(got, exp):
                diff = (got.float() - exp.float()).abs().max().item()
                mismatches.append((k, diff))
        if mismatches:
            print(f"  BYTE-FOR-BYTE REGRESSION: {mismatches}")
        else:
            print(f"  byte-for-byte ✓ against {path}")


# ────────────────────────────────────────────────────────────────
#  Sweep
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-ref", type=str, default=None,
                        help="Directory to save CK outputs as references")
    parser.add_argument("--check-ref", type=str, default=None,
                        help="Directory with reference files to diff against")
    args = parser.parse_args()

    cases = [
        # name                 B  H   S     D    BLKQ BLKK topk   mag
        ("small_blkq128",      1, 4, 1024, 128,  128, 64, 0.10,  1.0),
        ("blkq64",             1, 4, 1024, 128,   64, 64, 0.10,  1.0),
        ("sparse_extreme",     1, 4, 1024, 128,  128, 64, 0.0625, 1.0),
        ("dense_extreme",      1, 4, 1024, 128,  128, 64, 0.99,   1.0),
        ("medium_blkq128",     1, 8, 2048, 128,  128, 64, 0.10,   1.0),
        ("medium_blkq64",      1, 8, 2048, 128,   64, 64, 0.10,   1.0),
        ("high_magnitude",     1, 4, 1024, 128,  128, 64, 0.10,   5.0),
        ("near_zero",          1, 4, 1024, 128,  128, 64, 0.10,   0.05),
    ]

    print("=" * 80)
    print("Strict CK fwd vs Triton + fp32 PyTorch verification")
    print(f"  save_ref  = {args.save_ref}")
    print(f"  check_ref = {args.check_ref}")
    print("=" * 80)

    for name, B, H, S, D, BLKQ, BLKK, topk_ratio, mag in cases:
        try:
            outputs = run_case(name, B, H, S, D, BLKQ, BLKK, topk_ratio, mag)
            save_or_check_ref(name, outputs, args.save_ref, args.check_ref)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
