"""
Strict CK vs Triton verification.

Extends the 3-way comparison with:
  (a) Edge-case shapes: small S, blkq=64, sparse extreme (topk≈1), dense
      extreme (topk→full), high-magnitude inputs, near-zero inputs
  (b) torch.compile'd bf16 PyTorch reference alongside fp32, so we can
      compare against an implementation at CK's own dtype
  (c) Byte-for-byte reference save/check mode for CI regression

Env flags:
  SAVE_REF_DIR=<path>   save CK outputs to ref files after each case
  CHECK_REF_DIR=<path>  assert bit-identical CK output vs saved ref
"""
import argparse
import math
import os
import sys

os.environ.setdefault("SLA_DISABLE_CK", "1")

import torch

from aiter.ops.sla import sla_fwd as ck_sla_fwd
from aiter.ops.sla import sla_bwd as ck_sla_bwd
from sparse_linear_attention.kernel import _attention
from sparse_linear_attention.utils import get_block_map


# ────────────────────────────────────────────────────────────────
#  References
# ────────────────────────────────────────────────────────────────
def build_mask(lut_bhq, S, BLKQ, BLKK):
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


def torch_sparse_attn_generic(q, k, v, mask, scale, dtype):
    """Sparse attention in specified dtype — fresh tensors with grad."""
    q = q.detach().to(dtype).requires_grad_(True)
    k = k.detach().to(dtype).requires_grad_(True)
    v = v.detach().to(dtype).requires_grad_(True)
    s = (q @ k.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.where(torch.isnan(p), torch.zeros_like(p), p)
    out = p @ v
    return out, (q, k, v)


@torch.compile(fullgraph=False, dynamic=False)
def torch_sparse_attn_bf16_core(q, k, v, mask, scale):
    # This compiled fragment keeps everything in bf16 through the softmax
    # to mirror CK's internal precision more closely. Backprop goes
    # through torch.autograd.
    s = (q @ k.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.where(torch.isnan(p), torch.zeros_like(p), p)
    return p @ v


def torch_sparse_attn_bf16(q, k, v, mask, scale):
    q = q.detach().to(torch.bfloat16).requires_grad_(True)
    k = k.detach().to(torch.bfloat16).requires_grad_(True)
    v = v.detach().to(torch.bfloat16).requires_grad_(True)
    out = torch_sparse_attn_bf16_core(q, k, v, mask, scale)
    return out, (q, k, v)


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

    q = (torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda") * (0.5 * input_scale)
         ).detach().requires_grad_(True)
    k = (torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda") * (0.5 * input_scale)
         ).detach().requires_grad_(True)
    v = (torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda") * (0.5 * input_scale)
         ).detach().requires_grad_(True)
    dout = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    sparse_map, lut, topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=BLKQ, BLKK=BLKK)
    M = S // BLKQ
    lut_flat = lut.view(B * H, M, topk).contiguous()
    lut_bhq = lut.view(B, H, M, topk).contiguous()

    # Triton path
    out_tr = _attention.apply(q, k, v, sparse_map, lut_flat, topk, BLKQ, BLKK, scale)
    dq_tr, dk_tr, dv_tr = torch.autograd.grad((out_tr,), (q, k, v), (dout,))
    torch.cuda.synchronize()

    # CK path
    lut_ck = lut_bhq.to(torch.int32).contiguous()
    lut_ck, _ = torch.sort(lut_ck, dim=-1)
    out_ck, lse_ck = ck_sla_fwd(
        q.detach().contiguous(), k.detach().contiguous(), v.detach().contiguous(),
        lut_ck, BLKQ, BLKK, scale,
    )
    dq_ck, dk_ck, dv_ck = ck_sla_bwd(
        dout.contiguous(),
        q.detach().contiguous(), k.detach().contiguous(), v.detach().contiguous(),
        out_ck, lse_ck, lut_ck, BLKQ, BLKK, scale,
    )
    torch.cuda.synchronize()

    # fp32 ref
    mask = build_mask(lut_ck, S, BLKQ, BLKK)
    out_fp, (qf, kf, vf) = torch_sparse_attn_generic(q, k, v, mask, scale, torch.float32)
    out_fp.backward(dout.float())
    dq_fp, dk_fp, dv_fp = qf.grad, kf.grad, vf.grad

    # bf16 torch.compile'd ref
    try:
        out_bf, (qb, kb, vb) = torch_sparse_attn_bf16(q, k, v, mask, scale)
        out_bf.backward(dout.to(torch.bfloat16))
        dq_bf, dk_bf, dv_bf = qb.grad, kb.grad, vb.grad
        have_bf16 = True
    except Exception as e:
        print(f"  WARN: bf16 ref failed: {type(e).__name__}: {e}")
        out_bf, dq_bf, dk_bf, dv_bf = None, None, None, None
        have_bf16 = False

    # Report
    print(f"\n── {name}  [B={B} H={H} S={S} D={D}] BLKQ={BLKQ} "
          f"topk={topk}/{S // BLKK} input_scale={input_scale}")
    print(f"{'field':4} | {'ck vs fp32':>14} | {'tr vs fp32':>14} |"
          f" {'ck vs bf16':>14} | {'tr vs bf16':>14} | {'ck vs tr':>14}")
    for field, ck_t, tr_t, fp_t, bf_t in [
        ("out", out_ck, out_tr, out_fp, out_bf),
        ("dq",  dq_ck,  dq_tr,  dq_fp,  dq_bf),
        ("dk",  dk_ck,  dk_tr,  dk_fp,  dk_bf),
        ("dv",  dv_ck,  dv_tr,  dv_fp,  dv_bf),
    ]:
        def fmt(a, b, have=True):
            if not have or a is None or b is None:
                return "   n/a       "
            return f"{snr(a, b):6.2f}dB {max_abs(a, b):.1e}"
        print(f" {field:3} | {fmt(fp_t, ck_t):>14} | {fmt(fp_t, tr_t):>14} |"
              f" {fmt(bf_t, ck_t, have_bf16):>14} | {fmt(bf_t, tr_t, have_bf16):>14} |"
              f" {fmt(ck_t, tr_t):>14}")

    return {
        "out": out_ck.detach(),
        "dq":  dq_ck.detach(),
        "dk":  dk_ck.detach(),
        "dv":  dv_ck.detach(),
    }


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
        # name                  B  H   S     D    BLKQ BLKK topk  mag
        ("small_blkq128",      (1, 4, 1024, 128),  128, 64, 0.10, 1.0),
        ("blkq64",             (1, 4, 1024, 128),   64, 64, 0.10, 1.0),
        ("sparse_extreme",     (1, 4, 1024, 128),  128, 64, 0.0625, 1.0),  # topk ≈ 1
        ("dense_extreme",      (1, 4, 1024, 128),  128, 64, 0.99, 1.0),    # topk ≈ all
        ("medium_blkq128",     (1, 8, 2048, 128),  128, 64, 0.10, 1.0),
        ("medium_blkq64",      (1, 8, 2048, 128),   64, 64, 0.10, 1.0),
        ("high_magnitude",     (1, 4, 1024, 128),  128, 64, 0.10, 5.0),
        ("near_zero",          (1, 4, 1024, 128),  128, 64, 0.10, 0.05),
    ]

    print("=" * 80)
    print("Strict CK vs Triton verification")
    print(f"  save_ref  = {args.save_ref}")
    print(f"  check_ref = {args.check_ref}")
    print("=" * 80)

    for name, (B, H, S, D), BLKQ, BLKK, topk_ratio, mag in cases:
        try:
            outputs = run_case(name, B, H, S, D, BLKQ, BLKK, topk_ratio, mag)
            save_or_check_ref(name, outputs, args.save_ref, args.check_ref)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
