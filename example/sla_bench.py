import contextlib
from statistics import mean

import torch
torch.random.manual_seed(0)

ENABLE_PROFILING = False
ENABLE_SNR_CHECK = True
RUN_FLASH_V2 = True
RUN_SLA = True

# for NVIDIA
RUN_FLASH_V3 = False

# for AMD
RUN_PRIMUS_TURBO = True

def gen_rand_tensor(B : int, S : int, H : int, D : int):
    rng = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = torch.randn(
        (3, B, S, H, D),
        generator = rng,
        device="cpu",
        dtype=torch.bfloat16,
        requires_grad=False).to("cuda")
    dout = torch.randn(
        (B, S, H, D),
        generator = rng,
        device="cpu",
        dtype=torch.bfloat16,
        requires_grad=False).to("cuda")

    return (q, k, v, dout)

def print_partial_tensor(t : torch.Tensor, message : str):
    print("-" * 40)
    print(f"partial_tensor - {message}")
    print("shape={}, dtype={}, device={}".format(t.shape, t.dtype, t.device))

    N = 10
    x = t.flatten()
    L = x.numel()
    head = x[:N]
    mid = x[L//2:L//2+N]
    tail = x[-N:]
    print("head : {}".format(head))
    print("mid : {}".format(mid))
    print("tail : {}".format(tail))

    print("-" * 40, flush=True)

def get_time_elapsed(job_name, func, *args, **kwargs):
    NUM_WARMUPS = 0
    NUM_ITERS = 3

    def maybe_profile_context():
        if not ENABLE_PROFILING:
            return contextlib.nullcontext()

        assert NUM_ITERS >= 8
        return torch.profiler.profile(
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                schedule=torch.profiler.schedule(
                    wait = 3,
                    warmup = NUM_ITERS-3-3,
                    active = 2,
                    repeat = 0),
                on_trace_ready = torch.profiler.tensorboard_trace_handler(f"./profile_results_{job_name}"),
                record_shapes = True,
                profile_memory = False,
                with_stack = True)

    start_events = [torch.cuda.Event(enable_timing=True) for i in range(NUM_ITERS)]
    end_events = [torch.cuda.Event(enable_timing=True) for i in range(NUM_ITERS)]

    for i in range(NUM_WARMUPS):
        o = func(*args, **kwargs)

    with maybe_profile_context() as profiler:
        for i in range(NUM_ITERS):
            start_events[i].record()
            o = func(*args, **kwargs)
            end_events[i].record()
            if ENABLE_PROFILING:
                profiler.step()
        end_events[-1].synchronize()

    elapsed_times = [start_events[i].elapsed_time(end_events[i]) for i in range(NUM_ITERS)]
    elapsed_times.sort()
    if len(elapsed_times) >= 5:
        final_elapsed_times = elapsed_times[2:-2]
    else:
        final_elapsed_times = elapsed_times[1:-1] if len(elapsed_times) >= 3 else elapsed_times
    return (mean(final_elapsed_times), o)

def compute_snr(x: torch.Tensor, y: torch.Tensor, is_y_bshd : bool = True):
    if not is_y_bshd:
        # bhsd
        y = y.permute(0, 2, 1, 3)

    assert x.shape == y.shape
    x, y = x.float(), y.float()
    signal_power = torch.norm(x).pow(2)
    noise_power = torch.norm(x - y).pow(2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-12)).detach().item()

def run_test_case(
    B : int, S : int, H : int, D : int):

    def run_backward(out, dout):
        out.backward(dout, retain_graph=True)

    TOPK = 0.10
    BLKQ = 128
    BLKK = 64

    print("[B, S, H, D]=[{}, {}, {}, {}] topk={:.2f} BLKQ={}, BLKK={}".format(
        B, S, H, D, TOPK, BLKQ, BLKK))
    q, k, v, dout = gen_rand_tensor(B, S, H, D)
    print_partial_tensor(q, "q")
    print_partial_tensor(k, "k")
    print_partial_tensor(v, "v")
    print_partial_tensor(dout, "dout")

    if RUN_FLASH_V2:
        import flash_attn

        def run_flash_attn_v2_fwd(q, k, v):
            return flash_attn.flash_attn_func(q, k, v, causal=False, deterministic=False)

        q_fav2 = q.clone().detach().requires_grad_()
        k_fav2 = k.clone().detach().requires_grad_()
        v_fav2 = v.clone().detach().requires_grad_()
        dout_fav2 = dout.clone().detach()

        fav2_fwd_time, out_fav2 = get_time_elapsed("fav2_fwd", run_flash_attn_v2_fwd, q_fav2, k_fav2, v_fav2)
        fav2_bwd_time, _ = get_time_elapsed("fav2_bwd", run_backward, out_fav2, dout_fav2)
        print("fav2_fwd_time={:.3f} ms, fav2_bwd_time={:.3f} ms".format(fav2_fwd_time, fav2_bwd_time))

    if RUN_SLA:
        import sparse_linear_attention

        def run_sla_attn_fwd(sla_op, q, k, v):
            return sla_op(q, k, v, return_sparsity=False)

        sla_op = sparse_linear_attention.SparseLinearAttention(
            head_dim=D,
            topk=TOPK,
            feature_map="softmax",
            BLKQ=BLKQ,
            BLKK=BLKK).cuda()

        q_sla = q.transpose(1, 2).clone().detach().requires_grad_()
        k_sla = k.transpose(1, 2).clone().detach().requires_grad_()
        v_sla = v.transpose(1, 2).clone().detach().requires_grad_()
        dout_sla = dout.transpose(1, 2).clone().detach()

        sla_fwd_time, out_sla = get_time_elapsed("sla_fwd", run_sla_attn_fwd, sla_op, q_sla, k_sla, v_sla)
        sla_bwd_time, _ = get_time_elapsed("sla_bwd", run_backward, out_sla, dout_sla)
        print("sla_fwd_time={:.3f} ms, sla_bwd_time={:.3f} ms".format(sla_fwd_time, sla_bwd_time))

        print_partial_tensor(out_sla, "out_sla")
        print_partial_tensor(q_sla.grad, "q_sla.grad")
        print_partial_tensor(k_sla.grad, "k_sla.grad")
        print_partial_tensor(v_sla.grad, "v_sla.grad")

    if RUN_FLASH_V3:
        import flash_attn_interface

        def run_flash_attn_v3_fwd(q, k, v):
            return flash_attn_interface.flash_attn_func(q, k, v, causal=False, deterministic=False)

        q_fav3 = q.clone().detach().requires_grad_()
        k_fav3 = k.clone().detach().requires_grad_()
        v_fav3 = v.clone().detach().requires_grad_()
        dout_fav3 = dout.clone().detach()

        fav3_fwd_time, out_fav3 = get_time_elapsed("fav3_fwd", run_flash_attn_v3_fwd, q_fav3, k_fav3, v_fav3)
        fav3_bwd_time, _ = get_time_elapsed("fav3_bwd", run_backward, out_fav3, dout_fav3)
        print("fav3_fwd_time={:.3f} ms, fav3_bwd_time={:.3f} ms".format(fav3_fwd_time, fav3_bwd_time))

        if ENABLE_SNR_CHECK and RUN_FLASH_V2:
            assert compute_snr(out_fav2, out_fav3) > 40.0
            assert compute_snr(q_fav2.grad, q_fav3.grad) > 30.0
            assert compute_snr(k_fav2.grad, k_fav3.grad) > 30.0
            assert compute_snr(v_fav2.grad, v_fav3.grad) > 30.0

    if RUN_PRIMUS_TURBO:
        import primus_turbo.pytorch as pt

        q_turbo = q.clone().detach().requires_grad_()
        k_turbo = k.clone().detach().requires_grad_()
        v_turbo = v.clone().detach().requires_grad_()
        dout_turbo = dout.clone().detach()
        
        def run_primus_turbo_attn_fwd(q, k, v):
            return pt.ops.flash_attn_func(q, k, v, causal=False, deterministic=False)

        turbo_fwd_time, out_turbo = get_time_elapsed("turbo_fwd", run_primus_turbo_attn_fwd, q_turbo, k_turbo, v_turbo)
        turbo_bwd_time, _ = get_time_elapsed("turbo_bwd", run_backward, out_turbo, dout_turbo)
        print("turbo_fwd_time={:.3f} ms, turbo_bwd_time={:.3f} ms".format(turbo_fwd_time, turbo_bwd_time))

        if ENABLE_SNR_CHECK and RUN_FLASH_V2:
            assert compute_snr(out_fav2, out_turbo) > 40.0
            assert compute_snr(q_fav2.grad, q_turbo.grad) > 30.0
            assert compute_snr(k_fav2.grad, k_turbo.grad) > 30.0
            assert compute_snr(v_fav2.grad, v_turbo.grad) > 30.0


def main():
    attn_cases = [
       [1, 16384, 12, 128],
       #[4, 16384, 12, 128],
       #[1, 8192, 24, 128],
       #[4, 8192, 24, 128],
       #[1, 16384, 24, 64],
       #[4, 16384, 24, 64],
       #[1, 16384, 24, 128],
       #[4, 16384, 24, 128],
       #[1, 65536, 24, 128],
       #[4, 65536, 24, 128],
       #[1, 65536, 40, 128],
       #[1, 75600, 40, 128],
       #[4, 75600, 40, 128],
    ]
    for (B, S, H, D) in attn_cases:
        run_test_case(B, S, H, D)

if __name__ == "__main__":
    main()
