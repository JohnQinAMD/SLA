
import os
from enum import Enum
from typing import Callable

import triton
import torch

SLA_ENABLE_TRITON_AUTOTUNING = os.getenv("SLA_ENABLE_TRITON_AUTOTUNING", "0") == "1"
GPU_ARCH = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]

_tuned_kernel_caches = {}

class Phase(Enum):
    FORWARD = 1
    BACKWARD_DQ = 2
    BACKWARD_DKDV = 3

def get_heuristic_triton_config(L : int, D : int, phase : Phase):
    assert isinstance(phase, Phase)
    # TODO (limou)
    # optimize heuristic choosing
    match phase:
        case Phase.FORWARD:
            if D <= 64:
                return {
                    "num_warps" : 2,
                    "num_stages" : 1,
                    "matrix_instr_nonkdim" : 32,
                    "waves_per_eu" : 2,
                }
            else:
                return {
                    "num_warps" : 4,
                    "num_stages" : 1,
                    "matrix_instr_nonkdim" : 32,
                    "waves_per_eu" : 2,
                }
        case Phase.BACKWARD_DQ:
            return {
                "num_warps" : 4,
                "num_stages" : 2,
                "matrix_instr_nonkdim" : 16,
                "waves_per_eu" : 2,
            }
        case Phase.BACKWARD_DKDV:
            if D <= 64:
                return {
                    "num_warps" : 2,
                    "num_stages" : 1,
                    "matrix_instr_nonkdim" : 16,
                    "waves_per_eu" : 2,
                }
            else:
                return {
                    "num_warps" : 4,
                    "num_stages" : 1,
                    "matrix_instr_nonkdim" : 16,
                    "waves_per_eu" : 2,
                }

class HeuristicKernel(triton.KernelInterface):

    def __init__(self, kernel, meta):
        self.kernel = kernel
        self.meta = meta

    def run(self, *args, grid, warmup, **kwargs):
        launch_kwargs = {**self.meta, **kwargs}
        return self.kernel[grid](
            *args,
            **launch_kwargs,
        )

def get_triton_autotune_decorator(func : Callable, L : int, D : int, phase: Phase):
    """
    Args:
        L : sequence length
        D : head dimension
    """
            
    if not SLA_ENABLE_TRITON_AUTOTUNING:
        triton_kwargs = get_heuristic_triton_config(L, D, phase)
        return HeuristicKernel(func, triton_kwargs)
    
    tuned_key = (func, )
    if tuned_key in _tuned_kernel_caches:
        return _tuned_kernel_caches[tuned_key]

    def get_autotune_configs():
        # num_stages=0 fails to compile on gfx950 (PassManager::run failed);
        # num_warps=8 + matrix_instr_nonkdim=16 is what the curated sweep picks
        # as optimal for all three SLA kernels on gfx950.
        configs = []
        for num_warps in [2, 4, 8]:
            for num_stages in ([1, 2] if phase == Phase.FORWARD else [1, 2, 3]):
                for waves_per_eu in [2]:
                    for matrix_instr_nonkdim in [16, 32]:
                        for kpack in ([1] if GPU_ARCH == "gfx950" else [1, 2]):
                            configs.append(triton.Config(
                                {"waves_per_eu" : waves_per_eu, "matrix_instr_nonkdim" : matrix_instr_nonkdim, "kpack" : kpack},
                                num_warps=num_warps, num_stages=num_stages))
        return configs

    tuning_configs = get_autotune_configs()
    tuned_kernel = triton.autotune(
        configs = tuning_configs,
        key = ["L", "D", "BLOCK_M", "BLOCK_N"],
        warmup = 3,
        rep = 10,
        )(func)

    _tuned_kernel_caches[tuned_key] = tuned_kernel
    return tuned_kernel