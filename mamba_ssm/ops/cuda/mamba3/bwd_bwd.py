"""Hand-written CUDA replacement for the TileLang ``mamba_mimo_bwd_bwd`` kernel (see bwd_bwd.cu).

Opt-in with ``M3_CUDA_BWD=1``. It covers one configuration -- R=4, N=128, P=64, chunk 16, a single Q/K group,
reduceO with the Z gate and D skip, 32 rotary angles, bf16 activations and states, varlen, sm_90 -- and
returns None for anything else (or if the JIT build fails), so the caller keeps the TileLang kernel.
GH200, H=12 per TP rank, S=65536, two-level pass C: 6.2 ms vs 7.3 ms (TileLang).
"""

import os
import warnings
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    build = os.environ.get("M3_CUDA_BUILD_DIR", os.path.join(os.path.expanduser("~"), ".cache", "mamba3_cuda"))
    os.makedirs(build, exist_ok=True)
    return load(
        name="mamba3_cuda_bwd_bwd",
        sources=[os.path.join(here, "bwd_bwd.cu")],
        extra_include_paths=[here],
        extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=build,
        verbose=False,
    )


def cuda_bwd_bwd(B, H, G, N, P, R, hasZ, hasD, reduceO, packed_dout, isVarlen, chunk_size, rotary_dim_divisor,
                 dtype, states_dtype, has_init_state=False, state_only=False, blocked=False, **_):
    """A callable with the TileLang kernel's argument list, or None when this kernel does not apply."""
    if os.environ.get("M3_CUDA_BWD", "0") != "1":
        return None
    supported = (
        B == 1 and G == 1 and N == 128 and P == 64 and R == 4 and chunk_size == 16 and rotary_dim_divisor == 4
        and hasZ and hasD and reduceO and not packed_dout and isVarlen
        and str(dtype).replace("torch.", "") == "bfloat16" and states_dtype == torch.bfloat16
        and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
    )
    if not supported:
        return None
    try:
        ext = _ext()
    except Exception as e:  # noqa: BLE001 - build failure falls back to TileLang
        warnings.warn(f"mamba3 CUDA bwd_bwd unavailable ({e}); using the TileLang kernel")
        return None
    flags = (bool(blocked), bool(has_init_state), bool(state_only))
    return lambda *args: ext.bwd_bwd(*args, *flags)
