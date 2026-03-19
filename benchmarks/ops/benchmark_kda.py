

import torch
import triton
from flash_attn import flash_attn_func
from torch.nn import functional as F

from fla.ops.comba import chunk_comba
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
from fla.ops.kda import chunk_kda


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[4096, 32768, 65536, 131072, 262144, 524288, 1048576],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['kda'],
        # label name for the lines
        line_names=['kda'],
        # line styles
        styles=[('blue', '-'), ('red', '-.')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    ),
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    B, H, D = 1, 16, 128

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    do = torch.randn(B, T, H, D, dtype=dtype, device=device)
    if  provider == 'attn':
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        results = triton.testing.do_bench(
            lambda: flash_attn_func(
                q=q,
                k=k,
                v=v,
            ).backward(do),
            quantiles=quantiles,
        )
    elif provider == 'kda':
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
        results = triton.testing.do_bench(
            lambda: chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                safe_gate=True,  # Plese Carefully read doc strings
            )[0].backward(do),
            quantiles=quantiles,
        )
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
