
"""
perf record -F 99 -g -- python -m examples.conv_overhead
perf script > out.perf && /tmp/FlameGraph/stackcollapse-perf.pl out.perf > out.folded
/tmp/FlameGraph/flamegraph.pl out.folded > /mnt/shared/taylorrobie/public_html/kernel.svg
rm out.perf out.folded perf.data
"""
import time

import numpy as np
import torch
torch.set_num_threads(1)

from utils import Timer

#    C    M    H      W   kernel
params = [
    (1,   1,  128,   128, (3, 3)),
    (4,   8,  256,   256, (1, 1)),
    (8,   8,  256,   256, (3, 3)),
    (1,   4, 1024,  1024, (3, 3)),
    (16, 16,  128,   128, (7, 7)),
]

for C, M, h, w, kernel_size in params:

    conv2d_no_padding = torch.nn.Conv2d(C, M, kernel_size,
        padding=(0, 0),
        bias=True)

    conv2d_padding = torch.nn.Conv2d(C, M, kernel_size,
        padding=(1, 1),
        bias=True)

    def conv(x, weight, bias, model):
        return torch.convolution(
            x, weight, bias,
            model.stride, model.padding,
            model.dilation, model.transposed,
            model.output_padding, model.groups
        )


    print("\n" * 10)
    for use_mkl in [False, True]:
        torch._C._set_mkldnn_enabled(use_mkl)
        conv2d = conv2d_no_padding
        x = torch.ones((1, C, h, w))

        # weight = conv2d.weight.to_mkldnn() if use_mkl else conv2d.weight
        # bias = conv2d.bias.to_mkldnn() if use_mkl else conv2d.bias
        # x = x.to_mkldnn() if use_mkl else x
        weight = conv2d.weight
        bias = conv2d.bias

        if not use_mkl:
            torch._C._set_cost_enabled(True)
            for _ in range(500):
                conv(x, weight, bias, conv2d) if not use_mkl else None
                # print()
            torch._C._set_cost_enabled(False)

        m0 = Timer(
            "conv(x, weight, bias, conv2d)",
            globals={
                "conv": conv,
                "conv2d": conv2d,
                "weight": weight,
                "bias": bias,
                "x": x}
        ).blocked_autorange(min_run_time=1)
        print(m0)
    print()
    input("...")
