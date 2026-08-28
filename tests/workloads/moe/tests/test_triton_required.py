"""Hard preflight for the registered XPU/Triton MoE validation case."""

import torch

from latentmoe.kernels import HAS_TRITON


def test_xpu_triton_backend_is_available():
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), (
        "the registered MoE Triton case requires a working PyTorch XPU backend"
    )
    assert HAS_TRITON, (
        "Triton imported but latentmoe could not initialize its Triton kernels"
    )
