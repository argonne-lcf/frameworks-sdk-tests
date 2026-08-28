"""latentmoe: a standalone MoE example capturing DeepSeek-V4-Pro and Kimi-K3.

PyTorch + Triton only. Runs on NVIDIA (incl. Grace-Hopper), Intel XPU, and
(reference paths) CPU / Apple Silicon.
"""

from .accel import best_device, autocast_dtype
from .config import ModelConfig, deepseek_v4_mini, kimi_k3_mini, hybrid_mini, get_preset
from .model import LatentMoEModel

__all__ = [
    "best_device",
    "autocast_dtype",
    "ModelConfig",
    "deepseek_v4_mini",
    "kimi_k3_mini",
    "hybrid_mini",
    "get_preset",
    "LatentMoEModel",
]

__version__ = "0.1.0"
