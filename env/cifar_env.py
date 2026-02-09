"""
Placeholder for PyTorch CIFAR-10 training environment (MODE='REAL_WORLD').
"""
from typing import Optional

from configs.config import Config


class CIFAR10Env:
    """Placeholder: CIFAR-10 SFL training environment."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config
        self._total_layers = self.config.TOTAL_LAYERS
        self._d_max = self.config.D_MAX
        self._client_types = self.config.CLIENT_TYPES

    def reset(self) -> dict:
        """Reset environment. Returns initial obs dict (placeholder)."""
        return {"step": 0}

    def step(self, client_type_idx: int, cut_point: int) -> dict:
        """
        Execute one step: train with given client type and cut point.
        Returns dict with loss, metrics, etc. (placeholder).
        """
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "done": False,
        }
