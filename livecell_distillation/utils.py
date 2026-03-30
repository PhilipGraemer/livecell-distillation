"""Small utility functions shared across the pipeline."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Covers Python, NumPy, and PyTorch (CPU + all CUDA devices).
    Also forces deterministic cuDNN behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
