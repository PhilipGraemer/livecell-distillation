"""Shared configuration for the distillation pipeline.

All hardcoded values from the original scripts are collected here.  Override
via CLI flags or by editing a YAML config — the constants below serve as
defaults and documentation of the values used in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

CELL_TYPES: List[str] = [
    "A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"
]
NUM_CLASSES: int = len(CELL_TYPES)

# ──────────────────────────────────────────────────────────────────────
# Architecture registry
# ──────────────────────────────────────────────────────────────────────
# Maps short names (used in CLI / configs) to timm model strings.

ARCH_REGISTRY: Dict[str, str] = {
    "eva02": "eva02_base_patch14_224.mim_in22k",
    "efficientnet_b5": "tf_efficientnet_b5.ns_jft_in1k",
    "efficientnet_b0": "tf_efficientnet_b0.ns_jft_in1k",
    "swin": "swin_base_patch4_window7_224.ms_in22k",
    "vit_base": "vit_base_patch16_224.augreg_in21k",
}


# ──────────────────────────────────────────────────────────────────────
# Default hyperparameters
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DistillConfig:
    """Hyperparameters for a distillation run.

    These match the settings used in the paper.  Every field is
    overridable via CLI.
    """

    # Image / data
    img_size: int = 224
    batch_size: int = 128
    num_workers: int = 8
    seed: int = 42

    # Training
    learning_rate: float = 1e-4
    epochs: int = 100
    patience: int = 5

    # Distillation
    alpha: float = 0.2          # weight on hard (CE) loss; 1-alpha on soft (KL)
    temperature: float = 4.0
    data_fraction: float = 1.0  # fraction of training data to use

    # Normalisation (must match teacher training)
    norm_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    norm_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    # Data split ratios (used only when building sample lists from scratch)
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
