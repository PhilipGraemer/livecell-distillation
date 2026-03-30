"""Cache teacher logits to HDF5 for offline distillation.

Offline distillation is more efficient than online because teacher
inference happens once (not every epoch), it enables easy sweeps over
temperature and alpha without re-running the teacher, and it reduces
GPU memory since you don't need both models loaded during training.

This module supports two modes:

1. **Single teacher** — load one checkpoint, cache its logits.
2. **Council** — load N teacher checkpoints, run all of them, and
   average the logits in logit space before saving.

**Why average in logit space?**  Averaging softmax probabilities
discards information about the teacher's confidence magnitude.
Averaging logits preserves the relative scale and lets the temperature
parameter in the distillation loss control how soft the targets become.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from livecell_distillation.config import ARCH_REGISTRY, NUM_CLASSES


def load_teacher(
    checkpoint_path: str | Path,
    arch_name: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a single teacher model from a checkpoint file.

    Handles three common checkpoint formats: raw state_dict, or dicts
    with a ``model_state_dict``, ``state_dict``, or ``model`` key.
    """
    model_name = ARCH_REGISTRY[arch_name]
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)

    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if key in state:
            state = state[key]
            break

    model.load_state_dict(state)
    model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def cache_logits(
    teachers: List[torch.nn.Module],
    loader: DataLoader,
    n_samples: int,
    device: torch.device,
    split_name: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run teacher(s) on a data split and return (averaged) logits.

    For a single teacher this is a straightforward forward pass.
    For a council, logits are accumulated across all teachers and
    then divided by the teacher count.

    Returns
    -------
    logits : np.ndarray, shape (N, C)
    predictions : np.ndarray, shape (N,)
    labels : np.ndarray, shape (N,)
    accuracy : float
        Accuracy in percent.
    """
    n_teachers = len(teachers)
    all_logits = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    all_labels = np.zeros(n_samples, dtype=np.int64)

    print(f"\n  Caching {split_name} logits ({n_samples:,} samples, {n_teachers} teacher(s))...")

    for t_idx, teacher in enumerate(teachers):
        desc = f"    Teacher {t_idx + 1}/{n_teachers}" if n_teachers > 1 else f"    {split_name}"
        with torch.no_grad():
            for images, labels, indices in tqdm(loader, desc=desc):
                images = images.to(device)
                with torch.amp.autocast("cuda"):
                    logits = teacher(images)
                logits_np = logits.cpu().numpy()
                for i, idx in enumerate(indices):
                    idx = idx.item()
                    all_logits[idx] += logits_np[i]
                    if t_idx == 0:
                        all_labels[idx] = labels[i].item()

    all_logits /= n_teachers
    preds = np.argmax(all_logits, axis=1)
    accuracy = (preds == all_labels).mean() * 100

    print(f"  {split_name} accuracy: {accuracy:.2f}%")
    return all_logits, preds, all_labels, accuracy


def save_logits_h5(
    path: str | Path,
    logits: np.ndarray,
    labels: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
    attrs: Dict | None = None,
    compress: bool = True,
) -> None:
    """Write logits (and optional labels/predictions) to an HDF5 file."""
    kw = {"compression": "gzip"} if compress else {}
    with h5py.File(str(path), "w") as hf:
        hf.create_dataset("logits", data=logits, dtype="float32", **kw)
        if labels is not None:
            hf.create_dataset("labels", data=labels)
        if predictions is not None:
            hf.create_dataset("predictions", data=predictions)
        for k, v in (attrs or {}).items():
            hf.attrs[k] = v
