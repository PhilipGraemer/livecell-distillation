"""Dataset classes and data-loading utilities.

The HDF5 layout matches the format produced by the LIVECell classification
benchmark pipeline:

    images[i]   : vlen uint8  (flattened pixel data)
    shapes[i]   : (H, W, C)   (per-crop shape)

Each "sample" is a tuple ``(hdf5_path, label, index_in_file)``.
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from livecell_distillation.config import CELL_TYPES

# Type alias for a single sample identifier.
Sample = Tuple[str, int, int]  # (hdf5_path, label, index_in_file)


# ──────────────────────────────────────────────────────────────────────
# Core dataset
# ──────────────────────────────────────────────────────────────────────

class HDF5CellDataset(Dataset):
    """Reads single-cell crop images from per-class HDF5 files.

    Parameters
    ----------
    samples : list[Sample]
        Each entry is ``(hdf5_path, label, index_in_file)``.
    transform : callable or None
        Torchvision transform applied to each PIL image.
    """

    def __init__(self, samples: List[Sample], transform=None) -> None:
        self.samples = samples
        self.transform = transform
        self._h5_handles: Dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hdf5_path, label, index_in_file = self.samples[idx]

        if hdf5_path not in self._h5_handles:
            self._h5_handles[hdf5_path] = h5py.File(hdf5_path, "r")

        hf = self._h5_handles[hdf5_path]
        flat = hf["images"][index_in_file]
        shape = hf["shapes"][index_in_file]
        img = Image.fromarray(flat.reshape(shape))

        if self.transform:
            img = self.transform(img)

        return img, label, idx

    def __del__(self) -> None:
        for f in self._h5_handles.values():
            if f is not None:
                f.close()


class DistillationDataset(Dataset):
    """Wraps an :class:`HDF5CellDataset` and pairs each image with
    pre-computed teacher logits.

    This is what makes the distillation *offline*: teacher inference
    happens once (in the caching step), and the student trains against
    stored logits rather than running the teacher every epoch.

    Parameters
    ----------
    base_dataset : HDF5CellDataset
        The underlying image dataset.
    teacher_logits : np.ndarray
        Array of shape ``(N, num_classes)`` with teacher logits for each
        sample, indexed in the same order as *base_dataset*.
    """

    def __init__(self, base_dataset: HDF5CellDataset, teacher_logits: np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.teacher_logits = teacher_logits

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, label, _ = self.base_dataset[idx]
        logit = torch.tensor(self.teacher_logits[idx], dtype=torch.float32)
        return img, label, logit


# ──────────────────────────────────────────────────────────────────────
# Sample-list construction and I/O
# ──────────────────────────────────────────────────────────────────────

def build_sample_list(
    data_dir: str | Path,
    cell_types: List[str] = CELL_TYPES,
    subset_ratio: float = 1.0,
) -> List[Sample]:
    """Build a flat list of ``(hdf5_path, label, index)`` from per-class H5 files.

    Parameters
    ----------
    data_dir : path-like
        Directory containing ``<CellType>.h5`` files.
    cell_types : list[str]
        Cell-type names (order defines label indices).
    subset_ratio : float
        Fraction of each class to include (1.0 = all).
    """
    label_map = {ct: idx for idx, ct in enumerate(cell_types)}
    samples: List[Sample] = []

    for cell_type in cell_types:
        hdf5_path = os.path.join(str(data_dir), f"{cell_type}.h5")
        if not os.path.exists(hdf5_path):
            print(f"[warn] no HDF5 file for {cell_type} at {hdf5_path}")
            continue

        with h5py.File(hdf5_path, "r") as hf:
            n = len(hf["images"])

        label = label_map[cell_type]
        k = max(1, int(n * subset_ratio))
        indices = random.sample(range(n), k)

        for i in indices:
            samples.append((hdf5_path, label, i))

    return samples


def split_samples(
    samples: List[Sample],
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Stratified train / val / test split.

    Uses the same two-stage sklearn split as the original pipeline to
    ensure identical data partitions.
    """
    from sklearn.model_selection import train_test_split

    labels = [s[1] for s in samples]

    train, temp = train_test_split(
        samples,
        test_size=1 - train_split,
        random_state=seed,
        stratify=labels,
    )
    val, test = train_test_split(
        temp,
        test_size=test_split / (val_split + test_split),
        random_state=seed,
        stratify=[s[1] for s in temp],
    )
    return train, val, test


def save_sample_lists(
    train: List[Sample],
    val: List[Sample],
    test: List[Sample],
    output_path: str | Path,
    cell_types: List[str] = CELL_TYPES,
    seed: int = 42,
) -> None:
    """Persist sample lists as a pickle for downstream scripts."""
    payload = {
        "train_samples": train,
        "val_samples": val,
        "test_samples": test,
        "cell_types": cell_types,
        "seed": seed,
    }
    with open(str(output_path), "wb") as f:
        pickle.dump(payload, f)


def load_sample_lists(path: str | Path) -> Dict:
    """Load sample lists from a pickle saved by :func:`save_sample_lists`."""
    with open(str(path), "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────
# Standard transforms
# ──────────────────────────────────────────────────────────────────────

def make_train_transform(
    img_size: int = 224,
    mean: List[float] | None = None,
    std: List[float] | None = None,
) -> transforms.Compose:
    """Training transform: resize, light augmentation, normalise."""
    mean = mean or [0.5, 0.5, 0.5]
    std = std or [0.5, 0.5, 0.5]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def make_eval_transform(
    img_size: int = 224,
    mean: List[float] | None = None,
    std: List[float] | None = None,
) -> transforms.Compose:
    """Evaluation transform: resize and normalise only (no augmentation)."""
    mean = mean or [0.5, 0.5, 0.5]
    std = std or [0.5, 0.5, 0.5]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
