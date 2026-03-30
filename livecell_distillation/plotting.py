"""Plotting utilities for training diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: str | Path,
) -> None:
    """Save a two-panel figure: loss curves and validation accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-", label="Train")
    ax1.plot(epochs, val_losses, "r-", label="Val")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accs, "g-", label="Val Accuracy")
    ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    save_path: str | Path,
) -> None:
    """Save a row-normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=100,
    )
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix (%)")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
