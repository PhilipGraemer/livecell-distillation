"""Evaluation metrics for classification and calibration.

All metrics follow the convention of returning percentages (0–100) to
match the reporting format used in the paper.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Measures how well a model's confidence aligns with its accuracy.
    A perfectly calibrated model has ECE = 0.

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities, shape ``(N, C)``.
    labels : np.ndarray
        Ground-truth labels, shape ``(N,)``.
    n_bins : int
        Number of confidence bins.

    Returns
    -------
    float
        ECE as a fraction in [0, 1] (multiply by 100 for percentage).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += np.abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * prop_in_bin

    return ece


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    teacher_preds: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Comprehensive evaluation on a test set.

    Returns a dict with accuracy, macro F1, balanced accuracy, ECE,
    teacher agreement, per-class recall, inference timing, and raw
    predictions/probabilities for downstream plotting.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    start = time.time()

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    elapsed = time.time() - start

    preds = np.array(all_preds)
    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)

    accuracy = (preds == labels_np).mean() * 100
    macro_f1 = f1_score(labels_np, preds, average="macro") * 100
    balanced_acc = balanced_accuracy_score(labels_np, preds) * 100
    per_class_recall = recall_score(labels_np, preds, average=None) * 100
    ece = compute_ece(probs_np, labels_np) * 100
    agreement = (
        (preds == teacher_preds).mean() * 100
        if teacher_preds is not None
        else None
    )
    ms_per_image = (elapsed / len(labels_np)) * 1000

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_recall": per_class_recall,
        "ece": ece,
        "agreement_with_teacher": agreement,
        "inference_time_ms": ms_per_image,
        "predictions": preds,
        "labels": labels_np,
        "probabilities": probs_np,
    }
