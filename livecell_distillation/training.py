"""Training loop for offline knowledge distillation.

This module contains the epoch-level functions and the high-level
``train_distillation`` orchestrator that is called by the CLI.

**Design choice: model selection by accuracy, not distillation loss.**
The distillation loss includes a KL term that depends on the teacher's
soft targets, not just the ground-truth labels.  Selecting checkpoints
by distillation loss would effectively select for agreement with the
teacher rather than for the student's own accuracy.  We therefore
validate with standard cross-entropy and select by accuracy.
"""

from __future__ import annotations

import copy
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from livecell_distillation.config import CELL_TYPES, NUM_CLASSES, DistillConfig
from livecell_distillation.data import (
    DistillationDataset,
    HDF5CellDataset,
    Sample,
    make_eval_transform,
    make_train_transform,
)
from livecell_distillation.loss import DistillationLoss
from livecell_distillation.metrics import evaluate_model
from livecell_distillation.plotting import plot_confusion_matrix, plot_training_curves


# ──────────────────────────────────────────────────────────────────────
# Epoch-level functions
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> Tuple[float, float, float]:
    """One training epoch.  Returns ``(total_loss, hard_loss, soft_loss)``."""
    model.train()
    running = np.zeros(3)  # total, hard, soft

    for images, labels, teacher_logits in loader:
        images = images.to(device)
        labels = labels.to(device)
        teacher_logits = teacher_logits.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            student_logits = model(images)
            loss, hard, soft = criterion(student_logits, labels, teacher_logits)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running += np.array([loss.item() * bs, hard * bs, soft * bs])

    n = len(loader.dataset)
    return tuple(running / n)  # type: ignore[return-value]


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate with standard CE.  Returns ``(val_loss, val_accuracy_pct)``."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = ce(logits, labels)

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, (correct / total) * 100


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def train_distillation(
    *,
    cfg: DistillConfig,
    teacher_name: str,
    student_arch: str,
    train_samples_full: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    train_logits_full: np.ndarray,
    teacher_test_preds: Optional[np.ndarray],
    device: torch.device,
    output_dir: str | Path,
) -> Dict[str, object]:
    """Run a single distillation experiment.

    Returns a results dict suitable for CSV export.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Subsample training data if needed ─────────────────────────
    if cfg.data_fraction < 1.0:
        n_subset = int(len(train_samples_full) * cfg.data_fraction)
        labels_full = [s[1] for s in train_samples_full]
        subset_idx, _ = train_test_split(
            range(len(train_samples_full)),
            train_size=n_subset,
            random_state=cfg.seed,
            stratify=labels_full,
        )
        train_samples = [train_samples_full[i] for i in subset_idx]
        train_logits = train_logits_full[subset_idx]
    else:
        train_samples = train_samples_full
        train_logits = train_logits_full

    print(f"Training: {len(train_samples):,}  Val: {len(val_samples):,}  Test: {len(test_samples):,}")

    # ── Datasets and loaders ──────────────────────────────────────
    train_tf = make_train_transform(cfg.img_size, cfg.norm_mean, cfg.norm_std)
    eval_tf = make_eval_transform(cfg.img_size, cfg.norm_mean, cfg.norm_std)

    train_ds = DistillationDataset(HDF5CellDataset(train_samples, train_tf), train_logits)
    val_ds = DistillationDataset(
        HDF5CellDataset(val_samples, eval_tf),
        np.zeros((len(val_samples), NUM_CLASSES), dtype=np.float32),
    )
    test_ds = DistillationDataset(
        HDF5CellDataset(test_samples, eval_tf),
        np.zeros((len(test_samples), NUM_CLASSES), dtype=np.float32),
    )

    loader_kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    # ── Student model ─────────────────────────────────────────────
    model = timm.create_model(student_arch, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)
    print(f"Student: {student_arch}  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = DistillationLoss(alpha=cfg.alpha, temperature=cfg.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3,
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────
    best_val_acc = 0.0
    best_model_wts = None
    best_epoch = 0
    time_to_best = 0.0
    trigger_times = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []

    t0 = time.time()

    for epoch in range(cfg.epochs):
        t_ep = time.time()
        train_loss, hard_loss, soft_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
            f"Loss: {train_loss:.4f} (H:{hard_loss:.4f} S:{soft_loss:.4f}) | "
            f"Val: {val_loss:.4f} / {val_acc:.2f}% | "
            f"{time.time() - t_ep:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            time_to_best = (time.time() - t0) / 60
            trigger_times = 0
            print(f"  -> New best ({val_acc:.2f}%)")
        else:
            trigger_times += 1
            if trigger_times >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(val_loss)

    total_time = (time.time() - t0) / 60

    # ── Evaluate ──────────────────────────────────────────────────
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, output_dir / "best_model.pth")

    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, device, teacher_test_preds)

    # ── Plots ─────────────────────────────────────────────────────
    plot_training_curves(train_losses, val_losses, val_accs, output_dir / "training_curves.png")
    plot_confusion_matrix(metrics["labels"], metrics["predictions"], CELL_TYPES, output_dir / "confusion_matrix.png")

    # ── Results dict ──────────────────────────────────────────────
    results = {
        "teacher": teacher_name,
        "student": student_arch,
        "method": "distillation",
        "temperature": cfg.temperature,
        "data_fraction": cfg.data_fraction,
        "alpha": cfg.alpha,
        "best_epoch": best_epoch,
        "training_time_to_best_min": round(time_to_best, 1),
        "total_training_time_min": round(total_time, 1),
        "test_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "ece": metrics["ece"],
        "agreement_with_teacher": metrics["agreement_with_teacher"],
        "inference_time_ms": metrics["inference_time_ms"],
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "per_class_recall": metrics["per_class_recall"].tolist(),
    }

    print(f"\nTest Accuracy: {metrics['accuracy']:.2f}%  |  Macro-F1: {metrics['macro_f1']:.2f}%")

    return results


# ──────────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────────

def save_results_csv(results: List[Dict], path: str | Path) -> None:
    """Append-safe CSV export of experiment results."""
    if not results:
        return

    fieldnames = [
        "teacher", "student", "method", "temperature", "data_fraction", "alpha",
        "best_epoch", "training_time_to_best_min", "total_training_time_min",
        "test_accuracy", "macro_f1", "balanced_accuracy",
        "ece", "agreement_with_teacher", "inference_time_ms",
        "total_parameters",
    ]
    for ct in CELL_TYPES:
        fieldnames.append(f"recall_{ct}")

    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k != "per_class_recall"}
            for i, ct in enumerate(CELL_TYPES):
                row[f"recall_{ct}"] = r["per_class_recall"][i]
            writer.writerow(row)
