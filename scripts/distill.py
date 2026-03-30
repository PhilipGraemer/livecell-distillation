#!/usr/bin/env python3
"""Train a student model via offline knowledge distillation.

Replaces the three original scripts (distill_eva_to_enb0.py,
distill_enb5_to_enb0.py, distill_council_to_enb0.py) with a single
entry point.

Examples
--------
EVA-02 → EN-B0::

    python scripts/distill.py \
        --teacher-name "EVA-02" \
        --train-logits /path/to/teacher_logits_train.h5 \
        --test-logits /path/to/teacher_logits_test.h5 \
        --sample-lists /path/to/sample_lists.pkl \
        --student efficientnet_b0 \
        --output-dir /path/to/output

Sweep over temperatures and data fractions::

    python scripts/distill.py \
        --teacher-name "EN-B5" \
        --train-logits /path/to/enb5_teacher_logits_train.h5 \
        --test-logits /path/to/enb5_teacher_logits_test.h5 \
        --sample-lists /path/to/sample_lists.pkl \
        --student efficientnet_b0 \
        --temperatures 3 4 \
        --data-fractions 0.10 1.00 \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from livecell_distillation.config import DistillConfig
from livecell_distillation.data import load_sample_lists
from livecell_distillation.training import save_results_csv, train_distillation
from livecell_distillation.utils import set_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline knowledge distillation.")
    parser.add_argument("--teacher-name", required=True, help="Teacher name for logging.")
    parser.add_argument("--train-logits", required=True, help="HDF5 with teacher train logits.")
    parser.add_argument("--test-logits", required=True, help="HDF5 with teacher test logits + preds.")
    parser.add_argument("--sample-lists", required=True, help="Path to sample_lists.pkl.")
    parser.add_argument("--student", default="efficientnet_b0", help="timm model name for student.")
    parser.add_argument("--output-dir", required=True, help="Output root directory.")

    # Sweep parameters
    parser.add_argument("--temperatures", nargs="+", type=float, default=[3, 4])
    parser.add_argument("--data-fractions", nargs="+", type=float, default=[0.10, 1.00])
    parser.add_argument("--alpha", type=float, default=0.2)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)

    if not torch.cuda.is_available():
        print("[error] CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"DISTILLATION: {args.teacher_name} → {args.student}")
    print(f"Temperatures: {args.temperatures}  Data fractions: {args.data_fractions}")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────
    sample_data = load_sample_lists(args.sample_lists)
    train_samples = sample_data["train_samples"]
    val_samples = sample_data["val_samples"]
    test_samples = sample_data["test_samples"]

    with h5py.File(args.train_logits, "r") as hf:
        train_logits = hf["logits"][:].astype(np.float32)

    with h5py.File(args.test_logits, "r") as hf:
        teacher_test_preds = hf["predictions"][:]
        teacher_acc = float(hf.attrs.get("teacher_accuracy", 0))

    print(f"Train: {len(train_samples):,}  Val: {len(val_samples):,}  Test: {len(test_samples):,}")
    print(f"Teacher test accuracy: {teacher_acc:.2f}%")

    # ── Experiment grid ───────────────────────────────────────────
    all_results = []

    for frac in args.data_fractions:
        for temp in args.temperatures:
            run_name = f"T{temp}_data{int(frac * 100)}"
            run_dir = output_dir / run_name

            print(f"\n{'=' * 60}")
            print(f"{args.teacher_name} → {args.student}: T={temp}, Data={frac * 100:.0f}%")
            print(f"{'=' * 60}")

            cfg = DistillConfig(
                temperature=temp,
                data_fraction=frac,
                alpha=args.alpha,
                learning_rate=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )

            results = train_distillation(
                cfg=cfg,
                teacher_name=args.teacher_name,
                student_arch=args.student,
                train_samples_full=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                train_logits_full=train_logits,
                teacher_test_preds=teacher_test_preds,
                device=device,
                output_dir=run_dir,
            )

            all_results.append(results)
            save_results_csv(all_results, output_dir / "results_summary.csv")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\n{'Data%':>6} {'Temp':>5} {'Accuracy':>10} {'Macro-F1':>10} {'ECE':>8}")
    print("-" * 45)
    for r in all_results:
        print(
            f"{r['data_fraction'] * 100:>6.0f} {r['temperature']:>5.0f} "
            f"{r['test_accuracy']:>10.2f} {r['macro_f1']:>10.2f} {r['ece']:>8.2f}"
        )
    print(f"\nResults: {output_dir}")
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
