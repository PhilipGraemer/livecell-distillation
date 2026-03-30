#!/usr/bin/env python3
"""Cache teacher logits for offline distillation.

Replaces the three original scripts (cache_eva02_logits.py,
cache_enb5_teacher_logits.py, cache_council_logits.py) with a single
entry point.

Examples
--------
Single teacher (EVA-02)::

    python scripts/cache_logits.py \
        --teachers /path/to/eva02_seed42.pth \
        --teacher-archs eva02 \
        --name eva02_s42 \
        --sample-lists /path/to/sample_lists.pkl \
        --data-dir /path/to/LIVECell/Cells \
        --output-dir /path/to/output/distillation_eva02

Council (3× EVA-02)::

    python scripts/cache_logits.py \
        --teachers eva_s42.pth eva_s43.pth eva_s44.pth \
        --teacher-archs eva02 eva02 eva02 \
        --name eva3_council \
        --sample-lists /path/to/sample_lists.pkl \
        --output-dir /path/to/output/distillation_council

First run (no sample_lists.pkl yet — builds splits from scratch)::

    python scripts/cache_logits.py \
        --teachers /path/to/eva02.pth \
        --teacher-archs eva02 \
        --name eva02 \
        --data-dir /path/to/LIVECell/Cells \
        --output-dir /path/to/output/distillation \
        --build-splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from livecell_distillation.caching import cache_logits, load_teacher, save_logits_h5
from livecell_distillation.config import DistillConfig
from livecell_distillation.data import (
    HDF5CellDataset,
    build_sample_list,
    load_sample_lists,
    make_eval_transform,
    save_sample_lists,
    split_samples,
)
from livecell_distillation.utils import set_seeds

from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache teacher logits for distillation.")
    parser.add_argument("--teachers", nargs="+", required=True, help="Checkpoint paths.")
    parser.add_argument("--teacher-archs", nargs="+", required=True, help="Architecture per teacher.")
    parser.add_argument("--name", required=True, help="Name for this teacher/council.")
    parser.add_argument("--sample-lists", type=str, default=None, help="Path to sample_lists.pkl.")
    parser.add_argument("--data-dir", type=str, default=None, help="LIVECell/Cells directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--build-splits", action="store_true", help="Build splits from data-dir.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert len(args.teachers) == len(args.teacher_archs), \
        "Must provide one architecture name per teacher checkpoint."

    set_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[error] CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load or build sample lists ────────────────────────────────
    if args.build_splits:
        assert args.data_dir, "--data-dir required when --build-splits is set."
        print("Building sample lists from scratch...")
        all_samples = build_sample_list(args.data_dir)
        train, val, test = split_samples(all_samples, seed=args.seed)
        pkl_path = output_dir / "sample_lists.pkl"
        save_sample_lists(train, val, test, pkl_path, seed=args.seed)
        print(f"Saved sample lists to {pkl_path}")
    else:
        assert args.sample_lists, "--sample-lists required (or use --build-splits)."
        data = load_sample_lists(args.sample_lists)
        train = data["train_samples"]
        val = data.get("val_samples", [])
        test = data["test_samples"]

    print(f"Train: {len(train):,}  Test: {len(test):,}")

    # ── Eval transform and loaders ────────────────────────────────
    cfg = DistillConfig()
    transform = make_eval_transform(cfg.img_size, cfg.norm_mean, cfg.norm_std)

    train_loader = DataLoader(
        HDF5CellDataset(train, transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        HDF5CellDataset(test, transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Load teachers ─────────────────────────────────────────────
    print(f"\nLoading {len(args.teachers)} teacher(s)...")
    teachers = []
    for ckpt, arch in zip(args.teachers, args.teacher_archs):
        print(f"  {arch}: {ckpt}")
        teachers.append(load_teacher(ckpt, arch, device))

    # ── Cache ─────────────────────────────────────────────────────
    is_council = len(teachers) > 1
    prefix = "council_teacher" if is_council else f"{args.name}_teacher"

    train_logits, _, _, _ = cache_logits(teachers, train_loader, len(train), device, "train")
    test_logits, test_preds, test_labels, test_acc = cache_logits(
        teachers, test_loader, len(test), device, "test"
    )

    # ── Save ──────────────────────────────────────────────────────
    common_attrs = {"name": args.name, "n_teachers": len(teachers)}

    save_logits_h5(
        output_dir / f"{prefix}_logits_train.h5",
        train_logits,
        attrs=common_attrs,
    )
    save_logits_h5(
        output_dir / f"{prefix}_logits_test.h5",
        test_logits,
        labels=test_labels,
        predictions=test_preds,
        attrs={**common_attrs, "teacher_accuracy": test_acc},
    )

    print(f"\nTeacher test accuracy: {test_acc:.2f}%")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
