# livecell-distillation

Cross-architecture knowledge distillation for single-cell classification on [LIVECell](https://sartorius-research.github.io/LIVECell/). A didactic, well-commented implementation of temperature-scaled soft-label distillation across CNN and Vision Transformer families.

Part of the [LIVECell classification benchmark](https://github.com/PhilipGraemer/livecell-classification-benchmark) project.

## Key finding

**A weaker teacher can be a better teacher.** EVA-02 (92.26% macro F1) outperforms EfficientNet-B5 (93.53%) when distilling into EfficientNet-B0 - cross-architecture teaching transfers richer representations than same-family teaching despite lower raw accuracy. A pure 3×EVA-02 council achieves 93.42%.

## Method

Standard offline distillation with temperature-scaled soft labels:

```
L = α · CE(student_logits, y) + (1 − α) · T² · KL(soft_student ‖ soft_teacher)
```

where `soft_student = log_softmax(student_logits / T)` and `soft_teacher = softmax(teacher_logits / T)`.

**Council distillation** averages logits from multiple teacher checkpoints in logit space before computing the KL term. Averaging in logit space (rather than probability space) preserves confidence magnitude and lets the temperature parameter control softness.

## Installation

```bash
git clone https://github.com/PhilipGraemer/livecell-distillation.git
cd livecell-distillation
pip install -e ".[dev]"
```

## Usage

The pipeline has two steps: cache teacher logits, then train the student.

### Step 1: Cache teacher logits

```bash
# Single teacher (EVA-02)
python scripts/cache_logits.py \
    --teachers /path/to/eva02.pth \
    --teacher-archs eva02 \
    --name eva02 \
    --sample-lists /path/to/sample_lists.pkl \
    --output-dir output/distillation_eva02

# Council (3× EVA-02 seeds)
python scripts/cache_logits.py \
    --teachers eva_s42.pth eva_s43.pth eva_s44.pth \
    --teacher-archs eva02 eva02 eva02 \
    --name eva3_council \
    --sample-lists /path/to/sample_lists.pkl \
    --output-dir output/distillation_council

# First run (build train/val/test splits from scratch)
python scripts/cache_logits.py \
    --teachers /path/to/eva02.pth \
    --teacher-archs eva02 \
    --name eva02 \
    --data-dir /path/to/LIVECell/Cells \
    --output-dir output/distillation \
    --build-splits
```

### Step 2: Train student with distillation

```bash
# EVA-02 → EN-B0, sweep over T={3,4} and data={10%,100%}
python scripts/distill.py \
    --teacher-name "EVA-02" \
    --train-logits output/distillation_eva02/eva02_teacher_logits_train.h5 \
    --test-logits output/distillation_eva02/eva02_teacher_logits_test.h5 \
    --sample-lists output/distillation/sample_lists.pkl \
    --student efficientnet_b0 \
    --temperatures 3 4 \
    --data-fractions 0.10 1.00 \
    --output-dir output/distillation_eva02/enb0_results

# EN-B5 → EN-B0 (same student, different teacher)
python scripts/distill.py \
    --teacher-name "EN-B5" \
    --train-logits output/distillation_enb5/enb5_teacher_logits_train.h5 \
    --test-logits output/distillation_enb5/enb5_teacher_logits_test.h5 \
    --sample-lists output/distillation/sample_lists.pkl \
    --student efficientnet_b0 \
    --output-dir output/distillation_enb5/enb0_results
```

## Previous findings to replicate

| Teacher | Student | Student F1 |
|---|---|---|
| N/A (Baseline) | EfficientNet-B0 | **91.87%** |
| EVA-02 | EfficientNet-B0 | **92.67%** |
| EfficientNet-B5 | EfficientNet-B0 | 91.23% |
| 3×EVA-02 council | EfficientNet-B0 | **93.42%** |

## Repository structure

```
├── livecell_distillation/
│   ├── __init__.py
│   ├── caching.py           # Teacher logit extraction (single + council)
│   ├── config.py            # Constants, architecture registry, DistillConfig
│   ├── data.py              # HDF5CellDataset, DistillationDataset, transforms
│   ├── loss.py              # DistillationLoss (CE + T²·KL)
│   ├── metrics.py           # ECE, comprehensive evaluation
│   ├── plotting.py          # Training curves, confusion matrices
│   ├── training.py          # Training loop, orchestrator, CSV export
│   └── utils.py             # Seed setting
├── scripts/
│   ├── cache_logits.py      # CLI: cache teacher logits to HDF5
│   └── distill.py           # CLI: train student with offline distillation
├── tests/
│   ├── test_loss_and_metrics.py
│   └── test_config_and_data.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Design notes

This codebase is intentionally **didactic** — every module is commented to explain *why* each design choice was made, not just *what* the code does.

Key decisions documented inline:

- **Offline distillation** — why we cache teacher logits rather than running the teacher at train time (efficiency, sweep flexibility, memory)
- **T² scaling** — why the temperature squared factor is necessary in the loss to compensate for gradient magnitude reduction
- **Model selection by accuracy, not distillation loss** — why we use standard CE for validation rather than the distillation loss
- **Logit-space averaging for councils** — why we average logits rather than probabilities

## Tests

```bash
pytest
pytest --cov=livecell_distillation
```

## Citation

See [`livecell-classification-benchmark`](https://github.com/PhilipGraemer/livecell-classification-benchmark).

## License

MIT
