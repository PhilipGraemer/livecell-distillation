"""Tests for livecell_distillation — loss and metrics modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from livecell_distillation.loss import DistillationLoss
from livecell_distillation.metrics import compute_ece


class TestDistillationLoss:

    def test_output_shape(self) -> None:
        criterion = DistillationLoss(alpha=0.2, temperature=4.0)
        student = torch.randn(8, 8)
        teacher = torch.randn(8, 8)
        labels = torch.randint(0, 8, (8,))
        loss, hard, soft = criterion(student, labels, teacher)
        assert loss.shape == ()
        assert isinstance(hard, float)
        assert isinstance(soft, float)

    def test_loss_positive(self) -> None:
        criterion = DistillationLoss()
        student = torch.randn(16, 8)
        teacher = torch.randn(16, 8)
        labels = torch.randint(0, 8, (16,))
        loss, _, _ = criterion(student, labels, teacher)
        assert loss.item() > 0

    def test_perfect_agreement_low_soft_loss(self) -> None:
        """When student and teacher logits are identical, KL should be ~0."""
        criterion = DistillationLoss(alpha=0.2, temperature=4.0)
        logits = torch.randn(32, 8)
        labels = torch.randint(0, 8, (32,))
        _, _, soft = criterion(logits, labels, logits.clone())
        assert soft < 1e-5

    def test_alpha_boundaries(self) -> None:
        """alpha=1.0 means pure CE, alpha=0.0 means pure KL."""
        student = torch.randn(8, 8)
        teacher = torch.randn(8, 8)
        labels = torch.randint(0, 8, (8,))

        pure_ce = DistillationLoss(alpha=1.0, temperature=4.0)
        loss_ce, hard_ce, _ = pure_ce(student, labels, teacher)
        # Loss should equal the hard loss component
        assert abs(loss_ce.item() - hard_ce) < 1e-4

    def test_temperature_effect(self) -> None:
        """Higher temperature should produce a different loss value."""
        student = torch.randn(16, 8)
        teacher = torch.randn(16, 8)
        labels = torch.randint(0, 8, (16,))

        low_t = DistillationLoss(alpha=0.2, temperature=1.0)
        high_t = DistillationLoss(alpha=0.2, temperature=10.0)

        loss_low, _, _ = low_t(student, labels, teacher)
        loss_high, _, _ = high_t(student, labels, teacher)

        assert loss_low.item() != pytest.approx(loss_high.item(), abs=1e-6)

    def test_backward(self) -> None:
        """Loss should be differentiable."""
        criterion = DistillationLoss()
        student = torch.randn(8, 8, requires_grad=True)
        teacher = torch.randn(8, 8)
        labels = torch.randint(0, 8, (8,))
        loss, _, _ = criterion(student, labels, teacher)
        loss.backward()
        assert student.grad is not None


class TestECE:

    def test_perfect_calibration(self) -> None:
        """A model that always predicts 100% on the correct class has ECE ≈ 0."""
        n = 100
        labels = np.arange(n) % 5
        probs = np.zeros((n, 5))
        for i in range(n):
            probs[i, labels[i]] = 1.0
        assert compute_ece(probs, labels) < 0.01

    def test_uniform_predictions(self) -> None:
        """Uniform predictions (confidence=0.2 on 5 classes, 20% accuracy) → low ECE."""
        n = 1000
        labels = np.arange(n) % 5
        probs = np.full((n, 5), 0.2)
        ece = compute_ece(probs, labels)
        assert ece < 0.05  # nearly calibrated

    def test_overconfident(self) -> None:
        """Confident wrong predictions should have high ECE."""
        n = 100
        labels = np.zeros(n, dtype=int)
        probs = np.zeros((n, 5))
        probs[:, 1] = 0.95  # confidently wrong
        probs[:, 0] = 0.05
        ece = compute_ece(probs, labels)
        assert ece > 0.5

    def test_returns_scalar(self) -> None:
        probs = np.random.dirichlet(np.ones(8), size=50)
        labels = np.random.randint(0, 8, size=50)
        ece = compute_ece(probs, labels)
        assert isinstance(ece, float)
