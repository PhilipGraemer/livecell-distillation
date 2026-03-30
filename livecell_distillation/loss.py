"""Distillation loss: combined hard (CE) and soft (KL) objectives.

The standard offline distillation loss is:

    L = α · CE(student_logits, y) + (1 − α) · T² · KL(soft_student ‖ soft_teacher)

where:

- ``soft_student = log_softmax(student_logits / T)``
- ``soft_teacher = softmax(teacher_logits / T)``

The ``T²`` factor compensates for the reduced gradient magnitude that
comes from dividing logits by temperature before the softmax.  Without
it, increasing T would shrink the KL gradient and effectively reduce
the weight of the soft loss.

**Why α = 0.2?**  We found that giving 80% weight to the soft loss
and only 20% to the hard labels lets the student absorb more of the
teacher's inter-class structure while still being grounded by the
true labels.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combined hard + soft distillation loss.

    Parameters
    ----------
    alpha : float
        Weight on the hard (CE) loss.  ``1 - alpha`` is applied to the
        soft (KL) loss.  Default: 0.2.
    temperature : float
        Softmax temperature for the soft targets.  Higher T produces
        softer probability distributions that reveal more of the
        teacher's learned inter-class similarities.
    """

    def __init__(self, alpha: float = 0.2, temperature: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute the combined loss.

        Returns
        -------
        loss : Tensor
            Scalar loss for backpropagation.
        hard_loss : float
            Detached CE component (for logging).
        soft_loss : float
            Detached KL component before T² scaling (for logging).
        """
        hard_loss = self.ce_loss(student_logits, labels)

        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher)

        loss = self.alpha * hard_loss + (1 - self.alpha) * (T ** 2) * soft_loss

        return loss, hard_loss.item(), soft_loss.item()
