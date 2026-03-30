"""Tests for config and data modules."""

from __future__ import annotations

from livecell_distillation.config import ARCH_REGISTRY, CELL_TYPES, NUM_CLASSES, DistillConfig
from livecell_distillation.data import make_eval_transform, make_train_transform
from livecell_distillation.utils import set_seeds


class TestConfig:

    def test_cell_types_count(self) -> None:
        assert NUM_CLASSES == 8
        assert len(CELL_TYPES) == 8

    def test_arch_registry_keys(self) -> None:
        assert "eva02" in ARCH_REGISTRY
        assert "efficientnet_b0" in ARCH_REGISTRY
        assert "efficientnet_b5" in ARCH_REGISTRY

    def test_distill_config_defaults(self) -> None:
        cfg = DistillConfig()
        assert cfg.alpha == 0.2
        assert cfg.temperature == 4.0
        assert cfg.img_size == 224
        assert cfg.patience == 5


class TestTransforms:

    def test_train_transform_has_augmentation(self) -> None:
        tf = make_train_transform()
        names = [t.__class__.__name__ for t in tf.transforms]
        assert "RandomHorizontalFlip" in names
        assert "RandomRotation" in names

    def test_eval_transform_no_augmentation(self) -> None:
        tf = make_eval_transform()
        names = [t.__class__.__name__ for t in tf.transforms]
        assert "RandomHorizontalFlip" not in names
        assert "RandomRotation" not in names

    def test_custom_size(self) -> None:
        tf = make_eval_transform(img_size=128)
        # First transform is Resize
        assert tf.transforms[0].size == (128, 128)


class TestSetSeeds:

    def test_runs_without_error(self) -> None:
        set_seeds(42)
