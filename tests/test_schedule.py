"""Learning rate schedule and optimizer parameter grouping."""

from __future__ import annotations

import pytest
import torch

from dvit.engine import TrainConfig, build_optimizer, lr_at
from dvit.model import build_model


class TestSchedule:
    def test_warmup_starts_at_zero_and_reaches_peak(self) -> None:
        cfg = TrainConfig(epochs=10, lr=1e-3, warmup_epochs=2)
        assert lr_at(0, 100, cfg) == 0.0
        assert lr_at(200, 100, cfg) == pytest.approx(1e-3)

    def test_cosine_decays_to_zero_at_the_end(self) -> None:
        cfg = TrainConfig(epochs=10, lr=1e-3, warmup_epochs=2)
        assert lr_at(1000, 100, cfg) == pytest.approx(0.0, abs=1e-9)

    def test_schedule_never_exceeds_peak(self) -> None:
        cfg = TrainConfig(epochs=10, lr=1e-3, warmup_epochs=2)
        assert max(lr_at(s, 100, cfg) for s in range(0, 1001)) <= 1e-3 + 1e-12


class TestOptimizer:
    def test_norms_and_biases_are_excluded_from_weight_decay(self) -> None:
        opt = build_optimizer(build_model(), TrainConfig(weight_decay=0.05))
        decay, no_decay = opt.param_groups
        assert decay["weight_decay"] == 0.05
        assert no_decay["weight_decay"] == 0.0
        # Every no-decay tensor must be 1D or a learned token/position embedding.
        assert all(p.ndim <= 1 or p.shape[0] == 1 for p in no_decay["params"])
