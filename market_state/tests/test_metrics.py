"""Regression tests: IC metric must not report spurious skill on random input.

History: ``ic`` previously raveled the pooled ``[B, A]`` prediction/target
into one vector and computed a single Spearman. With a randomly-initialised
model the between-asset level offsets produced a large spurious IC (~0.5-0.95),
misleading the benchmark. The metric is now a per-timestep cross-sectional rank
IC averaged over time; on random input it must be ~0.
"""
from __future__ import annotations

import numpy as np
import pytest

from backtest.metrics import ic


def test_ic_is_zero_on_random_noised_model():
    rng = np.random.default_rng(1)
    # random model predictions with strong between-asset level offsets (this is
    # what triggered the old artifact) vs iid targets
    pred = 100.0 * rng.standard_normal((6, 1)).T * np.ones((500, 6)) + 0.01 * rng.standard_normal((500, 6))
    true = rng.standard_normal((500, 6))
    assert abs(ic(pred, true)) < 0.05


def test_ic_perfect_positive_for_monotone_target():
    # per-timestep cross-section perfectly ordered the same way as target
    rng = np.random.default_rng(2)
    true = rng.standard_normal((20, 5))
    pred = true * 3.0 + 0.0  # perfect monotone match cross-sectionally
    assert ic(pred, true) > 0.99


def test_ic_accepts_1d():
    rng = np.random.default_rng(3)
    p = rng.standard_normal(50)
    t = 2.0 * p + rng.standard_normal(50) * 0.01
    assert ic(p, t) > 0.99


def test_ic_naan_when_no_valid_timestep():
    rng = np.random.default_rng(4)
    pred = np.full((10, 3), 0.0)   # all zero -> filtered out
    true = rng.standard_normal((10, 3))
    assert np.isnan(ic(pred, true))
