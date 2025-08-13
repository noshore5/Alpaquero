"""Integration smoke tests for the deterministic feature transform stack."""
from __future__ import annotations

import numpy as np
import torch
import pytest

from transforms.cwt import MorletCWTBank, financial_periods
from transforms.pipeline import FeaturePipelineConfig, FeaturePipeline
from transforms.hermitian import HermitianGraph
from transforms.wavelet_coherence import WaveletCoherence
from models.market_state import build_from_config


@pytest.fixture
def config():
    return {
        "data": {"symbols": ["A", "B", "C", "D"]},
        "window": {"bars": 32},
        "targets": {
            "realized_vol_horizons": [5],
            "return_horizons": [5],
            "max_drawdown_horizon": 5,
            "correlation_horizon": 5,
            "regime_thresholds": [0.005, 0.02],
        },
    }


def _pipe(A, T, periods_bars, **kw):
    cfg = FeaturePipelineConfig(device="cpu", use_eigenvectors=False,
                                n_components=5, coi_factor=6.0)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return FeaturePipeline(cfg, A, periods_bars, T)


def test_pipeline_deterministic_and_finite():
    periods = financial_periods(["15min", "1h", "4h"], "5Min")
    A, T = 4, 2048
    lr = np.random.default_rng(0).standard_normal((A, T)).astype(np.float32)
    pl = _pipe(A, T, periods)
    f1, d1 = pl.compute(lr)
    f2, d2 = pl.compute(lr)
    assert np.allclose(f1, f2, atol=1e-6)          # deterministic
    assert np.isfinite(f1).all()                    # no NaNs/Infs
    assert f1.shape == (T - d1, pl.d_features)      # dims + warm-up drop
    assert d1 == d2


def test_hermitian_error_zero():
    periods = financial_periods(["15min", "1h"], "5Min")
    A, T = 3, 512
    lr = np.random.default_rng(0).standard_normal((A, T)).astype(np.float32)
    pl = _pipe(A, T, periods)
    coeffs, _ = pl.bank.transform_causal(torch.from_numpy(lr))
    out = pl.coherence(coeffs)
    H = pl.hermitian.build(out["complex_coherence"], coeffs.shape[-1])
    err = float(pl.hermitian.herm_error(H).max().item())
    assert err < 1e-5


def test_model_forward_shapes(config):
    periods = financial_periods(["15min", "1h"], "5Min")
    A, T = 4, 512
    lr = np.random.default_rng(0).standard_normal((A, T)).astype(np.float32)
    pl = _pipe(A, T, periods)
    feats, _ = pl.compute(lr)
    model = build_from_config(dict(state_dim=32, mamba_layers=1, n_assets=A),
                              pl.d_features, targets_cfg=config["targets"])
    x = torch.from_numpy(feats[-32:][None])         # [1, 32, D]
    out = model(x)
    assert "return_5" in out and "regime" in out
    assert out["return_5"].shape == (1, A)
    n_classes = len(config["targets"]["regime_thresholds"]) + 1
    assert out["regime"].shape == (1, A, n_classes)
