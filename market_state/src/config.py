"""Configuration loading for the market-state project.

All experimental knobs live in YAML config files under ``configs/`` and are
validated here. Configs are loaded into a nested dict that is also rendered
into the feature-cache key and experiment metadata (for reproducibility).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load a config YAML, falling back to ``configs/default.yaml``.

    Nested maps are deep-merged onto the defaults so a partial override file
    only has to specify the keys it wants to change.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    defaults = _defaults()
    cfg = _deep_merge(defaults, raw or {})
    validate_config(cfg)
    return cfg


def _defaults() -> dict[str, Any]:
    return {
        "data": {
            "timeframe": "5Min",
            "symbols": [],
            "start": "",
            "end": "",
            "raw_dir": "./data/raw",
        },
        "window": {"bars": 78},
        "wavelet": {
            "name": "morlet",
            "periods": [
                "15min",
                "30min",
                "1h",
                "2h",
                "4h",
                "8h",
                "1d",
            ],
            "normalization": "log",
            "nfreqs": 32,
        },
        "coherence": {
            "method": "gaussian_time",
            "smooth_time_steps": 5,
            "frequency_reduction": "magnitude_weighted",
            "frequency_weights": None,
        },
        "spectral": {
            "n_components": 8,
            "use_eigenvectors": False,
            "eigenvalue_normalize": "trace",
            "canonicalize_phase": True,
        },
        "model": {
            "state_dim": 64,
            "d_mode": 32,
            "d_freq": 64,
            "mamba_layers": 1,
            "mamba_d_state": 16,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "mamba_dropout": 0.0,
            "mamba_backend": "mamba3",
            "encoder": "projector",
        },
        "targets": {
            "realized_vol_horizons": [5, 12, 78],
            "return_horizons": [5, 12, 78],
            "max_drawdown_horizon": 78,
            "correlation_horizon": 78,
            "regime_thresholds": [0.005, 0.02],
        },
        "backtest": {
            "transaction_cost_bps": 1.0,
            "train_bars": 20000,
            "validate_bars": 5000,
            "test_bars": 2000,
            "step_bars": 5000,
            "expanding_window": True,
        },
        "inference": {"device": "cpu", "batch_size": 1},
        "reproducibility": {"seed": 42},
    }


def validate_config(cfg: dict[str, Any]) -> None:
    """Cheap sanity checks that catch the most common misconfigurations."""
    tf = cfg["data"]["timeframe"]
    if tf not in {"1Min", "5Min", "15Min", "1Hour", "1Day"}:
        raise ValueError(f"Unsupported timeframe {tf!r}")
    w = cfg["window"]["bars"]
    if not isinstance(w, int) or w < 4:
        raise ValueError("window.bars must be an int >= 4")
    if cfg["spectral"]["n_components"] < 1:
        raise ValueError("spectral.n_components must be >= 1")
    if cfg["coherence"]["frequency_reduction"] not in {"magnitude_weighted", "mean", "sqrt_weighted"}:
        raise ValueError("unsupported coherence.frequency_reduction")


def timeframe_bars_per_day(timeframe: str) -> int:
    """Approx bars per US trading session (6.5h) for a timeframe."""
    return {
        "1Min": 390,
        "5Min": 78,
        "15Min": 26,
        "1Hour": 6,
        "1Day": 1,
    }[timeframe]


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable 16-hex hash of the config, used for feature-cache addressing."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def symbol_set_hash(symbols: list[str]) -> str:
    payload = "\n".join(sorted(symbols))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
