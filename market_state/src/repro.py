"""Reproducibility helpers.

``reproducibility.seed`` in the config is applied through ``set_seed`` at the
top of every entry-point script (build_features, train, backtest, benchmark)
so a run is deterministic given (config, data).
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (if available) torch. Returns the seed used."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Best-effort; some ops (complex pscan) have no deterministic kernel.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:  # pragma: no cover - torch always present in this project
        pass
    return seed
