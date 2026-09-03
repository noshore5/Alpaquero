#!/usr/bin/env python
"""Run walk-forward backtesting over pre-built features.

Loads ``data/features/features.npz`` (written by build_features.py), which
already enforces the feature/target alignment (row i = real time dropped+i),
then runs every fold of the walk-forward protocol. ``--epochs`` defaults to 0
(a **randomly initialised** model -- the leakage/plumbing benchmark). Setting
``--epochs`` > 0 trains a fresh model per fold on that fold's training rows
(expanding-window walk-forward), which is the path to an actually predictive
model.

Usage:
    ../.venv_market/bin/python scripts/run_backtest.py [--config ...] [--features ...] [--epochs 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from repro import set_seed
from backtest.run import run_walk_forward


def load_features(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    feats: np.ndarray = data["features"]
    obr: np.ndarray = data["one_bar_returns"]
    targets: dict[str, np.ndarray] = {}
    for k in data.files:
        if k.startswith("target__"):
            targets[k[len("target__"):]] = data[k]
    return feats, targets, obr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default="data/features/features.npz")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_seed(cfg.get("reproducibility", {}).get("seed", 42))
    feats, targets, obr = load_features(Path(args.features))
    print(f"[backtest] features {feats.shape}, targets {sorted(targets.keys())}, seed {seed}")
    device = args.device or cfg["inference"].get("device", "cpu")

    rep = run_walk_forward(cfg, feats, targets, obr, d_spec=feats.shape[1],
                           epochs=args.epochs, device=device, seed=seed)
    n_folds = len(rep.fold_results)
    print(f"[backtest] {n_folds} folds; pooled metrics:")
    for k, v in sorted(rep.pooled_metrics.items()):
        print(f"    {k:24s} {v: .4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
