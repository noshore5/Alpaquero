#!/usr/bin/env python
"""Walk-forward TRAINING entry point for the market-state model.

Unlike ``run_backtest.py`` (which defaults to a randomly-initialised model and
is the leakage/plumbing benchmark), this script trains a fresh model on every
walk-forward fold with proper hygiene:

  * per-fold feature normalization fit on TRAIN rows only (z-score), applied
    to train / validation / test windows;
  * early stopping on the validation cross-sectional IC of the traded
    ``return_H`` head (``backtest.return_horizon`` = first return horizon);
  * deterministic given (config, data, seed) -- ``reproducibility.seed`` is
    applied, each fold uses ``seed + fold_idx``;
  * per-fold checkpoints (``fold_XXX.pt``: weights + norm stats + config) and a
    ``report.json`` summary written under ``--out``.

The held-out (test-fold) predictions drive the same cross-sectional long/short
portfolio and rank-IC metrics as the benchmark, so numbers are comparable.

Usage (from market_state/):
    ../.venv_market/bin/python scripts/train.py --config configs/crypto.yaml \
        --features data/features/features.npz --out runs/crypto_01 \
        --max-epochs 40 --patience 6 --device cpu

On a GPU box add ``--device cuda``. Build features first with build_features.py.
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


def load_features(path: Path):
    data = np.load(path, allow_pickle=False)
    feats = data["features"]
    obr = data["one_bar_returns"]
    targets = {k[len("target__"):]: data[k] for k in data.files if k.startswith("target__")}
    return feats, targets, obr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default="data/features/features.npz")
    ap.add_argument("--out", default="runs/latest", help="checkpoint + report directory")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: inference.device)")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="override reproducibility.seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = set_seed(args.seed if args.seed is not None else cfg.get("reproducibility", {}).get("seed", 42))
    device = args.device or cfg["inference"].get("device", "cpu")

    feats, targets, obr = load_features(Path(args.features))
    print(f"[train] features {feats.shape}  targets {sorted(targets)}  device {device}  seed {seed}")
    print(f"[train] max_epochs {args.max_epochs}  patience {args.patience}  lr {args.lr}"
          f"  normalize {not args.no_normalize}")

    rep = run_walk_forward(
        cfg, feats, targets, obr,
        d_spec=feats.shape[1],
        epochs=args.max_epochs,
        device=device,
        normalize=not args.no_normalize,
        patience=args.patience,
        min_epochs=args.min_epochs,
        lr=args.lr,
        seed=seed,
        save_dir=args.out,
    )

    n = len(rep.fold_results)
    print(f"\n[train] {n} folds trained; checkpoints + report.json -> {args.out}")
    print("[train] pooled walk-forward metrics (held-out test folds):")
    for k, v in sorted(rep.pooled_metrics.items()):
        print(f"    {k:26s} {v: .4f}")

    ret_h = cfg.get("targets", {}).get("return_horizons", [1])[0]
    key = f"return_{ret_h}_ic"
    if key in rep.pooled_metrics:
        ics = [fr.target_metrics.get(key) for fr in rep.fold_results]
        ics = np.array([x for x in ics if x is not None and x == x], dtype=float)
        if ics.size > 1:
            tstat = ics.mean() / (ics.std(ddof=1) / np.sqrt(ics.size))
            print(f"\n[train] traded head {key}: mean {ics.mean():+.4f} over {ics.size} folds, "
                  f"t-stat {tstat:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
