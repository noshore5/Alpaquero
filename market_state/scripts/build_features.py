#!/usr/bin/env python
"""Build aligned features + targets from raw bars into ``data/features``.

Pipeline:
    raw per-symbol parquet bars -> align onto a common UTC grid (sufficient
    coverage required, gaps stay NaN) -> log-returns -> causal FeaturePipeline
    -> per-timestep spectral features, plus aligned forward/realised targets and
    one-bar returns.

Causality / alignment invariant enforced here: the causal transform trims the
first ``dropped`` bars as warm-up. Feature row ``i`` corresponds to real
timeline index ``dropped + i``, so ALL target arrays and the one-bar return
matrix are sliced by ``[dropped : dropped + T_out]`` to stay row-aligned with
``features``.

Usage:
    ../.venv_market/bin/python scripts/build_features.py [--config ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, config_hash
from repro import set_seed
from data.alignment import align_bars
from transforms.cwt import financial_periods
from transforms.pipeline import FeaturePipeline, FeaturePipelineConfig
from datasets.targets import TargetEngine


def load_raw_bars(raw_dir: str, symbols: list[str], timeframe: str) -> dict[str, pd.DataFrame]:
    roots = {"stocks": Path(raw_dir) / "stocks", "crypto": Path(raw_dir) / "crypto"}
    bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        is_crypto = "/" in sym
        fname = sym.replace("/", "_") + ".parquet"
        p = roots["crypto" if is_crypto else "stocks"] / fname
        if not p.exists():
            print(f"  [warn] missing {sym} at {p}, skipping")
            continue
        bars[sym] = pd.read_parquet(p)
    return bars


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/features")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("reproducibility", {}).get("seed", 42))
    d = cfg["data"]
    bars = load_raw_bars(d["raw_dir"], d["symbols"], d["timeframe"])
    if not bars:
        print("[build] no raw bars found; run scripts/download.py first")
        return 1

    aligned = align_bars(bars, d["timeframe"],
                         min_coverage=0.7, start=d.get("start"), end=d.get("end"),
                         symbols=d["symbols"])
    prices = aligned.prices[aligned.valid_mask]
    print(f"[build] timeline {prices.shape[0]} bars x {prices.shape[1]} assets")

    # --- features ---
    from data.preprocessing import log_returns
    rets_df = log_returns(prices)
    lr = np.asarray(rets_df.T, dtype=np.float32)          # [A, T] log returns
    lr = np.nan_to_num(lr, nan=0.0)                        # gaps explicit-0 for CWT input
    T_full = lr.shape[1]
    A = lr.shape[0]

    periods = financial_periods(cfg["wavelet"]["periods"], d["timeframe"])
    w = cfg["wavelet"]
    fpc = FeaturePipelineConfig(
        device=cfg["inference"].get("device", "cpu"),
        smooth_time_steps=cfg["coherence"].get("smooth_time_steps", 5),
        frequency_reduction=cfg["coherence"].get("frequency_reduction", "magnitude_weighted"),
        n_components=cfg["spectral"].get("n_components", 8),
        eigenvalue_normalize=cfg["spectral"].get("eigenvalue_normalize", "trace"),
        canonicalize_phase=bool(cfg["spectral"].get("canonicalize_phase", True)),
        causal=bool(w.get("causal", True)),
        coi_factor=float(w.get("coi_factor", 3.0)),
        use_eigenvectors=bool(cfg["spectral"].get("use_eigenvectors", False)),
    )
    pl = FeaturePipeline(fpc, A, periods, T_full)
    features, dropped = pl.compute(lr)                    # [T_out, D]
    T_out = features.shape[0]
    print(f"[build] features {features.shape}, causal warm-up dropped={dropped}")

    # --- targets (aligned to feature rows via [dropped : dropped + T_out]) ---
    te = TargetEngine(cfg)
    tg_full = te.compute(prices.values.astype(np.float64))
    targets = {k: v[dropped : dropped + T_out] for k, v in tg_full.items()}

    obr_full = _log_one_bar(prices.values.astype(np.float64))
    one_bar_returns = obr_full[dropped : dropped + T_out]  # [T_out, A]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "features.npz",
        features=features,
        one_bar_returns=one_bar_returns,
        **{f"target__{k}": v for k, v in targets.items()},
    )
    meta = {
        "n_assets": A,
        "n_features": features.shape[1],
        "n_rows": T_out,
        "dropped": dropped,
        "d_features": pl.d_features,
        "config_hash": config_hash(cfg),
        "symbols": d["symbols"],
        "timeline_start": str(prices.index[0]),
        "timeline_end": str(prices.index[-1]),
        "target_keys": sorted(targets.keys()),
        "feature_pipeline": {
            "periods": cfg["wavelet"]["periods"],
            "coi_factor": fpc.coi_factor,
            "use_eigenvectors": fpc.use_eigenvectors,
            "n_components": fpc.n_components,
        },
    }
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[build] wrote features.npz + meta.json to {out_dir}")
    print(f"[build] targets: {sorted(targets.keys())}")
    return 0


def _log_one_bar(prices: np.ndarray) -> np.ndarray:
    out = np.empty_like(prices)
    out[0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.log(prices[1:] / prices[:-1])
    d = np.where(np.isfinite(out), out, 0.0)
    return d.astype(np.float32)


if __name__ == "__main__":
    sys.exit(main())
