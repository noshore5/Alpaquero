"""Backtest / evaluation metrics.

All metrics are computed on out-of-fold (held-out, walk-forward) predictions
only -- never on in-sample data. Reported per prediction series and/or pooled.
"""
from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, periods_per_year: float | None = None,
                 risk_free: float = 0.0) -> float:
    """Annualised Sharpe of a return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or np.std(r) == 0:
        return float("nan")
    ann = periods_per_year if periods_per_year else 1.0
    excess = r - risk_free
    return float(np.mean(excess) / np.std(r, ddof=1) * np.sqrt(ann))


def max_drawdown(equity: np.ndarray) -> float:
    """Max peak-to-trough drawdown of a cumulative-equity curve."""
    eq = np.asarray(equity, dtype=float)
    eq = eq[np.isfinite(eq)]
    if eq.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(eq)
    return float(np.max(1.0 - eq / running_max))


def annualized_return(equity: np.ndarray, periods_per_year: float) -> float:
    """CAGR from a cumulative-equity series (final / first)."""
    eq = np.asarray(equity, dtype=float)
    eq = eq[np.isfinite(eq)]
    if eq.size < 2 or eq[0] <= 0:
        return float("nan")
    total = eq[-1] / eq[0]
    return float(total ** (periods_per_year / (eq.size - 1)) - 1.0)


def hit_rate(pred: np.ndarray, true: np.ndarray, direction: bool = True) -> float:
    """Fraction of predictions matching the sign (direction) of the target."""
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() == 0:
        return float("nan")
    if direction:
        return float((np.sign(p[m]) == np.sign(t[m])).mean())
    return float((p[m] == t[m]).mean())


def ic(pred: np.ndarray, true: np.ndarray) -> float:
    """Cross-sectional (rank) IC: per-timestep Spearman across assets, averaged.

    Inputs are ``[B, A]`` per-asset prediction and target series (B decision
    times, A assets). For each timestep we Spearman-correlate the cross-section
    of predictions against the cross-section of realized targets, then average
    over timesteps with at least two finite observations.

    Why not a pooled raveled correlation? Pooling ``[B, A]`` into one vector
    mixes the cross-sectional and temporal dimensions and lets *between-asset*
    level offsets (e.g. a randomly-initialised model's per-asset biases, or the
    always-positive realized-vol target) create a large spurious correlation --
    exactly the IC~0.5..0.95 artifact seen with a randomly-initialised model.
    Returning to a per-timestep cross-section removes that contamination and
    measures the skill the long/short portfolio actually uses (ranking assets
    at each decision).

    A 1-D input is treated as a single timestep (Spearman over its entries),
    so ``ic`` also works for a univariate return series.
    """
    from scipy.stats import spearmanr
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    p = np.atleast_2d(p)
    t = np.atleast_2d(t)
    if p.shape != t.shape:
        raise ValueError(f"pred/true shape mismatch: {p.shape} vs {t.shape}")

    per_time = np.empty(p.shape[0], dtype=float)
    for b in range(p.shape[0]):
        pb = p[b]
        tb = t[b]
        m = np.isfinite(pb) & np.isfinite(tb) & ~np.isclose(pb, 0.0) & ~np.isclose(tb, 0.0)
        if m.sum() >= 2:
            per_time[b] = spearmanr(pb[m], tb[m])[0]
        else:
            per_time[b] = np.nan
    vals = per_time[np.isfinite(per_time)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanmean(vals))


def regime_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Accuracy for integer-coded regime classification."""
    p = np.asarray(pred, dtype=int).ravel()
    t = np.asarray(true, dtype=int).ravel()
    m = (t >= 0)
    if m.sum() == 0:
        return float("nan")
    return float((p[m] == t[m]).mean())


def summarize(metrics: dict[str, float]) -> dict[str, float]:
    return {k: (float("nan") if v is None or (isinstance(v, float) and v != v)
                else float(v)) for k, v in metrics.items()}
