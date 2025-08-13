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


def ic( pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson / Spearman information coefficient between pred and true."""
    from scipy.stats import pearsonr, spearmanr
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()
    m = np.isfinite(p) & np.isfinite(t) & ~np.isclose(t, 0.0) & ~np.isclose(p, 0.0)
    if m.sum() < 2:
        return float("nan")
    _, spe = spearmanr(p[m], t[m])
    return float(spe)


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
