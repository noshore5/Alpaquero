"""Causal target computation.

Every target is computed strictly from bars ``t+1 .. t+H`` and is stored at
row ``t`` (the decision time). The current bar ``t`` is never included in a
target. Missing closes (NaN) propagate to NaN targets (no fabrication).

Targets produced per config (all rows indexed by the aligned timeline):

  - realized_vol_H : realized volatility of log returns over ``t+1..t+H``,
                      annualised placeholder -- actually here just the sample
                      std of H one-bar log returns (a scalar proxy), scale-free.
  - return_H       : ``log(P[t+H]/P[t])`` (forward cumulative log return).
  - max_drawdown_H : worst peak-to-trough drawdown of the price path
                      ``t+1..t+H``, expressed as a positive fraction.
  - correlation    : sample correlation matrix of one-bar log returns over the
                      correlation horizon (N x N symmetric, diag=1).
  - regime         : deterministic class derived from realized_vol_5 relative to
                      ``regime_thresholds``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _log_returns(prices: np.ndarray) -> np.ndarray:
    out = np.empty_like(prices)
    out[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.log(prices[1:] / prices[:-1])
    return out


def realized_vol(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Sample std of H one-bar log returns over t+1..t+H, [T, A]."""
    T, A = prices.shape
    rets = _log_returns(prices)            # [T, A]
    out = np.full((T, A), np.nan)
    for t in range(T - horizon):
        seg = rets[t + 1 : t + 1 + horizon]
        out[t] = np.nanstd(seg, axis=0)
    return out


def forward_return(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Cumulative log return over (t, t+H]: log(P[t+H]/P[t]), [T, A]."""
    T, A = prices.shape
    out = np.full((T, A), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:-horizon] = np.log(prices[horizon:] / prices[:-horizon])
    return out


def max_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Worst peak-to-trough path drawdown in t+1..t+H (positive fraction), [T, A]."""
    T, A = prices.shape
    out = np.full((T, A), np.nan)
    for t in range(T - horizon):
        path = prices[t + 1 : t + 1 + horizon]
        cummax = np.maximum.accumulate(path, axis=0)
        dd = 1.0 - path / cummax
        out[t] = np.nanmax(dd, axis=0)
    return out


def correlation_matrix(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Sample corr matrix of one-bar returns over t+1..t+H, [T, A, A]."""
    T, A = prices.shape
    rets = _log_returns(prices)
    out = np.full((T, A, A), np.nan)
    for t in range(T - horizon):
        seg = rets[t + 1 : t + 1 + horizon]
        segd = np.nan_to_num(seg)
        if segd.shape[0] < 2:
            continue
        C = np.corrcoef(segd.T)
        C[np.isnan(C)] = 0.0
        np.fill_diagonal(C, 1.0)
        out[t] = C
    return out


def regime_classes(rv5: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Discretise realized-vol into regimes by threshold (per asset), [T, A] int."""
    th = np.asarray(sorted([float(x) for x in thresholds]))
    digit = np.digitize(np.nan_to_num(rv5, nan=-1.0), bins=th)
    return digit.astype(np.int64)


class TargetEngine:
    """Computes all targets for a given timeline/prices and config."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def compute(self, prices: np.ndarray) -> dict[str, np.ndarray]:
        """prices : [T, A] aligned close-price matrix (raw levels, not shifted).
        Returns target dict keyed by name (see module docstring) -> [T, ...]."""
        tc = self.cfg.get("targets", {})
        out: dict[str, np.ndarray] = {}

        for h in tc.get("realized_vol_horizons", []):
            out[f"realized_vol_{h}"] = realized_vol(prices, int(h))
        for h in tc.get("return_horizons", []):
            out[f"return_{h}"] = forward_return(prices, int(h))
        mdd_h = tc.get("max_drawdown_horizon")
        if mdd_h:
            out["max_drawdown"] = max_drawdown(prices, int(mdd_h))
        corr_h = tc.get("correlation_horizon")
        if corr_h:
            out["correlation"] = correlation_matrix(prices, int(corr_h))
        th = tc.get("regime_thresholds", [])
        if th:
            rv5 = out.get("realized_vol_5")
            if rv5 is None:
                rv5 = realized_vol(prices, 5)
            out["regime"] = regime_classes(rv5, th)
        return out