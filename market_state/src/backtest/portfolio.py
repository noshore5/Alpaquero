"""Portfolio construction and backtest simulation over held-out test folds.

Strategy: from the model's per-asset predicted forward returns at decision
time ``t``, build *target* weights for the holding window ``(t, t+H]``. We use
a deterministic, transparent crossing/rank scheme (no learned allocation):

  - "equal"        : equal-weight long-only across the shelf.
  - "signal"       : cross-sectional rank normalised to [-1, 1], long/short.
  - "shrunk_signal": signal weights shrunk toward equal-weight (reduces
                     turnover / transaction cost sensitivity for a
                     randomly-initialised model benchmark).

Weights are normalised so the gross exposure is 1.0 (fully invested). On each
decision step the realised portfolio return over the horizon is computed from
actual aligned asset returns, and a proportional transaction cost is charged
on turnover relative to the previous target. Costs use ``transaction_cost_bps``
(full round-turn on notional traded).

Returns the per-step equity curve and weight timeline for metrics/analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PortfolioResult:
    equity: np.ndarray            # [steps] cumulative equity (start 1.0)
    weights: np.ndarray           # [steps, A] target weights applied each step
    returns: np.ndarray           # [steps] realised net portfolio returns
    decision_idx: np.ndarray      # [steps] aligned rows at which decisions made
    n_steps: int


def _rank_signals(pred: np.ndarray, axis: int = -1) -> np.ndarray:
    """Cross-sectional ranks in [-1, 1] (ties averaged). NaN -> 0 weight."""
    n = pred.shape[-1]
    ranks = np.argsort(np.argsort(pred, axis=axis), axis=axis) + 1
    s = 2.0 * (ranks / max(n, 1)) - 1.0            # [.., A] in (-1, 1]
    s = np.where(np.isnan(pred) | (pred == 0), 0.0, s)
    # mean-center (zero net exposure) then normalise gross to 1
    s = s - np.nanmean(s, axis=axis, keepdims=True)
    s = np.where(np.isfinite(s), s, 0.0)
    gross = np.abs(s).sum(axis=axis, keepdims=True)
    s = np.divide(s, gross, out=np.zeros_like(s), where=gross > 1e-12)
    return s.astype(np.float32)


def build_target_weights(
    pred: np.ndarray,
    method: str = "shrunk_signal",
    shrink: float = 1.0,
) -> np.ndarray:
    """Target weight matrix [steps, A] from predicted returns [-inf, inf] + NaN.

    methods: "equal" | "signal" | "shrunk_signal"
    - "equal": all non-NaN assets get equal positive weight (gross 1).
    - "signal": cross-sectional rank long/short, zero net, gross 1.
    - "shrunk_signal": signal weights shrunk toward equal-weight by ``shrink``
      then renormalised to gross 1.
    """
    A = pred.shape[-1]
    if method == "equal":
        ok = np.isfinite(pred).astype(np.float32)
        w = ok / ok.sum(axis=-1, keepdims=True).clip(min=1e-12)
        return w.astype(np.float32)

    if method == "signal":
        return _rank_signals(pred)

    if method == "shrunk_signal":
        sig = _rank_signals(pred)                     # long/short, net 0
        eq = np.ones(A, dtype=np.float32) / max(A, 1)
        w = (1.0 - shrink) * eq + shrink * sig
        gross = np.abs(w).sum(axis=-1, keepdims=True).clip(min=1e-12)
        return (w / gross).astype(np.float32)

    raise ValueError(f"unknown portfolio method {method!r}")


class BacktestEngine:
    """Simulates a portfolio over held-out decision rows.

    The engine is *pure structure*: it takes actual asset one-bar returns
    (``rets[t, A]``, aligned), a set of decision rows, predictions, a holding
    horizon and costs, and produces an equity curve and weight timeline.
    """

    def __init__(
        self,
        *,
        transaction_cost_bps: float = 1.0,
        method: str = "shrunk_signal",
        shrink: float = 1.0,
    ) -> None:
        self.cost = float(transaction_cost_bps) * 1e-4
        self.method = method
        self.shrink = float(shrink)

    def run(
        self,
        returns: np.ndarray,
        pred_returns: np.ndarray,
        decision_idx: np.ndarray,
        *,
        hold: int = 1,
        weights: np.ndarray | None = None,
    ) -> PortfolioResult:
        """Simulate the portfolio.

        Parameters
        ----------
        returns     : [T, A] actual one-bar log returns, aligned.
        pred_returns: [steps, A] predicted H-ahead returns at each decision.
        decision_idx: [steps] aligned row indices of decisions (end of window).
        hold        : bars held per decision (== horizon H).
        weights     : optional precomputed target weights [steps, A]; else built
                      from pred_returns by ``method``.
        """
        steps, A = pred_returns.shape
        if weights is None:
            weights = build_target_weights(pred_returns, self.method, self.shrink)

        equity = np.empty(steps + 1, dtype=np.float32)
        equity[0] = 1.0
        net_ret = np.zeros(steps, dtype=np.float32)
        prev_w = None
        for s in range(steps):
            w_target = weights[s]
            # turnover cost on change from previous target; at t=0 we start
            # flat so the cost is on the full notional of the initial position
            ref = prev_w if prev_w is not None else np.zeros(A, dtype=np.float32)
            turnover = np.abs(w_target - ref).sum()

            idx = decision_idx[s]
            # hold window returns (t, t+H]: rows idx+1 .. idx+hold
            r_h = returns[idx + 1 : idx + 1 + hold]
            if r_h.shape[0] == 0:
                net_ret[s] = 0.0
                prev_w = w_target
                continue
            # multi-bar: cumulate the compounded portfolio return over H bars
            port_ret = 1.0
            for bar in r_h:
                bar = np.where(np.isfinite(bar), bar, 0.0)
                port_ret *= (1.0 + float(np.sum(w_target * bar)))
            net_ret[s] = port_ret - 1.0 - self.cost * turnover
            equity[s + 1] = equity[s] * (1.0 + net_ret[s])
            prev_w = w_target

        equity = equity[:-1]  # align to steps
        return PortfolioResult(
            equity=equity,
            weights=weights,
            returns=net_ret,
            decision_idx=decision_idx.astype(np.int64),
            n_steps=steps,
        )
