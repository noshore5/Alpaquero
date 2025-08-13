"""Top-level walk-forward orchestration over all folds."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from datasets.splits import walk_forward_folds
from models.market_state import build_from_config
from .walk_forward import WalkForwardRunner, FoldResult


@dataclass
class WalkForwardReport:
    fold_results: list[FoldResult]
    fold_config: dict
    pooled_metrics: dict[str, float] = field(default_factory=dict)

    def aggregate(self, aggregates=("ic", "hit", "acc")) -> dict[str, float]:
        pooled: dict[str, list] = {}
        for fr in self.fold_results:
            for k, v in fr.target_metrics.items():
                if v == v:  # finite
                    pooled.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in pooled.items()}


def run_walk_forward(
    config: dict,
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    one_bar_returns: np.ndarray | None = None,
    *,
    d_spec: int,
    epochs: int = 0,
    device: str = "cpu",
) -> WalkForwardReport:
    """Run every fold and collect results.

    Parameters
    ----------
    config : full project config (data.window.bars, targets, backtest, model).
    features : [T, D] causal feature rows (post causal-warm-up).
    targets : dict name -> [T, ...] aligned target arrays.
    one_bar_returns : [T, A] aligned one-bar log returns (for portfolio).
    """
    bt = config.get("backtest", {})
    n_rows = features.shape[0]
    folds = walk_forward_folds(
        n_rows,
        train_bars=int(bt.get("train_bars", 20000)),
        validate_bars=int(bt.get("validate_bars", 5000)),
        test_bars=int(bt.get("test_bars", 2000)),
        step_bars=int(bt.get("step_bars", 5000)),
        expanding_window=bool(bt.get("expanding_window", True)),
    )
    window_len = int(config.get("window", {}).get("bars", 78))
    model_cfg = dict(config.get("model", {}))
    model_cfg["n_assets"] = len(config.get("data", {}).get("symbols", []))

    def factory():
        return build_from_config(model_cfg, d_spec, targets_cfg=config.get("targets", {}))

    tgt_cfg = config.get("targets", {})
    ret_h = tgt_cfg.get("return_horizons", [1])[0]

    results = []
    for fi, fold in enumerate(folds):
        runner = WalkForwardRunner(
            factory,
            features,
            targets,
            fold,
            window_len=window_len,
            device=device,
            epochs=epochs,
            batch_size=int(bt.get("batch_size", 64)),
            cost_bps=float(bt.get("transaction_cost_bps", 1.0)),
            portfolio_method=bt.get("portfolio_method", "shrunk_signal"),
            return_horizon=int(ret_h),
            one_bar_returns=one_bar_returns,
        )
        results.append(runner.run_fold(fi))

    rep = WalkForwardReport(fold_results=results, fold_config=bt)
    rep.pooled_metrics = rep.aggregate()
    return rep
