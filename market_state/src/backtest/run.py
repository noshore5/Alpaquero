"""Top-level walk-forward orchestration over all folds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
            for k, v in (fr.portfolio_metrics or {}).items():
                if isinstance(v, (int, float)) and v == v:
                    pooled.setdefault(f"pf_{k}", []).append(v)
            vic = (fr.train_info or {}).get("best_ic")
            if isinstance(vic, (int, float)) and vic == vic:
                pooled.setdefault("val_ic", []).append(vic)
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
    normalize: bool = True,
    patience: int = 5,
    min_epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    save_dir: str | Path | None = None,
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

    save_path = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

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
            lr=float(lr),
            cost_bps=float(bt.get("transaction_cost_bps", 1.0)),
            portfolio_method=bt.get("portfolio_method", "shrunk_signal"),
            return_horizon=int(ret_h),
            one_bar_returns=one_bar_returns,
            normalize=normalize,
            patience=patience,
            min_epochs=min_epochs,
            seed=seed,
        )
        fr = runner.run_fold(fi)
        results.append(fr)
        if save_path is not None and epochs > 0 and getattr(runner, "last_model", None) is not None:
            import torch

            mean, std = (runner.last_norm if runner.last_norm is not None
                         else (np.zeros(d_spec, np.float32), np.ones(d_spec, np.float32)))
            torch.save(
                {"state_dict": runner.last_model.state_dict(),
                 "norm_mean": np.asarray(mean), "norm_std": np.asarray(std),
                 "model_cfg": model_cfg, "d_spec": d_spec, "window_len": window_len,
                 "return_horizon": int(ret_h), "fold_idx": fi,
                 "train_info": fr.train_info,
                 "test_metrics": fr.target_metrics,
                 "portfolio_metrics": fr.portfolio_metrics},
                save_path / f"fold_{fi:03d}.pt",
            )

    rep = WalkForwardReport(fold_results=results, fold_config=bt)
    rep.pooled_metrics = rep.aggregate()
    if save_path is not None:
        summary = {
            "pooled_metrics": rep.pooled_metrics,
            "n_folds": len(results),
            "per_fold": [
                {"fold": fr.fold_idx, "target_metrics": fr.target_metrics,
                 "portfolio_metrics": fr.portfolio_metrics,
                 "best_epoch": fr.train_info.get("best_epoch"),
                 "val_ic": fr.train_info.get("best_ic")}
                for fr in results
            ],
        }
        (save_path / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    return rep
