"""Walk-forward evaluation of the market-state model on held-out folds.

Orchestrates, per fold:

  1. Slice features/targets to the fold's train / validate / test rows.
  2. Build causal windows (length W ending at each decision row) for train
     and test.
  3. Optionally train a fresh model on the training windows, or (for the
     no-training benchmark) simply use a freshly initialised model.
  4. Run inference on the *test* windows and gather held-out predictions.
  5. Compute target metrics and run the portfolio backtest on those held-out
     predictions.

Causality is preserved end-to-end: feature rows are causal (bars <= t),
targets are strictly future (t+1..t+H), folds are chronological and
non-overlapping (splits.walk_forward_folds enforces the test-before-next-train
guard), and metrics/backtests are computed only on test folds, never training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from models.market_state import MarketStateModel
from datasets.windows import split_contiguous
from .portfolio import BacktestEngine
from . import metrics as M


@dataclass
class FoldResult:
    fold_idx: int
    preds: dict[str, np.ndarray]
    target_metrics: dict[str, float]
    portfolio: object | None = None
    portfolio_metrics: dict[str, float] = field(default_factory=dict)
    train_info: dict = field(default_factory=dict)


def model_forward(model: MarketStateModel, x: torch.Tensor) -> dict[str, torch.Tensor]:
    return model(x)


@torch.no_grad()
def _predict(model: MarketStateModel, x: np.ndarray, device) -> dict[str, np.ndarray]:
    model.eval()
    model.to(device)
    batches = torch.split(torch.as_tensor(x, dtype=torch.float32), 256)
    collected: dict[str, list] = {}
    for bx in batches:
        out = model(bx.to(device))
        for k, v in out.items():
            collected.setdefault(k, []).append(v.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


class TrainingLoop:
    """Minimal training loop over causal windows for the market-state model.

    Loss per head:
      - regression heads  : smooth-L1 (Huber) on finite targets.
      - correlation head  : MSE on the (symmetrised) matrix.
      - regime head       : cross-entropy on per-asset class logits.
    Only losses with at least one finite/sample target contribute.
    """

    def __init__(self, model: MarketStateModel, *, lr: float = 1e-3, weight_decay: float = 1e-5) -> None:
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    @staticmethod
    def _batch_loss(model: MarketStateModel, x: torch.Tensor, bt: dict[str, torch.Tensor]) -> torch.Tensor | None:
        out = model(x)
        total = None
        for name, spec in model.head_specs_by_name.items():
            t = bt.get(name)
            if t is None:
                continue
            p = out[name]
            if spec.kind == "regression":
                m = torch.isfinite(t)
                if not m.any():
                    continue
                loss = F.smooth_l1_loss(p[m], t[m])
            elif spec.kind == "correlation":
                loss = F.mse_loss(p, t)
            else:  # classification
                loss = F.cross_entropy(p.permute(0, 2, 1), t.long())
            total = loss if total is None else total + loss
        return total

    def _run_epoch(self, x, targets, *, batch_size, device, rng) -> float:
        self.model.train()
        n = x.shape[0]
        order = rng.permutation(n)
        last = 0.0
        for i in range(0, n, batch_size):
            idx = np.sort(order[i : i + batch_size])
            bx = torch.as_tensor(x[idx], dtype=torch.float32, device=device)
            bt = {k: torch.as_tensor(v[idx], dtype=torch.long if k == "regime" else torch.float32,
                                     device=device) for k, v in targets.items()}
            self.opt.zero_grad()
            loss = self._batch_loss(self.model, bx, bt)
            if loss is None:
                continue
            loss.backward()
            self.opt.step()
            last = float(loss.item())
        return last

    def fit(self, x: np.ndarray, targets: dict[str, np.ndarray], epochs: int = 1,
            batch_size: int = 32, device: str = "cpu", seed: int = 0) -> float:
        self.model.to(device)
        rng = np.random.default_rng(seed)
        last = 0.0
        for _ in range(epochs):
            last = self._run_epoch(x, targets, batch_size=batch_size, device=device, rng=rng)
        return last

    def fit_early_stop(
        self,
        x_train: np.ndarray,
        t_train: dict[str, np.ndarray],
        x_val: np.ndarray,
        t_val: dict[str, np.ndarray],
        *,
        monitor_key: str,
        max_epochs: int = 50,
        patience: int = 5,
        min_epochs: int = 1,
        batch_size: int = 32,
        device: str = "cpu",
        seed: int = 0,
        log: bool = True,
    ) -> dict:
        """Train up to ``max_epochs``, keeping the weights with the best
        validation cross-sectional IC on ``monitor_key`` (a ``return_H`` head).
        Restores the best weights into ``self.model`` before returning.
        """
        import copy

        self.model.to(device)
        rng = np.random.default_rng(seed)
        best_ic = -np.inf
        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch = -1
        history: list[dict] = []
        for ep in range(max_epochs):
            train_loss = self._run_epoch(x_train, t_train, batch_size=batch_size,
                                         device=device, rng=rng)
            vp = _predict(self.model, x_val, device)
            vt = t_val.get(monitor_key)
            val_ic = M.ic(vp[monitor_key], vt) if (vt is not None and monitor_key in vp) else float("nan")
            val_ic = float(val_ic) if val_ic == val_ic else -np.inf
            history.append({"epoch": ep, "train_loss": train_loss, "val_ic": val_ic})
            if log:
                print(f"    ep {ep:3d}  train_loss {train_loss: .4f}  val_ic {val_ic: .4f}"
                      + ("  *" if val_ic > best_ic else ""))
            if val_ic > best_ic:
                best_ic, best_epoch = val_ic, ep
                best_state = copy.deepcopy(self.model.state_dict())
            if ep + 1 >= min_epochs and ep - best_epoch >= patience:
                break
        self.model.load_state_dict(best_state)
        return {"best_ic": best_ic if np.isfinite(best_ic) else float("nan"),
                "best_epoch": best_epoch, "epochs_run": len(history), "history": history,
                "best_state": best_state}


class WalkForwardRunner:
    def __init__(
        self,
        model_factory,
        features: np.ndarray,          # [T, D] causal feature rows (post warm-up)
        targets: dict[str, np.ndarray],
        fold,
        *,
        window_len: int,
        device: str = "cpu",
        epochs: int = 0,
        batch_size: int = 64,
        lr: float = 1e-3,
        cost_bps: float = 1.0,
        portfolio_method: str = "shrunk_signal",
        return_horizon: int = 1,
        one_bar_returns: np.ndarray | None = None,
        normalize: bool = True,
        patience: int = 5,
        min_epochs: int = 1,
        seed: int = 0,
    ) -> None:
        self.model_factory = model_factory
        self.features = features
        self.targets = targets
        self.fold = fold
        self.window_len = int(window_len)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.cost_bps = cost_bps
        self.portfolio_method = portfolio_method
        self.return_horizon = return_horizon
        self.one_bar_returns = one_bar_returns
        self.normalize = bool(normalize)
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.seed = int(seed)
        self._norm: tuple[np.ndarray, np.ndarray] | None = None
        self.train_info: dict = {}

    def _windows(self, rows: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build [B, W, D] windows ending at each row in ``rows``, plus targets.

        Rows must be >= window_len-1 and within the feature series. The last
        valid decision row is the last row with finite targets.
        """
        t_total = self.features.shape[0]
        rows = rows[rows >= self.window_len - 1]
        rows = rows[rows < t_total]
        if rows.size == 0:
            raise ValueError("no valid decision rows for windows")
        # trim to rows where all targets are finite
        targ = {k: self.targets[k] for k in self.targets if self.targets[k].shape[0] == t_total}
        rows = rows[rows < min(arr.shape[0] for arr in targ.values())]
        W = self.window_len
        idx = rows[:, None] - np.arange(W - 1, -1, -1)[None, :]
        x = self.features[idx].astype(np.float32)
        if self._norm is not None:
            mean, std = self._norm
            x = (x - mean) / std
        bt = {k: arr[rows] for k, arr in targ.items()}
        return x, bt

    def _fit_normalization(self) -> None:
        self._norm = None
        if not self.normalize:
            return
        tr = self.features[self.fold.train]
        tr = tr[np.isfinite(tr).all(axis=1)]
        if tr.shape[0] < 2:
            return
        mean = tr.mean(axis=0)
        std = tr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self._norm = (mean.astype(np.float32), std.astype(np.float32))

    def run_fold(self, fold_idx: int) -> FoldResult:
        self._fit_normalization()
        x_test, tb_test = self._windows(self.fold.test)
        model = self.model_factory()
        if self.epochs > 0:
            x_train, tb_train = self._windows(self.fold.train)
            x_val, tb_val = self._windows(self.fold.validate)
            loop = TrainingLoop(model, lr=self.lr)
            info = loop.fit_early_stop(
                x_train, tb_train, x_val, tb_val,
                monitor_key=f"return_{self.return_horizon}",
                max_epochs=self.epochs, patience=self.patience,
                min_epochs=self.min_epochs, batch_size=self.batch_size,
                device=self.device, seed=self.seed + fold_idx,
            )
            self.train_info = {k: v for k, v in info.items() if k != "best_state"}
        self.last_model = model
        self.last_norm = self._norm
        preds = _predict(model, x_test, self.device)
        result = FoldResult(fold_idx=fold_idx, preds=preds,
                            target_metrics=self._compute_target_metrics(preds, tb_test),
                            train_info=dict(self.train_info))
        pr, pm = self._portfolio(preds, tb_test)
        result.portfolio, result.portfolio_metrics = pr, pm
        return result

    def _compute_target_metrics(self, preds, tb) -> dict[str, float]:
        mets: dict[str, float] = {}
        for k, p in preds.items():
            t = tb.get(k)
            if t is None:
                continue
            if k == "correlation":
                continue
            if k == "regime":
                pred_cls = p.argmax(-1) if p.ndim == 3 else p.astype(int)
                mets[f"{k}_acc"] = M.regime_accuracy(pred_cls, t)
            else:
                mets[f"{k}_ic"] = M.ic(p, t)
                mets[f"{k}_hit"] = M.hit_rate(p, t)
        return mets

    def _portfolio(self, preds, tb):
        if self.one_bar_returns is None:
            return None, {}
        hold = self.return_horizon
        pred_r = preds.get(f"return_{hold}")
        if pred_r is None or pred_r.ndim != 2:
            return None, {}
        eng = BacktestEngine(transaction_cost_bps=self.cost_bps, method=self.portfolio_method)
        decision_idx = tb.get("__rows")
        # decision rows recovered from the batch targets length
        n = pred_r.shape[0]
        # we lost original rows; rebuild from the test-fold slice
        rows = self.fold.test[self.fold.test >= self.window_len - 1]
        rows = rows[:n]
        pr = eng.run(self.one_bar_returns, pred_r, rows.astype(np.int64), hold=hold)
        eq = pr.equity
        mets = {
            "tot_ret": float(eq[-1] - 1.0),
            "max_dd": M.max_drawdown(np.concatenate([[1.0], eq])),
            "realized_vol": float(np.std(pr.returns)),
        }
        return pr, mets
