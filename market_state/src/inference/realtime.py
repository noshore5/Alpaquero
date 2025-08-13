"""Realtime / streaming market-state inference.

Given a live (or simulated-live) stream of aligned bars, this module keeps the
causal feature buffer for the most recent ``window + warm-up`` bars and, on
each new bar, produces:

  * the newest causal feature vector (from ``FeaturePipeline.transform_causal``
    on the trailing input signal), and
  * the model's prediction head outputs for the current market state.

Every computation is causal: the feature for real time ``tau`` uses only bars
``<= tau`` (CWT right-zero-padded past ``tau``, causal coherence smoothing).
No future information enters the state at any point.
"""
from __future__ import annotations

import numpy as np

from transforms.pipeline import FeaturePipeline
from models.market_state import MarketStateModel
import torch


class MarketStateInference:
    """Streaming inference over a causal feature pipeline + Mamba market model.

    Parameters
    ----------
    model : trained/randomly-initialised MarketStateModel.
    pipeline : FeaturePipeline (must be built with the same universe/config).
    window_len : model lookback (W), bars.
    """

    def __init__(self, model: MarketStateModel, pipeline: FeaturePipeline, window_len: int) -> None:
        self.model = model
        self.pipeline = pipeline
        self.window_len = int(window_len)
        self.device = pipeline.cfg.device
        # The pipeline recomputes on its fixed-length signal buffer each step.
        # We keep exactly ``bank.n_time`` bars so ``compute`` receives the full,
        # same-length signal it was constructed for (causal construction intact).
        self.n_time = int(pipeline.bank.n_time)
        self._lr_history: list[np.ndarray] = []   # list of [A] log-return vectors

    @property
    def warmup(self) -> int:
        return self.window_len + int(self.pipeline.bank.l_max)

    def update(self, log_return: np.ndarray) -> None:
        """Ingest one aligned log-return vector [A] for the newest bar."""
        self._lr_history.append(np.asarray(log_return, dtype=np.float32))
        if len(self._lr_history) > self.n_time:
            self._lr_history.pop(0)

    def step(self) -> dict[str, np.ndarray] | None:
        """Produce the newest causal feature + model output, or None during warm-up."""
        lr = np.stack(self._lr_history, axis=1) if self._lr_history else None
        if lr is None or lr.shape[1] < self.n_time:
            return None
        # Run the (already-causal) transform on the full trailing signal.
        feats, _ = self.pipeline.compute(lr[:, -self.n_time :])
        # feats rows are causal (warm-up already dropped); the last row is the
        # feature for the newest real time, and the last `window_len` rows form
        # the causal look-back window for the model.
        latest = feats[-1]                       # [D]
        win = feats[-self.window_len :]          # [W, D]
        x = torch.as_tensor(win[None, :, :], dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = {k: v.detach().cpu().numpy() for k, v in self.model(x).items()}
        return {"feature": latest, "window": win, "outputs": out}
