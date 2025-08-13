"""Composite market-state model: causal spectral features -> Mamba-3 -> heads.

The deterministic transform pipeline (transforms/*) turns the aligned
multi-asset price history into, per timestep, a fixed vector of *stable
spectral quantities* (sorted eigenvalues, normalized spectrum, spectral
entropy / concentration, node loadings from the gauge-invariant spectral
projector). This module consumes that ``[B, T, D_spec]`` sequence and:

  * projects to the latent state dim,
  * runs a complex-diagonal Mamba-3 (Mamba3Sequence) over time, taking the
    final timestep as the market state ``z``,
  * distributes ``z`` to the per-horizon prediction heads.

Separation of concerns: the model never touches raw prices or the Hermitian
matrix; all causal/leakage properties are enforced upstream (features at ``t``
use only bars ``<= t``, targets use ``t+1..t+H``). This keeps the learnable
component small and the benchmark forward-pass straightforward.

The ``run_heads`` entry point maps a HeadSpec list onto the heads; the
``build_from_config`` factory wires counts from the target configuration.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .mamba import Mamba3Sequence
from .heads import HeadSpec, OutputHeads


def _default_heads(model_cfg) -> list[tuple[HeadSpec, int]]:
    """Construct the head-spec list + per-head count from the model config.

    Count is ``n_assets`` for per-asset neat regression/classification targets;
    for correlation targets it is ``n_assets`` (the network internally builds
    an N x N matrix).
    """
    tc = model_cfg.get("targets", {})

    def hs(name, kind, horizon, n_targets=None, per_asset=True) -> HeadSpec:
        return HeadSpec(kind=kind, name=name, horizon=int(horizon),
                        n_targets=n_targets, per_asset=per_asset)

    n_assets = int(model_cfg.get("n_assets", 1))
    specs: list[tuple[HeadSpec, int]] = []
    for h in tc.get("realized_vol_horizons", []):
        specs.append((hs(f"realized_vol_{h}", "regression", h), n_assets))
    for h in tc.get("return_horizons", []):
        specs.append((hs(f"return_{h}", "regression", h), n_assets))
    mdd_h = tc.get("max_drawdown_horizon")
    if mdd_h:
        specs.append((hs("max_drawdown", "regression", mdd_h), n_assets))
    corr_h = tc.get("correlation_horizon")
    if corr_h:
        specs.append((hs("correlation", "correlation", corr_h), n_assets))
    thresholds = tc.get("regime_thresholds", [])
    if thresholds:
        specs.append((hs("regime", "classification", 5, n_targets=len(thresholds) + 1), n_assets))

    return specs


class MarketStateModel(nn.Module):
    def __init__(
        self,
        d_spec: int,
        n_assets: int,
        *,
        state_dim: int = 64,
        mamba_layers: int = 1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        d_model: int | None = None,
        head_specs: list[tuple[HeadSpec, int]] | None = None,
    ) -> None:
        super().__init__()
        self.d_spec = int(d_spec)
        self.n_assets = int(n_assets)
        d_model = int(d_model or state_dim)
        self.state_dim = d_model
        self.mamba = Mamba3Sequence(
            in_channels=self.d_spec,
            d_model=d_model,
            d_state=int(mamba_d_state),
            d_conv=int(mamba_d_conv),
            expand=int(mamba_expand),
            n_layers=int(mamba_layers),
            dropout=float(mamba_dropout),
        )
        specs = head_specs if head_specs is not None else []
        self.heads = OutputHeads(d_model, specs)
        self.head_specs_by_name = self.heads.specs_by_name

    def forward(self, spec_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """spec_seq : [B, T, D_spec] spectral feature sequence (causal).

        Returns dict of named head outputs (see heads.py)."""
        z = self.mamba(spec_seq)          # [B, state_dim]
        return self.heads(z)


def build_from_config(model_cfg: dict, d_spec: int, *, targets_cfg: dict | None = None, **overrides) -> MarketStateModel:
    """Factory building MarketStateModel from a model-config dict + d_spec.

    Head specs are derived from ``targets_cfg`` (the project's top-level
    ``config['targets']``); if not given, falls back to ``model_cfg['targets']``.
    """
    n_assets = int(model_cfg.get("n_assets", 1))
    if targets_cfg is None:
        targets_cfg = model_cfg.get("targets", {})
    eff_cfg = dict(model_cfg)
    eff_cfg["targets"] = targets_cfg
    kw = {
        "d_spec": d_spec,
        "n_assets": n_assets,
        "state_dim": int(model_cfg.get("state_dim", 64)),
        "mamba_layers": int(model_cfg.get("mamba_layers", 1)),
        "mamba_d_state": int(model_cfg.get("mamba_d_state", 16)),
        "mamba_d_conv": int(model_cfg.get("mamba_d_conv", 4)),
        "mamba_expand": int(model_cfg.get("mamba_expand", 2)),
        "mamba_dropout": float(model_cfg.get("mamba_dropout", 0.0)),
        "head_specs": _default_heads(eff_cfg),
    }
    kw.update(overrides)
    return MarketStateModel(**kw)
