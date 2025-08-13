"""Prediction heads for the market-state model.

After the Mamba-3 temporal encoder compresses each timestep into a fixed
latent vector ``z_t`` (shared across the cross-asset shelf), a set of heads
produces forecasts for the target horizons. Design decisions:

* **Causal targets**: each head predicts *future* quantities from information
  available at ``t`` only. The longitudinal series is ingested as the CWT /
  Hermitian class while the targets are computed strictly using bars at
  ``t+1 .. t+H`` (see datasets.targets). The encoder therefore does not need
  the targets at all -- it consumes the causal spectral sequence.
* Shared backbone, per-target heads. Each horizon uses its own head so the
  model can express horizon-specific dynamics.
* Heads are typed by kind:
    - "regression"     : continuous scalar(s) per asset (realized vol,
                         expected return, max drawdown).
    - "correlation"    : pairwise symmetric matrix prediction (real).
    - "classification" : regime class per asset.

All heads are wrapped by ``OutputHeads`` which turns the encoder latent
``[B, D]`` into an ``ModelOutput`` named-ish dict tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadSpec:
    kind: str          # "regression" | "correlation" | "classification"
    name: str          # e.g. "realized_vol", "return", "mdd", "correlation", "regime"
    horizon: int       # prediction horizon in bars
    n_targets: int | None = None   # number of regression targets / classes
    per_asset: bool = True         # True -> output per asset; correlation is NxN


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CorrelationHead(nn.Module):
    """Predict an N x N real matrix, symmetrised + hard/soft tanh clip to [-1,1].

    For a shelf of N assets we predict a full matrix from the shared latent
    then symmetrise ``(M + M^T)/2``. The diagonal is set to 1 after clipping.
    """

    def __init__(self, d_model: int, n_assets: int) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.net = nn.Linear(d_model, n_assets * n_assets)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        M = torch.tanh(self.net(z)).view(B, self.n_assets, self.n_assets)
        M = 0.5 * (M + M.transpose(-1, -2))
        return M


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, n_assets: int) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.net = nn.Linear(d_model, n_classes * n_assets)

    def forward(self, z: torch.Tensor, per_asset: bool = True) -> torch.Tensor:
        B = z.shape[0]
        logits = self.net(z)
        if per_asset:
            return logits.view(B, self.n_assets, -1)  # [B, A, n_classes]
        return logits.view(B, -1)


class OutputHeads(nn.Module):
    """Container of heads: builds each from a list of HeadSpec + counts."""

    def __init__(self, d_model: int, specs_with_counts: list[tuple[HeadSpec, int]]) -> None:
        super().__init__()
        self.d_model = d_model
        self.specs_by_name: dict[str, HeadSpec] = {}
        self.regression = nn.ModuleDict()
        self.correlation = nn.ModuleDict()
        self.classification = nn.ModuleDict()
        for spec, count in specs_with_counts:
            self.specs_by_name[spec.name] = spec
            if spec.kind == "regression":
                self.regression[spec.name] = RegressionHead(d_model, count)
            elif spec.kind == "correlation":
                self.correlation[spec.name] = CorrelationHead(d_model, count)
            elif spec.kind == "classification":
                self.classification[spec.name] = ClassificationHead(d_model, spec.n_targets, count)
            else:
                raise ValueError(spec.kind)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for name, h in self.regression.items():
            out[name] = h(z)
        for name, h in self.correlation.items():
            out[name] = h(z)
        for name, h in self.classification.items():
            out[name] = h(z)
        return out
