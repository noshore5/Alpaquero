"""Complex Hermitian market-graph construction.

Builds ``H(t) ∈ C^(N x N)`` from the complex wavelet coherence such that

    H_ij(t) = A_ij(t) exp(i phi_ij(t))      (i != j)
    H_ji(t) = conjugate(H_ij(t))
    H_ii(t) = real  (self node value)

The WCT frequency axis ``f`` is reduced to a single edge weight before the
graph is assembled. The EEG ``hermitian_ssm_cache`` instead keeps the full
frequency axis (graph is ``[F, T, C, C]``) and lets a learned encoder fuse
frequencies; here, because the first implementation uses a *deterministic,
configurable* reduction (no learned frequency aggregation yet), we reduce the
frequency dimension into one complex edge weight per pair.

Reduction choices (``frequency_reduction``):
  - "magnitude_weighted" (default): phase-coherent complex average weighted
    by coherence magnitude, i.e. ``H_ij = sum_f w_f * W_ij(f) / sum_f w_f``
    with ``w_f`` the per-frequency coherence magnitude (or an explicit
    ``frequency_weights``). This preserves phase and down-weights noisy bins.
  - "mean": unweighted complex mean over frequencies.
  - "sqrt_weighted": weight by sqrt(magnitude).

The construction then enforces exact Hermiticity by taking the conjugate
symmetry (we only ever compute the upper triangle). A validation helper
checks ``H ≈ H.conj().T`` within tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HermitianConfig:
    frequency_reduction: str = "magnitude_weighted"
    frequency_weights: list[float] | None = None
    diagonal: str = "one"  # "one" | "zero" | "power"


def reduce_frequency(
    complex_coh: torch.Tensor,
    method: str,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce the frequency axis of complex coherence to one value per pair.

    Parameters
    ----------
    complex_coh : [P, F, T] complex (pairs x frequency x time).
    method       : reduction method.
    weights      : optional [F] or [P, F, T] explicit weights.
    Returns [P, T] complex edge weight per pair.
    """
    if method == "mean":
        return complex_coh.mean(dim=1)
    if weights is None:
        w = torch.abs(complex_coh)           # [P, F, T]
    else:
        w = weights
        if w.ndim == 1:
            w = w.unsqueeze(0).unsqueeze(-1)
    if method in {"magnitude_weighted", "sqrt_weighted"}:
        eff_w = torch.sqrt(w) if method == "sqrt_weighted" else w
        num = (eff_w * complex_coh).sum(dim=1)
        den = eff_w.sum(dim=1).clamp_min(1e-12)
        return num / den
    raise ValueError(f"unsupported frequency_reduction {method!r}")


class HermitianGraph:
    """Builds the complex Hermitian market graph H(t) from complex coherence."""

    def __init__(
        self,
        n_assets: int,
        *,
        frequency_reduction: str = "magnitude_weighted",
        frequency_weights: list[float] | None = None,
        diagonal: str = "one",
        device=None,
        dtype=torch.complex64,
    ) -> None:
        self.n_assets = int(n_assets)
        self.cfg = HermitianConfig(frequency_reduction, frequency_weights, diagonal)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        iu, ju = torch.triu_indices(n_assets, n_assets, offset=1, device=self.device)
        self.iu, self.ju = iu, ju

    def build(self, complex_coh: torch.Tensor, n_time: int) -> torch.Tensor:
        """Assemble H(t) for every timestep.

        Parameters
        ----------
        complex_coh : [P, F, T] complex coherence between unique pairs.
        n_time      : number of timesteps T.
        Returns H : [T, N, N] complex64 Hermitian.
        """
        n_pair = self.iu.numel()
        if complex_coh.shape[0] != n_pair:
            raise ValueError(
                f"expected {n_pair} pairs, got {complex_coh.shape[0]}"
            )
        P, F, T = complex_coh.shape
        if n_time != T:
            n_time = T

        edge = reduce_frequency(complex_coh, self.cfg.frequency_reduction,
                                _make_weights(self.cfg, complex_coh))  # [P, T]
        H = torch.zeros((n_time, self.n_assets, self.n_assets), dtype=self.dtype,
                        device=self.device)
        # edge -> [T, P] set upper and lower triangles
        edge = edge.permute(1, 0)                      # [T, P]
        H[:, self.iu, self.ju] = edge
        H[:, self.ju, self.iu] = edge.conj()            # conjugate-symmetric

        if self.cfg.diagonal == "one":
            diag = torch.ones(n_time, self.n_assets, dtype=self.dtype, device=self.device)
        elif self.cfg.diagonal == "zero":
            diag = torch.zeros(n_time, self.n_assets, dtype=self.dtype, device=self.device)
        else:  # "power" -> needs per-node power, not available from pair coh; fallback one
            diag = torch.ones(n_time, self.n_assets, dtype=self.dtype, device=self.device)
        eye = torch.arange(self.n_assets, device=self.device)
        H[:, eye, eye] = diag
        return H

    def build_masked(self, complex_coh: torch.Tensor, valid: torch.Tensor, n_time: int) -> torch.Tensor:
        """Build H with NaNs at timesteps flagged invalid (multi-chunk API)."""
        H = self.build(complex_coh, n_time)
        if valid is not None:
            inv = ~valid if valid.dtype == torch.bool else (valid == 0)
            H[inv] = torch.nan
        return H

    @staticmethod
    def herm_error(H: torch.Tensor) -> torch.Tensor:
        """Max abs deviation from Hermitian symmetry, per matrix in batch."""
        diff = (H - H.conj().transpose(-1, -2)).abs()
        return diff.amax(dim=(-1, -2))


def _make_weights(cfg: HermitianConfig, complex_coh: torch.Tensor) -> torch.Tensor | None:
    if cfg.frequency_weights is None:
        return None
    w = torch.tensor(cfg.frequency_weights, dtype=complex_coh.dtype, device=complex_coh.device)
    return w
