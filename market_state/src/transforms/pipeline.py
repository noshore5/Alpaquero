"""Feature pipeline: aligned prices -> per-timestep spectral feature vectors.

This orchestrates the deterministic transform stack

    aligned log-returns [A, T]  --(causal CWT)-->  coeffs [A, F, T_valid]
                    --(wavelet coherence)-->  complex coherence [P, F, T_valid]
                    --(Hermitian graph)-->  H(t) [T_valid, N, N]
                    --(spectral)-->  per-timestep gauge-invariant features

Causality (critical requirement: signals at ``t`` only, never future): the
standard FFT/Morlet CWT is *acausal* -- the coefficient W(t, P) samples the
signal over ``[t - L_f, t + L_f]`` with trailing support ``L_f = coi_factor *
fb * P``, i.e. it "sees" up to ``L_f`` future bars. Feeding W(t) as a predictor
of ``t+1..t+H`` would leak up to the longest period ahead (for a 1-day period
that is enormous).

We handle causality at the input level (see ``MorletCWTBank.transform_causal``):
the input log-return signal is zero-padded on the right by ``l_max`` bars so
the future trailing support of every coefficient falls in the zeroed region.
The causal coefficient assigned to real time ``tau`` is ``W(tau - L_f, f)``
-- the wavelet whose trailing (right) edge touches ``tau``, i.e. evaluating the
wavelet centered at ``tau`` on signal truncated-and-zero-padded to end at
``tau``. Frequencies with longer periods use larger ``L_f`` (they need more
history). The first ``l_max`` output timesteps are the causal warm-up and are
dropped. ``coi_factor`` (default 3.0) is the single knob controlling ``L_f``
and is shared by the CWT padding and any cone-of-influence trimming so the two
stay consistent.

All downstream transforms operate on the causal coefficients, so coherence, the
Hermitian graph and the spectral features at ``tau`` use only bars ``<= tau``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .cwt import MorletCWTBank
from .wavelet_coherence import WaveletCoherence
from .hermitian import HermitianGraph
from .spectral import SpectralConfig, SpectralDecomposition


@dataclass
class FeaturePipelineConfig:
    device: str = "cpu"
    dtype: str = "complex64"
    smooth_time_steps: int = 5
    frequency_reduction: str = "magnitude_weighted"
    n_components: int = 8
    eigenvalue_normalize: str = "trace"
    canonicalize_phase: bool = True
    causal: bool = True
    # Trailing-support multiplier: L_f = coi_factor * fb * P_f. Used both for the
    # cone-of-influence (coi) trimming and for the causal right-zero-padding /
    # trailing-alignment so the two stay consistent. Default 3.0 (three
    # e-folding half-widths).
    coi_factor: float = 3.0
    use_eigenvectors: bool = False


class FeaturePipeline:
    """Computes per-timestep spectral feature vectors from aligned prices."""

    def __init__(self, cfg: FeaturePipelineConfig, n_assets: int, periods_bars: np.ndarray, n_time: int) -> None:
        self.cfg = cfg
        self.n_assets = n_assets
        self.bank = MorletCWTBank(periods_bars, n_time, device=cfg.device, coi_factor=cfg.coi_factor)
        self.coherence = WaveletCoherence(n_assets, smooth_time_steps=cfg.smooth_time_steps,
                                          device=cfg.device)
        self.hermitian = HermitianGraph(n_assets, frequency_reduction=cfg.frequency_reduction,
                                        device=cfg.device)
        self.spectral = SpectralDecomposition(
            SpectralConfig(n_components=cfg.n_components,
                           eigenvalue_normalize=cfg.eigenvalue_normalize,
                           canonicalize_phase=cfg.canonicalize_phase),
            device=cfg.device,
        )

    @property
    def d_features(self) -> int:
        """Feature vector dim per timestep for this pipeline's config/universe."""
        # n_components + normalized spectrum + |spec| + 3 scalars + (N loadings if eigvecs)
        k = min(self.cfg.n_components, self.n_assets)
        base = 3 * k + 4
        if self.cfg.use_eigenvectors:
            base += self.n_assets
        return base

    def compute(self, log_returns: np.ndarray) -> tuple[np.ndarray, int]:
        """Full-series feature computation.

        Parameters
        ----------
        log_returns : [n_assets, T] float32, aligned (causal-ready) log returns.
        Returns (features [T_out, D], n_dropped) where T_out = T - causal_warmup
        and n_dropped is the number of causal-warmup bars trimmed.
        """
        if isinstance(log_returns, np.ndarray):
            lr = torch.from_numpy(np.ascontiguousarray(log_returns, dtype=np.float32)).to(self.cfg.device)
        else:
            lr = log_returns.to(self.cfg.device)
        if self.cfg.causal:
            coeffs, max_L = self.bank.transform_causal(lr)   # [A, F, T - l_max] leak-free
        else:
            coeffs, max_L = self.bank.transform(lr), 0
        out = self.coherence(coeffs)                     # complex_coh [P, F, T']
        H = self.hermitian.build(out["complex_coherence"], coeffs.shape[-1])   # [T', N, N]
        dec = self.spectral.decompose(H)
        feats = self.spectral.features(dec["eigenvalues"],
                                       dec["eigenvectors"] if self.cfg.use_eigenvectors else None)
        return feats.cpu().numpy(), int(max_L)
