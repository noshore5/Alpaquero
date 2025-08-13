"""Pairwise wavelet coherence (WCT) between assets, keeping complex phase.

Adapted from the EEG_Benchmarks coherence machinery
(``wct_phase_gnn_classifier._compute_wct_window_features`` and
``hermitian_ssm_cache``'s ``compute_recording_spectral``). For assets i, j:

    cross-spectrum    X_ij(t,f) = W_i(t,f) conj(W_j(t,f))   complex
    auto-spectra      P_i(t,f)  = |W_i(t,f)|^2
    coherence         C_ij(t,f) = |<X_ij>| / sqrt(<P_i><P_j>)   in [0,1]
    relative phase    phi_ij(t,f) = atan2(Im <X_ij>, Re <X_ij>)
    complex coherence W_ij(t,f) = C_ij(t,f) * exp(i phi_ij(t,f))

The ``<>`` denotes Gaussian smoothing over the time axis (``smooth_time_steps``
bars), mirroring the EEG Hermitian cache's time-only smoothing operator (that
pipeline deliberately avoids cross-frequency smoothing so each frequency bin
remains an independent graph -- see the EEG module docstring).

Because coherence is a Hermitian quantity (C_ij = C_ji, phi_ij = -phi_ji), we
only ever compute the ``N(N-1)/2`` unique unordered pairs (upper triangle) and
leave the conjugate symmetry to the Hermitian construction, exploiting
symmetry rather than computing redundant pairs.
"""
from __future__ import annotations

import numpy as np
import torch


def _gaussian_kernel1d(width_steps: int, device, dtype=torch.float32, causal: bool = True) -> torch.Tensor:
    """(Trailing) Gaussian kernel over lags 0..W-1.

    ``causal=True`` returns a one-sided kernel whose mass is on the *present*
    and *past* lags only (lag 0 = present), used for leak-free time smoothing.
    ``causal=False`` returns the symmetric (acausal) kernel.
    """
    w = max(1, int(width_steps))
    if causal:
        # one-sided over lags {0..w-1}; peak at lag 0, decaying into the past.
        sigma = max((w - 1) / 2.0, 1e-3)
        k = torch.exp(-0.5 * (torch.arange(w, device=device, dtype=dtype) / sigma) ** 2)
    else:
        if w % 2 == 0:
            w += 1
        if w == 1:
            return torch.ones(1, device=device, dtype=dtype)
        sigma = (w - 1) / 2.0
        x = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2.0
        k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_time_causal(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Causal (past-only) weighted running mean along the last axis.

    ``out[tau] = sum_k kernel[k] * x[tau - k]`` with ``x`` treated as zero
    before time 0 (leading outputs are causal warm-up). Only present/past
    samples contribute, so a perturbation of future samples never changes
    ``out[tau]`` -- required for the no-leakage contract.
    """
    if kernel.numel() == 1:
        return x
    lead = x.shape[:-1]
    t = x.shape[-1]
    w = kernel.numel()
    xf = x.reshape(-1, 1, t)
    # left-pad by w-1 zeros, then a 'valid' conv with the reversed kernel yields
    # out[tau] = sum_k kernel[k] * x[tau - k].
    xp = torch.nn.functional.pad(xf, (w - 1, 0), mode="constant", value=0.0)
    kernel_rev = torch.flip(kernel, dims=[0]).view(1, 1, -1)
    out = torch.nn.functional.conv1d(xp, kernel_rev)
    if out.shape[-1] != t:
        out = out[..., :t]
    return out.reshape(*lead, t)


def _smooth_time(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Acausal (symmetric) smoothing, kept for ``causal=False`` only."""
    if kernel.numel() == 1:
        return x
    lead = x.shape[:-1]
    t = x.shape[-1]
    xf = x.reshape(-1, 1, t)
    pad = kernel.numel() // 2
    xf = torch.nn.functional.pad(xf, (pad, pad), mode="reflect")
    out = torch.nn.functional.conv1d(xf, kernel.view(1, 1, -1))
    return out.reshape(*lead, t)


class WaveletCoherence:
    """Computes complex pairwise wavelet coherence for a set of CWT coeffs.

    Parameters
    ----------
    n_assets : number of assets (nodes).
    smooth_time_steps : std of the time-smoothing Gaussian, in bars.
    device / dtype : tensor placement.
    """

    def __init__(
        self,
        n_assets: int,
        *,
        smooth_time_steps: int = 5,
        causal: bool = True,
        device=None,
        dtype=torch.complex64,
    ) -> None:
        self.n_assets = int(n_assets)
        self.smooth_time_steps = smooth_time_steps
        self.causal = causal
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.kernel = _gaussian_kernel1d(smooth_time_steps, self.device, causal=causal)
        # Unique unordered pairs (upper triangle): (C, 2) -> ([P], [P]).
        iu, ju = torch.triu_indices(n_assets, n_assets, offset=1, device=self.device)
        self.iu = iu
        self.ju = ju
        self.n_pairs = int(iu.numel())

    def forward(self, coeffs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute complex coherence for all unique pairs.

        Parameters
        ----------
        coeffs : [N, F, T] complex64 CWT coefficients (N assets, F freqs).
        Returns dict with:
            complex_coherence : [F_out?/F, P, T_out] complex W_ij
            coherence         : magnitude [F, P, T]
            phase             : [F, P, T] phase (radians)
            n_pairs           : int
            iu, ju            : pair indices
        """
        n_assets, n_freqs, n_time = coeffs.shape
        if n_assets != self.n_assets:
            raise ValueError(f"expected {self.n_assets} assets, got {n_assets}")

        w_i = coeffs[self.iu]      # [P, F, T] complex
        w_j = coeffs[self.ju]
        xwt = w_i * torch.conj(w_j)                      # [P, F, T] complex

        smoother = _smooth_time_causal if self.causal else _smooth_time
        xr = smoother(xwt.real.to(torch.float32), self.kernel)
        xi = smoother(xwt.imag.to(torch.float32), self.kernel)
        auto = (coeffs.real ** 2 + coeffs.imag ** 2).to(torch.float32)   # [N, F, T]
        p1 = smoother(auto[self.iu], self.kernel)
        p2 = smoother(auto[self.ju], self.kernel)

        denom = torch.sqrt(p1 * p2 + 1e-12)
        mag = torch.sqrt(xr ** 2 + xi ** 2 + 1e-20) / denom
        mag = mag.clamp(0.0, 1.0)
        phase = torch.atan2(xi, xr)
        complex_coh = mag.to(self.dtype) * torch.exp(1j * phase.to(self.dtype))

        return {
            "complex_coherence": complex_coh,   # [P, F, T]
            "coherence": mag,                    # [P, F, T]
            "phase": phase,                      # [P, F, T]
            "n_pairs": self.n_pairs,
            "iu": self.iu,
            "ju": self.ju,
        }

    def __call__(self, coeffs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward(coeffs)
