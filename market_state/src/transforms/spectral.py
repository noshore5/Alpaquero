"""Spectral decomposition of the Hermitian market graph.

Adapted from EEG ``hermitian_ssm_cache.compute_recording_spectral`` which
uses ``torch.linalg.eigh`` (a Hermitian-specific solver, *not* a generic
eigensolver), sorts eigenvalues by |lambda| descending, keeps the top-k, and
phase-canonicalises each eigenvector.

Important design decision (documented): because eigenvalues are invariant but
eigenvectors have a phase/sign ambiguity and become unstable near eigenvalue
crossings, the market-state model is configured by default to consume
**stable spectral quantities**:

  * sorted eigenvalues
  * normalised eigenvalue spectrum
  * spectral entropy / concentration
  * (optionally) phase-canonicalised eigenvector-derived quantities

Raw eigenvector *columns* are not assumed to have a stable identity across
time and are therefore not fed directly into Mamba by default
(``spectral.use_eigenvectors = False`` in config).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _sanitize_complex(H: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with 0 on a complex tensor, device-agnostic.

    ``torch.nan_to_num`` is not implemented for complex on MPS, so sanitise via
    the real/imag float parts (finite guards are sufficient for a Hermitian
    matrix built from bounded coherence).
    """
    re = torch.nan_to_num(H.real, nan=0.0, posinf=0.0, neginf=0.0)
    im = torch.nan_to_num(H.imag, nan=0.0, posinf=0.0, neginf=0.0)
    return re.to(H.dtype) + 1j * im.to(H.dtype)


@dataclass
class SpectralConfig:
    n_components: int = 8
    eigenvalue_normalize: str = "trace"
    canonicalize_phase: bool = True


def canonicalize_phase(evecs: torch.Tensor) -> torch.Tensor:
    """Rotate each eigenvector so its largest-|.| component is real +ve.

    ``evecs``: [..., k, C] complex. Phase gauge is arbitrary (``u -> e^{iθ}u``);
    fixing the largest-|.| entry to real+ makes the representation deterministic
    for the same underlying eigenvector. Does not fix order ambiguity (mode
    crossing / degeneracy) -- only phase.
    """
    amax = torch.argmax(evecs.abs(), dim=-1, keepdim=True)
    anchor = torch.gather(evecs, -1, amax)
    theta = torch.angle(anchor)
    return evecs * torch.exp(-1j * theta)


class SpectralDecomposition:
    """Eigendecomposition + stable spectral features of Hermitian matrices."""

    def __init__(self, cfg: SpectralConfig, *, device=None, dtype=torch.complex64) -> None:
        self.cfg = cfg
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

    def decompose(self, H: torch.Tensor) -> dict[str, torch.Tensor]:
        """Eigendecompose H [T, N, N] Hermitian -> sorted top-k eigenpairs.

        Uses ``torch.linalg.eigh`` (LAPACK syevd Hermitian solver). Returns:
            eigenvalues : [T, k]  (sorted |lambda| desc, top-k)
            eigenvectors: [T, k, N] complex (phase-canonicalised if configured)
            order       : [T, N]  the |lambda|-descending argorder
        """
        if H.dim() != 3 or H.shape[-1] != H.shape[-2]:
            raise ValueError("H must be [T, N, N]")
        T, N, _ = H.shape
        k = min(self.cfg.n_components, N)
        H = _sanitize_complex(H.to(self.dtype))
        # torch.linalg.eigh / eig are not implemented for the MPS device; run the
        # eigendecomposition on CPU and move the (small) results back.
        src_dev = H.device
        to_cpu = src_dev.type == "mps"
        work = H.cpu() if to_cpu else H
        try:
            evals, evecs = torch.linalg.eigh(work)   # ascending real evals, [T,N], [T,N,N]
        except torch._C._LinAlgError:
            # Degenerate / singular chunks: fall back to the general solver and
            # take real parts (Hermitian input => real spectrum). Mirrors EEG guard.
            ce, cv = torch.linalg.eig(work.to(torch.complex64))
            evals = ce.real.to(torch.float32)
            evecs = cv
        order = torch.argsort(evals.abs(), dim=-1, descending=True)  # [T, N]
        order_k = order[:, :k]
        sel_vals = torch.gather(evals, -1, order_k)                    # [T, k]
        sel_vecs = torch.gather(evecs, -1, order_k.unsqueeze(-2).expand(-1, N, -1))  # [T,N,k]
        sel_vecs = sel_vecs.permute(0, 2, 1)                          # [T, k, N]
        if self.cfg.canonicalize_phase:
            sel_vecs = canonicalize_phase(sel_vecs)
        if to_cpu:
            sel_vals = sel_vals.to(src_dev)
            sel_vecs = sel_vecs.to(src_dev)
            order_k = order_k.to(src_dev)
        return {
            "eigenvalues": sel_vals,
            "eigenvectors": sel_vecs,
            "order": order_k,
        }

    def features(self, eigvals: torch.Tensor, eigvecs: torch.Tensor | None = None) -> torch.Tensor:
        """Stable spectral feature vector for each timestep.

        eigvals : [T, k] real sorted |lambda| desc.
        eigvecs : [T, k, N] complex (optional; only used if eigenvectors enabled).
        Returns features [T, D] where D is a fixed, deterministic dimension.
        """
        T, k = eigvals.shape
        k_full = eigvals.shape[1]
        lam = torch.nan_to_num(eigvals, nan=0.0)
        lam_abs = lam.abs()
        denom = lam_abs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.cfg.eigenvalue_normalize == "trace":
            norm = lam_abs / denom                    # normalized magnitude spectrum [T,k]
            signed_norm = lam / denom                 # signed (keep eigenvalue sign/weight)
        else:
            norm = lam / denom
            signed_norm = norm

        # spectral entropy (normalized spectrum treated as a distribution)
        p = norm.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=-1) / np.log(k)          # [T] in [0,1]
        # spectral concentration (fraction in top-ish modes) & max eigenvalue
        concentration = p[..., 0]                                   # share of dominant mode
        topk_share = p[:, : min(3, k)].sum(-1) if k >= 1 else torch.zeros(T, device=lam.device)
        lam_max = lam_abs[:, 0]

        feats = [lam, signed_norm, norm,
                 lam_max.unsqueeze(-1),
                 entropy.unsqueeze(-1),
                 concentration.unsqueeze(-1),
                 topk_share.unsqueeze(-1)]

        if eigvecs is not None and eigvecs.ndim == 3:
            # eigenvector-derived magnitudes only (gauge-invariant |u|^2 sums)
            absu2 = eigvecs.abs() ** 2                              # [T,k,N]
            # per-mode participation / 'node loadings' pooled magnitude:
            # reuse gauge-invariant quantities: diagonal of projector etc.
            proj_diag = (lam.unsqueeze(-1).to(eigvecs.dtype) * absu2).sum(dim=1).real  # [T,N] real
            feats.append(proj_diag)                                 # gauge-invariant node loading
        out = torch.cat([f.to(torch.float32) for f in feats], dim=-1)
        return out

    def __call__(self, H: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decompose(H)
