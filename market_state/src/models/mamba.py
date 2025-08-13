"""Complex-diagonal selective state-space model ("Mamba-3").

Adapted from ``EEG_Benchmarks/Epilepsy/pipelines/mamba3.py``. The per-channel
state eigenvalue is complex:

    lambda = -exp(a) + i * omega        a, omega learned per (d_inner, d_state)

so each state channel has an exponential decay rate ``exp(-a)`` AND a
rotation frequency ``omega``. This is well-suited to a stream of complex
cross-spectral coherence / Hermitian-spectral features, whose phase evolves
over time (a real-diagonal SSM can only accumulate or decay, not rotate).

Selective (Mamba) machinery is kept: ``Delta``, ``B`` and ``C`` are
input-dependent; ``C`` is complex so the readout can resolve phase.

This module is self-contained: it vendors its own complex parallel scan
(``_ComplexPScan``) with a conjugation-correct backward so it does not depend
on a specific external pscan implementation, and falls back to a naive
sequential scan if ``mambapy`` is unavailable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional speed-up; falls back to a sequential scan
    from mambapy.pscan import PScan as _RealPScan, npo2 as _npo2, pad_npo2 as _pad_npo2
    _HAS_MAMBAPY = True
except Exception:  # pragma: no cover
    _HAS_MAMBAPY = False


def _next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


class _ComplexPScan(torch.autograd.Function):
    """Sequential (or Blelloch, when mambapy present) complex linear scan
    ``H_t = A_t * H_{t-1} + X_t`` elementwise, complex, with a
    conjugation-correct backward.

    torch's convention: for ``y = a*x``, ``grad_x = conj(a) grad_y``,
    ``grad_a = conj(x) grad_y``. The reverse recurrence is therefore
    ``G_t = grad_H[t] + conj(A_{t+1}) G_{t+1}``.
    """

    @staticmethod
    def forward(ctx, A_in: torch.Tensor, X_in: torch.Tensor) -> torch.Tensor:
        L = X_in.size(1)
        if _HAS_MAMBAPY:
            if L == _npo2(L):
                A, X = A_in.clone(), X_in.clone()
            else:
                A, X = _pad_npo2(A_in), _pad_npo2(X_in)
            A = A.transpose(2, 1).contiguous()
            X = X.transpose(2, 1).contiguous()
            _RealPScan.pscan(A, X)               # in-place; H now in X
            H = X.transpose(2, 1)[:, :L].contiguous()
        else:
            H = _sequential_fwd(A_in, X_in)
        ctx.save_for_backward(A_in, H)
        return H

    @staticmethod
    def backward(ctx, grad_in: torch.Tensor):
        A_in, H = ctx.saved_tensors
        if _HAS_MAMBAPY:
            L = grad_in.size(1)
            if L == _npo2(L):
                grad = grad_in.clone()
            else:
                grad = _pad_npo2(grad_in)
                A_in = _pad_npo2(A_in)
            grad = grad.transpose(2, 1).contiguous()
            A_t = A_in.transpose(2, 1)
            A_shift = F.pad(A_t[:, :, 1:], (0, 0, 0, 1)).conj().contiguous()
            _RealPScan.pscan_rev(A_shift, grad)
            G = grad.transpose(2, 1)[:, :L].contiguous()
            gradA = torch.zeros_like(H)
            gradA[:, :, 1:] = H[:, :, :-1].conj() * G[:, :, 1:]
            return gradA[:, :, :L], G
        # sequential backwards
        return _sequential_bwd(A_in, H, grad_in)

    def vmap(self, *args, **kwargs):  # pragma: no cover - not needed
        return self


def _sequential_fwd(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """[B, L, D, N] -> [B, L, D, N] H_t = A_t*H_{t-1} + X_t, H_0 = 0."""
    B, L, D, N = X.shape
    H = torch.zeros((B, L, D, N), dtype=X.dtype, device=X.device)
    prev = torch.zeros((B, D, N), dtype=X.dtype, device=X.device)
    for t in range(L):
        prev = A[:, t] * prev + X[:, t]
        H[:, t] = prev
    return H


def _sequential_bwd(A: torch.Tensor, H: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, L, D, N = H.shape
    gradA = torch.zeros_like(A)
    G = torch.zeros_like(H)
    carry = torch.zeros((B, D, N), dtype=H.dtype, device=H.device)
    for t in range(L - 1, -1, -1):
        carry = grad[:, t].to(carry.dtype) + A[:, t + 1].conj() * carry if t < L - 1 else grad[:, t]
        G[:, t] = carry
        gradA[:, t] = (H[:, t - 1].conj() * carry) if t > 0 else torch.zeros_like(carry)
    return gradA, G


class Mamba3Block(nn.Module):
    """One complex-diagonal selective-SSM layer over ``[rows, T, d_model]``."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = int(d_conv)
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=self.d_conv,
            groups=self.d_inner, padding=self.d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        a_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(a_init.clone())
        self.omega = nn.Parameter(0.01 * torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3)
        ).clamp_min(1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _scan(self, delta, Bm, Cm, x):
        """delta [R,T,di], Bm [R,T,ds] real, Cm [R,T,ds] complex, x [R,T,di].
        Returns y [R,T,di] real."""
        cdt = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        lam = (-torch.exp(self.A_log) + 1j * self.omega).to(cdt)          # [di,ds]
        deltaA = torch.exp(delta.unsqueeze(-1).to(cdt) * lam)             # [R,T,di,ds]
        u = (delta.unsqueeze(-1) * x.unsqueeze(-1) * Bm.unsqueeze(2)).to(cdt)
        h = _ComplexPScan.apply(deltaA, u)                                # [R,T,di,ds]
        return (h * Cm.unsqueeze(2)).sum(-1).real + self.D * x

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        r, t, _ = seq.shape
        xz = self.in_proj(seq)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :t].transpose(1, 2)
        x = F.silu(x)

        proj = self.x_proj(x)
        dt_lr, Br, Cri = torch.split(proj, [self.dt_rank, self.d_state, 2 * self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_lr))
        Cm = torch.complex(Cri[..., : self.d_state], Cri[..., self.d_state:])

        y = self._scan(delta, Br, Cm, x)
        y = y * F.silu(z)
        return self.out_proj(y)


class Mamba3Sequence(nn.Module):
    """Plain complex-diagonal Mamba-3 over a [B, T, D] sequence.

    Returns the latent at the final timestep [B, out_channels], suitable for
    a shared market-state latent that the heads consume.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()
        self.layers = nn.ModuleList(
            [Mamba3Block(d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq [B, T, in]
        seq = self.in_proj(seq)
        for layer, norm in zip(self.layers, self.norms):
            seq = seq + layer(norm(seq))
        return self.dropout(seq[:, -1, :])   # [B, d_model]


class Mamba3Temporal(nn.Module):
    """``[B, C_in, E, T] -> [B, out_channels, E, 1]``.

    Weight-shared across the ``E`` axis (folded into the row dim), complex
    SSM inside, last-timestep pooling. ``E`` may index edges, nodes,
    frequencies or be 1 -- the block does not care. Mirrors the EEG
    ``_Mamba3Temporal`` contract.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_proj = nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()
        self.layers = nn.ModuleList(
            [Mamba3Block(d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, out_channels)

    def forward(self, conv_in: torch.Tensor) -> torch.Tensor:
        b, c_in, e, t = conv_in.shape
        if c_in != self.in_channels:
            raise ValueError(
                f"built with in_channels={self.in_channels}, got {c_in}"
            )
        seq = conv_in.permute(0, 2, 3, 1).reshape(b * e, t, c_in)   # [B*E, T, C_in]
        seq = self.in_proj(seq)
        for layer, norm in zip(self.layers, self.norms):
            seq = seq + layer(norm(seq))
        pooled = self.dropout(seq[:, -1, :])
        out = self.out_proj(pooled).reshape(b, e, -1)
        return out.permute(0, 2, 1).unsqueeze(-1)                    # [B, out, E, 1]
