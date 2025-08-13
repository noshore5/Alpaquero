"""Financial continuous wavelet transform (Morlet), computed via FFT.

This is the direct analog of ``utils/torch_cwt.py`` from the EEG_Benchmarks
repo, adapted to financial time series. The EEG code uses a log-spaced
frequency grid in Hz (8-124 Hz) with a Morlet bandwidth ``fb = 2.0`` where a
wavelet of period ``P_sec = 1/f`` has a Gaussian envelope of std
``sigma_sec = fb/f`` seconds. Here we express everything in *bars* rather
than Hz, because financial data has no fixed sampling rate -- the "time" axis
is a sequence of bars at a fixed bar interval.

Mapping (documented decision):
    - ``sampling_rate = 1`` bar.
    - A financial "period" ``P`` (in bars, e.g. 12 bars for 1h of 5-min
      bars) maps to a CWT frequency ``f = 1/P`` cycles/bar.
    - The Morlet Gaussian envelope has std ``sigma_bars = fb * P`` bars
      (identical to the EEG relation ``sigma_sec = fb/f`` after converting
      seconds to bars at 1 bar/unit).
    - Boundary padding is sized to the widest wavelet (the longest period):
      ``pad = round(fb * P_max * 3)`` bars, matching ``_boundary_pad``.

Because it is a wavelet *transform*, the output is per-timestamp (not just a
set of average power bands), and can be causality-bounded by trimming the
cone-of-influence at each timestamp (see ``coi_valid`` and the leakage notes).

Configuration of periods is in wall-clock terms (e.g. "1h"), converted to
bars at the bar interval:  ``bars = seconds(period) / seconds(bar)``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

MORLET_FB = 2.0
MORLET_AMPLITUDE_SCALE = math.sqrt(2.0) * math.pi ** 0.25

_PERIOD_SECONDS = {
    "5min": 5 * 60,
    "15min": 15 * 60,
    "30min": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "8h": 8 * 60 * 60,
    "1d": 24 * 60 * 60,
    "2d": 2 * 24 * 60 * 60,
}

_BAR_SECONDS = {"1Min": 60, "5Min": 300, "15Min": 900, "1Hour": 3600, "1Day": 86400}


def period_to_bars(period: str, timeframe: str) -> float:
    """Convert a wall-clock period label to a number of bars."""
    if period not in _PERIOD_SECONDS:
        raise ValueError(f"unknown period {period!r}")
    sec = _PERIOD_SECONDS[period]
    bar_sec = _BAR_SECONDS[timeframe]
    bars = sec / bar_sec
    if bars < 1.0:
        raise ValueError(
            f"period {period} ({sec}s) < one {timeframe} bar ({bar_sec}s); "
            "wavelets cannot resolve periods shorter than the bar interval"
        )
    return bars


def _next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def _log_spaced(lo: float, hi: float, n: int) -> np.ndarray:
    """n log-spaced values in [lo, hi], expressed as periods (bars)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.array([hi])
    ratio = lo / hi
    exps = np.linspace(0.0, 1.0, n)
    return hi * ratio ** exps


def financial_periods(period_labels: list[str], timeframe: str) -> np.ndarray:
    """Wall-clock period labels -> sorted log-spaced period grid in bars."""
    return np.array(sorted(period_to_bars(p, timeframe) for p in period_labels))


def trailing_support_bars(periods_bars: np.ndarray, fb: float = MORLET_FB, coi_factor: float = 3.0) -> np.ndarray:
    """Per-frequency trailing support (bars) of the Morlet wavelet.

    ``L_f = ceil(coi_factor * fb * P_f)`` with default ``coi_factor = 3.0``
    (three e-folding half-widths, the conventional support edge). The wavelet
    coefficient centered at t samples the signal over
    ``[t - L_f, t + L_f]``, so ``L_f`` is both the *trailing* (future) reach
    that must be zeroed for a causal coefficient and its left lookback.
    """
    return np.ceil(coi_factor * fb * np.asarray(periods_bars, dtype=float)).astype(int)


def _boundary_pad_bars(p_max: float, coi_factor: float, fb: float = MORLET_FB) -> int:
    """Left zero-pad half-width in bars, sized to the widest wavelet's support."""
    return int(trailing_support_bars(np.array([p_max]), fb=fb, coi_factor=coi_factor).max())


class MorletCWTBank:
    """Precomputed frequency-domain Morlet filter bank for a period grid.

    Parameters
    ----------
    coi_factor : multiplier on the trailing-support ``L_f = coi_factor*fb*P_f``
        used for the cone-of-influence / causality padding and trimming. Default
        3.0 (three e-folding half-widths).
    """

    def __init__(self, periods_bars: np.ndarray, n_time: int, device=None, dtype=torch.complex64,
                 coi_factor: float = 3.0):
        self.periods = np.asarray(periods_bars, dtype=float)  # [F]
        self.n_time = int(n_time)
        self.device = device if device is not None else torch.device("cpu")
        self.coi_factor = float(coi_factor)
        # period (bars) -> CWT frequency f = 1/period (cycles/bar)
        freqs = 1.0 / self.periods  # [F], ascending period -> descending freq
        self.freqs = freqs
        self.trailing = trailing_support_bars(self.periods, MORLET_FB, self.coi_factor)  # [F]
        self.l_max = int(self.trailing.max())
        self.pad = _boundary_pad_bars(self.periods.max(), self.coi_factor)
        self.n_padded = _next_pow2(n_time + self.pad + self.l_max)
        # Build filter bank in float64 on CPU (MPS lacks fp64), like EEG.
        bin_freqs = torch.fft.rfftfreq(self.n_padded, d=1.0)  # cycles/bar
        sigma_bars = MORLET_FB * self.periods  # [F]
        freqs_t = torch.from_numpy(freqs).to(torch.float64)
        sb = torch.from_numpy(sigma_bars).to(torch.float64)
        delta = bin_freqs[None, :].to(torch.float64) - freqs_t[:, None]
        filters64 = MORLET_AMPLITUDE_SCALE * torch.exp(
            -2.0 * (math.pi ** 2) * (sb[:, None] ** 2) * (delta ** 2)
        )
        self.filters = filters64.to(dtype).to(self.device)
        self.freqs_t = freqs_t.to(torch.float32).to(self.device)

    def _padded_transform(self, signal: torch.Tensor) -> torch.Tensor:
        """CWT on the input zero-padded on the right by ``l_max`` (future
        region zeroed) and on the left by ``pad``. Returns coefficient array
        [N, F, n_padded] aligned so output index ``j`` reads ``x_pad[j-pad]``."""  # noqa: E501
        x = signal.to(torch.float32)
        x_padded = torch.nn.functional.pad(
            x, (self.pad, self.n_padded - self.n_time - self.pad)
        )
        spectrum = torch.fft.rfft(x_padded, n=self.n_padded, dim=-1)
        n_fft_bins = self.n_padded // 2 + 1
        product = spectrum.unsqueeze(-2) * self.filters
        full = torch.nn.functional.pad(product, (0, self.n_padded - n_fft_bins))
        return torch.fft.ifft(full, n=self.n_padded, dim=-1)

    def transform(self, signal: torch.Tensor) -> torch.Tensor:
        """(Acausal) CWT of signal [N, T] -> coeffs [N, F, T] complex64.

        Here ``self.pad`` (left) and ``l_max`` (right) zero-padding extends the
        signal; coefficient at index ``t`` samples ``x[t-...:t+...]`` and is
        therefore *acausal* (it sees up to ``coi_factor*fb*P_f`` future bars).
        This is the raw transform; use :meth:`transform_causal` for the
        leak-free (right-zero-padded / trailing-aligned) coefficients.
        """
        if signal.ndim != 2:
            raise ValueError("signal must be [N, T]")
        if signal.shape[-1] != self.n_time:
            raise ValueError(f"expected n_time={self.n_time}, got {signal.shape[-1]}")
        coeffs_padded = self._padded_transform(signal)
        coeffs = coeffs_padded[..., self.pad : self.pad + self.n_time].contiguous()
        return coeffs.to(torch.complex64), self.freqs_t

    def transform_causal(self, signal: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Causal (leak-free) CWT of signal [N, T] -> coeffs [N, F, T_out].

        The input is zero-padded on the right by ``l_max`` bars, so the future
        trailing support of every coefficient falls in the zeroed region. To
        make each *retained* coefficient depend only on signal ``<= t``, the
        coefficient assigned to real time ``tau`` is the one whose wavelet's
        trailing (right) edge touches ``tau`` -- i.e. ``W(tau - L_f, f)`` --
        which corresponds to evaluating the wavelet centered at ``tau`` on the
        signal truncated-and-zero-padded to end at ``tau``. The first
        ``l_max`` output timesteps (causal warm-up) are dropped; returns
        (coeffs [N, F, T - l_max], n_dropped = l_max).
        """
        if signal.ndim != 2 or signal.shape[-1] != self.n_time:
            raise ValueError("signal must be [N, T] with T == n_time")
        coeffs_padded = self._padded_transform(signal)
        acausal = coeffs_padded[..., self.pad : self.pad + self.n_time].contiguous()  # [N,F,T]
        N, F, T = acausal.shape
        n_drop = self.l_max
        out = torch.empty((N, F, T - n_drop), dtype=acausal.dtype, device=acausal.device)
        for f in range(F):
            lf = int(self.trailing[f])
            # out[tau - n_drop] = acausal[tau - lf]; with u = tau - n_drop:
            # out[u] = acausal[u + (n_drop - lf)]. No roll (no wraparound).
            start = n_drop - lf
            out[:, f, :] = acausal[:, f, start : start + (T - n_drop)]
        return out.to(torch.complex64), n_drop

    def coi_valid(self, n_time: int) -> torch.Tensor:
        """[T, F] bool: True where the wavelet at (t, period) stays within the
        signal bounds (within ``coi_factor`` e-folding half-widths). Used to
        causality-bound features and to track leakage-prone boundary regions."""
        centres = torch.arange(n_time, dtype=torch.float64)
        half = torch.from_numpy(self.trailing).to(torch.float64)      # [F] bars
        lo = centres[:, None] - half[None, :]
        hi = centres[:, None] + half[None, :]
        return (lo >= 0) & (hi <= n_time)



def cwt_torch(
    signal: np.ndarray | torch.Tensor,
    periods_bars: np.ndarray,
    *,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot batched CWT. ``signal``: [N, T] (N assets). Returns
    (coeffs [N, F, T] complex, freqs [F])."""
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))
    bank = MorletCWTBank(periods_bars, signal.shape[-1], device=device)
    return bank.transform(signal)
