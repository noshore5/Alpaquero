"""Windowing: contiguous-causal windows over the per-timestep feature array.

The feature pipeline produces one feature vector per aligned timestep
(already causal: bars <= t only). A model input window for decision time ``t``
is the last ``W`` feature rows ending at ``t`` inclusive: rows ``[t-W+1, t]``.
The targets for that window are the target rows at ``t`` (strictly future,
computed from t+1..t+H).

Because feature rows are themselves causal, the window is a
contiguous-causal block; there is no future information in any window slot.
The last ``W-1`` rows of the feature series have no full window and are
unavailable as decision times.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WindowedDataset:
    features: np.ndarray        # [B, W, D] model windows
    targets: dict[str, np.ndarray]  # name -> [B, ...] target rows
    sample_idx: np.ndarray      # [B] integer rows in the *feature* array
    timestamps: np.ndarray      # [B] original timestamps (object array) or None
    window_len: int
    feature_dim: int


def split_contiguous(
    feat_series: np.ndarray,
    horizon_starts: np.ndarray,
    window_len: int,
    targets_dict: dict[str, np.ndarray],
    timestamps=None,
) -> WindowedDataset:
    """Build windows over a single contiguous feature series.

    Parameters
    ----------
    feat_series   : [T, D] causal per-timestep feature vectors.
    horizon_starts: int start rows of any target horizon that must be valid
                    (a window at row t is only kept if t is <= last_valid_target).
    window_len    : W bars of causal lookback.
    targets_dict  : {name: [T, ...]} target arrays aligned to feat rows.
    timestamps    : optional [T] labels to carry out.
    """
    T, D = feat_series.shape
    W = int(window_len)
    if T < W:
        raise ValueError(f"need T>=W: {T}<{W}")
    max_avail = horizon_starts.min()
    last = min(T, max_avail)          # decision rows t with targets valid & window full
    n = max(0, last - (W - 1))
    indices = np.arange(W - 1, last)  # decision rows [W-1 .. last-1]
    if indices.size == 0:
        raise ValueError("no valid windows (window_len + target warmup exceed series)")
    # roll window slices: rows [i-W+1 .. i] for each decision row i
    rows = indices[:, None] - np.arange(W - 1, -1, -1)[None, :]   # [B, W]
    feats = feat_series[rows]                                       # [B, W, D]
    trgs = {name: arr[indices] for name, arr in targets_dict.items()}
    ts = timestamps[indices] if timestamps is not None else None
    return WindowedDataset(
        features=feats.astype(np.float32),
        targets={k: (v.astype(np.float32) if np.issubdtype(v.dtype, np.floating) else v) for k, v in trgs.items()},
        sample_idx=indices,
        timestamps=ts,
        window_len=W,
        feature_dim=D,
    )