"""Preprocessing: leakage-safe input representation.

Input representation is log returns (per the design), optionally augmented
with features (realised-vol proxy, volume transform). Feature *channels* are
configurable. Normalisation parameters are ALWAYS estimated on training data
only and stored; they are never recomputed on validation/test or at inference
in a way that sees the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FittedNormalization:
    """Per-channel (per-asset x feature) mean/std, fit on training data only."""
    mean: np.ndarray  # [n_assets, n_features]
    std: np.ndarray   # [n_assets, n_features]
    n_assets: int
    n_features: int

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_assets": self.n_assets,
            "n_features": self.n_features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FittedNormalization":
        return cls(
            mean=np.asarray(d["mean"], dtype=float),
            std=np.asarray(d["std"], dtype=float),
            n_assets=int(d["n_assets"]),
            n_features=int(d["n_features"]),
        )


@dataclass
class FeatureConfig:
    """What channels to build for each asset, plus how to combine them into
    the CWT input (a single serialised per-asset series by default)."""
    channels: list[str] = field(default_factory=lambda: ["return"])
    # How channels combine into one per-asset series for the CWT. For now we
    # CWT each channel separately and stack, keeping one return channel.
    cwt_channels: list[str] = field(default_factory=lambda: ["return"])


def compute_channels(
    prices: pd.DataFrame,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    """Compute per-asset feature channels.

    Returns (features, channel_names) where:
      features      : np.ndarray [T, n_assets, n_ch]
      channel_names : [n_ch] names for the channel axis
    """
    ret = log_returns(prices)  # [T, A]
    T, A = ret.shape
    chans = {}

    if "return" in cfg.channels:
        chans["return"] = ret.values  # [T, A]

    if "abs_return" in cfg.channels:
        chans["abs_return"] = np.abs(ret.values)

    if "realized_vol" in cfg.channels:
        rv = ret.rolling(5, min_periods=1).std().values
        chans["realized_vol"] = np.nan_to_num(rv)

    if "volume_return" in cfg.channels and "volume" in prices.attrs:
        vol = prices.attrs["volume"]  # [T, A]
        chans["volume_return"] = np.log1p(np.maximum(0.0, vol.values))

    if "volume" in cfg.channels:
        vol = prices.attrs.get("volume")
        if vol is not None:
            chans["volume"] = np.log1p(np.maximum(0.0, vol.values))

    names = [c for c in cfg.channels if c in chans]
    stacked = np.stack([chans[c] for c in names], axis=-1)  # [T, A, n_ch]
    return stacked, names


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from an aligned close-price matrix (rows already aligned)."""
    return np.log(prices / prices.shift(1))


def fit_normalization(features: np.ndarray, *, mask: np.ndarray | None = None) -> FittedNormalization:
    """Fit per-asset/per-channel mean & std on TRAINING data only.

    Parameters
    ----------
    features : [T, n_assets, n_ch] (pre-computation, pre-normalise)
    mask     : optional bool [T] marking the training rows to fit on.
    """
    if features.ndim != 3:
        raise ValueError("features must be [T, n_assets, n_ch]")
    T, A, C = features.shape
    sel = features if mask is None else features[mask]
    mean = np.nanmean(sel, axis=0)  # [A, C]
    std = np.nanstd(sel, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return FittedNormalization(mean=mean, std=std, n_assets=A, n_features=C)


def apply_normalization(features: np.ndarray, norm: FittedNormalization) -> np.ndarray:
    """Z-score using pre-fitted params (never re-estimated)."""
    return (features - norm.mean) / norm.std


def as_cwt_input(
    features: np.ndarray,
    cwt_channels: list[str],
    channel_positions: dict[str, int],
) -> np.ndarray:
    """Extract the per-asset series fed to the CWT.

    For a single return channel this is just ``features[..., return_idx]``
    reshaped to ``[n_assets, T]`` -- the current simple input. Additional
    channels can be added later by making ``cwt_channels`` longer and
    stacking.
    """
    idx = [channel_positions[c] for c in cwt_channels]
    return features[..., idx]  # [T, A, len(cwt_channels)]
