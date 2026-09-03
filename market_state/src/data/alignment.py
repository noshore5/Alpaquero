"""Temporal alignment of multi-symbol bar data.

The central difficulty of this project: US equities trade ~6.5h/day, crypto
trades 24/7, and individual instruments have sporadic missing bars. We must
*not* naively concat by timestamp. This module provides an explicit alignment
layer with the following policy:

1. Normalise every bar's timestamp to UTC.
2. Build a common modelling timeline as a regular UTC grid at the requested
   bar interval (e.g. every 5 minutes, aligned to the bar boundary).
3. For the first implementation, identify the subset of grid timestamps at
   which the *required* cross-asset coverage is sufficient (a configurable
   minimum fraction of the universe present), and restrict the model to those
   timestamps. This cleanly handles equity-hours vs crypto-24/7: crypto bars
   at 03:00 UTC (when US markets are closed) are *not fabricated* -- they are
   simply excluded because the equity shelf can't be populated there.
4. Missing bars are represented explicitly (NaN), never forward-filled, so
   returns are never silently fabricated.

Return series are computed *after* alignment, from present close prices, so
``r_i(t) = log(P_i(t)/P_i(t-1))`` only ever uses real, aligned observations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("market_state.data.alignment")

_BAR_SECONDS = {"1Min": 60, "5Min": 300, "15Min": 900, "1Hour": 3600, "1Day": 86400}


@dataclass
class AlignmentResult:
    timeline: pd.DatetimeIndex          # common modelling timeline (UTC)
    prices: pd.DataFrame                # index=timeline, cols=symbols (NaN where absent)
    coverage: pd.Series                 # fraction of universe present per timestep
    valid_mask: np.ndarray              # bool mask of timesteps passing coverage threshold
    symbols: list[str]
    timeframe: str
    dropped_timesteps: int = 0
    per_symbol_present: dict[str, float] = field(default_factory=dict)


def _bar_align(ts: pd.Timestamp, bar_seconds: int) -> pd.Timestamp:
    """Snap a timestamp to the nearest lower bar boundary (UTC)."""
    epoch = int(ts.value // 1_000_000_000)  # seconds
    aligned = epoch - (epoch % bar_seconds)
    return pd.Timestamp(aligned, unit="s", tz="UTC")


def build_timeline(
    bars: dict[str, pd.DataFrame],
    timeframe: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Build the common modelling timeline (UTC, regular bar grid).

    The timeline spans the union of all symbols' bar ranges, intersecting
    [start, end] if given. Aligned to bar boundaries.
    """
    bar_sec = _BAR_SECONDS[timeframe]
    if not bars:
        raise ValueError("no bars to align")
    # Normalise each symbol's timestamps to tz-aware UTC first: raw parquet can
    # come back as tz-naive even when the bar was authored UTC-aware.
    def _utc_ts(s: str) -> pd.Timestamp:
        ts = pd.Timestamp(s)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    ts_cols = [pd.to_datetime(df["timestamp"], utc=True) for df in bars.values() if len(df)]
    t0 = min(col.min() for col in ts_cols)
    t1 = max(col.max() for col in ts_cols)
    if start is not None:
        t0 = max(t0, _utc_ts(start))
    if end is not None:
        t1 = min(t1, _utc_ts(end))
    a0 = _bar_align(t0, bar_sec)
    a1 = _bar_align(t1, bar_sec)
    idx = pd.date_range(a0, a1, freq=timeframe, tz="UTC")
    return idx


def _close_matrix(
    bars: dict[str, pd.DataFrame],
    timeline: pd.DatetimeIndex,
    timeframe: str,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Build the aligned close-price matrix (index=timeline, cols=symbols).

    Uses the *open* bar's timestamp as the representation of a bar that
    starts at that grid point. Rows/symbols that have no bar in a given slot
    are NaN (explicit absence, not filled).
    """
    if symbols is None:
        symbols = sorted(bars.keys())
    bar_sec = _BAR_SECONDS[timeframe]
    grid = {ts: i for i, ts in enumerate(timeline)}
    data = np.full((len(timeline), len(symbols)), np.nan)
    for ci, sym in enumerate(symbols):
        df = bars[sym]
        if df is None or len(df) == 0:
            logger.warning("no data for %s", sym)
            continue
        ts = pd.to_datetime(df["timestamp"], utc=True)
        aligned = ts.map(lambda t: _bar_align(t, bar_sec)).values
        close = df["close"].to_numpy(dtype=float)
        for a, c in zip(aligned, close):
            i = grid.get(pd.Timestamp(a, tz="UTC"))
            if i is not None and np.isnan(data[i, ci]):
                data[i, ci] = c
    res = pd.DataFrame(data, index=timeline, columns=symbols)
    res.index.name = "timestamp"
    return res


def align_bars(
    bars: dict[str, pd.DataFrame],
    timeframe: str,
    *,
    min_coverage: float = 0.7,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    symbols: list[str] | None = None,
) -> AlignmentResult:
    """Align a dict of per-symbol bar DataFrames onto a common UTC grid.

    Parameters
    ----------
    bars : {symbol: DataFrame[timestamp, open, high, low, close, volume]}
    timeframe : bar interval string.
    min_coverage : minimum fraction of the universe that must be present for
        a grid timestep to be kept in the modelling timeline. Timesteps below
        this (e.g. US-hours requirement vs crypto-24/7) are dropped, not
        fabricated.
    """
    if symbols is None:
        symbols = sorted(bars.keys())
    syms = symbols
    timeline = build_timeline(bars, timeframe, start, end)
    prices = _close_matrix(bars, timeline, timeframe, syms)
    present = prices.notna()
    n_present = present.sum(axis=1)
    coverage = n_present / len(syms)
    valid_mask = coverage.values >= min_coverage

    per_symbol = present.mean(axis=0).to_dict()
    dropped = int((~valid_mask).sum())

    if dropped:
        logger.info("alignment dropped %d timesteps below coverage %.2f", dropped, min_coverage)

    # Sanity: no forward-fill of returns is done here; only present closes.
    return AlignmentResult(
        timeline=timeline,
        prices=prices,
        coverage=coverage,
        valid_mask=valid_mask,
        symbols=syms,
        timeframe=timeframe,
        dropped_timesteps=dropped,
        per_symbol_present=per_symbol,
    )


def log_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """``r_i(t) = log(P_i(t)/P_i(t-1))``.

    Uses only *present* aligned closes. A NaN close at t or t-1 yields a NaN
    return (missing is explicit, never forward-filled). A symbol that is a
    valid instrument at t (not NaN) but whose previous aligned close was NaN
    will have a NaN return for that step and is simply unavailable for the
    model, which is the honest representation of a gap.
    """
    prev = prices.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(prices / prev)
    return ret
