"""Binance public historical bar downloader (no API key required).

The Alpaca free feed cannot serve the equity universe this project wants
(IEX 15-minute restriction). Crypto has no such limit: Binance's public
``/api/v3/klines`` endpoint serves years of minute/5-minute OHLCV for every
liquid USDT pair, unauthenticated, 24/7 (no session gaps).

This downloader mirrors ``data.alpaca.AlpacaDownloader``'s output contract so
the rest of the pipeline is unchanged:

* Writes ``<raw_dir>/crypto/<PAIR>.parquet`` where ``<PAIR>`` is the config
  symbol with ``/`` replaced by ``_`` (e.g. ``BTC/USDT`` -> ``BTC_USDT``),
  matching ``scripts/build_features.py::load_raw_bars``.
* Each parquet has columns ``symbol, timestamp (UTC), open, high, low,
  close, volume``.
* Paged by ``startTime`` cursor (Binance caps a response at 1000 bars).
* Retries with exponential backoff on transient HTTP failures.
* Resumable: an existing file is loaded and only the missing leading/trailing
  ranges are fetched.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger("market_state.data.binance")

# data-api.binance.vision is the read-only market-data host (no key, generous
# limits, same payloads as api.binance.com).
_BASE = "https://data-api.binance.vision/api/v3/klines"

_INTERVAL_MAP = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "1Hour": "1h",
    "1Day": "1d",
}
_INTERVAL_MS = {
    "1Min": 60_000,
    "5Min": 300_000,
    "15Min": 900_000,
    "1Hour": 3_600_000,
    "1Day": 86_400_000,
}
_MAX_PAGE = 1000


@dataclass
class DownloadResult:
    symbols_requested: list[str]
    symbols_saved: list[str]
    unavailable: list[str]
    bars_per_symbol: dict[str, int]
    raw_dir: Path


def _as_ms(value: str | datetime) -> int:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _venue_symbol(sym: str) -> str:
    """``BTC/USDT`` / ``BTC-USDT`` / ``BTC/USD`` -> ``BTCUSDT`` (Binance form)."""
    s = sym.replace("/", "").replace("-", "").upper()
    if s.endswith("USD") and not s.endswith("USDT") and not s.endswith("BUSD"):
        s = s + "T"  # Binance quotes in USDT, not USD
    return s


class BinanceDownloader:
    def __init__(
        self,
        raw_dir: str | Path,
        timeframe: str = "5Min",
        *,
        max_retries: int = 5,
        retry_backoff_s: float = 1.5,
        session: requests.Session | None = None,
        request_pause_s: float = 0.15,
    ) -> None:
        if timeframe not in _INTERVAL_MAP:
            raise ValueError(f"Unsupported timeframe {timeframe!r}")
        self.raw_dir = Path(raw_dir)
        self.timeframe = timeframe
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.request_pause_s = request_pause_s
        self.session = session or requests.Session()

    # -- HTTP with retry -------------------------------------------------------

    def _get(self, params: dict) -> list:
        attempt = 0
        while True:
            try:
                r = self.session.get(_BASE, params=params, timeout=30)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (418, 429):  # rate limited / banned
                    wait = float(r.headers.get("Retry-After", self.retry_backoff_s * 4))
                    logger.warning("rate limited (%s), sleeping %.1fs", r.status_code, wait)
                    time.sleep(wait)
                    continue
                if r.status_code == 400:
                    # invalid symbol / bad params -> treat as unavailable
                    logger.warning("400 for %s: %s", params.get("symbol"), r.text[:200])
                    return []
                r.raise_for_status()
            except (requests.RequestException, ValueError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                sleep = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning("request failed (%s), retry %d in %.1fs", exc, attempt, sleep)
                time.sleep(sleep)

    # -- per-symbol fetch ----------------------------------------------------

    def _fetch_range(self, venue_sym: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        interval = _INTERVAL_MAP[self.timeframe]
        step = _INTERVAL_MS[self.timeframe]
        rows: list[dict] = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = self._get({
                "symbol": venue_sym,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": _MAX_PAGE,
            })
            if not batch:
                break
            for k in batch:
                open_ms = int(k[0])
                rows.append({
                    "timestamp": pd.Timestamp(open_ms, unit="ms", tz="UTC"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            last_open = int(batch[-1][0])
            nxt = last_open + step
            if nxt <= cursor:
                break
            cursor = nxt
            if len(batch) < _MAX_PAGE:
                break
            time.sleep(self.request_pause_s)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows)

    # -- cache helpers ------------------------------------------------------

    @staticmethod
    def _load_existing(fpath: Path) -> pd.DataFrame | None:
        if not fpath.exists():
            return None
        try:
            df = pd.read_parquet(fpath)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        except Exception:  # noqa: BLE001
            return None

    def _atomic_write(self, df: pd.DataFrame, fpath: Path) -> None:
        tmp = fpath.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(fpath)

    # -- public API --------------------------------------------------------

    def download(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> DownloadResult:
        start_ms = _as_ms(start)
        end_ms = _as_ms(end)
        step = _INTERVAL_MS[self.timeframe]
        sub_dir = self.raw_dir / "crypto"
        sub_dir.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []
        unavailable: list[str] = []
        counts: dict[str, int] = {}

        for sym in symbols:
            venue = _venue_symbol(sym)
            fpath = sub_dir / (sym.replace("/", "_") + ".parquet")
            cached = self._load_existing(fpath)

            ranges: list[tuple[int, int]] = []
            if cached is None or len(cached) == 0:
                ranges = [(start_ms, end_ms)]
            else:
                c0 = int(cached["timestamp"].min().timestamp() * 1000)
                c1 = int(cached["timestamp"].max().timestamp() * 1000)
                if start_ms < c0:
                    ranges.append((start_ms, c0 - step))
                if end_ms > c1:
                    ranges.append((c1 + step, end_ms))

            if not ranges:
                counts[sym] = int(len(cached))
                saved.append(sym)
                logger.info("%s: cached %d bars, up to date", sym, len(cached))
                continue

            frames = [cached] if cached is not None and len(cached) else []
            got = 0
            for (r0, r1) in ranges:
                if r0 > r1:
                    continue
                df = self._fetch_range(venue, r0, r1)
                if len(df):
                    frames.append(df)
                    got += len(df)

            if not frames:
                unavailable.append(sym)
                logger.warning("%s (%s): no data returned", sym, venue)
                continue

            full = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset="timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            full.insert(0, "symbol", sym)
            self._atomic_write(full, fpath)
            counts[sym] = int(len(full))
            saved.append(sym)
            logger.info("saved %s (%s): total %d bars (+%d new)", sym, venue, len(full), got)
            time.sleep(self.request_pause_s)

        return DownloadResult(
            symbols_requested=list(symbols),
            symbols_saved=sorted(saved),
            unavailable=sorted(set(unavailable)),
            bars_per_symbol=counts,
            raw_dir=self.raw_dir,
        )
