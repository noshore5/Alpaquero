"""Alpaca historical market-data downloader.

Uses the official ``alpaca-py`` SDK (Market Data API) for both stock and
crypto minute bars. Key properties:

* Credentials come from the environment (``APCA_API_KEY_ID`` /
  ``APCA_API_SECRET_KEY``, with the older ``ALPACA_API_KEY`` /
  ``ALPACA_SECRET_KEY`` names accepted as a fallback). Never hardcoded.
* Historical responses are paginated: we loop page by page using the API's
  cursor (``next_page_token``) and never assume one request has the data.
* Retries with exponential backoff on transient failures.
* Rate-limit aware (the Alpaca REST API limits per-minute request count).
* Raw bars are cached to Parquet so repeated ranges are not re-downloaded.
* Downloads are resumable: existing files are detected and only missing
  ranges are requested where practical.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("market_state.data.alpaca")


class AlpacaConfigError(RuntimeError):
    pass


def _try_load_dotenv() -> None:
    """Best-effort load of a local ``.env`` (repo root) into os.environ."""
    try:
        from dotenv import load_dotenv

        # Probe the repository root by walking up from the package location.
        here = Path(__file__).resolve().parent
        for _ in range(6):
            env = here / ".env"
            if env.exists():
                load_dotenv(env, override=False)
                return
            here = here.parent
    except Exception:  # pragma: no cover - dotenv is optional
        pass


def get_alpaca_credentials() -> dict[str, str]:
    """Read Alpaca credentials from the environment (never hardcoded).

    Prefers ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` (the names the task
    specifies and the SDK uses), falling back to the older ``ALPACA_API_KEY``
    / ``ALPACA_SECRET_KEY`` names used elsewhere in this repo.
    """
    key_id = (
        os.environ.get("APCA_API_KEY_ID")
        or os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_PAPER_KEY_ID")
        or ""
    )
    secret = (
        os.environ.get("APCA_API_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET_KEY")
        or os.environ.get("APCA_PAPER_SECRET_KEY")
        or ""
    )
    return {"key": key_id.strip(), "secret": secret.strip()}


# --------------------------------------------------------------------------
# Timeframe translation
# --------------------------------------------------------------------------

_TIME_FRAME_MAP = {
    "1Min": (1, "Minute"),
    "5Min": (5, "Minute"),
    "15Min": (15, "Minute"),
    "1Hour": (1, "Hour"),
    "1Day": (1, "Day"),
}

ALPACA_MAX_PAGE = 10000  # Alpaca caps historical bar page size


def _py_timeframe(timeframe: str) -> object:
    """Return the ``alpaca-py`` TimeFrame object for a timeframe string."""
    from alpaca.data import TimeFrame, TimeFrameUnit
    amount, unit = _TIME_FRAME_MAP[timeframe]
    if amount == 1:
        # alpaca-py ships singletons for 1-minute / 1-hour / 1-day bars.
        return {"Minute": TimeFrame.Minute, "Hour": TimeFrame.Hour, "Day": TimeFrame.Day}[unit]
    unit_enum = {
        "Minute": TimeFrameUnit.Minute,
        "Hour": TimeFrameUnit.Hour,
        "Day": TimeFrameUnit.Day,
    }[unit]
    return TimeFrame(amount, unit_enum)


# --------------------------------------------------------------------------
# Downloader
# --------------------------------------------------------------------------

@dataclass
class DownloadResult:
    symbols_requested: list[str]
    symbols_saved: list[str]
    unavailable: list[str]
    bars_per_symbol: dict[str, int]
    raw_dir: Path


class AlpacaDownloader:
    """Paged, cached, resumable historical bar downloader.

    Parameters
    ----------
    raw_dir : Path
        Root directory; stores ``stocks/<SYM>.parquet`` and
        ``crypto/<PAIR>.parquet``.
    timeframe : str
        One of 1Min/5Min/15Min/1Hour/1Day.
    max_retries : int
        Retry count with exponential backoff on transient errors.
    retry_backoff_s : float
        Base backoff seconds.
    page_size : int
        Bars requested per page (<= Alpaca cap).
    """

    def __init__(
        self,
        raw_dir: str | Path,
        timeframe: str = "5Min",
        *,
        max_retries: int = 5,
        retry_backoff_s: float = 2.0,
        page_size: int = ALPACA_MAX_PAGE,
        stock_feed: str = "iex",
    ) -> None:
        if timeframe not in _TIME_FRAME_MAP:
            raise ValueError(f"Unsupported timeframe {timeframe!r}")
        self.raw_dir = Path(raw_dir)
        self.timeframe = timeframe
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.page_size = min(int(page_size), ALPACA_MAX_PAGE)
        self.stock_feed = stock_feed
        self._client = None

    # -- client -----------------------------------------------------------------

    def _get_clients(self):
        if self._client is not None:
            return self._client
        _try_load_dotenv()
        creds = get_alpaca_credentials()
        if not creds["key"] or not creds["secret"]:
            raise AlpacaConfigError(
                "Alpaca credentials not found in environment (APCA_API_KEY_ID / "
                "APCA_API_SECRET_KEY). Set them in .env or export them."
            )
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.historical.crypto import CryptoHistoricalDataClient
            from alpaca.data import StockBarsRequest, CryptoBarsRequest
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("alpaca-py not installed") from exc

        self._stock = StockHistoricalDataClient(creds["key"], creds["secret"])
        self._crypto = CryptoHistoricalDataClient(creds["key"], creds["secret"])
        self._StockBarsRequest = StockBarsRequest
        self._CryptoBarsRequest = CryptoBarsRequest
        self._client = (self._stock, self._crypto)
        return self._client

    # -- request with retry ------------------------------------------------------

    def _with_retries(self, fn, *args, **kwargs):
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > self.max_retries:
                    raise
                sleep = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning("Request failed (%s), retry %d in %.1fs", exc, attempt, sleep)
                time.sleep(sleep)

    # -- download ----------------------------------------------------------------

    def download(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> DownloadResult:
        """Download bars for ``symbols`` in [start, end] into ``raw_dir``."""
        self._get_clients()
        start_dt = _as_dt(start).astimezone(timezone.utc)
        end_dt = _as_dt(end).astimezone(timezone.utc)

        stocks = []
        crypto = []
        for s in symbols:
            (crypto if "/" in s else stocks).append(s)

        saved: list[str] = []
        unavailable: list[str] = []
        counts: dict[str, int] = {}

        if stocks:
            avail, unav, n = self._download_asset_class(
                stocks, start_dt, end_dt, asset_class="stock"
            )
            saved += avail
            unavailable += unav
            counts.update(n)
        if crypto:
            avail, unav, n = self._download_asset_class(
                crypto, start_dt, end_dt, asset_class="crypto"
            )
            saved += avail
            unavailable += unav
            counts.update(n)

        return DownloadResult(
            symbols_requested=list(symbols),
            symbols_saved=sorted(saved),
            unavailable=sorted(set(unavailable)),
            bars_per_symbol=counts,
            raw_dir=self.raw_dir,
        )

    def _download_asset_class(
        self,
        symbols: list[str],
        start_dt: datetime,
        end_dt: datetime,
        asset_class: str,
    ) -> tuple[list[str], list[str], dict[str, int]]:
        """Download one asset class; returns (saved, unavailable, counts)."""
        sub_dir = self.raw_dir / asset_class
        sub_dir.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []
        unavailable: list[str] = []
        counts: dict[str, int] = {}

        # Detect existing files and only request missing ranges.
        for sym in symbols:
            fpath = sub_dir / f"{sym}.parquet"
            cached = self._load_existing(fpath)
            missing = self._missing_ranges(cached, start_dt, end_dt)
            if missing is None:
                counts[sym] = int(len(cached))
                saved.append(sym)
                continue

            frames = [cached] if cached is not None and len(cached) else []
            n_missing = 0
            try:
                for (r0, r1) in missing:
                    page = self._fetch_symbol(sym, r0, r1, asset_class, sub_dir, fpath)
                    if page is None:
                        unavailable.append(sym)
                        break
                    if len(page):
                        frames.append(page)
                        n_missing += len(page)
                else:
                    full = pd.concat(frames, ignore_index=True)
                    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp")
                    self._atomic_write(full, fpath)
                    counts[sym] = int(len(full))
                    saved.append(sym)
                    logger.info("saved %s: total %d bars (+%d new)", sym, len(full), n_missing)
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.error("failed downloading %s: %s", sym, exc)
                unavailable.append(sym)

        logger.info(
            "[alpaca:%s] saved=%d unavailable=%d",
            asset_class, len(saved), len(unavailable),
        )
        return saved, unavailable, counts

    def _fetch_symbol(
        self,
        sym: str,
        start_dt: datetime,
        end_dt: datetime,
        asset_class: str,
        sub_dir: Path,
        fpath: Path,
    ) -> pd.DataFrame | None:
        """Fetch one symbol's missing range with pagination; None => unavailable."""
        if asset_class == "stock":
            client = self._client[0]
            req_cls = self._StockBarsRequest
        else:
            client = self._client[1]
            req_cls = self._CryptoBarsRequest

        tf = _py_timeframe(self.timeframe)
        frames: list[pd.DataFrame] = []
        token = None
        while True:
            kwargs = dict(
                symbol_or_symbols=sym,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
                limit=self.page_size,
            )
            if asset_class == "stock":
                kwargs["feed"] = self.stock_feed
            req = req_cls(**kwargs)
            if token is not None:
                req.next_page_token = token
            req_fn = client.get_stock_bars if asset_class == "stock" else client.get_crypto_bars
            try:
                resp = self._with_retries(req_fn, req)
            except Exception as exc:  # noqa: BLE001
                # Alpaca raises an unknown-symbol / no-data style error.
                msg = str(exc).lower()
                if any(k in msg for k in ("unknown", "invalid", "no data", "not found", "forbidden", "unavailable")):
                    return None
                raise
            bars = getattr(resp, "bars", None)
            token = getattr(resp, "next_page_token", None)
            if bars is None:
                break
            symbol = sym
            if isinstance(bars, dict):
                symbol = next(iter(bars), sym)
                bars = bars.get(symbol, [])
            if bars is None:
                break
            frames.append(_normalize_bars(bars, explicit_symbol=symbol))
            if not token:
                break
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    # -- cache helpers -----------------------------------------------------------

    def _load_existing(self, fpath: Path) -> pd.DataFrame | None:
        if not fpath.exists():
            return None
        try:
            df = pd.read_parquet(fpath)
            if "timestamp" not in df.columns:
                return None
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        except Exception:  # noqa: BLE001
            return None

    def _missing_ranges(
        self, cached: pd.DataFrame | None, start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]] | None:
        """Return date-ranges missing from cache, or None if fully covered.
        To keep resumability simple and robust we treat a single contiguous
        gap: anything before first timestamp / after last timestamp needs
        downloading. (Ranges already cached are not re-fetched.)
        """
        if cached is None or len(cached) == 0:
            return [(start, end)]
        c0 = cached["timestamp"].min()
        c1 = cached["timestamp"].max()
        ranges = []
        if start < c0:
            ranges.append((start, c0 - timedelta(seconds=1)))
        if end > c1:
            ranges.append((c1 + timedelta(seconds=1), end))
        # Fully covered only if cached spans the whole [start, end] window.
        if c0 <= start and c1 >= end:
            return None
        return ranges

    def _atomic_write(self, df: pd.DataFrame, fpath: Path) -> None:
        tmp = fpath.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(fpath)

    def probe_symbols(self, symbols: list[str], count: int = 3) -> set[str]:
        """Quickly test which symbols resolve (fetch a few bars each)."""
        self._get_clients()
        ok: set[str] = set()
        for s in symbols:
            try:
                self._fetch_symbol(s, datetime.now(timezone.utc) - timedelta(days=30),
                                   datetime.now(timezone.utc), "crypto" if "/" in s else "stock",
                                   self.raw_dir / ("crypto" if "/" in s else "stock"),
                                   Path("/tmp/_probe.parquet"))
                ok.add(s.upper())
            except Exception:  # noqa: BLE001
                continue
        return ok


def _as_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _normalize_bars(bars, explicit_symbol: str | None = None) -> pd.DataFrame:
    """Convert alpaca-py Bar objects into a canonical long-form DataFrame.

    Returns columns: symbol, timestamp (UTC), open, high, low, close, volume.
    """
    rows = []
    for b in bars:
        ts = getattr(b, "timestamp", None) or getattr(b, "t", None)
        if ts is not None and not isinstance(ts, datetime):
            ts = pd.Timestamp(ts).to_pydatetime()
        rows.append(
            {
                "symbol": explicit_symbol or getattr(b, "symbol", None),
                "timestamp": ts,
                "open": float(getattr(b, "open", getattr(b, "o", float("nan")))),
                "high": float(getattr(b, "high", getattr(b, "h", float("nan")))),
                "low": float(getattr(b, "low", getattr(b, "l", float("nan")))),
                "close": float(getattr(b, "close", getattr(b, "c", float("nan")))),
                "volume": float(getattr(b, "volume", getattr(b, "v", 0.0))),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
