#!/usr/bin/env python
"""Fetch historical Alpaca bars for the configured universe into ``data/raw``.

Usage:
    ../.venv_market/bin/python scripts/download.py [--config configs/default.yaml]

Requires ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` (or ``ALPACA_*``); the
downloader loads a leading ``.env`` if present. Symbols the account cannot
serve are reported (not silently dropped).

Run the script from the ``market_state/`` directory so the ``../`` python path
and relative ``configs`` path resolve.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, config_hash


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to YAML config (default: configs/default.yaml)")
    ap.add_argument("--out", default=None, help="Override raw data directory")
    ap.add_argument("--provider", default=None,
                    help="alpaca | binance (default: data.provider in config, else alpaca)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    raw_dir = args.out or d["raw_dir"]
    syms = d["symbols"]
    provider = (args.provider or d.get("provider") or "alpaca").lower()

    if provider == "binance":
        from data.binance import BinanceDownloader, DownloadResult  # noqa: F401
        dl = BinanceDownloader(raw_dir=raw_dir, timeframe=d["timeframe"])
    elif provider == "alpaca":
        from data.alpaca import AlpacaDownloader, DownloadResult  # noqa: F401
        dl = AlpacaDownloader(raw_dir=raw_dir, timeframe=d["timeframe"])
    else:
        raise SystemExit(f"unknown provider {provider!r} (expected alpaca | binance)")
    print(f"[download] provider: {provider}")
    print(f"[download] {len(syms)} symbols, {d['timeframe']}, {d['start']}..{d['end']}")
    print(f"[download] config hash: {config_hash(cfg)}")
    res: DownloadResult = dl.download(syms, d["start"], d["end"])

    print(f"  saved      : {len(res.symbols_saved)}")
    print(f"  unavailable: {len(res.unavailable)}")
    for s in res.unavailable:
        print(f"    - {s}")
    totals = sum(res.bars_per_symbol.get(s, 0) for s in res.symbols_requested)
    print(f"  total bars : {totals} (per-symbol: {res.bars_per_symbol})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
