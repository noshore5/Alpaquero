#!/usr/bin/env python
"""CLI wrapper for the inference latency/throughput benchmark.

Usage:
    ../.venv_market/bin/python scripts/benchmark.py [--config ...] [--devices cpu,mps]
    [--window N] [--assets N] [--iters N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from benchmarks.benchmark_inference import benchmark, format_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--window", type=int, default=None,
                    help="Override lookback window length (default config.window.bars)")
    ap.add_argument("--assets", type=int, default=None,
                    help="Override number of assets (default len(symbols))")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_assets = args.assets or len(cfg["data"]["symbols"])
    window = args.window or cfg["window"]["bars"]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    report = benchmark(cfg, n_assets=n_assets, window_len=window,
                       n_batch=args.batch, n_iters=args.iters, devices=devices)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
