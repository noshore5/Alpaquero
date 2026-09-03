"""Symbol universe handling.

The initial universe is ~30 cross-asset instruments. This module owns the
classification of symbols (equity, crypto, etc.), reporting which symbols the
active data source can actually serve, and helpers to split likely-unavailable
symbols out without silently dropping them.
"""
from __future__ import annotations

from dataclasses import dataclass

# Asset classes the model reasons about (used for grouping / alignment policies).
EQUITIES = {
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "FEZ", "EWJ", "FXI",
    "VNQ", "XLE", "XLF", "SMH",
}
RATES = {"TLT", "IEF", "SHY", "TIP"}
COMMODITIES = {"GLD", "SLV", "USO", "UNG", "CPER"}
FX = {"UUP", "FXE", "FXY", "FXB"}
VOLATILITY = {"VIXY", "VXX"}
CRYPTO = {"BTC/USD", "ETH/USD"}


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    asset_class: str
    is_crypto: bool


def classify(symbol: str) -> SymbolInfo:
    """Return asset-class metadata for a symbol."""
    s = symbol.upper()
    if "/" in s:
        return SymbolInfo(s, "crypto", True)
    if s in CRYPTO:
        return SymbolInfo(s, "crypto", True)
    asset_class = "equity"
    if s in RATES:
        asset_class = "rates"
    elif s in COMMODITIES:
        asset_class = "commodity"
    elif s in FX:
        asset_class = "fx"
    elif s in VOLATILITY:
        asset_class = "volatility"
    return SymbolInfo(s, asset_class, False)


def default_crypto_symbols() -> list[str]:
    return sorted(CRYPTO)


def report_unavailable(requested: list[str], available: set[str]) -> dict[str, list[str]]:
    """Partition requested symbols into available / unavailable.

    Returns {"available": [...], "unavailable": [...]}. Unavailable symbols
    are *reported*, never silently dropped.
    """
    available = {s.upper() for s in available}
    req = [s.upper() for s in requested]
    return {
        "available": [s for s in req if s in available],
        "unavailable": [s for s in req if s not in available],
    }


def infer_missing_coverage(
    requested: list[str],
    reported_missing: list[str],
) -> list[str]:
    """List of requested symbols that were reported missing by the data
    source, in requested order (used to warn but continue)."""
    missing = set(reported_missing)
    return [s for s in requested if s.upper() in missing]
