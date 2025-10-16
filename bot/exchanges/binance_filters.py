"""Utilities for Binance Futures trading rules.

This module centralises the logic required to comply with Binance's
quantity and price filters.  Binance will reject any order whose
quantity is not a multiple of ``stepSize`` or whose price is not a
multiple of ``tickSize``.  The helpers exposed here keep that logic in a
single place so that both the live broker and potential backtests can
reuse it.

The functions intentionally accept plain dictionaries so they can be fed
with data coming either from python-binance (``exchangeInfo``) or from
ccxt's ``market`` description.  Only the keys that we care about are
extracted from the payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Any, Dict, Iterable, Optional
import json

# Give enough precision so flooring operations on very small ticks (e.g.
# 0.0001) remain precise when represented as decimals.
getcontext().prec = 18


@dataclass(frozen=True)
class SymbolFilters:
    """Relevant trading filters for a Binance symbol."""

    symbol: str
    step_size: Decimal
    tick_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    @property
    def precision(self) -> int:
        return max(0, -self.tick_size.normalize().as_tuple().exponent)


def _decimal(value: Any, default: str = "0") -> Decimal:
    if value in (None, "", 0):
        return Decimal(default)
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)


def _extract_filter(filters: Iterable[Dict[str, Any]], *names: str) -> Dict[str, Any]:
    for f in filters:
        if str(f.get("filterType", "")).upper() in {n.upper() for n in names}:
            return f
    return {}


def _get_filters_from_market(market: Dict[str, Any]) -> SymbolFilters:
    info = market.get("info", {}) if isinstance(market, dict) else {}
    filters = info.get("filters", []) if isinstance(info, dict) else []

    lot = _extract_filter(filters, "MARKET_LOT_SIZE", "LOT_SIZE") or {}
    price = _extract_filter(filters, "PRICE_FILTER") or {}
    min_notional_filter = _extract_filter(filters, "MIN_NOTIONAL", "NOTIONAL") or {}

    step = _decimal(lot.get("stepSize"), "0.001")
    tick = _decimal(price.get("tickSize"), "0.01")
    min_qty = _decimal(lot.get("minQty"), "0")
    min_notional = _decimal(min_notional_filter.get("notional")) if min_notional_filter else Decimal("0")

    return SymbolFilters(
        symbol=str(info.get("symbol") or market.get("symbol") or market.get("id") or ""),
        step_size=step,
        tick_size=tick,
        min_qty=min_qty,
        min_notional=min_notional,
    )


_FILTER_CACHE: dict[tuple[str, str], SymbolFilters] = {}


def build_filters(symbol: str, market: Dict[str, Any]) -> SymbolFilters:
    """Return the :class:`SymbolFilters` for *symbol*.

    ``symbol`` is only used as part of the cache key.  ``market`` should
    be the raw description of the symbol as returned by Binance's
    ``exchangeInfo`` or ccxt's ``market()`` call.
    """

    try:
        cache_key = (symbol, json.dumps(market, sort_keys=True))
    except Exception:
        cache_key = (symbol, str(market))

    cached = _FILTER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    filters = _get_filters_from_market(market)
    if not filters.symbol:
        object.__setattr__(filters, "symbol", symbol.replace("/", ""))
    _FILTER_CACHE[cache_key] = filters
    return filters


def quantize_qty(filters: SymbolFilters, quantity: float) -> float:
    """Floor *quantity* to the closest allowed multiple of ``stepSize``."""

    step = filters.step_size
    if step <= 0:
        return float(quantity)
    qty = Decimal(str(quantity))
    units = (qty / step).to_integral_value(rounding=ROUND_DOWN)
    quantized = units * step
    return float(quantized)


def quantize_price(filters: SymbolFilters, price: float) -> float:
    """Floor *price* to the closest allowed multiple of ``tickSize``."""

    tick = filters.tick_size
    if tick <= 0:
        return float(price)
    px = Decimal(str(price))
    units = (px / tick).to_integral_value(rounding=ROUND_DOWN)
    quantized = units * tick
    return float(quantized.quantize(tick, rounding=ROUND_DOWN))


def validate_order(filters: SymbolFilters, quantity: float, price: Optional[float]) -> None:
    """Ensure the order respects minQty and minNotional rules."""

    qty_dec = Decimal(str(quantity))
    if qty_dec <= 0:
        raise ValueError("Quantity must be positive after quantization")
    if filters.min_qty > 0 and qty_dec < filters.min_qty:
        raise ValueError(
            f"Quantity {quantity} is below minQty {float(filters.min_qty)} for {filters.symbol}"
        )
    if price is not None and filters.min_notional > 0:
        notion = qty_dec * Decimal(str(price))
        if notion < filters.min_notional:
            raise ValueError(
                f"Notional {float(notion)} is below minNotional {float(filters.min_notional)} for {filters.symbol}"
            )
