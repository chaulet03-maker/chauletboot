import math
from typing import Any, Dict


async def round_and_validate_qty(exchange, symbol: str, qty: float):
    """
    Usa filtros del exchange para stepSize / minQty / minNotional.
    Devuelve qty_redondeada o 0 si no cumple.
    """

    try:
        filters: Dict[str, Any] = await exchange.get_symbol_filters(symbol)
    except Exception:
        filters = {}

    step = float(filters.get("stepSize", 0) or 0)
    min_qty = float(filters.get("minQty", 0) or 0)
    min_notional = float(filters.get("minNotional", 0) or 0)

    price = await exchange.get_current_price(symbol)

    raw_qty = qty

    if step > 0:
        qty = math.floor(qty / step) * step

    risk_cap = raw_qty * 1.10 if raw_qty > 0 else None

    if (
        qty <= 0
        and step > 0
        and price is not None
        and price > 0
        and min_notional > 0
        and risk_cap is not None
    ):
        target = min_notional / price
        ceil_qty = math.ceil(target / step) * step
        if ceil_qty <= risk_cap:
            qty = ceil_qty

    if min_qty > 0 and qty < min_qty:
        return 0.0
    if min_notional > 0 and price and price * qty < min_notional:
        target = min_notional / price
        if step > 0:
            candidate = math.ceil(target / step) * step
        else:
            candidate = target
        if risk_cap is not None and candidate <= risk_cap:
            qty = candidate
        else:
            return 0.0
    return max(qty, 0.0)
