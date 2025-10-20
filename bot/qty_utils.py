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

    if step > 0:
        qty = math.floor(qty / step) * step
    if min_qty > 0 and qty < min_qty:
        return 0.0
    if min_notional > 0 and price and price * qty < min_notional:
        target = min_notional / price
        if step > 0:
            qty = math.ceil(target / step) * step
        else:
            qty = target
    return max(qty, 0.0)
