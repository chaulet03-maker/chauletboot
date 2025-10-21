from __future__ import annotations

import math
from typing import Optional


def _fmt_level(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def open_msg(
    symbol,
    side,
    margin,
    lev,
    notional,
    entry,
    tp1,
    tp2,
    sl,
    order_type,
    latency_ms,
):
    arrow = "↑" if side == "long" else "↓"
    side_u = "LONG" if side == "long" else "SHORT"
    tp1_val = None if tp1 is None else float(tp1)
    tp2_val = None if tp2 is None else float(tp2)
    sl_val = None if sl is None else float(sl)

    tp_line: str
    if tp1_val is None and tp2_val is None:
        tp_line = "TP: -"
    elif tp2_val is None or tp1_val is None:
        tp_line = f"TP: {_fmt_level(tp1_val or tp2_val)}"
    elif math.isclose(tp1_val, tp2_val, rel_tol=1e-9, abs_tol=1e-6):
        tp_line = f"TP: {_fmt_level(tp1_val)}"
    else:
        tp_line = f"TP1: {_fmt_level(tp1_val)} / TP2: {_fmt_level(tp2_val)}"

    return (
        f"{arrow} {side_u} {symbol.split('/')[0]}\n"
        f"Margen: ${margin:.2f} / lev x{lev} / total ${notional:.2f}\n"
        f"Entrada: ${entry:.2f}\n"
        f"{tp_line}\n"
        f"SL: {_fmt_level(sl_val)}\n"
        f"Orden: {order_type} / lat {latency_ms}ms"
    )

def close_msg(symbol, side, qty, entry, exitp, pnl_net, pct_str, holding_h, lev, ok=True):
    tick = "✅" if ok else "❌"
    return (f"{tick} CIERRE {symbol.split('/')[0]}\n"
            f"Qty entrada: {qty:.6f}\n"
            f"Entrada: ${entry:.2f} / Salida: ${exitp:.2f}\n"
            f"PnL neto: ${pnl_net:.2f} ({pct_str})\n"
            f"Holding: {holding_h}\n"
            f"Lev x{lev}")
