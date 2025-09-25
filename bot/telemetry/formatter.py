def open_msg(symbol, side, margin, lev, notional, entry, tp1, tp2, sl, order_type, latency_ms):
    arrow = "↑" if side=="long" else "↓"
    side_u = "LONG" if side=="long" else "SHORT"
    return (f"{arrow} {side_u} {symbol.split('/')[0]}\n"
            f"Margen: ${margin:.2f} / lev x{lev} / total ${notional:.2f}\n"
            f"Entrada: ${entry:.2f}\n"
            f"TP1: ${tp1:.2f} / TP2: ${tp2:.2f}\n"
            f"SL: ${sl:.2f}\n"
            f"Orden: {order_type} / lat {latency_ms}ms")

def close_msg(symbol, side, qty, entry, exitp, pnl_net, pct_str, holding_h, lev, ok=True):
    tick = "✅" if ok else "❌"
    return (f"{tick} CIERRE {symbol.split('/')[0]}\n"
            f"Qty entrada: {qty:.6f}\n"
            f"Entrada: ${entry:.2f} / Salida: ${exitp:.2f}\n"
            f"PnL neto: ${pnl_net:.2f} ({pct_str})\n"
            f"Holding: {holding_h}\n"
            f"Lev x{lev}")
