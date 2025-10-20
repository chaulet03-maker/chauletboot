from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import Any, Dict, List

from bot.identity import get_bot_id
from bot.ledger import DB_PATH


async def _get_bot_fills_since(mode: str, since_ts: int) -> List[Dict[str, Any]]:
    """Retrieve fills recorded by the bot for the given ``mode`` since ``since_ts``."""

    norm_mode = "live" if str(mode).lower() in {"live", "real"} else "paper"
    bot_id = get_bot_id()

    def _query() -> List[Dict[str, Any]]:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT symbol, side, qty, price, ts
                FROM fills
                WHERE mode=? AND bot_id=? AND ts>=?
                ORDER BY ts ASC
                """,
                (norm_mode, bot_id, int(since_ts)),
            ).fetchall()
        return [
            {
                "symbol": str(symbol),
                "side": str(side).upper(),
                "qty": float(qty),
                "price": float(price),
                "ts": int(ts),
            }
            for symbol, side, qty, price, ts in rows
        ]

    return await asyncio.to_thread(_query)


async def pnl_summary_bot(mode: str, mark_provider=None) -> Dict[str, Dict[str, float]]:
    """Return daily and weekly PnL (realized/unrealized) for the bot."""

    now = int(time.time())
    one_day = now - 86400
    one_week = now - 7 * 86400
    results: Dict[str, Dict[str, float]] = {}

    norm_mode = "live" if str(mode).lower() in {"live", "real"} else "paper"

    for label, since in (("daily", one_day), ("weekly", one_week)):
        fills = await _get_bot_fills_since(norm_mode, since)
        realized = 0.0
        per_symbol: Dict[str, tuple[float, float]] = {}
        for fill in fills:
            symbol = str(fill["symbol"]).upper()
            side = str(fill["side"]).upper()
            qty = float(fill["qty"])
            price = float(fill["price"])
            pos_qty, avg_price = per_symbol.get(symbol, (0.0, 0.0))

            if side == "BUY":
                if pos_qty >= 0:
                    new_notional = pos_qty * avg_price + qty * price
                    pos_qty = pos_qty + qty
                    avg_price = new_notional / pos_qty if pos_qty != 0 else 0.0
                else:
                    close_qty = min(qty, -pos_qty)
                    realized += close_qty * (avg_price - price)
                    pos_qty += close_qty
                    if qty > close_qty:
                        open_qty = qty - close_qty
                        pos_qty = open_qty
                        avg_price = price
            else:  # SELL
                if pos_qty <= 0:
                    new_notional = (-pos_qty) * avg_price + qty * price
                    pos_qty = pos_qty - qty
                    avg_price = new_notional / (-pos_qty) if pos_qty != 0 else 0.0
                else:
                    close_qty = min(qty, pos_qty)
                    realized += close_qty * (price - avg_price)
                    pos_qty -= close_qty
                    if qty > close_qty:
                        open_qty = qty - close_qty
                        pos_qty = -open_qty
                        avg_price = price
            per_symbol[symbol] = (pos_qty, avg_price)

        unrealized = 0.0
        if callable(mark_provider):
            for symbol, (pos_qty, avg_price) in per_symbol.items():
                if pos_qty == 0:
                    continue
                try:
                    mark = await mark_provider(symbol)
                except Exception:
                    mark = None
                if mark is None:
                    continue
                mark_f = float(mark)
                if pos_qty > 0:
                    unrealized += pos_qty * (mark_f - avg_price)
                else:
                    unrealized += (-pos_qty) * (avg_price - mark_f)

        results[label] = {
            "realized": realized,
            "unrealized": unrealized,
            "total": realized + unrealized,
        }

    return results


__all__ = ["pnl_summary_bot"]
