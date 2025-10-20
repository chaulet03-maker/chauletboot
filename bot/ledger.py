import os
import sqlite3
import time
from typing import Optional, Tuple, Dict, Any

DB_PATH = os.getenv("LEDGER_DB_PATH", "data/runtime/ledger.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _conn():
    return sqlite3.connect(DB_PATH)


def init():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS orders(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT, bot_id TEXT, symbol TEXT, side TEXT,
            client_oid TEXT, order_id TEXT, leverage INTEGER,
            qty REAL, price REAL, reduce_only INTEGER DEFAULT 0,
            opened_at INTEGER
        )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS fills(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT, bot_id TEXT, symbol TEXT, side TEXT,
            client_oid TEXT, order_id TEXT,
            qty REAL, price REAL, fee REAL, ts INTEGER
        )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS positions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT, bot_id TEXT, symbol TEXT,
            qty REAL, avg_price REAL, updated_at INTEGER
        )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_fills_time ON fills(ts)")
        c.commit()


def record_order(
    mode: str,
    bot_id: str,
    symbol: str,
    side: str,
    client_oid: str,
    order_id: Optional[str],
    leverage: int,
    qty: float,
    price: Optional[float],
    reduce_only: bool,
    opened_at: Optional[int] = None,
):
    opened_at = opened_at or int(time.time())
    with _conn() as c:
        c.execute(
            """INSERT INTO orders(mode,bot_id,symbol,side,client_oid,order_id,leverage,qty,price,reduce_only,opened_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mode,
                bot_id,
                symbol,
                side,
                client_oid,
                order_id or "",
                int(leverage),
                float(qty),
                float(price or 0.0),
                1 if reduce_only else 0,
                opened_at,
            ),
        )
        c.commit()


def record_fill(
    mode: str,
    bot_id: str,
    symbol: str,
    side: str,
    client_oid: str,
    order_id: Optional[str],
    qty: float,
    price: float,
    fee: float = 0.0,
    ts: Optional[int] = None,
):
    ts = ts or int(time.time())
    with _conn() as c:
        c.execute(
            """INSERT INTO fills(mode,bot_id,symbol,side,client_oid,order_id,qty,price,fee,ts)
                     VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                mode,
                bot_id,
                symbol,
                side,
                client_oid,
                order_id or "",
                float(qty),
                float(price),
                float(fee),
                ts,
            ),
        )
        c.commit()
    _rebuild_position(mode, bot_id, symbol)


def _rebuild_position(mode: str, bot_id: str, symbol: str):
    with _conn() as c:
        cur = c.execute(
            """SELECT side, qty, price FROM fills
                           WHERE mode=? AND bot_id=? AND symbol=? ORDER BY ts ASC""",
            (mode, bot_id, symbol),
        )
        qty = 0.0
        avg = 0.0
        for side, q, px in cur.fetchall():
            q = float(q)
            px = float(px)
            if side.upper() == "BUY":
                if qty >= 0:
                    new_notional = qty * avg + q * px
                    qty = qty + q
                    avg = new_notional / qty if qty != 0 else 0.0
                else:
                    close_qty = min(q, -qty)
                    qty += close_qty
                    if q > close_qty:
                        open_q = q - close_qty
                        new_notional = open_q * px
                        qty = open_q
                        avg = px
            else:
                if qty <= 0:
                    new_notional = (-qty) * avg + q * px
                    qty = qty - q
                    avg = new_notional / (-qty) if qty != 0 else 0.0
                else:
                    close_qty = min(q, qty)
                    qty -= close_qty
                    if q > close_qty:
                        open_q = q - close_qty
                        new_notional = open_q * px
                        qty = -open_q
                        avg = px
        now = int(time.time())
        c.execute(
            """DELETE FROM positions WHERE mode=? AND bot_id=? AND symbol=?""",
            (mode, bot_id, symbol),
        )
        c.execute(
            """INSERT INTO positions(mode,bot_id,symbol,qty,avg_price,updated_at)
                     VALUES(?,?,?,?,?,?)""",
            (mode, bot_id, symbol, qty, avg, now),
        )
        c.commit()


def bot_position(mode: str, bot_id: str, symbol: str) -> Tuple[float, float]:
    with _conn() as c:
        row = c.execute(
            """SELECT qty, avg_price FROM positions
                           WHERE mode=? AND bot_id=? AND symbol=?""",
            (mode, bot_id, symbol),
        ).fetchone()
        if not row:
            return (0.0, 0.0)
        return float(row[0] or 0.0), float(row[1] or 0.0)


def pnl_summary(
    mode: str,
    bot_id: str,
    now: Optional[int] = None,
    mark_provider=None,
) -> Dict[str, Any]:
    now = now or int(time.time())
    one_day = now - 86400
    one_week = now - 7 * 86400
    res: Dict[str, Any] = {}
    for label, since in [("daily", one_day), ("weekly", one_week)]:
        with _conn() as c:
            fls = c.execute(
                """SELECT symbol, side, qty, price, ts FROM fills
                               WHERE mode=? AND bot_id=? AND ts>=? ORDER BY ts ASC""",
                (mode, bot_id, since),
            ).fetchall()
        unreal = 0.0
        realized = 0.0
        marks: Dict[str, float] = {}
        symbols = {r[0] for r in fls}
        for sym in symbols:
            if callable(mark_provider):
                try:
                    marks[sym] = float(mark_provider(sym))
                except Exception:
                    marks[sym] = None
            else:
                marks[sym] = None
        for sym in symbols:
            qty = 0.0
            avg = 0.0
            entries = [
                (r[0], r[1], float(r[2]), float(r[3]), int(r[4]))
                for r in fls
                if r[0] == sym
            ]
            for _, side, q, px, ts in entries:
                if side.upper() == "BUY":
                    if qty >= 0:
                        new_notional = qty * avg + q * px
                        qty = qty + q
                        avg = new_notional / qty if qty != 0 else 0.0
                    else:
                        close_qty = min(q, -qty)
                        realized += close_qty * (avg - px)
                        qty += close_qty
                        if q > close_qty:
                            open_q = q - close_qty
                            qty = open_q
                            avg = px
                else:
                    if qty <= 0:
                        new_notional = (-qty) * avg + q * px
                        qty = qty - q
                        avg = new_notional / (-qty) if qty != 0 else 0.0
                    else:
                        close_qty = min(q, qty)
                        realized += close_qty * (px - avg)
                        qty -= close_qty
                        if q > close_qty:
                            open_q = q - close_qty
                            qty = -open_q
                            avg = px
            mark = marks.get(sym)
            if mark and qty != 0:
                if qty > 0:
                    unreal += qty * (mark - avg)
                else:
                    unreal += (-qty) * (avg - mark)
        res[label] = {
            "realized": realized,
            "unrealized": unreal,
            "total": realized + unreal,
        }
    return res


def prune_open_older_than(mode: str, bot_id: str, hours: int = 16):
    cutoff = int(time.time()) - hours * 3600
    with _conn() as c:
        c.execute(
            """DELETE FROM orders WHERE mode=? AND bot_id=? AND opened_at<?""",
            (mode, bot_id, cutoff),
        )
        c.commit()
