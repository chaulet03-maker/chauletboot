import json
import logging
import os
from datetime import datetime, timedelta, timezone

STATE_FILE = "runtime_state.json"


def _utcnow():
    return datetime.now(timezone.utc)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"paused_until": None}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def is_paused_now():
    state = load_state()
    pu = state.get("paused_until")
    if not pu:
        return False
    try:
        pu_dt = datetime.fromisoformat(pu)
    except Exception:
        return False
    return _utcnow() < pu_dt


def set_pause_hours(hours: float):
    until = _utcnow() + timedelta(hours=hours)
    state = load_state()
    state["paused_until"] = until.isoformat()
    save_state(state)
    logging.warning(f"[RISK] Trading en PAUSA hasta {until.isoformat()}.")


def clear_pause_if_expired():
    state = load_state()
    pu = state.get("paused_until")
    if not pu:
        return
    pu_dt = datetime.fromisoformat(pu)
    if _utcnow() >= pu_dt:
        state["paused_until"] = None
        save_state(state)
        logging.info("[RISK] Pausa expirada, reanudando trading.")


def in_ban_hours(ban_hours_str: str) -> bool:
    """ban_hours_str ejemplo: '0,1,2,3' (UTC)"""
    if not ban_hours_str:
        return False
    try:
        banned = {int(h.strip()) for h in ban_hours_str.split(",") if h.strip() != ""}
    except Exception:
        return False
    return _utcnow().hour in banned


def compute_adx(klines_14):  # klines_14 = ohlcv últimos ≥ 50 velas (ts, open, high, low, close, vol)
    # implementación mínima sin libs externas (Wilder’s smoothing simplificada)
    # Si ya usas `ta`/`pandas_ta`, reemplaza por eso.
    if len(klines_14) < 20:
        return None
    highs = [k[2] for k in klines_14]
    lows = [k[3] for k in klines_14]
    closes = [k[4] for k in klines_14]

    tr_list, plus_dm_list, minus_dm_list = [], [], []
    for i in range(1, len(klines_14)):
        high, low, prev_close = highs[i], lows[i], closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm_list.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm_list.append(down_move if (down_move > up_move and down_move > 0) else 0.0)

    n = 14
    if len(tr_list) < n:
        return None

    def rma(vals, period):
        alpha = 1.0 / period
        r = vals[0]
        for v in vals[1:]:
            r = r + alpha * (v - r)
        return r

    atr = rma(tr_list[-n:], n)
    plus_di = 100.0 * (rma(plus_dm_list[-n:], n) / atr) if atr else 0.0
    minus_di = 100.0 * (rma(minus_dm_list[-n:], n) / atr) if atr else 0.0
    denom = plus_di + minus_di
    dx = 100.0 * abs(plus_di - minus_di) / denom if denom else 0.0
    # ADX es RMA del DX
    adx = rma([dx] * n, n)  # simplificado
    return adx


def dyn_leverage_from_adx(adx: float, weak_thr=25.0, strong_x=10, weak_x=5):
    if adx is None:
        return weak_x
    return strong_x if adx >= weak_thr else weak_x


def get_24h_change_pct_ccxt(exchange, symbol):
    """
    Usa el % 24h del ticker si existe; fallback a OHLCV 1h últimas 24 velas.
    symbol ejemplo: 'BTC/USDT:USDT' en Binance Futures
    """
    try:
        t = exchange.fetch_ticker(symbol)
        pct = t.get("percentage")
        if pct is not None:
            return float(pct)
    except Exception:
        pass
    # Fallback
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=25)
        if len(ohlcv) >= 2:
            p0 = ohlcv[0][4]
            p1 = ohlcv[-1][4]
            return (p1 - p0) / p0 * 100.0
    except Exception:
        pass
    return 0.0


def maybe_trigger_shock_pause_on_loss(
    exchange,
    symbol,
    trade_pnl,
    shock_move_threshold_pct: float,
    shock_pause_hours: float,
):
    """Llamar cuando cierras un trade. Si fue perdedor y el movimiento 24h supera el umbral, activa pausa."""
    if trade_pnl is None or trade_pnl >= 0:
        return
    pct = abs(get_24h_change_pct_ccxt(exchange, symbol))
    if pct >= shock_move_threshold_pct:
        set_pause_hours(shock_pause_hours)
        logging.warning(
            f"[RISK] Shock detectado (Δ24h={pct:.2f}%) y trade perdedor -> PAUSA {shock_pause_hours}h."
        )
