from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from bot.telemetry.notifier import notify
from pause_manager import ShockPauseManager

DEFAULT_SHOCK_MOVE_THRESHOLD = 7.0
DEFAULT_SHOCK_PAUSE_HOURS = 10.0
DEFAULT_SHOCK_MESSAGE = (
    "Operación temporalmente suspendida por {hours} horas debido a volatilidad extrema "
    "(Δ24h={delta_pct:.2f}%, rango24h={range_pct:.2f}%). Se detectó una pérdida previa; "
    "se reanudará automáticamente al finalizar la pausa o con el comando /reanudar."
)


PAUSE_MANAGER = ShockPauseManager(state_path="pause_state.json")


def _utcnow():
    return datetime.now(timezone.utc)


def get_pause_manager() -> ShockPauseManager:
    return PAUSE_MANAGER


def is_paused_now():
    return PAUSE_MANAGER.is_paused()


def set_pause_hours(hours: float):
    until = PAUSE_MANAGER.set_pause_hours(hours)
    logging.warning(f"[RISK] Trading en PAUSA hasta {until.isoformat()}.")
    return until


def clear_pause_if_expired():
    if PAUSE_MANAGER.pause_until and not PAUSE_MANAGER.is_paused():
        PAUSE_MANAGER.clear_pause()
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


def _settings_to_mapping(settings: Any) -> Mapping[str, Any]:
    if isinstance(settings, Mapping):
        return settings
    if hasattr(settings, "config") and isinstance(getattr(settings, "config"), Mapping):
        return getattr(settings, "config")
    if hasattr(settings, "__dict__"):
        return {k: v for k, v in vars(settings).items() if not k.startswith("_")}
    return {}


def _nested_get(data: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _extract_shock_config(settings: Any) -> tuple[float, float, str]:
    mapping = _settings_to_mapping(settings)
    threshold = _nested_get(mapping, "risk", "guards", "shock", "move_threshold_pct")
    pause_hours = _nested_get(mapping, "risk", "guards", "shock", "pause_hours")
    template = _nested_get(mapping, "risk", "guards", "shock", "message_template")

    if threshold is None:
        threshold = mapping.get("shock_move_threshold_pct")
    if pause_hours is None:
        pause_hours = mapping.get("shock_pause_hours")
    if template is None:
        template = mapping.get("shock_message_template")

    try:
        threshold_f = float(threshold)
    except (TypeError, ValueError):
        threshold_f = DEFAULT_SHOCK_MOVE_THRESHOLD
    try:
        pause_hours_f = float(pause_hours)
    except (TypeError, ValueError):
        pause_hours_f = DEFAULT_SHOCK_PAUSE_HOURS

    template_str = str(template) if template else DEFAULT_SHOCK_MESSAGE
    return threshold_f, pause_hours_f, template_str


def _format_shock_message(
    template: str,
    *,
    hours: float,
    delta_pct: float,
    range_pct: float,
    until: Optional[datetime],
) -> str:
    context = {
        "hours": hours,
        "delta_pct": delta_pct,
        "range_pct": range_pct,
        "until": until.replace(microsecond=0).isoformat() if until else "",
    }
    try:
        return template.format(**context)
    except Exception:
        return DEFAULT_SHOCK_MESSAGE.format(**context)


def _notify_async(notifier: Any, message: str) -> None:
    if notifier is None or not hasattr(notifier, "send"):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(notifier.send(message))
    else:
        asyncio.run(notifier.send(message))


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


def get_24h_range_pct_ccxt(exchange, symbol) -> float:
    """Calcula el rango porcentual de las últimas 24h usando OHLCV de 1h."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=24)
        if not ohlcv:
            return 0.0
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]
        if not highs or not lows or not closes:
            return 0.0
        high = max(highs)
        low = min(lows)
        ref = closes[-1] or (high + low) / 2
        if ref:
            return (high - low) / ref * 100.0
    except Exception:
        pass
    return 0.0


def maybe_trigger_shock_pause_on_loss(
    exchange,
    symbol,
    trade_pnl,
    shock_move_threshold_pct: float,
    shock_pause_hours: float,
    *,
    message_template: Optional[str] = None,
):
    """Llamar cuando cierras un trade. Si fue perdedor y el movimiento 24h supera el umbral, activa pausa."""
    if trade_pnl is None or trade_pnl >= 0:
        return
    pct = abs(get_24h_change_pct_ccxt(exchange, symbol))
    if pct >= shock_move_threshold_pct:
        until = set_pause_hours(shock_pause_hours)
        range_pct = get_24h_range_pct_ccxt(exchange, symbol)
        logging.warning(
            f"[RISK] Shock detectado (Δ24h={pct:.2f}% | Rango24h={range_pct:.2f}%) y trade perdedor -> PAUSA {shock_pause_hours}h."
        )
        template = message_template or DEFAULT_SHOCK_MESSAGE
        message = _format_shock_message(
            template,
            hours=shock_pause_hours,
            delta_pct=pct,
            range_pct=range_pct,
            until=until,
        )
        notify(message)


def check_shock_pause_and_pause_if_needed(
    *,
    settings: Any,
    state: Any,
    market: Any,
    pause_manager: ShockPauseManager,
    notifier: Any = None,
) -> bool:
    """Evalúa si corresponde activar la pausa por shock tras el último trade perdedor."""

    if pause_manager is None:
        return False

    is_paused_attr = getattr(pause_manager, "is_paused_now", None)
    if callable(is_paused_attr) and is_paused_attr():
        return False
    if callable(getattr(pause_manager, "is_paused", None)) and pause_manager.is_paused():
        return False

    storage = getattr(state, "storage", None)
    get_last_trade = getattr(storage, "get_last_trade", None)
    last_trade: Optional[dict[str, Any]] = None
    if callable(get_last_trade):
        try:
            last_trade = get_last_trade()
        except Exception as exc:  # pragma: no cover - defensivo
            logging.debug("shock_guard: get_last_trade falló: %s", exc)
            last_trade = None

    if not last_trade:
        return False

    trade_id = last_trade.get("id")
    trade_key = trade_id if trade_id is not None else last_trade.get("close_timestamp")
    if trade_key is None:
        trade_key = last_trade.get("timestamp")
    last_processed = getattr(state, "_shock_guard_last_trade_id", None)
    if trade_key is not None and last_processed == trade_key:
        return False

    pnl_raw = last_trade.get("pnl")
    try:
        pnl_val = float(pnl_raw)
    except (TypeError, ValueError):
        pnl_val = None

    if pnl_val is None or pnl_val >= 0:
        if trade_key is not None:
            setattr(state, "_shock_guard_last_trade_id", trade_key)
        return False

    threshold, pause_hours, template = _extract_shock_config(settings)
    if threshold <= 0 or pause_hours <= 0:
        if trade_key is not None:
            setattr(state, "_shock_guard_last_trade_id", trade_key)
        return False

    symbol = None
    if isinstance(settings, Mapping):
        symbol = settings.get("symbol")
    if not symbol:
        symbol = getattr(state, "config", {}).get("symbol") if hasattr(state, "config") else None
    if not symbol:
        symbol = "BTC/USDT"

    symbol_candidates = [symbol]
    if isinstance(symbol, str) and symbol.endswith("/USDT") and ":USDT" not in symbol:
        symbol_candidates.append(f"{symbol}:USDT")

    clients: list[Any] = []
    for attr in ("client", "public_client"):
        cli = getattr(market, attr, None)
        if cli is not None and cli not in clients:
            clients.append(cli)

    if not clients:
        try:  # pragma: no cover - fallback
            from trading import PUBLIC_CCXT_CLIENT  # lazy import para evitar ciclos

            if PUBLIC_CCXT_CLIENT is not None:
                clients.append(PUBLIC_CCXT_CLIENT)
        except Exception:
            pass

    pct = 0.0
    range_pct = 0.0
    for cli in clients:
        for sym in symbol_candidates:
            pct = abs(get_24h_change_pct_ccxt(cli, sym))
            range_pct = get_24h_range_pct_ccxt(cli, sym)
            if pct or range_pct:
                break
        if pct or range_pct:
            break

    if max(pct, range_pct) < threshold:
        if trade_key is not None:
            setattr(state, "_shock_guard_last_trade_id", trade_key)
        return False

    until = set_pause_hours(pause_hours)
    message = _format_shock_message(
        template,
        hours=pause_hours,
        delta_pct=pct,
        range_pct=range_pct,
        until=until,
    )
    logging.warning(
        "[RISK] Shock gate activado (trade_id=%s, pnl=%.2f, Δ24h=%.2f%%, rango24h=%.2f%%) -> pausa hasta %s",
        trade_id,
        pnl_val,
        pct,
        range_pct,
        until,
    )
    notify(message)
    _notify_async(notifier, message)

    if trade_key is not None:
        setattr(state, "_shock_guard_last_trade_id", trade_key)
    return True
