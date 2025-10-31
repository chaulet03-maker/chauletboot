import os
import math
import logging
import asyncio
import math
import sqlite3
import time
import re
import inspect
import signal
from collections import deque
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from logging_setup import LOG_DIR, LOG_FILE
from time_fmt import fmt_ar
from config import S, MANUAL_OPEN_RISK_PCT
from bot.motives import MOTIVES
from bot.mode_manager import get_mode, _read_cfg as _read_cfg_yaml, _write_cfg as _write_cfg_yaml
from bot.identity import get_bot_id
from bot.ledger import bot_position
from bot.pnl import pnl_summary_bot
from bot.runtime_state import (
    get_equity_sim,
    get_protection_defaults,
    set_equity_sim,
    update_protection_defaults,
)
from bot.settings_utils import get_val, read_config_raw
from bot.telemetry.command_registry import CommandRegistry, normalize
from bot.telemetry.formatter import open_msg
from state_store import load_state, update_open_position
import trading
from position_service import fetch_live_equity_usdm

logger = logging.getLogger("telegram")

REGISTRY = CommandRegistry()


CLOSE_TEXT_RE_PATTERN = r"(?i)^(cerrar(?: posicion| posición)?|close)$"
OPEN_TEXT_RE_PATTERN = r"(?i)^open\s+(long|short)\s+x(\d+)$"
POSITION_TEXT_RE_PATTERN = r"(?i)^(posicion|posición|position)$"

CLOSE_TEXT_RE = re.compile(CLOSE_TEXT_RE_PATTERN)
OPEN_TEXT_RE = re.compile(OPEN_TEXT_RE_PATTERN)
POSITION_TEXT_RE = re.compile(POSITION_TEXT_RE_PATTERN)

# === util local para formateo ===


def _fmt_money(v: float) -> str:
    try:
        return f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(v)


def _fmt_sign_money(v: float) -> str:
    if v is None:
        return "0,00"
    s = "➕" if v >= 0 else "➖"
    return f"{s} ${_fmt_money(abs(v))}"


async def open_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(
        "Uso manual: `open long x10` o `open short x5`."
        " El bot calculará el tamaño usando MANUAL_OPEN_RISK_PCT.",
        parse_mode="Markdown",
    )


async def rendimiento_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await equity_command(update, context)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(
        "Hola! Escribí *ayuda* para ver los comandos disponibles.",
        parse_mode="Markdown",
    )


def _get_app_from_context(context: ContextTypes.DEFAULT_TYPE):
    application = context.application if context else None
    if application is None:
        return None
    app = application.user_data.get("__app__")
    if app is None:
        app = application.bot_data.get("engine")
    return app


async def estado_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app = _get_app_from_context(context)
    message = update.effective_message
    if app is None or message is None:
        return

    sym = app.config.get("symbol", "BTC/USDT") if getattr(app, "config", None) else "BTC/USDT"
    mode_txt = "REAL" if getattr(app, "is_live", False) else "SIMULADO"

    try:
        bal = await app.exchange.fetch_balance_usdt()
    except Exception:
        bal = 0.0

    try:
        fr = await app.exchange.fetch_current_funding_rate(sym)
    except Exception:
        fr = None

    day_stats = week_stats = None
    if hasattr(app, "get_period_stats"):
        try:
            day_stats = app.get_period_stats(days=1)
        except Exception:
            day_stats = None
        try:
            week_stats = app.get_period_stats(days=7)
        except Exception:
            week_stats = None

    def _fmt_stats(stats):
        if not stats:
            return "—"
        eq0 = float(stats.get("equity_ini") or 0.0)
        pnl = float(stats.get("pnl") or 0.0)
        pct = (pnl / eq0 * 100.0) if eq0 else 0.0
        sign = "+" if pnl >= 0 else ""
        return f"{sign}{pnl:,.2f} ({sign}{pct:.2f}%)"

    text = (
        "<b>🩺 Diagnóstico</b>\n"
        f"• Modo: <b>{mode_txt}</b>\n"
        f"• Símbolo: {sym.replace('/', '')}\n"
        f"• CCXT: {'AUTENTICADO' if getattr(app, 'is_live', False) else 'PÚBLICO'}\n"
        f"• Funding rate: {fr if fr is not None else '—'}\n"
        f"• Saldo: {bal:,.2f}\n"
        f"• PnL 24h: {_fmt_stats(day_stats)}\n"
        f"• PnL 7d:  {_fmt_stats(week_stats)}"
    )

    await message.reply_html(text)


async def posicion_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app = _get_app_from_context(context)
    message = update.effective_message
    if app is None or message is None:
        return

    try:
        pos = await app.exchange.get_open_position(app.config.get("symbol"))
    except Exception:
        pos = None

    if not pos:
        await message.reply_text("El BOT no tiene posición abierta.")
        return

    side = str(pos.get("side", "")).upper()
    qty = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("size") or 0.0)
    entry = float(pos.get("entryPrice") or pos.get("avgPrice") or 0.0)
    mark = float(pos.get("markPrice") or pos.get("mark") or entry)
    pnl = (mark - entry) * qty * (1.0 if side == "LONG" else -1.0)

    await message.reply_html(
        "<b>📌 Posición del BOT</b>\n"
        f"• {side} {qty:.6f}\n"
        f"• Entrada: {entry:,.2f} | Mark: {mark:,.2f}\n"
        f"• PnL actual: {pnl:+,.2f}"
    )


async def posiciones_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app = _get_app_from_context(context)
    message = update.effective_message
    if app is None or message is None:
        return

    symbol_cfg = app.config.get("symbol", "BTC/USDT") if getattr(app, "config", None) else "BTC/USDT"

    try:
        managed = await app.exchange.get_open_position(symbol_cfg)
    except Exception:
        managed = None

    try:
        allpos = await app.exchange.list_open_positions()
    except Exception:
        allpos = []

    others: List[Dict[str, Any]] = []
    if managed:
        target_symbol = str(managed.get("symbol") or symbol_cfg)
        target_qty = float(managed.get("contracts") or 0.0)
        for p in allpos or []:
            sym = str(p.get("symbol") or "")
            qty = float(p.get("contracts") or p.get("positionAmt") or p.get("size") or 0.0)
            if sym == target_symbol and abs(qty - target_qty) <= 1e-9:
                continue
            others.append(p)
    else:
        others = list(allpos or [])

    parts: List[str] = []
    if managed:
        s = managed
        side = str(s.get("side", "")).upper()
        qty = float(s.get("contracts") or 0.0)
        ep = float(s.get("entryPrice") or 0.0)
        mp = float(s.get("markPrice") or ep)
        pnl = (mp - ep) * qty * (1.0 if side == "LONG" else -1.0)
        parts.append(
            "<b>🤖 BOT</b>\n"
            f"• {s.get('symbol', symbol_cfg)} {side} {qty:.6f}\n"
            f"• Entrada: {ep:,.2f} | Mark: {mp:,.2f}\n"
            f"• PnL: {pnl:+,.2f}"
        )
    for s in others:
        side = str(s.get("side", "")).upper()
        qty = float(s.get("contracts") or s.get("positionAmt") or s.get("size") or 0.0)
        ep = float(s.get("entryPrice") or s.get("avgPrice") or 0.0)
        mp = float(s.get("markPrice") or s.get("mark") or ep)
        pnl = (mp - ep) * qty * (1.0 if side == "LONG" else -1.0)
        parts.append(
            "<b>👤 Manual/Otras</b>\n"
            f"• {s.get('symbol', '')} {side} {qty:.6f}\n"
            f"• Entrada: {ep:,.2f} | Mark: {mp:,.2f}\n"
            f"• PnL: {pnl:+,.2f}"
        )

    await message.reply_html("\n\n".join(parts) if parts else "No hay posiciones abiertas.")


# Valores por defecto para TP/SL expresados como % del precio de entrada.
DEFAULT_SL_PRICE_PCT = -10.0
DEFAULT_TP_PRICE_PCT = 10.0


# ===== Helpers de modo seguros (NO tocar is_live) =====


def _get_mode_from_engine(engine) -> str:
    # lee modo actual sin tocar nada
    try:
        if hasattr(engine, "mode") and isinstance(engine.mode, str):
            return "live" if engine.mode.lower() in ("live", "real") else "paper"
        if hasattr(engine, "is_live"):
            return "live" if bool(engine.is_live) else "paper"
    except Exception:
        pass
    return "paper"


def _is_engine_live(engine) -> bool:
    try:
        return _get_mode_from_engine(engine) == "live"
    except Exception:
        return False


def _format_position_block(
    *,
    symbol: str,
    side: str,
    qty: float,
    entry: float,
    mark: float,
    pnl: float,
    mode_txt: str,
    opened_at: float | None = None,
    is_bot: bool = False,
) -> str:
    """Bloque idéntico al de /posicion."""
    title = "📍 *Posición del BOT*" if is_bot else "📍 *Posición*"
    opened_line = ""
    if opened_at:
        try:
            from datetime import datetime, timezone

            opened_local = fmt_ar(datetime.fromtimestamp(float(opened_at), tz=timezone.utc))
            opened_line = f"• apertura: {opened_local}\n\n"
        except Exception:
            opened_line = ""
    return (
        f"{title}\n"
        f"{opened_line}"
        f"• Símbolo: *{symbol}* ({mode_txt.title()})\n"
        f"• Lado: *{(side or '').upper()}*\n"
        f"• Cantidad (bot qty): *{_num(qty, 4)}*\n"
        f"• Entrada: {_num(entry)}  |  Mark: {_num(mark)}\n"
        f"• PnL: *{_num(pnl)}*"
    )


def _signed_qty(qty: float, side: str) -> float:
    try:
        q = float(qty)
    except Exception:
        q = 0.0
    s = (side or "").upper()
    if q == 0:
        return 0.0
    if s in ("SHORT", "SELL") and q > 0:
        return -q
    if s in ("LONG", "BUY") and q < 0:
        return -q
    return q


# ==== Helpers de TP/SL por % del precio y métricas de presentación ====


def _side_from_qty(qty: float) -> str:
    return "LONG" if qty > 0 else ("SHORT" if qty < 0 else "FLAT")


def _normalize_side_name(side: str) -> str:
    """Normaliza representaciones de lado a LONG/SHORT/FLAT."""
    s = (side or "").strip().upper()
    if s in ("", "FLAT", "NEUTRAL", "NONE"):
        return "FLAT"
    if s in ("LONG", "BUY", "BULL", "L", "LARGO"):
        return "LONG"
    if s in ("SHORT", "SELL", "BEAR", "S", "CORTO"):
        return "SHORT"
    if s.startswith("L"):
        return "LONG"
    if s.startswith("S"):
        return "SHORT"
    return "LONG"


def _normalize_percent_value(value: Any, *, prefer_sign: int | None = None) -> Optional[float]:
    """Convierte valores que pueden venir como 0.10 ó 10 en porcentajes (10.0)."""
    if value in (None, ""):
        return None
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pct):
        return None
    if abs(pct) <= 1.0:
        pct *= 100.0
    if prefer_sign == -1:
        pct = -abs(pct)
    elif prefer_sign == 1:
        pct = abs(pct)
    return pct


def _target_from_price_pct(entry: float, side: str, pct: float) -> float:
    """Devuelve un precio objetivo aplicando un % sobre el precio de entrada."""
    if entry <= 0:
        return entry
    try:
        pct_val = float(pct)
    except (TypeError, ValueError):
        return entry
    if not math.isfinite(pct_val):
        return entry
    side_norm = _normalize_side_name(side)
    adj_pct = pct_val
    if side_norm == "SHORT":
        adj_pct = -adj_pct
    target = entry * (1 + adj_pct / 100.0)
    if target <= 0:
        target = entry * 0.1
    return target


def _pcts_for_target(entry: float, target: float, qty_abs: float, equity: float, side: str):
    """
    Devuelve:
      price_pct: % de movimiento de precio desde entry (con signo de 'beneficio' para ese side)
      pnl_pct_equity: % de impacto esperado en PnL sobre el equity (con signo)
    """
    if entry <= 0 or qty_abs <= 0:
        return 0.0, 0.0
    side_norm = _normalize_side_name(side)
    dir_sign = 1.0 if side_norm == "LONG" else -1.0
    price_pct = dir_sign * ((target - entry) / entry) * 100.0
    pnl_usd = dir_sign * (target - entry) * qty_abs
    pnl_pct_equity = (pnl_usd / equity) * 100.0 if equity > 0 else 0.0
    return price_pct, pnl_pct_equity


def _first_float_optional(*values):
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _build_bot_position_message(*, engine, symbol, qty, avg, mark_val) -> str:
    try:
        qty_val = float(qty)
    except Exception:
        qty_val = 0.0
    try:
        avg_val = float(avg)
    except Exception:
        avg_val = 0.0
    try:
        mark_value = float(mark_val)
    except Exception:
        mark_value = 0.0

    side = _side_from_qty(qty_val)
    lev = 1
    tp_price = None
    sl_price = None
    pos_state = None

    try:
        st = load_state() or {}
        sym_key1 = str(symbol).replace("/", "")
        sym_key2 = str(symbol)
        open_positions = st.get("open_positions", {}) or {}
        pos_state = open_positions.get(sym_key1) or open_positions.get(sym_key2)
        if isinstance(pos_state, dict):
            lev_raw = pos_state.get("leverage") or 1

            def _pick(d: dict | None, *keys):
                if not isinstance(d, dict):
                    return None
                for k in keys:
                    if k in d and d[k] not in (None, ""):
                        try:
                            return float(d[k])
                        except Exception:
                            continue
                return None

            try:
                lev_val = float(lev_raw)
            except Exception:
                lev_val = 1.0
            if lev_val > 0:
                lev = max(int(lev_val), 1)
            tp_price = _pick(
                pos_state,
                "tp",
                "tp_price",
                "take_profit",
                "tp2",
                "tp1",
            )
            sl_price = _pick(pos_state, "sl", "sl_price", "stop_loss", "stop")

            if tp_price is None and isinstance(pos_state, dict):
                alt_levels = None
                for key in ("tp_targets", "take_profits", "tps"):
                    if key in pos_state:
                        alt_levels = pos_state.get(key)
                        break
                if isinstance(alt_levels, (list, tuple)):
                    numeric_levels = []
                    for candidate in alt_levels:
                        try:
                            numeric_levels.append(float(candidate))
                        except Exception:
                            continue
                    if numeric_levels:
                        if side == "SHORT":
                            tp_price = min(numeric_levels)
                        else:
                            tp_price = max(numeric_levels)
    except Exception:
        pass

    try:
        defaults = get_protection_defaults(symbol) or {}
    except Exception:
        defaults = {}

    cfg = getattr(engine, "config", {}) or {}

    trader = getattr(engine, "trader", None)
    try:
        equity = float(trader.equity()) if trader is not None else 0.0
    except Exception:
        equity = 0.0

    qty_abs = abs(qty_val)
    pnl_usd = (mark_value - avg_val) * qty_val
    notional = qty_abs * avg_val
    margin_used = notional / max(lev, 1)
    if equity > 0:
        pnl_pct = (pnl_usd / equity) * 100.0
        pnl_pct_label = "equity"
    else:
        pnl_pct = 0.0 if margin_used <= 0 else (pnl_usd / margin_used) * 100.0
        pnl_pct_label = "margen"

    lev_txt = f" | lev x{lev}" if lev > 1 else ""

    def _line_tp_sl(name: str, target_val) -> str:
        if target_val in (None, ""):
            return f"• {name}: —"
        try:
            target = float(target_val)
        except (TypeError, ValueError):
            return f"• {name}: —"
        _, pnl_pct_equity = _pcts_for_target(avg_val, target, qty_abs, equity, side)
        price_move_pct = _pct_rel(avg_val, target)
        return (
            f"• {name}: {target:,.2f} "
            f"({price_move_pct:+.2f}% vs entrada | PnL {pnl_pct_equity:+.2f}%)"
        )

    entry_for_tp = avg_val
    if entry_for_tp <= 0 and isinstance(pos_state, dict):
        entry_for_tp = _first_float_optional((pos_state or {}).get("entry_price"), avg_val) or 0.0

    if tp_price is None and entry_for_tp > 0:
        tp_pct_candidates = []
        if isinstance(pos_state, dict):
            tp_pct_candidates.append(pos_state.get("tp_pct"))
        tp_pct_candidates.append(cfg.get("tp_pct"))
        strategy_cfg = cfg.get("strategy") if isinstance(cfg, dict) else None
        if isinstance(strategy_cfg, dict):
            tp_pct_candidates.append(strategy_cfg.get("tp_pct"))
        tp_pct_candidates.append(getattr(S, "tp_pct", None))
        try:
            raw_cfg = read_config_raw()
        except Exception:
            raw_cfg = None
        if isinstance(raw_cfg, dict):
            tp_pct_candidates.append(raw_cfg.get("tp_pct"))
            strat_cfg = raw_cfg.get("strategy")
            if isinstance(strat_cfg, dict):
                tp_pct_candidates.append(strat_cfg.get("tp_pct"))

        tp_pct_val = _first_float_optional(*tp_pct_candidates)
        if tp_pct_val is not None:
            tp_pct_f = abs(tp_pct_val)
            if tp_pct_f > 1:
                tp_pct_f = min(tp_pct_f, 100.0) / 100.0
            if tp_pct_f > 0:
                if side == "SHORT":
                    tp_price = entry_for_tp * (1 - tp_pct_f)
                else:
                    tp_price = entry_for_tp * (1 + tp_pct_f)

    if tp_price is None and entry_for_tp > 0:
        lev_key = str(int(lev)) if lev else "1"
        tp_pct_by_lev = {}
        if isinstance(cfg, dict):
            raw_map = cfg.get("tp_eq_pct_by_leverage")
            if isinstance(raw_map, dict):
                tp_pct_by_lev = raw_map
        kind = str(defaults.get("tp_last_kind") or "pct").lower()
        if kind == "price":
            tp_price = _first_float_optional(defaults.get("tp_price"))
        else:
            tp_pct = _first_float_optional(
                defaults.get("tp_pct_equity"),
                defaults.get("tp_pct"),
                tp_pct_by_lev.get(lev_key) if tp_pct_by_lev else None,
            )
            if tp_pct is not None:
                tp_price = _target_from_price_pct(entry_for_tp, side, float(tp_pct))

    entry_for_sl = avg_val
    if entry_for_sl <= 0 and isinstance(pos_state, dict):
        entry_for_sl = _first_float_optional((pos_state or {}).get("entry_price"), avg_val) or 0.0

    if sl_price is None and entry_for_sl > 0:
        kind = str(defaults.get("sl_last_kind") or "pct").lower()
        if kind == "price":
            sl_price = _first_float_optional(defaults.get("sl_price"))
        else:
            sl_pct = _first_float_optional(
                defaults.get("sl_pct_equity"),
                defaults.get("sl_pct"),
            )
            if sl_pct is not None:
                sl_price = _target_from_price_pct(entry_for_sl, side, float(sl_pct))

    tp_line = _line_tp_sl("TP", tp_price)
    sl_line = _line_tp_sl("SL", sl_price)

    mode_txt = "Simulado" if not _is_engine_live(engine) else "Real"

    msg = (
        f"📍 *Posición del BOT*\n"
        f"• Símbolo: {symbol} ({mode_txt})\n"
        f"• Lado: *{side}*\n"
        f"• Cantidad (bot qty): {qty_abs:.6f}\n"
        f"• Entrada: {avg_val:,.2f} | Mark: {mark_value:,.2f}{lev_txt}\n"
        f"• PnL: {pnl_usd:+.2f} (*{pnl_pct:+.2f}% {pnl_pct_label}*)\n"
        f"{tp_line}\n"
        f"{sl_line}"
    )

    return msg


async def _on_error(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    error = getattr(context, "error", None)
    if error:
        logger.error(
            "Telegram handler error: %s",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
    else:
        logger.exception("Telegram handler error")

    chat = None
    if update is not None:
        chat = getattr(update, "effective_chat", None)
    if chat is None:
        return
    try:
        await chat.send_message("Ocurrió un error procesando el comando. Revisá los logs.")
    except Exception:
        pass


class TelegramLoggingRequest(HTTPXRequest):
    def __init__(
        self,
        *args,
        latency_warning_s: float = 1.0,
        summary_interval_s: float = 60.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._latency_warning_s = max(latency_warning_s, 0.0)
        self._summary_interval_s = max(summary_interval_s, 0.0)
        self._log = logging.getLogger("telegram.http")
        self._ok_count = 0
        self._err_count = 0
        self._summary_started = time.monotonic()

    @staticmethod
    def _method_from_url(url: str) -> str:
        if not url:
            return ""
        return url.rstrip("/").split("/")[-1]

    def _bump_summary(self, ok: bool) -> None:
        if ok:
            self._ok_count += 1
        else:
            self._err_count += 1
        if self._summary_interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._summary_started >= self._summary_interval_s:
            self._log.info("tg_api/min ok=%d err=%d", self._ok_count, self._err_count)
            self._ok_count = 0
            self._err_count = 0
            self._summary_started = now

    def _log_response(self, method: str, status: int, elapsed_s: float) -> None:
        latency_ms = elapsed_s * 1000
        if status == 200 and elapsed_s <= self._latency_warning_s:
            self._log.debug(
                "tg_api method=%s status=%s latency_ms=%.0f", method, status, latency_ms
            )
            return

        level = logging.WARNING
        if status == 429 or status >= 500:
            level = logging.ERROR

        self._log.log(
            level,
            "tg_api method=%s status=%s latency_ms=%.0f",
            method,
            status,
            latency_ms,
        )

    async def do_request(self, url: str, method: str, *args, **kwargs):  # type: ignore[override]
        tg_method = self._method_from_url(url)
        start = time.perf_counter()
        try:
            status, payload = await super().do_request(url, method, *args, **kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._log.error(
                "tg_api method=%s error=%s latency_ms=%.0f",
                tg_method,
                exc.__class__.__name__,
                elapsed * 1000,
            )
            self._bump_summary(False)
            raise
        elapsed = time.perf_counter() - start
        self._log_response(tg_method, status, elapsed)
        self._bump_summary(status == 200)
        return status, payload


# =========================
# Utils de formato
# =========================
def _env(key, default=None):
    v = os.getenv(key, default)
    return v


def _first_float(*candidates, default: float) -> float:
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default

def _fmt_num(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)


def _num(val, decimals=2):
    try:
        return f"{float(val):,.{decimals}f}"
    except Exception:
        return str(val)


def _parse_fraction(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        raw = value.strip().rstrip("%") if isinstance(value, str) else value
        frac = float(raw)
    except (AttributeError, TypeError, ValueError):
        return None
    if frac <= 0:
        return None
    if frac > 1:
        frac = min(frac, 100.0) / 100.0
    return min(frac, 1.0)


def _parse_tp_pct(value: Any) -> float:
    """Parsea valores de porcentaje flexibles como "10", "10%" o "0.8%"."""

    if value is None:
        raise ValueError("valor vacío")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("valor vacío")
        has_pct = text.endswith("%")
        if has_pct:
            text = text[:-1].strip()
        text = text.replace(",", ".")
        if not text:
            raise ValueError("valor vacío")
        try:
            numeric = float(text)
        except ValueError as exc:  # pragma: no cover - defensivo
            raise ValueError("no es un número") from exc
        if not math.isfinite(numeric):
            raise ValueError("no es un número")
        if has_pct:
            numeric /= 100.0
        elif abs(numeric) > 1.0:
            numeric /= 100.0
        return numeric

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("no es un número") from exc
    if not math.isfinite(numeric):
        raise ValueError("no es un número")
    if abs(numeric) > 1.0:
        numeric /= 100.0
    return numeric


def _try_parse_tp_pct(value: Any, *, allow_zero: bool = False) -> Optional[float]:
    try:
        pct = _parse_tp_pct(value)
    except ValueError:
        return None
    if pct < 0:
        return None
    if pct == 0 and not allow_zero:
        return None
    return pct


def _collect_admin_ids(cfg: Dict[str, Any]) -> set[int]:
    ids: set[int] = set()

    def _add(raw_item: Any) -> None:
        if raw_item is None:
            return
        if isinstance(raw_item, (list, tuple, set)):
            for item in raw_item:
                _add(item)
            return
        text = str(raw_item).strip()
        if not text:
            return
        parts = re.split(r"[\s,]+", text)
        for part in parts:
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError:
                continue

    if isinstance(cfg, dict):
        _add(cfg.get("telegram_admin_ids"))
        _add(cfg.get("telegram_admins"))
        telegram_cfg = cfg.get("telegram") if isinstance(cfg.get("telegram"), dict) else {}
        if isinstance(telegram_cfg, dict):
            _add(telegram_cfg.get("admin_ids"))
            _add(telegram_cfg.get("admins"))

    env_admins = os.getenv("TELEGRAM_ADMIN_IDS")
    if env_admins:
        _add(env_admins)

    return ids


def _user_is_admin(update: Update, cfg: Dict[str, Any]) -> bool:
    admin_ids = _collect_admin_ids(cfg)
    if not admin_ids:
        return True
    user = update.effective_user if update else None
    if user is None or getattr(user, "id", None) is None:
        return False
    try:
        return int(user.id) in admin_ids
    except Exception:
        return False


def _persist_tp_mapping(lev_key: str, pct: float) -> None:
    try:
        raw_cfg = _read_cfg_yaml() or {}
    except Exception:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    mapping = raw_cfg.get("tp_eq_pct_by_leverage")
    if not isinstance(mapping, dict):
        mapping = {}
    mapping[str(lev_key)] = float(round(pct, 6))
    raw_cfg["tp_eq_pct_by_leverage"] = mapping
    try:
        _write_cfg_yaml(raw_cfg)
    except Exception:
        logger.warning("No se pudo persistir tp_eq_pct_by_leverage en config", exc_info=True)


def _get_equity_fraction(engine) -> float:
    # PRIORIDAD: ENV -> engine.config -> default
    env = _parse_fraction(os.getenv("EQUITY_PCT"))
    if env is not None:
        return env
    cfg = getattr(engine, "config", {}) or {}
    frac = cfg.get("order_sizing", {}).get("default_pct")
    parsed = _parse_fraction(frac)
    if parsed is not None:
        return parsed
    return 1.0

def _pct_rel(entry: float, level: float) -> float:
    try:
        entry = float(entry); level = float(level)
        if entry == 0:
            return 0.0
        return (level / entry - 1.0) * 100.0
    except Exception:
        return 0.0

def _fmt_price_with_pct(level: float, entry: float) -> str:
    """'62.890,50 (+0,86%)' (dos decimales y % relativo al entry)."""
    lvl = _fmt_num(level, 2)
    pct = _pct_rel(entry, level)
    sign = "+" if pct >= 0 else ""
    return f"{lvl} ({sign}{pct:.2f}%)"

def _cfg_csv_paths(cfg: dict):
    p = cfg.get("persistence", {}) if isinstance(cfg, dict) else {}
    base = p.get("dir", "data")
    equity = p.get("equity_csv", os.path.join(base, "equity.csv"))
    trades = p.get("trades_csv", os.path.join(base, "trades.csv"))
    return equity, trades

def _tz():
    return ZoneInfo("America/Argentina/Buenos_Aires")


def _app_config(app):
    if app is None:
        return {}
    if isinstance(getattr(app, "config", None), dict):
        return app.config
    if isinstance(getattr(app, "cfg", None), dict):
        return app.cfg
    return {}


def _engine_config(engine) -> Dict:
    cfg = _app_config(engine)
    if cfg:
        return cfg
    return {}


def _engine_sqlite_path(engine) -> str:
    cfg = _engine_config(engine)
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    path = getattr(engine, "sqlite_path", None) or storage.get("sqlite_path")
    if not path:
        base = storage.get("csv_dir", "data")
        path = os.path.join(base, "bot.sqlite")
    return path


def _engine_logs_path(engine) -> str:
    cfg = _engine_config(engine)
    persistence = cfg.get("persistence", {}) if isinstance(cfg, dict) else {}
    path = persistence.get("logs_file")
    if path:
        path = os.path.expanduser(str(path))
        if not os.path.isabs(path):
            return os.path.join(LOG_DIR, os.path.basename(path))
        return path
    return LOG_FILE


def _tail_log_file(path: str, limit: int) -> List[str]:
    with open(path, "rb") as fh:
        return [
            line.decode("utf-8", errors="ignore").rstrip("\n")
            for line in deque(fh, maxlen=limit)
        ]


def _tail_memory_logs(limit: int) -> List[str]:
    root_logger = logging.getLogger()
    buffer = getattr(root_logger, "_memh", None)
    if buffer and getattr(buffer, "buf", None):
        return list(buffer.buf)[-limit:]
    return []


def _format_local_timestamp(value) -> str:
    if isinstance(value, datetime):
        return fmt_ar(value)
    if isinstance(value, (int, float)):
        return fmt_ar(datetime.fromtimestamp(value, tz=timezone.utc))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
            return fmt_ar(datetime.fromisoformat(normalized))
        except Exception:
            return text
    return str(value)


def _chunk_text(text: str, max_len: int = 3800) -> List[str]:
    if not text:
        return [""]
    lines = text.split("\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in lines:
        line_with_newline = line if not current else "\n" + line
        if current_len + len(line_with_newline) > max_len:
            chunks.append("".join(current))
            current = [line]
            current_len = len(line)
        else:
            if not current:
                current.append(line)
                current_len = len(line)
            else:
                current.append("\n" + line)
                current_len += len(line) + 1
    if current:
        chunks.append("".join(current))
    return chunks or [text]


async def _reply_chunks(
    update: Update,
    text: str,
    chunk_size: int = 3800,
    delay: float = 0.0,
    **reply_kwargs,
):
    message = update.effective_message
    if message is None:
        return
    chunks = _chunk_text(text, max_len=chunk_size)
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        await message.reply_text(chunk, **reply_kwargs)
        if delay and idx + 1 < total:
            await asyncio.sleep(delay)


async def ayuda_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Genera dinámicamente la lista de comandos disponibles."""

    lines = ["📋 *Lista de Comandos*"]
    for name, desc in REGISTRY.help_lines():
        lines.append(f"- *{name}*: {desc}")
    text = "\n".join(lines)
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(text, parse_mode="Markdown")


def _build_rendimiento_text(engine) -> str:
    path = _engine_sqlite_path(engine)
    if not os.path.exists(path):
        return f"No encontré la base de trades ({path}). Todavía no hay operaciones registradas."
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*),
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END),
                    COALESCE(SUM(pnl), 0),
                    COALESCE(SUM(fee), 0)
            FROM trades
            WHERE ABS(pnl) > 1e-9
            """
        )
        row = cur.fetchone() or (0, 0, 0, 0.0, 0.0)
        total, wins, losses, pnl_total, fees_total = row
        cur.execute("SELECT MAX(pnl), MIN(pnl) FROM trades WHERE ABS(pnl) > 1e-9")
        best, worst = cur.fetchone() or (None, None)
        conn.close()
    except Exception as exc:
        return f"No pude leer la base de datos de trades ({path}): {exc}"

    total = int(total or 0)
    wins = int(wins or 0)
    losses = int(losses or 0)
    pnl_total = float(pnl_total or 0.0)
    fees_total = float(fees_total or 0.0)
    net_total = pnl_total - fees_total
    if total <= 0:
        return "Todavía no hay trades cerrados en la base de datos."
    winrate = (wins / total) * 100.0 if total else 0.0
    avg_pnl = pnl_total / total if total else 0.0
    lines = [
        "📈 Rendimiento acumulado",
        f"Operaciones cerradas: {total} (Ganadas: {wins} / Perdidas: {losses})",
        f"Winrate: {winrate:.2f}%",
        f"PnL bruto: ${_fmt_num(pnl_total, 2)}",
        f"Fees totales: ${_fmt_num(fees_total, 2)}",
        f"PnL neto: ${_fmt_num(net_total, 2)}",
        f"PnL promedio por trade: ${_fmt_num(avg_pnl, 2)}",
    ]
    if best is not None:
        lines.append(f"Mejor trade: ${_fmt_num(best, 2)} | Peor trade: ${_fmt_num(worst or 0.0, 2)}")
    return "\n".join(lines)


def _fmt_config_num(value, digits=2, suffix=""):
    try:
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "Sí" if value else "No"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value) if value is not None else "N/A"


async def _cmd_config(engine, reply):
    cfg = read_config_raw()
    modo = get_mode()
    timeframe = get_val(S, cfg, "tf", "timeframe", default="1h")

    size_mode = get_val(S, cfg, "size_mode")
    sl_atr_mult = get_val(S, cfg, "sl_atr_mult")
    max_hold_bars = get_val(S, cfg, "max_hold_bars")
    daily_stop_R = get_val(S, cfg, "daily_stop_R", "daily_stop_r")
    emerg_trade_stop_R = get_val(S, cfg, "emerg_trade_stop_R", "emerg_trade_stop_r")
    trail_to_be = get_val(S, cfg, "trail_to_be")

    entry_mode = get_val(S, cfg, "entry_mode")
    rsi_gate = get_val(S, cfg, "rsi_gate")
    target_eq_pnl_pct = get_val(S, cfg, "target_eq_pnl_pct")
    ema200_1h_confirm = get_val(S, cfg, "ema200_1h_confirm")
    ema200_4h_confirm = get_val(S, cfg, "ema200_4h_confirm")

    leverage_base = get_val(S, cfg, "leverage_base")
    leverage_strong = get_val(S, cfg, "leverage_strong")
    adx_strong_threshold = get_val(S, cfg, "adx_strong_threshold")

    pct_min = get_val(S, cfg, "order_pct_min")
    pct_def = get_val(S, cfg, "order_pct_default")
    pct_max = get_val(S, cfg, "order_pct_max")

    slippage_bps = get_val(S, cfg, "slippage_bps")
    leverage_set_last = getattr(engine, "last_leverage", None) if engine else None

    equity_line = "N/A"
    try:
        if modo == "simulado":
            store = (
                getattr(engine, "paper_store", None)
                or getattr(engine, "STORE", None)
                or getattr(trading.POSITION_SERVICE, "paper_store", None)
            )
            if store and getattr(store, "state", None):
                eq = store.state.get("equity")
                if eq is not None:
                    equity_line = f"{_fmt_config_num(eq, 2)} USDT"
        else:
            if hasattr(trading.POSITION_SERVICE, "get_balance"):
                bal = trading.POSITION_SERVICE.get_balance()
                if isinstance(bal, dict):
                    eq = bal.get("USDT") or bal.get("total") or bal.get("free")
                    if eq is not None:
                        equity_line = f"{_fmt_config_num(eq, 2)} USDT"
    except Exception:
        pass

    target_pct = target_eq_pnl_pct
    if target_eq_pnl_pct is not None:
        try:
            if target_eq_pnl_pct < 1:
                target_pct = target_eq_pnl_pct * 100
            else:
                target_pct = target_eq_pnl_pct
        except Exception:
            target_pct = target_eq_pnl_pct

    text = [
        "🛠️ *Configuración actual*",
        f"Modo: *{modo}* | Timeframe: *{timeframe}*",
        "",
        "— *Riesgo* —",
        f"Size mode: {size_mode or 'desconocido'} | Max hold bars: {max_hold_bars or 'N/A'}",
        f"SL ATR mult: {_fmt_config_num(sl_atr_mult)} | Stop diario (R): {_fmt_config_num(daily_stop_R)} | Emerg(R): {_fmt_config_num(emerg_trade_stop_R)}",
        f"Trail to BE: {_fmt_config_num(trail_to_be, 0)}",
        "",
        "— *Estrategia* —",
        f"Entry mode: {entry_mode or 'N/A'} | RSI gate: {rsi_gate if rsi_gate is not None else 'N/A'}",
        f"Target EQ PnL %: {_fmt_config_num(target_pct, 2, '%')}",
        f"EMA200 1h confirm: {_fmt_config_num(ema200_1h_confirm, 0)} | EMA200 4h confirm: {_fmt_config_num(ema200_4h_confirm, 0)}",
        "",
        "— *Tamaños de orden* —",
        f"Pct min/default/max: {_fmt_config_num(pct_min, 2, '%')} / {_fmt_config_num(pct_def, 2, '%')} / {_fmt_config_num(pct_max, 2, '%')}",
        "",
        "— *Leverage* —",
        f"Base: {_fmt_config_num(leverage_base, 1, 'x')} | Strong: {_fmt_config_num(leverage_strong, 1, 'x')} | ADX fuerte ≥ {_fmt_config_num(adx_strong_threshold, 0)}",
        "",
        "— *Ejecución* —",
        f"Leverage set: {_fmt_config_num(leverage_set_last, 1, 'x')} | Slippage bps: {_fmt_config_num(slippage_bps, 0)}",
        "",
        f"Equity USDT: {equity_line}",
    ]

    return await reply("\n".join(text), parse_mode="Markdown")


def _read_logs_text(engine, limit: int = 15) -> str:
    try:
        n = max(int(limit or 15), 1)
    except (TypeError, ValueError):
        n = 15
    path = _engine_logs_path(engine)
    try:
        if os.path.exists(path):
            lines = _tail_log_file(path, n)
        else:
            lines = _tail_memory_logs(n)
        if not lines:
            return "📄 Últimos logs:\n(sin logs disponibles)"
        return "📄 Últimos logs:\n" + "\n".join(lines)
    except Exception as exc:
        return f"No pude leer los logs ({path}): {type(exc).__name__}: {exc}"


def _set_killswitch(engine, enabled: bool) -> bool:
    if engine is None:
        return False
    trader = getattr(engine, "trader", None)
    if trader and hasattr(trader, "set_killswitch"):
        try:
            trader.set_killswitch(bool(enabled))
        except Exception:
            trader.state.killswitch = bool(enabled)
    elif trader and getattr(trader, "state", None):
        trader.state.killswitch = bool(enabled)
    if hasattr(engine, "allow_new_entries"):
        engine.allow_new_entries = not bool(enabled)
    return bool(enabled)


def _parse_adjust_value(raw: str):
    txt = raw.strip()
    low = txt.lower()
    if low in ("true", "on", "si", "sí", "1", "enable", "enabled"):
        return True
    if low in ("false", "off", "no", "0", "disable", "disabled"):
        return False
    num_txt = txt
    if re.match(r"^[+-]?\d{1,3}(?:\.\d{3})*(?:,\d+)?$", txt):
        num_txt = txt.replace(".", "").replace(",", ".")
    elif txt.count(",") == 1 and txt.replace(",", "").replace("-", "").isdigit():
        num_txt = txt.replace(",", ".")
    try:
        if num_txt.isdigit() or (num_txt.startswith("-") and num_txt[1:].isdigit()):
            return int(num_txt)
        return float(num_txt)
    except Exception:
        return txt


def _set_config_value(engine, path: Sequence[str], value) -> bool:
    cfg = _engine_config(engine)
    target = cfg
    for key in path[:-1]:
        if not isinstance(target, dict) or key not in target:
            return False
        target = target[key]
    key = path[-1]
    if isinstance(target, dict):
        target[key] = value
    else:
        return False

    # Sincronizar con estructuras internas comunes
    if path[0] == "strategy" and hasattr(engine, "strategy_conf") and isinstance(engine.strategy_conf, dict):
        engine.strategy_conf[key] = value
    elif path[0] == "order_sizing" and hasattr(engine, "order_sizes") and isinstance(engine.order_sizes, dict):
        engine.order_sizes[key] = value
    elif path[0] == "leverage" and hasattr(engine, "leverage_conf") and isinstance(engine.leverage_conf, dict):
        engine.leverage_conf[key] = value
    elif path[0] == "risk" and hasattr(engine, "cfg") and isinstance(engine.cfg.get("risk", {}), dict):
        engine.cfg["risk"][key] = value
    elif len(path) == 1 and hasattr(engine, key):
        setattr(engine, key, value)

    if path[0] == "portfolio_caps" and hasattr(engine, "portfolio_caps"):
        caps = getattr(engine, "portfolio_caps")
        if hasattr(caps, key):
            setattr(caps, key, value)
    return True


def _find_and_set_config(engine, param: str, value):
    if "." in param:
        path = [p for p in param.split(".") if p]
        if path and _set_config_value(engine, path, value):
            return path
    cfg = _engine_config(engine)
    stack: List[Tuple[Dict, List[str]]] = []
    if isinstance(cfg, dict):
        stack.append((cfg, []))
    while stack:
        current, path = stack.pop()
        if param in current:
            current[param] = value
            full_path = path + [param]
            _set_config_value(engine, full_path, value)
            return full_path
        for key, val in current.items():
            if isinstance(val, dict):
                stack.append((val, path + [key]))
    return []


def _extract_logs_limit(update: Update, context: ContextTypes.DEFAULT_TYPE, default: int = 15) -> int:
    args = getattr(context, "args", None) or []
    limit = None
    if args:
        try:
            limit = int(args[0])
        except Exception:
            limit = None
    if limit is None:
        text = (update.effective_message.text or "") if update.effective_message else ""
        m = re.search(r"(\d+)", text)
        if m:
            try:
                limit = int(m.group(1))
            except Exception:
                limit = None
    if limit is None:
        limit = default
    return max(1, min(200, limit))


def _get_engine_from_context(context: ContextTypes.DEFAULT_TYPE):
    app = getattr(context, "application", None)
    if app is None:
        return None
    return app.bot_data.get("engine")


def _default_symbol(engine) -> str:
    try:
        cfg = getattr(engine, "config", None)
        if isinstance(cfg, dict):
            return cfg.get("symbol", "BTC/USDT")
    except Exception:
        pass
    return "BTC/USDT"


def _normalized_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _bot_position_info(engine) -> Optional[dict[str, Any]]:
    try:
        status = trading.POSITION_SERVICE.get_status() if trading.POSITION_SERVICE else None
    except Exception:
        status = None
    if not status:
        return None
    side = _normalize_side_name(str(status.get("side", "FLAT")))
    if side == "FLAT":
        return None
    qty_raw = status.get("qty") or status.get("size") or status.get("pos_qty") or 0.0
    try:
        qty = abs(float(qty_raw))
    except Exception:
        qty = 0.0
    if qty <= 0:
        return None
    try:
        entry = float(status.get("entry_price") or status.get("avg_price") or 0.0)
    except Exception:
        entry = 0.0
    try:
        mark = float(status.get("mark") or 0.0)
    except Exception:
        mark = 0.0
    try:
        leverage = float(status.get("leverage") or 0.0)
    except Exception:
        leverage = 0.0
    symbol_conf = status.get("symbol") or ((getattr(engine, "config", {}) or {}).get("symbol") or "BTC/USDT")
    symbol_norm = _normalized_symbol(str(symbol_conf))
    return {
        "side": side,
        "qty": qty,
        "entry": entry,
        "mark": mark,
        "symbol": symbol_norm,
        "symbol_conf": symbol_conf,
        "leverage": leverage,
    }


async def _handle_position_controls(engine, reply_md, text: str) -> bool:
    """
    Solo maneja seteo/consulta de *defaults* de SL/TP cuando NO hay posición del BOT.
    Si hay posición abierta, devolvemos False y dejan trabajar a sl_command/tp_command.
    """
    if not text:
        return False

    text_raw = (text or "").strip()
    lower = text_raw.lower()

    if lower.startswith("sl") or lower.startswith("/sl") or lower.startswith("tp") or lower.startswith("/tp"):
        return False

    # ¿Hay posición del BOT?
    position = _bot_position_info(engine)
    if position is not None:
        # Con posición abierta, usar los comandos dedicados (/sl y /tp)
        return False

    # Sin posición → permitir setear/ver DEFAULTS
    cfg_engine = getattr(engine, "config", {}) or {}
    symbol_conf = cfg_engine.get("symbol", "BTC/USDT")
    defaults = get_protection_defaults(symbol_conf)

    # ----- SL -----
    if re.match(r"^/?sl\s*$", text_raw, re.IGNORECASE):
        stored_kind = defaults.get("sl_last_kind")
        if stored_kind == "pct" and defaults.get("sl_pct_equity") not in (None, ""):
            pct_val = float(defaults["sl_pct_equity"])
            await reply_md(f"SL predeterminado: {pct_val:+.2f}% del precio")
        elif stored_kind == "price" and defaults.get("sl_price") not in (None, ""):
            await reply_md(f"SL predeterminado: {_num(defaults['sl_price'])}")
        else:
            await reply_md("SL predeterminado: —")
        await reply_md("Uso: `sl +5`, `sl -2`, `sl 105000`")
        return True

    m_sl_pct = re.match(r"^/?sl\s*([+\-]?)(\d+(?:\.\d+)?)%?\s*$", text_raw, re.IGNORECASE)
    m_sl_abs = None if m_sl_pct else re.match(r"^/?sl\s*\$?\s*(\d+(?:\.\d+)?)\s*$", text_raw, re.IGNORECASE)
    if m_sl_pct:
        sign = m_sl_pct.group(1)
        pct = float(m_sl_pct.group(2))
        pct_signed = (+pct if sign == "+" else (-pct if sign == "-" else -pct))
        update_protection_defaults(symbol_conf, sl_last_kind="pct", sl_pct_equity=pct_signed, sl_price=None)
        await reply_md(f"✅ SL predeterminado guardado: {pct_signed:+.2f}% del precio.")
        return True
    if m_sl_abs:
        raw = m_sl_abs.group(1)
        target = float(str(raw).replace(".", "").replace(",", "."))
        update_protection_defaults(symbol_conf, sl_last_kind="price", sl_price=target, sl_pct_equity=None)
        await reply_md(f"✅ SL predeterminado guardado en {_num(target)}.")
        return True

    # ----- TP -----
    if re.match(r"^/?tp\s*$", text_raw, re.IGNORECASE):
        stored_kind = defaults.get("tp_last_kind")
        if stored_kind == "pct" and defaults.get("tp_pct_equity") not in (None, ""):
            pct_val = float(defaults["tp_pct_equity"])
            await reply_md(f"TP predeterminado: {pct_val:+.2f}% del precio")
        elif stored_kind == "price" and defaults.get("tp_price") not in (None, ""):
            await reply_md(f"TP predeterminado: {_num(defaults['tp_price'])}")
        else:
            await reply_md("TP predeterminado: —")
        await reply_md("Uso: `tp +5` o `tp 109000`")
        return True

    m_tp_pct = re.match(r"^/?tp\s*([+\-]?)(\d+(?:\.\d+)?)%?\s*$", text_raw, re.IGNORECASE)
    m_tp_abs = None if m_tp_pct else re.match(r"^/?tp\s*\$?\s*(\d+(?:\.\d+)?)\s*$", text_raw, re.IGNORECASE)
    if m_tp_pct:
        sign = m_tp_pct.group(1)
        pct = float(m_tp_pct.group(2))
        pct_signed = (+pct if sign == "+" else (-pct if sign == "-" else +pct))
        update_protection_defaults(symbol_conf, tp_last_kind="pct", tp_pct_equity=pct_signed, tp_price=None)
        await reply_md(f"✅ TP predeterminado guardado: {pct_signed:+.2f}% del precio.")
        return True
    if m_tp_abs:
        raw = m_tp_abs.group(1)
        target = float(str(raw).replace(".", "").replace(",", "."))
        update_protection_defaults(symbol_conf, tp_last_kind="price", tp_price=target, tp_pct_equity=None)
        await reply_md(f"✅ TP predeterminado guardado en {_num(target)}.")
        return True

    return False

def _format_close_result(result: Any) -> tuple[bool, str | None]:
    if not isinstance(result, dict):
        return False, None

    if str(result.get("status") or "").lower() != "closed":
        return False, None

    summary = result.get("summary") or {}
    side = str(summary.get("side") or "?").upper()
    qty = float(summary.get("qty") or summary.get("quantity") or 0.0)
    entry_price = float(summary.get("entry_price") or summary.get("entry") or 0.0)
    exit_price = float(summary.get("exit_price") or summary.get("exit") or 0.0)
    pnl_val = summary.get("realized_pnl")
    if pnl_val is None:
        pnl_val = summary.get("pnl_balance_delta")

    # Fallback: calcular PnL en USDT por diferencia de precios y qty
    try:
        pnl_usd_calc = (exit_price - entry_price) * qty
        if side == "SHORT":
            pnl_usd_calc = -pnl_usd_calc
    except Exception:
        pnl_usd_calc = None

    # Porcentaje según la entrada (movimiento de precio, no equity)
    try:
        delta_pct = ((exit_price - entry_price) / entry_price) * (-100 if side == "SHORT" else 100)
    except Exception:
        delta_pct = None

    msg = (
        "<b>✅ Cerré la posición del BOT</b>\n"
        f"• Lado: <b>{side}</b>\n"
        f"• Cantidad: {qty:.6f}\n"
        f"• Precio: entrada {entry_price:,.2f} | salida {exit_price:,.2f}"
    )
    # Mostrar PnL en USDT y el % (sobre precio de entrada)
    pnl_to_show = pnl_val if pnl_val is not None else pnl_usd_calc
    if pnl_to_show is not None:
        try:
            msg += f"\n• PnL: {float(pnl_to_show):+,.2f} USDT"
        except Exception:
            msg += f"\n• PnL: {pnl_to_show} USDT"
    if delta_pct is not None:
        try:
            msg += f" ({float(delta_pct):+.2f}%)"
        except Exception:
            pass
    return True, msg


async def cerrar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app = _get_app_from_context(context)
    message = update.effective_message
    if app is None or message is None:
        return

    symbol = app.config.get("symbol", "BTC/USDT") if getattr(app, "config", None) else "BTC/USDT"
    if getattr(app, "is_live", False):
        try:
            pos = await app.exchange.get_open_position(symbol)
        except Exception:
            pos = None
        if pos:
            side_pos = str(pos.get("side") or pos.get("positionSide") or "").upper()
            try:
                qty_val = float(
                    pos.get("contracts")
                    or pos.get("positionAmt")
                    or pos.get("size")
                    or pos.get("qty")
                    or 0.0
                )
            except Exception:
                qty_val = 0.0
            qty_val = abs(qty_val)
            if side_pos in {"LONG", "SHORT"} and qty_val > 0:
                client = trading.get_live_client()
                if client is None:
                    await message.reply_text("No hay cliente LIVE disponible para cerrar la posición.")
                    return
                try:
                    rules = await asyncio.to_thread(trading.get_symbol_rules, symbol)
                except Exception as exc:
                    await message.reply_text(f"Error al obtener filtros del símbolo: {exc}")
                    return
                try:
                    result = await asyncio.to_thread(
                        trading.close_position_hard,
                        client,
                        symbol,
                        side_pos,
                        qty_val if side_pos == "LONG" else -qty_val,
                        int(rules.get("price_precision") or 2),
                        int(rules.get("qty_precision") or 0),
                    )
                except Exception as exc:
                    await message.reply_text(f"Error al cerrar: {exc}")
                    return
                if not result.get("ok"):
                    await message.reply_text(
                        f"Error al cerrar: {result.get('error') or result.get('msg')}"
                    )
                    return
                await asyncio.to_thread(trading.bootstrap_real_state)
                qty_closed = float(result.get("qty") or qty_val)
                price_val = result.get("price")
                close_msg = (
                    "<b>✅ Cerré la posición del BOT</b>\n"
                    f"• Lado: <b>{side_pos}</b>\n"
                    f"• Cantidad: {qty_closed:.6f}"
                )
                if price_val not in (None, ""):
                    try:
                        close_msg += f"\n• Precio de cierre: {float(price_val):,.2f}"
                    except Exception:
                        pass
                await message.reply_html(close_msg)
                try:
                    app.manual_block_until = time.time() + 600
                    log = getattr(app, "log", None) or getattr(app, "logger", None)
                    if log is not None:
                        log.info(
                            "Cooldown activado por cierre manual: 10 min sin entradas nuevas."
                        )
                    return
                except Exception:
                    if getattr(app, "logger", None) is not None:
                        app.logger.debug(
                            "No se pudo establecer el cooldown manual tras cierre.",
                            exc_info=True,
                        )
                    return

    try:
        result = await app.close_all()
    except Exception as exc:
        await message.reply_text(f"⚠️ Error al cerrar la posición: {exc}")
        return

    ok, text = _format_close_result(result)
    if ok and text:
        await message.reply_html(text)
    else:
        await message.reply_text("✅ Cerré la posición del BOT.")

    if ok and app is not None:
        try:
            app.manual_block_until = time.time() + 600
            log = getattr(app, "log", None) or getattr(app, "logger", None)
            if log is not None:
                log.info(
                    "Cooldown activado por cierre manual: 10 min sin entradas nuevas."
                )
        except Exception:
            if getattr(app, "logger", None) is not None:
                app.logger.debug(
                    "No se pudo establecer el cooldown manual tras cierre.",
                    exc_info=True,
                )


async def handle_open_manual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return
    text = (message.text or "").strip()
    match = OPEN_TEXT_RE.match(text)
    if not match:
        return

    app = _get_app_from_context(context)
    if app is None:
        return
    if not getattr(app, "is_live", False):
        await message.reply_text("Abrir manual está disponible sólo en modo REAL.")
        return

    side_txt = match.group(1).upper()
    leverage = int(match.group(2))
    if leverage < 1 or leverage > 125:
        await message.reply_text("Apalancamiento inválido. Elegí un valor entre 1 y 125.")
        return

    symbol = app.config.get("symbol", "BTC/USDT") if getattr(app, "config", None) else "BTC/USDT"
    client = trading.get_live_client()
    if client is None:
        await message.reply_text("No hay cliente LIVE disponible para abrir la posición.")
        return

    symbol_clean = symbol.replace("/", "")

    def _set_leverage():
        return client.futures_change_leverage(symbol=symbol_clean, leverage=leverage)

    try:
        await asyncio.to_thread(_set_leverage)
    except Exception as exc:
        await message.reply_text(f"Error al setear leverage x{leverage}: {exc}")
        return

    balance = await asyncio.to_thread(trading.fetch_futures_usdt_balance)
    if balance is None or balance <= 0:
        await message.reply_text("No pude obtener el balance de futuros USDT.")
        return

    price = await asyncio.to_thread(trading.get_latest_price, symbol)
    if price is None or price <= 0:
        await message.reply_text("No pude obtener el precio actual del símbolo.")
        return

    risk_pct = float(MANUAL_OPEN_RISK_PCT or 0.5)
    notional = balance * (risk_pct / 100.0) * leverage
    qty_raw = notional / price if price else 0.0

    try:
        rules = await asyncio.to_thread(trading.get_symbol_rules, symbol)
    except Exception as exc:
        await message.reply_text(f"Error al obtener filtros del símbolo: {exc}")
        return

    step = float(rules.get("step_size") or 0.0)
    qty_precision = int(rules.get("qty_precision") or 0)
    if step > 0:
        qty = math.floor((qty_raw / step) + 1e-9) * step
    else:
        qty = qty_raw
    qty = round(qty, qty_precision)
    if qty <= 0:
        await message.reply_text("Qty resultó 0. Ajusta MANUAL_OPEN_RISK_PCT o el leverage.")
        return

    side_order = "BUY" if side_txt == "LONG" else "SELL"

    try:
        result = await asyncio.to_thread(
            trading.place_order_safe,
            side_order,
            qty,
            None,
            symbol=symbol,
            leverage=leverage,
        )
    except Exception as exc:
        await message.reply_text(f"Error al abrir: {exc}")
        return

    price_fill = trading._infer_fill_price(result, price)
    text_reply = f"✅ Abierta {side_txt} x{leverage} qty={qty:.6f} a mercado."
    if price_fill:
        try:
            text_reply += f" Precio≈{float(price_fill):,.2f}"
        except Exception:
            pass
    await message.reply_text(text_reply)

async def control_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toma control de la posición REAL actual: fuerza modo REAL y sincroniza desde el exchange."""

    engine = _get_engine_from_context(context)
    message = update.effective_message
    if engine is None or message is None:
        return

    result = trading.set_trading_mode("real", source="telegram/control")
    if not result.ok:
        await message.reply_text(f"❌ No pude activar modo REAL: {result.msg}", parse_mode="Markdown")
        return

    extra = result.msg or ""
    warn_lines = [
        line.strip()
        for line in extra.splitlines()
        if line.strip() and "Modo cambiado" not in line
    ]
    if warn_lines:
        await message.reply_text("\n".join(warn_lines), parse_mode="Markdown")

    try:
        if hasattr(engine, "exchange") and engine.exchange:
            await engine.exchange.upgrade_to_real_if_needed()
    except Exception:
        logger.debug("control_command: no se pudo reautenticar exchange tras activar REAL.", exc_info=True)

    try:
        if getattr(engine, "trader", None):
            engine.trader.reset_caches()
    except Exception:
        logger.debug("control_command: no se pudo resetear caches del trader.", exc_info=True)

    await message.reply_text("Modo REAL activado. Buscando posición LIVE en el exchange…")
    if not hasattr(engine, "sync_live_position"):
        await message.reply_text("❌ Engine no soporta sincronización automática.")
        return

    try:
        synced = await asyncio.to_thread(engine.sync_live_position)
    except Exception as exc:
        await message.reply_text(f"❌ Error al sincronizar con el exchange: {exc}")
        return

    if synced:
        await message.reply_text("✅ Posición LIVE sincronizada. Ya la estoy controlando.")
    else:
        await message.reply_text("ℹ️ No encontré posición LIVE en el exchange.")


async def sl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app = _get_app_from_context(context)
    message = update.effective_message
    if app is None or message is None:
        return

    args = context.args or []
    if not args:
        pct = float(getattr(app, "sl_equity_pct", 10.0) or 0.0)
        await message.reply_text(f"SL predeterminado: -{pct:.2f}% del equity.")
        return

    raw = " ".join(args).strip().replace(",", ".")
    if raw.startswith("$"):
        try:
            px = float(raw.replace("$", "").strip())
        except Exception:
            await message.reply_text("Formato inválido. Usá `sl 5` (equity %) o `sl $108000` (precio).")
            return
        try:
            pos = await app.exchange.get_open_position(app.config.get("symbol"))
        except Exception:
            pos = None
        if pos:
            await app.exchange.update_protections(
                symbol=app.config.get("symbol"),
                side=pos.get("side"),
                qty=float(pos.get("contracts") or pos.get("positionAmt") or 0.0),
                sl=px,
            )
            await message.reply_text(f"✅ SL actualizado a ${px:,.2f}")
        else:
            await message.reply_text("No hay posición del BOT abierta.")
        return

    try:
        pct = float(raw)
    except Exception:
        await message.reply_text("Formato inválido. Usá `sl 5` (equity %) o `sl $108000` (precio).")
        return
    if hasattr(app, "set_sl_equity_pct"):
        app.set_sl_equity_pct(pct)
    await message.reply_text(f"✅ SL predeterminado guardado en -{pct:.2f}% del equity.")


async def tp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app = _get_app_from_context(context)
    message = update.effective_message
    if app is None or message is None:
        return

    args = [arg.strip() for arg in (context.args or []) if arg.strip()]
    cfg = _engine_config(app)
    if not isinstance(cfg, dict):
        cfg = {}

    mapping = cfg.get("tp_eq_pct_by_leverage")
    if not isinstance(mapping, dict):
        mapping = {}

    default_candidates = [
        cfg.get("target_eq_pnl_pct"),
        cfg.get("tp_equity_pct"),
        cfg.get("tp_pct"),
    ]
    strategy_cfg = cfg.get("strategy")
    if isinstance(strategy_cfg, dict):
        default_candidates.extend(
            [
                strategy_cfg.get("target_eq_pnl_pct"),
                strategy_cfg.get("tp_pct"),
            ]
        )
    default_pct = 0.10
    for candidate in default_candidates:
        parsed_candidate = _try_parse_tp_pct(candidate, allow_zero=True)
        if parsed_candidate is not None:
            default_pct = parsed_candidate
            break

    def _current_pct_for(lev: int) -> float:
        raw_val = mapping.get(str(lev)) if isinstance(mapping, dict) else None
        if raw_val is None and isinstance(mapping, dict):
            raw_val = mapping.get(lev)
        parsed = _try_parse_tp_pct(raw_val, allow_zero=True)
        return parsed if parsed is not None else default_pct

    if not args:
        tp5 = _current_pct_for(5)
        tp10 = _current_pct_for(10)
        await message.reply_text(
            "TP actual:\n"
            f"• x5: {tp5 * 100:.3f}%\n"
            f"• x10: {tp10 * 100:.3f}%\n"
            f"(default {default_pct * 100:.3f}%)"
        )
        return

    if not _user_is_admin(update, cfg):
        await message.reply_text("No autorizado.")
        return

    if len(args) < 2:
        await message.reply_text("Uso: /tp x5 10   ó   /tp x10 0.8%")
        return

    lev_token = args[0].lower().lstrip("/")
    if lev_token.startswith("x"):
        lev_token = lev_token[1:]
    try:
        lev_value = int(lev_token)
    except Exception:
        lev_value = None
    if lev_value not in {5, 10}:
        await message.reply_text("Leverage inválido. Usá x5 o x10.")
        return

    raw_pct_text = " ".join(args[1:]).strip()
    try:
        pct_value = _parse_tp_pct(raw_pct_text)
    except ValueError as exc:
        await message.reply_text(f"Valor inválido: {exc}")
        return

    if not (0 < pct_value <= 0.5):
        await message.reply_text("Rango inválido. Recomendado: 0.1%–5%.")
        return

    mapping = dict(mapping) if not isinstance(mapping, dict) else mapping
    mapping[str(lev_value)] = float(round(pct_value, 6))
    cfg["tp_eq_pct_by_leverage"] = mapping

    try:
        strategy = getattr(app, "strategy", None)
        if strategy and isinstance(getattr(strategy, "config", None), dict):
            strategy.config["tp_eq_pct_by_leverage"] = mapping
    except Exception:
        logger.debug("No se pudo sincronizar strategy.config con tp_eq_pct_by_leverage", exc_info=True)

    _persist_tp_mapping(str(lev_value), pct_value)

    await message.reply_text(f"OK. TP x{lev_value} = {pct_value * 100:.3f}%")


async def precio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "Engine no disponible para consultar precios.")
        return
    text = (update.effective_message.text or "").strip() if update.effective_message else ""
    parts = text.split()
    symbols: Iterable[str]
    if len(parts) >= 2:
        tok = parts[1].upper(); tok = tok.split(':',1)[0] if ':' in tok else tok; symbols = [tok]
    else:
        symbols = getattr(engine, "symbols", []) or list(getattr(engine, "price_cache", {}).keys()) or ["BTC/USDT"]
    responses = []
    for sym in symbols:
        price = None
        cache = getattr(engine, "price_cache", {}) or {}
        if sym in cache:
            price = cache[sym]
        elif hasattr(engine, "price_of"):
            try:
                price = engine.price_of(sym)
            except Exception:
                price = None
        if price is None and hasattr(engine, "fetch_last_price"):
            try:
                result = engine.fetch_last_price(sym)
                if inspect.isawaitable(result):
                    price = await result
                else:
                    price = result
            except Exception:
                price = None
        if price is None:
            responses.append(f"{sym}: precio no disponible todavía.")
        else:
            responses.append(f"{sym}: ${_fmt_num(price, 2)}")
    await _reply_chunks(update, "\n".join(responses))


async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    message = update.effective_message
    if message is None:
        return
    await _cmd_config(engine, message.reply_text)


async def pausa_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "No pude acceder al engine.")
        return
    _set_killswitch(engine, True)
    await _reply_chunks(update, "⛔ Bot OFF: bloqueadas nuevas operaciones (killswitch ACTIVADO).")


async def reanudar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "No pude acceder al engine.")
        return
    _set_killswitch(engine, False)
    await _reply_chunks(update, "✅ Bot ON: habilitadas nuevas operaciones (killswitch desactivado).")


async def bot_on_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reanudar_command(update, context)


async def bot_off_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await pausa_command(update, context)


async def _cmd_modo_simulado(engine, reply):
    if _get_mode_from_engine(engine) == "paper":
        return await reply("✅ El bot ya se encontraba en *MODO SIMULADO*.")
    try:
        engine.set_mode("paper", source="telegram")  # usa el método que agregaste recién
        return await reply("✅ Modo cambiado a *SIMULADO*. El bot ahora opera en simulado.")
    except Exception as e:
        return await reply("⚠️ No pude cambiar a SIMULADO (revisá logs / configuración).")


async def _cmd_modo_real(engine, reply):
    if _get_mode_from_engine(engine) == "live":
        return await reply("✅ El bot ya está en *MODO REAL*.")

    try:
        engine.set_mode("live", source="telegram")
    except Exception as e:
        return await reply(f"⚠️ No pude cambiar a REAL: {e}")

    ex = getattr(engine, "exchange", None)
    if ex and hasattr(ex, "upgrade_to_real_if_needed"):
        try:
            await ex.upgrade_to_real_if_needed()
        except Exception as e:
            return await reply(f"⚠️ No pude autenticar el exchange: {e}")

    # Verificaciones
    authed = False
    try:
        authed = bool(getattr(ex, "is_authenticated", False))
        client = getattr(ex, "client", None)
        if client and getattr(client, "apiKey", None):
            authed = True
    except Exception:
        pass

    bal_txt = ""
    try:
        if ex and hasattr(ex, "fetch_balance_usdt"):
            bal = await ex.fetch_balance_usdt()
            bal_txt = f" | saldo USDT: {bal:,.2f}"
    except Exception:
        bal_txt = " | saldo: N/D"

    if not authed:
        return await reply("⚠️ MODO REAL activado pero *no veo auth en CCXT*. "
                            "Verificá API/SECRET, permisos de Futuros y `defaultType=future` en el cliente.")

    return await reply(f"✅ MODO REAL activado{bal_txt}")


async def modo_simulado_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return

    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "No pude acceder al engine.")
        return

    reply_md = lambda txt: message.reply_text(txt, parse_mode="Markdown")
    await _cmd_modo_simulado(engine, reply_md)


async def modo_real_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return

    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "No pude acceder al engine.")
        return

    reply_md = lambda txt: message.reply_text(txt, parse_mode="Markdown")
    await _cmd_modo_real(engine, reply_md)


async def killswitch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "Engine no disponible para killswitch.")
        return
    close_error: Optional[str] = None
    try:
        await engine.close_all()  # solo la del BOT
    except Exception as exc:  # pragma: no cover - defensivo
        close_error = str(exc)
    _set_killswitch(engine, True)
    if close_error:
        await _reply_chunks(
            update,
            "⚠️ Activé el killswitch pero no pude cerrar la **posición del BOT**: "
            f"{close_error}",
        )
    else:
        await _reply_chunks(
            update,
            "🛑 Killswitch ACTIVADO: se cerró la **posición del BOT** y se pausó el bot.",
        )


async def ajustar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "No pude acceder al engine para ajustar parámetros.")
        return
    args = getattr(context, "args", None) or []
    if len(args) >= 2:
        param = args[0]
        raw_value = " ".join(args[1:])
    else:
        text = (update.effective_message.text or "") if update.effective_message else ""
        match = re.match(
            r"ajustar\s+([\w.]+)\s+(.+)$",
            text.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            await _reply_chunks(update, "Uso: ajustar [parametro] [valor]. Ej: ajustar risk.max_hold_bars 20")
            return
        param = match.group(1)
        raw_value = match.group(2)
    value = _parse_adjust_value(raw_value)
    path = _find_and_set_config(engine, param, value)
    if path:
        await _reply_chunks(update, f"✅ Actualicé {'/'.join(path)} = {value}")
    else:
        await _reply_chunks(update, f"No encontré el parámetro '{param}' en la configuración.")


async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    limit = _extract_logs_limit(update, context)
    text = _read_logs_text(engine, limit)
    await _reply_chunks(update, text)


async def equity_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra o fija equity base (USDT) y porcentaje de sizing."""
    engine = _get_engine_from_context(context)
    message = update.effective_message
    if message is None:
        return
    if engine is None:
        await message.reply_text("No pude acceder al engine para ajustar el equity.")
        return

    txt = (message.text or "").strip().lower()
    pattern = re.compile(
        r"equity\s+(?:(usd|usdt|\$)\s+)?(\d+(?:\d+)?)(?:\s*(usd|usdt|\$|%))?"
    )
    match = pattern.search(txt)
    if not match:
        fraction = float(_get_equity_fraction(engine))
        pct = round(fraction * 100.0, 2)
        base_equity = get_equity_sim()
        await message.reply_text(
            "Equity base actual: {:.2f}\nEquity % actual: {:.2f}% (frac={:.4f})".format(
                float(base_equity or 0.0), pct, fraction
            )
        )
        return

    try:
        value = float(match.group(2).replace(",", "."))
    except Exception:
        await message.reply_text("Formato: equity 50  |  equity 50%  |  equity usd 1200")
        return

    prefix = (match.group(1) or "").strip()
    suffix = (match.group(3) or "").strip()
    is_usd = prefix in {"usd", "usdt", "$"} or suffix in {"usd", "usdt", "$"}

    if not is_usd:
        if not (0.0 < value <= 100.0):
            await message.reply_text("El porcentaje debe estar entre 0 y 100.")
            return

        frac = round(value / 100.0, 4)
        _find_and_set_config(engine, "order_sizing.default_pct", frac)
        os.environ["EQUITY_PCT"] = str(frac)

        try:
            current_mode = get_mode()
            is_real = current_mode == "real"
        except Exception:
            is_real = False

        if is_real:
            try:
                cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
                raw = read_config_raw(cfg_path) or {}
                raw.setdefault("order_sizing", {})
                raw["order_sizing"]["default_pct"] = float(frac)
                with open(cfg_path, "w", encoding="utf-8") as fh:
                    yaml.safe_dump(raw, fh, sort_keys=False, allow_unicode=True)
            except Exception:
                pass

        await message.reply_text(
            f"✅ Porcentaje de equity seteado: {value:.2f}% (frac={frac})"
        )
        return

    # Equity explícito en USDT
    set_equity_sim(value)
    try:
        trader = getattr(engine, "trader", None)
        if trader is not None:
            trader.set_paper_equity(value)
    except Exception:
        logger.debug("No se pudo actualizar trader.set_paper_equity", exc_info=True)

    await message.reply_text(f"✅ Equity base seteado: {value:.2f} USDT")


async def diag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Diagnóstico rápido del estado del bot."""
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "Engine no disponible.")
        return

    lines = ["🧪 *Diagnóstico*"]

    try:
        mode = str(get_mode()).upper()
    except Exception:
        mode = "DESCONOCIDO"
    lines.append(f"• Modo: `{mode}`")

    ex = getattr(engine, "exchange", None)
    if ex:
        try:
            authed = bool(getattr(ex, "is_authenticated", False))
            client = getattr(ex, "client", None)
            if authed and client is not None:
                authed = bool(getattr(client, "apiKey", None))
            lines.append(f"• CCXT: {'AUTENTICADO' if authed else 'PÚBLICO'}")
        except Exception:
            lines.append("• CCXT: (estado desconocido)")

        try:
            px = await ex.get_current_price()
            lines.append(f"• Precio cache: {px if px is not None else 'N/D'}")
            try:
                age = ex.get_price_age_sec()
            except Exception:
                age = None
            if age is not None and age != float("inf"):
                lines.append(f"• Edad precio WS: {age:.1f}s")
                if age > 10:
                    lines.append("⚠️ WS frío (>10s sin precio). Revisa conexión.")
        except Exception:
            lines.append("• Precio cache: error")

        try:
            symbol = engine.config.get("symbol", "BTC/USDT") if getattr(engine, "config", None) else "BTC/USDT"
            if getattr(ex, "public_client", None):
                fr = await asyncio.to_thread(ex.public_client.fetchFundingRate, symbol)
                val = float(fr.get("fundingRate")) if fr else None
            else:
                val = None
            lines.append(f"• Funding rate: {val if val is not None else 'N/D'}")
        except Exception:
            lines.append("• Funding rate: error")
    else:
        lines.append("• Exchange: N/D")

    try:
        trader = getattr(engine, "trader", None)
        if trader is not None:
            eq = await trader.get_balance(ex)
        else:
            eq = None
        lines.append(f"• Equity: {eq if eq is not None else 'N/D'}")
    except Exception:
        lines.append("• Equity: error")

    await _reply_chunks(update, "\n".join(lines), parse_mode="Markdown")


async def motivos_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Envía los últimos motivos registrados por los filtros de entrada."""
    items = MOTIVES.last(10)
    if not items:
        await _reply_chunks(update, "No hay rechazos recientes.")
        return

    tz = (
        getattr(S, "output_timezone", "America/Argentina/Buenos_Aires")
        if hasattr(S, "output_timezone")
        else "America/Argentina/Buenos_Aires"
    )
    lines = ["🕒 Motivos recientes (últimas 10 oportunidades NO abiertas):"]
    for it in items:
        lines.append(it.human_line(tz=tz))
    logger.debug(
        "TELEGRAM /motivos → %d items | 1ra: %s",
        len(items),
        lines[1] if len(lines) > 1 else "-",
    )
    await _reply_chunks(update, "\n".join(lines))


def _populate_registry() -> None:
    REGISTRY.register(
        "ayuda",
        ayuda_command,
        aliases=["help", "comandos"],
        help_text="Muestra esta ayuda",
    )
    REGISTRY.register(
        "precio",
        precio_command,
        aliases=["price", "precio actual", "cotizacion", "cotización", "btc"],
        help_text="Muestra el precio actual de BTC/USDT",
    )
    REGISTRY.register(
        "estado",
        estado_command,
        aliases=["status", "balance", "pnl"],
        help_text="Muestra PnL del día/semana y balance",
    )
    REGISTRY.register(
        "posicion",
        posicion_command,
        aliases=[
            "posición",
            "position",
            "pos",
            "posicion actual",
            "posición actual",
        ],
        help_text="Muestra el estado de la posición abierta (si existe)",
    )
    REGISTRY.register(
        "posiciones",
        posiciones_command,
        aliases=["positions", "open positions", "posicioness"],
        help_text="Lista todas las posiciones abiertas (la del bot en negrita).",
    )
    REGISTRY.register(
        "open",
        open_command,
        aliases=["abrir"],
        help_text="Abre una operación manual. Ej: open long x5",
    )
    REGISTRY.register(
        "diag",
        diag_command,
        aliases=["diagnostico", "health"],
        help_text="Muestra un diagnóstico rápido (modo, CCXT, precio, funding, equity).",
    )
    REGISTRY.register(
        "rendimiento",
        rendimiento_command,
        aliases=["performance", "estadisticas", "estadísticas"],
        help_text="Muestra estadísticas históricas completas",
    )
    REGISTRY.register(
        "motivos",
        motivos_command,
        aliases=[
            "razones",
            "motivo",
            "por que no entro",
            "por qué no entro",
            "porque no entro",
        ],
        help_text="Últimos rechazos y motivos claros",
    )
    REGISTRY.register(
        "config",
        config_command,
        aliases=["configuracion", "configuración", "parametros", "parámetros"],
        help_text="Muestra los parámetros actuales del bot",
    )
    REGISTRY.register(
        "logs",
        logs_command,
        aliases=["log", "ver logs", "log tail"],
        help_text="Muestra las últimas N líneas del log",
    )
    REGISTRY.register(
        "equity",
        equity_command,
        aliases=["equity%", "equitypct", "porcentaje", "size"],
        help_text="Muestra o fija equity base (USDT) o %. Ej: equity 1200 | equity 25%",
    )
    REGISTRY.register(
        "sl",
        sl_command,
        aliases=["stop", "stoploss"],
        help_text="Ver o setear SL como % del equity o precio fijo. Ej: `sl 10` o `sl $108000`.",
    )
    REGISTRY.register(
        "pausa",
        pausa_command,
        aliases=["pausar", "bot off", "bot apagar", "desactivar bot", "botoff"],
        help_text="Detiene la apertura de nuevas operaciones",
    )
    REGISTRY.register(
        "reanudar",
        reanudar_command,
        aliases=[
            "resume",
            "continuar",
            "bot on",
            "bot prender",
            "activar bot",
            "boton",
            "botón",
        ],
        help_text="Reanuda la apertura de operaciones",
    )
    REGISTRY.register(
        "cerrar",
        cerrar_command,
        aliases=["close", "cerrar posicion", "cerrar posición"],
        help_text="Cierra la posición abierta por el bot (paper/real)",
        show_in_help=True,
    )
    REGISTRY.register(
        "control",
        control_command,
        aliases=["sync", "tomarcontrol", "rescate"],
        help_text="Toma control de la posición LIVE del exchange (fuerza REAL y sincroniza).",
        show_in_help=True,
    )
    REGISTRY.register(
        "tp",
        tp_command,
        aliases=["takeprofit"],
        help_text="Consulta o ajusta el TP por apalancamiento. Ej: `tp x5 0.8%`.",
    )
    REGISTRY.register(
        "killswitch",
        killswitch_command,
        aliases=["panic", "cerrar todo", "panic button"],
        help_text="Cierra posición y pausa el bot",
        show_in_help=False,
    )
    REGISTRY.register(
        "ajustar",
        ajustar_command,
        aliases=["ajuste", "set", "config set"],
        help_text="Ajusta parámetros en caliente",
        show_in_help=False,
    )
    REGISTRY.register(
        "modo simulado",
        modo_simulado_command,
        aliases=[
            "simulado",
            "paper",
            "demo",
            "test",
            "modo demo",
            "activar simulado",
            "poner modo simulado",
        ],
        help_text="Cambia el bot a modo SIMULADO",
    )
    REGISTRY.register(
        "modo real",
        modo_real_command,
        aliases=["real", "live", "activar real", "poner modo real", "usar real"],
        help_text="Cambia el bot a modo REAL (requiere API keys). Si hay una posición del otro modo, sigue en segundo plano sin abrir nuevas en ese modo.",
    )


async def _dispatch_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    unknown_message: str,
) -> None:
    message = update.effective_message
    text = (message.text or "").strip() if message else ""
    if not text:
        return

    engine = _get_engine_from_context(context)
    norm = normalize(text)
    norm_noslash = norm.lstrip("/")
    candidates = (norm, norm_noslash)

    reply_md = None
    if engine is not None and message is not None:
        reply_md = lambda txt: message.reply_text(txt, parse_mode="Markdown")
        if await _handle_position_controls(engine, reply_md, text):
            return
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in ("modo simulado", "simulado", "paper") or candidate.startswith(
                "modo simulado "
            ):
                await _cmd_modo_simulado(engine, reply_md)
                return
            if candidate in ("modo real", "real", "live") or candidate.startswith("modo real "):
                await _cmd_modo_real(engine, reply_md)
                return
    chat = update.effective_chat
    notifier = None
    application = getattr(context, "application", None)
    if application is not None:
        notifier = application.bot_data.get("notifier")
    if notifier and getattr(notifier, "default_chat_id", None) is None and chat is not None:
        try:
            notifier.set_default_chat(chat.id)
        except Exception:
            logger.debug("No se pudo fijar default_chat_id automáticamente", exc_info=True)
    command = REGISTRY.resolve(text)
    logger.debug("Resolve: '%s' -> %s", text, command)
    if not command:
        if message is not None:
            await message.reply_text(unknown_message, parse_mode="Markdown")
        return
    handler = REGISTRY.handler_for(command)
    if handler is None:
        if message is not None:
            await message.reply_text(
                "Comando no disponible. Escribí *ayuda*.", parse_mode="Markdown"
            )
        return
    await handler(update, context)


def _prepare_args_for_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    args: List[str] = []
    if message and message.text:
        parts = message.text.strip().split()
        if parts:
            args = parts[1:]
    setattr(context, "args", args)


async def _slash_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _dispatch_command(
        update,
        context,
        unknown_message="Comando no reconocido. Escribí *ayuda*.",
    )


async def _text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _prepare_args_for_text(update, context)
    await _dispatch_command(
        update,
        context,
        unknown_message="No entendí. Escribí *ayuda* para ver comandos.",
    )


async def _status_plaintext_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await estado_command(update, context)


def register_commands(application: Application) -> None:
    _populate_registry()
    if not getattr(application, "_chaulet_bot_handlers_registered", False):
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler(["ayuda", "help"], ayuda_command))
        application.add_handler(CommandHandler(["estado", "status", "diagnostico"], estado_command))
        application.add_handler(CommandHandler(["posicion", "position"], posicion_command))
        application.add_handler(CommandHandler(["posiciones", "positions"], posiciones_command))
        application.add_handler(CommandHandler(["cerrar", "close"], cerrar_command))
        application.add_handler(CommandHandler(["sl", "stoploss"], sl_command))
        application.add_handler(CommandHandler(["tp", "takeprofit"], tp_command))
        setattr(application, "_chaulet_bot_handlers_registered", True)
    if getattr(application, "_chaulet_router_registered", False):
        return

    application.add_handler(MessageHandler(filters.COMMAND, _slash_router))
    application.add_handler(
        MessageHandler(filters.Regex(r"(?i)^(status|estado)$"), _status_plaintext_handler) # CORRECCIÓN
    )
    application.add_handler(
        MessageHandler(filters.Regex(CLOSE_TEXT_RE_PATTERN), cerrar_command) # CORRECCIÓN
    )
    application.add_handler(
        MessageHandler(filters.Regex(OPEN_TEXT_RE_PATTERN), handle_open_manual) # CORRECCIÓN
    )
    application.add_handler(
        MessageHandler(filters.Regex(POSITION_TEXT_RE_PATTERN), posicion_command) # CORRECCIÓN
    )
    generic_filter = filters.TEXT & (~filters.COMMAND)
    generic_filter = generic_filter & (~filters.Regex(CLOSE_TEXT_RE_PATTERN)) # CORRECCIÓN
    generic_filter = generic_filter & (~filters.Regex(OPEN_TEXT_RE_PATTERN)) # CORRECCIÓN
    generic_filter = generic_filter & (~filters.Regex(POSITION_TEXT_RE_PATTERN)) # CORRECCIÓN
    generic_filter = generic_filter & (~filters.Regex(r"(?i)^(status|estado)$")) # CORRECCIÓN
    application.add_handler(MessageHandler(generic_filter, _text_router))
    setattr(application, "_chaulet_router_registered", True)
    logger.info(
        "Router central de comandos registrado (%d comandos).",
        len(REGISTRY),
    )


def setup_telegram_bot(engine_instance):
    """Configura y devuelve la aplicación de Telegram con TODOS los handlers."""
    cfg = _engine_config(engine_instance)
    tconf = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}

    if not bool(tconf.get("enabled", True)):
        logger.info("Telegram disabled")
        return None

    token = (
        tconf.get("token")
        or cfg.get("telegram_token")
        or _env("TELEGRAM_TOKEN")
        or _env("TELEGRAM_BOT_TOKEN")
    )
    if not token:
        logger.warning("No TELEGRAM_TOKEN provided; Telegram disabled")
        return None

    latency_warn_s = _first_float(
        tconf.get("latency_warn_s"),
        tconf.get("latency_warn_seconds"),
        _env("TELEGRAM_LATENCY_WARN_S"),
        default=1.0,
    )
    summary_interval_s = _first_float(
        tconf.get("log_summary_interval_s"),
        tconf.get("log_summary_s"),
        _env("TELEGRAM_LOG_SUMMARY_S"),
        default=60.0,
    )

    request = TelegramLoggingRequest(
        latency_warning_s=latency_warn_s,
        summary_interval_s=summary_interval_s,
    )

    try:
        builder = Application.builder().token(token).request(request)
        application = builder.build()
    except Exception as exc:
        logger.warning("Telegram application init failed: %s", exc)
        return None

    application.add_error_handler(_on_error)
    application.bot_data["engine"] = engine_instance
    try:
        application.user_data["engine"] = engine_instance
        application.user_data["__app__"] = engine_instance
    except Exception:
        pass

    register_commands(application)
    return application


def _chat_id_from_env():
    chat_id_env = _env("TELEGRAM_CHAT_ID")
    if not chat_id_env:
        return None
    try:
        return int(chat_id_env)
    except (TypeError, ValueError):
        logger.warning("Invalid TELEGRAM_CHAT_ID provided; ignoring")
    return None

# =========================
# Notificador PRO
# =========================
class TelegramNotifier:
    """
    API que usa el engine para enviar mensajes “PRO” de apertura/cierre
    con porcentajes y datos que pediste.
    """
    def __init__(self, application, cfg: dict, default_chat_id=None):
        self.app = application
        self.cfg = cfg or {}
        self.default_chat_id = default_chat_id
        tcfg = (cfg or {}).get("telegram", {}) if isinstance(cfg, dict) else {}

        self.min_interval_s = float((tcfg.get("anti_spam", {}) or {}).get("min_interval_s", 1.0))
        self.dedup_window_s = int(tcfg.get("dedup_window_s", 60))
        self._last_sent_ts = 0.0
        self._dedup = deque(maxlen=200)
        self._rejections = deque(maxlen=50)  # ring buffer de motivos
        self.equity_csv, self.trades_csv = _cfg_csv_paths(self.cfg)

    def set_default_chat(self, chat_id):
        self.default_chat_id = chat_id

    def attach_app(self, application):
        self.app = application

    def _schedule(self, coro):
        if not self.app:
            logger.debug("telegram application not available; skipping notification")
            return
        try:
            self.app.create_task(coro)
        except Exception as exc:
            logger.warning("telegram scheduling failed: %s", exc)

    # API pública (no-async): el engine llama a estos
    def open(self, **kwargs):
        self._schedule(self._send_open(**kwargs))

    def tp1(self, **kwargs):
        self._schedule(self._send_tp1(**kwargs))

    def trailing(self, **kwargs):
        self._schedule(self._send_trailing(**kwargs))

    def close_tp(self, **kwargs):
        self._schedule(self._send_close(kind="TP", **kwargs))

    def close_sl(self, **kwargs):
        self._schedule(self._send_close(kind="SL", **kwargs))

    def close_manual(self, **kwargs):
        self._schedule(self._send_close(kind="MANUAL", **kwargs))

    def reject(self, **kwargs):
        self._schedule(self._send_reject(**kwargs))

    # Permite que el engine registre motivos para /motivos (o command-bot)
    def log_reject(self, symbol: str, side: str, code: str, detail: str = ""):
        self._rejections.appendleft({"symbol": symbol, "side": side, "code": code, "detail": detail})

    # ----------- Internals -----------
    async def _safe_send(self, text: str):
        try:
            now = asyncio.get_event_loop().time()
            if (now - self._last_sent_ts) < self.min_interval_s:
                await asyncio.sleep(self.min_interval_s - (now - self._last_sent_ts))
            await self.app.bot.send_message(self.default_chat_id, text)
            self._last_sent_ts = asyncio.get_event_loop().time()
        except Exception as e:
            logger.warning("telegram send failed: %s", e)

    async def _send_open(self, symbol: str, side: str, entry: float,
                          sl: float, tp1: float, tp2: float, qty: float, lev: int,
                          regime: str = "", conf: float = 0.0, reason: str = ""):
        msg = (
            f"🟢 OPEN {side.upper()}  | {symbol}\n"
            f"Precio: {_fmt_num(entry)}\n"
            f"SL: {_fmt_price_with_pct(sl, entry)}    "
            f"TP1: {_fmt_price_with_pct(tp1, entry)}    "
            f"TP2: {_fmt_price_with_pct(tp2, entry)}\n"
            f"Qty: {_fmt_num(qty, 6)}      Lev: x{lev}\n"
        )
        info = []
        if regime:
            info.append(f"Régimen: {str(regime).upper()}")
        if conf:
            info.append(f"Conf: {_fmt_num(conf, 2)}")
        if info:
            msg += " | ".join(info) + "\n"
        if reason:
            msg += f"Motivo: {reason}\n"
        await self._safe_send(msg)

    async def _send_tp1(self, symbol: str, side: str, price: float, entry: float,
                          qty_closed: float, pnl_partial: float, qty_remaining: float):
        msg = (
            f"✅ TP1 HIT      | {symbol} ({side.upper()})\n"
            f"Precio: {_fmt_price_with_pct(price, entry)}\n"
            f"Qty cerrada: {_fmt_num(qty_closed, 6)}    Qty remanente: {_fmt_num(qty_remaining, 6)}\n"
            f"PnL parcial: ${_fmt_num(pnl_partial)}"
        )
        await self._safe_send(msg)

    async def _send_trailing(self, symbol: str, side: str, new_sl: float, entry: float):
        await self._safe_send(
            f"🧷 TRAILING     | {symbol} ({side.upper()})\n"
            f"Nuevo SL: {_fmt_price_with_pct(new_sl, entry)}"
        )

    async def _send_close(self, kind: str, symbol: str, side: str, entry: float,
                          price: float, qty: float, pnl: float):
        tag = "🔴 SL" if kind == "SL" else ("✅ TP" if kind == "TP" else "🟡 CLOSE")
        msg = (
            f"{tag}             | {symbol} ({side.upper()})\n"
            f"Cierre: {_fmt_price_with_pct(price, entry)}\n"
            f"Qty: {_fmt_num(qty, 6)}    PnL: ${_fmt_num(pnl)}"
        )
        await self._safe_send(msg)

    async def _send_reject(self, symbol: str, side: str, code: str, detail: str = ""):
        txt = f"❌ NO-ENTRY {symbol} {side.upper()} — {code}"
        if detail:
            txt += f" ({detail})"
        await self._safe_send(txt)

# =========================
# Arranque del bot (con flags)
# =========================
async def run_telegram_bot(token: str, engine) -> Application:
    """Inicializa y lanza un bot básico con polling usando telegram.ext v20."""

    application = Application.builder().token(token).build()
    application.bot_data["engine"] = engine
    try:
        application.user_data["engine"] = engine
    except Exception:
        pass

    register_commands(application)
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    return application


async def start_telegram_bot(app, config):
    config_dict = config if isinstance(config, dict) else _app_config(app)

    application = getattr(app, "telegram_app", None)
    if application is None:
        application = setup_telegram_bot(app)
        if application is None:
            return
        setattr(app, "telegram_app", application)

    tconf = (config_dict or {}).get("telegram", {}) if isinstance(config_dict, dict) else {}
    inline_commands = bool(tconf.get("inline_commands", False))  # ← por defecto OFF
    reports_in_bot = bool(tconf.get("reports_in_bot", False))   # ← por defecto OFF

    default_chat_id = _chat_id_from_env()

    notifier = getattr(app, "notifier", None)
    if notifier is None:
        notifier = TelegramNotifier(application, config_dict, default_chat_id=default_chat_id)
        setattr(app, "notifier", notifier)
    else:
        notifier.attach_app(application)
        if default_chat_id is not None and getattr(notifier, "default_chat_id", None) is None:
            notifier.set_default_chat(default_chat_id)

    setattr(app, "telegram", notifier)
    application.bot_data["notifier"] = notifier

    # ============ (opcional) comandos inline ============
    if inline_commands:
        logger.info(
            "Inline commands habilitados: el router central procesa texto libre con alias normalizados."
        )
    else:
        logger.info("Inline commands deshabilitados (router central activo).")

    # ============ (opcional) reportes diarios/semanales en ESTE bot ============
    if reports_in_bot and not getattr(application, "_chaulet_reports_scheduled", False):
        try:
            j = application.job_queue
            tz = _tz()
            j.run_daily(lambda c: notifier.app.create_task(_report_periodic(notifier, days=1)),
                         time=dtime(hour=7, minute=0, tzinfo=tz), name="daily_report")
            j.run_daily(lambda c: notifier.app.create_task(_report_periodic(notifier, days=7)),
                         time=dtime(hour=7, minute=1, tzinfo=tz), days=(6,), name="weekly_report")
            logger.info("Reportes diarios/semanales programados en telegram_bot")
            setattr(application, "_chaulet_reports_scheduled", True)
        except Exception as e:
            logger.warning("No job_queue; reportes deshabilitados: %s", e)
    elif not reports_in_bot:
        logger.info("Reportes en telegram_bot deshabilitados (usa ReportingScheduler)")

    await application.initialize()
    # MUY IMPORTANTE: si alguna vez usaste webhook, hay que limpiarlo
    try:
        await application.bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        logger.info("No webhook to delete or failed to delete: %s", e)
    await application.start()
    logger.info("Telegram bot started")
    await application.updater.start_polling(drop_pending_updates=True)

# >>> MEJORA: Wrapper compatible con engine.py
def launch_telegram_bot(app, config, engine_api=None):
    """
    Arranca el bot de Telegram en segundo plano usando la config del bot/engine.
    engine_api se ignora porque 'app' YA expone la API que usa start_telegram_bot.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.create_task(start_telegram_bot(app, config))

# Reporte simple reutilizable si activás reports_in_bot
async def _report_periodic(notifier: TelegramNotifier, days: int):
    try:
        eq_csv, tr_csv = notifier.equity_csv, notifier.trades_csv
        equity_ini = equity_fin = pnl = 0.0
        total_trades = wins = losses = 0
        try:
            df_eq = pd.read_csv(eq_csv, parse_dates=["ts"])
            df_eq["ts"] = pd.to_datetime(df_eq["ts"], utc=True)
            now = pd.Timestamp.now(tz=_tz())
            since = now - pd.Timedelta(days=days)
            dfw = df_eq[df_eq["ts"] >= since.tz_convert("UTC")]
            if not dfw.empty:
                equity_ini = float(dfw["equity"].iloc[0])
                equity_fin = float(dfw["equity"].iloc[-1])
                # Corregido: pnl debe ser la diferencia de equity, no la suma de la columna pnl del equity DF
                pnl = equity_fin - equity_ini 
        except Exception:
            pass
        try:
            df_tr = pd.read_csv(tr_csv, parse_dates=["ts"])
            df_tr["ts"] = pd.to_datetime(df_tr["ts"], utc=True)
            now = pd.Timestamp.now(tz=_tz())
            since = now - pd.Timedelta(days=days)
            dft = df_tr[df_tr["ts"] >= since.tz_convert("UTC")]
            if not dft.empty:
                total_trades = len(dft)
                wins = int((dft["pnl"] > 0).sum())
                losses = int((dft["pnl"] < 0).sum())
        except Exception:
            pass

        txt = (
            f"🗓️ Reporte {'24h' if days==1 else '7d'}\n"
            f"Equity inicial: ${_fmt_num(equity_ini)}\n"
            f"Equity final:  ${_fmt_num(equity_fin)}\n"
            f"PnL neto:      ${_fmt_num(pnl):+,.2f}\n" # Agregamos formato de signo
            f"Trades: {total_trades} (W:{wins}/L:{losses})"
        )
        await notifier._safe_send(txt)
    except Exception as e:
        logger.warning("report periodic failed: %s", e)
