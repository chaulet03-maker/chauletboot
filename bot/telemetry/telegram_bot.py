import os
import logging
import asyncio
import sqlite3
import re
import inspect
import time
from collections import deque
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from logging_setup import LOG_DIR, LOG_FILE
from time_fmt import fmt_ar
from config import S
from bot.motives import MOTIVES
from bot.mode_manager import get_mode
from bot.identity import get_bot_id
from bot.ledger import bot_position
from bot.pnl import pnl_summary_bot
from bot.runtime_state import get_equity_sim, set_equity_sim
from bot.settings_utils import get_val, read_config_raw
from bot.telemetry.command_registry import CommandRegistry, normalize
from bot.telemetry.formatter import open_msg
from state_store import load_state, update_open_position
import trading

logger = logging.getLogger("telegram")

REGISTRY = CommandRegistry()


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


# ==== Helpers de TP/SL por % del EQUITY y métricas de presentación ====


def _side_from_qty(qty: float) -> str:
    return "LONG" if qty > 0 else ("SHORT" if qty < 0 else "FLAT")


def _target_from_equity_pct(entry: float, qty_abs: float, equity: float, side: str, pct: float) -> float:
    """pct puede ser + (TP) o - (SL). Basado en % del equity."""
    if qty_abs <= 0 or equity <= 0 or entry <= 0:
        return entry
    dollars = equity * abs(pct) / 100.0  # cuánto querés ganar/perder en USD
    if side == "LONG":
        sign = +1 if pct > 0 else -1
    else:  # SHORT
        sign = -1 if pct > 0 else +1
    return entry + sign * (dollars / qty_abs)


def _pcts_for_target(entry: float, target: float, qty_abs: float, equity: float, side: str):
    """
    Devuelve:
      price_pct: % de movimiento de precio desde entry (con signo de 'beneficio' para ese side)
      pnl_pct_equity: % de impacto esperado en PnL sobre el equity (con signo)
    """
    if entry <= 0 or qty_abs <= 0:
        return 0.0, 0.0
    dir_sign = 1.0 if side == "LONG" else -1.0
    price_pct = dir_sign * ((target - entry) / entry) * 100.0
    pnl_usd = dir_sign * (target - entry) * qty_abs
    pnl_pct_equity = (pnl_usd / equity) * 100.0 if equity > 0 else 0.0
    return price_pct, pnl_pct_equity


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
            tp_price = _pick(pos_state, "tp", "tp_price", "take_profit")
            sl_price = _pick(pos_state, "sl", "sl_price", "stop_loss")
    except Exception:
        pass

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
        price_pct, pnl_pct_equity = _pcts_for_target(avg_val, target, qty_abs, equity, side)
        return (
            f"• {name}: {target:,.2f} "
            f"({price_pct:+.2f}% precio | PnL {pnl_pct_equity:+.2f}%)"
        )

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


def _get_equity_fraction(engine) -> float:
    # PRIORIDAD: ENV -> engine.config -> default
    env = os.getenv("EQUITY_PCT")
    if env:
        try:
            f = float(env)
            if 0 < f <= 1:
                return f
        except Exception:
            pass
    cfg = getattr(engine, "config", {}) or {}
    frac = cfg.get("order_sizing", {}).get("default_pct")
    if isinstance(frac, (int, float)) and 0 < frac <= 1:
        return float(frac)
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
    side = str(status.get("side", "FLAT")).upper()
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
    if not text:
        return False

    broker = getattr(trading, "BROKER", None)
    if broker is None:
        return False

    position = _bot_position_info(engine)
    if position is None:
        return False

    lower = text.lower()
    qty_abs = float(position.get("qty") or 0.0)
    entry = float(position.get("entry") or 0.0)
    side_now = position["side"]
    symbol_norm = position["symbol"]
    symbol_conf = position.get("symbol_conf", symbol_norm)
    exchange = getattr(engine, "exchange", None)

    trader = getattr(engine, "trader", None)
    try:
        equity = float(trader.equity()) if trader is not None else 0.0
    except Exception:
        equity = 0.0

    async def _current_mark() -> Optional[float]:
        mark_val: Optional[float] = None
        if exchange is not None and hasattr(exchange, "get_current_price"):
            try:
                mark_raw = await exchange.get_current_price(symbol_norm)
                mark_val = float(mark_raw)
            except Exception:
                mark_val = None
        if mark_val is None:
            try:
                mark_val = float(position.get("mark")) if position.get("mark") is not None else None
            except Exception:
                mark_val = None
        return mark_val

    m_sl_abs = re.search(r"\bsl\s*\$([0-9][\d.,]*)\b", lower)
    m_sl_pct = re.search(r"\bsl\s*([+-]?)(\d+(?:\.\d+)?)\b", lower) if not m_sl_abs else None
    if m_sl_pct or m_sl_abs:
        if qty_abs <= 0 or entry <= 0:
            await reply_md("No tengo datos de la posición para calcular SL.")
            return True
        if m_sl_pct:
            sign = m_sl_pct.group(1)
            try:
                pct = float(m_sl_pct.group(2))
            except Exception:
                await reply_md("Formato SL: `sl +5`, `sl -2` o `sl 105000`")
                return True
            if equity <= 0 and trader is not None and not S.PAPER:
                try:
                    equity = float(trader.equity(force_refresh=True))
                except Exception:
                    logger.debug("No se pudo refrescar equity live para SL %%.", exc_info=True)
            if equity <= 0:
                if S.PAPER:
                    await reply_md("Equity <= 0. Configuralo con `equity 1000` antes de usar SL %.")
                else:
                    await reply_md("No pude obtener el equity live desde Binance. Intentá nuevamente en unos segundos.")
                return True
            pct_signed = (+pct if sign == "+" else (-pct if sign == "-" else -pct))
            target = _target_from_equity_pct(entry, qty_abs, equity, side_now, pct_signed)
        else:
            try:
                raw = m_sl_abs.group(1)  # type: ignore[union-attr]
                target = float(str(raw).replace(".", "").replace(",", "."))
            except Exception:
                await reply_md("Formato SL: `sl +5`, `sl -2` o `sl 105000`")
                return True
        price_pct, pnl_pct_equity = _pcts_for_target(entry, target, qty_abs, equity, side_now)
        try:
            broker.update_protections(symbol_norm, side_now, qty_abs, sl=target)
            update_open_position(symbol_conf, sl=target)
        except Exception as exc:
            await reply_md(f"No pude actualizar SL: {exc}")
            return True
        mark_val = await _current_mark()
        crossed = False
        if mark_val is not None:
            if side_now == "LONG" and mark_val <= target:
                crossed = True
            if side_now == "SHORT" and mark_val >= target:
                crossed = True
        if crossed:
            def _classify_close_result(payload: Any) -> str | None:
                if not isinstance(payload, dict):
                    return None
                info = str(payload.get("info") or "").strip().lower()
                if info and "sin posición" in info:
                    return "no_position"
                try:
                    ok_val = payload.get("ok")
                    if ok_val is False:
                        return "failed"
                except Exception:
                    pass
                executed = 0.0
                for key in (
                    "executedQty",
                    "executed_qty",
                    "filled",
                    "fills_qty",
                    "cumQty",
                    "cumqty",
                ):
                    if key not in payload:
                        continue
                    try:
                        executed = max(executed, abs(float(payload.get(key) or 0.0)))
                    except Exception:
                        continue
                if executed > 0:
                    return "filled"
                status = str(payload.get("status") or "").strip().upper()
                if status in {"FILLED", "PARTIALLY_FILLED"}:
                    return "filled"
                return None

            close_result: Any = None
            try:
                close_result = trading.close_bot_position_market()
            except Exception as exc:
                logger.debug("No se pudo cerrar posición tras cruzar SL: %s", exc)

            classification = _classify_close_result(close_result)
            if classification == "filled":
                await reply_md(f"✅ SL alcanzado. Posición cerrada (SL: ${target:,.2f}).")
            elif classification == "no_position":
                await reply_md("SL cruzado, pero no hay posición abierta para cerrar.")
            else:
                await reply_md(
                    f"SL cruzado (mark {_num(mark_val)}). Intentá cerrar manualmente."
                )
            return True
        await reply_md(
            f"✅ SL actualizado a ${target:,.2f} "
            f"({price_pct:+.2f}% precio | PnL {pnl_pct_equity:+.2f}%)"
        )
        return True

    m_tp_abs = re.search(r"\btp\s*\$([0-9][\d.,]*)\b", lower)
    m_tp_pct = re.search(r"\btp\s*([+-]?)(\d+(?:\.\d+)?)\b", lower) if not m_tp_abs else None
    if m_tp_pct or m_tp_abs:
        if qty_abs <= 0 or entry <= 0:
            await reply_md("No tengo datos de la posición para calcular TP.")
            return True
        if m_tp_pct:
            sign = m_tp_pct.group(1)
            try:
                pct = float(m_tp_pct.group(2))
            except Exception:
                await reply_md("Formato TP: `tp +5` o `tp 109000`")
                return True
            if equity <= 0:
                await reply_md("Equity <= 0. Configuralo con `equity 1000` antes de usar TP %.")
                return True
            pct_signed = (+pct if sign == "+" else (-pct if sign == "-" else +pct))
            target = _target_from_equity_pct(entry, qty_abs, equity, side_now, pct_signed)
        else:
            try:
                raw = m_tp_abs.group(1)  # type: ignore[union-attr]
                target = float(str(raw).replace(".", "").replace(",", "."))
            except Exception:
                await reply_md("Formato TP: `tp +5` o `tp 109000`")
                return True
        price_pct, pnl_pct_equity = _pcts_for_target(entry, target, qty_abs, equity, side_now)
        try:
            broker.update_protections(symbol_norm, side_now, qty_abs, tp=target)
            update_open_position(symbol_conf, tp=target)
        except Exception as exc:
            await reply_md(f"No pude actualizar TP: {exc}")
            return True
        mark_val = await _current_mark()
        reached = False
        if mark_val is not None:
            if side_now == "LONG" and mark_val >= target:
                reached = True
            if side_now == "SHORT" and mark_val <= target:
                reached = True
        if reached:
            closed = False
            try:
                trading.close_bot_position_market()
                closed = True
            except Exception:
                closed = False
            if closed:
                await reply_md(f"✅ TP alcanzado. Posición cerrada (TP: ${target:,.2f}).")
            else:
                await reply_md(
                    f"TP cruzado (mark {_num(mark_val)}). Intentá cerrar manualmente."
                )
            return True
        await reply_md(
            f"✅ TP actualizado a ${target:,.2f} "
            f"({price_pct:+.2f}% precio | PnL {pnl_pct_equity:+.2f}%)"
        )
        return True

    return False


def _position_status_message(engine) -> str:
    symbol_default = _default_symbol(engine)
    try:
        st = trading.POSITION_SERVICE.get_status() if trading.POSITION_SERVICE else None
    except Exception as exc:
        logger.debug("posicion/status error: %s", exc)
        st = None
    if not st or (st.get("side", "FLAT").upper() == "FLAT"):
        return f"Estado Actual: Sin posición\n----------------\nSímbolo: {symbol_default}"
    side = (st.get("side") or "").upper()
    symbol = st.get("symbol", symbol_default)
    entry_price = float(st.get("entry_price", 0.0) or 0.0)
    pnl = float(st.get("pnl", 0.0) or 0.0)
    return (
        "Estado Actual: Posición Abierta\n"
        "----------------\n"
        f"Símbolo: {symbol}\n"
        f"Lado: {side}\n"
        f"Precio de Entrada: ${entry_price:.2f}\n"
        f"PNL Actual: ${pnl:+.2f}\n"
    )


async def posicion_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return

    engine = _get_engine_from_context(context)
    if engine is None:
        await message.reply_text("No pude acceder al engine para consultar la posición.")
        return

    # Forzar upgrade a real para coherencia con /posiciones
    exchange = getattr(engine, "exchange", None)
    if exchange is not None and hasattr(exchange, "upgrade_to_real_if_needed"):
        try:
            await exchange.upgrade_to_real_if_needed()
        except Exception as exc:
            logger.debug("upgrade_to_real_if_needed desde /posicion falló: %s", exc)

    # Mostrar SOLO la posición del BOT (store local)
    try:
        st = trading.POSITION_SERVICE.get_status() if trading.POSITION_SERVICE else None
    except Exception:
        st = None
    if not st or (st.get("side", "FLAT").upper() == "FLAT"):
        reply_text = "📍 Posición del BOT: *SIN POSICIÓN*"
    else:
        q = float(st.get("qty") or st.get("size") or 0.0)
        side = (st.get("side") or "").upper()
        entry = float(st.get("entry_price") or 0.0)
        mark = float(st.get("mark") or 0.0)
        sym = st.get("symbol") or (engine.config or {}).get("symbol", "?")
        qty_signed = _signed_qty(q, side)
        reply_text = _build_bot_position_message(
            engine=engine,
            symbol=sym,
            qty=qty_signed,
            avg=entry,
            mark_val=mark,
        )
    await message.reply_text(reply_text, parse_mode="Markdown")


async def posiciones_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lista todas las posiciones abiertas; la del bot va en **negrita**."""
    message = update.effective_message
    if message is None:
        return

    engine = _get_engine_from_context(context)
    if engine is None:
        await message.reply_text("No pude acceder al engine para consultar posiciones.")
        return

    symbol_bot = (getattr(engine, "config", {}) or {}).get("symbol", "BTC/USDT")
    exchange = getattr(engine, "exchange", None)
    if exchange is None:
        await message.reply_text("Exchange no disponible.")
        return

    # 1) Forzar upgrade a real
    try:
        if hasattr(exchange, "upgrade_to_real_if_needed"):
            await exchange.upgrade_to_real_if_needed()
    except Exception as exc:
        logger.debug("upgrade_to_real_if_needed desde /posiciones falló: %s", exc)
    # 2) Traer TODAS las posiciones del exchange (usuario + bot)
    try:
        live_positions = await exchange.list_open_positions()
    except Exception as exc:  # pragma: no cover - robustez
        await message.reply_text(f"No pude leer posiciones: {exc}")
        return

    blocks: list[str] = []
    # Si el exchange no reporta, usar respaldo del BOT
    if not live_positions:
        try:
            st = trading.POSITION_SERVICE.get_status() if trading.POSITION_SERVICE else None
        except Exception:
            st = None
        if st and (st.get("side", "FLAT").upper() != "FLAT"):
            q = float(st.get("qty") or st.get("size") or 0.0)
            side = (st.get("side") or "").upper()
            entry = float(st.get("entry_price") or 0.0)
            mark = float(st.get("mark") or 0.0)
            sym = st.get("symbol") or symbol_bot
            qty_signed = _signed_qty(q, side)
            blocks.append(
                _build_bot_position_message(
                    engine=engine,
                    symbol=sym,
                    qty=qty_signed,
                    avg=entry,
                    mark_val=mark,
                )
            )
            await message.reply_text("\n".join(blocks), parse_mode="Markdown")
            return
        await message.reply_text("No hay posiciones abiertas.")
        return

    # Hay posiciones en el exchange → mostrarlas con el mismo formato
    for pos in live_positions:
        symbol = pos.get("symbol") or pos.get("symbolName") or symbol_bot
        side = (pos.get("side") or "").upper()
        size = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("size") or 0.0)
        entry = float(pos.get("entryPrice") or pos.get("avgEntryPrice") or pos.get("entry") or 0.0)
        mark = float(pos.get("markPrice") or pos.get("mark") or 0.0)
        upnl = float(pos.get("unrealizedPnl") or pos.get("unrealized_pnl") or 0.0)
        is_bot_symbol = str(symbol).upper() == str(symbol_bot).upper()
        mode_txt = "real" if not S.PAPER else "simulado"
        qty_signed = _signed_qty(size, side)
        if is_bot_symbol:
            blocks.append(
                _build_bot_position_message(
                    engine=engine,
                    symbol=symbol,
                    qty=qty_signed,
                    avg=entry,
                    mark_val=mark,
                )
            )
        else:
            blocks.append(
                _format_position_block(
                    symbol=symbol,
                    side=side,
                    qty=size,
                    entry=entry,
                    mark=mark,
                    pnl=upnl,
                    mode_txt=mode_txt,
                    opened_at=None,
                    is_bot=False,
                )
            )
    await message.reply_text("\n\n".join(blocks), parse_mode="Markdown")


def _sym(s):
    return (s or "BTCUSDT").replace("/", "").upper()


async def _cmd_open(engine, reply, raw_txt):
    t = (raw_txt or "").strip().lower()
    m = re.search(r"\bopen\s+(long|short)\s+x\s*(\d+)\b", t)
    if not m:
        return await reply("Formato: open long x5  |  open short x10")

    side_txt = m.group(1).upper()   # LONG/SHORT
    lev      = int(m.group(2))
    cfg      = getattr(engine, "config", {}) or {}
    symbol   = _sym(cfg.get("symbol"))
    side     = "BUY" if side_txt=="LONG" else "SELL"

    exchange = getattr(engine, "exchange", None)
    broker = getattr(engine, "broker", None)
    if exchange is None or broker is None:
        return await reply("No pude obtener acceso a exchange/broker para abrir.")

    # precio actual
    try:
        px = await exchange.get_current_price(symbol)
    except Exception:
        px = None
    if not px:
        return await reply("No pude obtener precio actual del símbolo.")

    # equity para sizing (como pediste: el que seteás con 'equity', en ambos modos)
    try:
        eq = float(engine.trader.equity())
    except Exception:
        eq = 0.0
    if eq <= 0:
        return await reply("Equity = 0. Setealo con: equity 1200")

    qty = (eq * lev) / float(px)
    try:
        round_qty_fn = getattr(exchange, "round_qty", None)
        if callable(round_qty_fn):
            maybe_qty = round_qty_fn(symbol, qty)
            qty = await maybe_qty if inspect.isawaitable(maybe_qty) else maybe_qty
    except Exception:
        pass
    if qty <= 0:
        return await reply("Cantidad final <= 0 (minQty/minNotional). Subí equity o bajá leverage.")

    try:
        res = await broker.place_market_order(symbol=symbol, side=side, quantity=qty, leverage=lev)
    except Exception as e:
        return await reply(f"Fallo al abrir: {e}")

    entry_price = float(
        res.get("price") if isinstance(res, dict) and res.get("price") is not None else px
    )

    try:
        # Enganchá TP/SL de tu estrategia (si tenés método)
        strategy = getattr(engine, "strategy", None)
        attach = getattr(strategy, "attach_tp_sl", None)
        if callable(attach):
            maybe_coro = attach(symbol=symbol, side_txt=side_txt, entry_price=entry_price)
            if inspect.isawaitable(maybe_coro):
                await maybe_coro
    except Exception:
        pass

    # ======= TP / SL (intento 1: que la estrategia los devuelva) =======
    tp = sl = None
    try:
        strategy = getattr(engine, "strategy", None)
        attach = getattr(strategy, "attach_tp_sl", None)
        if callable(attach):
            out = attach(symbol=symbol, side_txt=side_txt, entry_price=entry_price, return_levels=True)
            out = await out if inspect.isawaitable(out) else out
            if isinstance(out, dict):
                tp = out.get("tp") or out.get("tp_price") or out.get("take_profit")
                sl = out.get("sl") or out.get("sl_price") or out.get("stop_loss")
    except Exception:
        pass

    # ======= TP / SL (fallback por config si no hay niveles de la estrategia) =======
    from bot.settings_utils import get_val, read_config_raw
    from config import S

    raw_cfg = read_config_raw()
    tp_pct = get_val(S, raw_cfg, "tp_pct", default=None)   # ej: 0.01  -> 1%
    sl_pct = get_val(S, raw_cfg, "sl_pct", default=None)   # ej: 0.005 -> 0.5%
    if (tp is None or sl is None) and isinstance(tp_pct, (int, float)) and isinstance(sl_pct, (int, float)):
        if side_txt == "LONG":
            tp = tp or (entry_price * (1 + float(tp_pct)))
            sl = sl or (entry_price * (1 - float(sl_pct)))
        else:
            tp = tp or (entry_price * (1 - float(tp_pct)))
            sl = sl or (entry_price * (1 + float(sl_pct)))

    # ======= Mensaje final claro =======
    try:
        margin = float(engine.trader.equity()) if hasattr(engine, "trader") else float(eq)
    except Exception:
        margin = float(eq)
    notional = float(qty) * float(entry_price) * max(1, lev)
    tp1 = float(tp or 0.0)
    tp2 = tp1
    slv = float(sl or 0.0)
    latency_ms = 0

    return await reply(
        open_msg(
            symbol,
            "long" if side_txt == "LONG" else "short",
            margin,
            lev,
            notional,
            float(entry_price),
            tp1,
            tp2,
            slv,
            "MARKET",
            latency_ms,
        )
    )


async def open_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return

    engine = _get_engine_from_context(context)
    if engine is None:
        await message.reply_text("No pude acceder al engine para abrir la operación.")
        return

    reply_md = lambda txt: message.reply_text(txt, parse_mode="Markdown")
    await _cmd_open(engine, reply_md, message.text or "")


async def _cmd_estado(engine, reply):
    cfg = getattr(engine, "config", {}) or {}
    symbol = (cfg.get("symbol") or "BTCUSDT").replace("/", "").upper()
    mode = _get_mode_from_engine(engine)      # 'live' | 'paper'
    exchange = getattr(engine, "exchange", None)
    trader = getattr(engine, "trader", None)

    # EQUITY correcto según modo
    if mode == "live":
        try:
            equity = float(await exchange.get_account_equity()) if exchange else 0.0
        except Exception:
            equity = 0.0
    else:
        try:
            equity = float(trader.equity()) if trader else 0.0  # el que seteás con 'equity'
        except Exception:
            equity = 0.0

    # Mark para PnL no realizado
    async def _mark(sym):
        try:
            if not exchange:
                return None
            px = await exchange.get_current_price(sym)
            return float(px)
        except Exception:
            return None

    # PnL del BOT (no del exchange)
    pnl = await pnl_summary_bot(mode=("live" if mode=="live" else "paper"), mark_provider=_mark)
    d, w = pnl["daily"], pnl["weekly"]

    return await reply(
        f"Modo: *{'REAL' if mode=='live' else 'SIMULADO'}*\n"
        f"Símbolo: {symbol}\n"
        f"saldo: {equity:,.2f}\n"
        f"PnL Diario: {d['total']:+.2f} (R={d['realized']:+.2f} | U={d['unrealized']:+.2f})\n"
        f"PnL Semanal: {w['total']:+.2f} (R={w['realized']:+.2f} | U={w['unrealized']:+.2f})"
    )


async def estado_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return

    engine = _get_engine_from_context(context)
    if engine is None:
        await message.reply_text("No pude acceder al engine para consultar el estado.")
        return

    reply_md = lambda txt: message.reply_text(txt, parse_mode="Markdown")
    await _cmd_estado(engine, reply_md)


async def rendimiento_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Calcula y envía las estadísticas de rendimiento desde la base de datos."""
    chat = update.effective_chat if update else None
    if chat is None:
        return

    application = getattr(context, "application", None)
    engine = application.bot_data.get("engine") if application else None
    if engine is None:
        await chat.send_message("No pude acceder al motor (engine).")
        return

    db_path = (
        getattr(engine, "db_path", None)
        or getattr(engine, "metrics_db_path", None)
        or os.getenv("PERF_DB_PATH")
        or "data/perf.db"
    )
    path = Path(db_path)
    if not path.exists():
        await chat.send_message("No encuentro la base de rendimiento (data/perf.db).")
        return

    try:
        with sqlite3.connect(str(path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*), SUM(pnl), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) FROM trades"
            )
            total_trades, total_pnl, wins = cursor.fetchone()

        if not total_trades:
            reply_text = "Aún no hay operaciones completadas en el historial."
        else:
            total_pnl = total_pnl or 0
            wins = wins or 0
            losses = total_trades - wins
            winrate = (wins / total_trades) * 100 if total_trades > 0 else 0

            reply_text = (
                "**Rendimiento Histórico (Base de Datos)**\n"
                "----------------------------------\n"
                f"📈 **Trades Totales:** {total_trades}\n"
                f"✅ **Ganadas:** {wins}\n"
                f"❌ **Perdidas:** {losses}\n"
                f"🎯 **Winrate:** {winrate:.2f}%\n"
                f"💰 **PNL Neto Total:** {total_pnl:+.2f} USD"
            )
    except Exception as exc:
        reply_text = f"Error al leer la base de datos de rendimiento: {exc}"

    await chat.send_message(reply_text, parse_mode="Markdown")


async def cerrar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "Engine no disponible para cerrar posiciones.")
        return
    try:
        ok = await engine.close_all()
        if ok:
            await _reply_chunks(update, "✅ Cerré la **posición del BOT**.")
        else:
            await _reply_chunks(update, "⚠️ No había **posición del BOT** para cerrar.")
    except Exception as exc:
        await _reply_chunks(update, f"No pude cerrar la **posición del BOT**: {exc}")


async def control_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toma control de la posición REAL actual: fuerza modo REAL y sincroniza desde el exchange."""

    engine = _get_engine_from_context(context)
    message = update.effective_message
    if engine is None or message is None:
        return

    result = trading.switch_mode("real")
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
    """
    SL por % del equity (positivo o negativo) o por precio fijo.
      • sl            → muestra SL actual (si hay)
      • sl 5          → 5% (equity)  [equivale a +5]
      • sl +5         → +5% (equity)  → SL en ganancia
      • sl -5         → -5% (equity)  → SL en pérdida
      • sl 108000     → SL fijo a $108,000
      • sl $108000    → idem
    """

    message = update.effective_message
    engine = _get_engine_from_context(context)
    if engine is None or message is None:
        return

    def reply_md(text: str):
        return message.reply_text(text, parse_mode="Markdown")

    symbol_conf = (engine.config or {}).get("symbol", "BTC/USDT")
    symbol_norm = str(symbol_conf).replace("/", "")
    state = load_state() or {}
    open_positions = state.get("open_positions") or {}
    open_pos = open_positions.get(symbol_norm) or open_positions.get(symbol_conf)

    position_info = _bot_position_info(engine)
    if open_pos is None and position_info is None:
        await reply_md("No hay **posición del BOT** abierta.")
        return

    if position_info is not None:
        symbol_norm = position_info.get("symbol", symbol_norm)
        symbol_conf = position_info.get("symbol_conf", symbol_conf)
        side_now = str(position_info.get("side") or "LONG").upper()
        entry = float(position_info.get("entry") or position_info.get("entry_price") or 0.0)
        qty_abs = abs(float(position_info.get("qty") or 0.0))
        mark_cached = position_info.get("mark")
    else:
        side_now = str((open_pos or {}).get("side") or "LONG").upper()
        entry = _first_float(
            (open_pos or {}).get("entry_price"),
            (open_pos or {}).get("entry"),
            default=0.0,
        )
        qty_abs = abs(
            _first_float(
                (open_pos or {}).get("qty"),
                (open_pos or {}).get("contracts"),
                (open_pos or {}).get("size"),
                default=0.0,
            )
        )
        mark_cached = (open_pos or {}).get("mark")

    if qty_abs <= 0 or entry <= 0:
        await reply_md("No tengo datos de la posición para calcular SL.")
        return

    trader = getattr(engine, "trader", None)
    try:
        equity = float(trader.equity()) if trader is not None else 0.0
    except Exception:
        equity = 0.0

    exchange = getattr(engine, "exchange", None)

    async def _current_mark() -> Optional[float]:
        if exchange is not None and hasattr(exchange, "get_current_price"):
            try:
                mark_value = await exchange.get_current_price(symbol_conf)
                if mark_value is not None:
                    return float(mark_value)
            except Exception:
                logger.debug("sl_command: no se pudo obtener mark live", exc_info=True)
        for candidate in (
            mark_cached,
            (open_pos or {}).get("mark") if open_pos else None,
        ):
            if candidate in (None, ""):
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
        return None

    def _pick_level(data: dict | None, *keys: str) -> Optional[float]:
        if not isinstance(data, dict):
            return None
        for key in keys:
            if key in data and data[key] not in (None, ""):
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    continue
        return None

    text_raw = (message.text or "").strip()
    if re.match(r"^/?sl\s*$", text_raw, re.IGNORECASE):
        current = _pick_level(open_pos, "sl", "sl_price", "stop_loss")
        if current is not None:
            price_pct, pnl_pct_equity = _pcts_for_target(entry, current, qty_abs, equity, side_now)
            await reply_md(
                f"SL actual: {_num(current)} ({price_pct:+.2f}% precio | PnL {pnl_pct_equity:+.2f}%)"
            )
        else:
            await reply_md("SL actual: —")
        await reply_md("Uso: `sl +5`, `sl -2`, `sl 105000`")
        return

    m_pct = re.match(r"^/?sl\s*([+\-]?\d+(?:\.\d+)?)%?\s*$", text_raw, re.IGNORECASE)
    m_abs = None
    if not m_pct:
        m_abs = re.match(r"^/?sl\s*\$?\s*(\d+(?:\.\d+)?)\s*$", text_raw, re.IGNORECASE)
    if not m_pct and not m_abs:
        await reply_md("Formato SL: `sl +5`, `sl -2` o `sl 105000`")
        return

    if m_pct:
        pct = float(m_pct.group(1))
        target = _target_from_equity_pct(entry, qty_abs, equity, side_now, pct)
    else:
        target = float(m_abs.group(1))  # type: ignore[union-attr]

    price_pct, pnl_pct_equity = _pcts_for_target(entry, target, qty_abs, equity, side_now)

    broker = getattr(trading, "BROKER", None)
    if broker is None:
        await reply_md("No pude actualizar SL: broker no inicializado.")
        return

    try:
        broker.update_protections(symbol_norm, side_now, qty_abs, sl=target)
        update_open_position(symbol_conf, sl=target)
    except Exception as exc:
        await reply_md(f"No pude actualizar SL: {exc}")
        return

    mark_val = await _current_mark()
    crossed = False
    if mark_val is not None:
        if side_now == "LONG" and mark_val <= target:
            crossed = True
        if side_now == "SHORT" and mark_val >= target:
            crossed = True
    if crossed:
        try:
            trading.close_bot_position_market()
            await reply_md(f"✅ SL alcanzado. Posición cerrada (SL: {_num(target)}).")
            return
        except Exception:
            await reply_md(
                f"SL cruzado (mark {_num(mark_val)}). Intentá cerrar manualmente."
            )
            return

    await reply_md(
        f"✅ SL actualizado a {_num(target)} "
        f"({price_pct:+.2f}% precio | PnL {pnl_pct_equity:+.2f}%)"
    )


async def tp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Fija o muestra el TP por apalancamiento (como % del equity al abrir).
    Ejemplos:
      • tp x5 10        → setea x5 en 10%
      • tp x10 8        → setea x10 en 8%
      • tp              → muestra mapeo actual
      • tp x5           → muestra valor actual para x5
    """
    engine = _get_engine_from_context(context)
    message = update.effective_message
    if engine is None or message is None:
        return

    def _cfg() -> Dict:
        return _engine_config(engine) or {}

    txt = (message.text or "").strip().lower()

    # 1) "tp" o "tp x5" → mostrar
    m_show_one = re.match(r"^/?tp\s+x?(\d{1,3})\s*$", txt)
    m_show_all = re.match(r"^/?tp\s*$", txt)
    if m_show_all:
        cfg = _cfg()
        m = cfg.get("tp_eq_pct_by_leverage", {}) or {}
        lines = ["*TP por apalancamiento*"]
        if isinstance(m, dict) and m:
            for k in sorted([str(x) for x in m.keys()], key=lambda s: int(s)):
                v = float(m[k])
                v = v / 100.0 if v >= 1.0 else v
                lines.append(f"• x{int(k)}: {v*100:.2f}% del equity")
        else:
            default_pct = float((_engine_config(engine) or {}).get("target_eq_pnl_pct", 0.10))
            for lev in (5, 10):
                lines.append(f"• x{lev}: {default_pct*100:.2f}% del equity")
        await message.reply_text("\n".join(lines), parse_mode="Markdown")
        return
    if m_show_one:
        lev = int(m_show_one.group(1))
        cfg = _cfg()
        m = cfg.get("tp_eq_pct_by_leverage", {}) or {}
        v = m.get(str(lev), None)
        if v is None:
            await message.reply_text(f"x{lev}: _sin override_ (usa default)", parse_mode="Markdown")
        else:
            v = float(v)
            v = v / 100.0 if v >= 1.0 else v
            await message.reply_text(f"x{lev}: {v*100:.2f}% del equity", parse_mode="Markdown")
        return

    # 2) "tp x5 10" → setear
    m_set = re.match(r"^/?tp\s+x?(\d{1,3})\s+([0-9]+(?:\.[0-9]+)?)%?\s*$", txt)
    if not m_set:
        await message.reply_text("Uso: `tp x5 10` | `tp x10 8` | `tp`", parse_mode="Markdown")
        return
    lev = int(m_set.group(1))
    pct_in = float(m_set.group(2))
    pct = (pct_in / 100.0) if pct_in >= 1.0 else pct_in
    cfg = _cfg()
    mapping = dict(cfg.get("tp_eq_pct_by_leverage", {}) or {})
    mapping[str(lev)] = pct
    cfg["tp_eq_pct_by_leverage"] = mapping
    # guardamos en el objeto de runtime
    if hasattr(engine, "config") and isinstance(engine.config, dict):
        engine.config.update(cfg)
    await message.reply_text(
        f"✅ TP para x{lev} fijado en {pct*100:.2f}% del equity",
        parse_mode="Markdown",
    )

    # Aplicar inmediatamente sobre la posición abierta (si coincide el leverage)
    try:
        st = load_state() or {}
        symbol_conf = (engine.config or {}).get("symbol", "BTC/USDT")
        open_positions = (st.get("open_positions") or {})
        pos_state = open_positions.get(symbol_conf.replace("/", "")) or open_positions.get(symbol_conf)
        if pos_state:
            lev_now_raw = pos_state.get("leverage")
            try:
                lev_now = int(float(lev_now_raw)) if lev_now_raw not in (None, "") else 1
            except Exception:
                lev_now = 1
            if lev_now == int(lev):
                side_now = str(pos_state.get("side") or "LONG").upper()
                entry = _first_float(
                    pos_state.get("entry_price"),
                    pos_state.get("entry"),
                    default=0.0,
                )
                qty_abs = abs(
                    _first_float(
                        pos_state.get("qty"),
                        pos_state.get("contracts"),
                        pos_state.get("size"),
                        default=0.0,
                    )
                )
                trader = getattr(engine, "trader", None)
                try:
                    equity = float(trader.equity()) if trader is not None else 0.0
                except Exception:
                    equity = 0.0
                if entry > 0 and qty_abs > 0 and equity > 0:
                    pct_for_price = pct * 100.0
                    tp_price = _target_from_equity_pct(entry, qty_abs, equity, side_now, +pct_for_price)
                    broker = getattr(trading, "BROKER", None)
                    if broker is not None:
                        try:
                            broker.update_protections(symbol_conf.replace("/", ""), side_now, qty_abs, tp=tp_price)
                            update_open_position(symbol_conf, tp=tp_price)
                        except Exception:
                            logger.debug("tp_command: no se pudo actualizar protecciones", exc_info=True)
                    exchange = getattr(engine, "exchange", None)
                    mark_val: Optional[float] = None
                    if exchange is not None and hasattr(exchange, "get_current_price"):
                        try:
                            mark_val = await exchange.get_current_price(symbol_conf)
                            if mark_val is not None:
                                mark_val = float(mark_val)
                        except Exception:
                            logger.debug("tp_command: no se pudo obtener mark", exc_info=True)
                    if mark_val is not None:
                        crossed = False
                        if side_now == "LONG" and mark_val >= tp_price:
                            crossed = True
                        if side_now == "SHORT" and mark_val <= tp_price:
                            crossed = True
                        if crossed:
                            try:
                                trading.close_bot_position_market()
                                await message.reply_text(
                                    f"✅ TP alcanzado. Posición cerrada (TP: {_num(tp_price)}).",
                                    parse_mode="Markdown",
                                )
                                return
                            except Exception:
                                await message.reply_text(
                                    f"TP cruzado (mark {_num(mark_val)}). Intentá cerrar manualmente.",
                                    parse_mode="Markdown",
                                )
                                return
    except Exception:
        logger.debug("tp_command: fallback inmediato falló", exc_info=True)


async def precio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "Engine no disponible para consultar precios.")
        return
    text = (update.effective_message.text or "").strip() if update.effective_message else ""
    parts = text.split()
    symbols: Iterable[str]
    if len(parts) >= 2:
        symbols = [parts[1].upper()]
    else:
        symbols = getattr(engine, "symbols", []) or list(getattr(engine, "price_cache", {}).keys()) or ["BTC/USDT:USDT"]
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
        engine.set_mode("paper")   # usa el método que agregaste recién
        return await reply("✅ Modo cambiado a *SIMULADO*. El bot ahora opera en simulado.")
    except Exception as e:
        return await reply("⚠️ No pude cambiar a SIMULADO (revisá logs / configuración).")


async def _cmd_modo_real(engine, reply):
    if _get_mode_from_engine(engine) == "live":
        return await reply("✅ El bot ya se encontraba en *MODO REAL*.")
    try:
        engine.set_mode("live")    # usa el método que agregaste recién
        return await reply("✅ Modo cambiado a *REAL*. El bot ahora opera en real.")
    except Exception as e:
        return await reply("⚠️ No pude cambiar a REAL (revisá logs / configuración de modo).")


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
        await engine.close_all()   # solo la del BOT
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
        match = re.match(r"(?i)ajustar\s+([\w.]+)\s+(.+)$", text.strip())
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
    match = re.search(r"equity\s+(\d+(?:[.,]\d+)?)(\s*%?)", txt)
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
        value = float(match.group(1).replace(",", "."))
    except Exception:
        await message.reply_text("Formato: equity 1200  |  equity 25%")
        return

    suffix = match.group(2) or ""
    if "%" in suffix:
        if not (1.0 <= value <= 100.0):
            await message.reply_text("El porcentaje debe estar entre 1 y 100.")
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

    set_equity_sim(value)
    try:
        trader = getattr(engine, "trader", None)
        if trader is not None:
            trader.set_paper_equity(value)
    except Exception:
        logger.debug("No se pudo actualizar trader.set_paper_equity", exc_info=True)

    await message.reply_text(f"Equity seteado: {value:.2f}")


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
        aliases=["open", "abrir"],
        help_text="Abre una operación manual. Ej: open long x5",
    )
    REGISTRY.register(
        "diag",
        diag_command,
        aliases=["diagnostico", "status", "health"],
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
        help_text="Ver o setear SL global como % del equity. Ej: `sl 10` (10%)",
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
        aliases=["takeprofit", "tp%"],
        help_text="Fijá o mostrá el TP por apalancamiento. Ej: `tp x5 10` (10%)",
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


def register_commands(application: Application) -> None:
    _populate_registry()
    if getattr(application, "_chaulet_router_registered", False):
        return

    application.add_handler(MessageHandler(filters.COMMAND, _slash_router))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), _text_router))
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
            f"Qty: {_fmt_num(qty, 6)}     Lev: x{lev}\n"
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
            f"✅ TP1 HIT     | {symbol} ({side.upper()})\n"
            f"Precio: {_fmt_price_with_pct(price, entry)}\n"
            f"Qty cerrada: {_fmt_num(qty_closed, 6)}    Qty remanente: {_fmt_num(qty_remaining, 6)}\n"
            f"PnL parcial: ${_fmt_num(pnl_partial)}"
        )
        await self._safe_send(msg)

    async def _send_trailing(self, symbol: str, side: str, new_sl: float, entry: float):
        await self._safe_send(
            f"🧷 TRAILING    | {symbol} ({side.upper()})\n"
            f"Nuevo SL: {_fmt_price_with_pct(new_sl, entry)}"
        )

    async def _send_close(self, kind: str, symbol: str, side: str, entry: float,
                          price: float, qty: float, pnl: float):
        tag = "🔴 SL" if kind == "SL" else ("✅ TP" if kind == "TP" else "🟡 CLOSE")
        msg = (
            f"{tag}         | {symbol} ({side.upper()})\n"
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
async def start_telegram_bot(app, config):
    config_dict = config if isinstance(config, dict) else _app_config(app)

    application = getattr(app, "telegram_app", None)
    if application is None:
        application = setup_telegram_bot(app)
        if application is None:
            return
        setattr(app, "telegram_app", application)

    tconf = (config_dict or {}).get("telegram", {}) if isinstance(config_dict, dict) else {}
    inline_commands = bool(tconf.get("inline_commands", False))   # ← por defecto OFF
    reports_in_bot = bool(tconf.get("reports_in_bot", False))     # ← por defecto OFF

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
    await application.updater.start_polling()

# >>> MEJORA: Wrapper compatible con engine.py
def run_telegram_bot(app, config, engine_api=None):
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
                pnl = float(dfw["pnl"].sum())
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
            f"Equity final:   ${_fmt_num(equity_fin)}\n"
            f"PnL neto:       ${_fmt_num(pnl)}\n"
            f"Trades: {total_trades} (W:{wins}/L:{losses})"
        )
        await notifier._safe_send(txt)
    except Exception as e:
        logger.warning("report periodic failed: %s", e)

