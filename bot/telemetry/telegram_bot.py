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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from logging_setup import LOG_DIR, LOG_FILE
from time_fmt import fmt_ar
from config import S
from bot.motives import MOTIVES
from bot.mode_manager import get_mode
from bot.settings_utils import get_val, read_config_raw
from bot.telemetry.command_registry import CommandRegistry
import trading

logger = logging.getLogger("telegram")

REGISTRY = CommandRegistry()


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
    """Devuelve el equity% configurado como fracción.

    Prioriza, en orden:
    1) engine.config.order_sizing.default_pct
    2) getattr(engine, "order_sizes", {}).get("default_pct")
    3) Variable de entorno EQUITY_PCT
    Fallback: 1.0
    """

    try:
        cfg = getattr(engine, "config", {}) or {}
        osz = cfg.get("order_sizing") or {}
        value = osz.get("default_pct", None)
        if value is not None:
            fraction = float(value)
            if 0.01 <= fraction <= 1.0:
                return fraction
    except Exception:
        pass

    try:
        osz = getattr(engine, "order_sizes", {}) or {}
        value = osz.get("default_pct", None)
        if value is not None:
            fraction = float(value)
            if 0.01 <= fraction <= 1.0:
                return fraction
    except Exception:
        pass

    try:
        env_value = os.environ.get("EQUITY_PCT")
        if env_value is not None:
            fraction = float(env_value)
            if 0.01 <= fraction <= 1.0:
                return fraction
    except Exception:
        pass

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


def _calc_equity_stats(engine) -> Tuple[float, float, float]:
    equity = 0.0
    d1 = 0.0
    w1 = 0.0
    trader = getattr(engine, "trader", None)
    if trader and hasattr(trader, "equity"):
        try:
            equity = float(trader.equity())
        except Exception:
            equity = 0.0
    cfg = _engine_config(engine)
    equity_csv, _ = _cfg_csv_paths(cfg)
    try:
        df = pd.read_csv(equity_csv, parse_dates=["ts"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        now = pd.Timestamp.utcnow()
        d1 = float(df[df["ts"] >= (now - pd.Timedelta(days=1))]["pnl"].sum())
        w1 = float(df[df["ts"] >= (now - pd.Timedelta(days=7))]["pnl"].sum())
    except Exception:
        pass
    return equity, d1, w1


def _build_estado_text(engine) -> str:
    if engine is None:
        return "Engine no disponible."
    equity, d1, w1 = _calc_equity_stats(engine)
    trader = getattr(engine, "trader", None)
    per_symbol: Dict[str, int] = {}
    if trader and getattr(trader, "state", None):
        try:
            per_symbol = {sym: len(lots) for sym, lots in trader.state.positions.items()}
        except Exception:
            per_symbol = {}
    open_cnt = sum(per_symbol.values()) if per_symbol else 0
    ks = False
    if trader and getattr(trader.state, "killswitch", None) is not None:
        try:
            ks = bool(trader.state.killswitch)
        except Exception:
            ks = False
    mode_txt = "🧪 SIMULADO" if S.PAPER else "🔴 REAL"
    lines = [
        "📊 Estado del Bot",
        f"Modo: {mode_txt}",
        f"Saldo actual: ${_fmt_num(equity, 2)}",
        f"PnL 24h: ${_fmt_num(d1, 2)} | PnL 7d: ${_fmt_num(w1, 2)}",
        f"Operaciones abiertas: {open_cnt}",
    ]
    if S.PAPER:
        try:
            eq_sim = float(getattr(trading.BROKER, "equity"))
            lines.append(f"Equity sim: ${_fmt_num(eq_sim, 2)}")
        except Exception:
            pass
    if per_symbol:
        lines.append("Por símbolo: " + ", ".join(f"{sym}: {cnt}" for sym, cnt in per_symbol.items()))
    lines.append("Bot: OFF (killswitch ACTIVADO)" if ks else "Bot: ON (killswitch desactivado)")
    return "\n".join(lines)


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
        pnl = float(st.get("pnl") or 0.0)
        sym = st.get("symbol") or (engine.config or {}).get("symbol", "?")
        reply_text = (
            "📍 *Posición del BOT*\n"
            f"• Símbolo: *{sym}*\n"
            f"• Lado: *{side}*\n"
            f"• Cantidad (bot qty): *{_num(q, 4)}*\n"
            f"• Entrada: {_num(entry)}  |  Mark: {_num(mark)}\n"
            f"• PnL: *{_num(pnl)}*"
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
        positions = await exchange.fetch_positions(None)
    except Exception as exc:  # pragma: no cover - robustez
        await message.reply_text(f"No pude leer posiciones: {exc}")
        return

    if not positions:
        await message.reply_text("No hay posiciones abiertas.")
        return

    bot_status = None
    bot_qty = 0.0
    try:
        bot_status = trading.POSITION_SERVICE.get_status() if trading.POSITION_SERVICE else None
        bot_qty = float(bot_status.get("qty") or 0.0) if bot_status else 0.0
    except Exception:
        bot_status = None
        bot_qty = 0.0

    lines = ["📋 *Posiciones totales (usuario + BOT):*"]
    for pos in positions:
        symbol = pos.get("symbol") or pos.get("info", {}).get("symbol") or "?"
        size = (
            pos.get("contracts")
            or pos.get("size")
            or pos.get("contractsSize")
            or pos.get("amount")
            or 0.0
        )
        try:
            size_f = float(size or 0.0)
        except Exception:
            size_f = 0.0
        side = pos.get("side") or ("LONG" if size_f > 0 else ("SHORT" if size_f < 0 else "FLAT"))
        entry = (
            pos.get("entryPrice")
            or pos.get("entry_price")
            or pos.get("avgPrice")
            or pos.get("average")
            or 0.0
        )
        upnl = pos.get("unrealizedPnl") or pos.get("unrealized_pnl") or 0.0
        formatted = (
            f"{symbol} | {side} | qty={_num(size, 4)} | "
            f"entry=${_num(entry)} | uPnL=${_num(upnl)}"
        )
        # Resaltar SOLO si el BOT tiene porción abierta en este símbolo
        is_bot_symbol = str(symbol).upper() == str(symbol_bot).upper()
        if is_bot_symbol and bot_qty and abs(bot_qty) > 0:
            lines.append(f"*[BOT]* *{formatted}*  (bot qty={_num(bot_qty, 4)})")
        else:
            lines.append(formatted)

    await message.reply_text("\n".join(lines), parse_mode="Markdown")


async def estado_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return

    engine = _get_engine_from_context(context)
    if engine is None:
        await message.reply_text("No pude acceder al engine para consultar el estado.")
        return

    storage = getattr(engine, "storage", None)
    db_path = None
    if storage is not None:
        db_path = getattr(storage, "db_path", None)
    if not db_path:
        db_path = getattr(engine, "db_path", None)
    if not db_path:
        db_path = _engine_sqlite_path(engine)

    try:
        now = datetime.now()
        now_local_str = fmt_ar(datetime.now(timezone.utc))
        start_of_day = now - timedelta(hours=24)
        start_of_week = now - timedelta(days=7)

        pnl_day = 0.0
        pnl_week = 0.0
        if db_path and os.path.exists(db_path):
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT SUM(pnl) FROM trades WHERE close_timestamp >= ?",
                    (start_of_day.isoformat(),),
                )
                row = cursor.fetchone()
                pnl_day = float(row[0]) if row and row[0] is not None else 0.0

                cursor.execute(
                    "SELECT SUM(pnl) FROM trades WHERE close_timestamp >= ?",
                    (start_of_week.isoformat(),),
                )
                row = cursor.fetchone()
                pnl_week = float(row[0]) if row and row[0] is not None else 0.0

        trader = getattr(engine, "trader", None)
        from config import S as _S_

        balance_actual: Optional[float] = None
        if _S_.PAPER:
            try:
                if trader and hasattr(trader, "equity"):
                    balance_actual = float(trader.equity())
            except Exception:
                balance_actual = None

        if balance_actual is None:
            try:
                ex = getattr(engine, "exchange", None)
                if ex and hasattr(ex, "client") and not _S_.PAPER:
                    bal = await asyncio.to_thread(ex.client.fetch_balance)
                    usdt = bal.get("USDT") or {}
                    balance_actual = float(usdt.get("total") or usdt.get("free") or 0.0)
            except Exception:
                balance_actual = None

        if balance_actual is None:
            try:
                cfg = _engine_config(engine)
                equity_csv, _ = _cfg_csv_paths(cfg)
                df = pd.read_csv(equity_csv)
                if not df.empty:
                    balance_actual = float(df["equity"].iloc[-1])
            except Exception:
                balance_actual = None

        if balance_actual is None:
            balance_actual = float(getattr(_S_, "start_equity", 0.0))

        estado_lineas = [
            _position_status_message(engine),
            "",
            "Estado de Cuenta Rápido",
            "---------------------------",
            f"Hora local: {now_local_str}",
            f"PNL Hoy (24h): ${pnl_day:+.2f}",
            f"PNL Semana (7d): ${pnl_week:+.2f}",
            f"Balance Actual: ${balance_actual:,.2f}",
        ]
        reply_text = "\n".join(estado_lineas)
    except Exception as exc:
        reply_text = f"Error al generar el estado: {exc}"

    await message.reply_text(reply_text)


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
    Ver o setear el Stop Loss como % del equity al abrir (global para todos los leverages).
      • sl           → muestra el valor actual
      • sl 10        → setea 10% (también acepta decimales: 'sl 0.1' = 10%)
    """

    engine = _get_engine_from_context(context)
    message = update.effective_message
    if engine is None or message is None:
        return

    txt = (message.text or "").strip().lower()

    def _cfg() -> Dict:
        return _engine_config(engine) or {}

    if re.match(r"^/?sl\s*$", txt):
        cfg = _cfg()
        v = cfg.get("sl_eq_pct", cfg.get("stop_eq_pnl_pct", 0.05))
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.05
        if v < 1.0:
            pct = v * 100.0
        else:
            pct = v
            v = v / 100.0
        await message.reply_text(
            f"SL actual: {pct:.2f}% del equity",
            parse_mode="Markdown",
        )
        return

    m = re.match(r"^/?sl\s+([0-9]+(?:\.[0-9]+)?)%?\s*$", txt)
    if not m:
        await message.reply_text("Uso: `sl` | `sl 10` (10%) | `sl 0.1`", parse_mode="Markdown")
        return

    raw = float(m.group(1))
    value = raw / 100.0 if raw >= 1.0 else raw
    cfg = _cfg()
    cfg["sl_eq_pct"] = value

    if hasattr(engine, "config") and isinstance(engine.config, dict):
        engine.config.update(cfg)

    await message.reply_text(
        f"✅ SL fijado en {value * 100:.2f}% del equity",
        parse_mode="Markdown",
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
        default_pct = float(cfg.get("target_eq_pnl_pct", 0.10))
        lines = ["*TP por apalancamiento*"]
        if isinstance(m, dict) and m:
            for k in sorted([str(x) for x in m.keys()], key=lambda s: int(s)):
                v = float(m[k])
                v = v / 100.0 if v >= 1.0 else v
                lines.append(f"• x{int(k)}: {v*100:.2f}% del equity")
        else:
            lines.append("_(sin overrides; usando default)_")
        lines.append(f"• Default: {default_pct*100:.2f}% del equity (`target_eq_pnl_pct`)")
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


async def _modo_command(update: Update, context: ContextTypes.DEFAULT_TYPE, new_mode: str):
    engine = _get_engine_from_context(context)
    message = update.effective_message
    rescue_needed = False
    if new_mode == "real" and engine is not None:
        has_open_fn = getattr(engine, "has_open_position", None)
        if callable(has_open_fn):
            try:
                rescue_needed = bool(has_open_fn())
            except Exception:
                rescue_needed = False

    result = trading.switch_mode("real" if new_mode == "real" else "simulado")
    if result.ok:
        base_msg = f"✅ Modo cambiado a *{new_mode.upper()}*. El bot ya opera en {new_mode}."
        msg = f"{base_msg}\n{result.msg}" if result.msg else base_msg
        if new_mode == "real":
            try:
                if engine and hasattr(engine, "exchange") and engine.exchange:
                    await engine.exchange.upgrade_to_real_if_needed()
            except Exception:
                logger.debug("No se pudo reautenticar exchange tras cambio a REAL.", exc_info=True)
        else:
            try:
                if engine and hasattr(engine, "exchange") and engine.exchange:
                    await engine.exchange.downgrade_to_paper()
            except Exception:
                logger.debug("No se pudo pasar exchange a paper tras cambio a SIM.", exc_info=True)
        # Resetear caches del trader para evitar balances/posiciones viejas
        try:
            if engine and getattr(engine, "trader", None):
                engine.trader.reset_caches()
        except Exception:
            logger.debug("No se pudo resetear caches del trader tras cambio de modo.", exc_info=True)
        if message is not None:
            await message.reply_text(msg, parse_mode="Markdown")
        if new_mode == "real" and engine is not None and message is not None:
            if rescue_needed and hasattr(engine, "sync_live_position"):
                await message.reply_text(
                    "⚠️ Rescate: activé modo REAL y sincronizo la posición del exchange…"
                )
                try:
                    synced = await asyncio.to_thread(engine.sync_live_position)
                except Exception as exc:
                    await message.reply_text(
                        f"⚠️ Activé REAL pero falló la sincronización automática: {exc}"
                    )
                else:
                    if synced:
                        await message.reply_text(
                            "✅ Posición LIVE sincronizada. El bot ya la controla."
                        )
                    else:
                        await message.reply_text(
                            "ℹ️ No hay posición LIVE en el exchange. Estado local limpiado."
                        )
        return
    else:
        msg = f"❌ No pude cambiar el modo: {result.msg}"
    if message is None:
        return
    await message.reply_text(msg, parse_mode="Markdown")


async def modo_simulado_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _modo_command(update, context, "simulado")


async def modo_real_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _modo_command(update, context, "real")


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
    """Fija o muestra el porcentaje de equity (1–100%). Ej: 'equity 50%'."""
    engine = _get_engine_from_context(context)
    message = update.effective_message
    if message is None:
        return
    if engine is None:
        await message.reply_text("No pude acceder al engine para ajustar el equity.")
        return

    txt = (message.text or "").strip().lower()
    match = re.search(r"equity\s+(\d+(?:[.,]\d+)?)\s*%?", txt)
    if not match:
        fraction = float(_get_equity_fraction(engine))
        pct = round(fraction * 100.0, 2)
        await message.reply_text(
            f"Equity actual seteado: {pct:.2f}% (frac={fraction})\n"            
        )
        return

    try:
        pct_str = match.group(1).replace(",", ".")
        pct = float(pct_str)
    except Exception:
        await message.reply_text("No pude leer el porcentaje. Ej: equity 25%")
        return

    if not (1.0 <= pct <= 100.0):
        await message.reply_text("El porcentaje debe estar entre 1 y 100.")
        return

    frac = round(pct / 100.0, 4)
    _find_and_set_config(engine, "order_sizing.default_pct", frac)
    os.environ["EQUITY_PCT"] = str(frac)
    await message.reply_text(f"✅ Porcentaje de equity seteado: {pct:.2f}% (frac={frac})")


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
        help_text="Muestra o fija el % de equity (1–100). Ej: equity 37%",
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
        help_text="Cambia el bot a modo REAL (requiere API keys y sin posición abierta)",
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

