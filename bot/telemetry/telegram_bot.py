import os
import logging
import asyncio
import sqlite3
import re
import inspect
from collections import deque
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logger = logging.getLogger("telegram")

# =========================
# Utils de formato
# =========================
def _env(key, default=None):
    v = os.getenv(key, default)
    return v

def _fmt_num(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

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
        return path
    return os.path.join("logs", "bot.log")


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


async def _reply_chunks(update: Update, text: str):
    message = update.effective_message
    if message is None:
        return
    for chunk in _chunk_text(text):
        if chunk:
            await message.reply_text(chunk)


async def ayuda_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra la lista de comandos disponibles para interactuar con el bot."""
    ayuda_texto = (
        "**Lista de Comandos Disponibles**\n\n"
        "**Monitoreo:**\n"
        "`posicion` - Muestra el estado de la operación actual.\n"
        "`estado` - Muestra el PNL del día/semana y balance.\n"
        "`rendimiento` - Muestra estadísticas históricas completas.\n"
        "`precio` - Devuelve el precio actual de BTC/USDT.\n"
        "`motivos` - Muestra los últimos 10 rechazos de señales.\n"
        "`config` - Muestra los parámetros actuales del bot.\n"
        "`logs [N]` - Muestra las últimas N líneas del log.\n\n"
        "**Control:**\n"
        "`pausa` - Detiene la apertura de nuevas operaciones.\n"
        "`reanudar` - Reanuda la apertura de operaciones.\n"
        "`cerrar` - Cierra la posición actual.\n"
        "`killswitch` - Cierra la posición y pausa el bot."
    )
    await update.effective_message.reply_text(ayuda_texto, parse_mode='Markdown')


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
    lines = [
        "📊 Estado del Bot",
        f"Saldo actual: ${_fmt_num(equity, 2)}",
        f"PnL 24h: ${_fmt_num(d1, 2)} | PnL 7d: ${_fmt_num(w1, 2)}",
        f"Operaciones abiertas: {open_cnt}",
    ]
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


def _build_config_text(engine) -> str:
    cfg = _engine_config(engine)
    risk = cfg.get("risk", {}) if isinstance(cfg, dict) else {}
    strategy = cfg.get("strategy", {}) if isinstance(cfg, dict) else {}
    order = cfg.get("order_sizing", {}) if isinstance(cfg, dict) else {}
    execution = cfg.get("execution", {}) if isinstance(cfg, dict) else {}
    leverage = cfg.get("leverage", {}) if isinstance(cfg, dict) else {}
    lines = [
        "⚙️ Configuración actual",
        f"Modo: {cfg.get('mode', 'paper')} | Timeframe: {cfg.get('timeframe', '?')}",
        "--- Riesgo ---",
        f"Size mode: {risk.get('size_mode', 'desconocido')} | Max hold bars: {risk.get('max_hold_bars', 'N/A')}",
        f"Equity USDT: {risk.get('equity_usdt', 'N/A')} | Max riesgo trade: {risk.get('max_risk_per_trade_pct', 'N/A')}",
        "--- Estrategia ---",
        f"Entry mode: {strategy.get('entry_mode', 'N/A')} | RSI gate: {strategy.get('rsi4h_gate', 'N/A')}",
        f"Target EQ PnL %: {strategy.get('target_eq_pnl_pct', 'N/A')} | EMA200 1h confirm: {strategy.get('ema200_1h_confirm', 'N/A')}",
        "--- Tamaños de orden ---",
        f"Pct min/default/max: {order.get('min_pct', 'N/A')} / {order.get('default_pct', 'N/A')} / {order.get('max_pct', 'N/A')}",
        "--- Ejecución ---",
        f"Leverage set: {execution.get('leverage', leverage.get('default', 'N/A'))} | Slippage bps: {execution.get('slippage_bps', 'N/A')}",
    ]
    return "\n".join(lines)


def _read_logs_text(engine, limit: int = 15) -> str:
    path = _engine_logs_path(engine)
    try:
        if not os.path.exists(path):
            return f"No encontré el archivo de logs ({path})."
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]
        if not lines:
            return "El archivo de logs está vacío."
        tail = lines[-limit:]
        return "📄 Últimos logs:\n" + "\n".join(tail)
    except Exception as exc:
        return f"No pude leer los logs ({path}): {exc}"


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


async def posicion_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if message is None:
        return

    engine = _get_engine_from_context(context)
    if engine is None:
        await message.reply_text("No pude acceder al engine para consultar la posición.")
        return

    trader = getattr(engine, "trader", None)
    if trader is None or not hasattr(trader, "check_open_position"):
        await message.reply_text("No hay un trader disponible para consultar posiciones.")
        return

    try:
        result = trader.check_open_position(getattr(engine, "exchange", None))
        position = await result if inspect.isawaitable(result) else result
    except Exception as exc:
        await message.reply_text(f"Error al consultar la posición: {exc}")
        return

    if position:
        pnl = float(position.get("unrealizedPnl", 0) or 0)
        entry_price = float(position.get("entryPrice", 0) or 0)
        side = (position.get("side") or "N/A").upper()
        symbol = position.get("symbol", "N/A")

        reply_text = (
            "**Estado Actual: Posición Abierta**\n"
            "---------------------------------\n"
            f"Símbolo: {symbol}\n"
            f"Lado: {side}\n"
            f"Precio de Entrada: ${entry_price:,.2f}\n"
            f"PNL Actual: ${pnl:+.2f}"
        )
    else:
        reply_text = (
            "**Estado Actual: Esperando Señal**\n"
            "---------------------------------\n"
            "No hay ninguna posición abierta en este momento."
        )

    await message.reply_text(reply_text, parse_mode="Markdown")


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
        balance = 0.0
        if trader and hasattr(trader, "get_balance"):
            try:
                balance_result = trader.get_balance(getattr(engine, "exchange", None))
                balance = (
                    await balance_result
                    if inspect.isawaitable(balance_result)
                    else float(balance_result or 0.0)
                )
            except Exception:
                balance = 0.0

        reply_text = (
            "**Estado de Cuenta Rápido**\n"
            "---------------------------\n"
            f"PNL Hoy (24h): ${pnl_day:+.2f}\n"
            f"PNL Semana (7d): ${pnl_week:+.2f}\n"
            f"Balance Actual: ${balance:,.2f}"
        )
    except Exception as exc:
        reply_text = f"Error al generar el estado: {exc}"

    await message.reply_text(reply_text, parse_mode="Markdown")


async def rendimiento_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Calcula y envía las estadísticas de rendimiento desde la base de datos."""
    engine = context.application.user_data['engine']
    db_path = engine.db_path # Obtenemos la ruta de la DB desde el motor

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Hacemos una única consulta para obtener todos los datos
            cursor.execute("SELECT COUNT(*), SUM(pnl), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) FROM trades")
            total_trades, total_pnl, wins = cursor.fetchone()

        if total_trades == 0:
            reply_text = "Aún no hay operaciones completadas en el historial."
        else:
            total_pnl = total_pnl or 0
            wins = wins or 0
            losses = total_trades - wins
            winrate = (wins / total_trades) * 100 if total_trades > 0 else 0

            reply_text = (
                f"**Rendimiento Histórico (Base de Datos)**\n"
                f"----------------------------------\n"
                f"📈 **Trades Totales:** {total_trades}\n"
                f"✅ **Ganadas:** {wins}\n"
                f"❌ **Perdidas:** {losses}\n"
                f"🎯 **Winrate:** {winrate:.2f}%\n"
                f"💰 **PNL Neto Total:** {total_pnl:+.2f} USD"
            )

    except Exception as e:
        reply_text = f"Error al leer la base de datos de rendimiento: {e}"

    await update.message.reply_text(reply_text, parse_mode='MarkdownV2')


async def cerrar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "Engine no disponible para cerrar posiciones.")
        return
    try:
        await engine.close_all()
        await _reply_chunks(update, "✅ Cerré todas las posiciones abiertas.")
    except Exception as exc:
        await _reply_chunks(update, f"No pude cerrar las posiciones: {exc}")


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
    text = _build_config_text(engine)
    await _reply_chunks(update, text)


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


async def motivos_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Envía los últimos motivos registrados por los filtros de entrada."""
    engine = _get_engine_from_context(context)
    if engine is None:
        await _reply_chunks(update, "No pude acceder al engine para consultar motivos.")
        return

    log = list(getattr(engine, "rejection_log", []))
    if not log:
        await _reply_chunks(update, "No se ha registrado ningún rechazo de operación todavía.")
        return

    motivos_list = "\n".join(f"• {item}" for item in reversed(log))
    header = "Últimos 10 motivos para no entrar al mercado:\n"
    await _reply_chunks(update, header + "\n" + motivos_list)


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

    try:
        application = Application.builder().token(token).build()
    except Exception as exc:
        logger.warning("Telegram application init failed: %s", exc)
        return None

    application.bot_data["engine"] = engine_instance
    try:
        application.user_data["engine"] = engine_instance
    except Exception:
        pass

    text_filter = filters.TEXT & (~filters.COMMAND)

    application.add_handler(CommandHandler("ayuda", ayuda_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^ayuda$"), ayuda_command))

    application.add_handler(CommandHandler("posicion", posicion_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^posicion$"), posicion_command))

    application.add_handler(CommandHandler("estado", estado_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^estado$"), estado_command))

    application.add_handler(CommandHandler("rendimiento", rendimiento_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^rendimiento$"), rendimiento_command))

    application.add_handler(CommandHandler("cerrar", cerrar_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^cerrar$"), cerrar_command))

    application.add_handler(CommandHandler("precio", precio_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^precio(?:\s+\S+)?$"), precio_command))

    application.add_handler(CommandHandler("config", config_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^config$"), config_command))

    application.add_handler(CommandHandler("boton", bot_on_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^bot\s+on$"), bot_on_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^reanudar$"), bot_on_command))

    application.add_handler(CommandHandler("botoff", bot_off_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^bot\s+off$"), bot_off_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^pausa$"), bot_off_command))

    application.add_handler(CommandHandler("ajustar", ajustar_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^ajustar\b.*"), ajustar_command))

    application.add_handler(CommandHandler("logs", logs_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^logs(?:\s+\d+)?$"), logs_command))

    application.add_handler(CommandHandler("motivos", motivos_command))
    application.add_handler(MessageHandler(text_filter & filters.Regex(r"(?i)^motivos$"), motivos_command))

    logging.info("Todos los comandos de Telegram han sido registrados correctamente.")
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

    # ============ (opcional) comandos inline ============
    if inline_commands and not getattr(application, "_chaulet_inline_handler_added", False):
        async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
            try:
                text = (update.message.text or "").strip()
                low = text.lower()
                chat_id = update.effective_chat.id

                current_notifier = getattr(app, "notifier", notifier)
                if current_notifier and current_notifier.default_chat_id is None:
                    current_notifier.set_default_chat(chat_id)

                if re.match(r"(?i)^precio", low):
                    await precio_command(update, context)
                elif low == "posicion":
                    await posicion_command(update, context)
                elif low == "estado":
                    await estado_command(update, context)
                elif low == "rendimiento":
                    await rendimiento_command(update, context)
                elif low == "config":
                    await config_command(update, context)
                elif low in ("bot on", "prender bot", "activar bot", "reanudar", "bot prender"):
                    await bot_on_command(update, context)
                elif low in ("bot off", "apagar bot", "desactivar bot", "pausa", "bot apagar"):
                    await bot_off_command(update, context)
                elif low == "cerrar":
                    await cerrar_command(update, context)
                elif re.match(r"(?i)^ajustar\b", text):
                    await ajustar_command(update, context)
                elif re.match(r"(?i)^logs", text):
                    await logs_command(update, context)
                else:
                    await context.bot.send_message(
                        chat_id,
                        "Comandos: posicion | estado | rendimiento | precio [SIMBOLO] | config | bot on/off | cerrar | logs [n] | ajustar parametro valor",
                    )
            except Exception as e:
                logger.exception("telegram msg error: %s", e)

        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_message))
        setattr(application, "_chaulet_inline_handler_added", True)
    elif not inline_commands:
        logger.info("Inline commands deshabilitados (usando telegram_commands.py)")

    # ============ (opcional) reportes diarios/semanales en ESTE bot ============
    if reports_in_bot and not getattr(application, "_chaulet_reports_scheduled", False):
        try:
            j = application.job_queue
            tz = _tz()
            j.run_daily(lambda c: notifier.app.create_task(_report_periodic(notifier, days=1)),
                        time=dtime(hour=23, minute=59, tzinfo=tz), name="daily_report")
            j.run_daily(lambda c: notifier.app.create_task(_report_periodic(notifier, days=7)),
                        time=dtime(hour=23, minute=59, tzinfo=tz), days=(6,), name="weekly_report")
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

