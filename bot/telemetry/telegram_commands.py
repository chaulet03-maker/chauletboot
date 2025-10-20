import asyncio, logging, os, re, datetime as dt, csv, math, inspect
from typing import Dict, Tuple, List, Optional
from telegram.ext import Application, MessageHandler, filters
import unicodedata

import trading
from config import S
from bot.motives import MOTIVES
from bot.pnl import pnl_summary_bot
from bot.runtime_state import set_mode as runtime_set_mode
from .telegram_bot import (
    _build_config_text,
    _build_rendimiento_text,
    _find_and_set_config,
    _parse_adjust_value,
    _read_logs_text,
    _set_killswitch,
)

log = logging.getLogger("tg")


def _normalize_alias(txt: str) -> str:
    if not txt:
        return ""
    t = unicodedata.normalize("NFD", txt)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


ALIASES = {
    "precio": [
        "precio",
        "price",
        "precio actual",
        "btc",
        "cotizacion",
        "cotización",
    ],
    "estado": ["estado", "status", "pnl", "balance"],
    "posicion": [
        "posicion",
        "posición",
        "position",
        "pos",
        "posicion actual",
        "posición actual",
    ],
    "posiciones": ["posiciones", "positions", "open positions"],
    "motivos": [
        "motivos",
        "razones",
        "por que no entro",
        "porque no entro",
        "por qué no entro",
    ],
    "config": ["config", "configuracion", "configuración"],
    "pausa": ["pausa", "pausar"],
    "reanudar": ["reanudar", "resume", "continuar"],
    "cerrar": ["cerrar", "close", "cerrar todo"],
    "killswitch": ["killswitch", "panic"],
    "logs": ["logs", "log", "ver logs"],
    "modo_real": [
        "modo real",
        "poner modo real",
        "real",
        "activar real",
        "usar real",
    ],
    "modo_simulado": [
        "modo simulado",
        "poner modo simulado",
        "simulado",
        "activar simulado",
        "paper",
        "demo",
        "test",
    ],
}

ALIASES.update(
    {
        "open": ["open", "abrir"],
        "posicion": ["posicion", "posición", "position", "pos", "posicion actual", "posición actual"],
        "posiciones": ["posiciones", "positions", "open positions"],
    }
)
ALIASES["debug"] = ["debug"]


def _to_binance_symbol(sym: str) -> str:
    return sym.replace("/", "").upper()


def _is_engine_live(engine) -> bool:
    mode_attr = str(getattr(engine, "mode", "") or "").lower()
    if mode_attr in ("real", "live"):
        return True
    try:
        if bool(getattr(engine, "is_live")):
            return True
    except Exception:
        pass
    return str(trading.ACTIVE_MODE).lower() == "real"


def resolve_command(txt: str) -> Optional[str]:
    normalized = _normalize_alias(txt)
    if not normalized:
        return None
    for command, variants in ALIASES.items():
        for variant in variants:
            if normalized == _normalize_alias(variant):
                return command
    return None

# ========= Helpers de texto / formato =========

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[*_`~]+", "", s)                              # quitar markdown simple
    s = re.sub(r"[,.;:!?()\[\]{}<>\\|/]+", " ", s)             # quitar signos
    s = re.sub(r"\s+", " ", s).strip()                         # colapsar espacios
    s = unicodedata.normalize("NFD", s)                        # quitar acentos
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def _fmt_money(x):
    try:
        return f"${float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def _fmt_num(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _csv_dir(engine) -> str:
    return getattr(engine, "csv_dir", "data")

def _parse_iso(s: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

# ========= Bloques de ayuda / posiciones =========

async def _cmd_help(reply):
    texto = (
        "📖 *Ayuda*\n"
        "• *ayuda*: muestra esta ayuda.\n"
        "• *estado*: saldo, abiertas y últimos saldos.\n"
        "• *rendimiento*: estadísticas acumuladas desde la base SQLite.\n"
        "• *saldo* / *saldo=NUM*: consulta / fija saldo inicial (paper, si el engine lo soporta).\n"
        "• *posicion* / *posiciones*: detalle de posiciones abiertas.\n"
        "• *precio [SIMBOLO]*: precio actual o todos.\n"
        "• *config*: parámetros clave de la estrategia actual.\n"
        "• *logs [N]*: últimas N líneas del log.\n"
        "• *bot on* / *bot off*: habilita/deshabilita nuevas entradas.\n"
        "• *cerrar* / *cerrar todo* / *sos*: cierra posiciones (sos = apaga y cierra).\n"
        "• *ajustar parametro valor*: modifica la configuración en caliente.\n"
        "• *recientes* / *motivos*: últimos 10 motivos de NO-entrada.\n"
        "• *stats* / *stats semana*: PF, winrate, expectancy por símbolo/capa.\n"
        "• *report hoy* / *report semana*: reporte diario/semanal.\n"
        "• *diag on* / *diag off*: activa/desactiva diagnóstico.\n"
    )
    return await reply(texto)

async def _cmd_positions_detail(engine, reply):
    st = getattr(engine.trader, "state", None)
    positions = getattr(st, "positions", {}) if st else {}
    if not positions:
        return await reply("No hay posiciones abiertas.")
    price_cache = getattr(engine, "price_cache", {}) or {}
    lines = []
    for sym, lots in positions.items():
        px_now = price_cache.get(sym)
        try:
            px_now = float(px_now) if px_now is not None else None
        except Exception:
            px_now = None
        for L in lots:
            side = L.get("side","long")
            s_side = "long" if side == "long" else "short"
            lev = int(L.get("lev", 1) or 1)
            entry = float(L.get("entry", 0.0) or 0.0)
            qty = float(L.get("qty", 0.0) or 0.0)
            sl = float(L.get("sl", 0.0) or 0.0)
            tp = float(L.get("tp2", L.get("tp1", 0.0)) or 0.0)
            pnl_abs = 0.0
            pnl_pct = 0.0
            if px_now and entry:
                if side == "long":
                    pnl_abs = (px_now - entry) * qty * max(1, lev)
                    pnl_pct = (px_now / entry - 1.0) * 100.0 * max(1, lev)
                else:
                    pnl_abs = (entry - px_now) * qty * max(1, lev)
                    pnl_pct = (entry / px_now - 1.0) * 100.0 * max(1, lev)
            lines.append(
                f"{sym} {s_side} x{lev}\n"
                f"entrada: {entry:.2f}\n"
                f"pnl: {pnl_abs:+.2f} ({pnl_pct:+.2f}%)\n"
                f"sl: {sl:.2f}\n"
                f"tp: {tp:.2f}"
            )
            lines.append("")
    return await reply("\n".join(lines).strip())


async def _cmd_precio(engine, reply, symbol: Optional[str] = None):
    sym_default = "BTC/USDT"
    cfg = getattr(engine, "config", None)
    if isinstance(cfg, dict):
        sym_default = cfg.get("symbol", sym_default)
    sym = symbol or sym_default
    try:
        px = await engine.exchange.get_current_price(sym)
    except Exception as exc:
        log.debug("precio error para %s: %s", sym, exc)
        px = None
    if px is None:
        return await reply("No pude obtener el precio.")
    try:
        if S.PAPER and trading.POSITION_SERVICE:
            trading.POSITION_SERVICE.mark_to_market(float(px))
    except Exception:
        log.debug("No se pudo actualizar mark en PAPER desde /precio", exc_info=True)
    return await reply(f"{sym}: ${float(px):,.2f}")


async def _cmd_posicion(engine, reply):
    from bot.identity import get_bot_id
    from bot.ledger import bot_position

    sym_conf = engine.config.get("symbol", "BTCUSDT")
    symbol = _to_binance_symbol(sym_conf)
    mode = "live" if _is_engine_live(engine) else "paper"
    bid = get_bot_id()

    qty, avg = bot_position(mode, bid, symbol)
    if abs(qty) <= 0.0:
        qty, avg = bot_position(mode, bid, sym_conf)
    if abs(qty) <= 0.0:
        return await reply(f"Sin posición del bot en {symbol}.")

    try:
        mark = await engine.exchange.get_current_price(symbol)
    except Exception:
        mark = None

    mark_val = float(mark if mark is not None else avg)
    side = "LONG" if qty > 0 else "SHORT"
    pnl_u = (mark_val - avg) * abs(qty) if qty > 0 else (avg - mark_val) * abs(qty)

    return await reply(
        f"BOT {side} {symbol}\n"
        f"qty: {abs(qty):.6f} | entrada: {avg:.2f} | mark: {mark_val:.2f}\n"
        f"PnL no realizado: {pnl_u:+.2f}"
    )


async def _cmd_posiciones(engine, reply):
    from bot.identity import get_bot_id
    from bot.ledger import bot_position

    mode = "live" if _is_engine_live(engine) else "paper"
    bid = get_bot_id()
    sym_conf = engine.config.get("symbol", "BTCUSDT")
    try:
        ex_positions = await engine.exchange.fetch_positions()
    except Exception:
        ex_positions = []

    if not ex_positions:
        return await reply("No hay posiciones abiertas en el exchange.")

    lines = []
    for entry in ex_positions:
        try:
            symbol_raw = entry.get("symbol") or sym_conf
            symbol_clean = _to_binance_symbol(symbol_raw)
            amt = float(
                entry.get("positionAmt")
                or entry.get("contracts")
                or entry.get("amount")
                or 0.0
            )
            entry_price = float(entry.get("entryPrice") or entry.get("avgPrice") or 0.0)
            mark_price = float(entry.get("markPrice") or entry.get("mark") or entry_price)
            side = "LONG" if amt > 0 else ("SHORT" if amt < 0 else "FLAT")
        except Exception:
            continue

        qty_bot, avg_bot = bot_position(mode, bid, symbol_clean)
        if abs(qty_bot) <= 0.0:
            qty_bot, avg_bot = bot_position(mode, bid, symbol_raw)

        if abs(qty_bot) > 0.0:
            lines.append(
                f"**{symbol_clean} {side}**  | **BOT qty:** {abs(qty_bot):.6f}\n"
                f"**entrada BOT:** {avg_bot:.2f} | mark: {mark_price:.2f}"
            )
        else:
            lines.append(
                f"{symbol_clean} {side}\n"
                f"qty exch: {abs(amt):.6f} | entrada: {entry_price:.2f} | mark: {mark_price:.2f}"
            )
        lines.append("")

    return await reply("\n".join(lines).strip())


async def _cmd_motivos(engine, reply, n: int = 10):
    items = MOTIVES.last(n)
    if not items:
        return await reply("No hay rechazos recientes.")
    tz = getattr(S, "output_timezone", "America/Argentina/Buenos_Aires") if hasattr(S, "output_timezone") else "America/Argentina/Buenos_Aires"
    lines = ["🕒 Motivos recientes (últimas 10 oportunidades NO abiertas):"]
    for it in items:
        lines.append(it.human_line(tz=tz))
    log.debug("TELEGRAM /motivos → %d items | 1ra: %s", len(items), lines[1] if len(lines)>1 else "-")
    return await reply("\n".join(lines))

# ========= Stats / Reportes =========

def _read_csv(path: str) -> Tuple[List[Dict], List[str]]:
    rows, hdr = [], []
    if not os.path.exists(path):
        return rows, hdr
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        hdr = reader.fieldnames or []
        for r in reader:
            rows.append(r)
    return rows, hdr

def _filter_rows_last_days(rows: List[Dict], days: int, ts_key="ts") -> List[Dict]:
    if not rows:
        return []
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    since = now - dt.timedelta(days=days)
    out = []
    for r in rows:
        ts = r.get(ts_key) or r.get("timestamp") or r.get("time")
        dtp = _parse_iso(ts) if isinstance(ts, str) else None
        if dtp and dtp.tzinfo is None:
            dtp = dtp.replace(tzinfo=dt.timezone.utc)
        if dtp and dtp >= since:
            out.append(r)
    return out

def _safe_float(r: Dict, key: str, default: float=0.0) -> float:
    try:
        return float(r.get(key, default) or default)
    except Exception:
        return default

def _compute_stats(days: int, csv_dir: str):
    """
    Devuelve:
      closes, by_sym, by_layer, pf_total, winrate(0..1), expectancy, dur_avg_min, dd_max_frac
    No exige columnas estrictas; usa lo que haya.
    """
    trades_path = os.path.join(csv_dir, "trades.csv")
    eq_path     = os.path.join(csv_dir, "equity.csv")

    # --- Trades
    trs, _ = _read_csv(trades_path)
    trs = _filter_rows_last_days(trs, days, ts_key="ts")
    n = len(trs)

    wins = sum(1 for r in trs if _safe_float(r, "pnl") > 0)
    losses = sum(1 for r in trs if _safe_float(r, "pnl") < 0)
    pnl_total = sum(_safe_float(r, "pnl") for r in trs)
    fees_total = sum(_safe_float(r, "fees") for r in trs)
    winrate = (wins / n) if n else 0.0
    expectancy = (pnl_total / n) if n else 0.0

    # por símbolo
    by_sym: Dict[str, Dict] = {}
    for r in trs:
        sym = (r.get("symbol") or "").upper()
        if not sym:
            continue
        d = by_sym.setdefault(sym, {"n":0,"g":0.0,"l":0.0})
        d["n"] += 1
        pnl = _safe_float(r, "pnl")
        if pnl >= 0:
            d["g"] += pnl
        else:
            d["l"] += pnl

    # por “layer” si existiera columna layer/capa/strategy
    by_layer: Dict[str, Dict] = {}
    layer_key = None
    for cand in ("layer","capa","strategy","estrategia"):
        if trs and cand in trs[0]:
            layer_key = cand
            break
    if layer_key:
        for r in trs:
            lay = (r.get(layer_key) or "").upper() or "GEN"
            d = by_layer.setdefault(lay, {"n":0,"g":0.0,"l":0.0})
            d["n"] += 1
            pnl = _safe_float(r, "pnl")
            if pnl >= 0:
                d["g"] += pnl
            else:
                d["l"] += pnl

    # duraciones si hay duration_min
    durations = [_safe_float(r, "duration_min", math.nan) for r in trs]
    durations = [d for d in durations if not math.isnan(d)]
    dur_avg_min = (sum(durations)/len(durations)) if durations else None

    # Profit factor total
    gains = sum(_safe_float(r, "pnl") for r in trs if _safe_float(r, "pnl") > 0)
    losses_abs = abs(sum(_safe_float(r, "pnl") for r in trs if _safe_float(r, "pnl") < 0))
    pf_total = (gains / losses_abs) if losses_abs > 0 else (gains if gains>0 else 0.0)

    # --- DD máximo del período (si hay equity.csv es mejor)
    dd_max_frac = 0.0
    eq_rows, _ = _read_csv(eq_path)
    eq_rows = _filter_rows_last_days(eq_rows, days, ts_key="ts")
    if eq_rows and "equity" in eq_rows[0]:
        highs = -1e30
        for r in eq_rows:
            eq = _safe_float(r, "equity")
            if eq > highs:
                highs = eq
            if highs > 0:
                dd = (eq / highs) - 1.0
                dd_max_frac = min(dd_max_frac, dd)
    else:
        # fallback: curva con cumul de pnl (aprox)
        cum = 0.0
        highs = 0.0
        for r in trs:
            cum += _safe_float(r, "pnl")
            highs = max(highs, cum)
            if highs > 0:
                dd = (cum / highs) - 1.0
                dd_max_frac = min(dd_max_frac, dd)

    # “closes” es la cantidad de trades cerrados
    closes = n
    return closes, by_sym, by_layer, pf_total, winrate, expectancy, dur_avg_min, dd_max_frac, pnl_total, fees_total

def _build_report(days: int, csv_dir: str) -> str:
    closes, by_sym, by_layer, pf_total, wr, exp, dur_avg_min, dd, pnl_total, fees_total = _compute_stats(days, csv_dir)

    # equity cambio si hay equity.csv
    eq_path = os.path.join(csv_dir, "equity.csv")
    eq_rows, _ = _read_csv(eq_path)
    eq_rows = _filter_rows_last_days(eq_rows, days, ts_key="ts")
    eq_ini = eq_rows[0]["equity"] if eq_rows else None
    eq_fin = eq_rows[-1]["equity"] if eq_rows else None

    head = "📊 REPORTE DIARIO" if days == 1 else "📈 REPORTE SEMANAL"
    lines = [head]
    if eq_ini is not None and eq_fin is not None:
        try:
            delta = float(eq_fin) - float(eq_ini)
        except Exception:
            delta = 0.0
        lines.append(f"Equity inicial: {_fmt_money(eq_ini)}   Equity final: {_fmt_money(eq_fin)}   Δ: {_fmt_money(delta)}")
    lines.append(f"PnL neto: {_fmt_money(pnl_total)}   Fees: {_fmt_money(fees_total)}")
    lines.append(f"Trades cerrados: {closes}   Win rate: {wr*100:.2f}%   Expectancy: {_fmt_money(exp)}")
    if dur_avg_min is not None:
        lines.append(f"Tiempo promedio en trade: {dur_avg_min:.1f} min")
    lines.append(f"DD máx período: {dd*100:.2f}%")

    if by_sym:
        lines.append("Top símbolos:")
        # ordenar por pnl total
        ranking = []
        for sym, d in by_sym.items():
            pnl_sym = d["g"] + d["l"]
            ranking.append((pnl_sym, sym))
        ranking.sort(reverse=True)
        for pnl_sym, sym in ranking[:3]:
            lines.append(f"• {sym}: {_fmt_money(pnl_sym)}")

    if by_layer:
        lines.append("Por capa/estrategia:")
        for lay, d in by_layer.items():
            pnl_lay = d["g"] + d["l"]
            lines.append(f"• {lay}: {_fmt_money(pnl_lay)} (n={d['n']})")

    return "\n".join(lines)

# ========= Texto de estado =========

async def _cmd_open(engine, reply, raw_txt: str):
    t = raw_txt.strip().lower()
    m = re.search(r"\bopen\s+(long|short)\s+x\s*(\d+)\b", t)
    if not m:
        return await reply("Formato: open long x5  |  open short x10")

    side_txt = m.group(1).upper()
    lev = int(m.group(2))
    sym_conf = engine.config.get("symbol", "BTCUSDT")
    symbol = _to_binance_symbol(sym_conf)

    exchange = getattr(engine, "exchange", None)
    if exchange is None:
        return await reply("Exchange no disponible para abrir la operación.")

    try:
        px = await exchange.get_current_price(sym_conf)
    except Exception as e:
        return await reply(f"No pude obtener precio actual del símbolo: {e}")

    if px is None:
        return await reply("No pude obtener precio actual del símbolo.")

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
            if inspect.iscoroutinefunction(round_qty_fn):
                qty = await round_qty_fn(symbol, qty)
            else:
                qty = round_qty_fn(symbol, qty)
    except Exception:
        pass

    if qty is None or qty <= 0:
        try:
            from bot.qty_utils import round_and_validate_qty

            qty = await round_and_validate_qty(exchange, sym_conf, qty or 0.0)
        except Exception:
            pass

    if qty is None or qty <= 0:
        return await reply("Cantidad final <= 0 (minQty/minNotional). Subí el equity o bajá el leverage.")

    order_side = "SELL" if side_txt == "SHORT" else "BUY"
    broker = getattr(engine, "broker", None)
    if broker is None or not hasattr(broker, "place_market_order"):
        return await reply("Broker no disponible para abrir la operación.")

    try:
        place_fn = broker.place_market_order
        if inspect.iscoroutinefunction(place_fn):
            result = await place_fn(symbol=sym_conf, side=order_side, quantity=qty, leverage=lev)
        else:
            result = await asyncio.to_thread(
                place_fn,
                symbol=sym_conf,
                side=order_side,
                quantity=qty,
                leverage=lev,
            )
    except Exception as e:
        return await reply(f"Fallo al abrir: {e}")

    fill_price = float(px)
    if isinstance(result, dict):
        for key in ("avgPrice", "avg_price", "price"):
            val = result.get(key)
            if val is not None:
                try:
                    fill_price = float(val)
                    break
                except Exception:
                    continue

    strategy = getattr(engine, "strategy", None)
    if strategy is not None and hasattr(strategy, "attach_tp_sl"):
        try:
            attach_fn = strategy.attach_tp_sl
            kwargs = {
                "symbol": sym_conf,
                "side_txt": side_txt,
                "entry_price": fill_price,
            }
            if inspect.iscoroutinefunction(attach_fn):
                await attach_fn(**kwargs)
            else:
                await asyncio.to_thread(attach_fn, **kwargs)
        except Exception:
            log.debug("attach_tp_sl falló tras open", exc_info=True)

    return await reply(
        f"🟢 OPEN {side_txt} x{lev} | {symbol}\n"
        f"qty: {qty:.6f} @~{fill_price:.2f}\n"
        "Gestionada por la estrategia (TP/SL activos)."
    )


async def _cmd_debug_open(engine, reply, raw_txt: str):
    t = raw_txt.strip().lower()
    m = re.search(r"\bdebug\s+open\s+(long|short)\s+x\s*(\d+)\b", t)
    if not m:
        return await reply("Formato: debug open long x5  |  debug open short x10")

    side_txt = m.group(1).upper()
    lev = int(m.group(2))
    sym_conf = engine.config.get("symbol", "BTCUSDT")
    symbol = _to_binance_symbol(sym_conf)

    eq = float(engine.trader.equity())
    px = await engine.exchange.get_current_price(symbol)

    try:
        info = await engine.exchange.get_symbol_filters(symbol)
    except Exception:
        info = {}

    min_qty = info.get("minQty")
    step = info.get("stepSize")
    min_notional = info.get("minNotional")
    qty_raw = (eq * lev) / px if px else 0
    qty_final = qty_raw
    try:
        from bot.qty_utils import round_and_validate_qty

        qty_final = await round_and_validate_qty(engine.exchange, symbol, qty_raw)
    except Exception:
        pass

    return await reply(
        "DEBUG OPEN\n"
        f"modo: {'REAL' if _is_engine_live(engine) else 'SIM'} | symbol: {symbol}\n"
        f"eq: {eq} | px: {px} | lev: {lev}\n"
        f"minQty: {min_qty} | step: {step} | minNotional: {min_notional}\n"
        f"qty_raw: {qty_raw} | qty_final: {qty_final}"
    )


async def _cmd_estado(engine, reply):
    mode_is_live = _is_engine_live(engine)
    sym_conf = engine.config.get("symbol", "BTCUSDT")
    symbol = _to_binance_symbol(sym_conf)
    exchange = getattr(engine, "exchange", None)

    if mode_is_live:
        try:
            equity = float(await exchange.get_account_equity()) if exchange else 0.0
        except Exception:
            equity = 0.0
    else:
        try:
            equity = float(engine.trader.equity())
        except Exception:
            equity = 0.0

    async def _mark(sym: str):
        if exchange is None:
            return None
        candidates = [sym]
        if "/" in sym:
            candidates.append(sym.replace("/", ""))
        elif sym.endswith("USDT"):
            base = sym[:-4]
            candidates.append(f"{base}/USDT")
        for candidate in candidates:
            try:
                px = await exchange.get_current_price(candidate)
            except Exception:
                px = None
            if px is not None:
                try:
                    return float(px)
                except Exception:
                    continue
        return None

    try:
        pnl = await pnl_summary_bot(
            mode="live" if mode_is_live else "paper",
            mark_provider=_mark,
        )
    except Exception as exc:
        log.debug("pnl_summary_bot falló: %s", exc)
        pnl = {
            "daily": {"realized": 0.0, "unrealized": 0.0, "total": 0.0},
            "weekly": {"realized": 0.0, "unrealized": 0.0, "total": 0.0},
        }

    daily = pnl.get("daily", {"realized": 0.0, "unrealized": 0.0, "total": 0.0})
    weekly = pnl.get("weekly", {"realized": 0.0, "unrealized": 0.0, "total": 0.0})

    txt = (
        f"Modo: *{'REAL' if mode_is_live else 'SIMULADO'}*\n"
        f"Símbolo: {symbol}\n"
        f"Equity base: {equity:,.2f}\n"
        f"Balance estimado: {equity:,.2f}\n"
        f"PnL Diario: {daily['total']:+.2f} (R={daily['realized']:+.2f} | U={daily['unrealized']:+.2f})\n"
        f"PnL Semanal: {weekly['total']:+.2f} (R={weekly['realized']:+.2f} | U={weekly['unrealized']:+.2f})"
    )
    return await reply(txt)


async def _cmd_modo_simulado(engine, reply):
    if not _is_engine_live(engine):
        return await reply("✅ El bot ya se encontraba en *MODO SIMULADO*.")

    set_mode_fn = getattr(engine, "set_mode", None)
    try:
        if callable(set_mode_fn):
            if inspect.iscoroutinefunction(set_mode_fn):
                await set_mode_fn("paper")
            else:
                await asyncio.to_thread(set_mode_fn, "paper")
        else:
            setattr(engine, "mode", "paper")
    except Exception:
        setattr(engine, "mode", "paper")

    setattr(engine, "is_live", False)
    try:
        runtime_set_mode("paper")
    except Exception:
        pass

    exchange = getattr(engine, "exchange", None)
    if exchange is not None and hasattr(exchange, "downgrade_to_paper"):
        try:
            await exchange.downgrade_to_paper()
        except Exception:
            log.debug("downgrade_to_paper falló", exc_info=True)

    return await reply("✅ Modo cambiado a *SIMULADO*. El bot ahora opera en simulado.")


async def _cmd_modo_real(engine, reply):
    if _is_engine_live(engine):
        return await reply("✅ El bot ya se encontraba en *MODO REAL*.")

    set_mode_fn = getattr(engine, "set_mode", None)
    try:
        if callable(set_mode_fn):
            if inspect.iscoroutinefunction(set_mode_fn):
                await set_mode_fn("live")
            else:
                await asyncio.to_thread(set_mode_fn, "live")
        else:
            setattr(engine, "mode", "live")
    except Exception:
        setattr(engine, "mode", "live")

    setattr(engine, "is_live", True)
    try:
        runtime_set_mode("live")
    except Exception:
        pass

    exchange = getattr(engine, "exchange", None)
    if exchange is not None and hasattr(exchange, "upgrade_to_real_if_needed"):
        try:
            await exchange.upgrade_to_real_if_needed()
        except Exception:
            log.debug("upgrade_to_real_if_needed falló", exc_info=True)

    return await reply("✅ Modo cambiado a *REAL*. El bot ahora opera en real.")

# ========= Bot de comandos =========

class CommandBot:
    def __init__(self, app_engine):
        self.engine = app_engine
        # acepta ambos nombres de variable de entorno
        self.token = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")  # opcional

    async def _handle_text(self, update, context):
        msg_raw = (update.message.text or "")
        msg = msg_raw.strip().lower()
        norm_all = _normalize_text(msg_raw)
        norm = re.sub(r"[,.;:!?\s]+", " ", msg).strip()

        def reply(text):
            return update.message.reply_text(text)

        reply_md = lambda text: update.message.reply_text(text, parse_mode="Markdown")

        cmd_alias = resolve_command(msg_raw)
        if cmd_alias:
            if cmd_alias == "modo_real":
                return await _cmd_modo_real(self.engine, reply_md)

            if cmd_alias == "modo_simulado":
                return await _cmd_modo_simulado(self.engine, reply_md)

            if cmd_alias == "precio":
                return await _cmd_precio(self.engine, reply)

            if cmd_alias == "estado":
                return await _cmd_estado(self.engine, reply_md)

            if cmd_alias == "posicion":
                return await _cmd_posicion(self.engine, reply)

            if cmd_alias == "posiciones":
                return await _cmd_posiciones(self.engine, reply)

            if cmd_alias == "motivos":
                return await _cmd_motivos(self.engine, reply)

            if cmd_alias == "config":
                return await reply(_build_config_text(self.engine))

            if cmd_alias == "open":
                return await _cmd_open(self.engine, reply, msg_raw)

            if cmd_alias == "debug":
                return await _cmd_debug_open(self.engine, reply, msg_raw)

            if cmd_alias == "pausa":
                _set_killswitch(self.engine, True)
                return await reply("⛔ Bot OFF: bloqueadas nuevas operaciones (killswitch ACTIVADO).")

            if cmd_alias == "reanudar":
                _set_killswitch(self.engine, False)
                return await reply("✅ Bot ON: habilitadas nuevas operaciones (killswitch desactivado).")

            if cmd_alias == "cerrar":
                ok = await self.engine.close_all()
                if ok:
                    return await reply("Cerré todas las posiciones.")
                return await reply("No pude cerrar todo.")

            if cmd_alias == "killswitch":
                return await reply("Usá 'bot on' / 'bot off' para controlar el killswitch.")

            if cmd_alias == "logs":
                return await reply(_read_logs_text(self.engine, 15))

        # --- BOT ON / OFF (killswitch inverso) ---
        if norm in ("bot on", "prender bot", "activar bot", "bot prender", "reanudar"):
            _set_killswitch(self.engine, False)
            return await reply("✅ Bot ON: habilitadas nuevas operaciones (killswitch desactivado).")

        if norm in ("bot off", "apagar bot", "desactivar bot", "bot apagar", "pausa"):
            _set_killswitch(self.engine, True)
            return await reply("⛔ Bot OFF: bloqueadas nuevas operaciones (killswitch ACTIVADO).")

        # --- POSICION DETALLE ---
        if norm_all in ("posicion", "posición", "position", "pos", "posicion actual", "posición actual"):
            return await _cmd_posicion(self.engine, reply)
        if norm_all in ("posiciones", "positions", "open positions"):
            return await _cmd_posiciones(self.engine, reply)

        # --- AYUDA ---
        if norm_all in ('ayuda','menu','comandos','help'):
            return await _cmd_help(reply)

        # --- ESTADO ---
        if msg in ("estado", "status"):
            return await _cmd_estado(self.engine, reply_md)

        if msg == "rendimiento":
            return await reply(_build_rendimiento_text(self.engine))

        # --- SALDO / SALDO=NUM (best effort) ---
        if msg in ("equity", "saldo"):
            try:
                eq = self.engine.trader.equity()
            except Exception:
                eq = 0.0
            return await reply(f"Saldo: {_fmt_money(eq)}")

        m_eq_cmd = re.match(r"(equity|saldo)\s+([\d\.,]+)", norm_all)
        if m_eq_cmd:
            val_txt = m_eq_cmd.group(2).replace(".", "").replace(",", ".")
            try:
                new_eq = float(val_txt)
            except Exception:
                return await reply("Formato no válido. Ej: equity 1200")

            applied = False
            trader = getattr(self.engine, "trader", None)
            if trader is not None and hasattr(trader, "set_paper_equity"):
                try:
                    trader.set_paper_equity(new_eq)
                    applied = True
                except Exception:
                    applied = False

            if self.engine.is_paper:
                try:
                    from bot.paper_store import set_equity as paper_set_equity

                    paper_set_equity(new_eq)
                except Exception:
                    pass

            if not applied:
                for meth in ("set_equity", "set_equity_init"):
                    fn = getattr(self.engine, meth, None)
                    if callable(fn):
                        try:
                            fn(new_eq)
                            applied = True
                            break
                        except Exception:
                            continue

            if not applied:
                try:
                    os.makedirs(_csv_dir(self.engine), exist_ok=True)
                    with open(os.path.join(_csv_dir(self.engine), "equity_init.txt"), "w", encoding="utf-8") as f:
                        f.write(str(new_eq))
                    applied = True
                except Exception:
                    applied = False

            return await reply("Saldo inicial actualizado." if applied else "Guardé el valor, pero el engine no expone setter.")

        m = re.match(r"saldo\s*=\s*([\d\.,]+)", norm_all)
        if m:
            val_txt = m.group(1).replace(".", "").replace(",", ".")
            try:
                new_eq = float(val_txt)
                # intentar setear si el engine lo permite
                applied = False
                for meth in ("set_paper_equity", "set_equity", "set_equity_init"):
                    fn = getattr(self.engine, meth, None)
                    if callable(fn):
                        try:
                            fn(new_eq)
                            applied = True
                            break
                        except Exception:
                            pass
                if not applied:
                    # persistir para que otro componente lo lea si corresponde
                    try:
                        os.makedirs(_csv_dir(self.engine), exist_ok=True)
                        with open(os.path.join(_csv_dir(self.engine), "equity_init.txt"), "w", encoding="utf-8") as f:
                            f.write(str(new_eq))
                        applied = True
                    except Exception:
                        applied = False
                return await reply("Saldo inicial actualizado." if applied else "Guardé el valor, pero el engine no expone setter.")
            except Exception:
                return await reply("Formato no válido. Ej: saldo=1000")

        # --- POSICIONES (por símbolo) ---
        if msg in ("posicion", "posición"):
            return await _cmd_posicion(self.engine, reply)
        if msg == "posiciones":
            return await _cmd_posiciones(self.engine, reply)

        if norm_all.startswith("open ") or norm_all.startswith("abrir "):
            return await _cmd_open(self.engine, reply, msg_raw)
        if norm_all.startswith("debug open"):
            return await _cmd_debug_open(self.engine, reply, msg_raw)

        # --- PRECIO ---
        if msg.startswith("precio"):
            parts = msg.split()
            if len(parts) >= 2:
                token = parts[1].upper()
                if token in {"TODO", "TODOS", "ALL"}:
                    cache = getattr(self.engine, "price_cache", {}) or {}
                    if not cache:
                        return await reply("Todavía no tengo precios en caché.")
                    listado = "\n".join(f"• {k}: {_fmt_money(v)}" for k, v in cache.items())
                    return await reply("Precios:\n" + listado)
                return await _cmd_precio(self.engine, reply, token)
            return await _cmd_precio(self.engine, reply)

        if msg == "config":
            return await reply(_build_config_text(self.engine))

        # --- DIAGNOSTICO ---
        if norm_all in ("diag on", "diagnostico on", "diagnostico activar"):
            setattr(self.engine, "diag", True)
            return await reply("Modo diagnóstico activado: registrando motivos de no-entrada.")
        if norm_all in ("diag off", "diagnostico off", "diagnostico desactivar"):
            setattr(self.engine, "diag", False)
            return await reply("Modo diagnóstico desactivado.")

        # --- KILL / KILLSWITCH ---
        if msg in ("kill", "killswitch"):
            return await reply("Usá 'bot on' / 'bot off' para controlar el killswitch.")

        if msg.startswith("logs"):
            m = re.search(r"(\d+)", msg)
            limit = int(m.group(1)) if m else 15
            limit = max(1, min(200, limit))
            return await reply(_read_logs_text(self.engine, limit))

        if msg.startswith("ajustar"):
            m = re.match(r"(?i)ajustar\s+([\w.]+)\s+(.+)$", msg_raw.strip())
            if not m:
                return await reply("Uso: ajustar [parametro] [valor]. Ej: ajustar risk.max_hold_bars 20")
            param = m.group(1)
            value = _parse_adjust_value(m.group(2))
            path = _find_and_set_config(self.engine, param, value)
            if path:
                return await reply(f"✅ Actualicé {'/'.join(path)} = {value}")
            return await reply(f"No encontré el parámetro '{param}' en la configuración.")

        # --- STATS (24h / 7d) ---
        if norm_all in ('stats', 'estadisticas', 'estadísticas'):
            csv_dir = _csv_dir(self.engine)
            closes, by_sym, by_layer, pf_total, wr, exp, dur_avg_min, dd, pnl_total, fees_total = _compute_stats(1, csv_dir)
            partes = [
                f"📈 Estadísticas (24h):",
                f"PF total: {pf_total:.2f}",
                f"Winrate: {wr*100:.1f}%",
                f"Expectancy: {_fmt_money(exp)}",
                (f"Tiempo promedio en trade: {dur_avg_min:.1f} min" if dur_avg_min is not None else "Tiempo promedio en trade: N/A"),
                f"DD máx período: {dd*100:.2f}%",
                f"PnL neto: {_fmt_money(pnl_total)}   Fees: {_fmt_money(fees_total)}",
            ]
            if by_layer:
                for lay, d in by_layer.items():
                    g = d['g']; l = d['l']; pf = (g/abs(l)) if l<0 else (g if g>0 else 0.0)
                    partes.append(f'• {lay}: PF {pf:.2f} (n={d["n"]})')
            if by_sym:
                for sym, d in by_sym.items():
                    g = d['g']; l = d['l']; pf = (g/abs(l)) if l<0 else (g if g>0 else 0.0)
                    partes.append(f'• {sym}: PF {pf:.2f} (n={d["n"]})')
            return await reply('\n'.join(partes))

        if norm_all in ('stats semana', 'estadisticas semana', 'estadísticas semana'):
            csv_dir = _csv_dir(self.engine)
            closes, by_sym, by_layer, pf_total, wr, exp, dur_avg_min, dd, pnl_total, fees_total = _compute_stats(7, csv_dir)
            partes = [
                f"📈 Estadísticas (7 días):",
                f"PF total: {pf_total:.2f}",
                f"Winrate: {wr*100:.1f}%",
                f"Expectancy: {_fmt_money(exp)}",
                (f"Tiempo promedio en trade: {dur_avg_min:.1f} min" if dur_avg_min is not None else "Tiempo promedio en trade: N/A"),
                f"DD máx período: {dd*100:.2f}%",
                f"PnL neto: {_fmt_money(pnl_total)}   Fees: {_fmt_money(fees_total)}",
            ]
            if by_layer:
                for lay, d in by_layer.items():
                    g = d['g']; l = d['l']; pf = (g/abs(l)) if l<0 else (g if g>0 else 0.0)
                    partes.append(f'• {lay}: PF {pf:.2f} (n={d["n"]})')
            if by_sym:
                for sym, d in by_sym.items():
                    g = d['g']; l = d['l']; pf = (g/abs(l)) if l<0 else (g if g>0 else 0.0)
                    partes.append(f'• {sym}: PF {pf:.2f} (n={d["n"]})')
            return await reply('\n'.join(partes))

        # --- REPORTES (hoy / semana) ---
        if norm_all in ("report hoy","reporte hoy","report diario","reporte diario"):
            txt = _build_report(1, _csv_dir(self.engine))
            return await reply(txt)

        if norm_all in ("report semana","reporte semana","report semanal","reporte semanal"):
            txt = _build_report(7, _csv_dir(self.engine))
            return await reply(txt)

        # --- CERRAR TODO ---
        if msg in ("cerrar todo", "close all", "close_all", "cerrar", "cerrar de todo"):
            ok = await self.engine.close_all()
            if ok:
                return await reply("Cerré todas las posiciones.")
            return await reply("No pude cerrar todo.")

        # --- SOS: apaga y cierra todo ---
        if norm_all in ("sos","stop and close all","stop close all","stop_close_all"):
            await self.engine.close_all()
            try:
                ks = getattr(self.engine.trader.state, "killswitch", False)
            except Exception:
                ks = False
            if not ks:
                self.engine.toggle_killswitch()
            return await reply("🔒 Cerré todo y killswitch ACTIVADO.")

        # --- RECIENTES / MOTIVOS ---
        if norm_all in ("recientes", "motivos"):
            return await _cmd_motivos(self.engine, reply, 10)

        # Sin match → ayuda breve
        return await reply("No entendí. Escribí *ayuda* para ver comandos.")

    async def run(self):
        if not self.token:
            log.warning("TELEGRAM_TOKEN no configurado; comandos desactivados")
            return
        app = Application.builder().token(self.token).build()
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        await app.initialize()
        try:
            await app.bot.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass
        await app.start()
        try:
            await app.updater.start_polling()
        except Exception:
            pass
        log.info("Telegram command listener started")
        await asyncio.Event().wait()
