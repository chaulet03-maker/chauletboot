import os
import logging
import asyncio
from collections import deque
from datetime import time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger("telegram")

# =========================
# Utils y helpers de formato
# =========================

def _env(key, default=None):
    return os.getenv(key, default)

def _fmt_num(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _pct_rel(entry: float, level: float) -> float:
    """Porcentaje de variación de 'level' respecto a 'entry' (signo natural del cambio de precio)."""
    if not entry:
        return 0.0
    try:
        return (float(level) / float(entry) - 1.0) * 100.0
    except Exception:
        return 0.0

def _fmt_price_with_pct(level: float, entry: float) -> str:
    """Devuelve '62,890.5 (+0.86%)' sin miles, con 2 decimales."""
    lvl = _fmt_num(level, 2)
    pct = _pct_rel(entry, level)
    sign = "+" if pct >= 0 else ""
    return f"{lvl} ({sign}{pct:.2f}%)"

def _cfg_csv_paths(cfg: dict):
    """Resuelve rutas desde config.persistence si existen; si no, usa defaults."""
    p = cfg.get("persistence", {}) if isinstance(cfg, dict) else {}
    base = p.get("dir", "data")
    equity = p.get("equity_csv", os.path.join(base, "equity.csv"))
    trades = p.get("trades_csv", os.path.join(base, "trades.csv"))
    return equity, trades

def _tz():
    return ZoneInfo("America/Argentina/Buenos_Aires")

# =========================
# Notificador PRO
# =========================

class TelegramNotifier:
    """
    Expuesto como app.telegram para que el engine pueda llamar:
      app.telegram.open(symbol=..., side=..., entry=..., sl=..., tp1=..., tp2=..., qty=..., lev=..., regime=..., conf=..., reason=...)
      app.telegram.tp1(...)
      app.telegram.close_tp(...)
      app.telegram.close_sl(...)
      app.telegram.close_manual(...)
      app.telegram.reject(symbol=..., side=..., code="REJECT_...", detail="...")
    """

    def __init__(self, application, cfg, default_chat_id=None):
        self.app = application
        self.cfg = cfg or {}
        self.default_chat_id = default_chat_id
        # anti-spam / dedup
        tcfg = (self.cfg.get("telegram") or {})
        self.min_interval_s = float(((tcfg.get("anti_spam") or {}).get("min_interval_s", 1.0)))
        self.dedup_window_s = int(tcfg.get("dedup_window_s", 60))
        self._last_sent_ts = 0.0
        self._dedup = deque(maxlen=200)
        # ring buffer de motivos
        self._rejections = deque(maxlen=50)
        # paths
        self.equity_csv, self.trades_csv = _cfg_csv_paths(self.cfg)

    def set_default_chat(self, chat_id):
        self.default_chat_id = chat_id

    # ------------- API pública (sin await) -------------
    def open(self, **kwargs):        self.app.create_task(self._send_open(**kwargs))
    def tp1(self, **kwargs):         self.app.create_task(self._send_tp1(**kwargs))
    def trailing(self, **kwargs):    self.app.create_task(self._send_trailing(**kwargs))
    def close_tp(self, **kwargs):    self.app.create_task(self._send_close(kind="TP", **kwargs))
    def close_sl(self, **kwargs):    self.app.create_task(self._send_close(kind="SL", **kwargs))
    def close_manual(self, **kwargs):self.app.create_task(self._send_close(kind="MANUAL", **kwargs))
    def reject(self, **kwargs):      self.app.create_task(self._send_reject(**kwargs))

    # para que el engine loguee motivos a mostrar con /motivos (si usás comandos aparte)
    def log_reject(self, symbol: str, side: str, code: str, detail: str = ""):
        self._rejections.appendleft({
            "symbol": symbol, "side": side, "code": code, "detail": detail
        })

    # ------------- Internals -------------

    async def _safe_send(self, text: str):
        if not self.default_chat_id:
            # No hay chat preconfigurado; esperar a que el usuario escriba algo para capturar el chat_id
            logger.debug("No default chat_id; mensaje descartado: %s", text[:60])
            return
        # anti-spam básico: no más de 1 msg cada min_interval_s
        now = asyncio.get_running_loop().time()
        if now - self._last_sent_ts < self.min_interval_s:
            return
        # dedup simple por contenido en ventana
        key = (text.strip()[:200], int(now // self.dedup_window_s))
        if key in self._dedup:
            return
        self._dedup.append(key)
        self._last_sent_ts = now
        try:
            await self.app.bot.send_message(chat_id=self.default_chat_id, text=text)
        except Exception as e:
            logger.exception("telegram send error: %s", e)

    # ----- Mensajes de evento -----

    async def _send_open(self,
                         symbol: str, side: str,
                         entry: float,
                         sl: float, tp1: float, tp2: float,
                         qty: float, lev: int,
                         regime: str = "", conf: float = 0.0,
                         reason: str = ""):
        """
        Mensaje de apertura con % contra entry en SL/TPs.
        """
        side = (side or "").upper()
        head = "🟢 OPEN LONG" if side == "LONG" else "🔴 OPEN SHORT" if side == "SHORT" else "🟡 OPEN"
        msg = (
            f"{head}  | {symbol}\n"
            f"Precio: {_fmt_num(entry)}\n"
            f"SL: {_fmt_price_with_pct(sl, entry)}   "
            f"TP1: {_fmt_price_with_pct(tp1, entry)}   "
            f"TP2: {_fmt_price_with_pct(tp2, entry)}\n"
            f"Qty: {_fmt_num(qty, 6)}     Lev: x{lev}\n"
        )
        if regime or conf:
            msg += f"Régimen: {regime}   Conf: {_fmt_num(conf, 2)}\n"
        if reason:
            msg += f"Motivo: {reason}\n"

        await self._safe_send(msg)

    async def _send_tp1(self, symbol: str, side: str, price: float, entry: float,
                        qty_closed: float, pnl_partial: float, new_sl: float, balance: float):
        msg = (
            f"✅ TP1 HIT     | {symbol} ({side.upper()})\n"
            f"Precio: {_fmt_num(price)} {_fmt_price_with_pct(price, entry).split(' ',1)[1]}\n"
            f"Qty cerrada: {_fmt_num(qty_closed, 6)}   SL movido a BE: {_fmt_num(entry)}\n"
            f"PnL parcial: ${_fmt_num(pnl_partial)}   Balance: ${_fmt_num(balance)}"
        )
        await self._safe_send(msg)

    async def _send_trailing(self, symbol: str, side: str, new_sl: float, entry: float):
        # solo tiene sentido mostrar % respecto al entry
        msg = (
            f"🧷 TRAILING    | {symbol} ({side.upper()})\n"
            f"Nuevo SL: {_fmt_price_with_pct(new_sl, entry)}"
        )
        await self._safe_send(msg)

    async def _send_close(self, symbol: str, side: str, price: float, entry: float,
                          pnl: float, balance: float, kind: str = "TP"):
        kind = kind.upper()
        icon = "🎯" if kind == "TP" else "🛑" if kind == "SL" else "✋"
        msg = (
            f"{icon} CLOSE {kind} | {symbol} ({side.upper()})\n"
            f"Precio: {_fmt_num(price)} {_fmt_price_with_pct(price, entry).split(' ',1)[1]}\n"
            f"PnL trade: ${_fmt_num(pnl)}    Balance: ${_fmt_num(balance)}"
        )
        await self._safe_send(msg)

    async def _send_reject(self, symbol: str, side: str, code: str, detail: str = ""):
        msg = (
            f"🚫 RECHAZO OPEN | {symbol} ({side.upper()})\n"
            f"Motivo: {code}{(' — ' + detail) if detail else ''}"
        )
        # guardar para /motivos
        self.log_reject(symbol, side, code, detail)
        await self._safe_send(msg)

    # ----- Reportes programados -----

    async def report_daily(self):
        text = await self._build_report(days=1, title="📊 REPORTE DIARIO")
        await self._safe_send(text)

    async def report_weekly(self):
        text = await self._build_report(days=7, title="📈 REPORTE SEMANAL")
        await self._safe_send(text)

    async def _build_report(self, days: int, title: str) -> str:
        eq_path, tr_path = self.equity_csv, self.trades_csv
        equity_ini = equity_fin = pnl = fees = win_rate = 0.0
        wins = losses = total_trades = 0
        top_syms = {}

        # equity
        try:
            df_eq = pd.read_csv(eq_path, parse_dates=["ts"])
            df_eq["ts"] = pd.to_datetime(df_eq["ts"], utc=True)
            now = pd.Timestamp.now(tz=_tz())
            since = now - pd.Timedelta(days=days)
            dfw = df_eq[df_eq["ts"] >= since.tz_convert("UTC")]
            if not dfw.empty:
                equity_ini = float(dfw["equity"].iloc[0])
                equity_fin = float(dfw["equity"].iloc[-1])
                pnl = float(dfw["pnl"].sum())
        except Exception as e:
            logger.debug("equity.csv not ready: %s", e)

        # trades
        try:
            df_tr = pd.read_csv(tr_path, parse_dates=["ts"])
            df_tr["ts"] = pd.to_datetime(df_tr["ts"], utc=True)
            now = pd.Timestamp.now(tz=_tz())
            since = now - pd.Timedelta(days=days)
            dft = df_tr[df_tr["ts"] >= since.tz_convert("UTC")]
            if not dft.empty:
                total_trades = len(dft)
                wins = int((dft["pnl"] > 0).sum())
                losses = int((dft["pnl"] < 0).sum())
                win_rate = (wins / max(total_trades, 1)) * 100.0
                try:
                    fees = float(dft.get("fees", 0.0).sum())
                except Exception:
                    fees = 0.0
                # top por símbolo
                grp = dft.groupby("symbol")["pnl"].sum().sort_values(ascending=False).head(3)
                top_syms = {k: float(v) for k, v in grp.items()}
        except Exception as e:
            logger.debug("trades.csv not ready: %s", e)

        lines = [
            f"{title} — {pd.Timestamp.now(tz=_tz()).strftime('%Y-%m-%d')}",
            f"Equity inicial: ${_fmt_num(equity_ini)}   Equity final: ${_fmt_num(equity_fin)}   Δ: ${_fmt_num(equity_fin - equity_ini)}",
            f"PnL neto: ${_fmt_num(pnl)}   Fees: ${_fmt_num(fees)}",
            f"Trades cerrados: {total_trades}   Win rate: {_fmt_num(win_rate, 2)}%",
        ]
        if top_syms:
            tops = ", ".join([f"{sym} ${_fmt_num(val)}" for sym, val in top_syms.items()])
            lines.append(f"Top símbolos: {tops}")
        return "\n".join(lines)

# =========================
# Arranque y handler simple
# =========================

async def start_telegram_bot(app, config):
    # gated por config
    if not (config.get("telegram", {}).get("enabled", True)):
        logger.info("Telegram disabled")
        return

    token = _env("TELEGRAM_TOKEN")
    if not token:
        logger.warning("No TELEGRAM_TOKEN provided; Telegram disabled")
        return

    application = ApplicationBuilder().token(token).build()

    # Notificador con API para el engine
    chat_id_env = _env("TELEGRAM_CHAT_ID")  # opcional
    notifier = TelegramNotifier(application, config, default_chat_id=int(chat_id_env) if chat_id_env else None)
    # Exponerlo al resto del bot
    setattr(app, "telegram", notifier)

    # ==== Comandos tipo "texto plano" (los que ya tenías) ====

    async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            text = (update.message.text or "").strip()
            low = text.lower()
            chat_id = update.effective_chat.id

            # si no hay chat configurado aún, fijar este
            if notifier.default_chat_id is None:
                notifier.set_default_chat(chat_id)

            # rutas CSV (persistence)
            eq_csv, tr_csv = _cfg_csv_paths(config)

            if low == "precio":
                parts = []
                for sym in app.symbols:
                    p = app.price_of(sym) or await app.fetch_last_price(sym)
                    parts.append(f"{sym}: {_fmt_num(p)}")
                await context.bot.send_message(chat_id, "📈 " + " | ".join(parts))

            elif low == "estado":
                total = app.trader.equity()
                d1 = 0.0; w1 = 0.0
                try:
                    df = pd.read_csv(eq_csv, parse_dates=["ts"])
                    df['ts'] = pd.to_datetime(df['ts'], utc=True)
                    now = pd.Timestamp.utcnow()
                    d1 = float(df[df['ts'] >= (now - pd.Timedelta(days=1))]['pnl'].sum())
                    w1 = float(df[df['ts'] >= (now - pd.Timedelta(days=7))]['pnl'].sum())
                except Exception:
                    pass
                ks = "ON" if app.trader.state.killswitch else "OFF"
                pos_count = sum(len(v) for v in app.trader.state.positions.values())
                await context.bot.send_message(
                    chat_id,
                    f"⚙️ estado: equity=${_fmt_num(total)} | pnl(1d)=${_fmt_num(d1)} | pnl(7d)=${_fmt_num(w1)} | posiciones={pos_count} | killswitch={ks}"
                )

            elif low == "posicion":
                lines = []
                for sym, lots in app.trader.state.positions.items():
                    for i, L in enumerate(lots, 1):
                        lines.append(f"{sym} #{i} {L['side']} qty={_fmt_num(L['qty'],6)} entry={_fmt_num(L['entry'])} lev={L['lev']}")
                await context.bot.send_message(chat_id, "📊 posiciones:\n" + ("\n".join(lines) if lines else "Sin posiciones"))

            elif low == "saldo":
                total = app.trader.equity()
                await context.bot.send_message(chat_id, f"💰 saldo: ${_fmt_num(total)}")

            elif low.startswith("saldo="):
                total = app.trader.equity()
                try:
                    df = pd.read_csv(eq_csv, parse_dates=["ts"])
                    df['ts'] = pd.to_datetime(df['ts'], utc=True)
                    now = pd.Timestamp.utcnow()
                    d1 = float(df[df['ts'] >= (now - pd.Timedelta(days=1))]['pnl'].sum())
                    w1 = float(df[df['ts'] >= (now - pd.Timedelta(days=7))]['pnl'].sum())
                    await context.bot.send_message(chat_id, f"💰 saldo: ${_fmt_num(total)}\n📅 último día: ${_fmt_num(d1)}\n🗓️ última semana: ${_fmt_num(w1)}")
                except Exception:
                    await context.bot.send_message(chat_id, f"💰 saldo: ${_fmt_num(total)}\n(no hay datos de equity)")

            elif low.startswith("operaciones="):
                span = "hoy"
                if "semana" in low: span = "semana"
                if "mes" in low: span = "mes"
                try:
                    df = pd.read_csv(tr_csv, parse_dates=["ts"])
                    df['ts'] = pd.to_datetime(df['ts'], utc=True)
                    now = pd.Timestamp.utcnow()
                    delta = {"hoy":"1D","semana":"7D","mes":"30D"}[span]
                    cnt = int((df['ts'] >= (now - pd.to_timedelta(delta))).sum())
                    await context.bot.send_message(chat_id, f"🧾 operaciones {span}: {cnt}")
                except Exception:
                    await context.bot.send_message(chat_id, "No hay trades cargados todavía.")

            elif low == "killswitch":
                ks = app.toggle_killswitch()
                await context.bot.send_message(chat_id, f"🛑 killswitch {'ON' if ks else 'OFF'}")

            elif low == "cerrar":
                await app.close_all()
                app.trader.state.killswitch = True
                await context.bot.send_message(chat_id, "🔒 cerrado todo y killswitch ON")

            else:
                await context.bot.send_message(chat_id, "Comandos: precio | estado | posicion | saldo | saldo= | operaciones= (hoy/semana/mes) | killswitch | cerrar")

        except Exception as e:
            logger.exception("telegram msg error: %s", e)

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_message))

    # ===== Programación de reportes (si querés diario/semanal) =====
    # Diario y semanal a las 23:59 Buenos Aires
    try:
        j = application.job_queue
        if j is not None:
            tz = _tz()
            j.run_daily(lambda c: notifier.app.create_task(notifier.report_daily()),
                        time=dtime(hour=23, minute=59, tzinfo=tz), name="daily_report")
            # semanal: domingo 23:59
            j.run_daily(lambda c: notifier.app.create_task(notifier.report_weekly()),
                        time=dtime(hour=23, minute=59, tzinfo=tz), days=(6,), name="weekly_report")
    except Exception as e:
        logger.warning("No job_queue; reportes deshabilitados: %s", e)

    # Arranque del bot
    await application.initialize()
    await application.start()
    logger.info("Telegram bot started")
    await application.updater.start_polling()
