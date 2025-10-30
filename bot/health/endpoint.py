from typing import Optional
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes
import math
import logging

log = logging.getLogger(__name__)


# Este módulo NO crea ni inicia un bot nuevo.
# Se engancha al Application que ya arma tu engine (self.telegram_app).
#
# Uso:
#   from endpoint import attach_to_application
#   attach_to_application(trading_app)   # después de crear TradingApp

def _fmt_usd(x: Optional[float]) -> str:
    try:
        f = float(x)
        if math.isnan(f):
            return "n/d"
        return f"${f:,.2f}"
    except Exception:
        return "n/d"


async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "¡Hola! Soy tu Bot de Trading.\n\n"
        "Comandos:\n"
        "• /start o /help — Ayuda\n"
        "• /status — Estado actual (modo, equity, posición, precio)\n"
    )
    if update.effective_message:
        await update.effective_message.reply_text(txt)


async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await _cmd_start(update, context)


async def _cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Estado real usando el TradingApp que puso el engine en application.bot_data['app'].
    """
    app = context.application.bot_data.get("app")
    if app is None:
        if update.effective_message:
            await update.effective_message.reply_text("No hay app de trading enlazada todavía.")
        return

    try:
        symbol = app.config.get("symbol", "BTC/USDT")
        mode_txt = "REAL" if app.is_live else "SIMULADO"
        # Equity actual
        equity = await app.trader.get_balance(app.exchange)
        # Precio actual
        price = await app.exchange.get_current_price(symbol)
        # Posición (si hay)
        open_txt = "FLAT"
        entry = mark = None
        try:
            import trading
            st = trading.POSITION_SERVICE.get_status() if trading.POSITION_SERVICE else None
        except Exception:
            st = None
        if st:
            side = str(st.get("side", "FLAT")).upper()
            q = float(st.get("qty") or st.get("size") or 0.0)
            if side != "FLAT" and abs(q) > 0.0:
                open_txt = f"{side} {q:.6f}"
                entry = float(st.get("entry_price") or 0.0)
                mark = float(st.get("mark") or 0.0)

        # Funding (si engine lo guardó)
        fr_bps = app.config.get("_funding_rate_bps_now")
        fr_line = f"\nFunding ahora: {fr_bps:.2f} bps" if fr_bps is not None else ""

        msg = (
            f"📊 *Status del Bot*\n"
            f"Modo: {mode_txt}\n"
            f"Símbolo: {symbol}\n"
            f"Precio: {_fmt_usd(price)}\n"
            f"Equity: {_fmt_usd(equity)}\n"
            f"Posición: {open_txt}"
        )
        if entry is not None:
            msg += f"\nEntry: {_fmt_usd(entry)} | Mark: {_fmt_usd(mark)}"
        msg += fr_line

        if update.effective_message:
            await update.effective_message.reply_markdown(msg)
    except Exception as e:
        log.exception("status handler error: %s", e)
        if update.effective_message:
            await update.effective_message.reply_text(f"Error al obtener estado: {e}")


def attach_to_application(trading_app) -> None:
    """
    Registra /start, /help y /status en el Application ya creado por el engine:
      trading_app.telegram_app
    """
    app = getattr(trading_app, "telegram_app", None)
    if app is None:
        raise RuntimeError("No hay telegram_app en TradingApp; verificá setup_telegram_bot.")
    # Guardamos referencia al engine para usarla en handlers
    app.bot_data["app"] = trading_app
    app.add_handler(CommandHandler(["start", "help"], _cmd_start))
    app.add_handler(CommandHandler("status", _cmd_status))
    log.info("Handlers de Telegram (/start, /help, /status) registrados.")


# Nota: No hay bloque __main__. El engine ya hace run_polling().
