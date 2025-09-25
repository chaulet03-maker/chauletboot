import os, logging, pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

logger = logging.getLogger("telegram")

def _env(key, default=None):
    v = os.getenv(key, default)
    return v

async def start_telegram_bot(app, config):
    if not config.get("telegram",{}).get("enabled", True):
        logger.info("Telegram disabled")
        return
    token = _env("TELEGRAM_TOKEN")
    if not token:
        logger.warning("No TELEGRAM_TOKEN provided; Telegram disabled")
        return

    application = ApplicationBuilder().token(token).build()

    async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            text = (update.message.text or "").strip()
            low = text.lower()
            chat_id = update.effective_chat.id
            csv_dir = config.get("storage",{}).get("csv_dir","data")
            if low == "precio":
                parts = []
                for sym in app.symbols:
                    p = app.price_of(sym) or await app.fetch_last_price(sym)
                    parts.append(f"{sym}: {p:.2f}")
                await context.bot.send_message(chat_id, "📈 " + " | ".join(parts))
            elif low == "estado":
                total = app.trader.equity()
                d1 = 0.0; w1 = 0.0
                try:
                    df = pd.read_csv(f"{csv_dir}/equity.csv", parse_dates=["ts"])
                    df['ts'] = pd.to_datetime(df['ts'], utc=True)
                    now = pd.Timestamp.utcnow()
                    d1 = float(df[df['ts'] >= (now - pd.Timedelta(days=1))]['pnl'].sum())
                    w1 = float(df[df['ts'] >= (now - pd.Timedelta(days=7))]['pnl'].sum())
                except Exception:
                    pass
                ks = "ON" if app.trader.state.killswitch else "OFF"
                pos_count = sum(len(v) for v in app.trader.state.positions.values())
                await context.bot.send_message(chat_id, f"⚙️ estado: equity=${total:.2f} | pnl(1d)=${d1:.2f} | pnl(7d)=${w1:.2f} | posiciones={pos_count} | killswitch={ks}")
            elif low == "posicion":
                lines = []
                for sym, lots in app.trader.state.positions.items():
                    for i, L in enumerate(lots, 1):
                        lines.append(f"{sym} #{i} {L['side']} qty={L['qty']:.6f} entry={L['entry']:.2f} lev={L['lev']}")
                await context.bot.send_message(chat_id, "📊 posiciones:\n" + ("\n".join(lines) if lines else "Sin posiciones"))
            elif low == "saldo":
                total = app.trader.equity()
                await context.bot.send_message(chat_id, f"💰 saldo: ${total:.2f}")
            elif low.startswith("saldo="):
                total = app.trader.equity()
                try:
                    df = pd.read_csv(f"{csv_dir}/equity.csv", parse_dates=["ts"])
                    df['ts'] = pd.to_datetime(df['ts'], utc=True)
                    now = pd.Timestamp.utcnow()
                    d1 = float(df[df['ts'] >= (now - pd.Timedelta(days=1))]['pnl'].sum())
                    w1 = float(df[df['ts'] >= (now - pd.Timedelta(days=7))]['pnl'].sum())
                    await context.bot.send_message(chat_id, f"💰 saldo: ${total:.2f}\n📅 último día: ${d1:.2f}\n🗓️ última semana: ${w1:.2f}")
                except Exception:
                    await context.bot.send_message(chat_id, f"💰 saldo: ${total:.2f}\n(no hay datos de equity.csv suficientes)")
            elif low.startswith("operaciones="):
                span = "hoy"
                if "semana" in low: span = "semana"
                if "mes" in low: span = "mes"
                try:
                    df = pd.read_csv(f"{csv_dir}/trades.csv", parse_dates=["ts"])
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
    await application.initialize()
    await application.start()
    logger.info("Telegram bot started")
    await application.updater.start_polling()
