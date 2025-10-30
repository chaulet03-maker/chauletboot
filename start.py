# start.py (reemplazar completo)
import os
import sys
import logging
import atexit
from dotenv import load_dotenv

load_dotenv()

# Asegurar import relativo al proyecto
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logging_setup import setup_logging
from config import load_raw_config, get_telegram_token, get_telegram_chat_id
from bot.engine import TradingApp


def _resolve_mode(cfg):
    raw = (os.getenv("MODE")
           or str(cfg.get("trading_mode") or cfg.get("mode") or "paper")).lower()
    return "real" if raw in {"real", "live"} else "paper"


def main():
    # --- SINGLE INSTANCE LOCK ---
    import os, sys
    LOCK_PATH = "/tmp/chauletbot.lock"
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH, "r") as f:
                pid = int(f.read().strip() or "0")
            if pid and pid != os.getpid():
                print(f"[LOCK] Ya hay un bot ejecutándose (PID={pid}). Cerralo antes de iniciar otro.")
                sys.exit(1)
        except Exception:
            pass
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))

    @atexit.register
    def _cleanup_lock():
        try:
            if os.path.exists(LOCK_PATH):
                os.remove(LOCK_PATH)
        except Exception:
            pass

    cfg = load_raw_config()

    # Normalizamos modo y telegram acá (una sola fuente de verdad)
    cfg["mode"] = _resolve_mode(cfg)
    cfg["trading_mode"] = cfg["mode"]
    cfg["telegram_token"] = get_telegram_token(cfg.get("telegram_token"))
    cfg["telegram_chat_id"] = get_telegram_chat_id(cfg.get("telegram_chat_id"))

    setup_logging()
    logging.info("Iniciando bot en modo: %s", cfg["mode"])

    app = TradingApp(cfg)
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Bot detenido manualmente.")
