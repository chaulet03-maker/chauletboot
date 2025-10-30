# start.py (reemplazar completo)
import argparse
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
from bot.mode_manager import ensure_startup_mode
from paths import get_data_dir


def main(argv: list[str] | None = None):
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

    parser = argparse.ArgumentParser(description="Inicia el bot de trading")
    parser.add_argument(
        "--mode",
        dest="mode",
        help="Forzar modo de ejecución (real/live/paper/simulado)",
    )
    args = parser.parse_args(argv)

    cfg = load_raw_config()

    env_mode = os.getenv("TRADING_MODE") or os.getenv("MODE")
    effective_mode, mode_source, persisted = ensure_startup_mode(
        cfg,
        cli_mode=args.mode,
        env_mode=env_mode,
        persist=True,
    )

    # Normalizamos telegram acá (una sola fuente de verdad)
    cfg["telegram_token"] = get_telegram_token(cfg.get("telegram_token"))
    cfg["telegram_chat_id"] = get_telegram_chat_id(cfg.get("telegram_chat_id"))

    setup_logging()
    logging.info(
        "Iniciando bot en modo: %s (fuente=%s, persistido=%s)",
        "REAL" if effective_mode == "real" else "SIMULADO",
        mode_source,
        persisted,
    )
    logging.info("DATA_DIR=%s", get_data_dir())

    app = TradingApp(cfg, mode_source=mode_source)
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Bot detenido manualmente.")
