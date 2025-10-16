import os
import sys

# Asegura que la carpeta raíz del proyecto esté primera en sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import logging
from typing import Any, Dict

from bot.engine import TradingApp
from config import S, get_telegram_chat_id, get_telegram_token, load_raw_config
from logging_setup import setup_logging

def main():
    """Función principal que configura e inicia la aplicación."""
    cfg: Dict[str, Any] = load_raw_config()
    cfg.setdefault("trading_mode", S.trading_mode)
    cfg.setdefault("mode", "paper" if S.PAPER else "real")
    cfg.setdefault("start_equity", S.start_equity)

    cfg["telegram_token"] = get_telegram_token(cfg.get("telegram_token"))
    cfg["telegram_chat_id"] = get_telegram_chat_id(cfg.get("telegram_chat_id"))

    if S.LIVE:
        cfg["binance_api_key_real"] = (
            S.binance_api_key
            or os.getenv("BINANCE_API_KEY")
            or os.getenv("BINANCE_API_KEY_REAL")
        )
        cfg["binance_api_secret_real"] = (
            S.binance_api_secret
            or os.getenv("BINANCE_API_SECRET")
            or os.getenv("BINANCE_API_SECRET_REAL")
        )
    else:
        cfg["binance_api_key_test"] = os.getenv("BINANCE_API_KEY_TEST") or S.binance_api_key
        cfg["binance_api_secret_test"] = os.getenv("BINANCE_API_SECRET_TEST") or S.binance_api_secret
    
    setup_logging()

    app = TradingApp(cfg)
    app.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Bot detenido manualmente.")
