import os
import yaml
import logging
from dotenv import load_dotenv
from bot.engine import TradingApp
from logging_setup import setup_logging

def main():
    """Función principal que configura e inicia la aplicación."""
    load_dotenv()
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['telegram_token'] = os.getenv("TELEGRAM_TOKEN")
    cfg['telegram_chat_id'] = os.getenv("TELEGRAM_CHAT_ID")
    cfg['binance_api_key_real'] = os.getenv("BINANCE_API_KEY_REAL")
    cfg['binance_api_secret_real'] = os.getenv("BINANCE_API_SECRET_REAL")
    cfg['binance_api_key_test'] = os.getenv("BINANCE_API_KEY_TEST")
    cfg['binance_api_secret_test'] = os.getenv("BINANCE_API_SECRET_TEST")
    
    setup_logging()

    app = TradingApp(cfg)
    app.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Bot detenido manualmente.")
