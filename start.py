import os
import yaml
import asyncio
import logging
from dotenv import load_dotenv


def main():
    """Punto de entrada principal de la aplicación."""
    # 1. Cargar configuración y secretos
    load_dotenv()
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['telegram_token'] = os.getenv("TELEGRAM_TOKEN")
    cfg['telegram_chat_id'] = os.getenv("TELEGRAM_CHAT_ID")
    cfg['binance_api_key_real'] = os.getenv("BINANCE_API_KEY_REAL")
    cfg['binance_api_secret_real'] = os.getenv("BINANCE_API_SECRET_REAL")
    cfg['binance_api_key_test'] = os.getenv("BINANCE_API_KEY_TEST")
    cfg['binance_api_secret_test'] = os.getenv("BINANCE_API_SECRET_TEST")
    
    # 2. Importar el motor DESPUÉS de configurar el logging
    from bot.engine import TradingApp

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 3. Iniciar el bot
    app = TradingApp(cfg)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
