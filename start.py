import os
import yaml
import asyncio
import logging
from dotenv import load_dotenv

# Importar la clase principal del bot
from bot.engine import TradingApp


def main():
    """
    Función principal que configura e inicia la aplicación.
    """
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

    # 2. Configurar el logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler()
        ]
    )

    # 3. Iniciar la aplicación del bot
    app = TradingApp(cfg)

    # 4. El método run() ahora es el punto de entrada que maneja el bucle
    asyncio.run(app.run())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Bot detenido manualmente.")
