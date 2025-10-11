import ccxt
import logging


class Exchange:
    def __init__(self, cfg):
        self.config = cfg
        self.client = self._setup_client()

    def _setup_client(self):
        """ Configura e inicializa el cliente del exchange (ccxt). """
        mode = self.config.get('trading_mode', 'simulado')
        logging.info(f"Configurando el exchange en modo: {mode.upper()}")

        api_key_env = 'binance_api_key_real' if mode == 'real' else 'binance_api_key_test'
        api_secret_env = 'binance_api_secret_real' if mode == 'real' else 'binance_api_secret_test'

        client = ccxt.binance({
            'apiKey': self.config.get(api_key_env),
            'secret': self.config.get(api_secret_env),
            'options': {
                'defaultType': 'future',
            },
        })

        if mode != 'real':
            client.set_sandbox_mode(True)

        logging.info("Cliente de CCXT para Binance inicializado correctamente.")
        return client

    def get_klines(self, timeframe='1h', symbol=None, limit=300):
        """ Obtiene las velas (klines) de un par. """
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        # ccxt necesita que el símbolo para futuros no tenga la barra '/'
        ccxt_symbol = symbol.replace('/', '')

        ohlcv = self.client.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, limit=limit)
        # Aquí puedes añadir la lógica para convertir ohlcv a un DataFrame de pandas si es necesario
        # Por ahora, devolvemos los datos crudos
        return ohlcv

    def get_current_price(self, symbol=None):
        """ Obtiene el último precio de un par. """
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        ticker = self.client.fetch_ticker(symbol)
        return ticker.get('last')

    def set_leverage(self, leverage, symbol=None):
        """ Establece el apalancamiento para un par. """
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        ccxt_symbol = symbol.replace('/', '')  # Asegurarse de que el symbol es correcto para ccxt
        self.client.set_leverage(leverage, ccxt_symbol)
        logging.info(f"Apalancamiento establecido en x{leverage} para {symbol}")

    def create_order(self, side, quantity, sl_price, tp_price, symbol=None):
        """ Crea una nueva orden de mercado con SL y TP. """
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        # Lógica para crear la orden aquí...
        # Esto puede variar mucho, pero un ejemplo simple sería:
        logging.info(f"Creando orden {side} para {quantity} de {symbol}...")
        # order = self.client.create_market_order(symbol, side, quantity)
        # self.client.create_stop_loss_order(symbol, 'stop_market', side_opuesta, quantity, sl_price)
        # self.client.create_take_profit_order(symbol, 'take_profit_market', side_opuesta, quantity, tp_price)

        # Por ahora, solo simularemos la creación con un log
        logging.info("¡Orden (simulada) creada con éxito!")
        return {"status": "ok"}
