import logging
import asyncio
from collections import deque
import pandas as pd
import schedule
from telegram.ext import ContextTypes

from bot.exchange import Exchange
from bot.trader import Trader
from bot.storage import Storage
from bot.telemetry.notifier import Notifier
from bot.telemetry.telegram_bot import setup_telegram_bot
from core.strategy import Strategy
from core.indicators import add_indicators


class TradingApp:
    def __init__(self, cfg):
        self.config = cfg
        self._init_db()
        self.trader = Trader(cfg)
        self.exchange = Exchange(cfg)
        self.storage = Storage(cfg)
        self.strategy = Strategy(cfg)
        self.is_paused = False
        self.connection_lost = False
        self.rejection_log = deque(maxlen=10)
        self.telegram_app = setup_telegram_bot(self)
        self.notifier = Notifier(application=self.telegram_app, cfg=self.config)
        self.symbols = [self.config.get('symbol', 'BTC/USDT')]
        self.price_cache = {}

        schedule.every().day.at("08:00", "America/Argentina/Buenos_Aires").do(self._generate_daily_report)
        schedule.every().sunday.at("00:00", "America/Argentina/Buenos_Aires").do(self._generate_weekly_report)
        logging.info("Componentes y tareas de reporte inicializados.")

    async def trading_loop(self, context: ContextTypes.DEFAULT_TYPE):
        """Bucle principal de trading ejecutado por la JobQueue de Telegram."""
        try:
            schedule.run_pending()
            if self.connection_lost:
                await self.notifier.send("✅ **Conexión Reestablecida.**")
                self.connection_lost = False

            logging.info("Iniciando ciclo de análisis de mercado...")

            position = await self.trader.check_open_position(self.exchange)
            if position:
                logging.info(
                    "Posición abierta detectada: %s %s %s",
                    position.get('side'),
                    position.get('contracts'),
                    position.get('symbol'),
                )
                return

            if self.is_paused:
                logging.info("El bot está en pausa, no se buscan nuevas señales.")
                return

            klines_1h = await self.exchange.get_klines('1h')
            klines_4h = await self.exchange.get_klines('4h')

            if not klines_1h or not klines_4h:
                logging.warning("No se pudieron obtener los datos de mercado en este ciclo.")
                return

            df_1h = pd.DataFrame(klines_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms', utc=True)
            df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms', utc=True)
            df_1h.set_index('timestamp', inplace=True)
            df_4h.set_index('timestamp', inplace=True)

            current_funding_bps = await self.exchange.get_current_funding_rate_bps(self.config.get('symbol'))
            if current_funding_bps is not None:
                self.config["_funding_rate_bps_now"] = current_funding_bps
            else:
                self.config.pop("_funding_rate_bps_now", None)

            data = add_indicators(df_1h, df_4h)
            last_candle = data.iloc[-1]

            signal = self.strategy.check_entry_signal(data)
            if not signal:
                logging.info("No se encontraron señales de entrada válidas.")
                return

            # === equity al abrir (idéntico al simulador) ===
            eq_on_open = await self.trader.get_balance(self.exchange)

            # === leverage dinámico x5/x10 por ADX ===
            leverage = self.strategy.dynamic_leverage(last_candle)
            await self.exchange.set_leverage(
                leverage,
                self.config.get('symbol', 'BTC/USDT'),
            )

            entry_price = await self.exchange.get_current_price()
            # === sizing full_equity ===
            entry_price = float(entry_price)
            qty = (eq_on_open * leverage) / max(entry_price, 1e-12)

            # === SL / TP (TP único al 10% del equity al abrir, como el simulador) ===
            sl_price = self.strategy.calculate_sl(entry_price, last_candle, signal)
            tp_price = self.strategy.calculate_tp(entry_price, qty, eq_on_open, signal)

            order_result = await self.exchange.create_order(signal, qty, sl_price, tp_price)

            if order_result:
                await self.notifier.send(
                    f"🚀 **Nueva Operación Abierta: {signal}**\n"
                    f"Símbolo: {self.config['symbol']}\n"
                    f"Apalancamiento: x{leverage}"
                )
                await self.trader.set_position(order_result)

        except Exception as e:
            logging.error(f"Error grave en el trading_loop: {e}", exc_info=True)
            if not self.connection_lost:
                await self.notifier.send(f"💥 **Error inesperado en el bot:** {e}")
                self.connection_lost = True

    def price_of(self, symbol: str):
        return self.price_cache.get(symbol)

    async def fetch_last_price(self, symbol: str):
        return await self.exchange.get_current_price(symbol)

    async def _update_price_cache_job(self, context):
        try:
            symbols = self.symbols or [self.config.get('symbol', 'BTC/USDT')]
            for sym in symbols:
                px = await self.exchange.get_current_price(sym)
                if px is not None:
                    self.price_cache[sym] = float(px)
        except Exception as e:
            logging.warning(f"No pude actualizar la cache de precios: {e}")

    def run(self):
        job_queue = self.telegram_app.job_queue
        job_queue.run_repeating(self._update_price_cache_job, interval=10, first=1)
        job_queue.run_repeating(self.trading_loop, interval=60, first=5)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.notifier.send("✅ **Bot iniciado y corriendo.**"))
        else:
            loop.run_until_complete(self.notifier.send("✅ **Bot iniciado y corriendo.**"))

        logging.info("Bucle de trading programado. Iniciando polling de Telegram.")
        self.telegram_app.run_polling()

    def _init_db(self):
        pass

    def _generate_daily_report(self):
        pass

    def _generate_weekly_report(self):
        pass
