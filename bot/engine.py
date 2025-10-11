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

        schedule.every().day.at("08:00", "America/Argentina/Buenos_Aires").do(self._generate_daily_report)
        schedule.every().sunday.at("00:00", "America/Argentina/Buenos_Aires").do(self._generate_weekly_report)
        logging.info("Componentes y tareas de reporte inicializados.")

    def _get_dynamic_leverage(self, df_row):
        adx = float(df_row["adx"])
        if adx >= 25:
            leverage = 10.0
            trend_strength = "FUERTE"
        else:
            leverage = self.config.get('leverage', 5.0)
            trend_strength = "DEBIL"
        logging.info(f"Fuerza de tendencia: {trend_strength} (ADX={adx:.2f}). Usando apalancamiento: x{leverage}")
        return leverage

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

            data = add_indicators(df_1h, df_4h)
            last_candle = data.iloc[-1]

            signal = self.strategy.check_entry_signal(data)
            if not signal:
                logging.info("No se encontraron señales de entrada válidas.")
                return

            rejection_reason = self.strategy.check_all_filters(last_candle, signal)
            if rejection_reason:
                log_entry = f"[{pd.Timestamp.now(tz='UTC').strftime('%H:%M')}] Señal Rechazada: {rejection_reason}"
                self.rejection_log.append(log_entry)
                logging.info(log_entry)
                return

            leverage_for_this_trade = self._get_dynamic_leverage(last_candle)
            await self.exchange.set_leverage(leverage_for_this_trade)

            entry_price = await self.exchange.get_current_price()
            balance = await self.trader.get_balance(self.exchange)
            quantity = (balance * leverage_for_this_trade) / entry_price

            sl_price = self.strategy.calculate_sl(entry_price, last_candle, signal)
            tp_price = self.strategy.calculate_tp(entry_price, quantity, balance, signal)

            order_result = await self.exchange.create_order(signal, quantity, sl_price, tp_price)

            if order_result:
                await self.notifier.send(
                    f"🚀 **Nueva Operación Abierta: {signal}**\n"
                    f"Símbolo: {self.config['symbol']}\n"
                    f"Apalancamiento: x{leverage_for_this_trade}"
                )
                await self.trader.set_position(order_result)

        except Exception as e:
            logging.error(f"Error grave en el trading_loop: {e}", exc_info=True)
            if not self.connection_lost:
                await self.notifier.send(f"💥 **Error inesperado en el bot:** {e}")
                self.connection_lost = True

    def run(self):
        job_queue = self.telegram_app.job_queue
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
