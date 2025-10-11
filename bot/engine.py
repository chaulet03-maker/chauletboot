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

    async def trading_loop(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            schedule.run_pending()
            # Aquí va la lógica de trading completa...
            logging.info("Ciclo de trading...")
        except Exception as e:
            logging.error(f"Error en trading_loop: {e}")
            await self.notifier.send(f"💥 Error en ciclo de trading: {e}")

    def run(self):
        job_queue = self.telegram_app.job_queue
        job_queue.run_repeating(self.trading_loop, interval=60, first=5)
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Si el loop ya está corriendo, programamos la notificación y listo.
            loop.create_task(self.notifier.send("✅ **Bot iniciado y corriendo.**"))
        else:
            # Si no, lo usamos para la notificación inicial.
            loop.run_until_complete(self.notifier.send("✅ **Bot iniciado y corriendo.**"))

        logging.info("Bucle de trading programado. Iniciando polling de Telegram.")
        self.telegram_app.run_polling()
        
    # Funciones de reporte y base de datos van aquí
    def _generate_daily_report(self): pass
    def _generate_weekly_report(self): pass
    def _init_db(self): pass
