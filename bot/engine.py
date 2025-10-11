import logging
import time
import threading
import sqlite3
import pandas as pd
from collections import deque
import schedule
import requests
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Asumiendo que tus otros archivos/clases se llaman así
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

        # Estado del bot
        self.is_paused = False
        self.connection_lost = False
        self.rejection_log = deque(maxlen=10)

        # --- Componentes de Telemetría ---
        self.telegram_app = setup_telegram_bot(self)
        self.notifier = Notifier(application=self.telegram_app, cfg=self.config)

        # --- Programación de Tareas ---
        schedule.every().day.at("08:00", "America/Argentina/Buenos_Aires").do(self._generate_daily_report)
        schedule.every().sunday.at("00:00", "America/Argentina/Buenos_Aires").do(self._generate_weekly_report)
        logging.info("Tareas de reporte programadas correctamente.")

    def _get_dynamic_leverage(self, df_row):
        """
        Analiza la fuerza de la tendencia (ADX) y devuelve el apalancamiento
        apropiado según 2 niveles: x5 o x10.
        """
        adx = float(df_row["adx"])

        if adx >= 25:
            leverage = 10.0
            trend_strength = "FUERTE"
        else:
            leverage = self.config.get('leverage', 5.0) # Usa el apalancamiento base de la config
            trend_strength = "DEBIL"

        logging.info(f"Fuerza de tendencia: {trend_strength} (ADX={adx:.2f}). Usando apalancamiento: x{leverage}")
        return leverage

    def run(self):
        """ Inicia todos los procesos del bot. """
        telegram_thread = threading.Thread(target=self.telegram_app.run_polling)
        telegram_thread.daemon = True
        telegram_thread.start()
        self.notifier.send("✅ **Bot iniciado y corriendo.**")
        logging.info("Bot de Telegram iniciado y escuchando comandos...")
        logging.info("Iniciando bucle de trading...")
        
        while True:
            schedule.run_pending()
            try:
                if self.connection_lost:
                    self.notifier.send("✅ **Conexión Reestablecida.** El bot vuelve a operar normalmente.")
                    self.connection_lost = False
                
                # 1. ¿Hay posición abierta? Si sí, gestionarla.
                position = self.trader.check_open_position()
                if position:
                    closed_trade = self.strategy.manage_position(position)
                    if isinstance(closed_trade, dict) and {
                        "close_timestamp",
                        "side",
                        "pnl",
                        "note",
                    }.issubset(closed_trade.keys()):
                        self._persist_trade(closed_trade)
                    time.sleep(60) # Espera 1 minuto antes de volver a chequear
                    continue

                # 2. Si no hay posición y el bot no está en pausa, buscar señales.
                if self.is_paused:
                    logging.info("El bot está en pausa, no se buscan nuevas señales.")
                    time.sleep(60)
                    continue
                
                # 3. Obtener datos y calcular indicadores
                df_1h = self.exchange.get_klines('1h')
                df_4h = self.exchange.get_klines('4h')
                data = add_indicators(df_1h, df_4h)
                last_candle = data.iloc[-1]
                
                # 4. Buscar señal de entrada
                signal = self.strategy.check_entry_signal(data)
                if not signal:
                    logging.info("No se encontraron señales de entrada en este ciclo.")
                    time.sleep(60) # Espera
                    continue
                
                # 5. Aplicar filtros y apalancamiento dinámico
                rejection_reason = self.strategy.check_all_filters(last_candle, signal)
                if rejection_reason:
                    log_entry = f"[{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M')}] No se entró. Motivo: {rejection_reason}"
                    self.rejection_log.append(log_entry)
                    logging.info(log_entry)
                    time.sleep(60) # Espera
                    continue
                    
                # 6. ¡SEÑAL VÁLIDA! Proceder a abrir operación
                leverage_for_this_trade = self._get_dynamic_leverage(last_candle)
                self.exchange.set_leverage(leverage_for_this_trade)
                
                # 7. Calcular tamaño y abrir la orden
                entry_price = self.exchange.get_current_price()
                quantity = (self.trader.get_balance() * leverage_for_this_trade) / entry_price
                
                sl_price = self.strategy.calculate_sl(entry_price, last_candle)
                tp_price = self.strategy.calculate_tp(entry_price, quantity, self.trader.get_balance())

                self.exchange.create_order(signal, quantity, sl_price, tp_price)
                self.notifier.send(f"🚀 **Nueva Operación Abierta: {signal}**\nSymbol: {self.config['symbol']}\nLev: x{leverage_for_this_trade}")

            except (requests.exceptions.ConnectionError, BinanceAPIException) as e:
                logging.error(f"Error de conexión o API: {e}")
                if not self.connection_lost:
                    self.notifier.send(f"🔌 **Error de conexión.** El bot reintentará automáticamente.")
                    self.connection_lost = True
                time.sleep(60)
            except Exception as e:
                logging.error(f"Error inesperado en el bucle principal: {e}")
                self.notifier.send(f"💥 **Error inesperado en el bot:** {e}")
                time.sleep(300) # Espera 5 minutos ante un error desconocido

    def _init_db(self):
        """Crea la tabla de trades en la base de datos si no existe."""
        self.db_path = "trades_history.db" # La DB se guardará en la carpeta raíz
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    close_timestamp TEXT NOT NULL,
                    side TEXT,
                    pnl REAL,
                    note TEXT
                )
                """
            )
            conn.commit()

    def _persist_trade(self, trade_data):
        """Guarda los detalles de una operación cerrada en la base de datos."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO trades (close_timestamp, side, pnl, note) VALUES (?, ?, ?, ?)",
                    (
                        trade_data['close_timestamp'],
                        trade_data['side'],
                        trade_data['pnl'],
                        trade_data['note'],
                    )
                )
                conn.commit()
            logging.info("Trade cerrado guardado en la base de datos.")
        except Exception as e:
            logging.error(f"Error al guardar trade en la base de datos: {e}")

    # Aquí irían las funciones _generate_daily_report y _generate_weekly_report
    def _generate_daily_report(self):
        # ... tu código para los reportes ...
        pass
    def _generate_weekly_report(self):
        # ... tu código para los reportes ...
        pass
