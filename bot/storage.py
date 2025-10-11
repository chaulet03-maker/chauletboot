import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta

class Storage:
    def __init__(self, cfg):
        self.config = cfg
        self.db_path = "trades_history.db"
        self._init_db()

    def _init_db(self):
        """Crea la tabla de trades en la base de datos si no existe."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        close_timestamp TEXT NOT NULL,
                        symbol TEXT,
                        side TEXT,
                        pnl REAL,
                        note TEXT
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logging.error(f"Error al inicializar la base de datos: {e}")

    def persist_trade(self, trade_data):
        """Guarda los detalles de una operación cerrada en la base de datos."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO trades (close_timestamp, symbol, side, pnl, note) VALUES (?, ?, ?, ?, ?)",
                    (
                        datetime.now().isoformat(),
                        trade_data.get('symbol'),
                        trade_data.get('side'),
                        trade_data.get('pnl'),
                        trade_data.get('note'),
                    )
                )
                conn.commit()
            logging.info("Trade cerrado guardado en la base de datos.")
        except Exception as e:
            logging.error(f"Error al guardar trade en la base de datos: {e}")

    def get_trades_since(self, hours=None, days=None):
        """Recupera trades de la base de datos desde un punto en el tiempo."""
        trades = []
        try:
            if hours:
                since_time = datetime.now() - timedelta(hours=hours)
            elif days:
                since_time = datetime.now() - timedelta(days=days)
            else:
                return [] # No se especificó un período

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Para que devuelva diccionarios en lugar de tuplas
                conn.row_factory = sqlite3.Row 
                cursor.execute(
                    "SELECT * FROM trades WHERE close_timestamp >= ?", (since_time.isoformat(),)
                )
                trades = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Error al obtener trades de la base de datos: {e}")
        
        return trades
