import sqlite3
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

__path__ = [os.path.join(os.path.dirname(__file__), "storage")]
try:
    __spec__.submodule_search_locations = __path__
except Exception:
    pass

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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entry_locks (
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        anchor_epoch INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (symbol, side, anchor_epoch)
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

    def get_last_trade(self) -> Optional[dict]:
        """Obtiene el trade más reciente persistido, si existe."""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM trades ORDER BY datetime(close_timestamp) DESC, id DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return dict(row)
        except Exception as exc:
            logging.error(f"Error al obtener último trade: {exc}")
        return None

    def acquire_entry_lock(
        self,
        symbol: str,
        side: str,
        anchor_epoch: Optional[int],
        *,
        ttl_seconds: int = 12 * 60 * 60,
    ) -> bool:
        """Idempotentemente registra un lock para una señal de entrada.

        Devuelve ``True`` si el lock fue adquirido (no existía) y ``False``
        si ya había un registro previo dentro de la ventana de tiempo.
        """

        if not symbol or not side or anchor_epoch is None:
            return True

        symbol_u = str(symbol).upper().strip()
        side_u = str(side).upper().strip()
        try:
            anchor_epoch_i = int(anchor_epoch)
        except (TypeError, ValueError):
            return True

        ttl = max(int(ttl_seconds), 60)
        cutoff = anchor_epoch_i - ttl

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM entry_locks WHERE anchor_epoch < ?",
                    (cutoff,),
                )
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO entry_locks
                        (symbol, side, anchor_epoch, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        symbol_u,
                        side_u,
                        anchor_epoch_i,
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as exc:
            logging.debug(f"Error al registrar entry_lock: {exc}")
        return True
