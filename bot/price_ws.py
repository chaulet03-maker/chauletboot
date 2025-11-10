"""Utilidades para streamear precios vía WebSocket.

Este módulo abstrae el stream de precios de Binance USD-M para alimentar el
cache interno del ``Exchange`` cuando el bot opera en modo REAL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import websockets
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidStatusCode,
    WebSocketException,
)


CallbackType = Callable[[str, float, Optional[float]], None]


@dataclass(slots=True)
class _ParsedMessage:
    symbol: str
    price: float
    event_ts: Optional[float]


class PriceStream:
    """Consume precios en vivo desde Binance via WebSocket.

    Parameters
    ----------
    symbol:
        Símbolo configurado (por ej. ``"BTC/USDT"``).
    callback:
        Función que recibirá ``(symbol, price, event_ts)`` cuando arriben ticks.
    base_url:
        Endpoint base del WS. Se permite sobreescribirlo para testnet.
    stream:
        Tipo de stream a utilizar. Actualmente se soporta ``"mark"`` (por defecto)
        y ``"book_ticker"``.
    interval:
        Intervalo del stream de mark price. Binance admite ``1s`` o ``3s``.
    """

    def __init__(
        self,
        *,
        symbol: str,
        callback: CallbackType,
        base_url: str,
        stream: str = "mark",
        interval: str = "1s",
    ) -> None:
        if not callable(callback):
            raise TypeError("callback debe ser callable")

        self.symbol = symbol or "BTC/USDT"
        self._callback = callback
        self.base_url = (base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("base_url es requerido para PriceStream")
        self.stream = (stream or "mark").lower().strip()
        self.interval = interval or "1s"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._logger = logging.getLogger(__name__)
        self._running = False

    # --------- API pública -------------------------------------------------
    def start(self) -> None:
        """Inicia el hilo y loop asyncio que consume el WS."""

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"PriceStream-{self._normalized_symbol().upper()}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Detiene el stream y espera un cierre ordenado."""

        self._running = False
        self._stop_event.set()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(lambda: None)

        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._thread = None

    # --------- Internals ---------------------------------------------------
    def _normalized_symbol(self) -> str:
        return self.symbol.replace("/", "").lower()

    def _stream_path(self) -> str:
        norm = self._normalized_symbol()
        if self.stream in {"book_ticker", "bookticker"}:
            return f"{norm}@bookTicker"
        # default mark price
        interval = self.interval or "1s"
        return f"{norm}@markPrice@{interval}"

    def _parse_message(self, message: object) -> Optional[_ParsedMessage]:
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except Exception:
                return None
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                return None
        if not isinstance(message, dict):
            return None

        if "data" in message and isinstance(message["data"], dict):
            message = message["data"]

        if "result" in message and len(message) == 1:
            return None

        raw_symbol = message.get("s") or message.get("symbol") or self._normalized_symbol()
        symbol = str(raw_symbol)

        raw_ts = (
            message.get("E")
            or message.get("eventTime")
            or message.get("T")
            or message.get("time")
        )
        event_ts: Optional[float] = None
        if raw_ts is not None:
            try:
                ts_val = float(raw_ts)
                if ts_val > 1_000_000_000_000:
                    ts_val /= 1000.0
                elif ts_val > 1_000_000_000:
                    ts_val /= 1000.0
                event_ts = ts_val
            except (TypeError, ValueError):
                event_ts = None

        price_keys = (
            "p",
            "price",
            "c",
            "lastPrice",
            "markPrice",
            "ap",
        )
        price_value: Optional[float] = None
        for key in price_keys:
            raw = message.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                price_value = value
                break
        if price_value is None:
            return None

        return _ParsedMessage(symbol=symbol, price=price_value, event_ts=event_ts)

    async def _connect_once(self, url: str, open_timeout: float = 10.0):
        self._logger.debug("Conectando PriceStream a %s", url)
        return await asyncio.wait_for(
            websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_queue=None,
                max_size=2**20,
            ),
            timeout=open_timeout,
        )

    async def _on_message(self, raw: object) -> None:
        parsed = self._parse_message(raw)
        if not parsed:
            return
        try:
            self._callback(parsed.symbol, parsed.price, parsed.event_ts)
        except Exception:  # pragma: no cover - defensivo
            self._logger.exception("Error al procesar tick de precio")

    async def _run_forever(self) -> None:
        backoff = 1.0
        url = f"{self.base_url}/{self._stream_path()}"
        while self._running and not self._stop_event.is_set():
            ws = None
            try:
                ws = await self._connect_once(url)
                self._logger.info("WS de precios conectado a %s", url)
                backoff = 1.0

                async for raw in ws:
                    if not self._running or self._stop_event.is_set():
                        break
                    await self._on_message(raw)

            except (asyncio.TimeoutError, ConnectionClosedError, ConnectionClosedOK) as exc:
                if self._stop_event.is_set() or not self._running:
                    self._logger.debug("WS de precios cerrado: %s", exc)
                else:
                    self._logger.warning(
                        "WS de precios desconectado (%s). Reintentando...",
                        exc,
                    )
            except asyncio.CancelledError:
                self._logger.info("WS cancelado (shutdown).")
                break
            except (InvalidStatusCode, WebSocketException) as exc:
                self._logger.warning("WS de precios falló al conectar: %s", exc)
            except Exception as exc:
                if self._stop_event.is_set() or not self._running:
                    self._logger.debug("WS de precios detenido: %s", exc)
                else:
                    self._logger.exception("WS de precios error inesperado: %s", exc)
            finally:
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass

            if not self._running or self._stop_event.is_set():
                break

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

        self._logger.debug("Loop de PriceStream finalizado para %s", self.symbol)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_forever())
        finally:
            self._running = False
            tasks = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                self._loop.run_until_complete(
                    asyncio.gather(*tasks, return_exceptions=True)
                )
            self._loop.close()
            self._loop = None


__all__ = ["PriceStream"]
