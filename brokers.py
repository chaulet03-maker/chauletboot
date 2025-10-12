from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SimBroker:
    """Simula fills al precio de mercado que le pases (o último price)."""

    def __init__(self, equity_usdt: float):
        self.equity = float(equity_usdt)
        self.pos_base = 0.0
        self.avg_price: float | None = None

    def place_order(self, side: str, qty: float, price: float, **kwargs: Any) -> dict[str, Any]:
        oid = f"sim-{int(time.time() * 1000)}"
        logger.info("PAPER ORDER %s %.6f @ %.2f → FILLED [%s]", side, qty, price, oid)
        # TODO: fees/pnl/avg_price si lo estás llevando (igual a tu live)
        payload: dict[str, Any] = {
            "orderId": oid,
            "status": "FILLED",
            "price": price,
            "executedQty": qty,
            "side": side,
            "sim": True,
        }
        symbol = kwargs.get("symbol")
        if symbol:
            payload["symbol"] = symbol
        if "sl" in kwargs:
            payload["sl"] = kwargs["sl"]
        if "tp" in kwargs:
            payload["tp"] = kwargs["tp"]
        return payload


class BinanceBroker:
    """Envuelve al cliente real SOLO en live."""

    def __init__(self, client: Any):
        self.client = client

    def place_order(self, side: str, qty: float, price: float, **kwargs: Any) -> Any:
        symbol = kwargs.get("symbol")
        if not symbol:
            raise ValueError("BinanceBroker requiere 'symbol' para enviar la orden.")
        params = {k: v for k, v in kwargs.items() if k not in {"symbol", "sl", "tp"}}
        # tu ruta real (market/limit según tu estrategia)
        return self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty,
            **params,
        )


def build_broker(settings, client_factory: Callable[..., Any]):
    if settings.PAPER:
        logger.warning("MODO: 🧪 SIMULADO | equity inicial: %.2f USDT", settings.start_equity)
        return SimBroker(settings.start_equity)

    assert settings.binance_api_key and settings.binance_api_secret, (
        "Faltan credenciales Binance para modo 'real'."
    )
    client = client_factory(api_key=settings.binance_api_key, secret=settings.binance_api_secret)
    logger.warning("MODO: 🔴 REAL | Binance listo.")
    return BinanceBroker(client)
