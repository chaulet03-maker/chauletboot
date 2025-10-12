from __future__ import annotations

from config import S


class SafeBinanceClient:
    """Proxypass del SDK real, pero bloquea órdenes si estás en PAPER."""

    def __init__(self, real_client):
        self._c = real_client

    def futures_create_order(self, *args, **kwargs):
        if S.PAPER:
            raise RuntimeError("Bloqueado: intento de orden LIVE con trading_mode=simulado")
        return self._c.futures_create_order(*args, **kwargs)

    def futures_account(self, *args, **kwargs):
        if S.PAPER:
            raise RuntimeError("Bloqueado en PAPER: endpoint de cuenta")
        return self._c.futures_account(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._c, name)


def client_factory(api_key: str, secret: str):
    try:
        from binance.client import Client
    except ImportError as exc:  # pragma: no cover - solo al correr en live
        raise RuntimeError("python-binance es requerido para operar en modo real.") from exc

    rc = Client(api_key, secret)
    return SafeBinanceClient(rc)
