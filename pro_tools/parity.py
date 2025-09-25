
from __future__ import annotations
import os, logging
from .paper_futures_engine import PaperFuturesEngine

log = logging.getLogger(__name__)

class Parity:
    def __init__(self):
        self.mode = os.environ.get("MODE","paper").lower()
        self._paper = PaperFuturesEngine() if self.mode == "paper" else None
        self.is_paper = self.mode == "paper"

    # sync interface (most user codes are sync)
    def place_order_sync(self, symbol: str, side: str, type: str, qty: float|None=None, price: float|None=None, params: dict|None=None):
        if self.is_paper:
            return self._paper.place_order_sync(symbol=symbol, side=side, type=type, qty=qty, price=price, params=params or {})
        else:
            raise NotImplementedError("LIVE parity routing requires binding to your exchange client")

    def cancel_order_sync(self, order_id: str, symbol: str|None=None, params: dict|None=None):
        if self.is_paper:
            return self._paper.cancel_order_sync(order_id=order_id, symbol=symbol, params=params or {})
        else:
            raise NotImplementedError("LIVE parity routing requires binding to your exchange client")

# Global factory with resilience
def ensure_parity() -> Parity:
    try:
        return Parity()
    except Exception as e:
        log.exception("Cannot construct Parity: %s", e)
        # fallback to paper
        os.environ["MODE"] = "paper"
        return Parity()
