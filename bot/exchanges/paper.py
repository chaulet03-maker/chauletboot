from dataclasses import dataclass
import time

@dataclass
class Fill:
  price: float
  qty: float
  side: str
  ts: float

class PaperExchange:
    async def set_leverage(self, symbol: str, leverage: int):
        try:
            self._lev = getattr(self, '_lev', {})
            self._lev[symbol] = leverage
        except Exception:
            pass
        return True

    def __init__(self, fees: dict, slippage_bps: int = 5):
        self.fees = fees or {"taker":0.0002,"maker":0.0002}
        self.slippage_bps = int(slippage_bps)

    async def market_order(self, symbol: str, side: str, qty: float, ref_price: float):
        slip = self.slippage_bps / 10000.0
        price = ref_price * (1 + slip) if side == 'long' else ref_price * (1 - slip)
        fee = abs(price * qty) * self.fees.get("taker", 0.0002)
        return Fill(price=price, qty=qty, side=side, ts=time.time()), fee
