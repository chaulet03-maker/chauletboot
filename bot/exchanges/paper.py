from dataclasses import dataclass
import time

@dataclass
class Fill:
  price: float
  qty: float
  side: str
  ts: float

class PaperExchange:
    def __init__(self, fees: dict, slippage_bps: int = 5):
        self.fees = fees or {"taker":0.0002,"maker":0.0002}
        self.slippage_bps = int(slippage_bps)
        self._stops = {} # Para simular SLs
        self._takes = {} # Para simular TPs

    async def set_leverage(self, symbol: str, leverage: int):
        try:
            self._lev = getattr(self, '_lev', {})
            self._lev[symbol] = leverage
        except Exception:
            pass
        return True

    async def market_order(self, symbol: str, side: str, qty: float, ref_price: float):
        slip = self.slippage_bps / 10000.0
        price = ref_price * (1 + slip) if side == 'long' else ref_price * (1 - slip)
        fee = abs(price * qty) * self.fees.get("taker", 0.0002)
        return Fill(price=price, qty=qty, side=side, ts=time.time()), fee

    # --- MÉTODOS INTERNOS DE SIMULACIÓN (añadidos para completitud) ---
    def _place_stop_loss(self, symbol, side, qty, sl_price):
        """Simula el almacenamiento de una orden Stop Loss."""
        # En una simulación real, guardarías esto para verificarlo en cada tick.
        print(f"PAPER: Registrando SL para {symbol} ({side} {qty}) @ {sl_price}")
        self._stops[symbol] = {'side': side, 'price': sl_price}

    def _place_take_profit(self, symbol, side, qty, tp_price):
        """Simula el almacenamiento de una orden Take Profit."""
        # En una simulación real, guardarías esto para verificarlo en cada tick.
        print(f"PAPER: Registrando TP para {symbol} ({side} {qty}) @ {tp_price}")
        self._takes[symbol] = {'side': side, 'price': tp_price}

    # =============================================================
    # === PARCHE APLICADO AQUÍ ===
    # =============================================================
    async def place_protections(self, symbol, side, qty, sl, tp1=None, tp2=None):
        """
        PAPER: coloca un único SL y un único TP (sin parciales).
        Si vienen tp1 y tp2, prioriza tp2; si no, usa tp1.
        """
        tp_final = tp2 if tp2 is not None else tp1
        # registra/almacena internamente (dependiendo tu implementación)
        self._place_stop_loss(symbol, side, qty, sl)
        if tp_final is not None:
            self._place_take_profit(symbol, side, qty, tp_final)
        return True