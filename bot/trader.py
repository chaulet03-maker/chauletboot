from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


@dataclass
class TraderState:
    """
    Estado mínimo que el engine necesita:
    - equity: capital disponible (se actualiza con PnL neto)
    - killswitch: si está True, no se deben abrir nuevas posiciones
    - positions: mapa símbolo -> lista de lotes (dicts)
    - last_entry_ts_by_symbol: timestamp de la última entrada por símbolo (p/ cooldowns)
    """
    equity: float = 1000.0
    killswitch: bool = False
    positions: Dict[str, List[dict]] = field(default_factory=dict)
    last_entry_ts_by_symbol: Dict[str, float] = field(default_factory=dict)


class Trader:
    """
    Implementación simple y explícita usada por bot.engine.TradingApp.

    Interface consumida por el engine (según tu código):
      - equity() -> float
      - open_lot(symbol, side, qty, entry, lev, sl=0, tp1=0, tp2=0, fee=0, entry_adx=0, leg=1) -> None
      - close_lot(symbol, index, exit_price, fee=0, note="") -> float
      - state.killswitch (bool)
      - state.positions (dict: symbol -> list[dict])
      - state.last_entry_ts_by_symbol (cooldown)
    """

    def __init__(self, fees: Optional[dict] = None, equity0: float = 1000.0) -> None:
        # Estructura de fees esperada en el proyecto: {"taker": float, "maker": float}
        self.fees = (fees or {"taker": 0.0002, "maker": 0.0002}).copy()
        self.state = TraderState(equity=float(equity0))

    # --------------------------
    # Lectura de estado
    # --------------------------
    def equity(self) -> float:
        """Devuelve el equity (capital) actual."""
        return float(self.state.equity)

    def get_positions(self, symbol: Optional[str] = None) -> Dict[str, List[dict]] | List[dict]:
        """Devuelve todas las posiciones o solo las de un símbolo."""
        if symbol is None:
            return self.state.positions
        return self.state.positions.get(symbol, [])

    def position_count(self, symbol: Optional[str] = None) -> int:
        if symbol is None:
            return sum(len(v) for v in self.state.positions.values())
        return len(self.state.positions.get(symbol, []))

    # --------------------------
    # Control de riesgo/kill
    # --------------------------
    def get_killswitch(self) -> bool:
        return bool(self.state.killswitch)

    def set_killswitch(self, on: bool) -> None:
        self.state.killswitch = bool(on)

    # --------------------------
    # Operaciones
    # --------------------------
    def open_lot(
        self,
        symbol: str,
        side: str,             # 'long' | 'short'
        qty: float,
        entry: float,
        lev: int,
        sl: float = 0.0,
        tp1: float = 0.0,
        tp2: float = 0.0,
        fee: float = 0.0,      # fee total de apertura (USDT)
        entry_adx: float = 0.0,
        leg: int = 1,
    ) -> None:
        """
        Abre un lote agregándolo a positions[symbol].
        Descuenta el fee de apertura del equity inmediatamente (paper/real coherente).
        """
        lots = self.state.positions.setdefault(symbol, [])
        lots.append({
            "side": side,
            "qty": float(qty),
            "entry": float(entry),
            "lev": int(lev),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "fee": float(fee),          # fee de apertura
            "entry_adx": float(entry_adx),
            "leg": int(leg),
        })
        # impacto de fees de apertura en equity
        self.state.equity -= abs(float(fee))
        # cooldown por símbolo
        self.state.last_entry_ts_by_symbol[symbol] = time.time()

    def close_lot(
        self,
        symbol: str,
        index: int,
        exit_price: float,
        fee: float = 0.0,      # fee de cierre (USDT)
        note: str = "",
    ) -> float:
        """
        Cierra un lote por índice y devuelve el PnL neto (incluyendo fee de cierre).
        Si el índice es inválido o no hay posición, devuelve 0.0.
        """
        lots = self.state.positions.get(symbol)
        if not lots or index < 0 or index >= len(lots):
            return 0.0

        lot = lots.pop(index)
        side = lot.get("side", "long")
        qty = float(lot.get("qty", 0.0))
        entry = float(lot.get("entry", float(exit_price)))

        # PnL bruto (qty en "contratos" equivalentes a tamaño nocional/price si así lo maneja el exchange paper)
        if side == "long":
            pnl = (float(exit_price) - entry) * qty
        else:  # short
            pnl = (entry - float(exit_price)) * qty

        # PnL neto con fee de cierre (el de apertura ya se descontó en open_lot)
        pnl_net = float(pnl) - abs(float(fee))

        # Actualizar equity
        self.state.equity += pnl_net

        # Limpiar símbolo si quedó vacío
        if not lots:
            self.state.positions.pop(symbol, None)

        return float(pnl_net)

    # --------------------------
    # Utilidades opcionales
    # --------------------------
    def close_all_symbol(self, symbol: str, exit_price: float, fee_per_lot: float = 0.0) -> float:
        """Cierra todas las posiciones de un símbolo y devuelve PnL total."""
        total = 0.0
        # Cerrar desde el final evita reindexar
        for idx in range(len(self.state.positions.get(symbol, [])) - 1, -1, -1):
            total += self.close_lot(symbol, idx, exit_price, fee=fee_per_lot)
        return total

    def close_all(self, price_map: Dict[str, float], fee_per_lot: float = 0.0) -> float:
        """Cierra todas las posiciones de todos los símbolos usando un mapa de precios de salida."""
        total = 0.0
        for sym in list(self.state.positions.keys()):
            px = price_map.get(sym)
            if px is None:
                # si no hay precio provisto, intentar no cerrar ese símbolo
                continue
            total += self.close_all_symbol(sym, px, fee_per_lot=fee_per_lot)
        return total
