
from __future__ import annotations
import os, time, uuid, logging
from dataclasses import dataclass, field
from .leverage_policy import choose_leverage
from .risk_sizer import compute_position_size_usd
from .rounding import clamp_to_tick

log = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    side: str          # long/short
    qty: float
    entry_price: float
    leverage: int
    isolated: bool = True
    unrealized_pnl: float = 0.0

@dataclass
class Account:
    equity: float = 1000.0
    positions: dict = field(default_factory=dict)  # symbol -> Position

class PaperFuturesEngine:
    def __init__(self):
        self.acc = Account(equity=float(os.environ.get("PAPER_EQUITY_START","1000")))
        self.maker_fee = float(os.environ.get("FEE_MAKER","0.0002"))
        self.taker_fee = float(os.environ.get("FEE_TAKER","0.0004"))
        self.tick = float(os.environ.get("TICK_SIZE","0.1"))
        self.last_dd = 0.0

    def place_order_sync(self, symbol: str, side: str, type: str, qty: float|None=None, price: float|None=None, params: dict|None=None):
        """Simple market-fill simulator with leverage bounds and fee accounting."""
        params = params or {}
        # choose leverage adaptively (atr%/dd placeholders)
        lev = choose_leverage(atr_pct=0.02, dd_recent=self.last_dd)
        # compute qty if not provided, based on equity and signal strength
        if not qty or qty <= 0:
            price_ref = price or float(params.get("mark_price", 0.0)) or 100.0
            usd = compute_position_size_usd(self.acc.equity, price_ref, signal_strength=1.0)
            qty = usd / max(1e-9, price_ref)
        # compute fills (market assumption)
        px = float(price or params.get("mark_price", 0.0) or 100.0)
        px = clamp_to_tick(px, self.tick)
        fee = self.taker_fee * px * qty
        notional = px * qty
        # update positions bookkeeping (simplified netting)
        pos = self.acc.positions.get(symbol)
        if side.lower() == "buy":
            side_norm = "long"
        else:
            side_norm = "short"
        # Replace existing position for simplicity in this stub
        self.acc.positions[symbol] = Position(symbol=symbol, side=side_norm, qty=qty, entry_price=px, leverage=lev)
        self.acc.equity -= fee * 1.0  # pay fee
        order_id = "paper_" + uuid.uuid4().hex[:12]
        log.info("[PAPER] %s %s %s @ %s (qty=%s, lev=x%d) fee=%s", type.upper(), symbol, side, px, qty, lev, round(fee,4))
        return {
            "id": order_id,
            "status": "filled",
            "symbol": symbol,
            "type": type,
            "side": side,
            "price": px,
            "amount": qty,
            "cost": notional,
            "fee": {"cost": fee, "currency": "USDT"},
            "leverage": lev,
            "info": {"mode": "paper", "engine": "PaperFuturesEngine"}
        }

    def cancel_order_sync(self, order_id: str, symbol: str|None=None, params: dict|None=None):
        # In stub, nothing to cancel post-fill; return cancelled state
        return {"id": order_id, "status": "canceled", "symbol": symbol, "info": {"mode":"paper"}}
