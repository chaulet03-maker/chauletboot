
from __future__ import annotations
import os

def cap(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def compute_position_size_usd(equity_usd: float, price: float, signal_strength: float = 1.0) -> float:
    # Risk % per trade
    risk_pct = float(os.environ.get("RISK_PCT_TRADE","0.01"))  # 1%
    max_risk_usd = float(os.environ.get("MAX_RISK_USD","25"))
    gross_cap = float(os.environ.get("MAX_GROSS_EXPOSURE","3.0"))  # 300%
    sym_cap   = float(os.environ.get("MAX_SYMBOL_EXPOSURE","1.5")) # 150%
    # base risk allocation
    risk_usd = min(equity_usd * risk_pct, max_risk_usd)
    # convert to notionally target pnl size (simple proxy: risk_usd ~ 1% adverse move)
    notion = risk_usd * 100.0  # naive: 1% stop -> 100x
    return cap(notion*signal_strength, 10.0, equity_usd * gross_cap)
