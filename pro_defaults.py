
from __future__ import annotations
import os, logging

def apply_defaults():
    env = os.environ
    env.setdefault("MODE", "paper")
    env.setdefault("STRATEGY", "COTS")
    env.setdefault("SYMBOLS", "BTC/USDT,ETH/USDT")
    env.setdefault("TIMEFRAME", "1h")

    # Fees (Binance defaults)
    is_futures = True  # default to futures for parity testing
    if is_futures:
        env.setdefault("FEE_MAKER", "0.0002")
        env.setdefault("FEE_TAKER", "0.0004")
    else:
        env.setdefault("FEE_MAKER", "0.0010")
        env.setdefault("FEE_TAKER", "0.0010")

    # Parity v4
    env.setdefault("PAPER_EQUITY_START", "1000")
    env.setdefault("LEVERAGE_MIN", "5")
    env.setdefault("LEVERAGE_MAX", "15")
    env.setdefault("MARGIN_MODE", "isolated")
    env.setdefault("RISK_PCT_TRADE", "0.01")
    env.setdefault("MAX_RISK_USD", "25")
    env.setdefault("MAX_GROSS_EXPOSURE", "3.0")
    env.setdefault("MAX_SYMBOL_EXPOSURE", "1.5")
    env.setdefault("MAX_CONCURRENT_TRADES", "5")
    env.setdefault("FUNDING_DEFAULT_RATE", "0.0001")
    env.setdefault("SLIPPAGE_BPS_BASE", "2")
    env.setdefault("PARITY_AUTOPATCH", "1")

    env.setdefault("LOG_LEVEL", "INFO")
    logging.getLogger(__name__).info("Defaults applied: MODE=%s STRATEGY=%s", env["MODE"], env["STRATEGY"])
