# Compat: expone una fábrica simple que usa nuestras clases reales/paper
from .real import RealExchange
from .paper import PaperExchange

def get_exchange(mode: str, ccxt_client, fees: dict, slippage_bps: int = 5):
    mode = (mode or "paper").lower()
    if mode == "paper":
        return PaperExchange(fees, slippage_bps=slippage_bps)
    return RealExchange(ccxt_client, fees)
