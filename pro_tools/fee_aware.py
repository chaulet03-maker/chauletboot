
from __future__ import annotations
def sl_tp_fee_aware(entry: float, qty: float, risk_usd: float, target_usd: float, fee_in: float, fee_out: float, side: str) -> tuple[float,float]:
    if side == "buy":  # long
        sl = (entry*qty - risk_usd - fee_in*entry*qty) / (qty*(1 + fee_out))
        tp = (entry*qty + target_usd - fee_in*entry*qty) / (qty*(1 - fee_out))
    else:  # sell short
        sl = (entry*qty + risk_usd + fee_in*entry*qty) / (qty*(1 + fee_out))
        tp = (entry*qty - target_usd - fee_in*entry*qty) / (qty*(1 - fee_out))
    return sl, tp
