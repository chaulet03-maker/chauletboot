
from __future__ import annotations
def round_to_step(x: float, step: float) -> float:
    if step <= 0: return x
    return round(x/step)*step
def clamp_to_tick(price: float, tick: float) -> float:
    if tick <= 0: return price
    return round(price/tick)*tick
