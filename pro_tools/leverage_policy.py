
from __future__ import annotations
import os, math

def choose_leverage(atr_pct: float, dd_recent: float) -> int:
    lmin = int(float(os.environ.get("LEVERAGE_MIN","5")))
    lmax = int(float(os.environ.get("LEVERAGE_MAX","15")))
    atr_pct = max(0.001, min(0.2, atr_pct or 0.02))
    # Lower ATR -> higher leverage; simple inverse map with clamp
    ideal = lmax - (lmax-lmin) * (atr_pct/0.2)
    # DD penalty
    dd = max(0.0, dd_recent or 0.0)
    penalty = 0.0
    if dd > 0.05: penalty += 2
    if dd > 0.10: penalty += 3
    lev = int(max(lmin, min(lmax, round(ideal - penalty))))
    return lev
