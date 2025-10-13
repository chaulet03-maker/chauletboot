from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging, os, json, time
log = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

PRIORITY = [
    "pos_open","blackout","freeze_90","cooldown",
    "grid_oob","trend_block_long","trend_block_short",
    "gate_fail","risk_block","min_notional",
    "low_vol","adx_low","rsi_filter","funding_bad",
    "exchange_error","no_signal"
]

MSG = {
    "pos_open":          "Ya hay una posición abierta",
    "blackout":          "Fuera del horario de trading",
    "freeze_90":         "Congelamiento del 90% activo",
    "cooldown":          "En enfriamiento (cooldown)",
    "grid_oob":          "No tocó la zona de pullback",
    "trend_block_long":  "LONG bloqueado (precio < EMA200)",
    "trend_block_short": "SHORT bloqueado (precio > EMA200)",
    "gate_fail":         "Gate insuficiente (comisiones > ventaja)",
    "risk_block":        "Bloqueado por riesgo/stop",
    "min_notional":      "Tamaño mínimo del exchange no alcanzado",
    "low_vol":           "Volatilidad insuficiente (ATR bajo)",
    "adx_low":           "Tendencia débil (ADX bajo)",
    "rsi_filter":        "RSI fuera del rango válido",
    "funding_bad":       "Funding desfavorable",
    "exchange_error":    "Error del exchange",
    "no_signal":         "Sin señal utilizable",
}

@dataclass
class MotiveItem:
    ts: float
    symbol: str
    side_pref: Optional[str]
    price: float
    codes: List[str] = field(default_factory=list)
    ctx: Dict[str, Any] = field(default_factory=dict)

    def human_line(self, tz: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(self.ts, tz=timezone.utc)
        if tz and ZoneInfo:
            try: dt = dt.astimezone(ZoneInfo(tz))
            except Exception: pass
        t = dt.strftime("%H:%M")
        seen=set()
        ordered=[c for c in PRIORITY if c in self.codes and (c not in seen and not seen.add(c))]
        if not ordered: ordered=["no_signal"]
        return f"• {t} — {self.symbol}: " + " · ".join(MSG[c] for c in ordered)

class MotivesBuffer:
    def __init__(self, maxlen:int=200, persist_path: Optional[str]=None):
        self._buf=deque(maxlen=maxlen)
        self._persist_path=persist_path
        if self._persist_path:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)

    def add(self, item:MotiveItem):
        self._buf.append(item)
        log.debug("MOTIVES/ADD codes=%s ctx_keys=%s", item.codes, list(item.ctx.keys()))
        if self._persist_path:
            try:
                with open(self._persist_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": item.ts, "symbol": item.symbol, "codes": item.codes,
                        "side_pref": item.side_pref, "price": item.price
                    }, ensure_ascii=False) + "\n")
            except Exception as e:
                log.debug("MOTIVES persist fail: %s", e)

    def last(self, n:int=10) -> List[MotiveItem]:
        return list(self._buf)[-n:]

# Persistencia opcional para multi-proceso
PERSIST = os.getenv("MOTIVES_FILE") or os.path.join(os.getenv("DATA_DIR","/app/data"), "motives.jsonl")
MOTIVES = MotivesBuffer(maxlen=400, persist_path=PERSIST)

def compute_codes(ctx: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    if ctx.get("has_open"): codes.append("pos_open")
    if ctx.get("blackout"): codes.append("blackout")
    if ctx.get("freeze_90"): codes.append("freeze_90")
    if ctx.get("cooldown"): codes.append("cooldown")

    price,anchor,step,span = ctx.get("price"),ctx.get("anchor"),ctx.get("step"),ctx.get("span")
    if None not in (price,anchor,step,span):
        short_lo=anchor-span; short_hi=anchor-step
        long_lo=anchor+step;  long_hi=anchor+span
        in_short = short_lo <= price <= short_hi
        in_long  = long_lo  <= price <= long_hi
        if not (in_short or in_long): codes.append("grid_oob")
        ema1,ema4 = ctx.get("ema200_1h"),ctx.get("ema200_4h")
        if None not in (ema1,ema4):
            long_allowed  = price>ema1 and price>ema4
            short_allowed = price<ema1 and price<ema4
            if in_long and not long_allowed:   codes.append("trend_block_long")
            if in_short and not short_allowed: codes.append("trend_block_short")

    if ctx.get("gate_ok") is False: codes.append("gate_fail")
    if ctx.get("risk_ok") is False: codes.append("risk_block")
    atrp=ctx.get("atrp");  adx=ctx.get("adx"); adx_thr=float(ctx.get("adx_thr",25.0))
    if atrp is not None and atrp<0.006: codes.append("low_vol")
    if adx is not None and adx<adx_thr: codes.append("adx_low")

    for r in (ctx.get("reasons") or []):
        s=str(r).lower()
        if "rsi" in s and "fuera" in s: codes.append("rsi_filter")
        if "funding" in s and ("malo" in s or "desfavorable" in s): codes.append("funding_bad")
        if "min" in s and ("notional" in s or "tamaño" in s): codes.append("min_notional")
        if "exchange" in s or "api" in s or "binance" in s: codes.append("exchange_error")
        if "fuera de rango" in s or "grid" in s: codes.append("grid_oob")
        if "sin señal" in s: codes.append("no_signal")

    out,seen=[],set()
    for c in codes:
        if c not in seen: out.append(c); seen.add(c)
    if not out: out=["no_signal"]
    return out
