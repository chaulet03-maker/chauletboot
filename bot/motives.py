from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any

# Prioridad de motivos (orden de aparición)
PRIORITY = [
    "pos_open", "blackout", "freeze_90", "cooldown",
    "grid_oob", "trend_block_long", "trend_block_short",
    "gate_fail", "risk_block", "min_notional",
    "low_vol", "adx_low", "rsi_filter", "funding_bad",
    "exchange_error", "no_signal"
]

# Mensajes simples
MSG = {
    "pos_open":          "Ya hay una posición abierta",
    "blackout":          "Fuera del horario de trading",
    "freeze_90":         "Congelamiento del 90% activo",
    "cooldown":          "En enfriamiento (cooldown)",
    "grid_oob":          "No tocó la zona de pullback",
    "trend_block_long":  "LONG bloqueado (precio < EMA200)",
    "trend_block_short": "SHORT bloqueado (precio > EMA200)",
    "gate_fail":         "Ventaja insuficiente vs comisiones (gate)",
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
    side_pref: str | None
    price: float
    codes: List[str] = field(default_factory=list)
    ctx: Dict[str, Any] = field(default_factory=dict)

    def human_line(self) -> str:
        t = datetime.fromtimestamp(self.ts, tz=timezone.utc).strftime("%H:%M")
        # ordená por prioridad y remové duplicados preservando orden
        seen = set()
        ordered = [c for c in PRIORITY if c in self.codes and (c not in seen and not seen.add(c))]
        if not ordered:
            ordered = ["no_signal"]
        parts = [MSG[c] for c in ordered]
        return f"• {t} — {self.symbol}: " + " · ".join(parts)

class MotivesBuffer:
    def __init__(self, maxlen: int = 200):
        self._buf = deque(maxlen=maxlen)

    def add(self, item: MotiveItem):
        self._buf.append(item)

    def last(self, n: int = 10) -> List[MotiveItem]:
        return list(self._buf)[-n:]

MOTIVES = MotivesBuffer()

def compute_codes(ctx: Dict[str, Any]) -> List[str]:
    """
    ctx esperado (poné lo que tengas disponible):
      price, anchor, step, span,
      ema200_1h, ema200_4h,
      atrp, adx, rsi4h,
      gate_ok (bool), risk_ok (bool),
      has_open (bool), cooldown (bool), freeze_90 (bool), blackout (bool),
      reasons (list[str])  # opcional, por si querés mapear textos
    """
    codes: List[str] = []
    has_open = ctx.get("has_open")
    if has_open: codes.append("pos_open")
    if ctx.get("blackout"): codes.append("blackout")
    if ctx.get("freeze_90"): codes.append("freeze_90")
    if ctx.get("cooldown"): codes.append("cooldown")

    price = ctx.get("price")
    anchor = ctx.get("anchor")
    step = ctx.get("step")
    span = ctx.get("span")
    if price is not None and anchor is not None and step is not None and span is not None:
        short_lo = anchor - span
        short_hi = anchor - step
        long_lo  = anchor + step
        long_hi  = anchor + span
        in_short = short_lo <= price <= short_hi
        in_long  = long_lo  <= price <= long_hi
        if not (in_short or in_long):
            codes.append("grid_oob")

        ema1 = ctx.get("ema200_1h")
        ema4 = ctx.get("ema200_4h")
        if ema1 is not None and ema4 is not None:
            long_allowed  = (price > ema1) and (price > ema4)
            short_allowed = (price < ema1) and (price < ema4)
            if in_long and not long_allowed:
                codes.append("trend_block_long")
            if in_short and not short_allowed:
                codes.append("trend_block_short")

    if ctx.get("gate_ok") is False:
        codes.append("gate_fail")
    if ctx.get("risk_ok") is False:
        codes.append("risk_block")

    # Heurísticas de volatilidad/tendencia si pasaste los valores
    atrp = ctx.get("atrp")
    if atrp is not None and atrp < 0.006:  # 0.6% como umbral típico
        codes.append("low_vol")
    adx = ctx.get("adx")
    adx_thr = ctx.get("adx_thr", 25.0)
    if adx is not None and adx < adx_thr:
        codes.append("adx_low")

    # Mapear desde textos si te llegan
    for r in (ctx.get("reasons") or []):
        s = str(r).lower()
        if "rsi" in s and "fuera" in s: codes.append("rsi_filter")
        if "funding" in s and ("malo" in s or "desfavorable" in s): codes.append("funding_bad")
        if "min" in s and ("notional" in s or "tamaño" in s): codes.append("min_notional")
        if "exchange" in s or "api" in s or "binance" in s: codes.append("exchange_error")
        if "fuera de rango" in s or "grid" in s: codes.append("grid_oob")
        if "sin señal" in s: codes.append("no_signal")

    # compactá y preservá orden de aparición
    out, seen = [], set()
    for c in codes:
        if c not in seen:
            out.append(c); seen.add(c)
    return out
