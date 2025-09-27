from dataclasses import dataclass
from typing import Dict, Iterable
import math
import pandas as pd
from .market_regime import infer_regime  # acepta DF completo o una fila

@dataclass
class Signal:
    side: str   # 'long' | 'short' | 'flat'
    conf: float
    sl: float
    tp1: float
    tp2: float
    regime: str

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

_REQUIRED_COLS: Iterable[str] = (
    "close", "ema_fast", "ema_slow", "rsi", "atr", "bb_low", "bb_high"
)

def _has_required_cols(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in _REQUIRED_COLS)

def _is_bad_number(x) -> bool:
    try:
        return (x is None) or (not math.isfinite(float(x)))
    except Exception:
        return True

def generate_signal(df: pd.DataFrame, conf: Dict) -> Signal:
    """
    Señal con capas (régimen + momentum + volatilidad).
    Requiere columnas: close, ema_fast, ema_slow, rsi, atr, bb_low, bb_high.
    Usa macd_hist si está presente.
    """
    # 0) Guardas básicas
    if df is None or len(df) < 50 or not _has_required_cols(df):
        return Signal("flat", 0.0, 0.0, 0.0, 0.0, "UNKNOWN")

    # 1) Régimen (usar DF completo, no solo la última fila)
    try:
        regime = infer_regime(df)
    except Exception:
        regime = "UNKNOWN"

    # 2) Tomar última vela y campos
    last = df.iloc[-1]
    c        = float(last.get("close", 0.0))
    ema_fast = float(last.get("ema_fast", c))
    ema_slow = float(last.get("ema_slow", c))
    rsi      = float(last.get("rsi", 50.0))
    atr      = float(last.get("atr", 0.0))
    bb_low   = float(last.get("bb_low", c))
    bb_high  = float(last.get("bb_high", c))
    macd_h   = float(last.get("macd_hist", 0.0))

    # Si algún número clave es inválido, no operamos
    if any(_is_bad_number(x) for x in (c, ema_fast, ema_slow, rsi, atr, bb_low, bb_high)):
        return Signal("flat", 0.0, 0.0, 0.0, 0.0, regime)

    # 3) Umbrales (overridables por config)
    rsi_long  = float(conf.get("rsi_long", 52.0))
    rsi_short = float(conf.get("rsi_short", 48.0))
    rsi_low   = float(conf.get("rsi_low", 35.0))
    rsi_high  = float(conf.get("rsi_high", 65.0))

    stop_mult = float(conf.get("stop_mult", 1.5))
    tp1_R     = float(conf.get("tp1_r", 1.0))
    tp2_R     = float(conf.get("tp2_r", 2.0))

    side = "flat"
    confidence = 0.0

    # 4) Lógica por régimen
    if regime in ("TREND_UP", "TREND_DOWN", "trend_up", "trend_down"):
        is_up = regime in ("TREND_UP", "trend_up")
        if is_up:
            cond = (ema_fast > ema_slow) and (rsi >= rsi_long) and (c >= ema_fast) and (macd_h > 0)
            if cond:
                side = "long"
        else:
            cond = (ema_fast < ema_slow) and (rsi <= rsi_short) and (c <= ema_fast) and (macd_h < 0)
            if cond:
                side = "short"
        # confianza por separación EMAs normalizada por ATR
        sep = abs(ema_fast - ema_slow)
        confidence = _clip(sep / max(atr, 1e-8), 0.0, 1.0) * (0.8 if side != "flat" else 0.0)

    else:
        # Rango: comprar cerca del borde inferior con RSI bajo; vender cerca del superior con RSI alto
        width = max(bb_high - bb_low, 1e-8)
        near_low  = c <= (bb_low  + 0.15 * width)
        near_high = c >= (bb_high - 0.15 * width)

        if near_low and rsi <= rsi_low:
            side = "long"
            confidence = 0.55
        elif near_high and rsi >= rsi_high:
            side = "short"
            confidence = 0.55

    # 5) SL / TP por R (si ATR no alcanza, no operamos)
    if atr <= 0 or side == "flat":
        return Signal("flat", 0.0, 0.0, 0.0, 0.0, regime)

    if side == "long":
        sl  = c - stop_mult * atr
        rr  = c - sl
        tp1 = c + tp1_R * rr
        tp2 = c + tp2_R * rr
    else:
        sl  = c + stop_mult * atr
        rr  = sl - c
        tp1 = c - tp1_R * rr
        tp2 = c - tp2_R * rr

    # 6) Confianza final acotada
    confidence = _clip(confidence, 0.0, 1.0)

    return Signal(side, confidence, sl, tp1, tp2, regime)
