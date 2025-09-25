
from dataclasses import dataclass
import pandas as pd
from .market_regime import infer_regime

@dataclass
class Signal:
    side: str   # 'long' | 'short' | 'flat'
    conf: float
    sl: float
    tp1: float
    tp2: float
    regime: str

def _clip(v, lo, hi): return max(lo, min(hi, v))

def generate_signal(df: pd.DataFrame, conf: dict) -> Signal:
    """
    Señal con capas (régimen + momentum + volatilidad).
    Requiere columnas: close, ema_fast, ema_slow, rsi, adx, bb_width, bb_low, bb_high, atr.
    """
    last = df.iloc[-1]
    regime = infer_regime(last)

    c = float(last['close'])
    ema_fast = float(last['ema_fast'])
    ema_slow = float(last['ema_slow'])
    rsi = float(last.get('rsi', 50))
    atr = float(last.get('atr', 0.0))
    bb_low = float(last.get('bb_low', c))
    bb_high = float(last.get('bb_high', c))

    # thresholds (conservadores; tunear por config si hace falta)
    rsi_long = float(conf.get('rsi_long', 52.0))
    rsi_short = float(conf.get('rsi_short', 48.0))
    rsi_low = float(conf.get('rsi_low', 35.0))
    rsi_high = float(conf.get('rsi_high', 65.0))

    stop_mult = float(conf.get('stop_mult', 1.5))
    tp1_R = float(conf.get('tp1_r', 1.0))
    tp2_R = float(conf.get('tp2_r', 2.4))

    side = 'flat'
    confidence = 0.0

    if regime in ('trend_up', 'trend_down'):
        if regime == 'trend_up':
            cond_dir = ema_fast > ema_slow and rsi >= rsi_long and c >= ema_fast
            cond_dir = cond_dir and float(last.get('macd_hist', 0.0)) > 0
            if cond_dir:
                side = 'long'; confidence = 0.7
        else:
            cond_dir = ema_fast < ema_slow and rsi <= rsi_short and c <= ema_fast
            cond_dir = cond_dir and float(last.get('macd_hist', 0.0)) < 0
            if cond_dir:
                side = 'short'; confidence = 0.7
    else:
        near_low = c <= (bb_low + 0.15 * (bb_high - bb_low))
        near_high = c >= (bb_high - 0.15 * (bb_high - bb_low))
        if near_low and rsi <= rsi_low:
            side = 'long'; confidence = 0.55
        elif near_high and rsi >= rsi_high:
            side = 'short'; confidence = 0.55

    # SL / TP
    if atr <= 0:
        sl = tp1 = tp2 = 0.0
    else:
        if side == 'long':
            sl = c - stop_mult * atr
            rr = c - sl
            tp1 = c + tp1_R * rr
            tp2 = c + tp2_R * rr
        elif side == 'short':
            sl = c + stop_mult * atr
            rr = sl - c
            tp1 = c - tp1_R * rr
            tp2 = c - tp2_R * rr
        else:
            sl = tp1 = tp2 = 0.0

    return Signal(side, _clip(confidence, 0.0, 1.0), sl, tp1, tp2, regime)
