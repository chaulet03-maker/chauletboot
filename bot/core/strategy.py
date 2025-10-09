from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import math
import numpy as np
import pandas as pd
from .market_regime import infer_regime  # acepta DF completo o una fila

__all__ = ["Signal", "generate_signal"]

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

# ======== Filtros del simulador (opcionales) ========

def _passes_filters(row: pd.Series, conf: Dict) -> bool:
    """
    Filtros previos a la entrada:
      - Tendencia 4h: precio vs ema200_4h
      - Gate RSI 4h: rsi4h >= gate para LONG (<= 100-gate para SHORT)
      - Confirmación 1h opcional: precio vs ema200(1h) (si no hay, usa ema_slow como fallback)
      - ATR% gate: [min, max]
      - ban_hours: descarta si ts.hour está en la lista
    """
    side = row.get("_side_pref", None)  # 'LONG'/'SHORT' sugerido por la señal
    if side not in ("LONG", "SHORT"):
        return False

    price = float(row.get("close", np.nan))
    ema200_4h = float(row.get("ema200_4h", np.nan))
    rsi4h = float(row.get("rsi4h", np.nan))
    ema200_1h = float(row.get("ema200", row.get("ema_slow", np.nan)))  # fallback si no hay ema200 1h
    atr = float(row.get("atr", np.nan))

    # Tendencia 4h (si existen columnas)
    trend_filter = str(conf.get("trend_filter", "ema200_4h")).lower()
    if trend_filter == "ema200_4h" and np.isfinite(ema200_4h) and np.isfinite(price):
        if side == "LONG" and not (price > ema200_4h):
            return False
        if side == "SHORT" and not (price < ema200_4h):
            return False

    # RSI 4h gate
    rsi4h_gate = conf.get("rsi4h_gate", None)
    if rsi4h_gate is not None and np.isfinite(rsi4h):
        rsi4h_gate = float(rsi4h_gate)
        if side == "LONG" and not (rsi4h >= rsi4h_gate):
            return False
        if side == "SHORT" and not (rsi4h <= (100.0 - rsi4h_gate)):
            return False

    # Confirmación 1h opcional
    if conf.get("ema200_1h_confirm", False) and np.isfinite(ema200_1h) and np.isfinite(price):
        if side == "LONG" and not (price > ema200_1h):
            return False
        if side == "SHORT" and not (price < ema200_1h):
            return False

    # ATR% gate
    atrp_min = conf.get("atrp_gate_min", None)
    atrp_max = conf.get("atrp_gate_max", None)
    if (atrp_min is not None) or (atrp_max is not None):
        if not (np.isfinite(atr) and np.isfinite(price) and price > 0):
            return False
        atrp = (atr / price) * 100.0
        if atrp_min is not None and atrp < float(atrp_min):
            return False
        if atrp_max is not None and atrp > float(atrp_max):
            return False

    # ban_hours
    if "ts" in row.index and pd.notna(row["ts"]) and conf.get("ban_hours"):
        try:
            ban = {int(h) for h in str(conf["ban_hours"]).split(",") if str(h).strip() != ""}
            if pd.to_datetime(row["ts"], utc=True).hour in ban:
                return False
        except Exception:
            pass

    return True

def _signal_rsi_cross(row: pd.Series, prev: Optional[pd.Series], conf: Dict) -> Optional[str]:
    gate = float(conf.get("rsi_gate", 55.0))
    r = float(row.get("rsi", np.nan))
    pr = float(prev.get("rsi", r)) if prev is not None else r
    if not (np.isfinite(r) and np.isfinite(pr)):
        return None
    # LONG si cruza hacia arriba el gate; SHORT si cruza hacia abajo 100-gate
    if pr < gate <= r:
        return "LONG"
    if pr > (100.0 - gate) >= r:
        return "SHORT"
    return None

def _decide_grid_side(row: pd.Series, conf: Dict) -> Optional[str]:
    price = float(row.get("close", np.nan))
    ema200_4h = float(row.get("ema200_4h", np.nan))
    rsi4h = float(row.get("rsi4h", np.nan))
    gate = float(conf.get("rsi4h_gate", 52.0))
    if not all(np.isfinite(x) for x in (price, ema200_4h, rsi4h)):
        return None
    if (price > ema200_4h) and (rsi4h >= gate):
        return "LONG"
    if (price < ema200_4h) and (rsi4h <= (100.0 - gate)):
        return "SHORT"
    return None

def _anchor_price(row: pd.Series, conf: Dict) -> Optional[float]:
    """
    Ancla del grid. Por defecto usamos ema_fast como aprox de ema30 (si no hay ema30 en indicadores).
    También podés elegir 'ema200_4h' como ancla.
    """
    anchor = str(conf.get("grid_anchor", "ema_fast")).lower()
    if anchor == "ema30" or anchor == "ema_fast":
        v = row.get("ema_fast", np.nan)
    elif anchor == "ema200_4h":
        v = row.get("ema200_4h", np.nan)
    else:
        v = row.get(anchor, np.nan)
    return float(v) if np.isfinite(v) else None

def _signal_pullback_grid(row: pd.Series, conf: Dict) -> Optional[str]:
    atr = float(row.get("atr", np.nan))
    if not (np.isfinite(atr) and atr > 0):
        return None

    grid_side = str(conf.get("grid_side", "auto")).lower()
    if grid_side in ("long", "short"):
        side_pref = "LONG" if grid_side == "long" else "SHORT"
    else:
        side_pref = _decide_grid_side(row, conf)
        if side_pref is None:
            return None

    anchor = _anchor_price(row, conf)
    if anchor is None:
        return None

    price = float(row.get("close", np.nan))
    if not np.isfinite(price):
        return None

    step = float(conf.get("grid_step_atr", 0.6)) * atr
    half_span = float(conf.get("grid_span_atr", 2.5)) * atr

    if side_pref == "LONG":
        if (price < anchor) and (anchor - price >= step) and (anchor - price <= half_span):
            row["_side_pref"] = "LONG"
            return "LONG"
    else:
        if (price > anchor) and (price - anchor >= step) and (price - anchor <= half_span):
            row["_side_pref"] = "SHORT"
            return "SHORT"
    return None

def _signal_strength(row: pd.Series) -> float:
    """
    Fuerza de señal s en [0,1], combinando:
      - Desviación RSI 4h de 50
      - ADX 1h
      - Distancia precio-ema200_4h normalizada por ATR
    """
    rsi4h = float(row.get("rsi4h", np.nan))
    adx = float(row.get("adx", np.nan))
    price = float(row.get("close", np.nan))
    ema200_4h = float(row.get("ema200_4h", np.nan))
    atr = float(row.get("atr", np.nan))
    parts = []
    if np.isfinite(rsi4h):
        parts.append(np.clip(abs(rsi4h - 50.0) / 50.0, 0, 1))
    if np.isfinite(adx):
        parts.append(np.clip((adx - 10.0) / 30.0, 0, 1))
    if all(np.isfinite(x) for x in (price, ema200_4h, atr)) and atr > 0:
        parts.append(np.clip(abs(price - ema200_4h) / (3.0 * atr), 0, 1))
    return float(np.clip(np.mean(parts) if parts else 0.5, 0, 1))

# ======== Señal principal ========

def generate_signal(df: pd.DataFrame, conf: Dict) -> Signal:
    """
    Señal con capas.
    - Modo por defecto (legacy): tu régimen + momentum + volatilidad.
    - Modos del simulador (opcionales via conf['entry_mode']): 'rsi_cross' | 'pullback_grid'
      con filtros de tendencia/volatilidad estilo backtester.

    Requiere (según modo):
      Legacy: close, ema_fast, ema_slow, rsi, atr, bb_low, bb_high (y usa macd_hist si está)
      rsi_cross / pullback_grid: además puede usar rsi4h, ema200_4h, ema200 (o ema_slow fallback)
    """
    if df is None or len(df) < 50 or not _has_required_cols(df):
        return Signal("flat", 0.0, 0.0, 0.0, 0.0, "UNKNOWN")

    entry_mode = str(conf.get("entry_mode", "legacy")).lower()

    # ======== MODO LEGACY ========
    if entry_mode == "legacy":
        try:
            regime = infer_regime(df)
        except Exception:
            regime = "UNKNOWN"

        last = df.iloc[-1]
        c        = float(last.get("close", 0.0))
        ema_fast = float(last.get("ema_fast", c))
        ema_slow = float(last.get("ema_slow", c))
        rsi      = float(last.get("rsi", 50.0))
        atr      = float(last.get("atr", 0.0))
        bb_low   = float(last.get("bb_low", c))
        bb_high  = float(last.get("bb_high", c))
        macd_h   = float(last.get("macd_hist", 0.0))

        if any(_is_bad_number(x) for x in (c, ema_fast, ema_slow, rsi, atr, bb_low, bb_high)):
            return Signal("flat", 0.0, 0.0, 0.0, 0.0, regime)

        rsi_long  = float(conf.get("rsi_long", 52.0))
        rsi_short = float(conf.get("rsi_short", 48.0))
        rsi_low   = float(conf.get("rsi_low", 35.0))
        rsi_high  = float(conf.get("rsi_high", 65.0))

        side = "flat"
        confidence = 0.0

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
            sep = abs(ema_fast - ema_slow)
            confidence = _clip(sep / max(atr, 1e-8), 0.0, 1.0) * (0.8 if side != "flat" else 0.0)

        else:
            width = max(bb_high - bb_low, 1e-8)
            near_low  = c <= (bb_low  + 0.15 * width)
            near_high = c >= (bb_high - 0.15 * width)

            if near_low and rsi <= rsi_low:
                side = "long"; confidence = 0.55
            elif near_high and rsi >= rsi_high:
                side = "short"; confidence = 0.55

        if atr <= 0 or side == "flat":
            return Signal("flat", 0.0, 0.0, 0.0, 0.0, regime)

        # === SL/TP estilo simulador ===
        use_atr     = bool(conf.get("use_atr", False))
        sl_pct      = conf.get("sl_pct", 0.007)          # default sim
        tp_pct      = conf.get("tp_pct", 0.015)          # default sim
        sl_atr_mult = conf.get("sl_atr_mult", None)
        tp_atr_mult = conf.get("tp_atr_mult", None)

        sl = tp1 = tp2 = None
        if use_atr and sl_atr_mult is not None and tp_atr_mult is not None and atr > 0:
            move_sl = float(sl_atr_mult) * atr
            move_tp = float(tp_atr_mult) * atr
            if side == "long":
                sl  = c - move_sl
                tp2 = c + move_tp
            else:
                sl  = c + move_sl
                tp2 = c - move_tp
        else:
            # Porcentual (por defecto del sim)
            sl_pct = float(sl_pct)
            tp_pct = float(tp_pct)
            if side == "long":
                sl  = c * (1.0 - sl_pct)
                tp2 = c * (1.0 + tp_pct)
            else:
                sl  = c * (1.0 + sl_pct)
                tp2 = c * (1.0 - tp_pct)
        tp1 = 0.0  # TP único: el engine usa tp2

        confidence = _clip(confidence, 0.0, 1.0)
        return Signal(side, confidence, sl, tp1, tp2, regime)

    # ======== MODOS DEL SIMULADOR ========
    try:
        regime = infer_regime(df)
    except Exception:
        regime = "UNKNOWN"

    row = df.iloc[-1].copy()
    prev = df.iloc[-2] if len(df) >= 2 else None

    c   = float(row.get("close", 0.0))
    atr = float(row.get("atr", 0.0))
    if any(_is_bad_number(x) for x in (c, atr)) or atr <= 0:
        return Signal("flat", 0.0, 0.0, 0.0, 0.0, regime)

    side_sim = None
    mode = entry_mode
    if mode == "rsi_cross":
        side_sim = _signal_rsi_cross(row, prev, conf)
        if side_sim:
            row["_side_pref"] = side_sim  # para filtros
    elif mode == "pullback_grid":
        side_sim = _signal_pullback_grid(row, conf)
    else:
        return Signal("flat", 0.0, 0.0, 0.0, 0.0, regime)

    # Restricciones direccionales como en el simulador
    if bool(conf.get("only_longs", False)) and side_sim == "SHORT":
        side_sim = None
    if bool(conf.get("only_shorts", False)) and side_sim == "LONG":
        side_sim = None

    if side_sim is None:
        return Signal("flat", 0.0, 0.0, 0.0, 0.0, regime)

    # Filtros previos (tendencia/volatilidad/sesiones)
    if not _passes_filters(row, conf):
        return Signal("flat", 0.0, 0.0, 0.0, 0.0, regime)

    # === SL/TP estilo simulador ===
    use_atr     = bool(conf.get("use_atr", False))
    sl_pct      = conf.get("sl_pct", 0.007)          # default sim
    tp_pct      = conf.get("tp_pct", 0.015)          # default sim
    sl_atr_mult = conf.get("sl_atr_mult", None)
    tp_atr_mult = conf.get("tp_atr_mult", None)

    sl = tp1 = tp2 = None
    if use_atr and sl_atr_mult is not None and tp_atr_mult is not None and atr > 0:
        move_sl = float(sl_atr_mult) * atr
        move_tp = float(tp_atr_mult) * atr
        if side_sim == "LONG":
            sl  = c - move_sl
            tp2 = c + move_tp
        else:
            sl  = c + move_sl
            tp2 = c - move_tp
    else:
        # Porcentual (por defecto del sim)
        sl_pct = float(sl_pct)
        tp_pct = float(tp_pct)
        if side_sim == "LONG":
            sl  = c * (1.0 - sl_pct)
            tp2 = c * (1.0 + tp_pct)
        else:
            sl  = c * (1.0 + sl_pct)
            tp2 = c * (1.0 - tp_pct)
    tp1 = 0.0  # TP único: el engine usa tp2
    
    side_txt = "long" if side_sim == "LONG" else "short"

    # Confianza basada en fuerza de señal (0..1)
    confidence = _signal_strength(row)
    confidence = _clip(confidence, 0.0, 1.0)

    return Signal(side_txt, confidence, sl, tp1, tp2, regime)