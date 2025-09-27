# bot/core/market_regime.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Optional

# Regímenes estandarizados (en MAYÚSCULAS para evitar ambigüedades aguas arriba)
TREND_UP   = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE      = "RANGE"
CHOP       = "CHOP"
UNKNOWN    = "UNKNOWN"

def _median_last(df: pd.DataFrame, col: str, lookback: int) -> float:
    s = df[col].tail(lookback)
    return float(s.median()) if len(s) else float("nan")

def _trend_bias(df: pd.DataFrame) -> int:
    """+1 sesgo alcista, -1 bajista, 0 neutral (usa EMA fast/slow + close vs slow)."""
    last = df.iloc[-1]
    ema_fast = float(last.get("ema_fast", last["close"]))
    ema_slow = float(last.get("ema_slow", last["close"]))
    close    = float(last["close"])
    if ema_fast > ema_slow and close >= ema_slow:
        return +1
    if ema_fast < ema_slow and close <= ema_slow:
        return -1
    return 0

def _infer_from_df(df: pd.DataFrame, cfg: Optional[Dict] = None) -> str:
    """
    Clasifica con robustez usando la ventana reciente (medianas):
    - CHOP: adx muy bajo o bb_width muy angosto.
    - RANGE: ancho bajo y fuerza (adx) baja.
    - TREND_*: sesgo ema + umbrales mínimos de adx/bb_width.
    Notas:
      - bb_width se espera en bps (x10000) como sale de indicators.
      - Umbrales default pensados para 5m; podés afinarlos vía cfg.
    """
    if df is None or len(df) < 30:
        return UNKNOWN

    cfg = cfg or {}
    lb            = int(cfg.get("lookback", 30))
    adx_chop_max  = float(cfg.get("adx_chop_max", 12.0))
    bb_chop_max   = float(cfg.get("bb_chop_max_bps", 6.0))    # 6 bps = 0.06%
    adx_range_max = float(cfg.get("adx_range_max", 18.0))
    bb_range_max  = float(cfg.get("bb_range_max_bps", 12.0))  # 12 bps = 0.12%
    adx_trend_min = float(cfg.get("adx_trend_min", 18.0))
    bb_trend_min  = float(cfg.get("bb_trend_min_bps", 8.0))   # 8 bps = 0.08%

    adx_med = _median_last(df, "adx", lb)
    bbw_med = _median_last(df, "bb_width", lb)

    if adx_med < adx_chop_max or bbw_med < bb_chop_max:
        return CHOP

    if bbw_med < bb_range_max and adx_med < adx_range_max:
        return RANGE

    bias = _trend_bias(df)
    if bias > 0 and adx_med >= adx_trend_min and bbw_med >= bb_trend_min:
        return TREND_UP
    if bias < 0 and adx_med >= adx_trend_min and bbw_med >= bb_trend_min:
        return TREND_DOWN

    # Si no cumple thresholds de tendencia pero tampoco es chop claro, asumimos rango.
    return RANGE

def _infer_from_row(row: pd.Series, cfg: Optional[Dict] = None) -> str:
    """
    Compatibilidad si te pasan una sola fila (tu versión original).
    Usa thresholds idénticos a los del modo DF sobre la última vela.
    """
    cfg = cfg or {}
    adx_chop_max  = float(cfg.get("adx_chop_max", 12.0))
    bb_chop_max   = float(cfg.get("bb_chop_max_bps", 6.0))
    adx_range_max = float(cfg.get("adx_range_max", 18.0))
    bb_range_max  = float(cfg.get("bb_range_max_bps", 12.0))
    adx_trend_min = float(cfg.get("adx_trend_min", 18.0))
    bb_trend_min  = float(cfg.get("bb_trend_min_bps", 8.0))

    adx = float(row.get("adx", 0.0))
    bb  = float(row.get("bb_width", 0.0))
    ema_fast = float(row.get("ema_fast", row["close"]))
    ema_slow = float(row.get("ema_slow", row["close"]))
    close    = float(row["close"])

    if adx < adx_chop_max or bb < bb_chop_max:
        return CHOP
    if abs(bb) < bb_range_max and adx < adx_range_max:
        return RANGE
    if close > ema_slow and ema_fast > ema_slow and adx >= adx_trend_min and bb >= bb_trend_min:
        return TREND_UP
    if close < ema_slow and ema_fast < ema_slow and adx >= adx_trend_min and bb >= bb_trend_min:
        return TREND_DOWN
    return RANGE

def infer_regime(obj, cfg: Optional[Dict] = None) -> str:
    """
    API unificada:
      - Si recibe DataFrame -> usa ventana (recomendado).
      - Si recibe Series/fila -> usa la lógica de una vela (compatibilidad).
    """
    if isinstance(obj, pd.DataFrame):
        return _infer_from_df(obj, cfg)
    return _infer_from_row(obj, cfg)
