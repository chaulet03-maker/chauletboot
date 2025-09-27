# bot/core/market_regime.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Optional, Union

# Regímenes estandarizados (en MAYÚSCULAS para evitar ambigüedades aguas arriba)
TREND_UP   = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE      = "RANGE"
CHOP       = "CHOP"
UNKNOWN    = "UNKNOWN"

def _median_last(df: pd.DataFrame, col: str, lookback: int) -> float:
    """
    Mediana de la ventana reciente, ignorando NaNs.
    """
    if col not in df.columns:
        return float("nan")
    s = df[col].tail(max(1, int(lookback))).dropna()
    return float(s.median()) if len(s) else float("nan")

def _trend_bias(df: pd.DataFrame) -> int:
    """
    +1 sesgo alcista, -1 bajista, 0 neutral.
    Usa EMA fast/slow (si existen) y close vs slow como confirmación.
    """
    if df.empty:
        return 0
    last = df.iloc[-1]
    close    = float(last.get("close", float("nan")))
    ema_fast = float(last.get("ema_fast", close))
    ema_slow = float(last.get("ema_slow", close))

    if pd.isna(close) or pd.isna(ema_fast) or pd.isna(ema_slow):
        return 0

    if ema_fast > ema_slow and close >= ema_slow:
        return +1
    if ema_fast < ema_slow and close <= ema_slow:
        return -1
    return 0

def _infer_from_df(df: pd.DataFrame, cfg: Optional[Dict] = None) -> str:
    """
    Clasifica con robustez usando la ventana reciente (medianas):
      - CHOP: ADX muy bajo o BB_WIDTH muy angosto.
      - RANGE: ancho bajo y fuerza (ADX) baja.
      - TREND_*: sesgo EMA + umbrales mínimos de ADX/BB_WIDTH.

    Notas:
      - `bb_width` se espera en bps (x10000) como sale de tus `indicators`.
      - Umbrales **pensados para 1h** por defecto; afiná vía `cfg` si querés.

    cfg keys (opcionales):
      lookback: int = 30
      adx_chop_max: float = 12.0
      bb_chop_max_bps: float = 6.0
      adx_range_max: float = 18.0
      bb_range_max_bps: float = 12.0
      adx_trend_min: float = 18.0
      bb_trend_min_bps: float = 8.0
    """
    if df is None or len(df) < 30:
        return UNKNOWN

    cfg = cfg or {}
    lb            = int(cfg.get("lookback", 30))
    adx_chop_max  = float(cfg.get("adx_chop_max", 12.0))
    bb_chop_max   = float(cfg.get("bb_chop_max_bps", 6.0))     # 6 bps = 0.06%
    adx_range_max = float(cfg.get("adx_range_max", 18.0))
    bb_range_max  = float(cfg.get("bb_range_max_bps", 12.0))   # 12 bps = 0.12%
    adx_trend_min = float(cfg.get("adx_trend_min", 18.0))
    bb_trend_min  = float(cfg.get("bb_trend_min_bps", 8.0))    # 8 bps = 0.08%

    adx_med = _median_last(df, "adx", lb)
    bbw_med = _median_last(df, "bb_width", lb)

    # Si no hay datos confiables de fuerza/ancho, mejor no sobre-clasificar
    if pd.isna(adx_med) or pd.isna(bbw_med):
        return UNKNOWN

    # 1) CHOP: fuerza muy baja o rango muy angosto
    if adx_med < adx_chop_max or bbw_med < bb_chop_max:
        return CHOP

    # 2) RANGE: ambos bajos pero no tan extremos como CHOP
    if bbw_med < bb_range_max and adx_med < adx_range_max:
        return RANGE

    # 3) Tendencia si hay sesgo + thresholds mínimos
    bias = _trend_bias(df)
    if bias > 0 and adx_med >= adx_trend_min and bbw_med >= bb_trend_min:
        return TREND_UP
    if bias < 0 and adx_med >= adx_trend_min and bbw_med >= bb_trend_min:
        return TREND_DOWN

    # 4) Si no cumple thresholds de tendencia pero tampoco es CHOP claro, asumimos RANGE
    return RANGE

def _infer_from_row(row: pd.Series, cfg: Optional[Dict] = None) -> str:
    """
    Compatibilidad si te pasan una sola fila (última vela).
    Usa los mismos thresholds que el modo DataFrame pero evaluando un punto.
    """
    cfg = cfg or {}
    adx_chop_max  = float(cfg.get("adx_chop_max", 12.0))
    bb_chop_max   = float(cfg.get("bb_chop_max_bps", 6.0))
    adx_range_max = float(cfg.get("adx_range_max", 18.0))
    bb_range_max  = float(cfg.get("bb_range_max_bps", 12.0))
    adx_trend_min = float(cfg.get("adx_trend_min", 18.0))
    bb_trend_min  = float(cfg.get("bb_trend_min_bps", 8.0))

    adx = float(row.get("adx", float("nan")))
    bb  = float(row.get("bb_width", float("nan")))
    close    = float(row.get("close", float("nan")))
    ema_fast = float(row.get("ema_fast", close))
    ema_slow = float(row.get("ema_slow", close))

    if any(pd.isna(x) for x in (adx, bb, close, ema_fast, ema_slow)):
        return UNKNOWN

    if adx < adx_chop_max or bb < bb_chop_max:
        return CHOP
    if abs(bb) < bb_range_max and adx < adx_range_max:
        return RANGE
    if close > ema_slow and ema_fast > ema_slow and adx >= adx_trend_min and bb >= bb_trend_min:
        return TREND_UP
    if close < ema_slow and ema_fast < ema_slow and adx >= adx_trend_min and bb >= bb_trend_min:
        return TREND_DOWN
    return RANGE

def infer_regime(obj: Union[pd.DataFrame, pd.Series], cfg: Optional[Dict] = None) -> str:
    """
    API unificada:
      - Si recibe DataFrame -> usa ventana (recomendado).
      - Si recibe Series/fila -> usa la lógica de una vela (compatibilidad).
    """
    if isinstance(obj, pd.DataFrame):
        return _infer_from_df(obj, cfg)
    return _infer_from_row(obj, cfg)
