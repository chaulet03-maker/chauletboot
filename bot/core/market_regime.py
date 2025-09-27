# bot/core/market_regime.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Optional, Union, Literal

# Regímenes estandarizados (en MAYÚSCULAS para evitar ambigüedades aguas arriba)
TREND_UP   = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE      = "RANGE"
CHOP       = "CHOP"
UNKNOWN    = "UNKNOWN"

Timeframe = Optional[Literal["1h", "4h"]]

# ---------- Defaults por timeframe ----------
_DEFAULTS_1H = dict(
    lookback=30,
    adx_chop_max=12.0,
    bb_chop_max_bps=6.0,    # 0.06%
    adx_range_max=18.0,
    bb_range_max_bps=12.0,  # 0.12%
    adx_trend_min=18.0,
    bb_trend_min_bps=8.0,   # 0.08%
)

_DEFAULTS_4H = dict(
    lookback=30,
    # 4h suele mostrar bb_width más chico: bajamos ligeramente umbrales de ancho
    adx_chop_max=12.0,
    bb_chop_max_bps=5.0,    # 0.05%
    adx_range_max=18.0,
    bb_range_max_bps=10.0,  # 0.10%
    adx_trend_min=18.0,
    bb_trend_min_bps=6.0,   # 0.06%
)

def _merge_cfg(
    base: Dict[str, float],
    override: Optional[Dict[str, float]] = None,
    tf: Timeframe = None,
    regime_cfg: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Combina defaults + cfg explícito + overrides por timeframe (si se proveen).
    Prioridad: regime_cfg[tf] > override > base.
    """
    cfg = dict(base)
    if override:
        cfg.update(override)
    if regime_cfg and tf and tf in regime_cfg:
        cfg.update(regime_cfg[tf])
    return cfg

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
    if df is None or df.empty:
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

def _infer_from_df(
    df: pd.DataFrame,
    cfg: Dict,
) -> str:
    """
    Clasifica con robustez usando la ventana reciente (medianas):
      - CHOP: ADX muy bajo o BB_WIDTH muy angosto.
      - RANGE: ancho bajo y fuerza (ADX) baja.
      - TREND_*: sesgo EMA + umbrales mínimos de ADX/BB_WIDTH.

    Espera columnas: close, adx, bb_width, ema_fast, ema_slow (estas dos últimas opcionales).
    `bb_width` esperado en bps (x10000) como lo emiten tus indicadores.
    """
    if df is None or len(df) < max(30, int(cfg.get("lookback", 30))):
        return UNKNOWN

    lb            = int(cfg["lookback"])
    adx_chop_max  = float(cfg["adx_chop_max"])
    bb_chop_max   = float(cfg["bb_chop_max_bps"])
    adx_range_max = float(cfg["adx_range_max"])
    bb_range_max  = float(cfg["bb_range_max_bps"])
    adx_trend_min = float(cfg["adx_trend_min"])
    bb_trend_min  = float(cfg["bb_trend_min_bps"])

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

def _infer_from_row(row: pd.Series, cfg: Dict) -> str:
    """
    Compatibilidad si te pasan una sola fila (última vela).
    Usa los mismos thresholds que el modo DataFrame pero evaluando un punto.
    """
    adx_chop_max  = float(cfg["adx_chop_max"])
    bb_chop_max   = float(cfg["bb_chop_max_bps"])
    adx_range_max = float(cfg["adx_range_max"])
    bb_range_max  = float(cfg["bb_range_max_bps"])
    adx_trend_min = float(cfg["adx_trend_min"])
    bb_trend_min  = float(cfg["bb_trend_min_bps"])

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

def infer_regime(
    obj: Union[pd.DataFrame, pd.Series],
    cfg: Optional[Dict] = None,
    tf: Timeframe = None,
    regime_cfg: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """
    API unificada:
      - Si recibe DataFrame -> usa ventana (recomendado).
      - Si recibe Series/fila -> usa la lógica de una vela (compatibilidad).

    Parámetros:
      cfg: overrides globales de thresholds.
      tf: "1h" | "4h" para elegir defaults por timeframe (si no pasás regime_cfg).
      regime_cfg: mapa opcional por timeframe (e.g., {"1h": {...}, "4h": {...}}).
                  Si existe y coincide con `tf`, tiene prioridad.
    """
    # Elegir base por timeframe
    if tf == "4h":
        base = _DEFAULTS_4H
    else:
        # default 1h si no se especifica tf
        base = _DEFAULTS_1H

    merged_cfg = _merge_cfg(base=base, override=cfg, tf=tf, regime_cfg=regime_cfg)

    if isinstance(obj, pd.DataFrame):
        return _infer_from_df(obj, merged_cfg)
    return _infer_from_row(obj, merged_cfg)
