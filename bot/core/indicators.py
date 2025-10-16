import pandas as pd
from typing import Dict, Optional, Any, Iterable
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

__all__ = [
    "compute_indicators",
]

_REQUIRED_COLS = ("ts", "open", "high", "low", "close", "volume")


def _has_required_cols(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in _REQUIRED_COLS)


def _ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Convierte a numérico (coerce) sin mutar el df original."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_ts(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """
    Asegura que ts sea datetime UTC, ordenado asc y sin duplicados.
    Si hay duplicados en ts, se conserva la **última** fila.
    """
    out = df.copy()
    if ts_col not in out.columns:
        return out

    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    # Quitar filas sin ts válido
    out = out.dropna(subset=[ts_col])
    # Ordenar por ts
    out = out.sort_values(ts_col, kind="mergesort")
    # Quitar duplicados en ts (mantener la última lectura)
    out = out.drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    return out


def _empty_like() -> pd.DataFrame:
    """DF vacío con todas las columnas esperadas por aguas arriba."""
    return pd.DataFrame(
        columns=list(_REQUIRED_COLS)
        + [
            "ema_fast", "ema_slow",
            "macd", "macd_signal", "macd_hist",
            "rsi", "adx",
            "bb_high", "bb_low", "bb_mid", "bb_width",
            "atr",
            "vol_mean", "vol_ok",
            "ema200_4h", "rsi4h",   # quedan vacías si no se provee df_4h
        ]
    )


def _ensure_ohlc_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas estándar OHLCV y ts en UTC ascendente sin duplicados.
    """
    req = {"ts", "open", "high", "low", "close", "volume"}
    if not req.issubset(df.columns):
        raise ValueError(f"Faltan columnas OHLCV: {req - set(df.columns)}")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)
    out = (
        out.dropna(subset=["ts"])
           .sort_values("ts")
           .drop_duplicates("ts", keep="last")
           .reset_index(drop=True)
    )
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def compute_indicators(
    df: pd.DataFrame,
    conf: Optional[Dict[str, Any]] = None,
    df_4h: Optional[pd.DataFrame] = None,  # <— NUEVO (opcional). Si no viene, se comporta igual que antes.
) -> pd.DataFrame:
    """
    Calcula indicadores sobre OHLCV (1h) y, si se provee df_4h, agrega ema200_4h y rsi4h reindexeados a 1h (ffill).

    Espera columnas mínimas:
        ['ts','open','high','low','close','volume']

    Devuelve el mismo DF + columnas:
        - ema_fast, ema_slow
        - macd, macd_signal, macd_hist
        - rsi, adx
        - bb_high, bb_low, bb_mid, bb_width (en bps = ×10000)
        - atr
        - vol_mean, vol_ok
        - (opcional) ema200_4h, rsi4h  [si df_4h no es None]

    Notas de robustez:
        - Convierte 'ts' a datetime UTC, ordena y deduplica por ts.
        - Convierte precios/volumen a numérico (coerce).
        - Controla divisiones por cero o NaNs.
        - Soporta conf=None (usa defaults seguros).
    """
    if df is None or len(df) == 0 or not _has_required_cols(df):
        return _empty_like()

    conf = conf or {}

    # Copia y normalización de ts/números
    df = _normalize_ts(df, ts_col="ts")
    if len(df) == 0:
        return _empty_like()

    df = _ensure_numeric(df, ["open", "high", "low", "close", "volume"])

    # --- Parámetros (defaults) ---
    ema_fast_w = int(conf.get("ema_fast", 20))
    ema_slow_w = int(conf.get("ema_slow", 50))
    macd_fast  = int(conf.get("macd_fast", 12))
    macd_slow  = int(conf.get("macd_slow", 26))
    macd_sig   = int(conf.get("macd_signal", 9))
    rsi_len    = int(conf.get("rsi_len", 14))
    adx_len    = int(conf.get("adx_len", 14))
    bb_len     = int(conf.get("bb_len", 20))
    bb_dev     = float(conf.get("bb_dev", 2.0))
    atr_len    = int(conf.get("atr_len", 14))
    vol_mult   = float(conf.get("vol_multiplier_vs_mean", 1.2))
    vol_ma_win = int(conf.get("vol_ma_window", 20))

    # Validaciones suaves de ventana (evita ventanas < 1)
    def _w(n: int) -> int:
        return max(int(n), 1)

    ema_fast_w = _w(ema_fast_w)
    ema_slow_w = _w(ema_slow_w)
    macd_fast  = _w(macd_fast)
    macd_slow  = _w(macd_slow)
    macd_sig   = _w(macd_sig)
    rsi_len    = _w(rsi_len)
    adx_len    = _w(adx_len)
    bb_len     = _w(bb_len)
    atr_len    = _w(atr_len)
    vol_ma_win = _w(vol_ma_win)

    out = df.copy()

    # --- Tendencia: EMA ---
    out["ema_fast"] = EMAIndicator(close=out["close"], window=ema_fast_w).ema_indicator()
    out["ema_slow"] = EMAIndicator(close=out["close"], window=ema_slow_w).ema_indicator()

    # --- MACD ---
    macd = MACD(close=out["close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_sig)
    out["macd"]        = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"]   = macd.macd_diff()

    # --- Momentum / fuerza ---
    out["rsi"] = RSIIndicator(close=out["close"], window=rsi_len).rsi()
    out["adx"] = ADXIndicator(high=out["high"], low=out["low"], close=out["close"], window=adx_len).adx()

    # --- Volatilidad: Bandas de Bollinger + ATR ---
    bb = BollingerBands(close=out["close"], window=bb_len, window_dev=bb_dev)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"]  = bb.bollinger_lband()
    out["bb_mid"]  = bb.bollinger_mavg()

    # ancho en bps (×10000); evitar divisiones raras
    close_safe = out["close"].replace(0, pd.NA)
    width = (out["bb_high"] - out["bb_low"])
    out["bb_width"] = (width / close_safe * 10000.0).astype("float64")

    # ATR
    out["atr"] = AverageTrueRange(high=out["high"], low=out["low"], close=out["close"], window=atr_len).average_true_range()

    # --- Volumen y filtro de volumen ---
    out["vol_mean"] = out["volume"].rolling(vol_ma_win, min_periods=1).mean()
    out["vol_ok"] = (out["volume"] > (vol_mult * out["vol_mean"])) & out["vol_mean"].notna()

    # === Enriquecer con 4h si se provee ===
    if df_4h is not None and len(df_4h):
        d4 = _ensure_ohlc_schema(df_4h)
        ema200_w_4h = int(conf.get("ema200_4h_window", 200))
        rsi_len_4h  = int(conf.get("rsi4h_len", 14))
        d4["ema200_4h"] = EMAIndicator(close=d4["close"], window=_w(ema200_w_4h)).ema_indicator()
        d4["rsi4h"]     = RSIIndicator(close=d4["close"], window=_w(rsi_len_4h)).rsi()
        d4 = d4.dropna(subset=["ema200_4h", "rsi4h"]).set_index("ts")[["ema200_4h", "rsi4h"]]

        out = out.set_index("ts")
        out[["ema200_4h", "rsi4h"]] = d4.reindex(out.index, method="ffill")
        out = out.reset_index()

    # Limpieza: sacá filas donde aún falten datos por las ventanas de indicadores
    out = out.dropna().reset_index(drop=True)

    # Orden final de columnas (por prolijidad)
    ordered_cols = list(_REQUIRED_COLS) + [
        "ema_fast", "ema_slow",
        "macd", "macd_signal", "macd_hist",
        "rsi", "adx",
        "bb_high", "bb_low", "bb_mid", "bb_width",
        "atr",
        "vol_mean", "vol_ok",
        "ema200_4h", "rsi4h",  # solo estarán si se pasó df_4h
    ]
    # Mantener ts y OHLCV originales si trajeron columnas extra
    return out[[c for c in ordered_cols if c in out.columns] + [c for c in out.columns if c not in ordered_cols]]
