import pandas as pd
from typing import Dict
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

_REQUIRED_COLS = ("ts", "open", "high", "low", "close", "volume")

def _ensure_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _has_required_cols(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in _REQUIRED_COLS)

def compute_indicators(df: pd.DataFrame, conf: Dict) -> pd.DataFrame:
    """
    Espera columnas: ['ts','open','high','low','close','volume'] (OHLCV).
    Devuelve el mismo DF + columnas:
      ema_fast, ema_slow, macd, macd_signal, macd_hist, rsi, adx,
      bb_high, bb_low, bb_mid, bb_width (en bps), atr, vol_mean, vol_ok
    """
    if df is None or len(df) == 0 or not _has_required_cols(df):
        # devolvemos un df vacío con columnas esperadas para evitar crashes aguas arriba
        return pd.DataFrame(columns=list(_REQUIRED_COLS) + [
            "ema_fast","ema_slow","macd","macd_signal","macd_hist","rsi","adx",
            "bb_high","bb_low","bb_mid","bb_width","atr","vol_mean","vol_ok"
        ])

    df = df.copy()

    # Ordenar por timestamp si hiciera falta
    if not df["ts"].is_monotonic_increasing:
        df = df.sort_values("ts").reset_index(drop=True)

    # Asegurar numérico
    df = _ensure_numeric(df, ["open","high","low","close","volume"])

    # --- Parámetros (defaults alineados con el resto del bot) ---
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

    # --- Indicadores de tendencia ---
    df["ema_fast"] = EMAIndicator(close=df["close"], window=ema_fast_w).ema_indicator()
    df["ema_slow"] = EMAIndicator(close=df["close"], window=ema_slow_w).ema_indicator()

    macd = MACD(close=df["close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_sig)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # --- Momentum y fuerza ---
    df["rsi"] = RSIIndicator(close=df["close"], window=rsi_len).rsi()
    df["adx"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=adx_len).adx()

    # --- Volatilidad (Bandas + ATR) ---
    bb = BollingerBands(close=df["close"], window=bb_len, window_dev=bb_dev)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"]  = bb.bollinger_lband()
    df["bb_mid"]  = bb.bollinger_mavg()

    # ancho en bps (×10000) con división segura
    width = (df["bb_high"] - df["bb_low"])
    close_safe = df["close"].replace(0, pd.NA)
    df["bb_width"] = (width / close_safe * 10000.0).fillna(0.0)

    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=atr_len).average_true_range()

    # --- Volumen ---
    df["vol_mean"] = df["volume"].rolling(vol_ma_win, min_periods=1).mean()
    # Si no hay mean aún, lo consideramos False para no sobre-señalizar
    df["vol_ok"] = (df["volume"] > (vol_mult * df["vol_mean"])) & df["vol_mean"].notna()

    # Limpieza inicial de NaNs por ventanas
    df = df.dropna().reset_index(drop=True)
    return df
