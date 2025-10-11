import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange


def add_indicators(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores necesarios y los combina en un solo DataFrame.
    """
    # Indicadores de 1 hora
    c, h, l = df_1h["close"], df_1h["high"], df_1h["low"]
    df_1h["ema200"] = EMAIndicator(c, window=200).ema_indicator()
    df_1h["rsi"] = RSIIndicator(c, window=14).rsi()
    df_1h["adx"] = ADXIndicator(h, l, c, window=14).adx()
    df_1h["atr"] = AverageTrueRange(h, l, c, window=14).average_true_range()

    # Indicadores de 4 horas
    df_4h["ema200_4h"] = EMAIndicator(df_4h["close"], window=200).ema_indicator()
    df_4h["rsi4h"] = RSIIndicator(df_4h["close"], window=14).rsi()

    # Unir los datos de 4h al DataFrame de 1h
    df_4h_agg = df_4h[["ema200_4h", "rsi4h"]]
    combined_df = df_1h.join(df_4h_agg.reindex(df_1h.index, method="ffill"))

    return combined_df.dropna()
