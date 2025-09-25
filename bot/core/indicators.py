import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange

def compute_indicators(df: pd.DataFrame, conf: dict) -> pd.DataFrame:
    """
    df columns expected: ['ts','open','high','low','close','volume']
    """
    df = df.copy()
    ema_fast = EMAIndicator(close=df['close'], window=int(conf.get('ema_fast',21)))
    ema_slow = EMAIndicator(close=df['close'], window=int(conf.get('ema_slow',55)))
    df['ema_fast'] = ema_fast.ema_indicator()
    df['ema_slow'] = ema_slow.ema_indicator()

    macd = MACD(close=df['close'],
                window_fast=int(conf.get('macd_fast',12)),
                window_slow=int(conf.get('macd_slow',26)),
                window_sign=int(conf.get('macd_signal',9)))
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    rsi = RSIIndicator(close=df['close'], window=int(conf.get('rsi_len',14)))
    df['rsi'] = rsi.rsi()

    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'],
                       window=int(conf.get('adx_len',14)) if 'adx_len' in conf else 14)
    df['adx'] = adx.adx()

    bb = BollingerBands(close=df['close'],
                        window=int(conf.get('bb_len',20)),
                        window_dev=float(conf.get('bb_dev',2.0)))
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low']  = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close'] * 10000.0  # bps

    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'],
                           window=int(conf.get('atr_len',14)))
    df['atr'] = atr.average_true_range()

    # volumen
    df['vol_mean'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = df['volume'] > float(conf.get('vol_multiplier_vs_mean',1.2)) * df['vol_mean']

    return df.dropna().reset_index(drop=True)
