def infer_regime(row) -> str:
    """
    Clasifica mercado: 'trend_up', 'trend_down', 'range', 'chop' según EMA y ADX/BB.
    """
    adx = row['adx']
    bb = row['bb_width']
    ema_fast = row['ema_fast']
    ema_slow = row['ema_slow']
    close = row['close']

    if adx < 12 or bb < 6:
        return "chop"
    if abs(bb) < 8 and adx < 18:
        return "range"
    if close > ema_slow and ema_fast > ema_slow:
        return "trend_up"
    if close < ema_slow and ema_fast < ema_slow:
        return "trend_down"
    return "range"
