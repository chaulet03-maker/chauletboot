
from typing import Dict

def _max(a, b): return a if a > b else b
def _min(a, b): return a if a < b else b

def compute_trailing_stop(side: str,
                          price: float,
                          last_sl: float,
                          anchor: float,
                          ind_row: Dict[str, float],
                          params: Dict) -> float:
    """Devuelve nuevo SL propuesto (no empeora el SL anterior).
    side: 'long' | 'short'
    price: precio actual
    last_sl: stop loss actual
    anchor: último 'mejor precio' alcanzado a favor (para trailing)
    ind_row: fila de indicadores (atr, ema_fast, ema_slow, etc.)
    params:
      mode: 'atr' | 'ema' | 'percent'
      atr_k: float
      ema_len: str key in ind_row (ej. 'ema_fast' o 'ema_slow')
      percent: 0..100 (porcentaje del precio)
      min_step_atr: paso mínimo en múltiplos de ATR para mover el SL
      hard_stop_to_entry: si True, no deja el SL por debajo/encima del precio de entrada cuando ya se tomó TP1
    """
    mode = str(params.get('mode', 'atr')).lower()
    atr = float(ind_row.get('atr', 0.0) or 0.0)
    min_step_atr = float(params.get('min_step_atr', 0.5))
    # límite de movimiento mínimo del stop
    def _respect_min_step(proposed, last_sl):
        if atr <= 0:
            return proposed
        step = min_step_atr * atr
        if side == 'long':
            return proposed if proposed > last_sl + step else last_sl
        else:
            return proposed if proposed < last_sl - step else last_sl

    if side not in ('long','short'):
        return last_sl

    if mode == 'percent':
        pct = float(params.get('percent', 0.6)) / 100.0
        if side == 'long':
            proposed = price * (1 - pct)
        else:
            proposed = price * (1 + pct)

    elif mode == 'ema':
        ema_key = str(params.get('ema_key', 'ema_fast'))
        ema = float(ind_row.get(ema_key, 0.0) or 0.0)
        if ema <= 0:
            return last_sl
        k = float(params.get('ema_k', 1.0))
        if side == 'long':
            proposed = ema - k * atr if atr > 0 else ema
        else:
            proposed = ema + k * atr if atr > 0 else ema

    else:  # 'atr' (default)
        k = float(params.get('atr_k', 2.0))
        if side == 'long':
            proposed = price - k * atr if atr > 0 else last_sl
        else:
            proposed = price + k * atr if atr > 0 else last_sl

    # Histeresis: nunca empeorar el SL
    if side == 'long':
        proposed = _max(proposed, last_sl)
    else:
        proposed = _min(proposed, last_sl)

    # Aplicar paso mínimo
    proposed = _respect_min_step(proposed, last_sl)

    return proposed
