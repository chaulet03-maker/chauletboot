# bot/risk/trailing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------

@dataclass
class TrailingParams:
    mode: str = "atr"                 # 'atr' | 'ema' | 'percent'
    atr_k: float = 1.5                # múltiplos de ATR
    ema_key: str = "ema_fast"         # clave en ind_row para modo ema
    ema_k: float = 1.0                # offset en ATR alrededor de la EMA
    percent: float = 0.6              # % del precio (ej: 0.6 = 0.6%) para modo percent
    min_step_atr: float = 0.5         # pasos mínimos en múltiplos de ATR
    min_step_pct: float = 0.05        # pasos mínimos en % del precio (0.05 = 0.05%)
    hysteresis_pct: float = 0.05      # no mover SL si mejora < este %
    hard_stop_to_entry: bool = False  # al mover, no permitir SL < entry (long) o > entry (short) si ya TP1
    break_even_on_tp1: bool = True    # mover a break-even cuando TP1 se ejecutó

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _is_long(side: str) -> bool:
    return (side or "").lower() == "long"

def _pct_step_ok(side: str, proposed: float, last_sl: float, price: float, min_step_pct: float) -> bool:
    if min_step_pct <= 0:
        return True
    step = price * (min_step_pct / 100.0)
    if _is_long(side):
        return (proposed - last_sl) >= step
    else:
        return (last_sl - proposed) >= step

def _respect_monotonicity(side: str, proposed: float, last_sl: float) -> float:
    # Nunca empeorar el SL
    if _is_long(side):
        return max(proposed, last_sl)
    else:
        return min(proposed, last_sl)

def _respect_min_step_atr(side: str, proposed: float, last_sl: float, atr: float, min_step_atr: float) -> float:
    if atr <= 0 or min_step_atr <= 0:
        return proposed
    step = min_step_atr * atr
    if _is_long(side):
        return proposed if (proposed - last_sl) >= step else last_sl
    else:
        return proposed if (last_sl - proposed) >= step else last_sl

def _respect_entry_after_tp1(side: str, proposed: float, entry: float, tp1_hit: bool, hard_stop_to_entry: bool) -> float:
    if not hard_stop_to_entry or not tp1_hit:
        return proposed
    if _is_long(side):
        return max(proposed, entry)
    else:
        return min(proposed, entry)

def _hysteresis_ok(side: str, proposed: float, last_sl: float, hysteresis_pct: float) -> bool:
    """Evita micro-ajustes: exige mejora mínima relativa del SL."""
    if hysteresis_pct <= 0:
        return True
    if last_sl == 0:
        return True
    improve = (proposed - last_sl) / abs(last_sl) if _is_long(side) else (last_sl - proposed) / abs(last_sl)
    return improve >= (hysteresis_pct / 100.0)

# ---------------------------------------------------------------------
# Cálculo de trailing por modo
# ---------------------------------------------------------------------

def _proposed_sl_by_mode(
    side: str,
    price: float,
    ind_row: Dict[str, float],
    p: TrailingParams
) -> float:
    atr = float(ind_row.get("atr", 0.0) or 0.0)
    mode = (p.mode or "atr").lower()

    if mode == "percent":
        pct = float(p.percent) / 100.0
        if _is_long(side):
            return price * (1 - pct)
        else:
            return price * (1 + pct)

    if mode == "ema":
        ema = float(ind_row.get(p.ema_key, 0.0) or 0.0)
        if ema <= 0:
            # sin EMA válida, fallback a ATR si existe
            mode = "atr"

        else:
            if atr > 0 and p.ema_k != 0:
                if _is_long(side):
                    return ema - p.ema_k * atr
                else:
                    return ema + p.ema_k * atr
            else:
                return ema  # sin ATR, pegado a EMA

    # default / ATR
    k = float(p.atr_k or 0.0)
    if atr <= 0 or k <= 0:
        # sin ATR o k inválido: SL a un % conservador de emergencia
        pct = 0.8 / 100.0  # 0.8%
        return price * (1 - pct) if _is_long(side) else price * (1 + pct)

    return (price - k * atr) if _is_long(side) else (price + k * atr)

# ---------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------

def compute_trailing_stop(
    side: str,
    price: float,
    last_sl: float,
    anchor: float,
    ind_row: Dict[str, float],
    params: Dict
) -> float:
    """
    Compatibilidad con tu firma previa: devuelve nuevo SL propuesto,
    garantizando que no empeora el SL anterior y respetando paso mínimo.
    """
    p = TrailingParams(**params) if not isinstance(params, TrailingParams) else params
    atr = float(ind_row.get("atr", 0.0) or 0.0)

    proposed = _proposed_sl_by_mode(side, price, ind_row, p)
    # monotonicidad
    proposed = _respect_monotonicity(side, proposed, last_sl)
    # paso mínimo por ATR
    proposed = _respect_min_step_atr(side, proposed, last_sl, atr, p.min_step_atr)
    # paso mínimo por % (con fallback si ATR es muy chico)
    if not _pct_step_ok(side, proposed, last_sl, price, p.min_step_pct):
        return last_sl
    return proposed

def apply_trailing_to_lot(
    lot: Dict,
    price: float,
    ind_row: Dict[str, float],
    params: Dict | TrailingParams,
    tp1_hit: bool = False
) -> float:
    """
    Actualiza y devuelve el nuevo SL para un 'lot' en sitio.
    lot: { side, qty, entry, sl, tp1, tp2, lev, anchor? }
    - Mantiene/actualiza la 'anchor' cuando el precio hace nuevo máximo (long) o mínimo (short).
    - Respeta hysteresis y min_step.
    - Mueve a break-even si TP1 se ejecutó y la opción está activa.
    - Si 'hard_stop_to_entry' y TP1, no deja que el SL quede peor que el entry.
    """
    p = TrailingParams(**params) if not isinstance(params, TrailingParams) else params
    side = (lot.get("side") or "long").lower()
    entry = float(lot.get("entry", price))
    last_sl = float(lot.get("sl", entry))

    # actualizar anchor
    anchor = float(lot.get("anchor", entry))
    if _is_long(side):
        if price > anchor:
            anchor = price
    else:
        if price < anchor:
            anchor = price
    lot["anchor"] = anchor

    # break-even tras TP1
    if tp1_hit and p.break_even_on_tp1:
        if _is_long(side):
            last_sl = max(last_sl, entry)
        else:
            last_sl = min(last_sl, entry)

    # propuesto
    new_sl = compute_trailing_stop(
        side=side,
        price=price,
        last_sl=last_sl,
        anchor=anchor,
        ind_row=ind_row,
        params=p,
    )

    # hysteresis (mejora mínima del SL)
    if not _hysteresis_ok(side, new_sl, last_sl, p.hysteresis_pct):
        new_sl = last_sl

    # si ya TP1, no permitir SL peor que la entrada (opcional)
    new_sl = _respect_entry_after_tp1(side, new_sl, entry, tp1_hit, p.hard_stop_to_entry)

    # guardar y devolver
    lot["sl"] = new_sl
    return new_sl

# ---------------------------------------------------------------------
# Adaptador desde config.yaml
# ---------------------------------------------------------------------

def params_from_config(cfg: dict) -> TrailingParams:
    """
    Lee config.strategy.trailing y devuelve TrailingParams.
    Campos reconocidos:
      mode, atr_k, ema_period (-> ema_key='ema_fast' o 'ema_slow'), ema_k,
      percent, hysteresis_pct, min_step_pct
    """
    tr = (cfg.get("strategy") or {}).get("trailing", {}) or {}
    # mapear ema_period a ema_key si existiera
    ema_key = "ema_fast"
    ema_period = tr.get("ema_period")
    if ema_period is not None:
        # si tu pipeline expone ambas EMAs, podés mapear por valor
        ema_key = "ema_fast" if int(ema_period) <= int((cfg.get("strategy", {}).get("layers", {}).get("ma", {}) or {}).get("fast", 20)) else "ema_slow"

    return TrailingParams(
        mode=str(tr.get("mode", "atr")).lower(),
        atr_k=float(tr.get("atr_k", 1.5)),
        ema_key=str(tr.get("ema_key", ema_key)),
        ema_k=float(tr.get("ema_k", 1.0)),
        percent=float(tr.get("percent", 0.6)),
        hysteresis_pct=float(tr.get("hysteresis_pct", 0.05)),
        min_step_pct=float(tr.get("min_step_pct", 0.05)),
        min_step_atr=float(tr.get("min_step_atr", 0.5)),
        hard_stop_to_entry=bool(tr.get("hard_stop_to_entry", False)),
        break_even_on_tp1=bool(tr.get("break_even_on_tp1", True)),
    )
