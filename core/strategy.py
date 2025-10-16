import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bot.core.strategy import (
    Signal,
    _anchor_price as core_anchor_price,
    _apply_anchor_freeze as core_apply_anchor_freeze,
    _decide_grid_side as core_decide_grid_side,
    check_all_filters,
    generate_signal,
)

logger = logging.getLogger(__name__)


class Strategy:
    """Wrapper liviano que expone la API esperada por ``bot.engine``."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = cfg or {}
        self._symbol: str = str(self.config.get("symbol") or "")
        self._last_signal: Optional[Signal] = None
        self._last_candle: Optional[pd.Series] = None

    # === API principal utilizada por ``bot.engine`` ===
    def check_entry_signal(self, data: pd.DataFrame) -> Optional[str]:
        if data is None or len(data) == 0:
            self._last_signal = None
            self._last_candle = None
            return None

        try:
            self._last_candle = data.iloc[-1].copy()
        except Exception:
            self._last_candle = None

        try:
            signal = generate_signal(data, self.config)
        except Exception as exc:
            logger.debug("generate_signal failure: %s", exc)
            self._last_signal = None
            return None

        self._last_signal = signal
        side = str(signal.side or "").upper()
        if side in ("LONG", "SHORT"):
            return side
        return None

    def calculate_sl(self, entry_price: float, last_candle: pd.Series, side: str) -> float:
        atr = self._extract_atr(last_candle)
        if atr is None or not math.isfinite(entry_price):
            return float(getattr(self._last_signal, "sl", entry_price) or entry_price)

        mult = self._get_float("stop_atr_mult", fallback=self.config.get("sl_atr_mult", 1.5))
        if not math.isfinite(mult):
            mult = 1.5

        if str(side).upper() == "LONG":
            return float(entry_price - mult * atr)
        return float(entry_price + mult * atr)

    def calculate_tp(self, entry_price: float, quantity: float, equity_on_open: float, side: str) -> float:
        if self._last_signal is not None:
            tp_candidate = getattr(self._last_signal, "tp2", None) or getattr(self._last_signal, "tp1", None)
            if tp_candidate is not None and math.isfinite(float(tp_candidate)):
                return float(tp_candidate)

        atr = self._extract_atr(self._last_candle)
        if atr is not None and math.isfinite(entry_price):
            mult = self._get_float("tp_atr_mult", fallback=2.0)
            if math.isfinite(mult):
                if str(side).upper() == "LONG":
                    return float(entry_price + mult * atr)
                return float(entry_price - mult * atr)

        try:
            target_pct = float(self.config.get("target_eq_pnl_pct", 0.10))
            pnl_target = float(equity_on_open) * target_pct
            qty = float(quantity) if quantity is not None else 0.0
            move = pnl_target / qty if qty else 0.0
            if str(side).upper() == "LONG":
                return float(entry_price + move)
            return float(entry_price - move)
        except Exception:
            return float(entry_price)

    def dynamic_leverage(self, last_candle: pd.Series) -> float:
        lev_cfg = self._get_float("leverage", fallback=None)
        if lev_cfg is not None and math.isfinite(lev_cfg):
            return float(max(1.0, min(float(lev_cfg), 20.0)))

        try:
            adx = float(last_candle.get("adx"))
        except Exception:
            adx = float("nan")

        thr = self._get_float("adx_strong_threshold", fallback=25.0) or 25.0
        weak = self._get_float("lev_weak", fallback=self.config.get("leverage_base", 5.0)) or 5.0
        strong = self._get_float("lev_strong", fallback=self.config.get("leverage_strong", 10.0)) or 10.0

        if math.isfinite(adx):
            return float(strong if adx >= thr else weak)
        return float(weak)

    def explain_signal(self, data: pd.DataFrame) -> str:
        if self._last_signal and getattr(self._last_signal, "reason", ""):
            return str(self._last_signal.reason)

        if data is not None and len(data):
            row = data.iloc[-1]
            side_pref = self._safe_side_pref(row)
            reason = check_all_filters(row, self.config, side_pref)
            if reason:
                return str(reason)
        return "Sin detalle disponible"

    def get_rejection_reason(self, data: pd.DataFrame) -> Tuple[str, str, str]:
        reasons = self.get_rejection_reasons_all(data)
        if not reasons:
            return "no_signal", "Sin señal", ""
        code, detail = reasons[0]
        extras = "; ".join(f"{c}:{d}" for c, d in reasons[1:]) if len(reasons) > 1 else ""
        return code, detail, extras

    def get_rejection_reasons_all(self, data: pd.DataFrame) -> List[Tuple[str, str]]:
        if data is None or len(data) == 0:
            return [("no_signal", "Sin datos")]

        row = data.iloc[-1]
        reasons: List[Tuple[str, str]] = []

        side_pref = self._safe_side_pref(row)
        if side_pref is None:
            reasons.append(("no_side", "Sin dirección preferida"))
        else:
            reason = check_all_filters(row, self.config, side_pref)
            if reason:
                reasons.append(("filters", str(reason)))

        if self._last_signal is not None:
            side = str(self._last_signal.side or "").upper()
            if side not in ("LONG", "SHORT"):
                detail = str(getattr(self._last_signal, "reason", "No se generó señal utilizable"))
                reasons.append(("no_signal", detail))
        elif not reasons:
            reasons.append(("no_signal", "No se generó señal"))

        if not reasons:
            reasons.append(("no_signal", "Sin señal utilizable"))
        return reasons

    # === Helpers expuestos para el engine ===
    def _apply_anchor_freeze(self, side: Optional[str], price: float, anchor: float, step: float, span: float):
        try:
            return core_apply_anchor_freeze(self._symbol, side, price, anchor, step, span)
        except Exception as exc:
            logger.debug("anchor freeze fallback: %s", exc)
            return anchor, step, span, "error"

    def _anchor_price(self, row: pd.Series) -> Optional[float]:
        try:
            return core_anchor_price(row, self.config)
        except Exception as exc:
            logger.debug("anchor price fallback: %s", exc)
            return None

    def _decide_grid_side(self, row: pd.Series) -> Optional[str]:
        try:
            return core_decide_grid_side(row, self.config)
        except Exception as exc:
            logger.debug("grid side fallback: %s", exc)
            return None

    # === Utilidades internas ===
    def _extract_atr(self, row: Optional[pd.Series]) -> Optional[float]:
        if row is None:
            return None
        try:
            atr = float(row.get("atr"))
        except Exception:
            return None
        return atr if math.isfinite(atr) and atr > 0 else None

    def _get_float(self, key: str, fallback: Any) -> Optional[float]:
        value = self.config.get(key, fallback)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_side_pref(self, row: pd.Series) -> Optional[str]:
        try:
            side = core_decide_grid_side(row, self.config)
            if side in ("LONG", "SHORT"):
                return side
        except Exception:
            return None
        return None
