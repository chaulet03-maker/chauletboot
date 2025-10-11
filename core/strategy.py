import numpy as np
import pandas as pd

class Strategy:
    def __init__(self, cfg):
        self.config = cfg

    # Señal de entrada (pullback_grid) + filtros
    def check_entry_signal(self, data: pd.DataFrame) -> str | None:
        if len(data) < 2:
            return None
        row = data.iloc[-1]
        ts = data.index[-1]

        side_pref = self._decide_grid_side(row)
        if side_pref is None:
            return None
        if not self._passes_filters(row, ts, side_pref):
            return None

        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None

        anchor = self._anchor_price(row)
        if anchor is None or not np.isfinite(anchor):
            return None

        price = float(row["close"])
        step = float(self.config.get("grid_step_atr", 0.32)) * atr
        half_span = float(self.config.get("grid_span_atr", 3.0)) * atr

        if side_pref == "LONG":
            if (price < anchor) and (anchor - price >= step) and (anchor - price <= half_span):
                return "LONG"
        else:
            if (price > anchor) and (price - anchor >= step) and (price - anchor <= half_span):
                return "SHORT"
        return None

    # SL ATR x 1.3 y TP único al 10% del equity al abrir (idéntico al sim)
    def calculate_sl(self, entry_price: float, last_candle: pd.Series, side: str) -> float:
        atr = float(last_candle["atr"])
        sl_mult = float(self.config.get("sl_atr_mult", 1.3))
        return entry_price - (atr * sl_mult) if side == "LONG" else entry_price + (atr * sl_mult)

    def calculate_tp(self, entry_price: float, quantity: float, equity_on_open: float, side: str) -> float:
        # Idéntico al simulador:
        # move = (target_eq_pnl_pct * equity_on_open) / qty
        tp_pct = float(self.config.get("target_eq_pnl_pct", 0.10))
        pnl_target = equity_on_open * tp_pct
        move = pnl_target / max(quantity, 1e-12)
        return (entry_price + move) if side == "LONG" else (entry_price - move)

    # Apalancamiento dinámico por ADX (x5 base / x10 fuerte)
    def dynamic_leverage(self, last_candle: pd.Series) -> float:
        adx = float(last_candle.get("adx", np.nan))
        thr = float(self.config.get("adx_strong_threshold", 25.0))
        return float(self.config.get("leverage_strong", 10.0)) if (np.isfinite(adx) and adx >= thr) else float(self.config.get("leverage_base", 5.0))

    # Filtros auxiliares
    def _passes_filters(self, row: pd.Series, ts: pd.Timestamp, side: str) -> bool:
        if str(self.config.get("trend_filter", "ema200_4h")) == "ema200_4h":
            ema200_4h = float(row.get("ema200_4h", np.nan)); price = float(row.get("close", np.nan))
            if side == "LONG" and not (price > ema200_4h): return False
            if side == "SHORT" and not (price < ema200_4h): return False

        rsi_gate = self.config.get("rsi4h_gate", None)
        if rsi_gate is not None:
            rsi4h = float(row.get("rsi4h", np.nan)); g = float(rsi_gate)
            if side == "LONG" and not (rsi4h >= g): return False
            if side == "SHORT" and not (rsi4h <= (100.0 - g)): return False

        if bool(self.config.get("ema200_1h_confirm", False)):
            price = float(row.get("close", np.nan)); ema200_1h = float(row.get("ema200", np.nan))
            if side == "LONG" and not (price > ema200_1h): return False
            if side == "SHORT" and not (price < ema200_1h): return False

        atr = float(row.get("atr", np.nan)); close = float(row.get("close", np.nan))
        if np.isfinite(atr) and np.isfinite(close) and close > 0:
            atrp = (atr / close) * 100.0
            minp = self.config.get("atrp_gate_min", None); maxp = self.config.get("atrp_gate_max", None)
            if minp is not None and atrp < float(minp): return False
            if maxp is not None and atrp > float(maxp): return False

        ban_hours = set(self.config.get("ban_hours", []) or [])
        if len(ban_hours) and int(getattr(ts, "hour", 0)) in ban_hours: return False

        gate_bps = self.config.get("funding_gate_bps", None)
        if gate_bps is not None:
            g_frac = float(gate_bps) / 10000.0
            rate_dec = self.config.get("_funding_rate_now", None)
            rate_bps = self.config.get("_funding_rate_bps_now", None)
            if rate_dec is None and rate_bps is not None:
                try:
                    rate_dec = float(rate_bps) / 10000.0
                except Exception:
                    rate_dec = None
            if rate_dec is not None:
                r = float(rate_dec)
                if (side == "LONG" and r > +g_frac) or (side == "SHORT" and r < -g_frac):
                    return False

        return True

    def _decide_grid_side(self, row: pd.Series) -> str | None:
        side_cfg = str(self.config.get("grid_side", "auto")).lower()
        if side_cfg in ("long", "short"):
            return "LONG" if side_cfg == "long" else "SHORT"

        price = float(row.get("close", np.nan))
        ema200_4h = float(row.get("ema200_4h", np.nan))
        rsi4h = float(row.get("rsi4h", np.nan))
        gate = float(self.config.get("rsi4h_gate", 52.0))

        if not (np.isfinite(price) and np.isfinite(ema200_4h) and np.isfinite(rsi4h)):
            return None

        if (price > ema200_4h) and (rsi4h >= gate): return "LONG"
        if (price < ema200_4h) and (rsi4h <= (100.0 - gate)): return "SHORT"
        return None

    def _anchor_price(self, row: pd.Series) -> float | None:
        anchor = str(self.config.get("grid_anchor", "ema30")).lower()
        if anchor == "ema30": return float(row.get("ema30", np.nan))
        if anchor == "ema200_4h": return float(row.get("ema200_4h", np.nan))
        return None
