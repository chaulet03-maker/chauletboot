import numpy as np
import pandas as pd


class Strategy:
    def __init__(self, cfg):
        self.config = cfg

    # -------------------------------
    # Señal de entrada (pullback_grid) + filtros
    # -------------------------------
    def check_entry_signal(self, data: pd.DataFrame) -> str | None:
        if len(data) < 2:
            return None
        row = data.iloc[-1]
        ts = data.index[-1]

        # Filtros
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
            # pullback hacia abajo desde el ancla en rango [step, half_span]
            if (price < anchor) and (anchor - price >= step) and (anchor - price <= half_span):
                return "LONG"
        else:  # SHORT
            if (price > anchor) and (price - anchor >= step) and (price - anchor <= half_span):
                return "SHORT"

        return None

    # -------------------------------
    # Stop Loss y Take Profit
    # -------------------------------
    def calculate_sl(self, entry_price: float, last_candle: pd.Series, side: str) -> float:
        atr = float(last_candle["atr"])
        sl_mult = float(self.config.get("sl_atr_mult", 1.3))
        if side == "LONG":
            return entry_price - (atr * sl_mult)
        else:
            return entry_price + (atr * sl_mult)

    def calculate_tp(self, entry_price: float, quantity: float, equity_on_open: float, side: str) -> float:
        """
        TP al % del equity al abrir (idéntico al simulador):
          move = (target_eq_pnl_pct * equity_on_open) / qty
          LONG  -> entry + move
          SHORT -> entry - move
        """
        tp_pct = float(self.config.get("target_eq_pnl_pct", 0.10))
        pnl_target = equity_on_open * tp_pct
        move = pnl_target / max(quantity, 1e-12)
        return (entry_price + move) if side == "LONG" else (entry_price - move)

    # -------------------------------
    # Leverage dinámico por ADX
    # -------------------------------
    def dynamic_leverage(self, last_candle: pd.Series) -> float:
        adx = float(last_candle.get("adx", np.nan))
        thr = float(self.config.get("adx_strong_threshold", 25.0))
        if np.isfinite(adx) and adx >= thr:
            return float(self.config.get("leverage_strong", 10.0))
        return float(self.config.get("leverage_base", 5.0))

    # -------------------------------
    # Filtros auxiliares
    # -------------------------------
    def _passes_filters(self, row: pd.Series, ts: pd.Timestamp, side: str) -> bool:
        # Trend filter 4h
        if str(self.config.get("trend_filter", "ema200_4h")) == "ema200_4h":
            ema200_4h = float(row.get("ema200_4h", np.nan))
            price = float(row.get("close", np.nan))
            if side == "LONG" and not (price > ema200_4h):
                return False
            if side == "SHORT" and not (price < ema200_4h):
                return False

        # RSI4h gate
        rsi_gate = self.config.get("rsi4h_gate", None)
        if rsi_gate is not None:
            rsi4h = float(row.get("rsi4h", np.nan))
            if side == "LONG" and not (rsi4h >= float(rsi_gate)):
                return False
            if side == "SHORT" and not (rsi4h <= (100.0 - float(rsi_gate))):
                return False

        # Confirmación EMA200 1h
        if bool(self.config.get("ema200_1h_confirm", False)):
            price = float(row.get("close", np.nan))
            ema200_1h = float(row.get("ema200", np.nan))
            if side == "LONG" and not (price > ema200_1h):
                return False
            if side == "SHORT" and not (price < ema200_1h):
                return False

        # Volatilidad (ATR%) gates
        atr = float(row.get("atr", np.nan))
        close = float(row.get("close", np.nan))
        if np.isfinite(atr) and np.isfinite(close) and close > 0:
            atrp = (atr / close) * 100.0
            minp = self.config.get("atrp_gate_min", None)
            maxp = self.config.get("atrp_gate_max", None)
            if minp is not None and atrp < float(minp):
                return False
            if maxp is not None and atrp > float(maxp):
                return False

        # Ban hours (UTC)
        ban_hours = set(self.config.get("ban_hours", []) or [])
        if len(ban_hours):
            if int(getattr(ts, "hour", 0)) in ban_hours:
                return False

        # Funding gate (si engine carga el rate actual en config)
        gate_bps = self.config.get("funding_gate_bps", None)
        cur_bps = self.config.get("_funding_rate_bps_now", None)  # puede cargarlo el engine
        if gate_bps is not None and cur_bps is not None:
            # LONG abre si rate <= +gate; SHORT abre si rate >= -gate (simétrico al simulador)
            g = float(gate_bps)
            r = float(cur_bps)
            if side == "LONG" and not (r <= g):
                return False
            if side == "SHORT" and not (r >= -g):
                return False

        return True

    def _decide_grid_side(self, row: pd.Series) -> str | None:
        side_cfg = str(self.config.get("grid_side", "auto")).lower()
        if side_cfg in ("long", "short"):
            return "LONG" if side_cfg == "long" else "SHORT"

        # auto: usa relación precio vs ema200_4h + rsi4h gate
        price = float(row.get("close", np.nan))
        ema200_4h = float(row.get("ema200_4h", np.nan))
        rsi4h = float(row.get("rsi4h", np.nan))
        gate = float(self.config.get("rsi4h_gate", 52.0))

        if not (np.isfinite(price) and np.isfinite(ema200_4h) and np.isfinite(rsi4h)):
            return None

        if (price > ema200_4h) and (rsi4h >= gate):
            return "LONG"
        if (price < ema200_4h) and (rsi4h <= (100.0 - gate)):
            return "SHORT"
        return None

    def _anchor_price(self, row: pd.Series) -> float | None:
        anchor = str(self.config.get("grid_anchor", "ema30")).lower()
        if anchor == "ema30":
            return float(row.get("ema30", np.nan))
        elif anchor == "ema200_4h":
            return float(row.get("ema200_4h", np.nan))
        return None
