import logging

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

    def get_rejection_reason(self, data: pd.DataFrame) -> tuple[str, str, str]:
        """
        Devuelve (side, code, detail) si la última vela NO habilita entrada.
        code: 'trend_4h' | 'ema200_1h_confirm' | 'rsi4h_gate' | 'atr_gate' | 'ban_hours' | 'funding_gate' | 'grid_out_of_range' | 'no_signal'
        """
        import numpy as np
        if data is None or len(data) == 0:
            return ("", "no_signal", "Sin datos")

        row = data.iloc[-1]
        ts = data.index[-1] if hasattr(data, "index") and len(data.index) else None

        price     = float(row.get("close", float("nan")))
        ema200_4h = float(row.get("ema200_4h", float("nan")))
        ema200_1h = float(row.get("ema200", float("nan")))
        rsi4h     = float(row.get("rsi4h", float("nan")))
        atr       = float(row.get("atr", float("nan")))

        side_pref = self._decide_grid_side(row)
        side_out  = (side_pref or "").upper()

        atrp = (atr / price * 100.0) if (np.isfinite(atr) and np.isfinite(price) and price > 0) else float("nan")

        gate_bps = self.config.get("funding_gate_bps", None)
        r_dec = self.config.get("_funding_rate_now", None)
        r_bps = self.config.get("_funding_rate_bps_now", None)
        if r_dec is None and r_bps is not None:
            try: r_dec = float(r_bps) / 10000.0
            except Exception: r_dec = None
        g_frac = (float(gate_bps) / 10000.0) if gate_bps is not None else None

        # 1) Tendencia 4h
        if str(self.config.get("trend_filter", "ema200_4h")) == "ema200_4h" and np.isfinite(price) and np.isfinite(ema200_4h):
            if side_out == "LONG"  and not (price > ema200_4h):  return (side_out, "trend_4h", "Precio <= EMA200 (4h)")
            if side_out == "SHORT" and not (price < ema200_4h):  return (side_out, "trend_4h", "Precio >= EMA200 (4h)")

        # 2) RSI 4h
        rsi_gate = self.config.get("rsi4h_gate", None)
        if rsi_gate is not None and np.isfinite(rsi4h):
            g = float(rsi_gate)
            if side_out == "LONG"  and not (rsi4h >= g):              return (side_out, "rsi4h_gate", f"RSI4h {rsi4h:.2f} < {g:.2f}")
            if side_out == "SHORT" and not (rsi4h <= (100.0 - g)):    return (side_out, "rsi4h_gate", f"RSI4h {rsi4h:.2f} > {100.0 - g:.2f}")

        # 3) Confirmación EMA200 1h
        if bool(self.config.get("ema200_1h_confirm", False)) and np.isfinite(price) and np.isfinite(ema200_1h):
            if side_out == "LONG"  and not (price >= ema200_1h):      return (side_out, "ema200_1h_confirm", "Precio < EMA200 (1h)")
            if side_out == "SHORT" and not (price <= ema200_1h):      return (side_out, "ema200_1h_confirm", "Precio > EMA200 (1h)")

        # 4) ATR%
        minp = self.config.get("atrp_gate_min", None)
        maxp = self.config.get("atrp_gate_max", None)
        if (minp is not None or maxp is not None):
            if not (np.isfinite(atr) and np.isfinite(price) and price > 0):
                return (side_out, "atr_gate", "ATR/Precio inválidos")
            atrp_val = (atr / price) * 100.0
            if minp is not None and atrp_val < float(minp):            return (side_out, "atr_gate", f"ATR% {atrp_val:.2f} < Min {float(minp):.2f}")
            if maxp is not None and atrp_val > float(maxp):            return (side_out, "atr_gate", f"ATR% {atrp_val:.2f} > Max {float(maxp):.2f}")

        # 5) ban_hours
        if ts is not None and self.config.get("ban_hours"):
            try:
                ban = {int(h) for h in str(self.config.get("ban_hours")).split(",") if str(h).strip().isdigit()}
                if int(getattr(ts, "hour", 0)) in ban:
                    return (side_out, "ban_hours", f"Hora bloqueada={int(getattr(ts,'hour',0))}")
            except Exception:
                pass

        # 6) Funding gate
        if g_frac is not None and r_dec is not None:
            try:
                if (side_out == "LONG" and float(r_dec) > float(g_frac)) or (side_out == "SHORT" and float(r_dec) < -float(g_frac)):
                    return (side_out, "funding_gate", f"rate={float(r_dec):.6f} gate={float(g_frac):.6f}")
            except Exception:
                pass

        # 7) Grid fuera de rango (pullback)
        anchor_name = str(self.config.get("grid_anchor", "ema30")).lower()
        anchor = float(row.get("ema30" if anchor_name == "ema30" else "ema200_4h", float("nan")))
        step  = float(self.config.get("grid_step_atr", 0.32)) * (atr if np.isfinite(atr) else 0.0)
        span  = float(self.config.get("grid_span_atr", 3.0))  * (atr if np.isfinite(atr) else 0.0)
        if np.isfinite(price) and np.isfinite(anchor) and np.isfinite(step) and np.isfinite(span):
            if side_out == "LONG":
                ok = (price < anchor) and ((anchor - price) >= step) and ((anchor - price) <= span)
                if not ok: return (side_out, "grid_out_of_range", "Grid LONG fuera de [step,span]")
            else:
                ok = (price > anchor) and ((price - anchor) >= step) and ((price - anchor) <= span)
                if not ok: return (side_out, "grid_out_of_range", "Grid SHORT fuera de [step,span]")

        return (side_out, "no_signal", "No pasó filtros / sin señal utilizable")

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

    # --- DEBUG: explica por qué no hay señal en la última vela ---
    def explain_signal(self, data: pd.DataFrame) -> None:
        """
        Loguea un snapshot de condiciones y qué filtro(s) bloquearon la entrada.
        Se activa desde el engine si config.debug_signals = True.
        """
        log = logging.getLogger(__name__)
        if len(data) == 0:
            log.info("SIGNAL DEBUG: sin datos."); return
        row = data.iloc[-1]
        ts = data.index[-1]

        try:
            price = float(row.get("close", float("nan")))
            ema200_4h = float(row.get("ema200_4h", float("nan")))
            ema200_1h = float(row.get("ema200", float("nan")))
            rsi4h = float(row.get("rsi4h", float("nan")))
            atr = float(row.get("atr", float("nan")))
            adx = float(row.get("adx", float("nan")))
        except Exception:
            log.info("SIGNAL DEBUG: fila inválida."); return

        # Lado preferido por tendencia
        side_pref = None
        gate = float(self.config.get("rsi4h_gate", 52.0))
        if np.isfinite(price) and np.isfinite(ema200_4h) and np.isfinite(rsi4h):
            if (price > ema200_4h) and (rsi4h >= gate):
                side_pref = "LONG"
            elif (price < ema200_4h) and (rsi4h <= (100.0 - gate)):
                side_pref = "SHORT"

        # Grid geometry
        anchor_name = str(self.config.get("grid_anchor", "ema30")).lower()
        anchor = float(row.get("ema30" if anchor_name == "ema30" else "ema200_4h", float("nan")))
        step = float(self.config.get("grid_step_atr", 0.32)) * (atr if np.isfinite(atr) else 0.0)
        span = float(self.config.get("grid_span_atr", 3.0)) * (atr if np.isfinite(atr) else 0.0)

        # ATR%
        atrp = (atr / price * 100.0) if (np.isfinite(atr) and np.isfinite(price) and price > 0) else float("nan")

        # Funding gate runtime (si está presente)
        gate_bps = self.config.get("funding_gate_bps", None)
        r_dec = self.config.get("_funding_rate_now", None)
        r_bps = self.config.get("_funding_rate_bps_now", None)
        if r_dec is None and r_bps is not None:
            try:
                r_dec = float(r_bps) / 10000.0
            except Exception:
                r_dec = None
        g_frac = (float(gate_bps) / 10000.0) if gate_bps is not None else None

        reasons = []

        # Reglas de filtros (mismas que _passes_filters, pero explicadas)
        # Tendencia 4h
        if str(self.config.get("trend_filter", "ema200_4h")) == "ema200_4h" and np.isfinite(price) and np.isfinite(ema200_4h):
            if side_pref == "LONG" and not (price > ema200_4h): reasons.append("trend: price<=ema200_4h")
            if side_pref == "SHORT" and not (price < ema200_4h): reasons.append("trend: price>=ema200_4h")

        # Confirmación EMA200 1h
        if bool(self.config.get("ema200_1h_confirm", False)) and np.isfinite(price) and np.isfinite(ema200_1h):
            if side_pref == "LONG" and not (price > ema200_1h): reasons.append("confirm: price<=ema200_1h")
            if side_pref == "SHORT" and not (price < ema200_1h): reasons.append("confirm: price>=ema200_1h")

        # RSI4h gate
        if np.isfinite(rsi4h):
            if side_pref == "LONG" and not (rsi4h >= gate): reasons.append(f"rsi4h<{gate}")
            if side_pref == "SHORT" and not (rsi4h <= (100.0 - gate)): reasons.append(f"rsi4h>{100.0 - gate}")

        # ATR% gates
        minp = self.config.get("atrp_gate_min", None)
        maxp = self.config.get("atrp_gate_max", None)
        if minp is not None and np.isfinite(atrp) and atrp < float(minp): reasons.append(f"atrp<{minp}")
        if maxp is not None and np.isfinite(atrp) and atrp > float(maxp): reasons.append(f"atrp>{maxp}")

        # Ban hour UTC
        ban_hours = set(self.config.get("ban_hours", []) or [])
        if len(ban_hours) and int(getattr(ts, "hour", 0)) in ban_hours:
            reasons.append(f"ban_hour={int(getattr(ts, 'hour', 0))}UTC")

        # Funding gate
        if (g_frac is not None) and (r_dec is not None) and side_pref is not None:
            if (side_pref == "LONG" and r_dec > +g_frac) or (side_pref == "SHORT" and r_dec < -g_frac):
                reasons.append(f"funding_gate side={side_pref} rate={r_dec:.6f} gate={g_frac:.6f}")

        # Geometría del pullback
        if side_pref == "LONG" and np.isfinite(anchor) and np.isfinite(step) and np.isfinite(span) and np.isfinite(price):
            ok = (price < anchor) and ((anchor - price) >= step) and ((anchor - price) <= span)
            if not ok: reasons.append("grid LONG: fuera de rango [step,span]")
        if side_pref == "SHORT" and np.isfinite(anchor) and np.isfinite(step) and np.isfinite(span) and np.isfinite(price):
            ok = (price > anchor) and ((price - anchor) >= step) and ((price - anchor) <= span)
            if not ok: reasons.append("grid SHORT: fuera de rango [step,span]")

        log.info(
            "SIGNAL DEBUG ts=%s side_pref=%s price=%.2f anchor=%s step=%.2f span=%.2f atr=%.2f atrp=%.2f rsi4h=%.2f adx=%.2f ema200_4h=%.2f ema200_1h=%.2f funding_dec=%s gate_bps=%s reasons=%s",
            str(ts), side_pref, price, ("%.2f" % anchor) if np.isfinite(anchor) else "nan",
            step if np.isfinite(step) else 0.0, span if np.isfinite(span) else 0.0,
            atr if np.isfinite(atr) else float("nan"),
            atrp if np.isfinite(atrp) else float("nan"),
            rsi4h if np.isfinite(rsi4h) else float("nan"),
            adx if np.isfinite(adx) else float("nan"),
            ema200_4h if np.isfinite(ema200_4h) else float("nan"),
            ema200_1h if np.isfinite(ema200_1h) else float("nan"),
            ("%.6f" % r_dec) if (r_dec is not None) else "None",
            str(gate_bps) if (gate_bps is not None) else "None",
            reasons or ["OK (si no hay señal, probablemente grid no se activó)"]
        )
