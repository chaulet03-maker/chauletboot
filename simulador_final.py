#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtester con:
- Position sizing: modo "risk" (por R) o "full_equity" (usa todo el equity como margen * lev)
- TP / SL reales (por % o ATR)
- TP objetivo como % del equity al abrir (fijo) o dinámico en rango [min,max] según fuerza de señal
- Filtros de tendencia (EMA200 4h + confirmación 1h), gate de RSI 4h
- Filtros de volatilidad (ATR%), sesiones baneadas, funding gate y cobro de funding
- Time stop, daily stop en R, stop de emergencia en R, trailing a BE
- Entradas: rsi_cross | pullback_grid (grid de una pierna, disparo por distancia ATR a ancla)
- Presets (conservador / agresivo)
- Tag de corrida para nombrar archivos de salida

Ejemplos de ejecución al final del archivo (en comentarios)
"""

import argparse
import os
import json
import logging
import sqlite3
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)

# ============================
# Utilidades de carga / normalización OHLCV
# ============================

def _try_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp", engine="python", on_bad_lines="skip")

def _try_read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path, parse_dates=["timestamp"], index_col="timestamp", engine="openpyxl")

def _find_candidate(path: str) -> Optional[str]:
    p = os.path.normpath(path)
    if os.path.exists(p):
        return p
    base, _ = os.path.splitext(p)
    for cand in (p, base, base+".csv", base+".xlsx", base+".xls"):
        if os.path.exists(cand):
            return cand
    stem = os.path.basename(base).lower()
    search_dirs = list({
        os.path.dirname(p) or ".",
        os.path.join(".", "data"),
        os.path.join(".", "data", "hist"),
        os.path.join(".", "full", "data"),
        os.path.join(".", "full", "data", "hist"),
    })
    for sd in search_dirs:
        if not os.path.isdir(sd):
            continue
        for name in os.listdir(sd):
            if stem and stem in name.lower():
                cand = os.path.join(sd, name)
                if os.path.isfile(cand):
                    return cand
    return None

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name is None or str(df.index.name).lower() != "timestamp":
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            for alt in ("time", "date", "datetime"):
                if alt in df.columns:
                    df = df.set_index(alt)
                    break
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    df = df.rename(columns={c: c.lower() for c in df.columns})
    ren = {}
    if "open" not in df.columns:
        for alt in ("o", "op", "open_price"):
            if alt in df.columns: ren[alt] = "open"; break
    if "high" not in df.columns:
        for alt in ("h", "hi", "max", "high_price"):
            if alt in df.columns: ren[alt] = "high"; break
    if "low" not in df.columns:
        for alt in ("l", "lo", "min", "low_price"):
            if alt in df.columns: ren[alt] = "low"; break
    if "close" not in df.columns:
        for alt in ("c", "cl", "close_price", "price"):
            if alt in df.columns: ren[alt] = "close"; break
    if "volume" not in df.columns:
        for alt in ("v", "vol", "volume_usdt", "quote_volume", "base_volume"):
            if alt in df.columns: ren[alt] = "volume"; break
    if ren:
        df = df.rename(columns=ren)

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Columna requerida faltante: {col}")
    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[~df.index.duplicated(keep="last")]
    return df

def leer_archivo_smart(path: str) -> pd.DataFrame:
    cand = _find_candidate(path)
    if not cand:
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    try:
        logging.info(f"Leyendo CSV: {cand}")
        df = _try_read_csv(cand)
    except Exception as e_csv:
        try:
            logging.info(f"CSV falló, intento Excel: {cand} ({e_csv})")
            df = _try_read_excel(cand)
        except Exception as e_xl:
            raise FileNotFoundError(
                f"No se pudo leer '{path}'. Intentos en '{cand}'. CSV error: {e_csv}; Excel error: {e_xl}"
            )
    return _normalize_ohlc(df)

# ============================
# Indicadores
# ============================

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["close"], df["high"], df["low"]
    df["ema10"] = EMAIndicator(c, window=10).ema_indicator()
    df["ema30"] = EMAIndicator(c, window=30).ema_indicator()
    df["ema200"] = EMAIndicator(c, window=200).ema_indicator()  # 1h
    df["rsi"] = RSIIndicator(c, window=14).rsi()
    df["adx"] = ADXIndicator(h, l, c, window=14).adx()
    df["atr"] = AverageTrueRange(h, l, c, window=14).average_true_range()
    return df.dropna()

# ============================
# Métricas
# ============================

def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0

# ============================
# Backtester
# ============================

class RiskSizingBacktester:
    def __init__(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        initial_balance: float = 1000.0,
        fee_pct: float = 0.0005,
        lev: float = 5.0,
        taker_fee: Optional[float] = None,
        maker_fee: Optional[float] = None,
        slip_bps: float = 2.0,
        max_vol_frac: Optional[float] = None,
        margin_safety_pct: Optional[float] = None,
        entry_mode: str = "rsi_cross",  # o "pullback_grid"
        rsi_gate: float = 55.0,
        only_longs: bool = False,
        only_shorts: bool = False,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        # TP/SL base
        use_atr: bool = False,
        sl_pct: Optional[float] = 0.007,
        tp_pct: Optional[float] = 0.015,
        sl_atr_mult: Optional[float] = None,
        tp_atr_mult: Optional[float] = None,
        max_hold_bars: int = 24,
        # Position sizing
        size_mode: str = "risk",  # "risk" | "full_equity"
        risk_pct: Optional[float] = 0.01,
        risk_usd: Optional[float] = None,
        # TP como % de equity
        target_eq_pnl_pct: Optional[float] = None,  # ej 0.10
        target_eq_pnl_pct_min: Optional[float] = None,
        target_eq_pnl_pct_max: Optional[float] = None,
        # Funding
        funding_csv: Optional[str] = None,
        funding_default: float = 0.0001,
        funding_interval_hours: int = 8,
        funding_gate_bps: int = 80,
        # Filtros tendencia 4h + confirmación 1h
        trend_filter: str = "ema200_4h",
        rsi4h_gate: Optional[float] = 52.0,
        ema200_1h_confirm: bool = False,
        # Volatilidad & horario
        atrp_gate_min: Optional[float] = 0.15,
        atrp_gate_max: Optional[float] = 1.20,
        ban_hours: Optional[List[int]] = None,
        # Protecciones en R
        emerg_trade_stop_R: Optional[float] = 1.5,
        daily_stop_R: Optional[float] = 2.0,
        # Trailing
        trail_to_be: bool = False,
        # Cierre final
        no_end_close: bool = False,
        # Pullback grid
        grid_span_atr: float = 2.5,
        grid_step_atr: float = 0.6,
        grid_anchor: str = "ema30",
        grid_side: str = "auto",
        anchor_bias_frac: float = 1.0,
        # tagging
        tag: Optional[str] = None,
    ):
        self.entry_mode = entry_mode.lower().strip()
        if self.entry_mode not in ("rsi_cross", "pullback_grid"):
            raise ValueError("entry_mode debe ser 'rsi_cross' o 'pullback_grid'")

        # recortes
        if start is not None:
            df_1h = df_1h[df_1h.index >= start]
            df_4h = df_4h[df_4h.index >= start]
        if end is not None:
            df_1h = df_1h[df_1h.index <= end]
            df_4h = df_4h[df_4h.index <= end]

        # indicadores 1h y 4h
        self.df_1h = indicators(df_1h.copy())
        df4 = indicators(df_4h.copy())
        df4_agg = df4[["ema200", "rsi"]].rename(columns={"ema200": "ema200_4h", "rsi": "rsi4h"})
        self.df = self.df_1h.join(df4_agg.reindex(self.df_1h.index, method="ffill")).dropna()

        # estado
        self.balance = float(initial_balance)
        self.initial_balance = float(initial_balance)
        self.fee = float(fee_pct)
        self.lev = float(lev)
        self.rsi_gate = float(rsi_gate)
        self.only_longs = bool(only_longs)
        self.only_shorts = bool(only_shorts)

        # Fees por lado (si no vienen, caen al fee_pct existente)
        self.taker_fee = float(taker_fee) if taker_fee is not None else float(fee_pct)
        self.maker_fee = float(maker_fee) if maker_fee is not None else float(fee_pct)

        # Slippage básico y caps
        self.slip_frac = (float(slip_bps) / 10000.0) if slip_bps else 0.0
        self.max_vol_frac = float(max_vol_frac) if max_vol_frac is not None else None

        # Safety de margen
        self.margin_safety_pct = float(margin_safety_pct) if margin_safety_pct is not None else None

        # Orden pendiente para ejecutar en la próxima vela
        self.pending_order: Optional[Dict] = None

        # Margen inicial de la posición (para safety)
        self.initial_margin: Optional[float] = None

        # TP/SL
        self.use_atr = bool(use_atr)
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.max_hold_bars = int(max_hold_bars)

        # Sizing
        self.size_mode = size_mode
        self.risk_pct = risk_pct
        self.risk_usd_fixed = risk_usd

        # TP como % del equity
        self.target_eq_pnl_pct = target_eq_pnl_pct
        self.target_eq_pnl_pct_min = target_eq_pnl_pct_min
        self.target_eq_pnl_pct_max = target_eq_pnl_pct_max

        # Funding
        self.funding_interval_hours = int(funding_interval_hours)
        self.funding_default = float(funding_default)
        self.funding_series = self._load_funding_series(funding_csv) if funding_csv else None
        self.funding_gate_frac = float(funding_gate_bps)/10000.0 if funding_gate_bps is not None else 0.0

        # Filtros
        self.trend_filter = trend_filter.lower() if trend_filter else "none"
        self.rsi4h_gate = rsi4h_gate
        self.ema200_1h_confirm = bool(ema200_1h_confirm)
        self.atrp_gate_min = atrp_gate_min
        self.atrp_gate_max = atrp_gate_max
        self.ban_hours = set(ban_hours or [])

        # Protecciones R
        self.emerg_trade_stop_R = emerg_trade_stop_R
        self.daily_stop_R = daily_stop_R

        # Trailing
        self.trail_to_be = bool(trail_to_be)

        # Cierre final
        self.no_end_close = bool(no_end_close)

        # Pullback grid
        self.grid_span_atr = float(grid_span_atr)
        self.grid_step_atr = float(grid_step_atr)
        self.grid_anchor = grid_anchor.lower().strip()
        if self.grid_anchor not in ("ema30", "ema200_4h"):
            raise ValueError("grid_anchor debe ser 'ema30' o 'ema200_4h'")
        self.grid_side = grid_side.lower().strip()
        if self.grid_side not in ("auto", "long", "short"):
            raise ValueError("grid_side debe ser 'auto', 'long' o 'short'")
        # Sesgo del anchor hacia el precio (k=1 sin cambio; k=0.5 = 50% hacia price)
        self.anchor_bias_frac = float(anchor_bias_frac)

        # Tag
        self.tag = (tag or "").strip()

        # Posición
        self.active = False
        self.side: Optional[str] = None
        self.entry: Optional[float] = None
        self.entry_ts: Optional[pd.Timestamp] = None
        self.qty: float = 0.0
        self.sl_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.bars_in_position: int = 0
        self.risk_usd_trade: Optional[float] = None
        self.eq_on_open: Optional[float] = None
        self.entry_atr: Optional[float] = None
        self.last_sl_regime: Optional[str] = None
        self.last_sl_multiplier: Optional[float] = None

        # Tracking
        self.trades: List[Dict] = []
        self.equity_series: List[float] = [self.balance]
        self.daily_R: Dict[pd.Timestamp, float] = {}

        # Persistencia
        self.db_path = os.path.join("data", "backtests", "trades_history.db")
        self._init_db()

        logging.info(
            f"Backtest RISK SIZING | fee={self.fee*100:.3f}% lev={self.lev:.2f} "
            f"use_atr={self.use_atr} SL%={self.sl_pct} SL_ATR={self.sl_atr_mult} "
            f"TP%={self.tp_pct} TP_ATR={self.tp_atr_mult} entry_mode={self.entry_mode}"
        )

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    open_timestamp TEXT,
                    close_timestamp TEXT NOT NULL,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    qty REAL,
                    pnl REAL,
                    note TEXT,
                    vol_regime TEXT,
                    sl_price REAL,
                    tp_price REAL,
                    sl_multiplier REAL,
                    atr_on_entry REAL
                )
                """
            )
            conn.commit()

    def _persist_trade(
        self,
        open_ts: Optional[pd.Timestamp],
        close_ts: pd.Timestamp,
        side: Optional[str],
        entry_price: Optional[float],
        exit_price: float,
        qty: float,
        pnl: float,
        note: str,
        vol_regime: Optional[str],
        sl_price: Optional[float],
        tp_price: Optional[float],
        sl_multiplier: Optional[float],
        atr_on_entry: Optional[float],
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    open_timestamp,
                    close_timestamp,
                    side,
                    entry_price,
                    exit_price,
                    qty,
                    pnl,
                    note,
                    vol_regime,
                    sl_price,
                    tp_price,
                    sl_multiplier,
                    atr_on_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    open_ts.isoformat() if open_ts is not None else None,
                    close_ts.isoformat(),
                    side,
                    entry_price,
                    exit_price,
                    qty,
                    pnl,
                    note,
                    vol_regime,
                    sl_price,
                    tp_price,
                    sl_multiplier,
                    atr_on_entry,
                ),
            )
            conn.commit()

    # ------------- Funding utils -------------
    @staticmethod
    def _load_funding_series(path: str) -> pd.Series:
        cand = _find_candidate(path) or path
        fdf = pd.read_csv(cand)
        fdf = fdf.rename(columns={c: c.lower() for c in fdf.columns})
        ts_col = 'timestamp'
        if ts_col not in fdf.columns:
            for alt in ('time','date','datetime'):
                if alt in fdf.columns:
                    ts_col = alt; break
        rate_col = 'rate' if 'rate' in fdf.columns else ('funding_rate' if 'funding_rate' in fdf.columns else None)
        if rate_col is None:
            raise ValueError("Funding CSV debe tener columnas 'rate' o 'funding_rate'.")
        fdf[ts_col] = pd.to_datetime(fdf[ts_col], errors='coerce', utc=True)
        fdf = fdf.dropna(subset=[ts_col, rate_col]).sort_values(ts_col)
        return pd.Series(fdf[rate_col].astype(float).values, index=fdf[ts_col])

    def _is_funding_bar(self, ts: pd.Timestamp) -> bool:
        return (ts.minute == 0) and (self.funding_interval_hours > 0) and (ts.hour % self.funding_interval_hours == 0)

    def _funding_rate_at(self, ts: pd.Timestamp) -> float:
        if self.funding_series is None:
            return self.funding_default
        idx = self.funding_series.index.searchsorted(ts)
        if idx == 0:
            return float(self.funding_series.iloc[0])
        if idx >= len(self.funding_series):
            return float(self.funding_series.iloc[-1])
        prev_idx = idx - 1 if self.funding_series.index[idx] != ts else idx
        return float(self.funding_series.iloc[prev_idx])

    # ------------- Costos -------------
    def _fee_cost(self, notional_usd: float, fee_rate: Optional[float] = None) -> float:
        rate = self.taker_fee if fee_rate is None else float(fee_rate)
        return abs(notional_usd) * rate

    # ------------- Helpers de slippage y liquidez -------------
    def _apply_slippage_open(self, price: float, side: str) -> float:
        """Compra LONG => paga +slip; Venta SHORT => recibe -slip."""
        if self.slip_frac <= 0:
            return price
        sign = +1 if side == "LONG" else -1
        return price * (1.0 + sign * self.slip_frac)

    def _apply_slippage_close(self, price: float, side: str) -> float:
        """Cierre LONG (vende) => -slip; Cierre SHORT (compra) => +slip."""
        if self.slip_frac <= 0:
            return price
        sign = +1 if side == "LONG" else -1
        return price * (1.0 - sign * self.slip_frac)

    def _cap_qty_by_liquidity(self, qty: float, price: float, row: pd.Series) -> float:
        """Capea qty por fracción del volumen 1h. Aproxima volumen_quote = close * volume."""
        if not self.max_vol_frac or self.max_vol_frac <= 0:
            return qty
        vol_quote_est = float(row["close"] * row["volume"])  # si 'volume' ya es quote, igual funciona conservador
        max_notional = self.max_vol_frac * vol_quote_est
        max_qty = max_notional / max(price, 1e-12)
        return min(qty, max_qty)

    # ------------- Chequeo intrabar SL/TP (con peor fill conservador) -------------
    def _intrabar_sl_tp(self, row: pd.Series) -> Optional[Tuple[float, str]]:
        """
        Devuelve (fill_price, note) si hay salida intrabar por SL/TP.
        Regla conservadora:
          - LONG: si low<=SL y high>=TP => asume SL primero (peor caso).
                    SL: fill = min(open, SL). TP: fill = TP.
          - SHORT: simétrico (SL primero si choca ambos).
        """
        if not self.active or self.sl_price is None or self.tp_price is None:
            return None
        o = float(row["open"]); h = float(row["high"]); l = float(row["low"])

        if self.side == "LONG":
            hit_sl = (l <= self.sl_price)
            hit_tp = (h >= self.tp_price)
            if hit_sl and hit_tp:
                # peor caso: SL primero
                return (min(o, self.sl_price), "SL")
            if hit_sl:
                return (min(o, self.sl_price), "SL")
            if hit_tp:
                return (self.tp_price, "TP")  # TP como limit: no damos mejor que TP
            return None

        else:  # SHORT
            hit_sl = (h >= self.sl_price)
            hit_tp = (l <= self.tp_price)
            if hit_sl and hit_tp:
                return (max(o, self.sl_price), "SL")
            if hit_sl:
                return (max(o, self.sl_price), "SL")
            if hit_tp:
                return (self.tp_price, "TP")
            return None

    # ------------- Señales / Filtros -------------
    def _passes_trend_filters(self, row: pd.Series, side: str) -> bool:
        price = float(row["close"])
        if self.trend_filter == "ema200_4h":
            ema200_4h = float(row.get("ema200_4h", np.nan))
            if side == "LONG" and not (price > ema200_4h):
                return False
            if side == "SHORT" and not (price < ema200_4h):
                return False
        if self.rsi4h_gate is not None:
            rsi4h = float(row.get("rsi4h", np.nan))
            if side == "LONG" and not (rsi4h >= self.rsi4h_gate):
                return False
            if side == "SHORT" and not (rsi4h <= (100 - self.rsi4h_gate)):
                return False
        if self.ema200_1h_confirm:
            ema200_1h = float(row.get("ema200", np.nan))
            if side == "LONG" and not (price > ema200_1h):
                return False
            if side == "SHORT" and not (price < ema200_1h):
                return False
        return True

    def _volatility_and_session_ok(self, row: pd.Series, ts: pd.Timestamp) -> bool:
        if self.atrp_gate_min is not None or self.atrp_gate_max is not None:
            atr = float(row["atr"])
            close = float(row["close"])
            atrp = (atr / close) * 100.0
            if self.atrp_gate_min is not None and atrp < self.atrp_gate_min:
                return False
            if self.atrp_gate_max is not None and atrp > self.atrp_gate_max:
                return False
        if ts.hour in self.ban_hours:
            return False
        return True

    def _signal_rsi_cross(self, row: pd.Series, prev: pd.Series = None) -> Optional[str]:
        rsi = float(row["rsi"])
        prev_rsi = float(prev["rsi"]) if prev is not None else rsi
        long_cross = (prev_rsi < self.rsi_gate) and (rsi >= self.rsi_gate)
        short_cross = (prev_rsi > (100 - self.rsi_gate)) and (rsi <= (100 - self.rsi_gate))
        if long_cross and not self.only_shorts:
            return "LONG"
        if short_cross and not self.only_longs:
            return "SHORT"
        return None

    def _decide_grid_side(self, row: pd.Series) -> Optional[str]:
        if self.grid_side in ("long", "short"):
            return "LONG" if self.grid_side == "long" else "SHORT"
        price = float(row["close"])
        ema200_4h = float(row.get("ema200_4h", np.nan))
        rsi4h = float(row.get("rsi4h", np.nan))
        if np.isnan(ema200_4h) or np.isnan(rsi4h):
            return None
        if (price > ema200_4h) and (rsi4h >= (self.rsi4h_gate if self.rsi4h_gate is not None else 50)):
            return "LONG"
        if (price < ema200_4h) and (rsi4h <= (100 - (self.rsi4h_gate if self.rsi4h_gate is not None else 50))):
            return "SHORT"
        return None

    def _anchor_price(self, row: pd.Series) -> Optional[float]:
        """
        Devuelve anchor con sesgo hacia el precio:
            anchor_biased = price + k * (anchor_raw - price)
        donde k = self.anchor_bias_frac (1.0 = sin cambio; 0.5 = 50% hacia price).
        """
        import numpy as np

        if self.grid_anchor == "ema30":
            raw = row.get("ema30", np.nan)
        elif self.grid_anchor == "ema200_4h":
            raw = row.get("ema200_4h", np.nan)
        else:
            raw = np.nan

        if raw is None or not np.isfinite(raw):
            return None

        price = float(row.get("close", np.nan))
        if not np.isfinite(price):
            return float(raw)

        k = float(getattr(self, "anchor_bias_frac", 1.0))
        return float(price + k * (float(raw) - price))

    def _signal_pullback_grid(self, row: pd.Series) -> Optional[str]:
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None
        anchor = self._anchor_price(row)
        if anchor is None or not np.isfinite(anchor):
            return None
        price = float(row["close"])
        side_pref = self._decide_grid_side(row)
        if side_pref is None:
            return None

        step = self.grid_step_atr * atr
        half_span = self.grid_span_atr * atr
        if side_pref == "LONG":
            if (price < anchor) and (anchor - price >= step) and (anchor - price <= half_span):
                return "LONG"
        else:
            if (price > anchor) and (price - anchor >= step) and (price - anchor <= half_span):
                return "SHORT"
        return None

    # ------------- Fuerza de señal -> target dinámico -------------
    def _signal_strength(self, row: pd.Series, prev: Optional[pd.Series]) -> float:
        """Devuelve s en [0,1] combinando: RSI 4h alineado, ADX(1h), distancia a EMA200(4h) normalizada por ATR."""
        rsi4h = float(row.get("rsi4h", np.nan))
        adx = float(row.get("adx", np.nan))
        price = float(row.get("close", np.nan))
        ema200_4h = float(row.get("ema200_4h", np.nan))
        atr = float(row.get("atr", np.nan))
        parts = []
        if np.isfinite(rsi4h):
            parts.append(np.clip((abs(rsi4h-50.0)/50.0), 0, 1))  # 0 si neutro, 1 si extremo
        if np.isfinite(adx):
            parts.append(np.clip((adx-10)/30, 0, 1))  # ~0 a 40 -> 0..1
        if np.isfinite(price) and np.isfinite(ema200_4h) and np.isfinite(atr) and atr>0:
            parts.append(np.clip(abs(price-ema200_4h)/(3*atr), 0, 1))  # 0..~1
        if not parts:
            return 0.5
        return float(np.clip(np.mean(parts), 0, 1))

    def _dynamic_target_pct(self, row: pd.Series, prev: Optional[pd.Series]) -> Optional[float]:
        if self.target_eq_pnl_pct is not None:
            return float(self.target_eq_pnl_pct)
        if self.target_eq_pnl_pct_min is None or self.target_eq_pnl_pct_max is None:
            return None
        s = self._signal_strength(row, prev)
        return float(self.target_eq_pnl_pct_min + s * (self.target_eq_pnl_pct_max - self.target_eq_pnl_pct_min))

    # ------------- Mark-to-market -------------
    def _mark_equity(self, price: float) -> float:
        eq = self.balance
        if self.active and self.entry is not None and self.qty > 0:
            pnl = (price - self.entry) * self.qty if self.side == "LONG" else (self.entry - price) * self.qty
            eq += pnl
        self.equity_series.append(eq)
        return eq

    # ------------- SL/TP helpers -------------
    def _calc_sl_tp(
        self,
        side: str,
        entry: float,
        atr: float,
        eq_now: float,
        target_pct: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        sl = None; tp = None
        # SL
        if self.use_atr and self.sl_atr_mult is not None:
            move = self.sl_atr_mult * atr
            sl = entry - move if side == "LONG" else entry + move
        elif (not self.use_atr) and self.sl_pct is not None:
            sl = entry * (1 - self.sl_pct) if side == "LONG" else entry * (1 + self.sl_pct)
        # TP
        if target_pct is not None and self.size_mode == "full_equity":
            # fijar tp para alcanzar target_pct * equity
            # (tp - entry) * qty = target_pct * eq_now  => move = (target_pct * eq_now) / qty
            move = (target_pct * eq_now) / max(self.qty, 1e-12)
            tp = entry + move if side == "LONG" else entry - move
        elif self.use_atr and self.tp_atr_mult is not None:
            move = self.tp_atr_mult * atr
            tp = entry + move if side == "LONG" else entry - move
        elif (not self.use_atr) and self.tp_pct is not None:
            tp = entry * (1 + self.tp_pct) if side == "LONG" else entry * (1 - self.tp_pct)
        return sl, tp

    def _risk_usd(self, balance_now: float) -> float:
        if self.risk_usd_fixed is not None:
            return float(self.risk_usd_fixed)
        if self.risk_pct is None or self.risk_pct <= 0:
            raise ValueError("Debes especificar risk-pct o risk-usd")
        return balance_now * float(self.risk_pct)

    def _qty_from_risk(self, entry: float, sl: float, risk_usd: float) -> float:
        dollar_per_unit = abs(entry - sl)
        if dollar_per_unit <= 0:
            return 0.0
        qty = risk_usd / dollar_per_unit
        return max(qty, 0.0)

    def _qty_full_equity(self, price: float) -> float:
        # notional = equity * lev  => qty = notional / price
        notional = self.balance * self.lev
        qty = notional / max(price, 1e-12)
        return max(qty, 0.0)

    def _get_dynamic_leverage(self, row: pd.Series) -> float:
        """
        Analiza la fuerza de la tendencia (ADX) y devuelve el apalancamiento
        apropiado según 2 niveles: x5 o x10.
        """
        adx = float(row["adx"])

        # --- Definimos el umbral de tendencia ---
        if adx >= 25:
            # Tendencia FUERTE, usamos apalancamiento agresivo
            leverage = 10.0
            trend_strength = "FUERTE"
        else:
            # Tendencia DEBIL o LATERAL, usamos apalancamiento base
            leverage = 5.0
            trend_strength = "DEBIL"

        logging.info(f"Fuerza de tendencia: {trend_strength} (ADX={adx:.2f}). Usando apalancamiento: x{leverage}")

        return leverage

    # ------------- Funding -------------
    def _apply_funding(self, ts: pd.Timestamp, price: float):
        if not self._is_funding_bar(ts) or not self.active or self.qty <= 0:
            return
        rate = self._funding_rate_at(ts)
        notional = self.qty * price
        sign = -1 if self.side == "LONG" else +1
        # Funding correcto: sobre notional, SIN multiplicar por lev otra vez
        funding_pnl = sign * rate * notional
        if funding_pnl != 0:
            self.balance += funding_pnl
            self.trades.append({
                "timestamp": ts, "action": "FUNDING", "side": self.side, "price": float(price),
                "qty": float(self.qty), "regime": "RISK", "mode": "RISK",
                "pnl": float(funding_pnl), "balance": float(self.balance), "note": f"rate={rate:.6f}"
            })

    # ------------- Loop principal -------------
    def run(self):
        df = self.df.copy().reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})
        logging.info(f"Comienzo: {df['timestamp'].iloc[0]}  | Fin: {df['timestamp'].iloc[-1]}  | Velas: {len(df)}")

        def _day(ts: pd.Timestamp) -> pd.Timestamp:
            return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day, tz='UTC')

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            ts = pd.to_datetime(row["timestamp"], utc=True)
            o = float(row["open"]); h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

            # Cobro funding en esta barra si corresponde
            self._apply_funding(ts, c)

            # Mark-to-market al cierre de la barra anterior o saldo actual (si no hay posición)
            eq_now = self._mark_equity(c)

            # 1) Si hay orden pendiente del bar anterior y NO hay posición -> abrir en el OPEN de esta barra (i+1)
            if (not self.active) and (self.pending_order is not None):
                dkey = _day(ts)
                if self.daily_stop_R is None or self.daily_R.get(dkey, 0.0) > -float(self.daily_stop_R):
                    side = self.pending_order["side"]

                    # Precio de entrada = OPEN de esta barra, con slippage
                    entry_raw = o
                    entry_fill = self._apply_slippage_open(entry_raw, side)

                    # ATR actual para SL/TP
                    atr = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
                    self.last_sl_regime = None
                    self.last_sl_multiplier = self.sl_atr_mult if self.use_atr else None
                    target_pct = self._dynamic_target_pct(row, prev)

                    # Sizing
                    leverage_for_this_trade = self.lev
                    if self.size_mode == "full_equity":
                        leverage_for_this_trade = self._get_dynamic_leverage(row)
                        notional = self.balance * leverage_for_this_trade
                        qty = notional / max(entry_fill, 1e-12)
                    else:
                        # modo risk: necesito SL provisional para sizing
                        if self.use_atr and self.sl_atr_mult is not None:
                            move = self.sl_atr_mult * atr
                            sl_tmp = entry_fill - move if side == "LONG" else entry_fill + move
                        elif (not self.use_atr) and self.sl_pct is not None:
                            sl_tmp = entry_fill * (1 - self.sl_pct) if side == "LONG" else entry_fill * (1 + self.sl_pct)
                        else:
                            sl_tmp = None
                        if sl_tmp is None:
                            self.pending_order = None
                            continue
                        risk_u = self._risk_usd(eq_now)
                        qty = self._qty_from_risk(entry_fill, sl_tmp, risk_u)

                    # Cap por liquidez
                    qty = self._cap_qty_by_liquidity(qty, entry_fill, row)
                    self.qty = qty
                    if self.qty <= 0:
                        self.pending_order = None
                        continue

                    # Fees de apertura (taker) sobre notional
                    notional_open = self.qty * entry_fill
                    fee_open = self._fee_cost(notional_open, self.taker_fee)
                    self.balance -= fee_open

                    # Abrir
                    self.active = True
                    self.side = side
                    self.entry = entry_fill
                    self.entry_ts = ts
                    self.entry_atr = atr
                    self.sl_price, self.tp_price = self._calc_sl_tp(
                        side, entry_fill, atr, eq_now, target_pct
                    )
                    self.bars_in_position = 0
                    self.eq_on_open = eq_now

                    # Margen inicial (aprox) para safety: notional/lev
                    self.initial_margin = (self.qty * self.entry) / max(leverage_for_this_trade, 1e-12)

                    # R real en full_equity
                    if self.size_mode == "full_equity":
                        if self.sl_price is not None:
                            self.risk_usd_trade = abs(self.entry - self.sl_price) * self.qty
                        else:
                            self.risk_usd_trade = self.balance * (self.risk_pct if self.risk_pct is not None else 0.01)

                    note_msg = (
                        f"lev={leverage_for_this_trade:.2f} "
                        f"sl={self.sl_price if self.sl_price is not None else np.nan} "
                        f"tp={self.tp_price if self.tp_price is not None else np.nan} "
                        f"size={self.size_mode} "
                        f"target%={target_pct if target_pct is not None else np.nan}"
                    )
                    self.trades.append({
                        "timestamp": ts, "action": "OPEN", "side": self.side, "price": float(self.entry),
                        "qty": float(self.qty), "regime": "RISK", "mode": "RISK",
                        "pnl": -float(fee_open), "balance": float(self.balance),
                        "note": note_msg,
                        "vol_regime": self.last_sl_regime,
                        "sl_multiplier": self.last_sl_multiplier,
                    })

                # consumir la pendiente
                self.pending_order = None

            # 2) Gestión de posición activa (intrabar SL/TP + safety de margen + time stop)
            if self.active:
                self.bars_in_position += 1

                # Safety de margen (antes que SL/TP si hace falta)
                if self.margin_safety_pct and self.initial_margin and self.qty > 0:
                    if self.side == "LONG":
                        px_thr = self.entry - (self.margin_safety_pct * self.initial_margin) / self.qty
                        if l <= px_thr:  # se alcanzó intrabar
                            fill = min(o, px_thr)
                            fill = self._apply_slippage_close(fill, self.side)
                            self._close(ts, fill, note="MARGIN_SAFETY")
                            continue
                    else:  # SHORT
                        px_thr = self.entry + (self.margin_safety_pct * self.initial_margin) / self.qty
                        if h >= px_thr:
                            fill = max(o, px_thr)
                            fill = self._apply_slippage_close(fill, self.side)
                            self._close(ts, fill, note="MARGIN_SAFETY")
                            continue

                # Trailing a BE (igual que antes, usando 'c' para medir progreso)
                if self.trail_to_be and self.tp_price is not None and self.sl_price is not None and self.entry is not None:
                    dist_tp = (self.tp_price - self.entry) if self.side == "LONG" else (self.entry - self.tp_price)
                    if dist_tp is not None and dist_tp > 0:
                        half_way = self.entry + 0.5 * dist_tp if self.side == "LONG" else self.entry - 0.5 * dist_tp
                        if (self.side == "LONG" and c >= half_way):
                            self.sl_price = max(self.sl_price, self.entry)
                        elif (self.side == "SHORT" and c <= half_way):
                            self.sl_price = min(self.sl_price, self.entry)

                # Intrabar SL/TP (con peor fill conservador)
                hit = self._intrabar_sl_tp(row)
                if hit is not None:
                    exit_px, note = hit
                    exit_px = self._apply_slippage_close(exit_px, self.side)
                    self._close(ts, exit_px, note=note)
                    continue

                # Time stop (cierra al close con slippage)
                if self.max_hold_bars and self.bars_in_position >= self.max_hold_bars:
                    exit_px = self._apply_slippage_close(c, self.side)
                    self._close(ts, exit_px, note="TIME_STOP")
                    continue

                # Si seguimos activos, pasamos a la próxima barra
                continue

            # 3) Si NO hay posición, evaluar señal ACTUAL para abrir en la PRÓXIMA barra (pendiente)
            dkey = _day(ts)
            if self.daily_stop_R is not None and self.daily_R.get(dkey, 0.0) <= -float(self.daily_stop_R):
                # día bloqueado
                pass
            else:
                if self.entry_mode == "rsi_cross":
                    sig = self._signal_rsi_cross(row, prev)
                else:
                    sig = self._signal_pullback_grid(row)
                if sig is not None and self._passes_trend_filters(row, sig) and self._volatility_and_session_ok(row, ts):
                    # Funding gate en la barra de señal
                    open_rate = self._funding_rate_at(ts)
                    if not ((sig == "LONG" and open_rate > self.funding_gate_frac) or (sig == "SHORT" and open_rate < -self.funding_gate_frac)):
                        self.pending_order = {"side": sig}

        # Cierre al final
        if self.active:
            last = df.iloc[-1]
            ts = pd.to_datetime(last["timestamp"], utc=True)
            c = float(last["close"])
            if self.no_end_close:
                logging.info("No se fuerza cierre al final (--no-end-close). Posición queda abierta.")
            else:
                exit_px = self._apply_slippage_close(c, self.side)
                self._close(ts, exit_px, note="END")

        self._report()

    # ------------- Cerrar posición -------------
    def _close(self, ts: pd.Timestamp, price: float, note: str):
        if not self.active:
            return
        entry_price = self.entry
        entry_ts = self.entry_ts
        sl_price = self.sl_price
        tp_price = self.tp_price
        qty = self.qty
        vol_regime = self.last_sl_regime
        sl_multiplier = self.last_sl_multiplier
        atr_on_entry = self.entry_atr
        pnl_gross = (price - self.entry) * self.qty if self.side == "LONG" else (self.entry - price) * self.qty
        notional_close = self.qty * price

        # Fee de cierre: maker para TP (limit), taker para el resto (market)
        fee_rate = self.maker_fee if note == "TP" else self.taker_fee
        fee_close = self._fee_cost(notional_close, fee_rate)
        trade_pnl = pnl_gross - fee_close
        self.balance += trade_pnl

        # Actualizar daily R
        if self.risk_usd_trade and self.risk_usd_trade > 0:
            day_key = pd.Timestamp(year=ts.year, month=ts.month, day=ts.day, tz='UTC')
            self.daily_R[day_key] = self.daily_R.get(day_key, 0.0) + (trade_pnl / self.risk_usd_trade)

        self.trades.append({
            "timestamp": ts, "action": "CLOSE", "side": self.side, "price": float(price),
            "qty": float(self.qty), "regime": "RISK", "mode": "RISK",
            "pnl": float(trade_pnl), "balance": float(self.balance), "note": note,
            "vol_regime": vol_regime,
            "sl_multiplier": sl_multiplier,
        })

        try:
            self._persist_trade(
                entry_ts,
                ts,
                self.side,
                entry_price,
                float(price),
                float(qty),
                float(trade_pnl),
                note,
                vol_regime,
                sl_price,
                tp_price,
                sl_multiplier,
                atr_on_entry,
            )
        except Exception as exc:
            logging.error(f"Error al guardar trade en base de datos: {exc}")

        # Reset
        self.active = False
        self.side = None
        self.entry = None
        self.entry_ts = None
        self.qty = 0.0
        self.sl_price = None
        self.tp_price = None
        self.bars_in_position = 0
        self.risk_usd_trade = None
        self.eq_on_open = None
        self.initial_margin = None
        self.entry_atr = None

    # ------------- Reporte -------------
    def _suffix(self) -> str:
        return ("_" + self.tag) if self.tag else ""

    def _report(self):
        out_dir = os.path.join("data", "backtests")
        os.makedirs(out_dir, exist_ok=True)

        eq = pd.Series(self.equity_series, name="equity")
        equity_path = os.path.join(out_dir, f"equity_risk{self._suffix()}.csv")
        eq.to_csv(equity_path, index=False)

        df_tr = pd.DataFrame(self.trades)
        if df_tr.empty:
            print("\nNo se realizaron operaciones.")
            logging.info(f"Equity guardada en: {equity_path}")
            return

        closes = df_tr[df_tr["action"] == "CLOSE"]
        total_pnl = closes["pnl"].sum()
        wins = (closes["pnl"] > 0).sum()
        losses = (closes["pnl"] <= 0).sum()
        winrate = (wins / max(1, wins + losses)) * 100
        avg_win = closes.loc[closes["pnl"] > 0, "pnl"].mean() if wins else 0.0
        avg_loss = closes.loc[closes["pnl"] <= 0, "pnl"].mean() if losses else 0.0
        payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan
        mdd = max_drawdown(eq)

        # Resumen por modo
        try:
            per_mode = closes.groupby("mode")["pnl"].agg(["count", "sum", "mean"]).rename(columns={"count": "trades"})
            per_mode_win = (closes.assign(win=closes["pnl"] > 0).groupby("mode")["win"].mean() * 100).rename("winrate_%")
            per_mode = per_mode.join(per_mode_win)
        except Exception:
            per_mode = pd.DataFrame()

        # Resumen mensual
        closes = closes.copy()
        closes["month"] = pd.to_datetime(closes["timestamp"], utc=True).dt.to_period("M").astype(str)
        monthly = closes.groupby("month").agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            winrate_pct=("pnl", lambda x: (x.gt(0).mean() * 100) if len(x) else 0.0),
            avg_trade=("pnl", "mean"),
        ).reset_index()
        monthly_path = os.path.join(out_dir, f"monthly_stats{self._suffix()}.csv")
        monthly.to_csv(monthly_path, index=False)

        print("\n" + "=" * 64)
        print(" " * 20 + "REPORTE FINAL - RISK SIZING CON SL/TP")
        print("=" * 64)
        print(f"Resultado Neto Final: ${total_pnl:,.2f}")
        print(f"Balance Final:       ${self.balance:,.2f}")
        print(f"Rentabilidad Total:  {(self.balance / self.initial_balance - 1) * 100:.2f}%")
        print(f"Winrate:             {winrate:.2f}%  | Payoff: {payoff:.2f}")
        print(f"Max Drawdown:        {mdd * 100:.2f}%")
        print("-" * 64)
        print("PNL por modo (cierres):")
        if len(per_mode):
            print(per_mode.to_string())
        else:
            print("(sin datos)")
        print("-" * 64)
        print("Resumen mensual:")
        if len(monthly):
            print(monthly.to_string(index=False))
        else:
            print("Sin meses con cierres.")
        print("=" * 64)

        trades_path = os.path.join(out_dir, f"trades_risk{self._suffix()}.csv")
        df_tr.to_csv(trades_path, index=False)
        logging.info(f"Resultados guardados en: {trades_path}")
        logging.info(f"Equity curve guardada en: {equity_path}")
        logging.info(f"Resumen mensual guardado en: {monthly_path}")

# ============================
# PRESETS
# ============================

PRESETS = {
    "conservador": {
        "entry_mode": "pullback_grid",
        "use_atr": True,
        "sl_atr_mult": 1.3,
        "tp_atr_mult": 3.0,
        "trend_filter": "ema200_4h",
        "ema200_1h_confirm": True,
        "rsi4h_gate": 52.0,
        "grid_span_atr": 2.5,
        "grid_step_atr": 0.6,
        "grid_anchor": "ema30",
        "grid_side": "auto",
        "atrp_gate_min": 0.10,
        "atrp_gate_max": 1.20,
        "ban_hours": "0,1,2,3,4",
        "funding_gate_bps": 120,
        "risk_pct": 0.01,
        "max_hold_bars": 8,
        "daily_stop_R": 2.0,
        "emerg_trade_stop_R": 1.8,
    },
    "agresivo": {
        "entry_mode": "pullback_grid",
        "use_atr": True,
        "sl_atr_mult": 1.3,
        "tp_atr_mult": 3.2,
        "trend_filter": "ema200_4h",
        "ema200_1h_confirm": True,
        "rsi4h_gate": 52.0,
        "grid_span_atr": 2.5,
        "grid_step_atr": 0.4,
        "grid_anchor": "ema30",
        "grid_side": "auto",
        "atrp_gate_min": 0.08,
        "atrp_gate_max": 1.40,
        "ban_hours": "0,1,2,3,4",
        "funding_gate_bps": 120,
        "risk_pct": 0.01,  # tu valor por defecto; puedes sobreescribir por flag
        "max_hold_bars": 8,
        "daily_stop_R": 2.2,
        "emerg_trade_stop_R": 2.0,
    }
}

# ============================
# Helpers presets / parseo
# ============================

def _parse_ban_hours(s: Optional[str]) -> List[int]:
    if not s:
        return []
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            h = int(tok)
            if 0 <= h <= 23:
                out.append(h)
        except Exception:
            pass
    return out

def _apply_preset(args: argparse.Namespace, defaults: dict) -> argparse.Namespace:
    if not args.preset:
        return args
    name = args.preset.lower()
    if name not in PRESETS:
        raise ValueError(f"Preset desconocido: {args.preset}")
    preset = PRESETS[name]
    for k, v in preset.items():
        if not hasattr(args, k):
            continue
        current = getattr(args, k)
        default = defaults.get(k, None)
        if current == default or current is None:
            setattr(args, k, v)
    return args

def _save_run_config(args: argparse.Namespace, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    d = {}
    for k, v in vars(args).items():
        try:
            json.dumps({k: v}); d[k] = v
        except Exception:
            d[k] = str(v)
    suffix = ("_" + tag) if tag else ""
    path = os.path.join(out_dir, f"run_config{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    logging.info(f"Config efectiva guardada en: {path}")

# ============================
# Main / CLI
# ============================

def main():
    ap = argparse.ArgumentParser(description="Backtest con sizing por riesgo o full equity, TP/SL reales, filtros y protecciones. Incluye modo pullback_grid y TP objetivo como % del equity (fijo o dinámico).")
    ap.add_argument("--csv1h", required=True, help="Ruta 1h (CSV/Excel)")
    ap.add_argument("--csv4h", required=True, help="Ruta 4h (CSV/Excel)")
    ap.add_argument("--balance", type=float, default=1000.0, help="Balance inicial (equity)")
    ap.add_argument("--fee", type=float, default=0.0005, help="Fee proporcional por trade")
    ap.add_argument("--lev", type=float, default=5.0, help="Apalancamiento")

    # Tag / presets
    ap.add_argument("--preset", type=str, default=None, choices=["conservador", "agresivo"], help="Carga parámetros predefinidos. Tus flags explíticas prevalecen.")
    ap.add_argument("--tag", type=str, default=None, help="Sufijo para archivos de salida")

    # Señal / entrada
    ap.add_argument("--entry-mode", type=str, default="rsi_cross", choices=["rsi_cross", "pullback_grid"], help="Modo de entrada")
    ap.add_argument("--rsi-gate", type=float, default=55.0, help="RSI mínimo para LONGs (y 100-gate para SHORTs)")
    ap.add_argument("--only-longs", action="store_true", help="Solo LONGs")
    ap.add_argument("--only-shorts", action="store_true", help="Solo SHORTs")

    # Rango temporal
    ap.add_argument("--start", type=str, default=None, help="Inicio (ISO, ej 2024-01-01)")
    ap.add_argument("--end", type=str, default=None, help="Fin (ISO)")

    # TP/SL base
    ap.add_argument("--use-atr", action="store_true", help="Usar ATR para SL/TP (sino %)")
    ap.add_argument("--sl-pct", type=float, default=0.007, help="SL % (si no ATR)")
    ap.add_argument("--tp-pct", type=float, default=0.015, help="TP % (si no ATR y sin target de equity)")
    ap.add_argument("--sl-atr-mult", type=float, default=None, help="SL = mult*ATR")
    ap.add_argument("--tp-atr-mult", type=float, default=None, help="TP = mult*ATR (si no target equity)")
    ap.add_argument("--max-hold-bars", type=int, default=24, help="Time stop (velas 1h)")

    # Position sizing
    ap.add_argument("--size-mode", type=str, default="risk", choices=["risk", "full_equity"], help="risk = por R; full_equity = usa todo el equity")
    ap.add_argument("--risk-pct", type=float, default=0.01, help="Riesgo por trade como % del balance (modo risk)")
    ap.add_argument("--risk-usd", type=float, default=None, help="Riesgo fijo por trade en USD (anula risk-pct)")

    # TP como % del equity
    ap.add_argument("--target-eq-pnl-pct", type=float, default=None, help="Objetivo de PnL como % del equity al abrir (ej 0.10 = 10%)")
    ap.add_argument("--target-eq-pnl-pct-min", type=float, default=None, help="Mín para target dinámico")
    ap.add_argument("--target-eq-pnl-pct-max", type=float, default=None, help="Máx para target dinámico")

    # Funding
    ap.add_argument("--funding-csv", type=str, default=None, help="CSV de funding (timestamp, rate/funding_rate)")
    ap.add_argument("--funding-default", type=float, default=0.0001, help="Funding por defecto/8h si no hay CSV")
    ap.add_argument("--funding-interval-hours", type=int, default=8, help="Intervalo de funding (8h)")
    ap.add_argument("--funding-gate-bps", type=int, default=80, help="Evitar abrir si |rate| > gate (bps por 8h)")

    # Filtros tendencia 4h + confirmación 1h
    ap.add_argument("--trend-filter", type=str, default="ema200_4h", choices=["none", "ema200_4h"], help="Filtro de tendencia 4h")
    ap.add_argument("--rsi4h-gate", type=float, default=52.0, help="RSI 4h mínimo para LONG y (100-valor) para SHORT")
    ap.add_argument("--ema200-1h-confirm", action="store_true", help="Requiere precio > EMA200(1h) para LONG y < para SHORT")

    # Volatilidad & horario
    ap.add_argument("--atrp-gate-min", type=float, default=0.15, help="ATR%% mínimo (evitar chop)")
    ap.add_argument("--atrp-gate-max", type=float, default=1.20, help="ATR%% máximo (evitar latigazos)")
    ap.add_argument("--ban-hours", type=str, default="", help="Horas UTC a evitar (ej. 0,1,2,3)")

    # Protecciones en R
    ap.add_argument("--emerg-trade-stop-R", type=float, default=1.5, help="Cerrar si pérdida latente < -R*emerg")
    ap.add_argument("--daily-stop-R", type=float, default=2.0, help="No abrir más si el día acumula <= -R*daily")

    # Trailing / cierre final
    ap.add_argument("--trail-to-be", action="store_true", help="Mueve SL a BE al 50% del recorrido a TP")
    ap.add_argument("--no-end-close", action="store_true", help="No forzar cierre al final del período")

    # Pullback grid
    ap.add_argument("--grid-span-atr", type=float, default=2.5, help="Ancho total de banda desde ancla (en ATR)")
    ap.add_argument("--grid-step-atr", type=float, default=0.6, help="Paso de disparo de pullback (en ATR)")
    ap.add_argument("--grid-anchor", type=str, default="ema30", choices=["ema30", "ema200_4h"], help="Ancla del grid dinámico")
    ap.add_argument("--grid-side", type=str, default="auto", choices=["auto", "long", "short"], help="Lado preferido del grid")
    ap.add_argument(
        "--anchor-bias-frac",
        type=float,
        default=1.0,
        help="Sesgo del anchor hacia el precio: anchor'=price + k*(anchor-price). Usa 0.5 para mover el anchor 50% hacia price.",
    )

    # Fees por lado y slippage
    ap.add_argument("--taker-fee", type=float, default=None, help="Fee taker por lado (ej 0.0006)")
    ap.add_argument("--maker-fee", type=float, default=None, help="Fee maker por lado (ej 0.0003)")
    ap.add_argument("--slip-bps", type=float, default=2.0, help="Slippage básico en bps por lado (ej 2=0.02%)")

    # Cap de posición vs. liquidez y safety de margen
    ap.add_argument("--max-vol-frac", type=float, default=0.005, help="Fracción máxima del volumen 1h usada para el notional (0.005 = 0.5%)")
    ap.add_argument("--margin-safety-pct", type=float, default=0.5, help="Cierre de emergencia si pérdida latente >= pct del margen inicial (ej 0.5 = 50%)")

    args = ap.parse_args()
    defaults = {a.dest: a.default for a in ap._actions if a.dest and a.dest != "help"}
    args = _apply_preset(args, defaults)
    df_1h = leer_archivo_smart(args.csv1h)
    df_4h = leer_archivo_smart(args.csv4h)

    start_ts = pd.to_datetime(args.start, utc=True) if args.start else None
    end_ts = pd.to_datetime(args.end, utc=True) if args.end else None

    bt = RiskSizingBacktester(
        df_1h=df_1h,
        df_4h=df_4h,
        initial_balance=args.balance,
        fee_pct=args.fee,
        lev=args.lev,
        taker_fee=args.taker_fee,
        maker_fee=args.maker_fee,
        slip_bps=args.slip_bps,
        max_vol_frac=args.max_vol_frac,
        margin_safety_pct=args.margin_safety_pct,
        entry_mode=args.entry_mode,
        rsi_gate=args.rsi_gate,
        only_longs=args.only_longs,
        only_shorts=args.only_shorts,
        start=start_ts,
        end=end_ts,
        use_atr=bool(args.use_atr),
        sl_pct=args.sl_pct,
        tp_pct=args.tp_pct,
        sl_atr_mult=args.sl_atr_mult,
        tp_atr_mult=args.tp_atr_mult,
        max_hold_bars=args.max_hold_bars,
        size_mode=args.size_mode,
        risk_pct=args.risk_pct,
        risk_usd=args.risk_usd,
        target_eq_pnl_pct=args.target_eq_pnl_pct,
        target_eq_pnl_pct_min=args.target_eq_pnl_pct_min,
        target_eq_pnl_pct_max=args.target_eq_pnl_pct_max,
        funding_csv=args.funding_csv,
        funding_default=args.funding_default,
        funding_interval_hours=args.funding_interval_hours,
        funding_gate_bps=args.funding_gate_bps,
        trend_filter=args.trend_filter,
        rsi4h_gate=args.rsi4h_gate,
        ema200_1h_confirm=bool(args.ema200_1h_confirm),
        atrp_gate_min=args.atrp_gate_min,
        atrp_gate_max=args.atrp_gate_max,
        ban_hours=_parse_ban_hours(args.ban_hours),
        emerg_trade_stop_R=args.emerg_trade_stop_R,
        daily_stop_R=args.daily_stop_R,
        trail_to_be=bool(args.trail_to_be),
        no_end_close=bool(args.no_end_close),
        grid_span_atr=args.grid_span_atr,
        grid_step_atr=args.grid_step_atr,
        grid_anchor=args.grid_anchor,
        grid_side=args.grid_side,
        anchor_bias_frac=args.anchor_bias_frac,
        tag=(args.tag or "").strip(),
    )

    _save_run_config(args, out_dir=os.path.join("data", "backtests"), tag=(args.tag or ""))
    bt.run()

if __name__ == "__main__":
    main()
