from types import SimpleNamespace
import pandas as pd

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _slope_bps(series: pd.Series, k: int, price: float) -> float:
    """
    Pendiente en basis points por barra de una serie suavizada.
    """
    if k <= 0 or len(series) <= (k + 1) or price <= 0:
        return 0.0
    try:
        return ((series.iloc[-1] - series.iloc[-(k+1)]) / price) * 10000.0
    except Exception:
        return 0.0

def classify(row_or_df, cfg: dict):
    """
    Clasifica el mercado en: 'range' | 'uptrend' | 'downtrend' | 'chop'
    Usa EMA 9/20/50/200 + pendientes + ADX + ancho de bandas (bb_width_bps).

    row_or_df: última fila (pd.Series) o un DataFrame (pd.DataFrame).
    cfg: {
      "ema_set": [9,20,50,200],
      "slope_lookback": 10,
      "rules": {
        "range":     {"abs_slope_ema9_bps_max": 2, "abs_slope_ema50_bps_max": 1, "adx_max": 18, "bb_width_bps_max": 12},
        "uptrend":   {"ema_order": "9>20>50>200", "slope_ema20_bps_min": 2, "slope_ema50_bps_min": 1, "adx_min": 20, "bb_width_bps_min": 12},
        "downtrend": {"ema_order": "9<20<50<200", "slope_ema20_bps_max": -2, "slope_ema50_bps_max": -1, "adx_min": 20, "bb_width_bps_min": 12}
      }
    }
    """
    cfg = cfg or {}
    ema_set = list(cfg.get("ema_set", [9, 20, 50, 200]))
    # Garantizamos las 4 básicas aunque el usuario pase otra lista
    for n in (9, 20, 50, 200):
        if n not in ema_set:
            ema_set.append(n)
    ema_set = sorted(set(ema_set))

    slope_lookback = int(cfg.get("slope_lookback", 10))

    # Aceptar row o df
    if isinstance(row_or_df, pd.DataFrame):
        df = row_or_df
        row = df.iloc[-1]
    else:
        row = row_or_df
        df = None

    # Precio y salida de EMAs
    price = float(getattr(row, "close", getattr(row, "price", 0.0)) or 0.0)
    out_ema = {}

    # Asegurar EMAs presentes (si no están, se calculan con df si lo tenemos)
    for n in ema_set:
        col = f"ema{n}"
        val = getattr(row, col, None)
        if val is None or (hasattr(pd, "isna") and pd.isna(val)):
            if df is not None and "close" in df.columns:
                # Calculamos y cacheamos en df
                if col not in df.columns:
                    df[col] = _ema(df["close"], n)
                try:
                    val = float(df[col].iloc[-1])
                except Exception:
                    val = price
            else:
                val = price
        out_ema[col] = float(val)

    # Extraemos las 4 claves que usamos en las reglas base
    ema9   = out_ema.get("ema9", price)
    ema20  = out_ema.get("ema20", price)
    ema50  = out_ema.get("ema50", price)
    ema200 = out_ema.get("ema200", price)

    # Pendientes en bps/bar (si hay df)
    if df is not None and "close" in df.columns and len(df) > (slope_lookback + 1):
        ema9_series  = df["close"].ewm(span=9,  adjust=False).mean()
        ema20_series = df["close"].ewm(span=20, adjust=False).mean()
        ema50_series = df["close"].ewm(span=50, adjust=False).mean()
        s9  = _slope_bps(ema9_series,  slope_lookback, price)
        s20 = _slope_bps(ema20_series, slope_lookback, price)
        s50 = _slope_bps(ema50_series, slope_lookback, price)
    else:
        s9 = s20 = s50 = 0.0

    # Indicadores auxiliares
    adx = float(getattr(row, "adx", 0.0) or 0.0)

    # Aceptar varios nombres para el ancho de bandas
    bb_width_bps = None
    for k in ("bb_width_bps", "bb_width", "bbwidth_bps"):
        v = getattr(row, k, None)
        if v is not None and not (hasattr(pd, "isna") and pd.isna(v)):
            bb_width_bps = float(v)
            break
    if bb_width_bps is None:
        bb_width_bps = 0.0

    # Reglas
    rules = cfg.get("rules", {})
    r = rules.get("range", {})
    u = rules.get("uptrend", {})
    d = rules.get("downtrend", {})

    # Helper: orden de EMAs
    def ema_order_is(order: str) -> bool:
        if order == "9>20>50>200":
            return ema9 > ema20 > ema50 > ema200
        if order == "9<20<50<200":
            return ema9 < ema20 < ema50 < ema200
        return False

    # ===== RANGE =====
    if (
        abs(s9)  <= float(r.get("abs_slope_ema9_bps_max", 2)) and
        abs(s50) <= float(r.get("abs_slope_ema50_bps_max", 1)) and
        adx      <  float(r.get("adx_max", 18)) and
        bb_width_bps < float(r.get("bb_width_bps_max", 12)) and
        (abs(ema50 - ema200) / max(price, 1.0) * 10000.0) < 5.0  # ~5 bps de separación entre 50 y 200
    ):
        return SimpleNamespace(
            name="range",
            ema=(ema9, ema20, ema50, ema200),
            slope=(s9, s20, s50)
        )

    # ===== UPTREND =====
    if (
        ema_order_is(u.get("ema_order", "9>20>50>200")) and
        s20 >= float(u.get("slope_ema20_bps_min", 2)) and
        s50 >= float(u.get("slope_ema50_bps_min", 1)) and
        adx >= float(u.get("adx_min", 20)) and
        # --- CORRECCIÓN: Se usa `u` (uptrend config) en lugar de `r` ---
        bb_width_bps >= float(u.get("bb_width_bps_min", 12))
    ):
        return SimpleNamespace(
            name="uptrend",
            ema=(ema9, ema20, ema50, ema200),
            slope=(s9, s20, s50)
        )

    # ===== DOWNTREND =====
    if (
        ema_order_is(d.get("ema_order", "9<20<50<200")) and
        s20 <= float(d.get("slope_ema20_bps_max", -2)) and
        s50 <= float(d.get("slope_ema50_bps_max", -1)) and
        adx >= float(d.get("adx_min", 20)) and
        # --- CORRECCIÓN: Se usa `d` (downtrend config) en lugar de `r` ---
        bb_width_bps >= float(d.get("bb_width_bps_min", 12))
    ):
        return SimpleNamespace(
            name="downtrend",
            ema=(ema9, ema20, ema50, ema200),
            slope=(s9, s20, s50)
        )

    # ===== CHOP (default) =====
    return SimpleNamespace(
        name="chop",
        ema=(ema9, ema20, ema50, ema200),
        slope=(s9, s20, s50)
    )