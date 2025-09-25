
import math
import pandas as pd

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _slope_bps(series: pd.Series, k: int, price: float) -> float:
    if k <= 0 or len(series) <= k or price <= 0:
        return 0.0
    try:
        return ( (series.iloc[-1] - series.iloc[-k-1]) / price ) * 10000.0
    except Exception:
        return 0.0

def classify(row_or_df, cfg: dict):
    """
    Clasifica el mercado en: 'range' | 'uptrend' | 'downtrend' | 'chop'
    Usa EMA 9/20/50/200 + pendientes + ADX + ancho de bandas (bb_width_bps).
    Acepta un row con columnas ya calculadas o un df para calcular EMAs on-the-fly.
    """
    # Extrae parámetros
    ema_set = (cfg or {}).get("ema_set", [9,20,50,200])
    slope_lookback = int((cfg or {}).get("slope_lookback", 10))

    # Permitir row (última fila) o df
    if isinstance(row_or_df, pd.DataFrame):
        df = row_or_df
        row = df.iloc[-1]
    else:
        row = row_or_df
        df = None

    # Asegurar EMAs presentes
    price = float(getattr(row, "close", getattr(row, "price", 0)) or 0)
    out = {}
    for n in [9,20,50,200]:
        col = f"ema{n}"
        val = getattr(row, col, None)
        if pd.isna(val) if hasattr(pd, "isna") else (val is None):
            # calcular si nos dieron df
            if df is not None and "close" in df.columns:
                df[col] = _ema(df["close"], n)
                val = float(df[col].iloc[-1])
            else:
                val = price
        out[col] = float(val)

    ema9, ema20, ema50, ema200 = out["ema9"], out["ema20"], out["ema50"], out["ema200"]

    # Pendientes en bps/bar
    if df is not None and "close" in df.columns and len(df) > slope_lookback+1:
        s9  = _slope_bps(df["close"].ewm(span=9,  adjust=False).mean(),  slope_lookback, price)
        s20 = _slope_bps(df["close"].ewm(span=20, adjust=False).mean(),  slope_lookback, price)
        s50 = _slope_bps(df["close"].ewm(span=50, adjust=False).mean(),  slope_lookback, price)
    else:
        s9 = s20 = s50 = 0.0

    # Indicadores auxiliares si existen
    adx = float(getattr(row, "adx", 0) or 0)
    bb_width_bps = float(getattr(row, "bb_width_bps", 0) or 0)

    # Reglas del config
    rules = (cfg or {}).get("rules", {})

    # Helpers
    def ema_order_is(order: str) -> bool:
        if order == "9>20>50>200":
            return ema9 > ema20 > ema50 > ema200
        if order == "9<20<50<200":
            return ema9 < ema20 < ema50 < ema200
        return False

    # RANGE / CHATO
    r = rules.get("range", {})
    if (abs(s9) <= float(r.get("abs_slope_ema9_bps_max", 2)) and
        abs(s50) <= float(r.get("abs_slope_ema50_bps_max", 1)) and
        adx < float(r.get("adx_max", 18)) and
        bb_width_bps < float(r.get("bb_width_bps_max", 12)) and
        abs(ema50 - ema200)/max(price,1) * 10000.0 < 5.0):  # 5 bps de separación entre 50 y 200
        return type("Regime", (), {"name": "range", "ema": (ema9,ema20,ema50,ema200), "slope": (s9,s20,s50)})

    # TENDENCIA ALCISTA
    u = rules.get("uptrend", {})
    if (ema_order_is(u.get("ema_order","9>20>50>200")) and
        s20 >= float(u.get("slope_ema20_bps_min", 2)) and
        s50 >= float(u.get("slope_ema50_bps_min", 1)) and
        adx >= float(u.get("adx_min", 20)) and
        bb_width_bps >= float(rules.get("range", {}).get("bb_width_bps_max", 12))):
        return type("Regime", (), {"name": "uptrend", "ema": (ema9,ema20,ema50,ema200), "slope": (s9,s20,s50)})

    # TENDENCIA BAJISTA
    d = rules.get("downtrend", {})
    if (ema_order_is(d.get("ema_order","9<20<50<200")) and
        s20 <= float(d.get("slope_ema20_bps_max", -2)) and
        s50 <= float(d.get("slope_ema50_bps_max", -1)) and
        adx >= float(d.get("adx_min", 20)) and
        bb_width_bps >= float(rules.get("range", {}).get("bb_width_bps_max", 12))):
        return type("Regime", (), {"name": "downtrend", "ema": (ema9,ema20,ema50,ema200), "slope": (s9,s20,s50)})

    # CHOP / TRANSICIÓN (default si nada anterior)
    return type("Regime", (), {"name": "chop", "ema": (ema9,ema20,ema50,ema200), "slope": (s9,s20,s50)})
