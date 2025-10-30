from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Iterable

def load_csv(
    path: str,
    *,
    tz: str = "UTC",
    assume_unit: Optional[str] = None,   # "s" | "ms" | "ns" | None (auto)
    sep: Optional[str] = None,           # autodetect si None
    encoding: Optional[str] = None,      # deja que pandas adivine si None
    prefer_adj_close: bool = True
) -> pd.DataFrame:
    """
    Lee un CSV OHLCV y devuelve un DataFrame con index datetime tz-aware (UTC por defecto)
    y columnas: ['open','high','low','close','volume'].

    - Detecta alias de columnas comunes.
    - Auto-detecta unidad del timestamp (s/ms/ns) o parsea fechas de string.
    - Convierte columnas a numérico, limpia duplicados y ordena.
    """

    # 1) Carga
    df = pd.read_csv(path, sep=sep, encoding=encoding)

    if df.empty:
        raise ValueError("CSV vacío.")

    # 2) Normalización de nombres de columna
    norm = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=norm)

    cols_found = list(df.columns)

    # 3) Alias a canon
    aliases: Dict[str, Iterable[str]] = {
        "timestamp": ("timestamp", "time", "ts", "datetime", "date"),
        "open":      ("open", "o"),
        "high":      ("high", "h"),
        "low":       ("low", "l"),
        "close":     ("close", "c", "adj_close", "adjclose", "close_adj"),
        "volume":    ("volume", "vol", "base_volume"),
    }

    def pick(name: str) -> Optional[str]:
        for a in aliases[name]:
            if a in df.columns:
                return a
        return None

    ts_col = pick("timestamp")
    o_col  = pick("open")
    h_col  = pick("high")
    l_col  = pick("low")
    c_col  = pick("close")
    v_col  = pick("volume")

    # Preferir Adj Close si se pidió y está presente
    if prefer_adj_close and "adj_close" in df.columns:
        c_col = "adj_close"

    missing = [n for n, c in [("timestamp", ts_col), ("open", o_col), ("high", h_col),
                              ("low", l_col), ("close", c_col), ("volume", v_col)] if c is None]
    if missing:
        raise ValueError(f"Faltan columnas OHLCV: {missing}. Columnas disponibles: {cols_found}")

    # 4) Parseo de timestamp
    ts_series = df[ts_col]

    # Detectar si es numérico
    if pd.api.types.is_numeric_dtype(ts_series):
        unit = assume_unit
        if unit is None:
            # Heurística por magnitud
            mx = float(pd.to_numeric(ts_series, errors="coerce").dropna().max())
            if mx > 1e18:
                unit = "ns"
            elif mx > 1e12:
                unit = "ms"
            elif mx > 1e9:
                # podría ser ns con valores chicos, pero lo usual es s
                unit = "s"
            else:
                unit = "s"
        dt = pd.to_datetime(ts_series, unit=unit, utc=True, errors="coerce")
    else:
        # String → que pandas parsee, y lo volvemos UTC
        dt = pd.to_datetime(ts_series, utc=True, errors="coerce")

    if tz and tz.upper() != "UTC":
        dt = dt.dt.tz_convert(tz)

    # 5) Coerción a numérico
    num_cols = {
        "open":  pd.to_numeric(df[o_col], errors="coerce"),
        "high":  pd.to_numeric(df[h_col], errors="coerce"),
        "low":   pd.to_numeric(df[l_col], errors="coerce"),
        "close": pd.to_numeric(df[c_col], errors="coerce"),
        "volume": pd.to_numeric(df[v_col], errors="coerce"),
    }

    out = pd.DataFrame({"timestamp": dt, **num_cols}).dropna(subset=["timestamp", "open", "high", "low", "close"])

    # 6) Index, orden y duplicados
    out = out.set_index("timestamp")
    out = out[["open", "high", "low", "close", "volume"]]
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    # 7) (Opcional) validaciones suaves
    bad_range = out["high"] < out["low"]
    if bad_range.any():
        # no frenamos, pero marcamos
        out = out[~bad_range]

    return out
