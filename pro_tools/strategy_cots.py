
from __future__ import annotations
import numpy as np, pandas as pd

def _sigm(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-k * x))

def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def compute_cots_score(df: pd.DataFrame, atr_htf_window: int = 48, lookback_open: int = 3,
                       weights: dict | None = None, extras: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    atr = compute_atr(df, n=atr_htf_window)
    df["mid_ret"] = (df["close"] - df["open"]) / atr.replace(0, np.nan)
    df["follow"]  = (np.maximum(df["high"]-df["open"], df["open"]-df["low"])) / atr.replace(0, np.nan)
    vol = df["volume"].fillna(0)
    vol_chg = (vol - vol.rolling(lookback_open, min_periods=1).mean()) / (vol.rolling(lookback_open, min_periods=1).std() + 1e-9)
    df["delta_vol_norm"] = vol_chg.clip(-5, 5)

    if extras is not None:
        e = extras.reindex(df.index).fillna(method="ffill").fillna(0.0)
        obi = e.get("obi", pd.Series(0.0, index=df.index))
        spread_stab = e.get("spread_stability", pd.Series(0.0, index=df.index))
        impact = e.get("impact_proxy", pd.Series(0.0, index=df.index))
    else:
        obi = pd.Series(0.0, index=df.index)
        spread_stab = pd.Series(0.0, index=df.index)
        impact = pd.Series(0.0, index=df.index)

    w = {"delta_agg":0.25,"mid_ret":0.25,"obi":0.15,"spread_stab":0.15,"follow":0.20,"impact":0.20}
    if weights: w.update(weights)

    cots = (w["delta_agg"]*_sigm(df["delta_vol_norm"]) +
            w["mid_ret"]  *_sigm(df["mid_ret"]) +
            w["obi"]      *_sigm(obi) +
            w["spread_stab"]*_sigm(spread_stab) +
            w["follow"]   *_sigm(df["follow"]) -
            w["impact"]   *_sigm(impact))
    out = pd.DataFrame(index=df.index)
    out["cots"] = cots.clip(0,1)
    out["long_ok"]  = ((out["cots"]>=0.65) & (df["mid_ret"]>0)).astype(int)
    out["short_ok"] = ((out["cots"]>=0.65) & (df["mid_ret"]<0)).astype(int)
    return out
