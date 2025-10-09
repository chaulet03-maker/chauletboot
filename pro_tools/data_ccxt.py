
from __future__ import annotations
import pandas as pd
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower():c for c in df.columns}
    for k in ["timestamp","open","high","low","close","volume"]:
        assert any(k==c.lower() for c in df.columns), f"CSV missing {k}"
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", unit="ms").fillna(pd.to_datetime(df["timestamp"], errors="coerce"))
    elif "date" in cols:
        df["timestamp"] = pd.to_datetime(df[cols["date"]])
    df = df[["timestamp","open","high","low","close","volume"]].dropna()
    df = df.set_index("timestamp").sort_index()
    return df
