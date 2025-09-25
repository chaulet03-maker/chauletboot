
from __future__ import annotations
import pandas as pd, numpy as np, math
from .strategy_cots import compute_cots_score

def walk_forward(df: pd.DataFrame, train_months: int = 6, test_months: int = 1, threshold: float = 0.65) -> pd.DataFrame:
    df = df.copy().sort_index()
    monthly = df.resample("MS").first().index
    rows = []
    for i in range(0, len(monthly)-(train_months+test_months)):
        train_end = monthly[i+train_months]
        test_end  = monthly[i+train_months+test_months]
        test_df = df.loc[train_end:test_end - pd.Timedelta(milliseconds=1)]
        sig = compute_cots_score(test_df, atr_htf_window=48)
        longs = (sig["cots"]>=threshold) & (sig["long_ok"]==1)
        shorts= (sig["cots"]>=threshold) & (sig["short_ok"]==1)
        # naive PnL: next bar close-close
        pnl = 0.0; trades=0; wins=0; res=[]
        for ts, ok in longs.items():
            nxt = test_df.index.get_loc(ts)+1
            if nxt < len(test_df.index):
                ret = test_df.iloc[nxt]["close"] - test_df.iloc[nxt]["close"].shift(0)
        rows.append({"train_end":str(train_end.date()),"test_end":str((test_end-pd.Timedelta(days=1)).date()),
                     "trades": trades, "win_rate": (wins/max(1,trades)) if trades else 0.0, "pnl": pnl})
    return pd.DataFrame(rows)
