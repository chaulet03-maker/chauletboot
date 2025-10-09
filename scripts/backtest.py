#!/usr/bin/env python
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

def load_ohlcv(csv_path: str):
    df = pd.read_csv(csv_path)
    # Esperado: timestamp,open,high,low,close,volume
    for c in ["timestamp","open","high","low","close","volume"]:
        if c not in df.columns:
            raise ValueError(f"CSV sin columna {c}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def calc_metrics(trades_df: pd.DataFrame):
    if trades_df.empty:
        return {
            "trades": 0, "winrate": 0.0, "net_profit": 0.0,
            "profit_factor": 0.0, "expectancy_per_trade": 0.0,
            "max_drawdown": 0.0, "cagr": None, "mar": None, "sharpe": 0.0
        }
    rets = trades_df["pnl"].astype(float)
    # equity en 1.0 base si no existe
    if "equity" not in trades_df.columns:
        equity = 1.0 + rets.cumsum()
    else:
        equity = trades_df["equity"].astype(float)
    gross_profit = rets[rets > 0].sum()
    gross_loss = -rets[rets < 0].sum()
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = float(rets.mean())
    # max drawdown sobre equity
    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = float(abs(dd.min())) if len(dd) else 0.0
    # CAGR si hay fechas
    cagr = None
    if "timestamp" in trades_df.columns and trades_df["timestamp"].notna().all():
        t0, t1 = trades_df["timestamp"].iloc[0], trades_df["timestamp"].iloc[-1]
        years = max(1e-9, (t1 - t0).days / 365.25)
        cagr = float((equity.iloc[-1] / equity.iloc[0])**(1/years) - 1) if years > 0 else None
    mar = float(cagr / mdd) if (cagr is not None and mdd > 0) else None
    # Sharpe simple (asumiendo rets por trade)
    sh = 0.0
    if rets.std() != 0:
        sh = float(np.sqrt(252) * (rets.mean()) / rets.std())
    return {
        "trades": int(len(trades_df)),
        "winrate": float((rets > 0).mean()),
        "net_profit": float(rets.sum()),
        "profit_factor": float(pf),
        "expectancy_per_trade": float(expectancy),
        "max_drawdown": mdd,
        "cagr": cagr,
        "mar": mar,
        "sharpe": float(sh),
    }

def run_backtest(df: pd.DataFrame, symbol: str, fee_bps: int, slippage_bps: int):
    """
    Ejemplo básico:
    - Entradas por cruce EMA(9) > EMA(20) (LONG) y viceversa (SHORT)
    - TP/SL fijos razonables para demo
    - pnl expresado como retorno fraccional por trade (ya neteado fee/slippage)
    """
    close = df["close"].astype(float).values
    ts = df["timestamp"].values
    ema_fast = pd.Series(close).ewm(span=9, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=20, adjust=False).mean().values

    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0

    pos = 0  # 0 none, 1 long, -1 short
    entry = None
    trades = []
    equity = 1.0

    TP = 0.008  # 0.8%
    SL = 0.006  # 0.6%

    for i in range(1, len(close)):
        # señales
        long_sig = ema_fast[i-1] <= ema_slow[i-1] and ema_fast[i] > ema_slow[i]
        short_sig= ema_fast[i-1] >= ema_slow[i-1] and ema_fast[i] < ema_slow[i]

        price = close[i]
        if pos == 0:
            if long_sig:
                pos = 1
                entry = price*(1+slip)  # slippage entrar
            elif short_sig:
                pos = -1
                entry = price*(1-slip)
        else:
            # check TP/SL
            if pos == 1:
                rr = (price - entry)/entry
                tp_hit = rr >= TP
                sl_hit = rr <= -SL
                exit_price = price*(1-slip)
                if tp_hit or sl_hit or short_sig:  # cierre por señal contraria también
                    gross = (exit_price - entry)/entry
                    net = gross - 2*fee  # entrada+salida
                    equity *= (1.0 + net)
                    trades.append({"timestamp": df["timestamp"].iloc[i], "side":"LONG","entry":float(entry),"exit":float(exit_price),"pnl":float(net),"equity":float(equity)})
                    pos = 0; entry = None
            else:  # short
                rr = (entry - price)/entry
                tp_hit = rr >= TP
                sl_hit = rr <= -SL
                exit_price = price*(1+slip)
                if tp_hit or sl_hit or long_sig:
                    gross = (entry - exit_price)/entry
                    net = gross - 2*fee
                    equity *= (1.0 + net)
                    trades.append({"timestamp": df["timestamp"].iloc[i], "side":"SHORT","entry":float(entry),"exit":float(exit_price),"pnl":float(net),"equity":float(equity)})
                    pos = 0; entry = None

    trades_df = pd.DataFrame(trades)
    return trades_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--config", required=False)  # lo dejo por compatibilidad, no lo uso acá
    ap.add_argument("--fee_bps", type=int, default=2, help="fee en bps (default 2)")
    ap.add_argument("--slippage_bps", type=int, default=5, help="slippage en bps (default 5)")
    args = ap.parse_args()

    df = load_ohlcv(args.csv)
    trades = run_backtest(df, args.symbol, args.fee_bps, args.slippage_bps)

    out_dir = Path("data/backtests")/args.symbol.replace("/","_")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir/"trades.csv"
    trades.to_csv(trades_path, index=False)

    summary = calc_metrics(trades)
    with open(out_dir/"summary.json","w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[OK] Trades: {trades_path}")
    print(f"[OK] Summary: {out_dir/'summary.json'}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
