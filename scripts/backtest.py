import argparse, os, pandas as pd, numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def compute_indicators(df, conf):
    c=df['close']; h=df['high']; l=df['low']
    df['ema_fast']=EMAIndicator(c,window=conf.get('ema_fast',21)).ema_indicator()
    df['ema_slow']=EMAIndicator(c,window=conf.get('ema_slow',55)).ema_indicator()
    df['rsi']=RSIIndicator(c,window=conf.get('rsi_period',14)).rsi()
    macd=MACD(c,window_fast=conf.get('macd_fast',12),window_slow=conf.get('macd_slow',26),window_sign=conf.get('macd_signal',9))
    df['macd']=macd.macd(); df['macd_signal']=macd.macd_signal()
    bb=BollingerBands(c,window=conf.get('bb_period',20),window_dev=conf.get('bb_std',2))
    df['bb_width_bps']=(bb.bollinger_hband()-bb.bollinger_lband())/c*10000
    df['adx']=ADXIndicator(h,l,c,window=14).adx()
    df['atr']=AverageTrueRange(h,l,c,window=conf.get('atr_period',14)).average_true_range()
    return df.dropna().reset_index(drop=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="OHLCV CSV con columnas timestamp,open,high,low,close,volume")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--fees", default="taker", choices=["taker","maker"])
    args=p.parse_args()
    os.makedirs("data/backtests", exist_ok=True)

    df=pd.read_csv(args.csv, parse_dates=["timestamp"])
    df=compute_indicators(df, {"ema_fast":21,"ema_slow":55,"rsi_period":14,"macd_fast":12,"macd_slow":26,"macd_signal":9,"bb_period":20,"bb_std":2,"atr_period":14})
    bal=1000.0; fee=0.0005 if args.fees=="taker" else 0.0002
    risk=0.01; stop_mult=2.0; tp1r=1.0; tp2r=2.0

    open_pos=None; rows=[]
    for i in range(2,len(df)):
        r=df.iloc[i]; p1=df.iloc[i-1]
        # simple signal (igual que runtime)
        long_ok = r.ema_fast>r.ema_slow and (r.macd-r.macd_signal)>(p1.macd-p1.macd_signal) and r.rsi>52 and r.adx>=15 and r.bb_width_bps>=12
        short_ok= r.ema_fast<r.ema_slow and (r.macd-r.macd_signal)<(p1.macd-p1.macd_signal) and r.rsi<48 and r.adx>=15 and r.bb_width_bps>=12

        if open_pos is None:
            if long_ok or short_ok:
                side="long" if long_ok else "short"
                price=r.close; atr=r.atr
                risk_usdt=bal*risk; stop_dist=atr*stop_mult
                qty= risk_usdt/stop_dist
                notional=qty*price; bal -= notional*fee
                sl= price - stop_dist if side=="long" else price + stop_dist
                tp1= price + (atr*tp1r if side=="long" else -atr*tp1r)
                tp2= price + (atr*tp2r if side=="long" else -atr*tp2r)
                open_pos={"side":side,"qty":qty,"entry":price,"sl":sl,"tp1":tp1,"tp2":tp2}
                rows.append({"timestamp":r.timestamp,"action":"OPEN","price":price})
        else:
            price=r.close
            if (open_pos["side"]=="long" and price<=open_pos["sl"]) or (open_pos["side"]=="short" and price>=open_pos["sl"]):
                pnl=(price-open_pos["entry"])*open_pos["qty"]*(1 if open_pos["side"]=="long" else -1)
                bal += pnl - (open_pos["qty"]*price*fee)
                rows.append({"timestamp":r.timestamp,"action":"STOP","price":price,"pnl":pnl})
                open_pos=None
            elif (open_pos["side"]=="long" and price>=open_pos["tp1"]) or (open_pos["side"]=="short" and price<=open_pos["tp1"]):
                pnl=(price-open_pos["entry"])*open_pos["qty"]*(1 if open_pos["side"]=="long" else -1)
                bal += pnl - (open_pos["qty"]*price*fee)
                rows.append({"timestamp":r.timestamp,"action":"TP1","price":price,"pnl":pnl})
                open_pos=None

    out=pd.DataFrame(rows)
    out_path=f"data/backtests/{args.symbol.replace('/','')}_results.csv"
    out.to_csv(out_path, index=False)
    print("Backtest listo:", out_path)

if __name__=="__main__":
    main()
