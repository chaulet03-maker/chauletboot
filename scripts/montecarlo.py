import pandas as pd, numpy as np, argparse, os, json
def main():
    p=argparse.ArgumentParser()
    p.add_argument("results_csv")
    p.add_argument("--runs", type=int, default=1000)
    args=p.parse_args()
    df=pd.read_csv(args.results_csv)
    rets=df["pnl"].dropna().values
    if len(rets)==0: print("No hay pnl en CSV"); return
    stats=[]
    for _ in range(args.runs):
        s=np.random.choice(rets, size=len(rets), replace=True)
        eq=1000+np.cumsum(s)
        dd=np.min(eq-np.maximum.accumulate(eq))
        stats.append({"final":eq[-1],"maxDD":dd})
    out=pd.DataFrame(stats)
    os.makedirs("data/metrics", exist_ok=True)
    path="data/metrics/montecarlo.json"
    out.to_json(path, orient="records")
    print("Monte Carlo:", path)
if __name__=="__main__":
    main()
