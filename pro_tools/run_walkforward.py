
from __future__ import annotations
import argparse, os
from .data_ccxt import load_csv
from .walkforward import walk_forward

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--train_months", type=int, default=6)
    ap.add_argument("--test_months", type=int, default=1)
    args = ap.parse_args()
    df = load_csv(args.csv)
    rep = walk_forward(df, args.train_months, args.test_months)
    os.makedirs("wf_out", exist_ok=True)
    rep.to_csv("wf_out/walkforward_report.csv", index=False)
    print("WF listo: wf_out/walkforward_report.csv")

if __name__ == "__main__":
    main()
