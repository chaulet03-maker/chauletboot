import argparse
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

# --- Importa lógica del bot (paridad total) ---
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bot.core.indicators import compute_indicators
from bot.core.strategy import generate_signal
import yaml

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class Backtester:
    def __init__(self, df: pd.DataFrame, symbol: str, config: dict):
        self.df = compute_indicators(df, config["strategy"])
        self.symbol = symbol
        self.balance = config["risk"]["equity_usdt"]
        self.risk_pct = config["risk"]["max_risk_per_trade_pct"]
        self.fee = config["fees"]["taker"]
        self.slippage_bps = config["execution"]["slippage_bps"]
        self.trades = []
        self.open_position = None
        logging.info(f"Backtester {symbol} inicializado. Balance inicial: ${self.balance:.2f}")

    def _size_position(self, entry, stop):
        risk_usd = self.balance * self.risk_pct
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return 0
        return risk_usd / risk_per_unit

    def _slip(self, price, side):
        slip = price * self.slippage_bps / 10000
        return price + slip if side == "LONG" else price - slip

    def _log_trade(self, ts, action, side, price, qty, pnl=0.0):
        self.trades.append({
            "timestamp": ts,
            "symbol": self.symbol,
            "action": action,
            "side": side,
            "price": price,
            "qty": qty,
            "pnl": pnl,
            "balance": self.balance
        })

    def run(self):
        for i in range(50, len(self.df)):  # arranca con historial suficiente
            row = self.df.iloc[: i + 1]
            last = row.iloc[-1]

            # gestion de posicion abierta
            if self.open_position:
                self._manage(last)

            # buscar señal nueva
            if not self.open_position:
                sig = generate_signal(row, {})
                if sig.side in ["long", "short"]:
                    entry = self._slip(last.close, "LONG" if sig.side == "long" else "SHORT")
                    sl, tp1, tp2 = sig.sl, sig.tp1, sig.tp2
                    qty = self._size_position(entry, sl)
                    if qty > 0:
                        self.open_position = {
                            "side": sig.side.upper(),
                            "qty": qty,
                            "entry": entry,
                            "sl": sl,
                            "tp1": tp1,
                            "tp2": tp2,
                            "tp1_hit": False,
                        }
                        logging.info(f"{last.timestamp} | OPEN {sig.side.upper()} @ {entry:.2f}")
                        self._log_trade(last.timestamp, "OPEN", sig.side.upper(), entry, qty)

        return self._report()

    def _manage(self, r):
        pos = self.open_position
        if not pos:
            return
        price = r.close
        is_long = pos["side"] == "LONG"
        reason = None

        # TP1
        if not pos["tp1_hit"] and ((is_long and price >= pos["tp1"]) or (not is_long and price <= pos["tp1"])):
            half_qty = pos["qty"] / 2
            pnl = (price - pos["entry"]) * half_qty * (1 if is_long else -1)
            self.balance += pnl - (half_qty * price * self.fee)
            pos["qty"] = half_qty
            pos["tp1_hit"] = True
            pos["sl"] = pos["entry"]  # break-even
            self._log_trade(r.timestamp, "TP1", pos["side"], price, half_qty, pnl)

        # TP2 o SL
        if (is_long and price <= pos["sl"]) or (not is_long and price >= pos["sl"]):
            reason = "STOP"
        elif pos["tp2"] and ((is_long and price >= pos["tp2"]) or (not is_long and price <= pos["tp2"])):
            reason = "TP2"

        if reason:
            pnl = (price - pos["entry"]) * pos["qty"] * (1 if is_long else -1)
            self.balance += pnl - (pos["qty"] * price * self.fee)
            self._log_trade(r.timestamp, reason, pos["side"], price, pos["qty"], pnl)
            self.open_position = None

    def _report(self):
        df = pd.DataFrame(self.trades)
        if df.empty:
            print("No hubo trades.")
            return df

        total_pnl = df["pnl"].sum()
        wins = df[df["pnl"] > 0]; losses = df[df["pnl"] < 0]
        win_rate = len(wins) / len(df[df["action"] != "OPEN"]) * 100 if len(df) > 0 else 0
        maxdd = (df["balance"].cummax() - df["balance"]).max()
        sharpe = (df["pnl"].mean() / df["pnl"].std() * np.sqrt(252)) if df["pnl"].std() > 0 else 0

        print("\n=== REPORTE FINAL ===")
        print(f"Balance final: ${self.balance:.2f}")
        print(f"PNL neto: ${total_pnl:.2f}")
        print(f"Operaciones: {len(df[df['action'] != 'OPEN'])}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Max Drawdown: ${maxdd:.2f}")
        print(f"Sharpe: {sharpe:.2f}")

        out_path = f"data/backtests/backtest_{self.symbol}.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Resultados guardados en {out_path}")
        return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ruta al CSV OHLCV (1 símbolo, timeframe config.yaml).")
    parser.add_argument("--symbol", required=True, help="Símbolo (ej: BTC/USDT:USDT).")
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    config = load_config(args.config)

    bt = Backtester(df, args.symbol, config)
    bt.run()

if __name__ == "__main__":
    main()
