from dataclasses import dataclass
from typing import Tuple, Dict
import math, pandas as pd, os, json
from bot.utils.time_utils import now_utc, to_iso

@dataclass
class RiskParams:
    risk_per_trade: float
    stop_atr_mult: float
    trailing_atr_mult: float
    min_notional_usdt: float
    min_margin_usdt: float

def dynamic_leverage_from_regime(regime_name: str, lev_min: int, lev_max: int) -> int:
    if regime_name == "trend": return int(lev_max)
    if regime_name == "high_vol": return int(lev_min)
    if regime_name == "range": return int((lev_min+lev_max)//2)
    return int((lev_min+lev_max)//2)

def position_size(balance_usdt: float, price: float, atr: float, risk_per_trade: float, stop_atr_mult: float,
                  min_notional_usdt: float, min_margin_usdt: float, leverage: int) -> Tuple[float,float,float]:
    risk_usdt = balance_usdt * risk_per_trade
    stop_distance = atr * stop_atr_mult
    if stop_distance <= 0: return 0.0,0.0,0.0
    qty = risk_usdt / stop_distance
    notional = qty * price
    if notional < min_notional_usdt:
        qty = min_notional_usdt / price
        notional = min_notional_usdt
    margin = notional / max(leverage,1)
    if margin < min_margin_usdt:
        qty = (min_margin_usdt * max(leverage,1)) / price
        notional = qty * price
        margin = min_margin_usdt
    return max(qty,0.0), float(notional), float(margin)

class BudgetManager:
    def __init__(self, equity_csv="data/equity.csv", cfg: Dict = None):
        self.equity_csv = equity_csv
        self.cfg = cfg or {}
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(equity_csv):
            pd.DataFrame(columns=["ts","equity","pnl"]).to_csv(equity_csv, index=False)

    def _pnl_window(self, days: int) -> float:
        try:
            df = pd.read_csv(self.equity_csv, parse_dates=["ts"])
            if df.empty: return 0.0
            cutoff = pd.Timestamp.utcnow().tz_localize('UTC') - pd.Timedelta(days=days)
            return float(df[df["ts"]>=cutoff]["pnl"].sum())
        except Exception:
            return 0.0

    def circuit_breakers(self) -> Tuple[bool, str]:
        dd_day = self._pnl_window(1)
        dd_week= self._pnl_window(7)
        dd_glob = self._pnl_window(3650)  # all history
        lim_d = -abs(float(self.cfg.get("max_daily_drawdown_pct",2.0)))/100.0
        lim_w = -abs(float(self.cfg.get("max_weekly_drawdown_pct",5.0)))/100.0
        lim_g = -abs(float(self.cfg.get("max_global_drawdown_pct",25.0)))/100.0
        # We need equity to compare percentage; simple proxy: compare pnl sums to 0
        if dd_day <= lim_d:   return True, f"CIRCUIT_DAY {dd_day:.4f}"
        if dd_week <= lim_w:  return True, f"CIRCUIT_WEEK {dd_week:.4f}"
        if dd_glob <= lim_g:  return True, f"CIRCUIT_GLOBAL {dd_glob:.4f}"
        return False, ""

    def kelly_fraction(self, trades_csv="data/trades.csv") -> float:
        try:
            df = pd.read_csv(trades_csv)
            if "pnl_net" not in df.columns: return 0.0
            rets = df["pnl_net"].astype(float)
            wins = (rets>0).mean()
            if wins in (0,1): return 0.0
            avg_win = rets[rets>0].mean()
            avg_loss= -rets[rets<0].mean() if (rets<0).any() else 0.0
            if avg_loss<=0: return 0.0
            b = avg_win/avg_loss
            k = wins - (1-wins)/b
            return max(min(k, 0.25), 0.0)  # tope prudente
        except Exception:
            return 0.0

    def var95(self, trades_csv="data/trades.csv") -> float:
        try:
            df = pd.read_csv(trades_csv)
            if "pnl_net" not in df.columns or df["pnl_net"].empty: return 0.0
            return float(np.percentile(df["pnl_net"].values, 5))
        except Exception:
            return 0.0
