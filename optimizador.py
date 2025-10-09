#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools
import math
import os
import random
import time
import pandas as pd
import numpy as np

# Importa tu simulador
from simulador_final import Backtester, leer_archivo_smart

# ---------- Métricas / Objetivo ----------
def score_result(res):
    """
    Función objetivo:
      - base = balance_final
      - penalización por max drawdown (mdd) y por payoff bajo
    """
    balance_final = res["balance_final"]
    mdd = res["max_drawdown"]
    payoff = res.get("payoff", 1.0)
    winrate = res.get("winrate", 0.5)

    # Penalty por drawdown (no lineal; >30% duele fuerte)
    dd_penalty = 1.0 / (1.0 + 5.0 * max(0.0, mdd - 0.30))
    # Premio leve a payoff y winrate
    payoff_boost = min(payoff, 3.0) / 2.0
    wr_boost = (winrate / 100.0 + 0.5) / 1.5

    return balance_final * dd_penalty * payoff_boost * wr_boost

# ---------- Search Spaces ----------
def default_search_space():
    """
    Devuelve listas para grid + valores para random.
    Ajustá si querés buscar más/menos.
    """
    grid = {
        "modes": [
            {"TENDENCIA"},
            {"RANGO"},
            {"TENDENCIA", "RANGO"},
        ],
        "risk_pct": [0.005, 0.01],
        "atr_k_sl": [1.6, 1.8, 2.0],
        "tp1_atr": [0.8, 1.0, 1.2],
        "tp2_atr": [1.8, 2.2, 2.6],
        "trail_atr_k": [1.2, 1.5],
        "sl_pct": [None, 0.30],  # None => SL por ATR; 0.30 => SL porcentual 30%
    }
    # Solo aplica si RANGO está en modes
    grid_range = {
        "bbw_max_range": [0.008, 0.010, 0.012],  # 0.8%-1.2%
        "adx_max_range": [16.0, 18.0],
        "tstop_bars_range": [12, 24],
    }
    # Solo aplica si TENDENCIA está en modes
    grid_trend = {
        "tstop_bars_trend": [36, 48, 72],
    }

    # Random bounds (si querés explorar un poco)
    random_bounds = {
        "risk_pct": (0.003, 0.015),
        "atr_k_sl": (1.4, 2.2),
        "tp1_atr": (0.6, 1.4),
        "tp2_atr": (1.6, 3.0),
        "trail_atr_k": (1.0, 2.0),
        "bbw_max_range": (0.006, 0.015),
        "adx_max_range": (14.0, 20.0),
        "tstop_bars_range": (8, 36),
        "tstop_bars_trend": (24, 96),
    }

    return grid, grid_range, grid_trend, random_bounds

def expand_configs(grid, grid_range, grid_trend, n_random=30, seed=1337):
    """
    Crea configuraciones a evaluar: primero grid determinístico, después random.
    """
    random.seed(seed)
    base_keys = ["modes", "risk_pct", "atr_k_sl", "tp1_atr", "tp2_atr", "trail_atr_k", "sl_pct"]
    base_vals = [grid[k] for k in base_keys]
    base_grid = [dict(zip(base_keys, vals)) for vals in itertools.product(*base_vals)]

    configs = []
    for cfg in base_grid:
        cfg_full = cfg.copy()
        if "RANGO" in cfg["modes"]:
            # meter defaults de rango
            for k, vals in grid_range.items():
                cfg_full[k] = vals[0]  # se ajustará luego en product
        if "TENDENCIA" in cfg["modes"]:
            for k, vals in grid_trend.items():
                cfg_full[k] = vals[0]
        configs.append(cfg_full)

    # Expandir combinaciones específicas por modo
    expanded = []
    for cfg in configs:
        blocks = []
        if "RANGO" in cfg["modes"]:
            blocks.append(grid_range)
        if "TENDENCIA" in cfg["modes"]:
            blocks.append(grid_trend)

        if blocks:
            # combinamos los dicts de blocks
            keys = []
            vals = []
            for b in blocks:
                for k, v in b.items():
                    keys.append(k)
                    vals.append(v)
            for tail in itertools.product(*vals):
                newc = cfg.copy()
                for k, v in zip(keys, tail):
                    newc[k] = v
                expanded.append(newc)
        else:
            expanded.append(cfg)

    # Agregar algunas random configs
    _, _, _, rb = default_search_space()
    for _ in range(n_random):
        rcfg = {}
        # elegir modos al azar (1 ó 2)
        modes_choices = [
            {"TENDENCIA"},
            {"RANGO"},
            {"TENDENCIA", "RANGO"},
        ]
        rcfg["modes"] = random.choice(modes_choices)
        # muestrear float en rango
        def frange(lo, hi):
            return lo + random.random() * (hi - lo)
        rcfg["risk_pct"] = round(frange(*rb["risk_pct"]), 4)
        rcfg["atr_k_sl"] = round(frange(*rb["atr_k_sl"]), 2)
        rcfg["tp1_atr"] = round(frange(*rb["tp1_atr"]), 2)
        rcfg["tp2_atr"] = round(frange(*rb["tp2_atr"]), 2)
        rcfg["trail_atr_k"] = round(frange(*rb["trail_atr_k"]), 2)
        rcfg["sl_pct"] = random.choice([None, 0.25, 0.30, 0.35])

        if "RANGO" in rcfg["modes"]:
            rcfg["bbw_max_range"] = round(frange(*rb["bbw_max_range"]), 4)
            rcfg["adx_max_range"] = round(frange(*rb["adx_max_range"]), 1)
            rcfg["tstop_bars_range"] = int(frange(*rb["tstop_bars_range"]))
        if "TENDENCIA" in rcfg["modes"]:
            rcfg["tstop_bars_trend"] = int(frange(*rb["tstop_bars_trend"]))

        expanded.append(rcfg)

    return expanded

# ---------- Runner ----------
def run_backtest(df_1h, df_4h, cfg):
    bt = Backtester(
        df_1h=df_1h,
        df_4h=df_4h,
        initial_balance=1000.0,
        fee_pct=0.0005,
        modes=cfg["modes"],
        risk_pct=cfg["risk_pct"],
        atr_k_sl=cfg["atr_k_sl"],
        tp1_atr=cfg["tp1_atr"],
        tp2_atr=cfg["tp2_atr"],
        trail_atr_k=cfg["trail_atr_k"],
        trail_hysteresis=0.05,
        sl_pct=cfg.get("sl_pct", None),
        bbw_max_range=cfg.get("bbw_max_range", 0.012),
        adx_max_range=cfg.get("adx_max_range", 18.0),
        tstop_bars_range=cfg.get("tstop_bars_range", 24),
        tstop_bars_trend=cfg.get("tstop_bars_trend", 48),
    )
    bt.run()

    # Leer outputs en memoria (las guardas siguen saliendo a /data/backtests)
    trades = pd.DataFrame(bt.trades)
    equity = pd.Series(bt.equity, name="equity")

    # Métricas principales
    closes = trades[trades["action"] == "CLOSE"] if not trades.empty else pd.DataFrame()
    total_pnl = float(closes["pnl"].sum()) if not closes.empty else 0.0
    balance_final = float(bt.balance)
    wins = int((closes["pnl"] > 0).sum()) if not closes.empty else 0
    losses = int((closes["pnl"] <= 0).sum()) if not closes.empty else 0
    winrate = (wins / max(1, wins + losses)) * 100.0
    avg_win = float(closes.loc[closes["pnl"] > 0, "pnl"].mean()) if wins else 0.0
    avg_loss = float(closes.loc[closes["pnl"] <= 0, "pnl"].mean()) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("nan")

    roll_max = equity.cummax() if len(equity) else pd.Series([1.0])
    dd = equity / roll_max - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0

    modes_str = ",".join(sorted(cfg["modes"]))
    result = {
        "modes": modes_str,
        "risk_pct": cfg["risk_pct"],
        "atr_k_sl": cfg["atr_k_sl"],
        "tp1_atr": cfg["tp1_atr"],
        "tp2_atr": cfg["tp2_atr"],
        "trail_atr_k": cfg["trail_atr_k"],
        "sl_pct": cfg.get("sl_pct", None),
        "bbw_max_range": cfg.get("bbw_max_range", None),
        "adx_max_range": cfg.get("adx_max_range", None),
        "tstop_bars_range": cfg.get("tstop_bars_range", None),
        "tstop_bars_trend": cfg.get("tstop_bars_trend", None),
        "pnl": total_pnl,
        "balance_final": balance_final,
        "winrate": winrate,
        "payoff": payoff,
        "max_drawdown": mdd,
        "trades": int(len(closes)),
    }
    result["score"] = score_result(result)
    return result

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("Optimizador de estrategias (1h + 4h)")
    ap.add_argument("--csv1h", required=True, help="Ruta 1h (CSV/Excel o sin extensión).")
    ap.add_argument("--csv4h", required=True, help="Ruta 4h (CSV/Excel o sin extensión).")
    ap.add_argument("--n-random", type=int, default=30, help="Cantidad de configuraciones aleatorias extra.")
    ap.add_argument("--top-k", type=int, default=20, help="Guardar top-K en CSV.")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # Carga de data
    df_1h = leer_archivo_smart(args.csv1h)
    df_4h = leer_archivo_smart(args.csv4h)

    # Preparar configs
    grid, grid_range, grid_trend, _ = default_search_space()
    configs = expand_configs(grid, grid_range, grid_trend, n_random=args.n_random, seed=args.seed)

    print(f"Probando {len(configs)} configuraciones...")
    results = []
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        res = run_backtest(df_1h, df_4h, cfg)
        results.append(res)
        if i % 10 == 0:
            dt = time.time() - t0
            print(f"[{i}/{len(configs)}] Tiempo {dt:.1f}s | Mejor score hasta ahora: {max(r['score'] for r in results):.2f}")

    # Ranking
    df = pd.DataFrame(results)
    df.sort_values("score", ascending=False, inplace=True)

    os.makedirs("data/opt", exist_ok=True)
    out_csv = "data/opt/top_configs.csv"
    df.head(args.top_k).to_csv(out_csv, index=False)

    best = df.iloc[0].to_dict()
    print("\n=== MEJOR CONFIGURACIÓN ===")
    for k in ["modes", "risk_pct", "atr_k_sl", "tp1_atr", "tp2_atr", "trail_atr_k", "sl_pct",
              "bbw_max_range", "adx_max_range", "tstop_bars_range", "tstop_bars_trend",
              "balance_final", "winrate", "payoff", "max_drawdown", "trades", "score"]:
        if k in best and pd.notna(best[k]):
            print(f"{k}: {best[k]}")

    # Comando sugerido
    modes_cli = best["modes"]
    parts = [
        f'--csv1h "{args.csv1h}"',
        f'--csv4h "{args.csv4h}"',
        f'--modes {modes_cli}',
        f'--risk-pct {best["risk_pct"]}',
        f'--atr-k-sl {best["atr_k_sl"]}',
        f'--tp1-atr {best["tp1_atr"]}',
        f'--tp2-atr {best["tp2_atr"]}',
        f'--trail-atr-k {best["trail_atr_k"]}',
    ]
    if not pd.isna(best.get("sl_pct", np.nan)):
        parts.append(f'--sl-pct {best["sl_pct"]}')
    if "RANGO" in modes_cli:
        if not pd.isna(best.get("bbw_max_range", np.nan)):
            parts.append(f'--bbw-max-range {best["bbw_max_range"]}')
        if not pd.isna(best.get("adx_max_range", np.nan)):
            parts.append(f'--adx-max-range {best["adx_max_range"]}')
        if not pd.isna(best.get("tstop_bars_range", np.nan)):
            parts.append(f'--tstop-bars-range {int(best["tstop_bars_range"])}')
    if "TENDENCIA" in modes_cli:
        if not pd.isna(best.get("tstop_bars_trend", np.nan)):
            parts.append(f'--tstop-bars-trend {int(best["tstop_bars_trend"])}')

    cli = "python .\\simulador_final.py " + " ".join(parts)
    print("\nComando sugerido para reproducir el mejor resultado:\n" + cli)
    print(f"\nTop {args.top_k} configs guardadas en: {out_csv}")

if __name__ == "__main__":
    main()
