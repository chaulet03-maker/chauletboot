import argparse
import os
import pandas as pd
import numpy as np
import logging

# --- Indicadores (los que ya tenías) ---
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_indicators_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores necesarios para un DataFrame de velas."""
    c = df['close']
    h = df['high']
    l = df['low']
    
    # Indicadores de Tendencia
    df['ema_fast'] = EMAIndicator(c, window=10).ema_indicator()
    df['ema_slow'] = EMAIndicator(c, window=30).ema_indicator()
    df['adx'] = ADXIndicator(h, l, c, window=14).adx()

    # Indicadores de Volatilidad y Momentum
    bb = BollingerBands(c, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['rsi'] = RSIIndicator(c, window=14).rsi()
    df['atr'] = AverageTrueRange(h, l, c, window=14).average_true_range()
    
    return df.dropna().reset_index(drop=True)

class Backtester:
    def __init__(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame, initial_balance: float, fee_pct: float):
        self.df_5m = compute_indicators_for_df(df_5m)
        self.df_1h = compute_indicators_for_df(df_1h)
        self.balance = initial_balance
        self.fee = fee_pct
        self.trades = []
        self.open_position = None
        logging.info(f"Backtester inicializado. Balance inicial: ${initial_balance:.2f}, Fee: {fee_pct*100:.2f}%")

    def _detectar_regimen(self, current_timestamp):
        """Analiza el mercado en 1h para decidir el régimen."""
        df_1h_hist = self.df_1h[self.df_1h['timestamp'] <= current_timestamp]
        if len(df_1h_hist) < 25:
            return "RANGO" # Por defecto, si no hay suficientes datos

        last_1h_row = df_1h_hist.iloc[-1]
        adx = last_1h_row['adx']
        bb_width = last_1h_row['bb_width']

        if adx > 23:
            return "TENDENCIA"
        if adx < 19 and bb_width < 0.08:
            return "RANGO"
        return "RANGO" # Si es ambiguo, se queda en modo seguro

    def _run_strategy_tendencia(self, r, p1):
        """Lógica para la estrategia de Tendencia (2% riesgo, Multi-TP)."""
        # (Esta es una señal de ejemplo, podés ajustarla a la tuya)
        long_ok = r.ema_fast > r.ema_slow and r.adx > 25
        short_ok = r.ema_fast < r.ema_slow and r.adx > 25
        
        if long_ok or short_ok:
            side = "LONG" if long_ok else "SHORT"
            entry_price = r.close
            stop_dist = r.atr * 2.0 # Stop Loss a 2x ATR
            
            # --- GESTIÓN DE RIESGO: 2% ---
            risk_usd = self.balance * 0.02
            qty = risk_usd / stop_dist
            
            sl_price = entry_price - stop_dist if side == "LONG" else entry_price + stop_dist
            tp1_price = entry_price + stop_dist * 1.0 # Ratio 1:1 para TP1
            tp2_price = entry_price + stop_dist * 2.0 # Ratio 2:1 para TP2
            
            self.open_position = {
                "side": side, "qty": qty, "entry_price": entry_price, 
                "sl": sl_price, "tp1": tp1_price, "tp2": tp2_price,
                "tp1_hit": False, "regime": "TENDENCIA"
            }
            logging.info(f"{r.timestamp} | TENDENCIA | OPEN {side} | Price: {entry_price:.2f} | Qty: {qty:.4f}")
            self._log_trade(r.timestamp, "OPEN", side, entry_price, qty, "TENDENCIA")

    def _run_strategy_rango(self, r, p1):
        """Lógica para la estrategia de Rango (1% riesgo, Scalping)."""
        long_ok = r.close <= r.bb_lower and r.rsi < 35
        short_ok = r.close >= r.bb_upper and r.rsi > 65
        
        if long_ok or short_ok:
            side = "LONG" if long_ok else "SHORT"
            entry_price = r.close
            stop_dist = r.atr * 1.5 # Stop Loss más ajustado a 1.5x ATR
            
            # --- GESTIÓN DE RIESGO: 1% ---
            risk_usd = self.balance * 0.01
            qty = risk_usd / stop_dist

            sl_price = entry_price - stop_dist if side == "LONG" else entry_price + stop_dist
            tp_price = r.bb_mid # TP único en la media móvil de las bandas
            
            self.open_position = {
                "side": side, "qty": qty, "entry_price": entry_price,
                "sl": sl_price, "tp1": tp_price, "tp2": None, # No hay TP2
                "tp1_hit": False, "regime": "RANGO"
            }
            logging.info(f"{r.timestamp} | RANGO | OPEN {side} | Price: {entry_price:.2f} | Qty: {qty:.4f}")
            self._log_trade(r.timestamp, "OPEN", side, entry_price, qty, "RANGO")

    def _manage_open_position(self, r):
        """Gestiona la posición abierta, verificando SL y TPs."""
        if not self.open_position:
            return

        pos = self.open_position
        price = r.close
        pnl = 0.0
        reason = ""

        # --- Lógica de Múltiples TPs para TENDENCIA ---
        if pos['regime'] == 'TENDENCIA' and not pos['tp1_hit']:
            if (pos['side'] == 'LONG' and price >= pos['tp1']) or \
               (pos['side'] == 'SHORT' and price <= pos['tp1']):
                
                half_qty = pos['qty'] / 2
                pnl = (price - pos['entry_price']) * half_qty * (1 if pos['side'] == 'LONG' else -1)
                self.balance += pnl - (half_qty * price * self.fee)
                
                pos['qty'] = half_qty # Reducimos la cantidad a la mitad
                pos['sl'] = pos['entry_price'] # Movemos SL a Break-Even
                pos['tp1_hit'] = True
                logging.info(f"{r.timestamp} | TENDENCIA | TP1 HIT | PnL: ${pnl:.2f} | Moving SL to Break-Even")
                self._log_trade(r.timestamp, "TP1", pos['side'], price, half_qty, pos['regime'], pnl)

        # --- Lógica de Cierre Total (SL o TP final) ---
        is_long = pos['side'] == 'LONG'
        final_tp = pos['tp2'] if pos['regime'] == 'TENDENCIA' else pos['tp1']

        if (is_long and price <= pos['sl']) or (not is_long and price >= pos['sl']):
            reason = "STOP"
        elif final_tp and ((is_long and price >= final_tp) or (not is_long and price <= final_tp)):
            reason = "TAKE_PROFIT"

        if reason:
            pnl = (price - pos['entry_price']) * pos['qty'] * (1 if is_long else -1)
            self.balance += pnl - (pos['qty'] * price * self.fee)
            logging.info(f"{r.timestamp} | {pos['regime']} | CLOSE {reason} | PnL: ${pnl:.2f} | New Balance: ${self.balance:.2f}")
            self._log_trade(r.timestamp, reason, pos['side'], price, pos['qty'], pos['regime'], pnl)
            self.open_position = None

    def _log_trade(self, ts, action, side, price, qty, regime, pnl=0.0):
        self.trades.append({
            "timestamp": ts,
            "action": action,
            "side": side,
            "price": price,
            "qty": qty,
            "regime": regime,
            "pnl": pnl,
            "balance": self.balance
        })

    def run(self):
        """Ejecuta el bucle principal de la simulación."""
        for i in range(1, len(self.df_5m)):
            current_row = self.df_5m.iloc[i]
            prev_row = self.df_5m.iloc[i-1]
            
            self._manage_open_position(current_row)

            if not self.open_position:
                regime = self._detectar_regimen(current_row.timestamp)
                if regime == "TENDENCIA":
                    self._run_strategy_tendencia(current_row, prev_row)
                elif regime == "RANGO":
                    self._run_strategy_rango(current_row, prev_row)
        
        return self._generate_report()

    def _generate_report(self):
        """Genera y muestra un reporte final del backtest."""
        df_trades = pd.DataFrame(self.trades)
        if df_trades.empty:
            print("\nNo se realizaron operaciones.")
            return

        logging.info("Generando reporte de backtest...")
        df_trades['pnl'] = df_trades['pnl'].astype(float)
        
        total_pnl = df_trades['pnl'].sum()
        total_trades = len(df_trades[df_trades['action'] != 'OPEN'])
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] < 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        # Reporte por estrategia
        tendencia_pnl = df_trades[df_trades['regime'] == 'TENDENCIA']['pnl'].sum()
        rango_pnl = df_trades[df_trades['regime'] == 'RANGO']['pnl'].sum()
        
        print("\n" + "="*50)
        print("          REPORTE FINAL DEL BACKTEST")
        print("="*50)
        print(f"Resultado Neto Final: ${total_pnl:,.2f}")
        print(f"Balance Final: ${self.balance:,.2f}")
        print(f"Operaciones Totales: {total_trades}")
        print(f"Tasa de Acierto (Win Rate): {win_rate:.2f}%")
        print(f"Ganancia Promedio: ${wins['pnl'].mean():,.2f}" if not wins.empty else "Ganancia Promedio: $0.00")
        print(f"Pérdida Promedio: ${losses['pnl'].mean():,.2f}" if not losses.empty else "Pérdida Promedio: $0.00")
        print("-"*50)
        print("--- Rendimiento por Estrategia ---")
        print(f"PnL Estrategia TENDENCIA: ${tendencia_pnl:,.2f}")
        print(f"PnL Estrategia RANGO: ${rango_pnl:,.2f}")
        print("="*50)
        
        out_path = f"data/backtests/mutante_results.csv"
        df_trades.to_csv(out_path, index=False)
        logging.info(f"Resultados detallados guardados en: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Backtester para bot de trading con doble estrategia.")
    parser.add_argument("--csv5m", required=True, help="Ruta al archivo CSV con datos OHLCV de 5 minutos.")
    parser.add_argument("--csv1h", required=True, help="Ruta al archivo CSV con datos OHLCV de 1 hora.")
    parser.add_argument("--balance", type=float, default=1000.0, help="Balance inicial en USDT.")
    parser.add_argument("--fee", type=float, default=0.0005, help="Comisión por operación (ej: 0.0005 para 0.05%).")
    args = parser.parse_args()

    os.makedirs("data/backtests", exist_ok=True)

    try:
        df_5m = pd.read_csv(args.csv5m, parse_dates=["timestamp"])
        df_1h = pd.read_csv(args.csv1h, parse_dates=["timestamp"])
    except FileNotFoundError as e:
        logging.error(f"Error: No se encontró el archivo de datos: {e}")
        return

    backtester = Backtester(df_5m=df_5m, df_1h=df_1h, initial_balance=args.balance, fee_pct=args.fee)
    backtester.run()

if __name__ == "__main__":
    main()
