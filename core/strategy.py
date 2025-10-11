import pandas as pd


class Strategy:
    def __init__(self, cfg):
        self.config = cfg

    def check_entry_signal(self, data: pd.DataFrame) -> str | None:
        """Revisa la última vela en busca de una señal de entrada."""
        last_candle = data.iloc[-1]

        # Lógica de Pullback Grid
        atr = float(last_candle["atr"])
        price = float(last_candle["close"])

        # Asumimos que el ancla es la EMA200 de 4h para simplificar
        anchor = float(last_candle.get("ema200_4h", price))

        step = self.config.get('grid_step_atr', 0.6) * atr
        half_span = self.config.get('grid_span_atr', 2.5) * atr

        # Determinar dirección de la tendencia
        side_pref = None
        if price > anchor:
            side_pref = "LONG"
        else:
            side_pref = "SHORT"

        if side_pref == "LONG":
            if (price < anchor) and (anchor - price >= step) and (anchor - price <= half_span):
                return "LONG"
        else:  # SHORT
            if (price > anchor) and (price - anchor >= step) and (price - anchor <= half_span):
                return "SHORT"

        return None

    def check_all_filters(self, row: pd.Series, side: str) -> str | None:
        """Aplica todos los filtros de calidad a una señal."""
        price = float(row["close"])

        # Filtro de Tendencia (redundante pero seguro)
        if side == "LONG" and price < row['ema200_4h']:
            return "Filtro de Tendencia (Precio < EMA200 4h)"
        if side == "SHORT" and price > row['ema200_4h']:
            return "Filtro de Tendencia (Precio > EMA200 4h)"

        # Filtro de RSI 4h
        rsi4h = float(row["rsi4h"])
        rsi4h_gate = self.config.get('rsi4h_gate', 52)
        if side == "LONG" and rsi4h < rsi4h_gate:
            return f"Filtro RSI 4h ({rsi4h:.1f} < {rsi4h_gate})"
        if side == "SHORT" and rsi4h > (100 - rsi4h_gate):
            return f"Filtro RSI 4h ({rsi4h:.1f} > {100-rsi4h_gate})"

        # ... Aquí se podrían añadir más filtros del config.yaml ...

        return None  # Si pasa todos los filtros

    def calculate_sl(self, entry_price, last_candle):
        """Calcula el precio del Stop Loss."""
        atr = float(last_candle["atr"])
        sl_mult = self.config.get('sl_atr_mult', 1.3)
        # Lógica simplificada, asumiendo LONG
        return entry_price - (atr * sl_mult)

    def calculate_tp(self, entry_price, quantity, balance):
        """Calcula el precio del Take Profit."""
        tp_pct = self.config.get('target_eq_pnl_pct', 0.10)
        pnl_target = balance * tp_pct
        # Lógica simplificada, asumiendo LONG
        return entry_price + (pnl_target / quantity)
