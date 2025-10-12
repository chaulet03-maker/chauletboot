import logging
import asyncio
from collections import deque
from typing import cast
import pandas as pd
import schedule
from telegram.ext import ContextTypes

from anchor_freezer import Side
from deps import FREEZER
from bot.exchange import Exchange
from bot.trader import Trader
from bot.storage import Storage
from bot.telemetry.notifier import Notifier
from bot.telemetry.telegram_bot import setup_telegram_bot
from core.strategy import Strategy
from core.indicators import add_indicators
from config import S
from trading import BROKER


class TradingApp:
    def __init__(self, cfg):
        self.config = cfg
        self.config.setdefault("trading_mode", S.trading_mode)
        self.config.setdefault("mode", "paper" if S.PAPER else "real")
        self.config.setdefault("start_equity", S.start_equity)
        self.logger = logging.getLogger(__name__)
        self._init_db()
        self.trader = Trader(cfg)
        self.exchange = Exchange(cfg)
        self.storage = Storage(cfg)
        self.strategy = Strategy(cfg)
        self.is_paused = False
        self.connection_lost = False
        self.rejection_log = deque(maxlen=10)
        self.telegram_app = setup_telegram_bot(self)
        self.notifier = Notifier(application=self.telegram_app, cfg=self.config)
        self.symbols = [self.config.get('symbol', 'BTC/USDT')]
        self.price_cache = {}
        self._funding_series = None
        self._daily_R: dict = {}
        self._risk_usd_trade = 0.0
        self._eq_on_open = 0.0
        self._entry_ts = None
        self._bars_in_position = 0

        schedule.every().day.at("08:00", "America/Argentina/Buenos_Aires").do(self._generate_daily_report)
        schedule.every().sunday.at("00:00", "America/Argentina/Buenos_Aires").do(self._generate_weekly_report)
        logging.info("Componentes y tareas de reporte inicializados.")

        if str(self.config.get("trading_mode", "simulado")).lower() == "simulado":
            funding_csv = self.config.get("funding_csv")
            if funding_csv:
                try:
                    self._load_funding_series(funding_csv)
                except Exception as exc:
                    logging.warning("No se pudo cargar la serie de funding desde %s: %s", funding_csv, exc)

    # === Rejection recording helpers ===
    def record_rejection(self, symbol: str, side: str, code: str, detail: str = "", ts=None):
        try:
            from datetime import datetime, timezone
            from collections import deque
            if not hasattr(self, "rejection_log"):
                self.rejection_log = deque(maxlen=50)  # si no existe, lo creo
            if ts is None:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            item = {"iso": ts, "symbol": symbol, "side": side, "code": code, "detail": detail}
            self.rejection_log.append(item)

            # Reflejá también en el notificador PRO si existe
            tg = getattr(self, "telegram", None) or getattr(self, "notifier", None)
            if tg is not None and hasattr(tg, "log_reject"):
                tg.log_reject(symbol=symbol, side=side, code=code, detail=detail)
        except Exception as e:
            import logging
            logging.debug(f"record_rejection failed: {e}")

    async def trading_loop(self, context: ContextTypes.DEFAULT_TYPE):
        """Bucle principal de trading ejecutado por la JobQueue de Telegram."""
        try:
            schedule.run_pending()
            if self.connection_lost:
                await self.notifier.send("✅ **Conexión Reestablecida.**")
                self.connection_lost = False

            logging.info("Iniciando ciclo de análisis de mercado...")

            position = await self.trader.check_open_position(self.exchange)
            if position:
                logging.info(
                    "Posición abierta detectada: %s %s %s",
                    position.get('side'),
                    position.get('contracts'),
                    position.get('symbol'),
                )
                try:
                    now_ts = pd.Timestamp.utcnow().tz_localize("UTC")
                    last_price = await self.exchange.get_current_price(self.config.get('symbol'))
                    side = (position.get('side') or '').upper()
                    qty_raw = position.get('contracts') or position.get('size')
                    qty = float(qty_raw) if qty_raw is not None else 0.0
                    if last_price is not None and side and qty > 0:
                        self._apply_funding_if_needed(now_ts, float(last_price), side, qty)
                except Exception as exc:
                    logging.debug("No se pudo aplicar funding a la posición abierta: %s", exc)
                return

            if self.is_paused:
                logging.info("El bot está en pausa, no se buscan nuevas señales.")
                return

            klines_1h = await self.exchange.get_klines('1h')
            klines_4h = await self.exchange.get_klines('4h')

            if not klines_1h or not klines_4h:
                logging.warning("No se pudieron obtener los datos de mercado en este ciclo.")
                return

            df_1h = pd.DataFrame(klines_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms', utc=True)
            df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms', utc=True)
            df_1h.set_index('timestamp', inplace=True)
            df_4h.set_index('timestamp', inplace=True)

            data = add_indicators(df_1h, df_4h)
            last_candle = data.iloc[-1]
            now_ts = data.index[-1]

            if self._should_block_new_trade_today(now_ts):
                logging.info("Daily stop alcanzado, no se abrirán nuevas operaciones hoy.")
                return

            if self.config.get("funding_interval_hours", 8) and self.config.get("funding_gate_bps") is not None:
                try:
                    current_rate_dec = await self.exchange.fetch_current_funding_rate(self.config.get('symbol'))
                except Exception as exc:
                    logging.debug("Fallo al obtener funding runtime: %s", exc)
                    current_rate_dec = self._funding_rate_at(now_ts)
                if current_rate_dec is not None:
                    self.config["_funding_rate_now"] = float(current_rate_dec)
                    self.config["_funding_rate_bps_now"] = float(current_rate_dec) * 10000.0
                else:
                    self.config.pop("_funding_rate_now", None)
                    self.config.pop("_funding_rate_bps_now", None)

            signal = self.strategy.check_entry_signal(data)
            if not signal:
                try:
                    code, detail, extras = self.strategy.get_rejection_reason(data)
                    item_detail = detail if not extras else f"{detail} | {extras}"
                    self.record_rejection(
                        symbol=self.config.get("symbol", ""),
                        side=str(self.config.get("grid_side", "auto")).upper(),
                        code=code or "pre_open_checks",
                        detail=item_detail or "Sin detalle"
                    )
                except Exception:
                    pass

                if bool(self.config.get("debug_signals", False)):
                    try:
                        self.strategy.explain_signal(data)
                    except Exception as e:
                        self.logger.info(f"SIGNAL DEBUG fallo: {e}")

                logging.info("No se encontraron señales de entrada válidas.")
                return

            eq_now = await self.trader.get_balance(self.exchange)
            eq_on_open = eq_now

            leverage = self.strategy.dynamic_leverage(last_candle)
            if S.PAPER:
                self.logger.info(
                    "PAPER: leverage lógico=%s (no se setea en Binance)", leverage
                )
            else:
                await self.exchange.set_leverage(
                    leverage,
                    self.config.get('symbol', 'BTC/USDT'),
                )

            entry_price = await self.exchange.get_current_price()
            if entry_price is None:
                logging.warning("No se pudo obtener el precio de entrada actual.")
                return
            entry_price = float(entry_price)
            qty = (eq_on_open * leverage) / max(entry_price, 1e-12)

            sl_price = self.strategy.calculate_sl(entry_price, last_candle, signal)
            side = signal
            tp_price = self.strategy.calculate_tp(entry_price, qty, eq_on_open, side)

            order_result = await self.exchange.create_order(signal, qty, sl_price, tp_price)

            if order_result:
                await self.notifier.send(
                    f"🚀 **Nueva Operación Abierta: {signal}**\n"
                    f"Símbolo: {self.config['symbol']}\n"
                    f"Apalancamiento: x{leverage}"
                )
                await self.trader.set_position(order_result)
                self._risk_usd_trade = abs(entry_price - sl_price) * qty
                self._eq_on_open = eq_on_open
                self._entry_ts = now_ts
                self._bars_in_position = 0
                symbol = str(self.config.get("symbol", ""))
                if symbol:
                    for s in ("LONG", "SHORT"):
                        FREEZER.clear(symbol, cast(Side, s))

        except Exception as e:
            logging.error(f"Error grave en el trading_loop: {e}", exc_info=True)
            if not self.connection_lost:
                await self.notifier.send(f"💥 **Error inesperado en el bot:** {e}")
                self.connection_lost = True

    def price_of(self, symbol: str):
        return self.price_cache.get(symbol)

    async def fetch_last_price(self, symbol: str):
        return await self.exchange.get_current_price(symbol)

    async def _update_price_cache_job(self, context):
        try:
            symbols = self.symbols or [self.config.get('symbol', 'BTC/USDT')]
            for sym in symbols:
                px = await self.exchange.get_current_price(sym)
                if px is not None:
                    self.price_cache[sym] = float(px)
        except Exception as e:
            logging.warning(f"No pude actualizar la cache de precios: {e}")

    def run(self):
        job_queue = self.telegram_app.job_queue
        job_queue.run_repeating(self._update_price_cache_job, interval=10, first=1)
        job_queue.run_repeating(self.trading_loop, interval=60, first=5)

        loop = asyncio.get_event_loop()
        mode_msg = "🧪 Modo SIMULADO activo" if S.PAPER else "🔴 Modo REAL activo"
        if S.PAPER:
            try:
                equity = float(getattr(BROKER, "equity"))
                mode_msg += f"\nEquity sim: ${equity:.2f} USDT"
            except Exception:
                pass
        boot_message = f"{mode_msg}\n✅ **Bot iniciado y corriendo.**"

        if loop.is_running():
            loop.create_task(self.notifier.send(boot_message))
        else:
            loop.run_until_complete(self.notifier.send(boot_message))

        logging.info("Bucle de trading programado. Iniciando polling de Telegram.")
        self.telegram_app.run_polling()

    def _init_db(self):
        pass

    def _generate_daily_report(self):
        pass

    def _generate_weekly_report(self):
        pass

    # ===== Funding helpers (modo simulado) =====
    def _load_funding_series(self, path: str):
        import pandas as pd

        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        ts_col = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else "date")
        rate_col = "rate" if "rate" in df.columns else ("funding_rate" if "funding_rate" in df.columns else None)
        if rate_col is None:
            raise ValueError("Funding CSV debe tener 'rate' o 'funding_rate'.")
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        ser = pd.Series(df[rate_col].astype(float).values, index=ts).dropna()
        self._funding_series = ser.sort_index()

    def _funding_rate_at(self, ts):
        ser = getattr(self, "_funding_series", None)
        if ser is None:
            return float(self.config.get("funding_default", 0.0))
        idx = ser.index.searchsorted(ts)
        if idx == 0:
            return float(ser.iloc[0])
        if idx >= len(ser):
            return float(ser.iloc[-1])
        prev_idx = idx - 1 if ser.index[idx] != ts else idx
        return float(ser.iloc[prev_idx])

    def _is_funding_bar(self, ts):
        interval = int(self.config.get("funding_interval_hours", 8))
        return (interval > 0) and (getattr(ts, "minute", 0) == 0) and (getattr(ts, "hour", 0) % interval == 0)

    def _apply_funding_if_needed(self, now_ts, last_price, side, qty):
        if str(self.config.get("trading_mode", "simulado")).lower() != "simulado":
            return 0.0
        if not self._is_funding_bar(now_ts):
            return 0.0
        rate = self._funding_rate_at(now_ts)
        notional = qty * last_price
        sign = -1 if side == "LONG" else +1
        funding_pnl = sign * rate * notional
        if hasattr(self.trader, "_balance"):
            self.trader._balance += funding_pnl
        try:
            self._log_trade_event(now_ts, "FUNDING", side, last_price, qty, funding_pnl, note=f"rate={rate:.6f}")
        except Exception:
            pass
        return funding_pnl

    # ===== Daily R tracking =====
    def _day_key(self, ts):
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is not None:
                return ts.tz_convert("UTC").floor("D")
            return ts.tz_localize("UTC").floor("D")
        return pd.Timestamp(ts).tz_localize("UTC").floor("D")

    def _should_block_new_trade_today(self, now_ts):
        daily_stop_R = float(self.config.get("daily_stop_R", 0.0) or 0.0)
        if daily_stop_R <= 0:
            return False
        dkey = self._day_key(now_ts)
        return self._daily_R.get(dkey, 0.0) <= -daily_stop_R

    def _register_close_R(self, now_ts, trade_pnl):
        risk = max(1e-12, getattr(self, "_risk_usd_trade", 0.0))
        if risk <= 0:
            return
        R = trade_pnl / risk
        dkey = self._day_key(now_ts)
        self._daily_R[dkey] = self._daily_R.get(dkey, 0.0) + R
