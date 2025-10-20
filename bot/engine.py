import logging
import math
import asyncio
import os
import threading
import inspect
from collections import deque
from typing import Any, Dict, Optional, cast
from time import time as _now
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
from brokers import ACTIVE_LIVE_CLIENT
from bot.identity import get_bot_id, make_client_oid
from bot.ledger import bot_position, prune_open_older_than, init as ledger_init
from state_store import create_position, load_state, persist_open, save_state
from bot.motives import MOTIVES, MotiveItem, compute_codes
from core.strategy import Strategy
from core.indicators import add_indicators
from config import S
import trading
from risk_guards import (
    clear_pause_if_expired,
    get_pause_manager,
    in_ban_hours,
    is_paused_now,
)
from reanudar_listener import listen_reanudar

class _EngineBrokerAdapter:
    def __init__(self, app: "TradingApp"):
        self.app = app

    async def place_market_order(self, symbol: str, side: str, quantity: float, leverage: int = 1):
        trading.ensure_initialized()
        ledger_init()

        sym = symbol or self.app.config.get("symbol", "BTC/USDT")
        sym_clean = sym.replace("/", "").upper()
        mode = "live" if self.app.is_live else "paper"
        side_u = str(side).upper()

        if self.app.is_live:
            try:
                await self.app.exchange.set_leverage(leverage, sym)
            except Exception:
                self.app.logger.debug("No se pudo establecer leverage", exc_info=True)

        bot_id = get_bot_id()
        client_oid = make_client_oid(bot_id, sym_clean, mode)

        result = await asyncio.to_thread(
            trading.place_order_safe,
            side_u,
            float(quantity),
            None,
            symbol=sym,
            leverage=int(leverage),
            newClientOrderId=client_oid,
            order_type="MARKET",
        )

        if isinstance(result, dict):
            result.setdefault("newClientOrderId", client_oid)
            result.setdefault("clientOrderId", client_oid)
        return result


class TradingApp:
    def __init__(self, cfg):
        self.config = cfg
        self.config.setdefault("trading_mode", S.trading_mode)
        self.config.setdefault("mode", "paper" if S.PAPER else "real")
        self.config.setdefault("start_equity", S.start_equity)
        self.logger = logging.getLogger(__name__)

        trading.ensure_initialized()

        self.broker = _EngineBrokerAdapter(self)

        storage_cfg = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
        self.db_path = (
            storage_cfg.get("db_path")
            or os.getenv("PERF_DB_PATH")
            or "data/perf.db"
        )
        self._init_db()
        self.trader = Trader(cfg)
        self.exchange = Exchange(cfg)
        self.storage = Storage(cfg)
        self.strategy = Strategy(cfg)
        self.is_paused = False
        self.pause_manager = get_pause_manager()
        self._reanudar_thread: Optional[threading.Thread] = None
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
        self._entry_locks: Dict[tuple[str, str, int], float] = {}
        self._entry_lock_gc_last = 0.0
        try:
            ttl_cfg = cfg.get("entry_lock_ttl_seconds", 12 * 60 * 60)
            self._entry_lock_ttl_s = max(int(float(ttl_cfg)), 60)
        except Exception:
            self._entry_lock_ttl_s = 12 * 60 * 60
        self._shock_guard_last_trade_id: Optional[Any] = None

        schedule.every().day.at("07:00", "America/Argentina/Buenos_Aires").do(self._generate_daily_report)
        schedule.every().sunday.at("07:01", "America/Argentina/Buenos_Aires").do(self._generate_weekly_report)
        schedule.every(30).minutes.do(self._prune_old_orders)
        logging.info("Componentes y tareas de reporte inicializados.")

        if str(self.config.get("trading_mode", "simulado")).lower() == "simulado":
            funding_csv = self.config.get("funding_csv")
            if funding_csv:
                try:
                    self._load_funding_series(funding_csv)
                except Exception as exc:
                    logging.warning("No se pudo cargar la serie de funding desde %s: %s", funding_csv, exc)

        self._start_reanudar_listener()

    @property
    def active_mode(self) -> str:
        return str(trading.ACTIVE_MODE).lower()

    @property
    def is_live(self) -> bool:
        return self.active_mode in {"real", "live"}

    @property
    def is_paper(self) -> bool:
        return not self.is_live

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

    def _prune_old_orders(self):
        try:
            if getattr(S, "PAPER", False):
                mode = "paper"
            else:
                trading_mode = str(getattr(S, "trading_mode", "real")).lower()
                mode = "live" if trading_mode in {"real", "live"} else "paper"
            prune_open_older_than(mode, get_bot_id(), hours=16)
        except Exception:
            self.logger.debug("No se pudo ejecutar prune_open_older_than", exc_info=True)

    def _record_motive(self, ctx_overrides: Optional[Dict[str, Any]] = None):
        try:
            def _safe_float(value: Any) -> Optional[float]:
                if value is None:
                    return None
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    return None
                return f if math.isfinite(f) else None

            ctx: Dict[str, Any] = {
                "price": _safe_float(getattr(self, "last_price", None)),
                "anchor": _safe_float(getattr(self, "anchor", None)),
                "step": _safe_float(getattr(self, "step", None)),
                "span": _safe_float(getattr(self, "span", None)),
                "ema200_1h": _safe_float(getattr(self, "ema200_1h", None)),
                "ema200_4h": _safe_float(getattr(self, "ema200_4h", None)),
                "atrp": _safe_float(getattr(self, "atrp", None)),
                "adx": _safe_float(getattr(self, "adx", None)),
                "rsi4h": _safe_float(getattr(self, "rsi4h", None)),
                "gate_ok": getattr(self, "gate_ok", None),
                "risk_ok": getattr(self, "risk_ok", None),
                "has_open": bool(getattr(self, "position_open", False)),
                "cooldown": bool(getattr(self, "cooldown_active", False)),
                "freeze_90": bool(getattr(self, "freeze_90_active", False)),
                "blackout": bool(getattr(self, "blackout_active", False)),
                "reasons": list(getattr(self, "reasons", []) or []),
                "adx_thr": float(getattr(self, "adx_strong_threshold", 25.0)),
            }

            if ctx_overrides:
                for key, value in ctx_overrides.items():
                    if key == "reasons":
                        if value is None:
                            ctx["reasons"] = []
                        elif isinstance(value, (list, tuple, set)):
                            ctx["reasons"] = [str(v) for v in value if v is not None]
                        else:
                            ctx["reasons"] = [str(value)]
                    else:
                        ctx[key] = value

            reasons = []
            seen = set()
            for item in ctx.get("reasons", []):
                if item is None:
                    continue
                text = str(item)
                if text not in seen:
                    seen.add(text)
                    reasons.append(text)
            ctx["reasons"] = reasons

            price_val = ctx.get("price")
            try:
                price_out = float(price_val) if price_val is not None else 0.0
            except Exception:
                price_out = 0.0

            codes = compute_codes(ctx)
            MOTIVES.add(
                MotiveItem(
                    ts=_now(),
                    symbol=self.config.get("symbol", "BTC/USDT"),
                    side_pref=str(getattr(self, "side_pref", None)) if getattr(self, "side_pref", None) is not None else None,
                    price=price_out,
                    codes=codes,
                    ctx=ctx,
                )
            )
            self.logger.debug("MOTIVES/REC %s", codes)
        except Exception as e:
            self.logger.debug("MOTIVES/REC fail: %s", e)

    def _cleanup_entry_locks(self, now: float, ttl: int) -> None:
        if not self._entry_locks:
            return
        if (now - self._entry_lock_gc_last) < 60.0:
            return
        cutoff = now - ttl
        stale = [key for key, ts in self._entry_locks.items() if ts < cutoff]
        for key in stale:
            self._entry_locks.pop(key, None)
        self._entry_lock_gc_last = now

    def _start_reanudar_listener(self) -> None:
        if self._reanudar_thread is not None and self._reanudar_thread.is_alive():
            return

        def _on_reanudar() -> None:
            try:
                self.pause_manager.clear_pause()
                self.is_paused = False
                logging.info("[RISK] Pausa eliminada manualmente vía Telegram.")
            except Exception as exc:  # pragma: no cover - defensivo
                logging.debug("reanudar callback falló: %s", exc)

        try:
            thread = threading.Thread(target=listen_reanudar, args=(_on_reanudar,), daemon=True)
            thread.start()
            self._reanudar_thread = thread
        except Exception as exc:  # pragma: no cover - defensivo
            logging.debug("No se pudo iniciar listener de reanudar: %s", exc)

    def _acquire_entry_lock(
        self,
        symbol: Optional[str],
        side: Optional[str],
        anchor_epoch: Optional[int],
    ) -> bool:
        if not symbol or not side or anchor_epoch is None:
            return True

        ttl = getattr(self, "_entry_lock_ttl_s", 12 * 60 * 60)
        storage = getattr(self, "storage", None)
        if storage is not None and hasattr(storage, "acquire_entry_lock"):
            try:
                acquired = storage.acquire_entry_lock(
                    symbol,
                    side,
                    anchor_epoch,
                    ttl_seconds=ttl,
                )
                if not acquired:
                    return False
            except Exception as exc:  # pragma: no cover - defensivo
                self.logger.debug("entry_lock storage error: %s", exc)

        key = (str(symbol).upper(), str(side).upper(), int(anchor_epoch))
        now = _now()
        self._cleanup_entry_locks(now, ttl)
        if key in self._entry_locks:
            return False
        self._entry_locks[key] = now
        return True

    async def trading_loop(self, context: ContextTypes.DEFAULT_TYPE):
        """Bucle principal de trading ejecutado por la JobQueue de Telegram."""
        try:
            schedule.run_pending()
            clear_pause_if_expired()
            self.is_paused = bool(is_paused_now())
            if self.connection_lost:
                await self.notifier.send("✅ **Conexión Reestablecida.**")
                self.connection_lost = False

            logging.info("Iniciando ciclo de análisis de mercado...")

            try:
                from risk_guards import check_shock_pause_and_pause_if_needed

                check_shock_pause_and_pause_if_needed(
                    settings=self.config,
                    state=self,
                    market=self.exchange,
                    pause_manager=self.pause_manager,
                    notifier=self.notifier,
                )
            except Exception as e:  # pragma: no cover - defensivo
                self.logger.exception(f"Shock gate check failed: {e}")

            # 0) Refrescar mark-to-market en paper utilizando el precio actual
            if S.PAPER and trading.POSITION_SERVICE is not None:
                try:
                    last_px = await self.exchange.get_current_price(self.config.get('symbol', 'BTC/USDT'))
                    if last_px is not None:
                        trading.POSITION_SERVICE.mark_to_market(float(last_px))
                except Exception as exc:
                    logging.debug("No se pudo refrescar mark-to-market inicial: %s", exc)

            position = None
            bot_has_open = False
            try:
                st = trading.POSITION_SERVICE.get_status() if trading.POSITION_SERVICE else None
            except Exception as exc:
                st = None
                logging.debug("PositionService status error: %s", exc)
            try:
                # --- Evaluar posición local del BOT con tolerancia ---
                qty_local = 0.0
                side = "FLAT"
                if st:
                    side = (st.get("side") or "FLAT").upper()
                    # tolerancia para evitar falsos positivos
                    qty_local = float(st.get("qty") or st.get("size") or 0.0) or 0.0
                bot_has_open = side != "FLAT" and abs(qty_local) > 1e-12

                # --- En REAL nunca confiar en local si el exchange no reporta abierta ---
                if not S.PAPER:
                    try:
                        ex_pos = await self.exchange.get_open_position(self.config.get("symbol"))
                        ex_qty = float(
                            (ex_pos or {}).get("contracts")
                            or (ex_pos or {}).get("positionAmt")
                            or (ex_pos or {}).get("size")
                            or 0.0
                        )
                        live_has_open = abs(ex_qty) > 0.0
                    except Exception:
                        ex_pos = None
                        live_has_open = False

                    if not live_has_open:
                        # limpiar cualquier rastro local para no “revivir” posiciones
                        bot_has_open = False
                        position = None
                        try:
                            if hasattr(self.trader, "_open_position"):
                                self.trader._open_position = None
                        except Exception:
                            pass
                    else:
                        # si hay en el exchange, mapear a la cache local coherente
                        bot_has_open = True
                        side = ((ex_pos or {}).get("side") or side or "FLAT").upper()
                        position = {
                            "symbol": (ex_pos or {}).get("symbol", self.config.get("symbol")),
                            "side": side,
                            "contracts": ex_qty,
                            "entryPrice": float((ex_pos or {}).get("entryPrice") or 0.0),
                            "markPrice": float((ex_pos or {}).get("markPrice") or 0.0),
                        }
                        await self.trader.set_position(position)
                else:
                    # SIM: sólo si qty_local > 0 (|qty| > 1e-12) y side != FLAT
                    if bot_has_open:
                        position = {
                            "symbol": st.get("symbol", self.config.get("symbol")),
                            "side": side,
                            "contracts": qty_local,
                            "entryPrice": float(st.get("entry_price") or 0.0),
                            "markPrice": float(st.get("mark") or 0.0),
                        }
                        await self.trader.set_position(position)
                    else:
                        # sin qty local -> asegurar limpieza de cache
                        try:
                            if hasattr(self.trader, "_open_position"):
                                self.trader._open_position = None
                        except Exception:
                            pass
            except Exception as exc:
                logging.debug("PositionService mapping fail: %s", exc)

            cooldown_active = bool(getattr(self, "cooldown_active", False))
            freeze_90_active = bool(getattr(self, "freeze_90_active", False))
            blackout_active = bool(getattr(self, "blackout_active", False))

            base_ctx: Dict[str, Any] = {
                "price": None,
                "anchor": None,
                "step": None,
                "span": None,
                "ema200_1h": None,
                "ema200_4h": None,
                "atrp": None,
                "adx": None,
                "rsi4h": None,
                "gate_ok": None,
                "risk_ok": None,
                "has_open": bool(bot_has_open),
                "cooldown": cooldown_active,
                "freeze_90": freeze_90_active,
                "blackout": blackout_active,
                "reasons": [],
                "adx_thr": float(getattr(self, "adx_strong_threshold", 25.0)),
            }
            side_pref: Optional[str] = None

            self.last_price = None
            self.anchor = None
            self.step = None
            self.span = None
            self.ema200_1h = None
            self.ema200_4h = None
            self.atrp = None
            self.adx = None
            self.rsi4h = None
            self.gate_ok = None
            self.risk_ok = None
            self.reasons = []
            self.side_pref = None
            self.position_open = bool(bot_has_open)
            self.cooldown_active = cooldown_active
            self.freeze_90_active = freeze_90_active
            self.blackout_active = blackout_active

            def _emit_motive(
                overrides: Optional[Dict[str, Any]] = None,
                price_value: Optional[float] = None,
                side_value: Optional[str] = None,
            ) -> None:
                ctx_local = dict(base_ctx)
                if overrides:
                    if "reasons" in overrides:
                        val = overrides["reasons"]
                        if val is None:
                            ctx_local["reasons"] = []
                        elif isinstance(val, (list, tuple, set)):
                            ctx_local["reasons"] = [str(v) for v in val if v is not None]
                        else:
                            ctx_local["reasons"] = [str(val)]
                    for key, value in overrides.items():
                        if key == "reasons":
                            continue
                        ctx_local[key] = value
                if price_value is not None:
                    ctx_local["price"] = price_value

                local_side = side_value if side_value is not None else side_pref

                self.last_price = ctx_local.get("price")
                self.anchor = ctx_local.get("anchor")
                self.step = ctx_local.get("step")
                self.span = ctx_local.get("span")
                self.ema200_1h = ctx_local.get("ema200_1h")
                self.ema200_4h = ctx_local.get("ema200_4h")
                self.atrp = ctx_local.get("atrp")
                self.adx = ctx_local.get("adx")
                self.rsi4h = ctx_local.get("rsi4h")
                self.gate_ok = ctx_local.get("gate_ok")
                self.risk_ok = ctx_local.get("risk_ok")
                self.reasons = list(ctx_local.get("reasons") or [])
                self.position_open = bool(ctx_local.get("has_open", self.position_open))
                self.cooldown_active = bool(ctx_local.get("cooldown", self.cooldown_active))
                self.freeze_90_active = bool(ctx_local.get("freeze_90", self.freeze_90_active))
                self.blackout_active = bool(ctx_local.get("blackout", self.blackout_active))
                self.side_pref = local_side

                self._record_motive(ctx_local)

            def _safe_float_local(value: Any) -> Optional[float]:
                if value is None:
                    return None
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    return None
                return f if math.isfinite(f) else None

            if position:
                logging.info(
                    "Posición abierta detectada: %s %.6f %s",
                    (position.get('side') or '').upper(),
                    float(position.get('contracts') or position.get('size') or 0.0),
                    position.get('symbol') or self.config.get('symbol'),
                )
                last_price = None
                try:
                    now_ts = pd.Timestamp.utcnow().tz_localize("UTC")
                    last_price = await self.exchange.get_current_price(self.config.get('symbol'))
                    side = (position.get('side') or '').upper()
                    qty_raw = position.get('contracts') or position.get('size') or 0.0
                    qty = float(qty_raw) if qty_raw is not None else 0.0
                    if last_price is not None and side and qty > 0:
                        self._apply_funding_if_needed(now_ts, float(last_price), side, qty)
                except Exception as exc:
                    logging.debug("No se pudo aplicar funding a la posición abierta: %s", exc)

                price_val = None
                try:
                    price_val = float(last_price) if last_price is not None else None
                except Exception:
                    price_val = None

                ctx_overrides = {"has_open": True}
                _emit_motive(ctx_overrides, price_value=price_val, side_value=position.get('side'))
                return

            if blackout_active:
                logging.info("Blackout activo, no se evaluarán nuevas entradas.")
                price_val = None
                try:
                    price_cached = self.price_cache.get(self.config.get('symbol', 'BTC/USDT'))
                    price_val = float(price_cached) if price_cached is not None else None
                except Exception:
                    price_val = None
                ctx_overrides = {"blackout": True}
                _emit_motive(ctx_overrides, price_value=price_val)
                return

            if freeze_90_active:
                logging.info("Congelamiento 90% activo, no se evaluarán nuevas entradas.")
                price_val = None
                try:
                    price_cached = self.price_cache.get(self.config.get('symbol', 'BTC/USDT'))
                    price_val = float(price_cached) if price_cached is not None else None
                except Exception:
                    price_val = None
                ctx_overrides = {"freeze_90": True}
                _emit_motive(ctx_overrides, price_value=price_val)
                return

            if cooldown_active:
                logging.info("Cooldown activo, no se evaluarán nuevas entradas.")
                price_val = None
                try:
                    price_cached = self.price_cache.get(self.config.get('symbol', 'BTC/USDT'))
                    price_val = float(price_cached) if price_cached is not None else None
                except Exception:
                    price_val = None
                ctx_overrides = {"cooldown": True}
                _emit_motive(ctx_overrides, price_value=price_val)
                return

            if self.pause_manager.is_paused_now():
                until = self.pause_manager.get_pause_until()
                self.is_paused = True
                logging.info(
                    "El bot está en pausa hasta %s. No se buscarán nuevas señales.",
                    until,
                )
                reason_msg = (
                    f"bot en pausa hasta {until.strftime('%Y-%m-%d %H:%M UTC')}"
                    if until is not None
                    else "bot en pausa"
                )
                _emit_motive({"reasons": [reason_msg]})
                return

            if self.is_paused:
                logging.info("El bot está en pausa, no se buscan nuevas señales.")
                _emit_motive({"reasons": ["bot en pausa"]})
                return

            ban_hours_cfg = self.config.get("ban_hours", "")
            if isinstance(ban_hours_cfg, (list, tuple, set)):
                try:
                    ban_hours_str = ",".join(
                        str(int(h))
                        for h in ban_hours_cfg
                        if h is not None and str(h).strip() != ""
                    )
                except Exception:
                    ban_hours_str = ""
            else:
                ban_hours_str = str(ban_hours_cfg or "")

            if ban_hours_str and in_ban_hours(ban_hours_str):
                logging.info("[RISK] Hora baneada (UTC). No se abre nueva posición.")
                _emit_motive({"reasons": ["hora baneada"]})
                return

            klines_1h = await self.exchange.get_klines('1h')
            klines_4h = await self.exchange.get_klines('4h')

            if not klines_1h or not klines_4h:
                logging.warning("No se pudieron obtener los datos de mercado en este ciclo.")
                _emit_motive({"reasons": ["exchange_error: sin datos de klines"]})
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
                price_stop = last_candle.get('close')
                try:
                    price_stop = float(price_stop) if price_stop is not None else None
                except Exception:
                    price_stop = None
                ctx_overrides = {"risk_ok": False, "reasons": ["Daily stop alcanzado"]}
                _emit_motive(ctx_overrides, price_value=price_stop)
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
            if signal:
                side_pref = signal
                self.side_pref = signal
            base_ctx["reasons"] = []

            price_signal = _safe_float_local(last_candle.get("close"))
            ema200_1h_val = _safe_float_local(last_candle.get("ema200"))
            ema200_4h_val = _safe_float_local(last_candle.get("ema200_4h"))
            rsi4h_val = _safe_float_local(last_candle.get("rsi4h"))
            atr_val = _safe_float_local(last_candle.get("atr"))
            adx_val = _safe_float_local(last_candle.get("adx"))

            anchor_val = None
            try:
                anchor_raw = self.strategy._anchor_price(last_candle)
            except Exception:
                anchor_raw = None
            anchor_val = _safe_float_local(anchor_raw)

            step_val = None
            span_val = None
            if atr_val is not None:
                step_mult_cfg = _safe_float_local(self.config.get("grid_step_atr", 0.32))
                span_mult_cfg = _safe_float_local(self.config.get("grid_span_atr", 3.0))
                if step_mult_cfg is not None:
                    step_val = step_mult_cfg * atr_val
                if span_mult_cfg is not None:
                    span_val = span_mult_cfg * atr_val
                if (
                    signal in ("LONG", "SHORT")
                    and price_signal is not None
                    and anchor_val is not None
                    and step_val is not None
                    and span_val is not None
                ):
                    try:
                        anchor_adj, step_adj, span_adj, _ = self.strategy._apply_anchor_freeze(
                            signal,
                            price_signal,
                            anchor_val,
                            step_val,
                            span_val,
                        )
                        anchor_val = _safe_float_local(anchor_adj)
                        step_val = _safe_float_local(step_adj)
                        span_val = _safe_float_local(span_adj)
                    except Exception:
                        pass

            atrp_val = None
            if (
                atr_val is not None
                and price_signal is not None
                and price_signal != 0.0
            ):
                try:
                    atrp_val = atr_val / price_signal
                except Exception:
                    atrp_val = None

            gate_ok_val = None
            gate_bps_cfg = self.config.get("funding_gate_bps")
            rate_now_val = self.config.get("_funding_rate_now")
            if (
                gate_bps_cfg is not None
                and rate_now_val is not None
                and signal in ("LONG", "SHORT")
            ):
                try:
                    g_frac = float(gate_bps_cfg) / 10000.0
                    r_dec = float(rate_now_val)
                    gate_ok_val = not (
                        (signal == "LONG" and r_dec > g_frac)
                        or (signal == "SHORT" and r_dec < -g_frac)
                    )
                except (TypeError, ValueError):
                    gate_ok_val = None

            base_ctx.update(
                {
                    "price": price_signal,
                    "ema200_1h": ema200_1h_val,
                    "ema200_4h": ema200_4h_val,
                    "rsi4h": rsi4h_val,
                    "atrp": atrp_val,
                    "adx": adx_val,
                    "anchor": anchor_val,
                    "step": step_val,
                    "span": span_val,
                    "gate_ok": gate_ok_val,
                }
            )

            anchor_epoch = None
            try:
                anchor_epoch = int(now_ts.value // 1_000_000_000)
            except Exception:
                anchor_epoch = None
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
                try:
                    row = data.iloc[-1] if len(data) else None

                    price = _safe_float_local(row.get("close")) if row is not None else None
                    ema200_1h = _safe_float_local(row.get("ema200")) if row is not None else None
                    ema200_4h = _safe_float_local(row.get("ema200_4h")) if row is not None else None
                    rsi4h = _safe_float_local(row.get("rsi4h")) if row is not None else None
                    atr = _safe_float_local(row.get("atr")) if row is not None else None
                    adx = _safe_float_local(row.get("adx")) if row is not None else None

                    side_pref = None
                    if row is not None:
                        try:
                            side_pref = self.strategy._decide_grid_side(row)
                        except Exception:
                            side_pref = None

                    anchor = None
                    if row is not None:
                        try:
                            anchor_val = self.strategy._anchor_price(row)
                        except Exception:
                            anchor_val = None
                        anchor = _safe_float_local(anchor_val)

                    step = None
                    span = None
                    if atr is not None:
                        step_mult = _safe_float_local(self.config.get("grid_step_atr", 0.32))
                        span_mult = _safe_float_local(self.config.get("grid_span_atr", 3.0))
                        if step_mult is not None:
                            step = step_mult * atr
                        if span_mult is not None:
                            span = span_mult * atr

                    if (
                        price is not None
                        and anchor is not None
                        and step is not None
                        and span is not None
                    ):
                        try:
                            anchor_frozen, step_frozen, span_frozen, _ = self.strategy._apply_anchor_freeze(
                                side_pref if side_pref in ("LONG", "SHORT") else None,
                                price,
                                anchor,
                                step,
                                span,
                            )
                            anchor = _safe_float_local(anchor_frozen)
                            step = _safe_float_local(step_frozen)
                            span = _safe_float_local(span_frozen)
                        except Exception:
                            pass

                    atrp = None
                    if atr is not None and price is not None and price != 0.0:
                        atrp = atr / price

                    gate_ok = None
                    gate_bps = self.config.get("funding_gate_bps")
                    rate_now = self.config.get("_funding_rate_now")
                    if (
                        gate_bps is not None
                        and rate_now is not None
                        and side_pref in ("LONG", "SHORT")
                    ):
                        try:
                            g_frac = float(gate_bps) / 10000.0
                            r_dec = float(rate_now)
                            gate_ok = not (
                                (side_pref == "LONG" and r_dec > g_frac)
                                or (side_pref == "SHORT" and r_dec < -g_frac)
                            )
                        except (TypeError, ValueError):
                            gate_ok = None

                    reasons_list = []
                    try:
                        reasons_list = self.strategy.get_rejection_reasons_all(data)
                    except Exception:
                        reasons_list = []
                    reasons = [f"{c}: {d}" for c, d in reasons_list if c or d]
                    overrides_ctx = {
                        "ema200_1h": ema200_1h,
                        "ema200_4h": ema200_4h,
                        "rsi4h": rsi4h,
                        "adx": adx,
                        "anchor": anchor,
                        "step": step,
                        "span": span,
                        "atrp": atrp,
                        "gate_ok": gate_ok,
                        "reasons": reasons,
                    }
                    _emit_motive(overrides_ctx, price_value=price, side_value=side_pref)
                except Exception as _e:
                    logging.debug("No se pudo registrar motivo: %s", _e)
                return

            symbol_name = str(self.config.get("symbol", "BTC/USDT"))
            if not self._acquire_entry_lock(symbol_name, signal, anchor_epoch):
                logging.info(
                    "Entrada duplicada bloqueada por entry_lock (symbol=%s side=%s anchor_epoch=%s)",
                    symbol_name,
                    signal,
                    anchor_epoch,
                )
                reason_msg = (
                    f"entry_lock activo (anchor_epoch={anchor_epoch})"
                    if anchor_epoch is not None
                    else "entry_lock activo"
                )
                _emit_motive({"reasons": [reason_msg]}, price_value=price_signal, side_value=signal)
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
                _emit_motive({"reasons": ["exchange_error: precio de entrada no disponible"]}, side_value=signal)
                return
            entry_price = float(entry_price)
            qty = (eq_on_open * leverage) / max(entry_price, 1e-12)
            if qty <= 0:
                logging.warning(
                    "Qty=0: eq=%.2f lev=%.2f px=%.2f → no se abre.",
                    eq_on_open,
                    leverage,
                    entry_price,
                )
                return

            sl_price = self.strategy.calculate_sl(entry_price, last_candle, signal, eq_on_open)
            side = signal
            tp_price = self.strategy.calculate_tp(entry_price, qty, eq_on_open, side, leverage)

            order_result = await self.exchange.create_order(signal, qty, sl_price, tp_price)

            # 1) Verificar FILL > 0 (evita anuncios falsos)
            filled = 0.0
            try:
                filled = float(
                    (order_result or {}).get("executedQty")
                    or (order_result or {}).get("filled")
                    or (order_result or {}).get("size")
                    or 0.0
                )
            except Exception:
                filled = 0.0
            if filled <= 0:
                logging.warning("Orden enviada pero SIN FILL (>0). No se anuncia apertura.")
                return

            # 2) Esperar a que el store (POSITION_SERVICE) deje de estar FLAT
            st = None
            if trading.POSITION_SERVICE is not None:
                for _ in range(6):  # espera total ~300 ms
                    try:
                        st = trading.POSITION_SERVICE.get_status()
                        if st and str(st.get("side", "FLAT")).upper() != "FLAT":
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)

            # 3) Cachear desde el store si está disponible
            cached_position = order_result
            try:
                if st and str(st.get("side", "FLAT")).upper() != "FLAT":
                    cached_position = {
                        "symbol": st.get("symbol", self.config.get("symbol")),
                        "side": str(st.get("side")).upper(),
                        "contracts": float(st.get("qty") or 0.0),
                        "entryPrice": float(st.get("entry_price") or 0.0),
                        "markPrice": float(st.get("mark") or 0.0),
                    }
            except Exception as e:
                logging.debug("post-open cache position fail: %s", e)
            await self.trader.set_position(cached_position)

            # 4) Ahora sí: anunciar la apertura (usá tu formato nuevo si ya lo tenés)
            base = str(self.config.get("symbol", "BTC/USDT")).split("/")[0]
            mode_txt = "real" if not S.PAPER else "simulado"
            await self.notifier.send(
                "🚀 operacion "
                + f"{base} Abierta: {signal} ({mode_txt})\n"
                + f"Apalancamiento: x{leverage:.1f}\n"
                + f"precio: ${entry_price:.2f}\n"
                + f"tp : ${tp_price:.2f}\n"
                + f"sl:  ${sl_price:.2f}"
            )
            self._risk_usd_trade = abs(entry_price - sl_price) * qty
            self._eq_on_open = eq_on_open
            self._entry_ts = now_ts
            self._bars_in_position = 0
            symbol = str(self.config.get("symbol", ""))
            if symbol:
                for s in ("LONG", "SHORT"):
                    FREEZER.clear(symbol, cast(Side, s))
            return

            _emit_motive({}, side_value=signal)
            return

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

    def has_open_position(self) -> bool:
        """Indica si el estado local del bot registra una posición abierta."""

        try:
            trading.ensure_initialized()
        except Exception:
            return False

        service = getattr(trading, "POSITION_SERVICE", None)
        if service is None:
            return False

        try:
            status = service.get_status() or {}
        except Exception as exc:
            self.logger.debug("has_open_position: no se pudo obtener status: %s", exc)
            return False

        side = str(status.get("side", "FLAT")).upper()
        try:
            qty = float(status.get("qty") or status.get("pos_qty") or 0.0)
        except Exception:
            qty = 0.0
        return side != "FLAT" and abs(qty) > 0.0

    def sync_live_position(self) -> bool:
        """
        Trae la posición LIVE del exchange y la sincroniza al estado local del bot.
        Devuelve True si encontró y tomó control de una posición; False si no hay.
        """

        symbol = getattr(self, "symbol", None) or self.config.get("symbol", "BTC/USDT")
        trading.ensure_initialized()

        broker = getattr(trading, "BROKER", None)
        client = getattr(broker, "client", None) if broker is not None else None
        if client is None:
            client = getattr(trading, "ACTIVE_LIVE_CLIENT", None)
        if client is None:
            client = ACTIVE_LIVE_CLIENT
        if client is None:
            raise RuntimeError("No hay cliente LIVE disponible para sincronizar la posición.")

        symbol_no_slash = str(symbol).replace("/", "").upper()

        def _call_positions():
            if hasattr(client, "futures_position_information"):
                return client.futures_position_information(symbol=symbol_no_slash)
            if hasattr(client, "fapiPrivate_get_positionrisk"):
                return client.fapiPrivate_get_positionrisk({"symbol": symbol_no_slash})
            if hasattr(client, "fapiPrivateGetPositionRisk"):
                return client.fapiPrivateGetPositionRisk({"symbol": symbol_no_slash})
            raise RuntimeError("El cliente live no permite consultar posiciones abiertas.")

        try:
            raw_positions = _call_positions()
        except Exception as exc:
            self.logger.exception("sync_live_position: error consultando broker: %s", exc)
            raise

        if raw_positions is None:
            entries: list[dict[str, Any]] = []
        elif isinstance(raw_positions, list):
            entries = [entry for entry in raw_positions if isinstance(entry, dict)]
        elif isinstance(raw_positions, dict):
            entries = [raw_positions]
        else:
            entries = []

        def _as_float(entry: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
            for key in keys:
                if entry.get(key) is None:
                    continue
                try:
                    return float(entry[key])
                except Exception:
                    continue
            return float(default)

        amount = 0.0
        entry_price = 0.0
        leverage = 1.0
        mark_price = 0.0
        for entry in entries:
            entry_symbol = str(entry.get("symbol") or entry.get("symbolName") or "").upper()
            if entry_symbol != symbol_no_slash:
                continue
            amt = _as_float(entry, "positionAmt", "position_amt", "amount", "qty")
            if abs(amt) <= 0.0:
                continue
            amount = amt
            entry_price = _as_float(entry, "entryPrice", "entry_price", "avgPrice", "avgEntryPrice")
            leverage = max(1.0, _as_float(entry, "leverage"))
            mark_price = _as_float(entry, "markPrice", "mark_price", default=entry_price)
            position_side = str(entry.get("positionSide", "")).upper()
            if position_side in {"LONG", "SHORT", "BOTH"}:
                break

        service = getattr(trading, "POSITION_SERVICE", None)
        store = getattr(service, "store", None)

        if abs(amount) <= 0.0:
            self.logger.info(
                "sync_live_position: sin posición LIVE para %s. Limpiando estado local.",
                symbol,
            )
            if store is not None:
                try:
                    store.save(pos_qty=0.0, avg_price=0.0)
                except Exception:
                    self.logger.debug("No se pudo limpiar store al sincronizar.", exc_info=True)
            try:
                state = load_state()
                open_positions = state.get("open_positions", {})
                if open_positions.pop(symbol, None) is not None:
                    save_state(state)
            except Exception:
                self.logger.debug("No se pudo limpiar state_store al sincronizar.", exc_info=True)
            try:
                if hasattr(self.trader, "_open_position"):
                    self.trader._open_position = None
            except Exception:
                self.logger.debug("No se pudo limpiar cache de trader al sincronizar.", exc_info=True)
            self.position_open = False
            return False

        mode = "live" if self.is_live else "paper"
        bot_id = get_bot_id()
        sym_clean = symbol.replace("/", "").upper()
        sym = symbol
        qty_bot, avg_bot = bot_position(mode, bot_id, sym_clean)

        if abs(qty_bot) <= 0.0:
            qty_bot, avg_bot = bot_position(mode, bot_id, symbol)

        if abs(qty_bot) <= 0.0:
            try:
                if store is not None:
                    store.save(pos_qty=0.0, avg_price=0.0)
            except Exception:
                self.logger.debug("No se pudo limpiar store al sincronizar.", exc_info=True)
            try:
                state = load_state()
                open_positions = state.get("open_positions", {})
                if open_positions.pop(sym, None) is not None:
                    save_state(state)
            except Exception:
                self.logger.debug("No se pudo limpiar state_store al sincronizar.", exc_info=True)
            try:
                if hasattr(self.trader, "_open_position"):
                    self.trader._open_position = None
            except Exception:
                self.logger.debug("No se pudo limpiar cache de trader al sincronizar.", exc_info=True)
            self.position_open = False
            self.logger.info(
                "sync_live_position: el bot está FLAT (puede haber posiciones manuales en la cuenta)."
            )
            return False

        leverage_i = int(leverage) if leverage >= 1 else 1
        mark_val = float(mark_price or avg_bot or entry_price or 0.0)
        side_bot = "LONG" if qty_bot > 0 else "SHORT"
        qty_abs = abs(float(qty_bot))
        avg_bot_f = float(avg_bot)

        if store is not None:
            try:
                store.save(pos_qty=float(qty_bot), avg_price=avg_bot_f, mark=mark_val)
            except Exception:
                self.logger.debug("No se pudo actualizar el store con la posición live.", exc_info=True)

        try:
            pos = create_position(
                symbol=sym,
                side=side_bot,
                qty=qty_abs,
                entry_price=avg_bot_f,
                leverage=float(leverage_i),
                mode="live",
            )
            persist_open(pos)
        except Exception:
            self.logger.debug("No se pudo persistir la posición live en state_store.", exc_info=True)

        try:
            if hasattr(self.trader, "_open_position"):
                self.trader._open_position = {
                    "symbol": sym,
                    "side": side_bot,
                    "contracts": qty_abs,
                    "entryPrice": avg_bot_f,
                    "markPrice": mark_val,
                }
        except Exception:
            self.logger.debug("No se pudo actualizar la cache del trader con la posición live.", exc_info=True)

        self.position_open = True
        self.logger.info(
            "sync_live_position: posición del BOT adoptada %s %.6f @ %.2f",
            side_bot,
            qty_abs,
            avg_bot_f,
        )
        return True

    async def _preload_position_from_store(self) -> None:
        if trading.POSITION_SERVICE is None:
            return
        try:
            status = trading.POSITION_SERVICE.get_status()
        except Exception as exc:
            logging.debug("Preload posición falló: %s", exc)
            return

        side = (status.get("side") or "FLAT").upper()
        if side == "FLAT":
            return

        symbol = status.get("symbol", self.config.get("symbol", "BTC/USDT"))
        qty = float(status.get("qty") or status.get("size") or 0.0)
        # Precarga de posición: SOLO en SIM. En REAL validar con exchange primero.
        if S.PAPER:
            position = {
                "symbol": symbol,
                "side": side,
                "contracts": qty,
                "entryPrice": float(status.get("entry_price") or 0.0),
                "markPrice": float(status.get("mark") or 0.0),
            }
            await self.trader.set_position(position)
            logging.info("Posición precargada desde store (SIM): %s %.6f %s", side, qty, symbol)
        else:
            try:
                ex_pos = await self.exchange.get_open_position(self.config.get("symbol"))
                ex_qty = float(
                    (ex_pos or {}).get("contracts")
                    or (ex_pos or {}).get("positionAmt")
                    or (ex_pos or {}).get("size")
                    or 0.0
                )
                live_has_open = abs(ex_qty) > 0.0
            except Exception as e:
                logging.debug("Validación de posición en exchange falló: %s", e)
                live_has_open = False

            if live_has_open:
                await self.trader.set_position(ex_pos)
                logging.info("Posición precargada desde EXCHANGE (REAL): %s", ex_pos)
            else:
                # No hay posición live → NO precargar, y limpiar cualquier rastro local
                try:
                    if hasattr(self.trader, "_open_position"):
                        self.trader._open_position = None
                except Exception:
                    pass
                logging.info("Sin posición live en exchange (REAL): no se precarga nada")

    def run(self):
        job_queue = self.telegram_app.job_queue
        job_queue.run_repeating(self._update_price_cache_job, interval=10, first=1)
        job_queue.run_repeating(self.trading_loop, interval=60, first=5)

        loop = asyncio.get_event_loop()
        bootstrap_tasks = [
            self.exchange.set_position_mode(one_way=not self.exchange.hedge_mode),
            self._preload_position_from_store(),
        ]
        for task in bootstrap_tasks:
            if loop.is_running():
                loop.create_task(task)
            else:
                loop.run_until_complete(task)

        mode_msg = "🧪 Modo SIMULADO activo" if S.PAPER else "🔴 Modo REAL activo"
        if S.PAPER:
            try:
                equity = float(getattr(trading.BROKER, "equity"))
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

    async def close_all(self) -> bool:
        """Cierra SOLO la posición del BOT (reduceOnly). No toca posiciones manuales del usuario."""
        try:
            symbol = (self.config or {}).get("symbol", "BTC/USDT")

            ex = getattr(self, "exchange", None)

            async def _safe_fetch_balance() -> Optional[float]:
                if ex is None:
                    return None
                fetcher = getattr(ex, "fetch_balance_usdt", None)
                if not callable(fetcher):
                    return None
                try:
                    result = fetcher()
                    if inspect.isawaitable(result):
                        result = await result
                    return float(result)
                except Exception:
                    return None

            # --- snapshot antes del cierre (para PnL por delta de balance) ---
            bal_before = await _safe_fetch_balance()

            # --- cerrar ahora (reduceOnly + positionSide correcto) ---
            await asyncio.to_thread(trading.close_now, symbol)

            # --- snapshot después del cierre ---
            bal_after = await _safe_fetch_balance()

            # --- persistir trade si podemos calcular PnL ---
            pnl_real = None
            if bal_before is not None and bal_after is not None:
                pnl_real = float(bal_after) - float(bal_before)

            storage = getattr(self, "storage", None)
            if storage and pnl_real is not None:
                side = "LONG"
                try:
                    from position_service import POSITION_SERVICE

                    status = POSITION_SERVICE.get_status() if POSITION_SERVICE else {}
                    side = (status.get("side") or side).upper()
                except Exception:
                    pass
                storage.persist_trade(
                    {
                        "symbol": symbol,
                        "side": side,
                        "pnl": pnl_real,
                        "note": "close_all/manual",
                    }
                )
            return True
        except Exception as exc:
            self.logger.exception("close_all failed: %s", exc)
            return False

    def _init_db(self):
        pass

    def _csv_paths(self):
        cfg = getattr(self, "config", {}) or {}
        persistence_cfg = cfg.get("persistence", {}) if isinstance(cfg, dict) else {}
        base_dir = persistence_cfg.get("dir", "data")
        equity_path = persistence_cfg.get("equity_csv", os.path.join(base_dir, "equity.csv"))
        trades_path = persistence_cfg.get("trades_csv", os.path.join(base_dir, "trades.csv"))
        return equity_path, trades_path

    def _sum_income(self, income_type: str, start_ms: int, end_ms: int) -> float:
        if S.PAPER or ACTIVE_LIVE_CLIENT is None:
            return 0.0
        client = ACTIVE_LIVE_CLIENT
        total = 0.0
        cursor = start_ms
        try:
            while True:
                batch = client.futures_income_history(  # type: ignore[attr-defined]
                    startTime=cursor,
                    endTime=end_ms,
                    incomeType=income_type,
                    limit=1000,
                )
                if not batch:
                    break
                last_ts = None
                for row in batch:
                    ts = int(row.get("time", 0))
                    if ts > end_ms:
                        continue
                    try:
                        total += float(row.get("income", 0.0))
                    except Exception:
                        continue
                    last_ts = ts if last_ts is None else max(last_ts, ts)
                if last_ts is None or last_ts >= end_ms or len(batch) < 1000:
                    break
                cursor = last_ts + 1
        except Exception as exc:
            logging.debug("income_history(%s) falló: %s", income_type, exc)
        return total

    def _generate_period_report(self, days: int):
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo("America/Argentina/Buenos_Aires")
        except Exception:
            tz = None
        try:
            import pandas as pd

            equity_csv, trades_csv = self._csv_paths()
            equity_ini = equity_fin = pnl = 0.0
            total_trades = wins = losses = 0

            try:
                df_eq = pd.read_csv(equity_csv, parse_dates=["ts"])
                df_eq["ts"] = pd.to_datetime(df_eq["ts"], utc=True, errors="coerce")
                now = (
                    pd.Timestamp.now(tz=tz).tz_convert("UTC")
                    if tz
                    else pd.Timestamp.utcnow()
                )
                since = now - pd.Timedelta(days=days)
                df_window = df_eq[df_eq["ts"] >= since]
                if not df_window.empty:
                    equity_ini = float(df_window["equity"].iloc[0])
                    equity_fin = float(df_window["equity"].iloc[-1])
                    if "pnl" in df_window.columns:
                        pnl = float(df_window["pnl"].sum())
                    else:
                        pnl = equity_fin - equity_ini
            except Exception:
                pass

            try:
                df_tr = pd.read_csv(trades_csv, parse_dates=["ts"])
                df_tr["ts"] = pd.to_datetime(df_tr["ts"], utc=True, errors="coerce")
                now = (
                    pd.Timestamp.now(tz=tz).tz_convert("UTC")
                    if tz
                    else pd.Timestamp.utcnow()
                )
                since = now - pd.Timedelta(days=days)
                df_trades = df_tr[df_tr["ts"] >= since]
                if not df_trades.empty:
                    total_trades = int(len(df_trades))
                    pnl_col = df_trades.get("pnl")
                    if pnl_col is not None:
                        wins = int((pnl_col > 0).sum())
                        losses = int((pnl_col < 0).sum())
            except Exception:
                pass

            title = "🗓️ Reporte 24h" if days == 1 else "📈 Reporte 7d"
            msg = (
                f"{title}\n"
                f"Equity inicial: ${equity_ini:,.2f}\n"
                f"Equity final:   ${equity_fin:,.2f}\n"
                f"PnL neto:       ${pnl:,.2f}\n"
                f"Trades: {total_trades} (W:{wins}/L:{losses})"
            )
            if not S.PAPER:
                try:
                    start_ms = int(since.timestamp() * 1000)
                    end_ms = int(now.timestamp() * 1000)
                    fees_real = self._sum_income("COMMISSION", start_ms, end_ms)
                    funding_real = self._sum_income("FUNDING_FEE", start_ms, end_ms)
                    msg += (
                        f"\nFees cobradas:   ${fees_real:,.2f}\n"
                        f"Funding neto:    ${funding_real:,.2f}"
                    )
                except Exception:
                    logging.debug("No se pudieron obtener fees/funding reales", exc_info=True)

            if hasattr(self, "notifier") and self.notifier:
                try:
                    coro = self.notifier.send(msg)
                    if asyncio.iscoroutine(coro):
                        asyncio.create_task(coro)
                except Exception:
                    pass
            else:
                logging.info(msg)
        except Exception:
            logging.warning("No se pudo generar el reporte de periodo.", exc_info=True)

    def _generate_daily_report(self):
        self._generate_period_report(1)

    def _generate_weekly_report(self):
        self._generate_period_report(7)

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
