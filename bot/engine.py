import logging
import math
import asyncio
import os
import threading
import inspect
import time
from collections import deque
from typing import Any, Dict, Optional, cast
from time import monotonic, time as _now

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback when zoneinfo is unavailable
    ZoneInfo = None  # type: ignore[assignment]
import pandas as pd
import schedule
from telegram.ext import ContextTypes

from anchor_freezer import Side
from deps import FREEZER
from bot.exchange import Exchange
from bot.exchange_client import ensure_position_mode
from bot.trader import Trader
from bot.storage import Storage
from bot.telemetry.notifier import Notifier
from bot.telemetry.telegram_bot import setup_telegram_bot
from bot.health.endpoint import attach_to_application
from brokers import ACTIVE_LIVE_CLIENT
from bot.identity import get_bot_id, make_client_oid
from bot.ledger import bot_position, prune_open_older_than, init as ledger_init
from state_store import create_position, load_state, persist_open, save_state
from bot.motives import MOTIVES, MotiveItem, compute_codes
from core.strategy import Strategy
from core.indicators import add_indicators
from config import S
import trading
from position_service import reconcile_bot_store_with_account
from risk_guards import (
    clear_pause_if_expired,
    get_pause_manager,
    in_ban_hours,
    is_paused_now,
)
from reanudar_listener import listen_reanudar
from bot.runtime_state import get_mode as runtime_get_mode
from paths import get_data_dir


def _compute_order_qty_from_equity(
    equity_usdt: float,
    price: float,
    leverage: float,
    risk_pct: float = 1.0,
) -> float:
    try:
        eq = float(equity_usdt)
        px = float(price)
        lev = float(leverage)
        risk = float(risk_pct)
    except Exception:
        return 0.0
    if eq <= 0 or px <= 0 or risk <= 0 or lev <= 0:
        return 0.0
    notional = eq * risk * lev
    return notional / px


def _quantize_amount(
    filters: Optional[Dict[str, float]],
    raw_qty: float,
    price: Optional[float] = None,
) -> float:
    try:
        qty = float(raw_qty)
    except Exception:
        return 0.0
    if qty <= 0 or not math.isfinite(qty):
        return 0.0
    if not filters:
        return qty

    step = float(filters.get("stepSize", 0.0) or 0.0)
    min_qty = float(filters.get("minQty", 0.0) or 0.0)
    min_notional = float(filters.get("minNotional", 0.0) or 0.0)

    if step > 0:
        qty = math.floor(qty / step) * step

    if min_qty > 0 and qty < min_qty:
        return 0.0

    if min_notional > 0 and price is not None:
        try:
            if qty * float(price) < min_notional:
                return 0.0
        except Exception:
            return 0.0

    return qty


def _runtime_mode_value() -> str:
    try:
        return (runtime_get_mode() or "paper").lower()
    except Exception:
        return "paper"


class _EngineBrokerAdapter:
    def __init__(self, app: "TradingApp"):
        self.app = app

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int = 1,
        **extra: Any,
    ):
        trading.ensure_initialized()
        ledger_init()

        sym = symbol or self.app.config.get("symbol", "BTC/USDT")
        sym_clean = sym.replace("/", "").upper()
        mode = "live" if self.app.is_live else "paper"
        side_u = str(side).upper()

        bot_id = get_bot_id()
        client_oid = make_client_oid(bot_id, sym_clean, mode)

        order_kwargs = dict(extra or {})
        order_kwargs.setdefault("symbol", sym)
        order_kwargs.setdefault("leverage", int(leverage))
        order_kwargs.setdefault("newClientOrderId", client_oid)
        order_kwargs.setdefault("order_type", "MARKET")

        result = await asyncio.to_thread(
            trading.place_order_safe,
            side_u,
            float(quantity),
            None,
            **order_kwargs,
        )

        if isinstance(result, dict):
            result.setdefault("newClientOrderId", client_oid)
            result.setdefault("clientOrderId", client_oid)
        return result


class TradingApp:
    def __init__(self, cfg, mode_source: str | None = None):
        self.config = cfg
        runtime_mode = _runtime_mode_value()
        default_trading_mode = "real" if runtime_mode in {"real", "live"} else "simulado"
        resolved_mode = str(self.config.get("trading_mode", default_trading_mode)).lower()
        if resolved_mode not in {"real", "live"}:
            resolved_mode = "simulado"
        effective_mode = "real" if resolved_mode in {"real", "live"} else "simulado"
        self.config["trading_mode"] = "real" if effective_mode == "real" else "simulado"
        self.config["mode"] = "real" if effective_mode == "real" else "paper"
        self.config.setdefault("start_equity", S.start_equity)
        self.logger = logging.getLogger(__name__)
        self.log = self.logger

        trading.ensure_initialized(mode=effective_mode)
        self.mode_source = mode_source or trading.LAST_MODE_CHANGE_SOURCE or "startup"
        self.data_dir = trading.ACTIVE_DATA_DIR or get_data_dir()
        self.store_path = trading.ACTIVE_STORE_PATH
        self.mode = "live" if effective_mode == "real" else "paper"
        self.last_signal_ts = None

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
        attach_to_application(self)
        self.notifier = Notifier(application=self.telegram_app, cfg=self.config)
        self.symbols = [self.config.get('symbol', 'BTC/USDT')]
        self.price_cache = {}
        self._funding_series = None
        self._daily_R: dict = {}
        self._risk_usd_trade = 0.0
        self._eq_on_open = 0.0
        self._entry_ts = None
        self._bars_in_position = 0
        self.manual_block_until = 0
        try:
            self.sl_equity_pct = float(
                self.config.get(
                    "sl_equity_pct",
                    os.getenv("DEFAULT_SL_EQUITY_PCT", "10"),
                )
            )
        except Exception:
            self.sl_equity_pct = 10.0
        try:
            self.tp_equity_pct = float(self.config.get("tp_equity_pct", 0.0) or 0.0)
        except Exception:
            self.tp_equity_pct = 0.0
        try:
            self.max_position_hours = int(self.config.get("max_position_hours", 16))
        except Exception:
            self.max_position_hours = 16
        self._entry_locks: Dict[tuple[str, str, int], float] = {}
        self._entry_lock_gc_last = monotonic()
        try:
            ttl_cfg = cfg.get("entry_lock_ttl_seconds", 12 * 60 * 60)
            self._entry_lock_ttl_s = max(int(float(ttl_cfg)), 60)
        except Exception:
            self._entry_lock_ttl_s = 12 * 60 * 60
        self._shock_guard_last_trade_id: Optional[Any] = None

        buenos_aires_tz = None
        if ZoneInfo is not None:
            try:
                buenos_aires_tz = ZoneInfo("America/Argentina/Buenos_Aires")
            except Exception:  # pragma: no cover - defensive
                buenos_aires_tz = None

        def _schedule_at(job, time_str: str, callback):
            try:
                if buenos_aires_tz is not None:
                    job = job.at(time_str, timezone=buenos_aires_tz)
                else:
                    job = job.at(time_str)
            except TypeError:
                job = job.at(time_str)
            job.do(callback)

        _schedule_at(schedule.every().day, "07:00", self._generate_daily_report)
        _schedule_at(schedule.every().sunday, "07:01", self._generate_weekly_report)
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

    def set_mode(self, mode: str, *, source: str = "engine") -> str:
        """
        Cambia modo del bot sin tocar propiedades de solo-lectura.
        Acepta: 'live'|'real'|'paper'|'simulado'|'sim'.
        Propaga a trader/strategy/exchange si esos objetos exponen 'mode' o setters similares.
        Devuelve el modo normalizado ('live'|'paper') activo tras el cambio.
        """

        m = (mode or "").lower()
        target = "live" if m in ("live", "real") else "paper"
        current = "live" if self.is_live else "paper"

        # bandera propia
        try:
            self.mode = target
        except Exception:
            pass

        # flags tipicas / legacy
        for flag, val in (("PAPER", target == "paper"), ("paper", target == "paper")):
            try:
                if hasattr(self, flag):
                    setattr(self, flag, val)
            except Exception:
                self.logger.debug("No se pudo actualizar flag %s", flag, exc_info=True)

        # Actualizar diccionario de config si aplica
        try:
            if isinstance(self.config, dict):
                self.config["mode"] = "real" if target == "live" else "paper"
                self.config["trading_mode"] = "real" if target == "live" else "simulado"
        except Exception:
            self.logger.debug("No se pudo reflejar modo en config", exc_info=True)

        # Propagar a subcomponentes comunes
        for comp_name in ("trader", "exchange", "strategy"):
            comp = getattr(self, comp_name, None)
            if not comp:
                continue
            try:
                if hasattr(comp, "mode"):
                    setattr(comp, "mode", target)
            except Exception:
                self.logger.debug("No se pudo fijar atributo mode en %s", comp_name, exc_info=True)
            for fn in (
                "set_mode",
                "set_trading_mode",
                "switch_mode",
                "enable_live",
                "enable_real",
                "enable_paper",
                "set_paper",
            ):
                try:
                    fn_ref = getattr(comp, fn, None)
                    if callable(fn_ref):
                        fn_ref(target)
                except Exception:
                    self.logger.debug(
                        "No se pudo propagar %s.%s(%s)", comp_name, fn, target, exc_info=True
                    )

        # Sincronizar stack de trading principal (usa ModeResult para logs)
        desired_trading_mode = "real" if target == "live" else "simulado"
        try:
            result = trading.set_trading_mode(desired_trading_mode, source=source)
            if hasattr(result, "ok") and not result.ok:
                self.logger.warning("set_trading_mode devolvió error: %s", getattr(result, "msg", ""))
            if hasattr(result, "ok") and result.ok:
                self.mode_source = getattr(trading, "LAST_MODE_CHANGE_SOURCE", source)
                self.store_path = trading.ACTIVE_STORE_PATH
                self.data_dir = trading.ACTIVE_DATA_DIR or self.data_dir
        except Exception:
            self.logger.warning(
                "No se pudo sincronizar trading.set_trading_mode(%s)",
                desired_trading_mode,
                exc_info=True,
            )

        return current if current == target else target

    def set_sl_equity_pct(self, pct: float) -> None:
        try:
            value = max(0.1, float(pct))
        except Exception:
            value = 10.0
        self.sl_equity_pct = value
        if isinstance(self.config, dict):
            self.config["sl_equity_pct"] = float(value)

    def set_tp_equity_pct(self, pct: float | None) -> None:
        try:
            value = max(0.0, float(pct or 0.0))
        except Exception:
            value = 0.0
        self.tp_equity_pct = value
        if isinstance(self.config, dict):
            self.config["tp_equity_pct"] = float(value)

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
        cutoff = now - float(ttl)
        stale = [key for key, ts in self._entry_locks.items() if ts < cutoff]
        for key in stale:
            self._entry_locks.pop(key, None)
        self._entry_lock_gc_last = now

    def _start_reanudar_listener(self) -> None:
        if self._reanudar_thread is not None and self._reanudar_thread.is_alive():
            return

        if self.telegram_app is not None:
            logging.debug(
                "Reanudar listener integrado al bot de Telegram; se omite polling paralelo.")
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
        now = monotonic()
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
            symbol_cfg = self.config.get("symbol", "BTC/USDT")
            try:
                await reconcile_bot_store_with_account(
                    self.trader,
                    self.exchange,
                    symbol_cfg,
                    0.0,
                )
            except Exception:
                self.logger.debug(
                    "No se pudo reconciliar posición del BOT con la cuenta.",
                    exc_info=True,
                )
            try:
                self.last_signal_ts = pd.Timestamp.now(tz="UTC")
            except Exception:
                self.last_signal_ts = None

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
            if runtime_get_mode() != "real" and trading.POSITION_SERVICE is not None:
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
                if runtime_get_mode() == "real":
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
                now_ts = pd.Timestamp.now(tz="UTC")
                side = (position.get('side') or '').upper()
                qty_raw = position.get('contracts') or position.get('size') or 0.0
                qty = float(qty_raw) if qty_raw is not None else 0.0
                entry_px = float(position.get('entryPrice') or 0.0)
                mark_px = float(position.get('markPrice') or position.get('mark') or entry_px)
                sign = 1.0 if side == "LONG" else -1.0
                pnl_now = (mark_px - entry_px) * qty * sign

                if getattr(self, "_entry_ts", None) is None:
                    raw_entry = (
                        position.get("timestamp")
                        or position.get("time")
                        or position.get("entryTime")
                        or position.get("updateTime")
                    )
                    parsed_ts = None
                    if raw_entry is not None:
                        try:
                            parsed_ts = pd.Timestamp(int(raw_entry), unit="ms", tz="UTC")
                        except Exception:
                            try:
                                parsed_ts = pd.to_datetime(raw_entry, utc=True)
                            except Exception:
                                parsed_ts = None
                    self._entry_ts = parsed_ts or now_ts

                if self.max_position_hours and getattr(self, "_entry_ts", None) is not None:
                    try:
                        age_hours = (now_ts - self._entry_ts).total_seconds() / 3600.0
                    except Exception:
                        age_hours = 0.0
                    if age_hours >= float(self.max_position_hours):
                        await self.notifier.send(
                            f"⏰ Máximo {self.max_position_hours}h alcanzado. Cerrando posición del BOT."
                        )
                        await self.close_all()
                        return

                if float(self.tp_equity_pct) > 0 and float(self._eq_on_open) > 0:
                    target_pnl = self._eq_on_open * float(self.tp_equity_pct) / 100.0
                    if pnl_now >= target_pnl:
                        await self.notifier.send(
                            f"🎯 TP +{float(self.tp_equity_pct):.1f}% del equity alcanzado. Cierro posición."
                        )
                        await self.close_all()
                        return

                last_price = None
                try:
                    last_price = await self.exchange.get_current_price(
                        self.config.get('symbol', 'BTC/USDT')
                    )
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

            now = time.time()
            if now < self.manual_block_until:
                remaining = int(self.manual_block_until - now)
                self.log.info(
                    "Bloqueo manual activo %ds; no se abren nuevas posiciones.",
                    remaining,
                )
                _emit_motive(
                    {"reasons": ["bloqueo manual activo"]},
                    price_value=price_signal,
                    side_value=signal,
                )
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
            symbol_conf = str(self.config.get('symbol', 'BTC/USDT'))
            runtime_mode = _runtime_mode_value()
            if runtime_mode in {"real", "live"}:
                await self.exchange.set_leverage(
                    leverage,
                    symbol_conf,
                )
            else:
                self.logger.info(
                    "SIMULADO: leverage lógico=%s (no se setea en Binance)", leverage
                )

            entry_price = await self.exchange.get_current_price(symbol_conf)
            if entry_price is None:
                logging.warning("No se pudo obtener el precio de entrada actual.")
                _emit_motive({"reasons": ["exchange_error: precio de entrada no disponible"]}, side_value=signal)
                return
            entry_price = float(entry_price)
            try:
                risk_pct = float(self.config.get("risk_pct", 0.02))
            except Exception:
                risk_pct = 0.02
            if risk_pct <= 0:
                risk_pct = 0.02
            raw_qty = _compute_order_qty_from_equity(
                eq_on_open,
                entry_price,
                leverage,
                risk_pct,
            )

            raw_filters = await self.exchange.get_symbol_filters(symbol_conf)

            def _coerce_float(value: Any) -> float:
                try:
                    if value in (None, ""):
                        return 0.0
                    return float(value)
                except Exception:
                    return 0.0

            def _flatten_symbol_filters(raw: Any) -> Dict[str, float]:
                collected: Dict[str, Any] = {}
                if isinstance(raw, dict):
                    collected = dict(raw)
                elif isinstance(raw, (list, tuple, set)):
                    for item in raw:
                        if not isinstance(item, dict):
                            continue
                        for key in (
                            "stepSize",
                            "step_size",
                            "minQty",
                            "min_qty",
                            "minNotional",
                            "min_notional",
                        ):
                            if key in item and item[key] not in (None, ""):
                                collected.setdefault(key, item[key])
                        filter_type = str(
                            item.get("filterType") or item.get("filter_type") or ""
                        ).upper()
                        if filter_type in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
                            for key in ("stepSize", "step_size", "minQty", "min_qty"):
                                if item.get(key) not in (None, ""):
                                    collected.setdefault(key, item.get(key))
                        if filter_type == "MIN_NOTIONAL":
                            for key in ("minNotional", "min_notional"):
                                if item.get(key) not in (None, ""):
                                    collected.setdefault(key, item.get(key))

                return {
                    "stepSize": _coerce_float(
                        collected.get("stepSize") or collected.get("step_size")
                    ),
                    "minQty": _coerce_float(
                        collected.get("minQty") or collected.get("min_qty")
                    ),
                    "minNotional": _coerce_float(
                        collected.get("minNotional") or collected.get("min_notional")
                    ),
                }

            filters = _flatten_symbol_filters(raw_filters)
            qty = _quantize_amount(filters, raw_qty, entry_price)
            if qty <= 0:
                reason = (
                    f"Equity insuficiente o stepSize/minNotional demasiado altos. "
                    f"equity={eq_on_open:.2f} raw_qty={raw_qty:.8f}"
                )
                logging.warning("Qty=0 tras cuantización: %s", reason)
                try:
                    await self.notifier.send(
                        f"⚠️ No se pudo abrir posición: {reason}"
                    )
                except Exception:
                    logging.debug("No se pudo notificar fallo de qty=0", exc_info=True)
                return

            def _finite_or_none(value: Any) -> Optional[float]:
                if value in (None, ""):
                    return None
                try:
                    val = float(value)
                except Exception:
                    return None
                return val if math.isfinite(val) else None

            sl_price = _finite_or_none(
                self.strategy.calculate_sl(entry_price, last_candle, signal, eq_on_open)
            )
            try:
                eq_value = float(eq_on_open)
            except Exception:
                eq_value = 0.0
            if eq_value > 0 and qty > 0 and float(self.sl_equity_pct) > 0:
                risk_usd = eq_value * float(self.sl_equity_pct) / 100.0
                move = risk_usd / qty if risk_usd > 0 else 0.0
                if move > 0:
                    if signal == "LONG":
                        sl_price = entry_price - move
                    else:
                        sl_price = entry_price + move
            if sl_price is None:
                _emit_motive({"reasons": ["SL inválido"]}, side_value=signal)
                return
            side = signal
            tp_price = _finite_or_none(
                self.strategy.calculate_tp(entry_price, qty, eq_on_open, side, leverage)
            )

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
            runtime_mode = _runtime_mode_value()
            mode_txt = "real" if runtime_mode in {"real", "live"} else "simulado"
            await self.notifier.send(
                "🚀 operacion "
                + f"{base} Abierta: {signal} ({mode_txt})\n"
                + f"Apalancamiento: x{leverage:.1f}\n"
                + f"precio: ${entry_price:.2f}\n"
                + f"tp : {f'${tp_price:.2f}' if tp_price is not None else 'N/A'}\n"
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
                try:
                    px = await asyncio.wait_for(
                        self.exchange.get_current_price(sym), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logging.warning("Timeout obteniendo precio para %s", sym)
                    continue
                except Exception as exc:
                    logging.warning(
                        "Error inesperado obteniendo precio para %s: %s", sym, exc
                    )
                    continue

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
        entry_time_raw = None
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
            if entry_time_raw is None:
                entry_time_raw = (
                    entry.get("updateTime")
                    or entry.get("entryTime")
                    or entry.get("time")
                    or entry.get("timestamp")
                )
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
        parsed_entry_ts = None
        if entry_time_raw is not None:
            try:
                parsed_entry_ts = pd.Timestamp(int(entry_time_raw), unit="ms", tz="UTC")
            except Exception:
                try:
                    parsed_entry_ts = pd.to_datetime(entry_time_raw, utc=True)
                except Exception:
                    parsed_entry_ts = None
        if parsed_entry_ts is None:
            parsed_entry_ts = pd.Timestamp.now(tz="UTC")
        self._entry_ts = parsed_entry_ts
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
        runtime_mode = _runtime_mode_value()
        is_live_runtime = runtime_mode in {"real", "live"}

        # Precarga de posición: SOLO en SIM. En REAL validar con exchange primero.
        if not is_live_runtime:
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

    def _start_internal_scheduler(self, loop: asyncio.AbstractEventLoop) -> None:
        """Fallback scheduler when Telegram is disabled."""

        async def _run_periodic(coro, interval: float, first: float = 0.0):
            await asyncio.sleep(max(first, 0.0))
            context = None
            while True:
                try:
                    await coro(context)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensivo
                    self.logger.exception("Error en tarea periódica interna: %s", exc)
                await asyncio.sleep(max(interval, 0.0))

        tasks = [
            loop.create_task(_run_periodic(self._update_price_cache_job, interval=10, first=1)),
            loop.create_task(_run_periodic(self.trading_loop, interval=60, first=5)),
        ]
        self._internal_jobs = getattr(self, "_internal_jobs", [])
        self._internal_jobs.extend(tasks)

    def run(self):
        loop = asyncio.get_event_loop()

        if self.telegram_app is not None:
            job_queue = self.telegram_app.job_queue
            job_queue.run_repeating(self._update_price_cache_job, interval=10, first=1)
            job_queue.run_repeating(self.trading_loop, interval=60, first=5)
        else:
            logging.info(
                "Telegram deshabilitado; iniciando planificador interno basado en asyncio."
            )
            self._start_internal_scheduler(loop)

        bootstrap_tasks = [
            asyncio.to_thread(ensure_position_mode, self.exchange.hedge_mode),
            self._preload_position_from_store(),
        ]
        for task in bootstrap_tasks:
            if loop.is_running():
                loop.create_task(task)
            else:
                loop.run_until_complete(task)

        runtime_mode = (runtime_get_mode() or "paper").lower()
        mode_msg = "🔴 Modo REAL activo" if runtime_mode in {"real", "live"} else "🧪 Modo SIMULADO activo"
        if runtime_mode not in {"real", "live"}:
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

        if self.telegram_app is not None:
            logging.info("Bucle de trading programado. Iniciando polling de Telegram.")
            self.telegram_app.run_polling()
        else:
            logging.info(
                "Bucle de trading programado. Ejecutando bucle interno sin Telegram."
            )
            try:
                loop.run_forever()
            except KeyboardInterrupt:  # pragma: no cover - defensivo
                logging.info("Ejecución interrumpida manualmente.")

    async def close_all(self) -> Dict[str, Any] | bool:
        """Cierra SOLO la posición del BOT (reduceOnly) y devuelve un resumen del cierre."""
        try:
            symbol = (self.config or {}).get("symbol", "BTC/USDT")
            symbol_norm = str(symbol).replace("/", "").upper()

            def _find_open_snapshot(state_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                open_positions = state_obj.get("open_positions", {}) if isinstance(state_obj, dict) else {}
                if not isinstance(open_positions, dict):
                    return None
                candidates = [symbol, symbol_norm, str(symbol).upper(), str(symbol).lower()]
                seen = set()
                for key in candidates:
                    key_clean = str(key).replace("/", "") if isinstance(key, str) else key
                    for candidate in {key, key_clean}:
                        if candidate in seen:
                            continue
                        seen.add(candidate)
                        pos = open_positions.get(candidate)
                        if pos:
                            return dict(pos)
                return None

            def _latest_closed(state_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                closed = state_obj.get("closed_positions", []) if isinstance(state_obj, dict) else []
                if not isinstance(closed, list):
                    return None
                for entry in reversed(closed):
                    if not isinstance(entry, dict):
                        continue
                    sym_entry = str(entry.get("symbol", "")).replace("/", "").upper()
                    if sym_entry == symbol_norm:
                        return dict(entry)
                return None

            state_before = load_state()
            snapshot_before = _find_open_snapshot(state_before)

            service = getattr(trading, "POSITION_SERVICE", None)
            status_before: Optional[Dict[str, Any]] = None
            if service is not None:
                try:
                    status_before = service.get_status() or {}
                except Exception:
                    status_before = None

            summary_base: Dict[str, Any] = {"symbol": symbol, "side": "LONG"}

            def _set_if_missing(key: str, value) -> None:
                if value in (None, ""):
                    return
                try:
                    if key in {"qty", "entry_price", "leverage", "mark", "opened_at"}:
                        summary_base.setdefault(key, float(value))
                    else:
                        summary_base.setdefault(key, value)
                except Exception:
                    return

            if isinstance(snapshot_before, dict):
                summary_base["side"] = str(snapshot_before.get("side") or summary_base["side"]).upper()
                _set_if_missing("qty", snapshot_before.get("qty"))
                _set_if_missing("entry_price", snapshot_before.get("entry_price"))
                _set_if_missing("leverage", snapshot_before.get("leverage"))
                _set_if_missing("opened_at", snapshot_before.get("opened_at"))

            if isinstance(status_before, dict):
                side_now = status_before.get("side") or status_before.get("positionSide")
                if side_now:
                    summary_base["side"] = str(side_now).upper()
                for qty_key in ("qty", "pos_qty", "contracts", "positionAmt"):
                    if qty_key in status_before and status_before[qty_key] not in (None, ""):
                        _set_if_missing("qty", status_before[qty_key])
                        break
                for entry_key in ("entry_price", "avg_price", "entryPrice", "avgEntryPrice"):
                    if entry_key in status_before and status_before[entry_key] not in (None, ""):
                        _set_if_missing("entry_price", status_before[entry_key])
                        break
                for mark_key in ("mark", "mark_price", "markPrice"):
                    if mark_key in status_before and status_before[mark_key] not in (None, ""):
                        _set_if_missing("mark", status_before[mark_key])
                        break
                if "leverage" in status_before:
                    _set_if_missing("leverage", status_before.get("leverage"))

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

            bal_before = await _safe_fetch_balance()

            close_result = await asyncio.to_thread(trading.close_now, symbol)
            if not isinstance(close_result, dict):
                close_result = {"status": "ok", "order": close_result}

            if str(close_result.get("status", "")).lower() == "noop":
                reason = close_result.get("msg") or "No había posición del BOT para cerrar."
                return {"status": "noop", "reason": reason}

            bal_after = await _safe_fetch_balance()
            pnl_real = None
            if bal_before is not None and bal_after is not None:
                pnl_real = float(bal_after) - float(bal_before)

            state_after = load_state()
            closed_snapshot = _latest_closed(state_after)

            summary = dict(summary_base)

            # Preferir precio de cierre directo del resultado si está disponible
            close_price = close_result.get("close_price")

            def _assign_float(target: Dict[str, Any], key: str, value) -> None:
                if value in (None, ""):
                    return
                try:
                    target[key] = float(value)
                except Exception:
                    return

            if isinstance(closed_snapshot, dict):
                summary["side"] = str(closed_snapshot.get("side") or summary.get("side", "LONG")).upper()
                _assign_float(summary, "qty", closed_snapshot.get("qty"))
                _assign_float(summary, "entry_price", closed_snapshot.get("entry_price"))
                _assign_float(summary, "exit_price", closed_snapshot.get("exit_price"))
                # Si close_result trae precio directo, lo usamos como fuente de verdad (evita snapshot viejo)
                if close_price is not None:
                    _assign_float(summary, "exit_price", close_price)
                _assign_float(summary, "realized_pnl", closed_snapshot.get("realized_pnl"))
                if closed_snapshot.get("closed_at") is not None:
                    summary["closed_at"] = closed_snapshot.get("closed_at")
            else:
                # Ya tenemos close_price calculado arriba
                if close_price is not None:
                    _assign_float(summary, "exit_price", close_price)
                entry_px = summary.get("entry_price")
                qty_val = summary.get("qty")
                side_val = str(summary.get("side", "LONG")).upper()
                if (
                    entry_px not in (None, 0)
                    and qty_val not in (None, 0)
                    and close_price is not None
                ):
                    sign = 1.0 if side_val == "LONG" else -1.0
                    realized = (float(close_price) - float(entry_px)) * float(qty_val) * sign
                    summary["realized_pnl"] = realized

            if pnl_real is not None:
                summary["pnl_balance_delta"] = pnl_real

            storage = getattr(self, "storage", None)
            realized_for_store = summary.get("realized_pnl")
            if storage and (pnl_real is not None or realized_for_store is not None):
                trade_pnl = pnl_real if pnl_real is not None else float(realized_for_store)
                storage.persist_trade(
                    {
                        "symbol": symbol,
                        "side": str(summary.get("side", "LONG")).upper(),
                        "pnl": float(trade_pnl),
                        "note": "close_all/manual",
                    }
                )

            if close_result.get("close_price") is not None and "exit_price" not in summary:
                _assign_float(summary, "exit_price", close_result.get("close_price"))

            try:
                now_ts = pd.Timestamp.now(tz="UTC")
                trade_pnl = summary.get("realized_pnl") or summary.get("pnl_balance_delta")
                if trade_pnl is not None:
                    self._register_close_R(now_ts, float(trade_pnl))
            except Exception:
                pass

            self._entry_ts = None
            self._eq_on_open = 0.0
            self._risk_usd_trade = 0.0

            return {"status": "closed", "summary": summary, "order": close_result.get("order")}
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

    def get_period_stats(self, days: int) -> dict | None:
        try:
            import pandas as pd

            equity_csv, _ = self._csv_paths()
            df = pd.read_csv(equity_csv, parse_dates=["ts"])
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            now = pd.Timestamp.utcnow()
            since = now - pd.Timedelta(days=days)
            window = df[df["ts"] >= since]
            if window.empty:
                return None
            equity_ini = float(window["equity"].iloc[0])
            equity_fin = float(window["equity"].iloc[-1])
            if "pnl" in window.columns:
                pnl_val = float(window["pnl"].sum())
            else:
                pnl_val = equity_fin - equity_ini
            return {"equity_ini": equity_ini, "equity_fin": equity_fin, "pnl": pnl_val}
        except Exception:
            return None

    def _sum_income(self, income_type: str, start_ms: int, end_ms: int) -> float:
        if runtime_get_mode() != "real" or ACTIVE_LIVE_CLIENT is None:
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
            pct = ((equity_fin - equity_ini) / equity_ini * 100.0) if equity_ini else 0.0
            msg = (
                f"{title}\n"
                f"Equity inicial: ${equity_ini:,.2f}\n"
                f"Equity final:   ${equity_fin:,.2f}\n"
                f"PnL neto:       ${pnl:,.2f} ({pct:+.2f}%)\n"
                f"Trades: {total_trades} (W:{wins}/L:{losses})"
            )
            if runtime_get_mode() == "real":
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
