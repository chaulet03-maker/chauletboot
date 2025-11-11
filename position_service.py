from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Literal

import brokers
from paper_store import PaperStore
from bot.runtime_state import get_mode as runtime_get_mode
from bot.logger import _warn

try:
    from bot import ledger as ledger_module  # type: ignore
except Exception:  # pragma: no cover - defensivo
    ledger_module = None

try:
    from bot.identity import get_bot_id  # type: ignore
except Exception:  # pragma: no cover - defensivo
    def get_bot_id() -> str:  # type: ignore
        return "bot"

try:
    from bot.exchange_client import normalize_symbol as normalize_exchange_symbol  # type: ignore
except Exception:  # pragma: no cover - defensivo
    def normalize_exchange_symbol(symbol: str) -> str:  # type: ignore
        value = str(symbol or "").strip()
        if not value:
            return ""
        value = value.upper()
        if value.endswith(":USDT"):
            return value
        if value.endswith("/USDT"):
            return f"{value}:USDT"
        if value.endswith("USDT") and "/" not in value:
            base = value[:-4]
            return f"{base}/USDT:USDT"
        return value


Mode = Literal["simulado", "real"]


def _runtime_is_paper() -> bool:
    try:
        return (runtime_get_mode() or "paper").lower() not in {"real", "live"}
    except Exception:
        return True

logger = logging.getLogger(__name__)

EPS_QTY = 1e-9


async def async_fetch_live_equity_usdm() -> float:
    """Equity para USDⓈ-M (USDT) usando cliente CCXT asíncrono."""

    try:
        from bot import exchange_client

        client = exchange_client.get_ccxt()
        if client is None:
            return 0.0

        # Ruta estándar CCXT
        try:
            bal = await client.fetch_balance({"type": "future"})
            # ccxt puede exponerlo en bal['USDT']['total'] o similar
            for key in ("total", "free", "used"):
                try:
                    v = bal.get("USDT", {}).get(key)
                    if v is not None:
                        return float(v)
                except Exception:
                    pass
            # Fallback: algunos ccxt exponen en info
            info = bal.get("info") or {}
            if isinstance(info, dict):
                assets = info.get("assets") or info.get("balances") or []
                if isinstance(assets, list):
                    for it in assets:
                        if (it or {}).get("asset") == "USDT":
                            for k in (
                                "walletBalance",
                                "balance",
                                "cashBalance",
                                "availableBalance",
                            ):
                                v = (it or {}).get(k)
                                if v is not None:
                                    return float(v)
        except Exception as exc:
            _warn("POSITION", "fetch_balance(type=future) fallo", exc=exc, level="debug")

        # Fallback Binance nativo: /fapi/v2/balance
        try:
            raw_method = getattr(client, "fapiPrivateV2GetBalance", None)
            if raw_method is not None:
                payload = raw_method()
                if asyncio.iscoroutine(payload):
                    payload = await payload
                if isinstance(payload, list):
                    for it in payload:
                        if (it or {}).get("asset") == "USDT":
                            for k in (
                                "walletBalance",
                                "balance",
                                "cashBalance",
                                "availableBalance",
                            ):
                                v = (it or {}).get(k)
                                if v is not None:
                                    return float(v)
        except Exception as exc:
            _warn("POSITION", "fapiPrivateV2GetBalance fallo", exc=exc, level="debug")

    except Exception as exc:
        _warn("POSITION", "fetch_live_equity_usdm() error", exc=exc, level="debug")
    return 0.0


def fetch_live_equity_usdm() -> float:
    """Equity para USDⓈ-M (USDT) en contexto síncrono."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_fetch_live_equity_usdm())
    if loop.is_running():
        raise RuntimeError(
            "fetch_live_equity_usdm no puede ejecutarse en un loop activo; use async_fetch_live_equity_usdm"
        )
    return loop.run_until_complete(async_fetch_live_equity_usdm())


def _pnl(side: str, qty: float, entry: float, mark: float) -> float:
    if not qty or qty < EPS_QTY:
        return 0.0
    if not entry or entry <= 0 or not mark or mark <= 0:
        return 0.0
    sign = 1 if (side or "").upper() == "LONG" else -1
    try:
        return round((mark - entry) * qty * sign, 2)
    except Exception:
        return 0.0


def _extract_qty_and_side(data: Optional[Dict[str, Any]]) -> tuple[float, str, float]:
    if not isinstance(data, dict):
        return 0.0, "FLAT", 0.0
    raw_qty = data.get("qty") or data.get("contracts") or data.get("positionAmt")
    try:
        signed_qty = float(raw_qty)
    except Exception:
        signed_qty = 0.0
    side_raw = str(data.get("side") or "").upper()
    if not side_raw:
        if signed_qty > EPS_QTY:
            side_raw = "LONG"
        elif signed_qty < -EPS_QTY:
            side_raw = "SHORT"
    qty = abs(signed_qty)
    if qty < EPS_QTY:
        qty = 0.0
        signed_qty = 0.0
        if not side_raw:
            side_raw = "FLAT"
    return qty, side_raw or "FLAT", signed_qty


def _extract_price(data: Optional[Dict[str, Any]], *keys: str) -> float:
    if not isinstance(data, dict):
        return 0.0
    for key in keys:
        value = data.get(key)
        if value in (None, ""):
            continue
        try:
            price = float(value)
        except Exception:
            continue
        if price > 0:
            return price
    return 0.0


def split_total_vs_bot(
    acct_pos: Optional[Dict[str, Any]],
    bot_pos: Optional[Dict[str, Any]],
    mark: float,
) -> Dict[str, Dict[str, float | str]]:
    """Separa la posición total en porciones BOT vs manual y calcula PnL estimado.

    La implementación anterior asumía que las posiciones manuales siempre iban en la
    misma dirección que la posición total reportada por la cuenta. Ese supuesto
    dejaba de ser cierto cuando coexistían posiciones del bot y manuales con lados
    opuestos (por ejemplo, el bot LONG y una cobertura manual SHORT). Como la API de
    Binance reporta cantidades con signo, usamos ese dato para obtener la porción
    manual como una diferencia firmada en vez de limitarse a restar magnitudes.
    Esto permite distinguir correctamente las posiciones manuales aunque el neto de
    la cuenta esté del lado opuesto o incluso se acerque a cero.
    """

    total_qty, total_side, total_signed = _extract_qty_and_side(acct_pos)
    bot_qty, bot_side, bot_signed = _extract_qty_and_side(bot_pos)

    if bot_side == "FLAT" and abs(bot_qty) <= EPS_QTY:
        bot_qty = 0.0
        bot_signed = 0.0
        bot_side = "FLAT"

    if total_side == "FLAT" and abs(total_qty) <= EPS_QTY:
        total_qty = 0.0
        total_signed = 0.0
        total_side = "FLAT"

    total_entry = _extract_price(acct_pos, "entry_price", "entryPrice", "avgPrice")
    bot_entry = _extract_price(bot_pos, "entry_price", "entryPrice", "avgPrice")

    manual_signed = total_signed - bot_signed
    manual_qty = abs(manual_signed)
    if manual_qty <= EPS_QTY:
        manual_qty = 0.0
        manual_signed = 0.0

    manual_side = "FLAT"
    if manual_signed > EPS_QTY:
        manual_side = "LONG"
    elif manual_signed < -EPS_QTY:
        manual_side = "SHORT"

    manual_entry = 0.0
    if manual_side != "FLAT":
        try:
            total_notional = float(total_entry or 0.0) * total_signed
            bot_notional = float(bot_entry or 0.0) * bot_signed
            manual_notional = total_notional - bot_notional
            manual_entry = manual_notional / manual_signed if manual_signed else 0.0
        except Exception:
            manual_entry = 0.0
        if manual_entry <= 0:
            manual_entry = 0.0

    mark_val = float(mark or 0.0)

    total_side_output = "FLAT"
    if total_signed > EPS_QTY:
        total_side_output = "LONG"
    elif total_signed < -EPS_QTY:
        total_side_output = "SHORT"

    bot_side_output = "FLAT"
    if bot_signed > EPS_QTY:
        bot_side_output = "LONG"
    elif bot_signed < -EPS_QTY:
        bot_side_output = "SHORT"

    return {
        "total": {
            "side": total_side_output,
            "qty": round(abs(total_signed), 6),
            "entry_price": round(total_entry, 6) if total_entry else 0.0,
            "pnl": _pnl(total_side_output, abs(total_signed), total_entry, mark_val),
        },
        "bot": {
            "side": bot_side_output,
            "qty": round(abs(bot_signed), 6),
            "entry_price": round(bot_entry, 6) if bot_entry else 0.0,
            "pnl": _pnl(bot_side_output, abs(bot_signed), bot_entry, mark_val),
        },
        "manual": {
            "side": manual_side,
            "qty": round(manual_qty, 6),
            "entry_price": round(manual_entry, 6) if manual_entry else 0.0,
            "pnl": _pnl(manual_side, manual_qty, manual_entry, mark_val),
        },
    }


async def reconcile_bot_store_with_account(trader, exchange, symbol: str, mark: float) -> None:
    """Reconcilia el store del BOT con la posición total de la cuenta."""

    if trader is None or exchange is None or not hasattr(exchange, "fetch_positions"):
        return

    try:
        acct_positions = await exchange.fetch_positions(symbol)
    except Exception:
        acct_positions = None

    acct = None
    if isinstance(acct_positions, list):
        acct = acct_positions[0] if acct_positions else None
    elif isinstance(acct_positions, dict):
        acct = acct_positions

    try:
        bot = await trader.check_open_position(exchange=None)
    except Exception:
        bot = None

    total_qty, _, _ = _extract_qty_and_side(acct)
    bot_qty, _, _ = _extract_qty_and_side(bot)

    if total_qty < EPS_QTY and bot_qty > EPS_QTY:
        await trader.set_position(None)
        logger.info(
            "Reconciliación: cuenta FLAT pero store del BOT tenía posición -> limpiado."
        )
        try:
            import trading

            service = getattr(trading, "POSITION_SERVICE", None)
            if service is not None:
                close_ledger = getattr(service, "_force_close_ledger", None)
                block_rehydrate = getattr(service, "_block_rehydrate", None)
                store = getattr(service, "store", None)
                if callable(close_ledger):
                    close_ledger(symbol, reason="exchange_flat")
                if store is not None:
                    try:
                        store.save(pos_qty=0.0, avg_price=0.0)
                    except Exception as exc:
                        _warn(
                            "RECON",
                            "No se pudo limpiar store desde reconciliación.",
                            exc=exc,
                            level="debug",
                        )
                if callable(block_rehydrate):
                    block_rehydrate(symbol, seconds=120.0)
        except Exception as _e:  # pragma: no cover - defensivo
            _warn("RECON", "No pude bloquear rehidratación", exc=_e)
        return

    if total_qty + EPS_QTY < bot_qty:
        await trader.set_position(None)
        logger.info(
            "Reconciliación: total < bot_qty (cerraste manual). BOT marcado FLAT."
        )


class PositionService:
    """Centraliza la información de posición tanto en paper como en live."""

    def __init__(
        self,
        *,
        paper_store: Optional[PaperStore] = None,
        live_client: Optional[Any] = None,
        ccxt_client: Optional[Any] = None,
        symbol: str = "BTC/USDT",
        mode: Mode | str | None = None,
    ) -> None:
        self.store = paper_store
        self.client = live_client
        self.public_client = ccxt_client or live_client
        self.symbol = symbol
        self.mode: Mode = self._resolve_mode(mode)
        self._ledger = ledger_module
        self._bot_id: Optional[str] = None
        if self._ledger is not None:
            try:
                self._bot_id = get_bot_id()
            except Exception:  # pragma: no cover - defensivo
                self._bot_id = None
        self._cache: dict[str, dict[str, Any]] = {}
        self._rehydrate_block_until: dict[str, float] = {}

    @staticmethod
    def _resolve_mode(mode: Mode | str | None) -> Mode:
        if mode is None:
            return "simulado" if _runtime_is_paper() else "real"
        value = str(mode).strip().lower()
        if value in {"real", "live"}:
            return "real"
        return "simulado"

    def _cache_key(self, symbol: Optional[str] = None) -> str:
        base = symbol or self.symbol or ""
        normalized = normalize_exchange_symbol(base)
        return normalized or base

    def _ledger_symbol_key(self, symbol: Optional[str] = None) -> str:
        base = symbol or self.symbol or ""
        cleaned = base.replace("/", "")
        return cleaned.upper()

    def _ledger_mode(self) -> str:
        return "paper" if self.mode == "simulado" else "live"

    def _invalidate_cache(self, symbol: Optional[str] = None) -> None:
        key = self._cache_key(symbol)
        if key:
            self._cache.pop(key, None)

    def _block_rehydrate(self, symbol: Optional[str] = None, seconds: float = 120.0) -> None:
        key = self._cache_key(symbol)
        if not key:
            return
        self._rehydrate_block_until[key] = time.time() + float(seconds)

    def _rehydrate_allowed(self, symbol: Optional[str] = None) -> bool:
        key = self._cache_key(symbol)
        if not key:
            return True
        until = self._rehydrate_block_until.get(key)
        if until is None:
            return True
        if until <= time.time():
            self._rehydrate_block_until.pop(key, None)
            return True
        return False

    def _force_close_ledger(self, symbol: Optional[str] = None, *, reason: str = "exchange_flat") -> None:
        if self._ledger is None:
            return
        if self._bot_id is None:
            return
        action = getattr(self._ledger, "force_close", None)
        if action is None:
            return
        try:
            action(self._ledger_mode(), self._bot_id, self._ledger_symbol_key(symbol), reason=reason)
        except Exception as exc:  # pragma: no cover - defensivo
            _warn("RECON", "No se pudo forzar cierre en ledger.", exc=exc, level="debug")

    def _fetch_from_exchange(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        position = self._fetch_live_position()
        if position is None:
            return None
        result = dict(position)
        if symbol or self.symbol:
            result.setdefault("symbol", symbol or self.symbol)
        return result

    def _rehydrate_from_ledger(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.mode == "real":
            return None
        if self._ledger is None or self._bot_id is None:
            return None
        if not self._rehydrate_allowed(symbol):
            return None
        action = getattr(self._ledger, "bot_position", None)
        if action is None:
            return None
        try:
            qty, avg = action(self._ledger_mode(), self._bot_id, self._ledger_symbol_key(symbol))
        except Exception as exc:
            _warn("RECON", "No se pudo leer posición desde ledger.", exc=exc, level="debug")
            return None
        if abs(qty) <= EPS_QTY:
            return None
        side = "LONG" if qty > 0 else "SHORT"
        return {
            "symbol": symbol or self.symbol,
            "side": side,
            "qty": abs(float(qty)),
            "entry_price": float(avg or 0.0),
        }

    def current_position(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Devuelve la posición actual considerando el modo y bloqueos de rehidratación."""

        sym = symbol or self.symbol
        key = self._cache_key(sym)

        if not sym:
            return None

        pos: Optional[Dict[str, Any]] = None

        if str(self.mode).lower() == "real":
            pos = self._fetch_from_exchange(sym)
        else:
            block_until = self._rehydrate_block_until.get(key, 0.0)
            now = time.time()
            if block_until > now:
                self._cache.pop(key, None)
                return None

            pos = self._rehydrate_from_ledger(sym)

            if pos is None and self.store is not None:
                try:
                    state = self.store.load() or {}
                except Exception as exc:
                    _warn(
                        "POSITION",
                        "No se pudo leer store paper para rehidratación.",
                        exc=exc,
                        level="debug",
                    )
                    state = {}

                try:
                    qty_store = float(state.get("pos_qty") or 0.0)
                except Exception:
                    qty_store = 0.0

                if abs(qty_store) > EPS_QTY:
                    side_store = "LONG" if qty_store > 0 else "SHORT"
                    pos = {
                        "symbol": sym,
                        "side": side_store,
                        "qty": abs(qty_store),
                        "entry_price": float(state.get("avg_price") or 0.0),
                        "mark_price": float(state.get("mark") or 0.0),
                    }

            if pos is None:
                cached = self._cache.get(key)
                if isinstance(cached, dict):
                    pos = dict(cached)

        if not pos:
            self._cache.pop(key, None)
            return None

        qty_raw = pos.get("qty") or pos.get("contracts") or pos.get("size")
        try:
            qty_val = abs(float(qty_raw))
        except Exception:
            qty_val = 0.0

        side = str(pos.get("side") or "").upper()
        if not side:
            side = "LONG" if qty_val > 0 else "FLAT"

        if qty_val <= EPS_QTY or side == "FLAT":
            self._cache.pop(key, None)
            return None

        entry_price = _extract_price(pos, "entry_price", "entryPrice", "avg_price", "avgPrice")
        mark_price = _extract_price(pos, "mark_price", "markPrice", "mark")

        normalized = {
            "symbol": pos.get("symbol") or sym,
            "side": side,
            "qty": qty_val,
            "entry_price": entry_price,
            "mark_price": mark_price,
        }

        self._cache[key] = dict(normalized)
        return normalized

    def refresh(self) -> None:
        """Recarga el estado desde el store compartido."""
        if not self.store:
            return
        try:
            _ = self.store.load()
            self._invalidate_cache()
        except Exception:
            raise

    # ------------------------------------------------------------------
    def mark_to_market(self, mark: float) -> None:
        if not (self.mode == "simulado" and self.store):
            return
        try:
            self.store.save(mark=float(mark))
            self._invalidate_cache()
        except Exception as exc:
            _warn(
                "POSITION",
                "No se pudo actualizar mark del store paper.",
                exc=exc,
                level="debug",
            )

    # ------------------------------------------------------------------
    def _fetch_public_mark(self) -> Optional[float]:
        client = self.public_client
        if client is None:
            return None

        symbol_no_slash = self.symbol.replace("/", "")

        # Preferimos el endpoint de premium index si está disponible
        try:
            if hasattr(client, "fapiPublicGetPremiumIndex"):
                data = client.fapiPublicGetPremiumIndex({"symbol": symbol_no_slash})
            elif hasattr(client, "fapiPublic_get_premiumindex"):
                data = client.fapiPublic_get_premiumindex({"symbol": symbol_no_slash})
            else:
                data = None
            if data:
                if isinstance(data, list):
                    for item in data:
                        if str(item.get("symbol", "")).upper() == symbol_no_slash.upper():
                            mark_price = item.get("markPrice") or item.get("markprice")
                            if mark_price is not None:
                                return float(mark_price)
                elif isinstance(data, dict):
                    mark_price = data.get("markPrice") or data.get("markprice")
                    if mark_price is not None:
                        return float(mark_price)
        except Exception as exc:
            _warn(
                "POSITION",
                "No se pudo obtener markPrice desde premium index público.",
                exc=exc,
                level="debug",
            )

        # Fallback: ticker público
        if hasattr(client, "fetch_ticker"):
            candidates = [self.symbol]
            base_symbol = self.symbol
            if base_symbol.endswith("/USDT") and ":USDT" not in base_symbol:
                candidates.append(f"{base_symbol}:USDT")
            candidates.append(symbol_no_slash)
            seen = set()
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    ticker = client.fetch_ticker(candidate)
                    if ticker and ticker.get("last") is not None:
                        return float(ticker.get("mark", ticker.get("info", {}).get("markPrice", ticker["last"])))
                except Exception:
                    continue

        return None

    def _maybe_refresh_paper_mark(self, state: dict[str, Any]) -> dict[str, Any]:
        if not (self.mode == "simulado" and self.store):
            return state
        updated = int(state.get("updated") or 0)
        mark = float(state.get("mark") or 0.0)
        now = int(time.time())
        if mark > 0 and (now - updated) <= 30:
            return state
        price = self._fetch_public_mark()
        if price is None:
            return state
        try:
            self.store.save(mark=float(price))
            return self.store.load()
        except Exception as exc:
            _warn(
                "POSITION",
                "No se pudo refrescar mark del store paper.",
                exc=exc,
                level="debug",
            )
            return state

    # ------------------------------------------------------------------
    def _status_paper(self) -> dict[str, Any]:
        if not self.store:
            raise RuntimeError("PaperStore no inicializado en modo paper.")

        state = self.store.load()
        state = self._maybe_refresh_paper_mark(state)

        pos_qty = float(state.get("pos_qty", 0.0) or 0.0)
        avg_price = float(state.get("avg_price", 0.0) or 0.0)
        mark = float(state.get("mark", 0.0) or 0.0)
        if abs(pos_qty) <= EPS_QTY:
            ledger_state = self._rehydrate_from_ledger(self.symbol)
            if ledger_state is not None:
                side_ledger = str(ledger_state.get("side") or "FLAT").upper()
                qty_ledger = float(ledger_state.get("qty") or 0.0)
                if side_ledger == "SHORT":
                    pos_qty = -abs(qty_ledger)
                else:
                    pos_qty = abs(qty_ledger)
                avg_price = float(ledger_state.get("entry_price") or avg_price)
        side = "FLAT"
        if pos_qty > 0:
            side = "LONG"
        elif pos_qty < 0:
            side = "SHORT"

        unreal = 0.0
        if pos_qty != 0.0 and avg_price > 0.0 and mark > 0.0:
            if pos_qty > 0:
                unreal = (mark - avg_price) * abs(pos_qty)
            else:
                unreal = (avg_price - mark) * abs(pos_qty)

        base_equity = float(state.get("equity") or 0.0)
        realized = float(state.get("realized_pnl") or 0.0)
        fees = float(state.get("fees") or 0.0)
        equity = base_equity + realized + unreal - fees

        return {
            "symbol": self.symbol,
            "side": side,
            "entry_price": round(avg_price, 2),
            "qty": float(abs(pos_qty)),
            "pnl": round(unreal, 2),
            "equity": round(equity, 2),
            "mark": round(mark, 2),
        }

    def _fetch_live_position(self) -> Optional[Dict[str, Any]]:
        client = self.client
        if client is None:
            return None

        symbol_no_slash = self.symbol.replace("/", "")
        payload: Any = None

        if hasattr(client, "futures_position_information"):
            try:
                payload = client.futures_position_information(symbol=symbol_no_slash)
            except Exception:
                payload = None

        if not payload:
            if hasattr(client, "fapiPrivate_get_positionrisk"):
                try:
                    payload = client.fapiPrivate_get_positionrisk({"symbol": symbol_no_slash})
                except Exception:
                    payload = None
            elif hasattr(client, "fapiPrivateGetPositionRisk"):
                try:
                    payload = client.fapiPrivateGetPositionRisk({"symbol": symbol_no_slash})
                except Exception:
                    payload = None

        entries: list[Dict[str, Any]] = []
        if isinstance(payload, list):
            entries = [entry for entry in payload if isinstance(entry, dict)]
        elif isinstance(payload, dict):
            entries = [payload]

        for entry in entries:
            entry_symbol = str(entry.get("symbol") or entry.get("symbolName") or "").upper()
            if entry_symbol != symbol_no_slash.upper():
                continue
            try:
                raw_amount = (
                    entry.get("positionAmt")
                    or entry.get("position_amt")
                    or entry.get("amount")
                    or entry.get("qty")
                    or 0.0
                )
                amount = float(raw_amount)
            except Exception:
                amount = 0.0

            if abs(amount) <= EPS_QTY:
                continue

            side = "LONG" if amount > 0 else "SHORT"

            try:
                entry_price = float(entry.get("entryPrice") or entry.get("avgPrice") or 0.0)
            except Exception:
                entry_price = 0.0

            try:
                mark_price = float(
                    entry.get("markPrice")
                    or entry.get("mark_price")
                    or entry.get("entryPrice")
                    or entry.get("avgPrice")
                    or 0.0
                )
            except Exception:
                mark_price = 0.0

            return {
                "side": side,
                "qty": abs(amount),
                "entry_price": entry_price,
                "mark_price": mark_price,
            }

        return None

    def _status_live(self) -> dict[str, Any]:
        client = self.client
        symbol_no_slash = self.symbol.replace("/", "")

        mark = 0.0
        if client and hasattr(client, "futures_mark_price"):
            try:
                mp = client.futures_mark_price(symbol=symbol_no_slash)
                if isinstance(mp, dict):
                    mark = float(mp.get("markPrice") or 0.0)
            except Exception as exc:
                _warn("POSITION", "No se pudo obtener markPrice privado.", exc=exc, level="debug")

        store_has_pos = False
        if self.store is not None:
            try:
                state_live = self.store.load() or {}
                pos_store = float(state_live.get("pos_qty") or 0.0)
                store_has_pos = abs(pos_store) > EPS_QTY
            except Exception as exc:
                _warn("POSITION", "No se pudo leer store live", exc=exc, level="debug")

        live_position = self._fetch_from_exchange(self.symbol)
        exchange_flat = True
        if live_position is not None:
            try:
                qty_live = float(live_position.get("qty") or 0.0)
            except Exception:
                qty_live = 0.0
            exchange_flat = abs(qty_live) <= EPS_QTY
        
        if self.mode == "real" and exchange_flat and store_has_pos:
            self._invalidate_cache(self.symbol)
            if self.store is not None:
                try:
                    self.store.save(pos_qty=0.0, avg_price=0.0)
                except Exception as exc:
                    _warn("POSITION", "No se pudo limpiar el store en live.", exc=exc, level="debug")
            self._force_close_ledger(self.symbol, reason="exchange_flat")
            self._block_rehydrate(self.symbol)
            logger.info(
                "Reconciliación: exchange FLAT -> store limpiado, ledger cerrado y bloqueo de rehidratación."
            )

        pos_qty = 0.0
        avg_price = 0.0
        side = "FLAT"

        if live_position and not exchange_flat:
            side = str(live_position.get("side") or "FLAT").upper()
            qty_live = float(live_position.get("qty") or 0.0)
            avg_price = float(live_position.get("entry_price") or 0.0)
            mark_live = float(live_position.get("mark_price") or 0.0)
            if mark <= 0 and mark_live > 0:
                mark = mark_live
            pos_qty = -abs(qty_live) if side == "SHORT" else abs(qty_live)
        else:
            side = "FLAT"
            pos_qty = 0.0

        if mark <= 0:
            public_mark = self._fetch_public_mark()
            if public_mark is not None:
                mark = float(public_mark)

        equity = 0.0
        try:
            equity = fetch_live_equity_usdm()
        except Exception as exc:
            _warn("POSITION", "No se pudo obtener equity live.", exc=exc, level="debug")

        pnl = 0.0
        qty = abs(pos_qty)
        if qty > 0 and avg_price > 0 and mark > 0:
            if side == "SHORT":
                pnl = (avg_price - mark) * qty
            else:
                pnl = (mark - avg_price) * qty

        status = {
            "symbol": self.symbol,
            "side": side,
            "entry_price": round(avg_price, 2),
            "qty": float(qty),
            "pnl": round(pnl, 2),
            "equity": round(float(equity or 0.0), 2),
            "mark": round(float(mark or 0.0), 2),
        }
        self._cache[self._cache_key(self.symbol)] = dict(status)
        return status

    def apply_fill(
        self, side: str, qty: float, price: float, fee: float = 0.0
    ) -> dict[str, Any] | None:
        """Actualiza el store local con un fill ejecutado."""
        if not self.store:
            return None
        try:
            state = self.store.load()
            pos_qty = float(state.get("pos_qty") or 0.0)
            avg_price = float(state.get("avg_price") or 0.0)
            direction = str(side).upper()
            dq = float(qty) if direction in {"LONG", "BUY"} else -float(qty)
            new_qty = pos_qty + dq
            price_f = float(price or 0.0)
            realized = 0.0

            if pos_qty * dq < 0:
                closing_qty = min(abs(dq), abs(pos_qty))
                if closing_qty > 0:
                    if pos_qty > 0:
                        realized = (price_f - avg_price) * closing_qty
                    else:
                        realized = (avg_price - price_f) * closing_qty
                if abs(new_qty) < 1e-12:
                    avg_new = 0.0
                elif abs(abs(dq) - closing_qty) > 1e-12:
                    avg_new = price_f
                else:
                    avg_new = avg_price
            else:
                total = abs(pos_qty) + abs(dq)
                if total <= 1e-12:
                    avg_new = 0.0
                elif abs(pos_qty) < 1e-12:
                    avg_new = price_f
                else:
                    avg_new = (
                        (abs(pos_qty) * avg_price) + (abs(dq) * price_f)
                    ) / total

            changes: dict[str, Any] = {
                "pos_qty": float(new_qty),
                "avg_price": float(avg_new),
            }
            if fee:
                changes["fees"] = float(state.get("fees") or 0.0) + float(fee)
            if realized:
                changes["realized_pnl"] = float(state.get("realized_pnl") or 0.0) + float(realized)
            logger.info(
                "STORE SAVE: pos_qty=%.6f avg=%.2f",
                changes["pos_qty"],
                changes["avg_price"],
            )
            result = self.store.save(**changes)
            self._invalidate_cache()
            return result
        except Exception as exc:
            _warn("POSITION", "apply_fill fallo", exc=exc, level="debug")
            return None

    # ------------------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        use_paper = self.mode == "simulado"
        if not use_paper and _runtime_is_paper():
            # Defensa: si el runtime sigue en paper pero el servicio está en real,
            # priorizamos el modo configurado explícitamente.
            use_paper = False
        status = self._status_paper() if use_paper else self._status_live()
        self._cache[self._cache_key(self.symbol)] = dict(status)
        return status


def build_position_service(
    settings,
    *,
    store: Optional[PaperStore] = None,
    client: Optional[Any] = None,
    ccxt_client: Optional[Any] = None,
    symbol: str = "BTC/USDT",
) -> PositionService:
    if settings.PAPER:
        if store is None:
            store = brokers.ACTIVE_PAPER_STORE or brokers._build_paper_store(settings.start_equity)
            brokers.ACTIVE_PAPER_STORE = store
        svc = PositionService(
            paper_store=store,
            ccxt_client=ccxt_client,
            symbol=symbol,
            mode="simulado",
        )
    else:
        if client is None:
            client = brokers.ACTIVE_LIVE_CLIENT
        svc = PositionService(
            paper_store=None,
            live_client=client,
            ccxt_client=ccxt_client,
            symbol=symbol,
            mode="real",
        )
    global pos_svc
    pos_svc = svc
    return svc


pos_svc: Optional[PositionService] = None


__all__ = [
    "PositionService",
    "build_position_service",
    "pos_svc",
    "fetch_live_equity_usdm",
    "async_fetch_live_equity_usdm",
]
