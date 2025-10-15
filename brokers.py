from __future__ import annotations

import logging
import os
import time
from threading import Lock
from pathlib import Path
from typing import Any, Callable, Optional

from paper_store import PaperStore

try:
    from binance.error import BinanceAPIException, BinanceOrderException, BinanceRequestException  # type: ignore
except Exception:  # pragma: no cover - python-binance opcional en tests
    BinanceAPIException = BinanceOrderException = BinanceRequestException = tuple()  # type: ignore

from bot.exchanges.binance_filters import (
    SymbolFilters,
    build_filters,
    quantize_price,
    quantize_qty,
    validate_order,
)

logger = logging.getLogger(__name__)

_RAW_DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data")).expanduser()
if not _RAW_DATA_DIR.is_absolute():
    _RAW_DATA_DIR = (Path("/app") / _RAW_DATA_DIR).resolve()
_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

_raw_store_path = os.getenv("PAPER_STORE_PATH")
if _raw_store_path:
    candidate = Path(_raw_store_path)
    if not candidate.is_absolute():
        candidate = (_RAW_DATA_DIR / candidate).resolve()
else:
    candidate = (_RAW_DATA_DIR / "paper_state.json").resolve()

PAPER_STORE_PATH = candidate

ACTIVE_PAPER_STORE: PaperStore | None = None
ACTIVE_LIVE_CLIENT: Any | None = None


class SimBroker:
    """Simula fills y persiste el estado en disco."""

    def __init__(self, store: PaperStore, fee_rate: float = 0.0):
        self.store = store
        self.fee_rate = max(0.0, float(fee_rate))

    def _apply_fill(self, side: str, qty: float, price: float) -> dict[str, Any]:
        state = self.store.load()
        pos_qty = float(state.get("pos_qty", 0.0) or 0.0)
        avg_price = float(state.get("avg_price", 0.0) or 0.0)
        dq = float(qty) if side.upper() == "BUY" else -float(qty)
        fee = abs(float(price) * float(qty)) * self.fee_rate

        updates: dict[str, Any] = {}
        if pos_qty == 0.0 or pos_qty * dq >= 0.0:
            # Abrir o aumentar posición en la misma dirección.
            new_qty = pos_qty + dq
            if abs(new_qty) < 1e-12:
                new_avg = 0.0
            elif pos_qty == 0.0:
                new_avg = float(price)
            else:
                total_notional = abs(pos_qty) * avg_price + abs(dq) * float(price)
                new_avg = total_notional / max(abs(new_qty), 1e-12)
            updates.update({
                "pos_qty": new_qty,
                "avg_price": new_avg,
            })
        else:
            # Cerrando posición parcial o completamente.
            closing_qty = min(abs(pos_qty), abs(dq))
            realized = 0.0
            if closing_qty > 0:
                if pos_qty > 0:
                    realized = (float(price) - avg_price) * closing_qty
                else:
                    realized = (avg_price - float(price)) * closing_qty
            remaining_qty = pos_qty + dq
            updates["pos_qty"] = remaining_qty
            if abs(remaining_qty) < 1e-12:
                updates["avg_price"] = 0.0
            elif abs(abs(dq) - closing_qty) > 1e-12:
                # Revirtió y abrió nueva en dirección contraria.
                updates["avg_price"] = float(price)
            else:
                updates["avg_price"] = avg_price
            updates["realized_pnl"] = state.get("realized_pnl", 0.0) + realized

        if fee > 0:
            updates["fees"] = state.get("fees", 0.0) + fee

        return self.store.save(**updates)

    def place_order(self, side: str, qty: float, price: float | None, **kwargs: Any) -> dict[str, Any]:
        fill_price: float | None = None if price is None else float(price)
        if fill_price is None:
            state = self.store.load()
            fallback = state.get("mark") or state.get("avg_price")
            if fallback is None:
                raise ValueError("SimBroker requiere un precio para simular fills")
            fill_price = float(fallback)

        oid = f"sim-{int(time.time() * 1000)}"
        new_state = self._apply_fill(side, qty, float(fill_price))
        logger.info("PAPER ORDER %s %.6f @ %.2f → FILLED [%s]", side, qty, float(fill_price), oid)
        payload: dict[str, Any] = {
            "orderId": oid,
            "status": "FILLED",
            "price": float(fill_price),
            "executedQty": float(qty),
            "side": str(side).upper(),
            "sim": True,
            "state": new_state,
        }
        symbol = kwargs.get("symbol")
        if symbol:
            payload["symbol"] = symbol
        if "sl" in kwargs:
            payload["sl"] = kwargs["sl"]
        if "tp" in kwargs:
            payload["tp"] = kwargs["tp"]
        return payload


class BinanceBroker:
    """Envuelve al cliente real SOLO en live."""

    def __init__(self, client: Any):
        self.client = client
        self._symbol_filters: dict[str, SymbolFilters] = {}
        self._filters_lock = Lock()
        self._reject_count = 0
        self._reject_window_ts = 0.0

    # ------------------------------------------------------------------
    # Exchange info helpers
    # ------------------------------------------------------------------
    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    def _load_symbol_filters(self, symbol: str) -> SymbolFilters:
        norm = self._normalize_symbol(symbol)
        with self._filters_lock:
            if norm in self._symbol_filters:
                return self._symbol_filters[norm]

            info = {}
            try:
                # ``futures_exchange_info`` returns the full payload once.
                exchange_info = self.client.futures_exchange_info()
                symbols = exchange_info.get("symbols", []) if isinstance(exchange_info, dict) else []
                for entry in symbols:
                    if str(entry.get("symbol", "")).upper() == norm:
                        info = entry
                        break
            except Exception as exc:  # pragma: no cover - dependiente de red
                logger.warning("No se pudo obtener exchangeInfo: %s", exc)

            if not info:
                info = {"symbol": norm, "filters": []}

            filters = build_filters(norm, {"info": info, "symbol": norm})
            self._symbol_filters[norm] = filters
            return filters

    # ------------------------------------------------------------------
    # Error handling / retry
    # ------------------------------------------------------------------
    def _extract_error_code(self, exc: Exception) -> Optional[int]:
        for attr in ("code", "status_code", "error_code"):
            value = getattr(exc, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        for arg in getattr(exc, "args", []):
            if isinstance(arg, (int, float)):
                return int(arg)
            if isinstance(arg, str) and "code" in arg:
                for token in arg.replace("{", " ").replace("}", " ").replace(",", " ").split():
                    if token.startswith("-") or token.isdigit():
                        try:
                            return int(token)
                        except Exception:
                            continue
        return None

    def _classify_error(self, exc: Exception) -> str:
        code = self._extract_error_code(exc)
        if isinstance(exc, BinanceRequestException):
            return "network"
        if isinstance(exc, BinanceAPIException):
            if code in (-1003, -1015):
                return "ratelimit"
            if code in (-2019, -2021):
                return "insufficient_margin"
            if code in (-2010, -1013, -1102):
                return "invalid_order"
        if isinstance(exc, BinanceOrderException):
            return "invalid_order"
        if code in (-1003, -1015):
            return "ratelimit"
        if code in (-2019, -2021):
            return "insufficient_margin"
        if code in (-2010, -1013, -1102):
            return "invalid_order"
        message = str(exc).lower()
        if "insufficient" in message and "margin" in message:
            return "insufficient_margin"
        if "precision" in message or "lot size" in message:
            return "invalid_order"
        if "timed out" in message or "timeout" in message or "temporarily unavailable" in message:
            return "network"
        return "unknown"

    def _should_retry(self, classification: str) -> bool:
        return classification in {"ratelimit", "network"}

    def _register_reject(self) -> None:
        now = time.time()
        if now - self._reject_window_ts > 60:
            self._reject_window_ts = now
            self._reject_count = 0
        self._reject_count += 1
        if self._reject_count >= 3:
            logger.warning(
                "⚠️ Múltiples rejects en <60s (=%s). Revisar cuantización/margen.",
                self._reject_count,
            )
            # Si existe un notifier global, se puede enviar alerta aquí.
            # try:
            #     notifier.send(f"⚠️ Múltiples rejects en <60s (={self._reject_count})")
            # except Exception:
            #     pass

    def _call_with_backoff(self, fn, *args, **kwargs):
        retries = int(kwargs.pop("_retries", 4))
        base_delay = float(kwargs.pop("_base_delay", 0.25))
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                classification = self._classify_error(exc)
                should_retry = self._should_retry(classification)
                if not should_retry or attempt >= retries:
                    if classification in {"insufficient_margin", "invalid_order"}:
                        logger.error("Orden rechazada (%s): %s", classification, exc)
                    if classification not in {"network", "ratelimit"}:
                        self._register_reject()
                    raise
                delay = base_delay * (1.8 ** attempt)
                logger.warning(
                    "Retry %s/%s tras error %s (%.3fs)", attempt + 1, retries, classification, delay
                )
                attempt += 1
                time.sleep(delay)

    def place_order(self, side: str, qty: float, price: float | None, **kwargs: Any) -> Any:
        symbol = kwargs.get("symbol")
        if not symbol:
            raise ValueError("BinanceBroker requiere 'symbol' para enviar la orden.")
        params = {k: v for k, v in kwargs.items() if k not in {"symbol", "sl", "tp", "client_order_id"}}
        explicit_type = params.pop("order_type", None)
        reduce_only = params.pop("reduce_only", False)
        if explicit_type is None:
            order_type = "MARKET" if price in (None, 0, "0") else "LIMIT"
        else:
            order_type = str(explicit_type).upper()

        filters = self._load_symbol_filters(symbol)

        qty_f = quantize_qty(filters, qty)
        px_f: Optional[float] = None

        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Orden LIMIT requiere precio")
            px_f = quantize_price(filters, price)
            params["price"] = px_f
        else:
            params.pop("price", None)

        if reduce_only:
            params["reduceOnly"] = True

        validate_order(filters, qty_f, px_f)

        client_order_id = params.pop("newClientOrderId", None)
        if client_order_id is None:
            provided = kwargs.get("client_order_id")
            if provided:
                client_order_id = str(provided)
            else:
                client_order_id = f"bot-{int(time.time() * 1000)}-{os.getpid()}"
        params["newClientOrderId"] = client_order_id

        order_payload = dict(
            symbol=self._normalize_symbol(symbol),
            side=side.upper(),
            type=order_type,
            quantity=qty_f,
            **params,
        )

        response = self._call_with_backoff(self.client.futures_create_order, **order_payload)

        sl_price = kwargs.get("sl")
        tp_price = kwargs.get("tp")
        position_side = params.get("positionSide")
        protections = {}
        if sl_price:
            protections["SL"] = sl_price
        if tp_price:
            protections["TP"] = tp_price
        if protections:
            self._place_protections(filters, side, qty_f, protections, position_side)

        return response

    # ------------------------------------------------------------------
    # Reduce-only protections
    # ------------------------------------------------------------------
    def _place_protections(
        self,
        filters: SymbolFilters,
        side: str,
        qty: float,
        protections: dict[str, float],
        position_side: Optional[str] = None,
    ) -> None:
        closing_side = "SELL" if side.upper() == "BUY" else "BUY"
        for label, raw_price in protections.items():
            if raw_price is None:
                continue
            px = quantize_price(filters, raw_price)
            try:
                validate_order(filters, qty, px)
            except ValueError as exc:
                logger.warning("Protección %s inválida: %s", label, exc)
                continue
            order_type = "STOP_MARKET" if label == "SL" else "TAKE_PROFIT_MARKET"
            params = {
                "symbol": filters.symbol,
                "side": closing_side,
                "type": order_type,
                "stopPrice": px,
                "closePosition": False,
                "reduceOnly": True,
                "quantity": qty,
                "workingType": "CONTRACT_PRICE",
                "newClientOrderId": f"bot-{label.lower()}-{int(time.time() * 1000)}",
            }
            if position_side:
                params["positionSide"] = position_side
            try:
                self._call_with_backoff(self.client.futures_create_order, **params)
            except Exception as exc:
                logger.warning("No se pudo colocar %s reduceOnly: %s", label, exc)


def _build_paper_store(start_equity: float) -> PaperStore:
    path = PAPER_STORE_PATH
    if path.is_dir():
        path = path / "paper_state.json"
    return PaperStore(path, start_equity=start_equity)


def build_broker(settings, client_factory: Callable[..., Any]):
    global ACTIVE_PAPER_STORE, ACTIVE_LIVE_CLIENT
    if settings.PAPER:
        ACTIVE_PAPER_STORE = _build_paper_store(settings.start_equity)
        logger.warning(
            "MODO: 🧪 SIMULADO | equity inicial: %.2f USDT | store=%s",
            settings.start_equity,
            ACTIVE_PAPER_STORE.path,
        )
        fee_rate = float(os.getenv("PAPER_FEE_RATE", "0") or 0.0)
        return SimBroker(ACTIVE_PAPER_STORE, fee_rate=fee_rate)

    assert settings.binance_api_key, "Falta BINANCE_API_KEY (modo real)."
    assert settings.binance_api_secret, "Falta BINANCE_API_SECRET (modo real)."
    # Siempre recrear el cliente al cambiar a REAL para evitar credenciales viejas
    ACTIVE_LIVE_CLIENT = None
    client = client_factory(api_key=settings.binance_api_key, secret=settings.binance_api_secret)
    ACTIVE_LIVE_CLIENT = client
    logger.warning("MODO: 🔴 REAL | Binance listo.")
    return BinanceBroker(client)


__all__ = [
    "SimBroker",
    "BinanceBroker",
    "build_broker",
    "ACTIVE_PAPER_STORE",
    "ACTIVE_LIVE_CLIENT",
    "PAPER_STORE_PATH",
]
