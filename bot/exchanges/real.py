import asyncio, logging, hashlib, inspect
from dataclasses import dataclass
from time import time as _t
from typing import Any, Dict, List, Optional

from ccxt.base.errors import ExchangeError

from .order_store import OrderStore


async def _create_order_tolerant(
    place_order,
    symbol: str,
    order_type: str,
    side: str,
    qty: float,
    price: float | None = None,
    params: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
):
    """Intenta crear una orden sin ``positionSide`` y reintenta en modo hedge."""

    params_dict = dict(params or {})
    side_u = (side or "").upper()

    try:
        order = await place_order(symbol, order_type, side_u, qty, price, params_dict)
        return order, params_dict
    except ExchangeError as exc:
        msg = str(exc or "").lower()
        code = getattr(exc, "code", None)
        mismatch = any(
            token in msg for token in ("-4061", "position side does not match")
        ) or code == -4061
        if not mismatch:
            raise

        retry_params = dict(params_dict)
        retry_params["positionSide"] = "LONG" if side_u in {"BUY", "LONG"} else "SHORT"

        if logger is not None:
            logger.debug(
                "Reintento de orden con positionSide=%s tras mismatch de modo de posición",
                retry_params["positionSide"],
            )

        order = await place_order(symbol, order_type, side_u, qty, price, retry_params)
        return order, retry_params

@dataclass
class Fill:
    price: float
    qty: float
    side: str
    ts: float

class RealExchange:
    def __init__(
        self,
        ccxt_client,
        fees: dict,
        store_path: str = "./runtime/orders.json",
        symbol: str = "BTC/USDT",
        native_client: Any | None = None,
    ):
        default_symbol = symbol or "BTC/USDT"
        self.client = ccxt_client
        self.ccxt = ccxt_client
        self.fees = fees or {"taker": 0.0002, "maker": 0.0002}
        self.log = logging.getLogger("RealExchange")
        self.store = OrderStore(store_path)
        self.symbol = default_symbol
        self._sym_noslash = default_symbol.replace("/", "")
        self._native_client = native_client

    def _idemp_key(self, prefix: str, **fields) -> str:
        raw = prefix + "|" + "|".join(f"{k}={fields[k]}" for k in sorted(fields.keys()))
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return f"{prefix}-{h}"

    def _normalize_order_side(self, side: str) -> str:
        side_u = (side or "").upper()
        if side_u in {"BUY", "LONG"}:
            return "BUY"
        if side_u in {"SELL", "SHORT"}:
            return "SELL"
        raise ValueError(f"Lado de orden inválido: {side}")

    async def _place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        qty: float,
        price: float | None,
        params: Optional[dict] = None,
    ):
        return await self._place(
            self.client.create_order,
            symbol,
            order_type,
            side,
            qty,
            price,
            params,
        )

    async def _create_order_adaptive(
        self,
        symbol: str,
        order_type: str,
        side: str,
        qty: float,
        price: float | None,
        params: Optional[dict] = None,
    ):
        return await _create_order_tolerant(
            self._place_order,
            symbol,
            order_type,
            side,
            qty,
            price,
            params,
            logger=self.log,
        )

    async def _place(self, fn, *args, retries=4, base_delay=0.25, **kwargs):
        last = None
        for i in range(retries):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                last = e
                await asyncio.sleep(base_delay * (1.8 ** i) * (1 + 0.2))
        raise last

    async def set_position_mode(self, one_way: bool = True):
        self.log.debug(
            "set_position_mode(one_way=%s) omitido: se utiliza el modo configurado en la cuenta",
            one_way,
        )

    async def _ensure_cross_margin(self, symbol: str):
        """Intenta poner CROSS para el símbolo dado."""
        try:
            fn = getattr(self.client, "set_margin_mode", None) or getattr(self.client, "set_marginMode", None)
            if callable(fn):
                await fn("cross", symbol)
        except Exception as exc:
            self.log.warning("No se pudo establecer margin_mode CROSS para %s: %s", symbol, exc)

    async def set_leverage(self, symbol: str, lev: int):
        """Wrapper compatible con CCXT Python para USDM."""
        try:
            lev_int = int(float(lev))
            await self._ensure_cross_margin(symbol)
            # Preferir la API de alto nivel de CCXT
            return await self.client.set_leverage(lev_int, symbol)
        except Exception as e:
            # Fallback explícito al endpoint oficial si hiciera falta
            try:
                m = self.client.market(symbol)  # m["id"] -> 'BTCUSDT'
                return await self.client.fapiPrivatePostLeverage(
                    {"symbol": m["id"], "leverage": lev_int}
                )
            except Exception as ex:
                self.log.warning("set_leverage(%s,%s) failed: %s / %s", symbol, lev_int, e, ex)
                raise

    async def market_order(self, symbol: str, side: str, qty: float, price_hint: float = None):
        params = {"newClientOrderId": self._idemp_key("MO", symbol=symbol, side=side, qty=qty)}
        order_side = self._normalize_order_side(side)
        o, _ = await self._create_order_adaptive(symbol, "market", order_side, qty, None, params)
        price = float(
            o.get("average")
            or o.get("price")
            or (o.get("info", {}) or {}).get("avgPrice")
            or (o.get("info", {}) or {}).get("price")
            or (price_hint or 0.0)
        )
        fee = abs(price * qty) * self.fees.get("taker", 0.0002)
        return Fill(price=price, qty=qty, side=side, ts=_t()), fee

    async def create_order(
        self,
        symbol: str,
        internal_side: str,
        qty: float,
        price: float | None = None,
        type_: str = "market",
        params: dict | None = None,
    ):
        params = dict(params or {})

        order_type = (type_ or "market").upper()
        binance_type = order_type.lower()
        if order_type not in {"MARKET", "LIMIT"}:
            self.log.warning(
                "Tipo de orden no contemplado (%s), enviándolo directamente a CCXT",
                order_type,
            )

        order_side = self._normalize_order_side(internal_side)
        internal_norm = "LONG" if order_side == "BUY" else "SHORT"
        price_arg = price if order_type == "LIMIT" else None

        try:
            order, used_params = await self._create_order_adaptive(
                symbol,
                binance_type,
                order_side,
                qty,
                price_arg,
                params,
            )
        except Exception as e:
            self.log.exception("Fallo creando orden real: %s", e)
            raise

        oid = (
            order.get("id")
            or order.get("orderId")
            or order.get("clientOrderId")
            or (order.get("info", {}) or {}).get("orderId")
            or (order.get("info", {}) or {}).get("clientOrderId")
        )
        status = (order.get("status") or (order.get("info", {}) or {}).get("status") or "").upper()
        if not oid:
            raise RuntimeError(f"Binance no devolvió id de orden: {order}")

        self.store.append(
            {
                "ts": _t(),
                "symbol": symbol,
                "internal_side": internal_norm,
                "side": order_side,
                "positionSide": (used_params or {}).get("positionSide"),
                "type": order_type,
                "qty": qty,
                "price": price_arg,
                "status": status,
                "order": order,
            }
        )

        return order

    def _format_symbol(self, symbol: str) -> str:
        if not symbol:
            return symbol
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        return symbol

    def _get_native_client(self):
        if self._native_client is not None:
            return self._native_client
        try:
            from brokers import ACTIVE_LIVE_CLIENT  # type: ignore

            self._native_client = ACTIVE_LIVE_CLIENT
        except Exception:
            self._native_client = None
        return self._native_client

    async def _call_native(self, func, *args, **kwargs):
        if func is None:
            return None
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    async def get_open_position(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        sym = symbol or self.symbol or "BTC/USDT"
        target = sym.replace("/", "")

        ccxt_client = getattr(self, "ccxt", None) or self.client
        if ccxt_client is not None and hasattr(ccxt_client, "fetch_positions"):
            try:
                pos_list = await self._place(ccxt_client.fetch_positions, [sym])
                for entry in pos_list or []:
                    info = entry.get("info") or {}
                    exch_sym = str(info.get("symbol") or entry.get("symbol") or "").upper()
                    # normalizar para comparar: quitar "/"
                    exch_id = exch_sym.replace("/", "")
                    if exch_id != target.upper():
                        continue
                    raw_amt = info.get("positionAmt") or info.get("positionamt")
                    if raw_amt is None:
                        raw_amt = entry.get("contracts") or entry.get("size") or 0
                    amt_f = float(raw_amt or "0")
                    if abs(amt_f) == 0.0:
                        return None
                    side = "LONG" if amt_f > 0 else "SHORT"
                    entry_price = float(info.get("entryPrice") or entry.get("entryPrice") or 0.0)
                    mark_price = float(
                        info.get("markPrice")
                        or info.get("markprice")
                        or entry.get("markPrice")
                        or entry.get("markprice")
                        or 0.0
                    )
                    return {
                        "symbol": sym,
                        "side": side,
                        "contracts": abs(amt_f),
                        "entryPrice": entry_price,
                        "markPrice": mark_price,
                    }
            except Exception:
                pass

        native_client = self._get_native_client()
        if native_client is None:
            return None

        try:
            account = await self._call_native(getattr(native_client, "futures_account", None))
            if not account:
                return None
            for pos in account.get("positions", []):
                exch_sym = str(pos.get("symbol") or "").upper()
                # normalizar para comparar: quitar "/"
                exch_id = exch_sym.replace("/", "")
                if exch_id != target.upper():
                    continue
                raw_amt = pos.get("positionAmt") or pos.get("positionamt")
                amt_f = float(raw_amt or "0")
                if abs(amt_f) == 0.0:
                    return None
                side = "LONG" if amt_f > 0 else "SHORT"
                entry_price = float(pos.get("entryPrice") or 0.0)
                mark_raw = pos.get("markPrice")
                mark_price = float(mark_raw or 0.0)
                if mark_price == 0.0 and hasattr(native_client, "futures_mark_price"):
                    try:
                        mp = await self._call_native(
                            native_client.futures_mark_price,
                            symbol=target.upper(),
                        )
                    except Exception:
                        mp = None
                    if mp:
                        try:
                            mark_price = float(mp.get("markPrice") or 0.0)
                        except Exception:
                            mark_price = 0.0
                return {
                    "symbol": sym,
                    "side": side,
                    "contracts": abs(amt_f),
                    "entryPrice": entry_price,
                    "markPrice": mark_price,
                }
        except Exception:
            return None
        return None

    async def list_open_positions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        ccxt_client = getattr(self, "ccxt", None) or self.client
        if ccxt_client is not None and hasattr(ccxt_client, "fetch_positions"):
            try:
                pos_list = await self._place(ccxt_client.fetch_positions)
                for entry in pos_list or []:
                    info = entry.get("info") or {}
                    raw_amt = info.get("positionAmt") or info.get("positionamt")
                    if raw_amt is None:
                        raw_amt = entry.get("contracts") or entry.get("size") or 0
                    amt_f = float(raw_amt or "0")
                    if abs(amt_f) == 0.0:
                        continue
                    side = "LONG" if amt_f > 0 else "SHORT"
                    sym_raw = str(info.get("symbol") or entry.get("symbol") or "")
                    out.append(
                        {
                            "symbol": self._format_symbol(sym_raw),
                            "side": side,
                            "contracts": abs(amt_f),
                            "entryPrice": float(info.get("entryPrice") or entry.get("entryPrice") or 0.0),
                            "markPrice": float(
                                info.get("markPrice")
                                or info.get("markprice")
                                or entry.get("markPrice")
                                or entry.get("markprice")
                                or 0.0
                            ),
                        }
                    )
                return out
            except Exception:
                pass

        native_client = self._get_native_client()
        if native_client is None:
            return []

        try:
            account = await self._call_native(getattr(native_client, "futures_account", None))
            if not account:
                return []
            for pos in account.get("positions", []):
                raw_amt = pos.get("positionAmt") or pos.get("positionamt")
                amt_f = float(raw_amt or "0")
                if abs(amt_f) == 0.0:
                    continue
                side = "LONG" if amt_f > 0 else "SHORT"
                sym_raw = str(pos.get("symbol") or "")
                mark_price = float(pos.get("markPrice") or 0.0)
                if mark_price == 0.0 and hasattr(native_client, "futures_mark_price"):
                    try:
                        mp = await self._call_native(
                            native_client.futures_mark_price,
                            symbol=str(sym_raw).replace("/", ""),
                        )
                    except Exception:
                        mp = None
                    if mp:
                        try:
                            mark_price = float(mp.get("markPrice") or 0.0)
                        except Exception:
                            mark_price = 0.0
                out.append(
                    {
                        "symbol": self._format_symbol(sym_raw),
                        "side": side,
                        "contracts": abs(amt_f),
                        "entryPrice": float(pos.get("entryPrice") or 0.0),
                        "markPrice": mark_price,
                    }
                )
        except Exception:
            return []
        return out

    # =============================================================
    # === PARCHE APLICADO AQUÍ ===
    # =============================================================
    async def place_protections(self, symbol: str, side: str, qty: float,
                                sl: float = 0.0, tp1: float = 0.0, tp2: float = 0.0,
                                position_side: str = None):
        """
        REAL: envía un único SL y un único TP (compat con firmas previas).
        Si vienen tp1 y tp2, prioriza tp2; si no, usa tp1.
        """
        placed = []
        if qty <= 0:
            return placed

        # Determina el TP final, dando prioridad a tp2.
        tp_final = tp2 if tp2 and tp2 > 0 else (tp1 if tp1 and tp1 > 0 else None)
        s = (side or "").lower()
        closing_side = "sell" if s == "long" else "buy"
        closing_order_side = self._normalize_order_side(closing_side)
        params_base = {"reduceOnly": True}
        if position_side:
            params_base["positionSide"] = position_side

        # --- STOP LOSS ---
        if sl and sl > 0:
            coid_sl = self._idemp_key("SL", symbol=symbol, side=closing_side, qty=qty, sl=sl)
            params_sl = dict(params_base, newClientOrderId=coid_sl, stopPrice=float(sl), workingType="CONTRACT_PRICE")
            try:
                o_sl, _ = await self._create_order_adaptive(
                    symbol,
                    "STOP_MARKET",
                    closing_order_side,
                    qty,
                    None,
                    params_sl,
                )
                placed.append(o_sl)
            except Exception as e:
                self.log.warning("place SL failed: %s", e)

        # --- TAKE PROFIT (único) ---
        if tp_final:
            coid_tp = self._idemp_key("TP", symbol=symbol, side=closing_side, qty=qty, tp=tp_final)
            params_tp = dict(params_base, newClientOrderId=coid_tp, stopPrice=float(tp_final), workingType="CONTRACT_PRICE")
            try:
                o_tp, _ = await self._create_order_adaptive(
                    symbol,
                    "TAKE_PROFIT_MARKET",
                    closing_order_side,
                    qty,
                    None,
                    params_tp,
                )
                placed.append(o_tp)
            except Exception as e:
                self.log.warning("place TP failed: %s", e)

        return placed

    async def close_all(self, positions: dict):
        for sym, lots in positions.items():
            net = sum(l["qty"] if l["side"] == "long" else -l["qty"] for l in lots)
            if abs(net) < 1e-12:
                continue
            side = "sell" if net > 0 else "buy"
            try:
                await self._place(self.client.create_order, sym, "market", side, abs(net), None,
                                  {"reduceOnly": True, "newClientOrderId": self._idemp_key("CLOSE", symbol=sym, side=side, qty=abs(net))})
            except Exception as e:
                self.log.warning("close_all %s failed: %s", sym, e)

    async def fetch_balance_usdt(self):
        try:
            bal = await self._place(self.client.fetch_balance)
            usdt = bal.get("USDT") or bal.get("total", {}).get("USDT")
            if isinstance(usdt, dict):
                return float(usdt.get("free", 0.0) + usdt.get("used", 0.0))
            return float(usdt or 0.0)
        except Exception:
            return 0.0

    async def get_mark_price(self, symbol: str | None = None) -> float:
        sym = symbol or self.symbol or "BTC/USDT"
        client = self.ccxt or self.client
        if client is None:
            raise RuntimeError("CCXT client no disponible")

        sym_clean = sym.replace("/", "").upper()

        async def _call(name: str):
            method = getattr(client, name, None)
            if method is None:
                return None
            try:
                result = method({"symbol": sym_clean})
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception:
                return None

        for name in ("public_get_premiumindex", "fapiPublicGetPremiumIndex"):
            result = await _call(name)
            if not result:
                continue
            payloads: list[dict] = []
            if isinstance(result, dict):
                payloads = [result]
            elif isinstance(result, list):
                payloads = [r for r in result if isinstance(r, dict)]
            for payload in payloads:
                sym_match = payload.get("symbol") or payload.get("pair")
                if sym_match and str(sym_match).upper() != sym_clean:
                    continue
                for key in ("markPrice", "markprice", "indexPrice", "indexprice"):
                    value = payload.get(key)
                    if value is None:
                        continue
                    try:
                        return float(value)
                    except Exception:
                        continue

        ticker = await self._place(client.fetch_ticker, sym)
        for key in ("last", "close", "bid", "ask"):
            value = ticker.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue

        raise RuntimeError(f"No pude obtener precio para {sym}")

    async def fetch_positions(self):
        try:
            return await self._place(self.client.fetch_positions)
        except Exception:
            return []
