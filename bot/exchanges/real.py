import asyncio, logging, hashlib
from dataclasses import dataclass
from time import time as _t
from typing import Any, Dict, List, Optional

from .order_store import OrderStore
from .side_map import normalize_side

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
        try:
            hedged = not one_way
            if hasattr(self.client, "setPositionMode"):
                return await self.client.setPositionMode(hedged)
            return await self.client.fapiPrivate_post_positionside_dual({"dualSidePosition": "true" if hedged else "false"})
        except Exception as e:
            self.log.warning("set_position_mode failed: %s", e)

    async def set_leverage(self, symbol: str, lev: int):
        """Wrapper compatible con CCXT Python para USDM."""
        try:
            lev_int = int(float(lev))
            # Preferir la API de alto nivel de CCXT
            return await asyncio.to_thread(self.client.set_leverage, lev_int, symbol)
        except Exception as e:
            # Fallback explícito al endpoint oficial si hiciera falta
            try:
                m = self.client.market(symbol)  # m["id"] -> 'BTCUSDT'
                return await asyncio.to_thread(
                    self.client.fapiPrivatePostLeverage,
                    {"symbol": m["id"], "leverage": lev_int},
                )
            except Exception as ex:
                self.log.warning("set_leverage(%s,%s) failed: %s / %s", symbol, lev_int, e, ex)
                raise

    async def market_order(self, symbol: str, side: str, qty: float, price_hint: float = None):
        params = {"newClientOrderId": self._idemp_key("MO", symbol=symbol, side=side, qty=qty)}
        o = await self._place(self.client.create_order, symbol, "market", side, qty, None, params=params)
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

        options = getattr(self.client, "options", {}) or {}
        default_type = options.get("defaultType") or options.get("default_type")
        client_type = getattr(self.client, "type", None)
        is_futures = (default_type in {"future", "swap", "delivery"}) or (
            client_type in {"future", "swap", "delivery"}
        )
        hedge_mode = bool(options.get("hedgeMode") or options.get("hedge_mode"))

        side_map = normalize_side(internal_side, futures=is_futures, hedge_mode=hedge_mode)

        order_type = (type_ or "market").upper()
        binance_type = order_type.lower()
        if order_type not in {"MARKET", "LIMIT"}:
            self.log.warning("Tipo de orden no contemplado (%s), usando CCXT tal cual", order_type)

        binance_side = side_map.order_side
        if side_map.position_side:
            params.setdefault("positionSide", side_map.position_side)

        price_arg = price if order_type == "LIMIT" else None

        try:
            order = await self._place(
                self.client.create_order,
                symbol,
                binance_type,
                binance_side,
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
                "internal_side": side_map.internal,
                "side": binance_side,
                "positionSide": params.get("positionSide"),
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
        target_upper = target.upper()

        ccxt_client = getattr(self, "ccxt", None) or self.client
        if ccxt_client is not None and hasattr(ccxt_client, "fetch_positions"):
            try:
                pos_list = await self._place(ccxt_client.fetch_positions, [sym])
                for entry in pos_list or []:
                    info = entry.get("info") or {}
                    exch_sym = str(info.get("symbol") or entry.get("symbol") or "").upper()
                    exch_id = exch_sym.replace("/", "")
                    if exch_id != target_upper:
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
                exch_id = exch_sym.replace("/", "")
                if exch_id != target_upper:
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
                            symbol=target_upper,
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
        params_base = {"reduceOnly": True}
        if position_side:
            params_base["positionSide"] = position_side

        # --- STOP LOSS ---
        if sl and sl > 0:
            coid_sl = self._idemp_key("SL", symbol=symbol, side=closing_side, qty=qty, sl=sl)
            params_sl = dict(params_base, newClientOrderId=coid_sl, stopPrice=float(sl), workingType="CONTRACT_PRICE")
            try:
                o_sl = await self._place(self.client.create_order, symbol, "STOP_MARKET", closing_side, qty, None, params_sl)
                placed.append(o_sl)
            except Exception as e:
                self.log.warning("place SL failed: %s", e)

        # --- TAKE PROFIT (único) ---
        if tp_final:
            coid_tp = self._idemp_key("TP", symbol=symbol, side=closing_side, qty=qty, tp=tp_final)
            params_tp = dict(params_base, newClientOrderId=coid_tp, stopPrice=float(tp_final), workingType="CONTRACT_PRICE")
            try:
                o_tp = await self._place(self.client.create_order, symbol, "TAKE_PROFIT_MARKET", closing_side, qty, None, params_tp)
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

    async def fetch_positions(self):
        try:
            return await self._place(self.client.fetch_positions)
        except Exception:
            return []