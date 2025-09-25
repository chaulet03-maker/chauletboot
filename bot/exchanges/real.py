import asyncio, logging, hashlib
from dataclasses import dataclass
from time import time as _t

@dataclass
class Fill:
    price: float
    qty: float
    side: str
    ts: float

class RealExchange:
    def __init__(self, ccxt_client, fees: dict):
        self.client = ccxt_client
        self.fees = fees or {"taker": 0.0002, "maker": 0.0002}
        self.log = logging.getLogger("RealExchange")

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
        try:
            if hasattr(self.client, "setLeverage"):
                return await self.client.setLeverage(lev, symbol)
            m = self.client.market(symbol)
            return await self.client.fapiPrivate_post_leverage({"symbol": m["id"], "leverage": lev})
        except Exception as e:
            self.log.warning("set_leverage(%s,%s) failed: %s", symbol, lev, e)

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

    async def place_protections(self, symbol: str, side: str, qty: float,
                                sl: float = 0.0, tp1: float = 0.0, tp2: float = 0.0,
                                position_side: str = None):
        placed = []
        if qty <= 0:
            return placed
        closing_side = "sell" if side == "long" else "buy"
        params_base = {"reduceOnly": True}
        if position_side:
            params_base["positionSide"] = position_side

        if sl and sl > 0:
            coid = self._idemp_key("SL", symbol=symbol, side=closing_side, qty=qty, sl=sl)
            params_sl = dict(params_base, newClientOrderId=coid, stopPrice=float(sl), workingType="CONTRACT_PRICE")
            try:
                o_sl = await self._place(self.client.create_order, symbol, "STOP_MARKET", closing_side, qty, None, params_sl)
                placed.append(o_sl)
            except Exception as e:
                self.log.warning("place SL failed: %s", e)

        if tp2 and tp2 > 0:
            coid2 = self._idemp_key("TP", symbol=symbol, side=closing_side, qty=qty, tp=tp2)
            params_tp = dict(params_base, newClientOrderId=coid2, stopPrice=float(tp2), workingType="CONTRACT_PRICE")
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
