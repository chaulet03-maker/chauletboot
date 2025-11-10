import asyncio
import math

from bot.qty_utils import round_and_validate_qty


class DummyExchange:
    def __init__(self, filters, price):
        self._filters = filters
        self._price = price

    async def get_symbol_filters(self, symbol):
        return self._filters

    async def get_current_price(self, symbol):
        return self._price


def test_round_and_validate_qty_ceils_to_step_for_min_notional():
    exchange = DummyExchange({"stepSize": "0.1", "minNotional": "10"}, 100)
    qty = asyncio.run(round_and_validate_qty(exchange, "BTCUSDT", 0.095))
    assert math.isclose(qty, 0.1)


def test_round_and_validate_qty_ceils_even_if_increase_exceeds_ten_percent():
    exchange = DummyExchange({"stepSize": "0.1", "minNotional": "10"}, 100)
    qty = asyncio.run(round_and_validate_qty(exchange, "BTCUSDT", 0.05))
    assert math.isclose(qty, 0.1)
