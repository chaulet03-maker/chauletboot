import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.exchanges.binance_filters import (
    build_filters,
    quantize_price,
    quantize_qty,
    validate_order,
)
from brokers import BinanceBroker


class DummyClient:
    def __init__(self):
        self._orders = []
        self._info_calls = 0

    def futures_exchange_info(self):
        self._info_calls += 1
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }

    def futures_create_order(self, **payload):
        self._orders.append(payload)
        return {"order": payload}


@pytest.fixture
def btc_filters():
    market = {
        "info": {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        }
    }
    return build_filters("BTCUSDT", market)


def test_quantize_qty_and_price(btc_filters):
    assert quantize_qty(btc_filters, 0.001234) == pytest.approx(0.001)
    assert quantize_price(btc_filters, 68234.123) == pytest.approx(68234.1)


def test_validate_order_min_notional(btc_filters):
    with pytest.raises(ValueError):
        validate_order(btc_filters, 0.001, 100.0)


def test_binance_broker_quantizes_and_sets_ids(monkeypatch):
    client = DummyClient()
    broker = BinanceBroker(client)
    result = broker.place_order(
        "BUY",
        0.001234,
        68234.123,
        symbol="BTC/USDT",
        client_order_id="test-123",
        sl=67000.333,
        tp=70000.777,
        order_type="LIMIT",
    )
    assert "order" in result
    assert len(client._orders) >= 1
    entry = client._orders[0]
    assert entry["quantity"] == pytest.approx(0.001)
    assert entry["price"] == pytest.approx(68234.1)
    assert entry["newClientOrderId"] == "test-123"
    assert entry["type"] == "LIMIT"
    assert entry["timeInForce"] == "GTC"
    # Protections should be reduce-only and quantized
    protection_orders = client._orders[1:]
    assert protection_orders, "Stop loss/TP orders were not placed"
    for protection in protection_orders:
        assert protection["reduceOnly"] is True
        assert protection["quantity"] == pytest.approx(0.001)
        expected = quantize_price(
            build_filters("BTCUSDT", {"info": client.futures_exchange_info()["symbols"][0]}),
            protection["stopPrice"],
        )
        assert protection["stopPrice"] == pytest.approx(expected)


def test_binance_broker_retries_network(monkeypatch):
    class FlakyClient(DummyClient):
        def __init__(self):
            super().__init__()
            self._fails = 0

        def futures_create_order(self, **payload):
            if self._fails < 1:
                self._fails += 1
                raise Exception("Request timed out")
            return super().futures_create_order(**payload)

    broker = BinanceBroker(FlakyClient())
    broker.place_order("BUY", 0.0015, 20000.0, symbol="BTCUSDT", order_type="LIMIT")


def test_binance_broker_does_not_retry_margin(monkeypatch):
    class MarginError(Exception):
        code = -2019

    class MarginClient(DummyClient):
        def futures_create_order(self, **payload):
            raise MarginError("insufficient margin")

    broker = BinanceBroker(MarginClient())
    with pytest.raises(MarginError):
        broker.place_order("BUY", 0.01, 25000.0, symbol="BTCUSDT", order_type="LIMIT")


def test_binance_broker_market_defaults(monkeypatch):
    client = DummyClient()
    broker = BinanceBroker(client)
    broker.place_order("SELL", 0.01, None, symbol="BTC/USDT", order_type="market")
    entry = client._orders[0]
    assert entry["type"] == "MARKET"
    assert "price" not in entry
    assert "timeInForce" not in entry
