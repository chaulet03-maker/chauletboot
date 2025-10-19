import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brokers import SimBroker
from paper_store import PaperStore


@pytest.fixture
def paper_store(tmp_path):
    path = tmp_path / "paper_state.json"
    return PaperStore(path, start_equity=1000.0)


def test_sim_broker_updates_position_immediately(paper_store):
    broker = SimBroker(paper_store)
    result = broker.place_order("BUY", 0.5, 20000.0, symbol="BTCUSDT")
    assert result["status"] == "FILLED"
    assert pytest.approx(result["executedQty"], rel=1e-6) == 0.5
    assert pytest.approx(result["avgPrice"], rel=1e-6) == 20000.0
    state = paper_store.load()
    assert pytest.approx(state["pos_qty"], rel=1e-6) == 0.5
    assert pytest.approx(state["avg_price"], rel=1e-6) == 20000.0
    assert result["state"]["pos_qty"] == pytest.approx(0.5)
