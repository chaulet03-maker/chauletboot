import pytest

from position_service import split_total_vs_bot


def _acct_position(signed_qty: float, entry_price: float) -> dict[str, float]:
    return {"positionAmt": str(signed_qty), "entryPrice": float(entry_price)}


def _bot_position(side: str, qty: float, entry_price: float) -> dict[str, float]:
    return {"side": side, "qty": float(qty), "entry_price": float(entry_price)}


def test_split_total_vs_bot_only_bot_position():
    acct = _acct_position(1.0, 100.0)
    bot = _bot_position("LONG", 1.0, 100.0)

    split = split_total_vs_bot(acct, bot, mark=105.0)

    assert split["total"]["side"] == "LONG"
    assert split["total"]["qty"] == pytest.approx(1.0)
    assert split["manual"]["qty"] == pytest.approx(0.0)
    assert split["manual"]["pnl"] == pytest.approx(0.0)


def test_split_total_vs_bot_manual_same_direction():
    # Total LONG 1.5 @ 96.6666…, bot LONG 0.5 @ 110 ⇒ manual LONG 1 @ 90
    acct = _acct_position(1.5, 96.6666667)
    bot = _bot_position("LONG", 0.5, 110.0)

    split = split_total_vs_bot(acct, bot, mark=100.0)

    manual = split["manual"]
    assert manual["side"] == "LONG"
    assert manual["qty"] == pytest.approx(1.0)
    assert manual["entry_price"] == pytest.approx(90.0, rel=1e-6, abs=1e-6)
    assert manual["pnl"] == pytest.approx(10.0, rel=1e-6, abs=1e-6)


def test_split_total_vs_bot_manual_opposite_direction():
    # Total SHORT 0.2 @ 222.5, bot LONG 0.5 @ 205 ⇒ manual SHORT 0.7 @ 210
    acct = _acct_position(-0.2, 222.5)
    bot = _bot_position("LONG", 0.5, 205.0)

    split = split_total_vs_bot(acct, bot, mark=215.0)

    assert split["total"]["side"] == "SHORT"
    assert split["total"]["qty"] == pytest.approx(0.2, rel=1e-6, abs=1e-6)

    manual = split["manual"]
    assert manual["side"] == "SHORT"
    assert manual["qty"] == pytest.approx(0.7, rel=1e-6, abs=1e-6)
    assert manual["entry_price"] == pytest.approx(210.0, rel=1e-6, abs=1e-6)
    assert manual["pnl"] == pytest.approx(-3.5, rel=1e-6, abs=1e-6)
