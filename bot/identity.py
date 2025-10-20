import os
import uuid
import pathlib

_ID_FILE = pathlib.Path("data/bot_id.txt")


def get_bot_id() -> str:
    bid = os.getenv("BOT_ID")
    if bid:
        return bid.strip()
    if _ID_FILE.exists():
        return _ID_FILE.read_text(encoding="utf-8").strip()
    _ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    bid = ("BOT-" + uuid.uuid4().hex[:10]).upper()
    _ID_FILE.write_text(bid, encoding="utf-8")
    return bid


def make_client_oid(prefix: str, symbol: str, mode: str) -> str:
    """
    newClientOrderId para Binance (máx ~32 chars). Ej: BOT_ABCDEF_L_BTCUSDT_1729440000
    """
    import time
    import re

    ts = int(time.time())
    sym = re.sub(r"[^A-Z0-9]", "", symbol.upper())[:10]
    base = f"{prefix}_{mode[0].upper()}_{sym}_{ts}"
    return base[:32]
