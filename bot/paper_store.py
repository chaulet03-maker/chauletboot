import json
import os
import time

PATH = "data/runtime/paper_state.json"


def _ensure():
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    if not os.path.exists(PATH):
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump({"equity": 0.0, "updated": int(time.time())}, f)


def set_equity(value: float):
    _ensure()
    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["equity"] = float(value)
    data["updated"] = int(time.time())
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def get_equity() -> float:
    _ensure()
    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("equity", 0.0))
