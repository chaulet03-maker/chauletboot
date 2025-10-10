import json, os, time, datetime as dt

STATE_PATH = "data/state.json"
SIMPLE_STATE_PATH = "data/estado.json"
CMD_QUEUE_PATH = "data/cmd_queue.json"
os.makedirs("data", exist_ok=True)


def _simple_state_template():
    return {
        "posicion_activa": False,
        "side": None,
        "entry_price": 0.0,
        "qty": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
    }


def _normalize_simple_state(data: dict | None):
    tmpl = _simple_state_template()
    if not isinstance(data, dict):
        return tmpl
    out = {}
    out["posicion_activa"] = bool(data.get("posicion_activa", False))
    out["side"] = data.get("side") if isinstance(data.get("side"), str) else None
    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return 0.0
    out["entry_price"] = _to_float(data.get("entry_price"))
    out["qty"] = _to_float(data.get("qty"))
    out["stop_loss"] = _to_float(data.get("stop_loss"))
    out["take_profit"] = _to_float(data.get("take_profit"))
    return {**tmpl, **out}


def _simple_state_from_snapshot(snapshot: dict):
    simple = _simple_state_template()
    positions = (snapshot or {}).get("positions") or {}
    if isinstance(positions, dict):
        for lots in positions.values():
            if not isinstance(lots, list) or not lots:
                continue
            lot = lots[0]
            if not isinstance(lot, dict):
                continue
            simple["posicion_activa"] = True
            side = lot.get("side")
            simple["side"] = side if isinstance(side, str) else None
            def _pick_float(key, fallback=0.0):
                try:
                    return float(lot.get(key, fallback) or fallback)
                except Exception:
                    return fallback
            simple["entry_price"] = _pick_float("entry", 0.0)
            simple["qty"] = _pick_float("qty", 0.0)
            simple["stop_loss"] = _pick_float("sl", 0.0)
            tp = lot.get("tp2", lot.get("tp1", 0.0))
            try:
                simple["take_profit"] = float(tp or 0.0)
            except Exception:
                simple["take_profit"] = 0.0
            break
    return simple


def load_simple_state():
    """Carga el estado compacto de posición para reinicios rápidos."""
    if not os.path.exists(SIMPLE_STATE_PATH):
        save_simple_state(_simple_state_template())
    try:
        with open(SIMPLE_STATE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return _normalize_simple_state(data)
    except Exception:
        return _simple_state_template()


def save_simple_state(state: dict):
    data = _normalize_simple_state(state)
    os.makedirs(os.path.dirname(SIMPLE_STATE_PATH), exist_ok=True)
    with open(SIMPLE_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

def _iso_utc(ts=None):
    if ts is None: ts = time.time()
    return dt.datetime.utcfromtimestamp(ts).isoformat()+"Z"

def load_state():
    """Carga estado genérico {allow_new_entries, positions, equity, updated_at}. Tolerante a formatos viejos."""
    if not os.path.exists(STATE_PATH):
        save_state({"allow_new_entries": True, "positions": {}, "equity": 1000.0, "updated_at": _iso_utc()})
    try:
        with open(STATE_PATH,"r",encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):  # esquemas viejos
            data = {}
        if isinstance(data.get("positions", {}), list):
            data["positions"] = {}
        if "equity" in data and not isinstance(data.get("equity"), (int,float)):
            data["equity"] = 1000.0
        data.setdefault("allow_new_entries", True)
        data.setdefault("updated_at", _iso_utc())
        try:
            save_simple_state(_simple_state_from_snapshot(data))
        except Exception:
            pass
        return data
    except Exception:
        fallback = {"allow_new_entries": True, "positions": {}, "equity": 1000.0, "updated_at": _iso_utc()}
        try:
            save_simple_state(_simple_state_from_snapshot(fallback))
        except Exception:
            pass
        return fallback

def save_state(st: dict):
    st = dict(st or {})
    st["updated_at"] = _iso_utc()
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH,"w",encoding="utf-8") as f:
        json.dump(st,f,ensure_ascii=False,indent=2)
    try:
        save_simple_state(_simple_state_from_snapshot(st))
    except Exception:
        pass

def enqueue_cmd(cmd: dict):
    q = []
    if os.path.exists(CMD_QUEUE_PATH):
        try:
            with open(CMD_QUEUE_PATH,"r",encoding="utf-8") as f: q = json.load(f)
        except Exception: q=[]
    q.append(cmd)
    with open(CMD_QUEUE_PATH,"w",encoding="utf-8") as f: json.dump(q,f,ensure_ascii=False,indent=2)

def read_and_clear_cmds():
    q=[]
    if os.path.exists(CMD_QUEUE_PATH):
        with open(CMD_QUEUE_PATH,"r",encoding="utf-8") as f: q = json.load(f)
        try: os.remove(CMD_QUEUE_PATH)
        except Exception: pass
    return q
