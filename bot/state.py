import json, os, time, datetime as dt

STATE_PATH = "data/state.json"
CMD_QUEUE_PATH = "data/cmd_queue.json"
os.makedirs("data", exist_ok=True)

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
        return data
    except Exception:
        return {"allow_new_entries": True, "positions": {}, "equity": 1000.0, "updated_at": _iso_utc()}

def save_state(st: dict):
    st = dict(st or {})
    st["updated_at"] = _iso_utc()
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH,"w",encoding="utf-8") as f: 
        json.dump(st,f,ensure_ascii=False,indent=2)

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
