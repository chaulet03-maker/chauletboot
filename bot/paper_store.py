from paper_store import PaperStore


def _store() -> PaperStore:
    return PaperStore()


def set_equity(value: float):
    store = _store()
    store.save(equity=float(value))


def get_equity() -> float:
    store = _store()
    state = store.get_state()
    return float(state.get("equity", 0.0))
