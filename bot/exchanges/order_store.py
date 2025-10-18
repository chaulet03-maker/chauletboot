import json
from pathlib import Path
from typing import Iterable


class OrderStore:
    """Persistencia sencilla para órdenes ejecutadas."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read(self) -> list:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def append(self, record: dict):
        data = self._read()
        data.append(record)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def last_n(self, n: int = 50) -> Iterable[dict]:
        data = self._read()
        return data[-n:]
