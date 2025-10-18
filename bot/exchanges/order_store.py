import json
import os
from pathlib import Path
from typing import Iterable


class OrderStore:
    """Persistencia sencilla para órdenes ejecutadas."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        runtime_dir = self.path.parent
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"No se puede crear la carpeta de runtime en '{runtime_dir}': {exc}"
            ) from exc

        if not os.access(runtime_dir, os.W_OK | os.X_OK):
            raise PermissionError(
                f"La carpeta de runtime '{runtime_dir}' no es escribible para el bot."
            )

        if not self.path.exists():
            try:
                self.path.write_text("[]", encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"No se pudo inicializar el archivo de órdenes en '{self.path}': {exc}"
                ) from exc

    def _read(self) -> list:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def append(self, record: dict):
        data = self._read()
        data.append(record)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def last_n(self, n: int = 50) -> Iterable[dict]:
        data = self._read()
        return data[-n:]
