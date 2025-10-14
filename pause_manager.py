import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union


class ShockPauseManager:
    """
    - Guarda/lee 'pause_until' en disco para sobrevivir reinicios.
    - Soporta override manual de 'reanudar'.
    """

    def __init__(self, state_path: str = "pause_state.json"):
        self.state_file = Path(state_path)
        self.pause_until: Optional[datetime] = None
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                if "pause_until" in data and data["pause_until"]:
                    self.pause_until = datetime.fromisoformat(data["pause_until"])
            except Exception:
                self.pause_until = None

    def _save(self) -> None:
        data = {"pause_until": self.pause_until.isoformat() if self.pause_until else None}
        try:
            self.state_file.write_text(json.dumps(data))
        except Exception:
            pass

    def set_pause_hours(self, hours: Union[int, float]):
        now = datetime.now(timezone.utc)
        self.pause_until = now + timedelta(hours=float(hours))
        self._save()
        return self.pause_until

    def clear_pause(self) -> None:
        self.pause_until = None
        self._save()

    def is_paused(self) -> bool:
        if not self.pause_until:
            return False
        return datetime.now(timezone.utc) < self.pause_until

    def is_paused_now(self) -> bool:
        """Compat helper to expose a more explicit check name."""

        return self.is_paused()

    def get_pause_until(self) -> Optional[datetime]:
        """Devuelve el instante hasta el cual la pausa permanece activa."""

        return self.pause_until

    def remaining(self) -> Optional[timedelta]:
        if not self.pause_until:
            return None
        delta = self.pause_until - datetime.now(timezone.utc)
        return delta if delta.total_seconds() > 0 else None
