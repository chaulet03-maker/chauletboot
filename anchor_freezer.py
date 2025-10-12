from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Literal

Side = Literal["LONG", "SHORT"]


@dataclass
class Frozen:
    anchor: float
    step: float
    span: float
    expires_at: datetime  # UTC


class AnchorFreezer:
    """
    Congela (anchor, step, span) por un tiempo fijo cuando el precio se acerca
    a la zona de entrada. Evita que 'se mueva el arco' mientras el precio llega.
    """

    def __init__(self, approach_ratio: float = 0.90, freeze_secs: int = 300):
        self.approach_ratio = approach_ratio      # 0.90 = 90% de step
        self.freeze_secs = freeze_secs            # 300 s = 5 min
        self._frozen: Dict[Tuple[str, Side], Frozen] = {}

    def clear(self, symbol: str, side: Side) -> None:
        self._frozen.pop((symbol, side), None)

    def expires_at(self, symbol: str, side: Side) -> datetime | None:
        frozen = self._frozen.get((symbol, side))
        if frozen is None:
            return None
        now = datetime.now(timezone.utc)
        if frozen.expires_at <= now:
            self._frozen.pop((symbol, side), None)
            return None
        return frozen.expires_at

    def _should_freeze(self, side: Side, price: float, anchor: float, step: float, span: float) -> bool:
        r = self.approach_ratio
        if side == "SHORT":
            # Se acerca si supera anchor + r*step, pero aún está dentro del corredor
            return (price >= anchor + r * step) and (price <= anchor + span)
        else:  # LONG
            return (price <= anchor - r * step) and (price >= anchor - span)

    def apply(
        self,
        symbol: str,
        side: Side,
        price: float,
        anchor: float,
        step: float,
        span: float,
        now: datetime | None = None,
    ):
        """
        Devuelve (anchor_usado, step_usado, span_usado, status)
        status: 'active' si hay congelamiento vigente, 'armed' si acaba de armar,
                'none' si no aplica.
        """
        key = (symbol, side)
        now = now or datetime.now(timezone.utc)

        # 1) Si hay congelamiento vigente y no venció, usarlo
        frozen = self._frozen.get(key)
        if frozen:
            if now < frozen.expires_at:
                return frozen.anchor, frozen.step, frozen.span, "active"
            else:
                # venció
                self._frozen.pop(key, None)

        # 2) Si no hay, ver si debemos armarlo
        if self._should_freeze(side, price, anchor, step, span):
            expires = now + timedelta(seconds=self.freeze_secs)
            self._frozen[key] = Frozen(anchor=anchor, step=step, span=span, expires_at=expires)
            return anchor, step, span, "armed"

        # 3) Nada que congelar
        return anchor, step, span, "none"
