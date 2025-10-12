from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

AR = ZoneInfo("America/Argentina/Buenos_Aires")


def fmt_ar(ts: datetime) -> str:
    """Return the hour/minute in Argentina's local time."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(AR).strftime("%H:%M")


def fmt_int(x: float) -> str:
    """Format numbers using dot as thousands separator without decimals."""
    return f"{int(round(x)):,}".replace(",", ".")
