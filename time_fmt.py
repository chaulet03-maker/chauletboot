from datetime import datetime, timezone
from zoneinfo import ZoneInfo

AR = ZoneInfo("America/Argentina/Buenos_Aires")


def fmt_ar(ts: datetime, with_secs: bool = False) -> str:
    """Return timestamp formatted in Argentina's local time."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    fmt = "%Y-%m-%d %H:%M:%S" if with_secs else "%Y-%m-%d %H:%M"
    return ts.astimezone(AR).strftime(fmt)
