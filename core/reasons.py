"""Helpers to convert technical rejection contexts into human messages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - zoneinfo may be unavailable on some platforms
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def simplify_reason(ctx: Mapping[str, Any]) -> str:
    """Map a technical context dict to a friendly explanation."""

    price = _safe_float(ctx.get("price"))
    ema1h = _safe_float(ctx.get("ema200_1h"))
    ema4h = _safe_float(ctx.get("ema200_4h"))
    rsi4h = _safe_float(ctx.get("rsi4h"))
    adx = _safe_float(ctx.get("adx"))

    trend = "Zona mixta/lateral"
    if price is not None and ema1h is not None and ema4h is not None:
        if price < ema1h and price < ema4h:
            trend = "Tendencia bajista"
        elif price > ema1h and price > ema4h:
            trend = "Tendencia alcista"

    long_allowed = ctx.get("long_allowed", True)
    short_allowed = ctx.get("short_allowed", True)

    if (
        not long_allowed
        and price is not None
        and ema1h is not None
        and price < ema1h
    ):
        return f"{trend} • Precio debajo de EMA200 → no LONG."
    if (
        not short_allowed
        and price is not None
        and ema1h is not None
        and price > ema1h
    ):
        return f"{trend} • Precio arriba de EMA200 → no SHORT."

    in_long = bool(ctx.get("in_long"))
    in_short = bool(ctx.get("in_short"))
    if not in_long and not in_short:
        return f"{trend} • Precio fuera del rango de entrada (no tocó bandas)."

    if adx is not None and adx < 18:
        return "Mercado sin fuerza (ADX bajo) • Espero."
    if rsi4h is not None and rsi4h > 70:
        return "Sobrecompra • Evito LONG."
    if rsi4h is not None and rsi4h < 30:
        return "Sobreventa • Evito SHORT."

    if ctx.get("min_notional_ok") is False:
        return "Monto mínimo de Binance insuficiente."
    if ctx.get("step_ok") is False:
        return "Paso/lote mínimo inválido para el par."

    return f"{trend} • Sin señal clara."


def _ensure_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    return datetime.fromtimestamp(datetime.now(tz=timezone.utc).timestamp(), tz=timezone.utc)


def _maybe_localize(ts: datetime, tz_name: str | None) -> datetime:
    if tz_name and ZoneInfo:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:  # pragma: no cover - invalid timezone
            return ts
        if ts.tzinfo is None:
            return ts.replace(tzinfo=tz)
        return ts.astimezone(tz)
    return ts


def render_reasons_simple(
    last_ctx_list: Sequence[Mapping[str, Any]],
    tz_name: str | None = None,
) -> str:
    """Render a human summary for the most recent rejection contexts."""

    lines = ["🕘 Motivos (últimas oportunidades NO abiertas):"]
    for ctx in list(last_ctx_list)[-10:]:
        ctx_map: MutableMapping[str, Any] = dict(ctx)
        ts_raw = ctx_map.get("ts")
        ts_dt = _maybe_localize(_ensure_datetime(ts_raw), tz_name)
        ctx_map["ts"] = ts_dt
        symbol = ctx_map.get("symbol") or "—"
        hhmm = ts_dt.strftime("%H:%M")
        lines.append(f"• {hhmm} — {symbol}: {simplify_reason(ctx_map)}")
    if len(lines) == 1:
        lines.append("(sin datos)")
    return "\n".join(lines)


def render_logs_summary(last_events: Sequence[Mapping[str, Any]]) -> str:
    """Render a concise summary of recent log contexts."""

    lines = ["📄 Últimos logs (resumen):"]
    for event in list(last_events)[-10:]:
        ctx_map: MutableMapping[str, Any] = dict(event)
        ts_dt = _ensure_datetime(ctx_map.get("ts"))
        lines.append(f"{ts_dt.strftime('%Y-%m-%d %H:%M:%S')} — {simplify_reason(ctx_map)}")
    if len(lines) == 1:
        lines.append("(sin eventos recientes)")
    return "\n".join(lines)
