from __future__ import annotations

from datetime import datetime

from human_fmt import fmt_ar, fmt_int


def motivo_simplificado(
    side: str,
    price: float,
    anchor: float,
    step: float,
    span: float,
    gate_bps: float | None = None,
    ts_utc: datetime | None = None,
) -> str:
    """Devuelve una línea amigable resumiendo el estado del pullback."""

    ts_txt = fmt_ar(ts_utc) if ts_utc else "--:--"
    side = (side or "").upper()

    if side == "SHORT":
        min_short = anchor + step
        max_short = anchor + span
        if price < min_short:
            faltan = min_short - price
            core = (
                f"SHORT: entrar ≥ {fmt_int(min_short)}; ahora {fmt_int(price)} "
                f"(faltan {fmt_int(faltan)})"
            )
        elif price > max_short:
            exceso = price - max_short
            core = (
                f"SHORT: se pasó (máx {fmt_int(max_short)}); ahora {fmt_int(price)} "
                f"(exceso {fmt_int(exceso)})"
            )
        else:
            extra = f", falta ventaja {gate_bps / 100:.2f}%" if gate_bps else ""
            core = f"SHORT: dentro de rango{extra}"
        rango = f"[rango: {fmt_int(min_short)}–{fmt_int(max_short)}]"

    else:  # LONG
        min_long = anchor - span
        max_long = anchor - step
        if price > max_long:
            faltan = price - max_long
            core = (
                f"LONG: debe caer ≤ {fmt_int(max_long)}; ahora {fmt_int(price)} "
                f"(faltan {fmt_int(faltan)})"
            )
        elif price < min_long:
            exceso = min_long - price
            core = (
                f"LONG: se pasó (mín {fmt_int(min_long)}); ahora {fmt_int(price)} "
                f"(exceso {fmt_int(exceso)})"
            )
        else:
            extra = f", falta ventaja {gate_bps / 100:.2f}%" if gate_bps else ""
            core = f"LONG: dentro de rango{extra}"
        rango = f"[rango: {fmt_int(min_long)}–{fmt_int(max_long)}]"

    return f"{ts_txt} — {core} {rango}"
