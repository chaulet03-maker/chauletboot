from dataclasses import dataclass


@dataclass(frozen=True)
class SideMap:
    """Representa los parámetros de dirección según el modo de trading."""

    # Lado lógico interno: 'LONG'/'SHORT'
    internal: str
    # Para Binance order.side: 'BUY'/'SELL'
    order_side: str
    # Para Binance Futures en hedge: 'LONG'/'SHORT' o None si spot/simple
    position_side: str | None


def normalize_side(internal_side: str, *, futures: bool, hedge_mode: bool) -> SideMap:
    """Normaliza el lado interno a los campos requeridos por Binance/CCXT."""

    s = (internal_side or "").upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError(f"Internal side inválido: {internal_side}")

    if futures:
        if hedge_mode:
            if s == "LONG":
                return SideMap(internal="LONG", order_side="BUY", position_side="LONG")
            return SideMap(internal="SHORT", order_side="SELL", position_side="SHORT")
        return SideMap(internal=s, order_side=("BUY" if s == "LONG" else "SELL"), position_side=None)

    return SideMap(internal=s, order_side=("BUY" if s == "LONG" else "SELL"), position_side=None)
