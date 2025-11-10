"""Command helpers for the trading engine.

This module keeps the order sizing helpers used by the trading
engine commands.  The project originally shipped these utilities in a
private method called ``_compute_order_size`` which, according to the
product requirements, must apply the configured leverage as a direct
multiplier of the capital that will be deployed.  The regression that
motivated this patch skipped the leverage factor, effectively assuming
``1x`` leverage and resulting in significantly smaller positions.

The helpers defined here focus on making the logic easy to unit test
while still providing sensible defaults when optional configuration
values are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Optional

from config import RAW_CONFIG, S

# -- Helpers -----------------------------------------------------------------


def _parse_fraction(value: Any) -> Optional[float]:
    """Return ``value`` as a decimal fraction if possible.

    The helper accepts numbers (``0.25`` or ``25``) and strings (``"25%"``,
    ``"0.25"``) and always returns a decimal fraction – e.g. ``0.25`` for 25%.
    Invalid or empty values return ``None`` so that the caller can choose a
    sensible fallback.
    """

    if value in (None, ""):
        return None

    candidate: float
    try:
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "").replace(",", ".")
            if cleaned == "":
                return None
            candidate = float(cleaned)
        else:
            candidate = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(candidate) or candidate <= 0:
        return None

    return candidate / 100.0 if candidate > 1.0 else candidate


def _default_order_fraction() -> float:
    """Resolve the default risk fraction for order sizing.

    The function looks for two optional configuration sources:

    * ``S.ORDER_SIZING.DEFAULT_PCT`` – when running with a settings object
      that exposes nested order sizing defaults (not available in every
      environment of this code base).
    * ``RAW_CONFIG['order_sizing']['default_pct']`` – the raw configuration
      dictionary loaded from ``config.yaml``.

    When neither of the above is available we fall back to ``0.02`` (2%).
    """

    fallbacks = [0.02]

    order_sizing_cfg: Dict[str, Any] = {}
    try:
        raw_order_sizing = RAW_CONFIG.get("order_sizing")
        if isinstance(raw_order_sizing, dict):
            order_sizing_cfg = raw_order_sizing
    except Exception:
        order_sizing_cfg = {}

    # Try the settings object first
    settings_order_sizing = getattr(S, "ORDER_SIZING", None)
    if settings_order_sizing is not None:
        default_pct = getattr(settings_order_sizing, "DEFAULT_PCT", None)
        frac = _parse_fraction(default_pct)
        if frac is not None:
            return frac

    # Then try the raw configuration
    raw_default = order_sizing_cfg.get("default_pct") if order_sizing_cfg else None
    frac = _parse_fraction(raw_default)
    if frac is not None:
        return frac

    # Finally return the first fallback
    return fallbacks[0]


# -- Order sizing -------------------------------------------------------------


@dataclass
class OrderSize:
    """Result of the order sizing routine."""

    qty: float
    notional: float
    capital_usdt: float
    leverage: float
    risk_fraction: float


class OrderSizingCommands:
    """Utility helpers that emulate the behaviour of the trading commands."""

    def __init__(self, leverage_resolver: Optional[Callable[[str], float]] = None):
        self._leverage_resolver = leverage_resolver

    def leverage(self, symbol: str) -> float:
        if self._leverage_resolver is None:
            return 1.0
        try:
            value = self._leverage_resolver(symbol)
            return float(value) if value is not None else 1.0
        except Exception:
            return 1.0

    def _compute_order_size(
        self,
        symbol: str,
        price: float,
        equity: float,
        risk_pct_val: Any = None,
    ) -> OrderSize:
        """Compute the order size for ``symbol``.

        The important bit of this routine is that leverage is applied as a
        multiplier over the capital being risked so that the notional exposure
        respects the configured leverage.
        """

        try:
            eq = float(equity)
        except (TypeError, ValueError):
            eq = 0.0

        try:
            price_val = float(price)
        except (TypeError, ValueError):
            price_val = 0.0

        leverage = self.leverage(symbol)
        if leverage <= 0:
            leverage = 1.0

        risk_pct_frac = _parse_fraction(risk_pct_val)
        if risk_pct_frac is None:
            risk_pct_frac = _default_order_fraction()

        capital_usdt = eq * risk_pct_frac
        notional = capital_usdt * leverage
        qty = notional / price_val if price_val > 0 else 0.0

        return OrderSize(
            qty=max(qty, 0.0),
            notional=max(notional, 0.0),
            capital_usdt=max(capital_usdt, 0.0),
            leverage=float(leverage),
            risk_fraction=float(risk_pct_frac),
        )


__all__ = [
    "OrderSize",
    "OrderSizingCommands",
    "_compute_order_size",
    "_default_order_fraction",
    "_parse_fraction",
]


# Backwards compatibility: expose a module level helper mirroring the
# original private function so tests can import it directly.

def _compute_order_size(
    symbol: str,
    price: float,
    equity: float,
    risk_pct_val: Any = None,
    leverage_resolver: Optional[Callable[[str], float]] = None,
) -> OrderSize:
    command = OrderSizingCommands(leverage_resolver=leverage_resolver)
    return command._compute_order_size(symbol, price, equity, risk_pct_val=risk_pct_val)
