from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

Side = str  # 'long' | 'short'

@dataclass
class Limits:
    max_total_positions: int
    max_per_symbol: int
    no_hedge: bool

@dataclass
class Caps:
    max_portfolio_leverage: float = 8.0
    max_portfolio_margin_pct: float = 1.0
    max_cluster_side_exposure_pct: float = 60.0
    clusters: Optional[Dict[str, str]] = None

# --- helpers internos -------------------------------------------------

def _caps_from_any(caps_in: Any) -> Caps:
    """Permite pasar dict o Caps sin romper."""
    if isinstance(caps_in, Caps):
        return caps_in
    if isinstance(caps_in, dict):
        return Caps(
            max_portfolio_leverage=float(caps_in.get("max_portfolio_leverage", 8.0)),
            max_portfolio_margin_pct=float(caps_in.get("max_portfolio_margin_pct", 1.0)),
            max_cluster_side_exposure_pct=float(caps_in.get("max_cluster_side_exposure_pct", 60.0)),
            clusters=caps_in.get("clusters"),
        )
    return Caps()

def _norm_side(side: str) -> Side:
    s = (side or "").strip().lower()
    return s if s in ("long", "short") else "long"

def _positions_total_count(all_positions: Dict[str, List[dict]]) -> int:
    return sum(len(v) for v in (all_positions or {}).values())

def _price(sym: str, price_by_symbol: Dict[str, float]) -> Optional[float]:
    p = price_by_symbol.get(sym)
    try:
        return float(p) if p is not None else None
    except Exception:
        return None

def _cluster_of(sym: str, clusters: Optional[Dict[str, str]]) -> str:
    return (clusters or {}).get(sym, "UNCLUSTERED")

# --- reglas por símbolo -----------------------------------------------

def can_open(symbol: str, side: str, all_positions: Dict[str, List[dict]], limits: Limits) -> Tuple[bool, str]:
    """
    - Máx. total de posiciones
    - Máx. por símbolo
    - No-hedge en el mismo símbolo
    """
    side = _norm_side(side)
    total = _positions_total_count(all_positions)
    per_sym = len((all_positions or {}).get(symbol, []))

    if total >= limits.max_total_positions:
        return False, "REJECT_MAX_TOTAL"
    if per_sym >= limits.max_per_symbol:
        return False, "REJECT_MAX_PER_SYMBOL"

    if limits.no_hedge:
        for lot in (all_positions or {}).get(symbol, []):
            lot_side = _norm_side(lot.get("side", ""))
            if lot_side != side:
                return False, "REJECT_NO_HEDGE"

    return True, "OK"

# --- reglas de cartera -------------------------------------------------

def portfolio_caps_ok(
    equity: float,
    positions: Dict[str, List[dict]],
    price_by_symbol: Dict[str, float],
    caps: Any  # acepta Caps o dict
) -> Tuple[bool, str]:
    """
    positions: { symbol: [ { side, qty, lev?, ... }, ... ] }
    price_by_symbol: { symbol: last_price }
    caps: Caps | dict
    """
    caps = _caps_from_any(caps)

    equity = float(equity or 0.0)
    if equity <= 0:
        return False, "REJECT_NO_EQUITY"

    # apalancamiento/margen
    notional_total = 0.0
    margin_total = 0.0
    for sym, lots in (positions or {}).items():
        p = _price(sym, price_by_symbol)
        if p is None or p <= 0:
            continue
        for L in (lots or []):
            try:
                qty = abs(float(L.get("qty", 0.0)))
                if qty <= 0:
                    continue
                notional = qty * p
                notional_total += notional
                lev = max(int(L.get("lev", 1)), 1)
                margin_total += notional / lev
            except Exception:
                continue

    lev_port = (notional_total / equity) if equity else 0.0
    margin_pct = (margin_total / equity) if equity else 0.0

    if lev_port > float(caps.max_portfolio_leverage):
        return False, "REJECT_PORTFOLIO_LEVERAGE"
    if margin_pct > float(caps.max_portfolio_margin_pct):
        return False, "REJECT_PORTFOLIO_MARGIN"

    # exposición por clúster y lado
    max_cluster_pct = float(caps.max_cluster_side_exposure_pct or 0.0)
    if max_cluster_pct > 0:
        expo: Dict[Tuple[str, Side], float] = {}
        for sym, lots in (positions or {}).items():
            p = _price(sym, price_by_symbol)
            if p is None or p <= 0:
                continue
            cluster = _cluster_of(sym, caps.clusters)
            for L in (lots or []):
                side = _norm_side(L.get("side", ""))
                try:
                    qty = abs(float(L.get("qty", 0.0)))
                except Exception:
                    qty = 0.0
                if qty <= 0:
                    continue
                expo[(cluster, side)] = expo.get((cluster, side), 0.0) + qty * p

        for (cluster, side), notional in expo.items():
            pct_equity = (notional / equity) * 100.0
            if pct_equity > max_cluster_pct:
                return False, f"REJECT_CLUSTER_SIDE_EXPOSURE:{cluster}:{side}:{pct_equity:.1f}%>{max_cluster_pct:.1f}%"

    return True, "OK"
