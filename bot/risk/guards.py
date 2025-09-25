from dataclasses import dataclass

@dataclass
class Limits:
    max_total_positions: int
    max_per_symbol: int
    no_hedge: bool

def can_open(symbol, side, all_positions, limits: Limits):
    total = sum(len(v) for v in all_positions.values())
    per_sym = len(all_positions.get(symbol, []))
    if total >= limits.max_total_positions:
        return False, "REJECT_MAX_TOTAL"
    if per_sym >= limits.max_per_symbol:
        return False, "REJECT_MAX_PER_SYMBOL"
    if limits.no_hedge:
        for lot in all_positions.get(symbol, []):
            if lot['side'] != side:
                return False, "REJECT_NO_HEDGE"
    return True, ""

def portfolio_caps_ok(equity, positions, price_by_symbol, caps: dict):
    notional_total = 0.0
    margin_total = 0.0
    for sym, lots in positions.items():
        p = price_by_symbol.get(sym)
        if p is None: continue
        for L in lots:
            notional = abs(L['qty']) * p
            notional_total += notional
            lev = max(int(L.get('lev',1)), 1)
            margin_total += notional / lev
    lev_port = (notional_total / equity) if equity else 0.0
    margin_pct = (margin_total / equity) if equity else 0.0

    if 'max_portfolio_leverage' in caps and lev_port > float(caps['max_portfolio_leverage']):
        return False, "REJECT_MAX_PORTF_LEV"
    if 'max_portfolio_margin_pct' in caps and margin_pct > float(caps['max_portfolio_margin_pct']):
        return False, "REJECT_MAX_PORTF_MARGIN"
    return True, ""
