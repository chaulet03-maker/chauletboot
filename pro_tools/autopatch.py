
from __future__ import annotations
import os, logging
from .parity import ensure_parity

log = logging.getLogger(__name__)

def _try_patch_ccxt(parity_obj):
    try:
        if os.environ.get("MODE","paper").lower() != "paper":
            return False
        import ccxt
        # Save originals once
        if not hasattr(ccxt.Exchange, "_pro_orig_create_order"):
            ccxt.Exchange._pro_orig_create_order = ccxt.Exchange.create_order
        if not hasattr(ccxt.Exchange, "_pro_orig_cancel_order"):
            ccxt.Exchange._pro_orig_cancel_order = ccxt.Exchange.cancel_order

        def _create_order_sync(self, symbol, type, side, amount, price=None, params={}):
            return parity_obj.place_order_sync(symbol=symbol, side=side, type=type, qty=amount, price=price, params=params)

        def _cancel_order_sync(self, id, symbol=None, params={}):
            return parity_obj.cancel_order_sync(order_id=id, symbol=symbol, params=params)

        ccxt.Exchange.create_order = _create_order_sync
        ccxt.Exchange.cancel_order = _cancel_order_sync
        log.info("CCXT patched: create_order/cancel_order routed to PARITY (PAPER)")
        return True
    except Exception as e:
        log.warning("CCXT patch skipped: %s", e)
        return False

def autopatch_if_enabled():
    if os.environ.get("PARITY_AUTOPATCH","1") not in ("1","true","yes","on"):
        return False
    p = ensure_parity()
    ok = _try_patch_ccxt(p)
    log.info("Parity ready (paper=%s), ccxt_patched=%s", p.is_paper, ok)
    # Global accessible singleton
    globals()["parity"] = p
    return True
