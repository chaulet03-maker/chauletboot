from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import brokers
from config import S
from paper_store import PaperStore

logger = logging.getLogger(__name__)

EPS_QTY = 1e-9


def fetch_live_equity_usdm() -> float:
    """Obtiene el equity real de la cuenta de Futuros USD-M (USDT)."""

    # Import lazily to avoid circular dependencies during application start-up.
    from bot.exchange_client import get_ccxt

    client = get_ccxt()

    try:
        bal = client.fetch_balance({"type": "future"})
        if isinstance(bal, dict):
            usdt = bal.get("USDT")
            if isinstance(usdt, dict):
                total = usdt.get("total")
                if total is not None:
                    return float(total)
                alt_total = (usdt.get("free", 0.0) or 0.0) + (usdt.get("used", 0.0) or 0.0)
                return float(alt_total)
            info = bal.get("info") or {}
            if isinstance(info, dict):
                twb = info.get("totalWalletBalance")
                if twb is not None:
                    return float(twb)
                assets = info.get("assets") or []
                for asset in assets:
                    if (asset or {}).get("asset") == "USDT":
                        for key in ("walletBalance", "marginBalance", "crossWalletBalance"):
                            value = (asset or {}).get(key)
                            if value is not None:
                                return float(value)
    except Exception:
        logger.debug("fetch_balance(type=future) falló; intento fallback", exc_info=True)

    try:
        if hasattr(client, "fapiPrivateV2GetAccount"):
            acct = client.fapiPrivateV2GetAccount()
        else:
            acct = client.fapiPrivateGetAccount()
        twb = acct.get("totalWalletBalance") if isinstance(acct, dict) else None
        if twb is not None:
            return float(twb)
        assets = acct.get("assets") if isinstance(acct, dict) else []
        for asset in assets or []:
            if (asset or {}).get("asset") == "USDT":
                for key in ("walletBalance", "marginBalance"):
                    value = (asset or {}).get(key)
                    if value is not None:
                        return float(value)
    except Exception:
        logger.debug("fapiPrivateV2GetAccount falló", exc_info=True)

    return 0.0


def _pnl(side: str, qty: float, entry: float, mark: float) -> float:
    if not qty or qty < EPS_QTY:
        return 0.0
    if not entry or entry <= 0 or not mark or mark <= 0:
        return 0.0
    sign = 1 if (side or "").upper() == "LONG" else -1
    try:
        return round((mark - entry) * qty * sign, 2)
    except Exception:
        return 0.0


def _extract_qty_and_side(data: Optional[Dict[str, Any]]) -> tuple[float, str, float]:
    if not isinstance(data, dict):
        return 0.0, "FLAT", 0.0
    raw_qty = data.get("qty") or data.get("contracts") or data.get("positionAmt")
    try:
        signed_qty = float(raw_qty)
    except Exception:
        signed_qty = 0.0
    side_raw = str(data.get("side") or "").upper()
    if not side_raw:
        if signed_qty > EPS_QTY:
            side_raw = "LONG"
        elif signed_qty < -EPS_QTY:
            side_raw = "SHORT"
    qty = abs(signed_qty)
    if qty < EPS_QTY:
        qty = 0.0
        signed_qty = 0.0
        if not side_raw:
            side_raw = "FLAT"
    return qty, side_raw or "FLAT", signed_qty


def _extract_price(data: Optional[Dict[str, Any]], *keys: str) -> float:
    if not isinstance(data, dict):
        return 0.0
    for key in keys:
        value = data.get(key)
        if value in (None, ""):
            continue
        try:
            price = float(value)
        except Exception:
            continue
        if price > 0:
            return price
    return 0.0


def split_total_vs_bot(
    acct_pos: Optional[Dict[str, Any]],
    bot_pos: Optional[Dict[str, Any]],
    mark: float,
) -> Dict[str, Dict[str, float | str]]:
    """Separa la posición total en porciones BOT vs manual y calcula PnL estimado."""

    total_qty, total_side, _ = _extract_qty_and_side(acct_pos)
    bot_qty, bot_side, _ = _extract_qty_and_side(bot_pos)

    if bot_side == "FLAT" and bot_qty <= EPS_QTY:
        bot_qty = 0.0

    if total_side == "FLAT" and total_qty <= EPS_QTY:
        total_qty = 0.0

    total_entry = _extract_price(acct_pos, "entry_price", "entryPrice", "avgPrice")
    bot_entry = _extract_price(bot_pos, "entry_price", "entryPrice", "avgPrice")

    manual_qty = max(total_qty - bot_qty, 0.0)
    if manual_qty < EPS_QTY:
        manual_qty = 0.0

    manual_entry = 0.0
    if manual_qty > EPS_QTY and total_entry > 0 and total_qty > 0:
        try:
            manual_entry = (total_entry * total_qty - bot_entry * bot_qty) / manual_qty
        except ZeroDivisionError:
            manual_entry = 0.0

    manual_side = total_side if manual_qty > EPS_QTY else "FLAT"
    mark_val = float(mark or 0.0)

    return {
        "total": {
            "side": total_side if total_qty > EPS_QTY else "FLAT",
            "qty": round(total_qty, 6),
            "entry_price": round(total_entry, 6) if total_entry else 0.0,
            "pnl": _pnl(total_side, total_qty, total_entry, mark_val),
        },
        "bot": {
            "side": bot_side if bot_qty > EPS_QTY else "FLAT",
            "qty": round(bot_qty, 6),
            "entry_price": round(bot_entry, 6) if bot_entry else 0.0,
            "pnl": _pnl(bot_side, bot_qty, bot_entry, mark_val),
        },
        "manual": {
            "side": manual_side,
            "qty": round(manual_qty, 6),
            "entry_price": round(manual_entry, 6) if manual_entry else 0.0,
            "pnl": _pnl(manual_side, manual_qty, manual_entry, mark_val),
        },
    }


async def reconcile_bot_store_with_account(trader, exchange, symbol: str, mark: float) -> None:
    """Reconcilia el store del BOT con la posición total de la cuenta."""

    if trader is None or exchange is None or not hasattr(exchange, "fetch_positions"):
        return

    try:
        acct_positions = await exchange.fetch_positions(symbol)
    except Exception:
        acct_positions = None

    acct = None
    if isinstance(acct_positions, list):
        acct = acct_positions[0] if acct_positions else None
    elif isinstance(acct_positions, dict):
        acct = acct_positions

    try:
        bot = await trader.check_open_position(exchange=None)
    except Exception:
        bot = None

    total_qty, _, _ = _extract_qty_and_side(acct)
    bot_qty, _, _ = _extract_qty_and_side(bot)

    if total_qty < EPS_QTY and bot_qty > EPS_QTY:
        await trader.set_position(None)
        logger.info(
            "Reconciliación: cuenta FLAT pero store del BOT tenía posición -> limpiado."
        )
        return

    if total_qty + EPS_QTY < bot_qty:
        await trader.set_position(None)
        logger.info(
            "Reconciliación: total < bot_qty (cerraste manual). BOT marcado FLAT."
        )


class PositionService:
    """Centraliza la información de posición tanto en paper como en live."""

    def __init__(
        self,
        *,
        paper_store: Optional[PaperStore] = None,
        live_client: Optional[Any] = None,
        ccxt_client: Optional[Any] = None,
        symbol: str = "BTC/USDT",
    ) -> None:
        self.store = paper_store
        self.client = live_client
        self.public_client = ccxt_client or live_client
        self.symbol = symbol

    # --- NEW: helpers to pull live position from exchange as source of truth ---
    def _fetch_account_position_ccxt(self) -> tuple[float, float] | None:
        """Return (qty, entry_price) using CCXT if available."""
        client = getattr(self, "public_client", None) or getattr(self, "client", None)
        symbol = self.symbol or ""
        if not symbol:
            return None
        try:
            if client and hasattr(client, "fetch_positions"):
                res = client.fetch_positions([symbol]) if symbol else client.fetch_positions()
                seq = res or []
                if isinstance(seq, dict):
                    seq = [seq]
                sym_norm = symbol.replace("/", "").upper()
                for item in seq:
                    info = item.get("info") if isinstance(item, dict) else None
                    item_sym = (item.get("symbol") or (info or {}).get("symbol") or "")
                    item_sym = str(item_sym).replace("/", "").upper()
                    if item_sym == sym_norm:
                        qty = (
                            item.get("contracts")
                            or item.get("positionAmt")
                            or (info or {}).get("positionAmt")
                            or 0
                        )
                        entry = (
                            item.get("entryPrice")
                            or item.get("avgEntryPrice")
                            or (info or {}).get("entryPrice")
                            or 0
                        )
                        return float(qty), float(entry)
        except Exception:
            logger.debug("CCXT fetch_positions fallo.", exc_info=True)
        return None

    def _fetch_account_position_sdk(self) -> tuple[float, float] | None:
        """Return (qty, entry_price) using python-binance futures_account if available."""
        client = getattr(self, "client", None)
        if not client or not hasattr(client, "futures_account"):
            return None
        try:
            acct = client.futures_account()
            positions = acct.get("positions") or []
            sym = (self.symbol or "").replace("/", "").upper()
            if not sym:
                return None
            for p in positions:
                if str(p.get("symbol", "")).upper() == sym:
                    qty = float(p.get("positionAmt") or 0)
                    entry = float(p.get("entryPrice") or 0)
                    return qty, entry
        except Exception:
            logger.debug("SDK futures_account fallo.", exc_info=True)
        return None

    def refresh(self) -> None:
        """Recarga el estado desde el store compartido."""
        if not self.store:
            return
        try:
            _ = self.store.load()
        except Exception:
            raise

    # ------------------------------------------------------------------
    def mark_to_market(self, mark: float) -> None:
        if not (S.PAPER and self.store):
            return
        try:
            self.store.save(mark=float(mark))
        except Exception:
            logger.debug("No se pudo actualizar mark del store paper.", exc_info=True)

    # ------------------------------------------------------------------
    def _fetch_public_mark(self) -> Optional[float]:
        client = self.public_client
        if client is None:
            return None

        symbol_no_slash = (self.symbol or "").replace("/", "")

        # Preferimos el endpoint de premium index si está disponible
        try:
            if hasattr(client, "fapiPublicGetPremiumIndex"):
                data = client.fapiPublicGetPremiumIndex({"symbol": symbol_no_slash})
            elif hasattr(client, "fapiPublic_get_premiumindex"):
                data = client.fapiPublic_get_premiumindex({"symbol": symbol_no_slash})
            else:
                data = None
            if data:
                if isinstance(data, list):
                    for item in data:
                        if str(item.get("symbol", "")).upper() == symbol_no_slash.upper():
                            mark_price = item.get("markPrice") or item.get("markprice")
                            if mark_price is not None:
                                return float(mark_price)
                elif isinstance(data, dict):
                    mark_price = data.get("markPrice") or data.get("markprice")
                    if mark_price is not None:
                        return float(mark_price)
        except Exception:
            logger.debug("No se pudo obtener markPrice desde premium index público.", exc_info=True)

        # Fallback: ticker público
        if hasattr(client, "fetch_ticker"):
            candidates = [self.symbol]
            base_symbol = self.symbol
            if base_symbol.endswith("/USDT") and ":USDT" not in base_symbol:
                candidates.append(f"{base_symbol}:USDT")
            candidates.append(symbol_no_slash)
            seen = set()
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    ticker = client.fetch_ticker(candidate)
                    if ticker and ticker.get("last") is not None:
                        return float(ticker.get("mark", ticker.get("info", {}).get("markPrice", ticker["last"])))
                except Exception:
                    continue

        return None

    def _maybe_refresh_paper_mark(self, state: dict[str, Any]) -> dict[str, Any]:
        if not (S.PAPER and self.store):
            return state
        updated = int(state.get("updated") or 0)
        mark = float(state.get("mark") or 0.0)
        now = int(time.time())
        if mark > 0 and (now - updated) <= 30:
            return state
        price = self._fetch_public_mark()
        if price is None:
            return state
        try:
            self.store.save(mark=float(price))
            return self.store.load()
        except Exception:
            logger.debug("No se pudo refrescar mark del store paper.", exc_info=True)
            return state

    # ------------------------------------------------------------------
    def _status_paper(self) -> dict[str, Any]:
        if not self.store:
            raise RuntimeError("PaperStore no inicializado en modo paper.")

        state = self.store.load()
        state = self._maybe_refresh_paper_mark(state)

        pos_qty = float(state.get("pos_qty", 0.0) or 0.0)
        avg_price = float(state.get("avg_price", 0.0) or 0.0)
        mark = float(state.get("mark", 0.0) or 0.0)
        side = "FLAT"
        if pos_qty > 0:
            side = "LONG"
        elif pos_qty < 0:
            side = "SHORT"

        unreal = 0.0
        if pos_qty != 0.0 and avg_price > 0.0 and mark > 0.0:
            if pos_qty > 0:
                unreal = (mark - avg_price) * abs(pos_qty)
            else:
                unreal = (avg_price - mark) * abs(pos_qty)

        base_equity = float(state.get("equity") or 0.0)
        realized = float(state.get("realized_pnl") or 0.0)
        fees = float(state.get("fees") or 0.0)
        equity = base_equity + realized + unreal - fees

        return {
            "symbol": self.symbol,
            "side": side,
            "entry_price": round(avg_price, 2),
            "qty": float(abs(pos_qty)),
            "pnl": round(unreal, 2),
            "equity": round(equity, 2),
            "mark": round(mark, 2),
        }

    def _status_live(self) -> dict[str, Any]:
        # LIVE: Binance es la verdad. No confiar en store local para qty/entry.
        EPS = EPS_QTY if "EPS_QTY" in globals() else 1e-9

        pos_qty = 0.0
        avg_price = 0.0

        # 1) CCXT -> 2) SDK -> 3) último recurso: store sólo para mostrar
        got = self._fetch_account_position_ccxt()
        if got is None:
            got = self._fetch_account_position_sdk()
        if isinstance(got, tuple):
            try:
                pos_qty, avg_price = float(got[0] or 0.0), float(got[1] or 0.0)
            except Exception:
                pos_qty, avg_price = 0.0, 0.0
        if abs(pos_qty) <= EPS and getattr(self, "store", None):
            try:
                state = self.store.load()
                pos_qty = float(state.get("pos_qty") or 0.0)
                avg_price = float(state.get("avg_price") or 0.0)
            except Exception:
                logger.debug("No se pudo leer store live", exc_info=True)

        # Mark: primero privado, luego público
        mark = 0.0
        client = self.client
        symbol_no_slash = (self.symbol or "").replace("/", "")
        if client and hasattr(client, "futures_mark_price"):
            try:
                mp = client.futures_mark_price(symbol=symbol_no_slash)
                if isinstance(mp, dict):
                    mark = float(mp.get("markPrice") or 0.0)
            except Exception:
                logger.debug("No se pudo obtener markPrice privado.", exc_info=True)
        if mark <= 0:
            public_mark = self._fetch_public_mark()
            if public_mark is not None:
                mark = float(public_mark)

        # Side con épsilon
        side = "FLAT"
        if pos_qty > EPS:
            side = "LONG"
        elif pos_qty < -EPS:
            side = "SHORT"

        # Equity y PnL
        equity = 0.0
        try:
            equity = fetch_live_equity_usdm()
        except Exception:
            logger.debug("No se pudo obtener equity live.", exc_info=True)

        qty = abs(pos_qty)
        pnl = 0.0
        try:
            if side == "LONG":
                pnl = (mark - avg_price) * qty
            elif side == "SHORT":
                pnl = (avg_price - mark) * qty
        except Exception:
            pnl = 0.0

        return {
            "symbol": self.symbol,
            "side": side,
            "entry_price": round(avg_price, 2),
            "qty": float(qty),
            "pnl": round(pnl, 2),
            "equity": round(float(equity or 0.0), 2),
            "mark": round(float(mark or 0.0), 2),
        }

    def apply_fill(
        self, side: str, qty: float, price: float, fee: float = 0.0
    ) -> dict[str, Any] | None:
        """Actualiza el store local con un fill ejecutado."""
        if not self.store:
            return None
        try:
            state = self.store.load()
            pos_qty = float(state.get("pos_qty") or 0.0)
            avg_price = float(state.get("avg_price") or 0.0)
            direction = str(side).upper()
            dq = float(qty) if direction in {"LONG", "BUY"} else -float(qty)
            new_qty = pos_qty + dq
            price_f = float(price or 0.0)
            realized = 0.0

            if pos_qty * dq < 0:
                closing_qty = min(abs(dq), abs(pos_qty))
                if closing_qty > 0:
                    if pos_qty > 0:
                        realized = (price_f - avg_price) * closing_qty
                    else:
                        realized = (avg_price - price_f) * closing_qty
                if abs(new_qty) < 1e-12:
                    avg_new = 0.0
                elif abs(abs(dq) - closing_qty) > 1e-12:
                    avg_new = price_f
                else:
                    avg_new = avg_price
            else:
                total = abs(pos_qty) + abs(dq)
                if total <= 1e-12:
                    avg_new = 0.0
                elif abs(pos_qty) < 1e-12:
                    avg_new = price_f
                else:
                    avg_new = (
                        (abs(pos_qty) * avg_price) + (abs(dq) * price_f)
                    ) / total

            changes: dict[str, Any] = {
                "pos_qty": float(new_qty),
                "avg_price": float(avg_new),
            }
            if fee:
                changes["fees"] = float(state.get("fees") or 0.0) + float(fee)
            if realized:
                changes["realized_pnl"] = float(state.get("realized_pnl") or 0.0) + float(realized)
            logger.info(
                "STORE SAVE: pos_qty=%.6f avg=%.2f",
                changes["pos_qty"],
                changes["avg_price"],
            )
            return self.store.save(**changes)
        except Exception:
            logger.debug("apply_fill fallo", exc_info=True)
            return None

    # ------------------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        if S.PAPER:
            return self._status_paper()
        return self._status_live()


def build_position_service(
    settings,
    *,
    store: Optional[PaperStore] = None,
    client: Optional[Any] = None,
    ccxt_client: Optional[Any] = None,
    symbol: str = "BTC/USDT",
) -> PositionService:
    if settings.PAPER:
        if store is None:
            store = brokers.ACTIVE_PAPER_STORE or brokers._build_paper_store(settings.start_equity)
            brokers.ACTIVE_PAPER_STORE = store
        svc = PositionService(paper_store=store, ccxt_client=ccxt_client, symbol=symbol)
    else:
        if client is None:
            client = brokers.ACTIVE_LIVE_CLIENT
        svc = PositionService(
            paper_store=None,
            live_client=client,
            ccxt_client=ccxt_client,
            symbol=symbol,
        )
    global pos_svc
    pos_svc = svc
    return svc


pos_svc: Optional[PositionService] = None


__all__ = [
    "PositionService",
    "build_position_service",
    "pos_svc",
    "fetch_live_equity_usdm",
]
