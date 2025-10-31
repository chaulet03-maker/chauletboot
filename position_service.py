from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import brokers
from paper_store import PaperStore
from bot.runtime_state import get_mode as runtime_get_mode


def _runtime_is_paper() -> bool:
    try:
        return (runtime_get_mode() or "paper").lower() not in {"real", "live"}
    except Exception:
        return True

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
    """Separa la posición total en porciones BOT vs manual y calcula PnL estimado.

    La implementación anterior asumía que las posiciones manuales siempre iban en la
    misma dirección que la posición total reportada por la cuenta. Ese supuesto
    dejaba de ser cierto cuando coexistían posiciones del bot y manuales con lados
    opuestos (por ejemplo, el bot LONG y una cobertura manual SHORT). Como la API de
    Binance reporta cantidades con signo, usamos ese dato para obtener la porción
    manual como una diferencia firmada en vez de limitarse a restar magnitudes.
    Esto permite distinguir correctamente las posiciones manuales aunque el neto de
    la cuenta esté del lado opuesto o incluso se acerque a cero.
    """

    total_qty, total_side, total_signed = _extract_qty_and_side(acct_pos)
    bot_qty, bot_side, bot_signed = _extract_qty_and_side(bot_pos)

    if bot_side == "FLAT" and abs(bot_qty) <= EPS_QTY:
        bot_qty = 0.0
        bot_signed = 0.0
        bot_side = "FLAT"

    if total_side == "FLAT" and abs(total_qty) <= EPS_QTY:
        total_qty = 0.0
        total_signed = 0.0
        total_side = "FLAT"

    total_entry = _extract_price(acct_pos, "entry_price", "entryPrice", "avgPrice")
    bot_entry = _extract_price(bot_pos, "entry_price", "entryPrice", "avgPrice")

    manual_signed = total_signed - bot_signed
    manual_qty = abs(manual_signed)
    if manual_qty <= EPS_QTY:
        manual_qty = 0.0
        manual_signed = 0.0

    manual_side = "FLAT"
    if manual_signed > EPS_QTY:
        manual_side = "LONG"
    elif manual_signed < -EPS_QTY:
        manual_side = "SHORT"

    manual_entry = 0.0
    if manual_side != "FLAT":
        try:
            total_notional = float(total_entry or 0.0) * total_signed
            bot_notional = float(bot_entry or 0.0) * bot_signed
            manual_notional = total_notional - bot_notional
            manual_entry = manual_notional / manual_signed if manual_signed else 0.0
        except Exception:
            manual_entry = 0.0
        if manual_entry <= 0:
            manual_entry = 0.0

    mark_val = float(mark or 0.0)

    total_side_output = "FLAT"
    if total_signed > EPS_QTY:
        total_side_output = "LONG"
    elif total_signed < -EPS_QTY:
        total_side_output = "SHORT"

    bot_side_output = "FLAT"
    if bot_signed > EPS_QTY:
        bot_side_output = "LONG"
    elif bot_signed < -EPS_QTY:
        bot_side_output = "SHORT"

    return {
        "total": {
            "side": total_side_output,
            "qty": round(abs(total_signed), 6),
            "entry_price": round(total_entry, 6) if total_entry else 0.0,
            "pnl": _pnl(total_side_output, abs(total_signed), total_entry, mark_val),
        },
        "bot": {
            "side": bot_side_output,
            "qty": round(abs(bot_signed), 6),
            "entry_price": round(bot_entry, 6) if bot_entry else 0.0,
            "pnl": _pnl(bot_side_output, abs(bot_signed), bot_entry, mark_val),
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
        if not (_runtime_is_paper() and self.store):
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

        symbol_no_slash = self.symbol.replace("/", "")

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
        if not (_runtime_is_paper() and self.store):
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

    def _fetch_live_position(self) -> Optional[Dict[str, Any]]:
        client = self.client
        if client is None:
            return None

        symbol_no_slash = self.symbol.replace("/", "")
        payload: Any = None

        if hasattr(client, "futures_position_information"):
            try:
                payload = client.futures_position_information(symbol=symbol_no_slash)
            except Exception:
                payload = None

        if not payload:
            if hasattr(client, "fapiPrivate_get_positionrisk"):
                try:
                    payload = client.fapiPrivate_get_positionrisk({"symbol": symbol_no_slash})
                except Exception:
                    payload = None
            elif hasattr(client, "fapiPrivateGetPositionRisk"):
                try:
                    payload = client.fapiPrivateGetPositionRisk({"symbol": symbol_no_slash})
                except Exception:
                    payload = None

        entries: list[Dict[str, Any]] = []
        if isinstance(payload, list):
            entries = [entry for entry in payload if isinstance(entry, dict)]
        elif isinstance(payload, dict):
            entries = [payload]

        for entry in entries:
            entry_symbol = str(entry.get("symbol") or entry.get("symbolName") or "").upper()
            if entry_symbol != symbol_no_slash.upper():
                continue
            try:
                raw_amount = (
                    entry.get("positionAmt")
                    or entry.get("position_amt")
                    or entry.get("amount")
                    or entry.get("qty")
                    or 0.0
                )
                amount = float(raw_amount)
            except Exception:
                amount = 0.0

            if abs(amount) <= EPS_QTY:
                continue

            side = "LONG" if amount > 0 else "SHORT"

            try:
                entry_price = float(entry.get("entryPrice") or entry.get("avgPrice") or 0.0)
            except Exception:
                entry_price = 0.0

            try:
                mark_price = float(
                    entry.get("markPrice")
                    or entry.get("mark_price")
                    or entry.get("entryPrice")
                    or entry.get("avgPrice")
                    or 0.0
                )
            except Exception:
                mark_price = 0.0

            return {
                "side": side,
                "qty": abs(amount),
                "entry_price": entry_price,
                "mark_price": mark_price,
            }

        return None

    def _status_live(self) -> dict[str, Any]:
        client = self.client
        symbol_no_slash = self.symbol.replace("/", "")

        mark = 0.0
        if client and hasattr(client, "futures_mark_price"):
            try:
                mp = client.futures_mark_price(symbol=symbol_no_slash)
                if isinstance(mp, dict):
                    mark = float(mp.get("markPrice") or 0.0)
            except Exception:
                logger.debug("No se pudo obtener markPrice privado.", exc_info=True)

        live_position = self._fetch_live_position()

        pos_qty = 0.0
        avg_price = 0.0
        side = "FLAT"

        if live_position:
            side = str(live_position.get("side") or "FLAT").upper()
            qty_live = float(live_position.get("qty") or 0.0)
            avg_price = float(live_position.get("entry_price") or 0.0)
            mark_live = float(live_position.get("mark_price") or 0.0)
            if mark <= 0 and mark_live > 0:
                mark = mark_live
            if side == "SHORT":
                pos_qty = -qty_live
            else:
                pos_qty = qty_live

        if not live_position and self.store:
            try:
                state = self.store.load()
                pos_qty = float(state.get("pos_qty") or 0.0)
                avg_price = float(state.get("avg_price") or 0.0)
                side = "LONG" if pos_qty > 0 else "SHORT" if pos_qty < 0 else "FLAT"
            except Exception:
                logger.debug("No se pudo leer store live", exc_info=True)

        if mark <= 0:
            public_mark = self._fetch_public_mark()
            if public_mark is not None:
                mark = float(public_mark)

        equity = 0.0
        try:
            equity = fetch_live_equity_usdm()
        except Exception:
            logger.debug("No se pudo obtener equity live.", exc_info=True)

        pnl = 0.0
        qty = abs(pos_qty)
        if qty > 0 and avg_price > 0 and mark > 0:
            if side == "SHORT":
                pnl = (avg_price - mark) * qty
            else:
                pnl = (mark - avg_price) * qty

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
        if _runtime_is_paper():
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
