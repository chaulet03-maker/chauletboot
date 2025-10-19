from __future__ import annotations

import logging
import time
from typing import Any, Optional

import brokers
from config import S
from paper_store import PaperStore

logger = logging.getLogger(__name__)


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

    def _fetch_live_equity(self, client: Any) -> float:
        try:
            if hasattr(client, "fapiPrivateGetAccount"):
                account = client.fapiPrivateGetAccount()
            elif hasattr(client, "fapiPrivate_get_account"):
                account = client.fapiPrivate_get_account()
            else:
                account = client.futures_account()
        except Exception as exc:  # pragma: no cover - live only
            raise RuntimeError(f"No se pudo obtener equity live: {exc}") from exc

        equity = 0.0
        if isinstance(account, dict):
            for key in ("totalWalletBalance", "totalMarginBalance", "equity"):
                if account.get(key) is not None:
                    try:
                        equity = float(account.get(key) or 0.0)
                        break
                    except Exception:
                        continue
            else:
                assets = account.get("assets") or []
                for asset in assets:
                    if str(asset.get("asset", "")).upper() == "USDT":
                        for k in ("walletBalance", "marginBalance", "balance"):
                            if asset.get(k) is not None:
                                try:
                                    equity = float(asset.get(k) or 0.0)
                                    break
                                except Exception:
                                    continue
                        break
        return equity

    def _status_live(self) -> dict[str, Any]:
        pos_qty = 0.0
        avg_price = 0.0
        if self.store:
            try:
                state = self.store.load()
                pos_qty = float(state.get("pos_qty") or 0.0)
                avg_price = float(state.get("avg_price") or 0.0)
            except Exception:
                logger.debug("No se pudo leer store live", exc_info=True)

        mark = 0.0
        client = self.client
        symbol_no_slash = self.symbol.replace("/", "")
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

        side = "FLAT"
        if pos_qty > 0:
            side = "LONG"
        elif pos_qty < 0:
            side = "SHORT"

        equity = 0.0
        if client:
            try:
                equity = self._fetch_live_equity(client)
            except Exception:
                logger.debug("No se pudo obtener equity live.", exc_info=True)

        pnl = 0.0
        qty = abs(pos_qty)
        if qty > 0 and avg_price > 0 and mark > 0:
            if side == "LONG":
                pnl = (mark - avg_price) * qty
            else:
                pnl = (avg_price - mark) * qty

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


__all__ = ["PositionService", "build_position_service", "pos_svc"]
