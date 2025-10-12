from __future__ import annotations

import logging
from typing import Any, Optional

import brokers
from config import S
from paper_store import PaperStore

logger = logging.getLogger(__name__)


class PositionService:
    def __init__(
        self,
        *,
        paper_store: Optional[PaperStore] = None,
        live_client: Optional[Any] = None,
        symbol: str = "BTC/USDT",
    ) -> None:
        self.store = paper_store
        self.client = live_client
        self.symbol = symbol

    # ------------------------------------------------------------------
    def mark_to_market(self, mark: float) -> None:
        if S.PAPER and self.store:
            try:
                self.store.save(mark=float(mark))
            except Exception:
                logger.debug("No se pudo actualizar mark-to-market en store paper.", exc_info=True)

    # ------------------------------------------------------------------
    def _status_paper(self) -> dict[str, Any]:
        if not self.store:
            raise RuntimeError("PaperStore no inicializado en modo paper.")
        st = self.store.load()
        pos_qty = float(st.get("pos_qty", 0.0) or 0.0)
        avg_price = float(st.get("avg_price", 0.0) or 0.0)
        mark = float(st.get("mark", 0.0) or 0.0)
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
        base_equity = float(st.get("equity") or 0.0)
        realized = float(st.get("realized_pnl") or 0.0)
        fees = float(st.get("fees") or 0.0)
        equity = base_equity + realized + unreal - fees
        return {
            "symbol": self.symbol,
            "side": side,
            "entry_price": round(avg_price, 2),
            "qty": abs(pos_qty),
            "pnl": round(unreal, 2),
            "equity": round(equity, 2),
            "mark": round(mark, 2),
        }

    def _status_live(self) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("Cliente live no inicializado.")
        sym = self.symbol.replace("/", "")
        try:
            positions = self.client.futures_position_information(symbol=sym)
        except Exception as exc:  # pragma: no cover - solo en live
            raise RuntimeError(f"No se pudo obtener la posición live: {exc}") from exc

        pos = positions[0] if positions else {}
        entry = float(pos.get("entryPrice") or 0.0)
        amt = float(pos.get("positionAmt") or 0.0)
        side = "LONG" if amt > 0 else "SHORT" if amt < 0 else "FLAT"
        qty = abs(amt)
        mark = float(pos.get("markPrice") or 0.0)
        if mark <= 0:
            try:
                mp = self.client.futures_mark_price(symbol=sym)
                if isinstance(mp, dict):
                    mark = float(mp.get("markPrice") or 0.0)
            except Exception:
                logger.debug("No se pudo obtener markPrice desde endpoint dedicado.", exc_info=True)

        try:
            account = self.client.futures_account()
        except Exception as exc:  # pragma: no cover - live only
            raise RuntimeError(f"No se pudo obtener equity live: {exc}") from exc

        equity = 0.0
        if isinstance(account, dict):
            for key in ("totalWalletBalance", "totalMarginBalance"):
                if account.get(key) is not None:
                    equity = float(account.get(key) or 0.0)
                    break
            else:
                assets = account.get("assets") or []
                for asset in assets:
                    if str(asset.get("asset", "")).upper() == "USDT":
                        equity = float(asset.get("walletBalance") or asset.get("marginBalance") or 0.0)
                        break

        pnl = 0.0
        if qty > 0 and entry > 0 and mark > 0:
            if side == "LONG":
                pnl = (mark - entry) * qty
            elif side == "SHORT":
                pnl = (entry - mark) * qty

        return {
            "symbol": self.symbol,
            "side": side,
            "entry_price": round(entry, 2),
            "qty": qty,
            "pnl": round(pnl, 2),
            "equity": round(equity, 2),
            "mark": round(mark, 2),
        }

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
    symbol: str = "BTC/USDT",
) -> PositionService:
    if settings.PAPER:
        if store is None:
            store = brokers.ACTIVE_PAPER_STORE or brokers._build_paper_store(settings.start_equity)
            brokers.ACTIVE_PAPER_STORE = store
        svc = PositionService(paper_store=store, symbol=symbol)
    else:
        if client is None:
            client = brokers.ACTIVE_LIVE_CLIENT
        svc = PositionService(paper_store=None, live_client=client, symbol=symbol)
    global pos_svc
    pos_svc = svc
    return svc


pos_svc: Optional[PositionService] = None


__all__ = ["PositionService", "build_position_service", "pos_svc"]
