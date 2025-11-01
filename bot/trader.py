import asyncio
import logging
import math
from typing import Any, Dict, Optional

from config import S
import trading
from bot.paper_store import get_equity as paper_get_equity, set_equity as paper_set_equity
from position_service import fetch_live_equity_usdm
from bot.runtime_state import get_mode as runtime_get_mode


def _runtime_is_paper() -> bool:
    try:
        return (runtime_get_mode() or "paper").lower() not in {"real", "live"}
    except Exception:
        return getattr(S, "PAPER", True)


class Trader:
    def __init__(self, cfg):
        self.config = cfg
        trading.ensure_initialized()
        default_balance = float(self.config.get('balance', S.start_equity))
        if _runtime_is_paper():
            try:
                default_balance = float(paper_get_equity())
            except Exception:
                try:
                    default_balance = float(getattr(trading.BROKER, "equity", default_balance))
                except Exception:
                    pass
        self._balance = default_balance
        self._open_position: Optional[Dict[str, Any]] = None
        self._last_mode = "paper" if _runtime_is_paper() else "real"

    def reset_caches(self):
        """Limpiar caches de balance/posición al cambiar de modo."""
        base = float(self.config.get('balance', S.start_equity))
        if _runtime_is_paper():
            try:
                base = float(paper_get_equity())
            except Exception:
                pass
        self._balance = base
        self._open_position = None
        self._last_mode = "paper" if _runtime_is_paper() else "real"

    def set_paper_equity(self, value: float) -> None:
        try:
            val = float(value)
        except Exception:
            raise ValueError("Equity inválido") from None

        if _runtime_is_paper():
            try:
                paper_set_equity(val)
            except Exception:
                logging.debug("No se pudo persistir equity en bot.paper_store.", exc_info=True)
        self._balance = val

    def _ensure_mode_consistency(self):
        curr = "paper" if _runtime_is_paper() else "real"
        if curr != getattr(self, "_last_mode", curr):
            self.reset_caches()

    async def get_balance(self, exchange=None) -> float:
        """Devuelve el balance actual de la cuenta."""
        self._ensure_mode_consistency()
        if _runtime_is_paper():
            try:
                self._balance = float(self.equity())
                return self._balance
            except Exception:
                logging.debug("No se pudo refrescar equity en modo paper.", exc_info=True)

        try:
            live_equity = await asyncio.to_thread(fetch_live_equity_usdm)
            self._balance = float(live_equity)
            return self._balance
        except Exception:
            logging.debug("No se pudo obtener equity live vía CCXT.", exc_info=True)

        if exchange and getattr(exchange, 'client', None):
            try:
                balance = await asyncio.to_thread(exchange.client.fetch_balance)
                usdt_info = balance.get('USDT') if isinstance(balance, dict) else None
                if isinstance(usdt_info, dict):
                    total = usdt_info.get('total') or usdt_info.get('free')
                    if total is not None:
                        self._balance = float(total)
            except Exception as exc:
                logging.warning("No se pudo actualizar el balance desde el exchange: %s", exc)
        return self._balance

    def equity(self, force_refresh: bool = False) -> float:
        """Devuelve el equity actual (USDT) usando las fuentes disponibles."""
        self._ensure_mode_consistency()
        trading.ensure_initialized()

        if not force_refresh and math.isfinite(self._balance):
            cached = float(self._balance)
        else:
            cached = 0.0

        equity: Optional[float] = None

        service = getattr(trading, "POSITION_SERVICE", None)
        if service is not None and equity is None:
            try:
                status = service.get_status() or {}
                raw_equity = status.get("equity")
                if raw_equity is not None:
                    equity = float(raw_equity)
            except Exception:
                logging.debug("No se pudo obtener equity desde PositionService.", exc_info=True)

        if _runtime_is_paper() and equity is None:
            try:
                equity = float(paper_get_equity())
            except Exception:
                logging.debug("No se pudo leer equity desde bot.paper_store.", exc_info=True)

        if equity is None:
            equity = cached

        if not math.isfinite(equity):
            equity = 0.0

        self._balance = float(equity)
        return self._balance

    async def check_open_position(self, exchange=None) -> Optional[Dict[str, Any]]:
        """Devuelve la posición abierta (si la hay) y cachea el resultado."""
        self._ensure_mode_consistency()

        if self._open_position:
            return self._open_position

        symbol_cfg = self.config.get("symbol", "BTC/USDT")
        service = getattr(trading, "POSITION_SERVICE", None)

        if service is not None:
            position_state: Optional[Dict[str, Any]] = None

            try:
                fetch_method = getattr(service, "current_position", None)
                if callable(fetch_method):
                    position_state = fetch_method(symbol_cfg)
            except Exception as exc:
                logging.debug("PositionService.current_position fallo: %s", exc)

            if position_state is None:
                try:
                    status = service.get_status()
                    side_status = (status.get("side") or "FLAT").upper()
                    qty_status = float(status.get("qty") or status.get("size") or 0.0)
                    if side_status != "FLAT" and qty_status > 0:
                        position_state = {
                            "symbol": status.get("symbol", symbol_cfg),
                            "side": side_status,
                            "qty": qty_status,
                            "entry_price": float(status.get("entry_price") or 0.0),
                            "mark_price": float(status.get("mark") or 0.0),
                        }
                except Exception as exc:
                    logging.debug("PositionService.get_status fallback fallo: %s", exc)

            if position_state is not None:
                side = (position_state.get("side") or "FLAT").upper()
                try:
                    qty_val = float(
                        position_state.get("qty")
                        or position_state.get("contracts")
                        or position_state.get("size")
                        or 0.0
                    )
                except Exception:
                    qty_val = 0.0

                if side != "FLAT" and qty_val > 0:
                    entry_px = position_state.get("entry_price") or position_state.get("entryPrice") or 0.0
                    mark_px = (
                        position_state.get("mark_price")
                        or position_state.get("markPrice")
                        or position_state.get("mark")
                        or 0.0
                    )
                    self._open_position = {
                        "symbol": position_state.get("symbol", symbol_cfg),
                        "side": side,
                        "contracts": float(qty_val),
                        "entryPrice": float(entry_px or 0.0),
                        "markPrice": float(mark_px or 0.0),
                    }
                    return self._open_position

                self._open_position = None
                return self._open_position

        # IMPORTANTE:
        # La posición del BOT vive en el store (self._open_position).
        # La posición TOTAL de la cuenta se consulta por fuera sólo para reportes/reconciliación.
        return self._open_position

    async def set_position(self, position_data: Optional[Dict[str, Any]]) -> None:
        """Actualiza el estado de la posición almacenada."""
        self._open_position = position_data
        logging.info("Nuevo estado de posición: %s", position_data)
