from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from paths import get_data_dir

try:  # pragma: no cover - opcional
    from prometheus_client import Counter, Gauge, start_http_server
except Exception:  # pragma: no cover - sin prometheus
    Counter = None  # type: ignore
    Gauge = None  # type: ignore


_PROM_SERVER_STARTED = False
_PROM_LOCK = threading.Lock()


def _start_prometheus_if_needed(port: Optional[int]) -> None:
    if Counter is None or start_http_server is None:
        return
    if port is None or port <= 0:
        return
    global _PROM_SERVER_STARTED
    with _PROM_LOCK:
        if _PROM_SERVER_STARTED:
            return
        try:
            start_http_server(int(port))
            _PROM_SERVER_STARTED = True
        except Exception:
            # Si no se puede iniciar el server, dejamos que continúe sin Prometheus
            _PROM_SERVER_STARTED = False


@dataclass
class _PromHandles:
    counters: Dict[str, Counter] = field(default_factory=dict)
    gauges: Dict[str, Gauge] = field(default_factory=dict)


class MetricsRecorder:
    """Acumula métricas de trading y las persiste en CSV / Prometheus."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._csv_path = Path(
            os.getenv("TRADING_METRICS_CSV", Path(get_data_dir()) / "metrics" / "trading_metrics.csv")
        )
        self._persist_interval = max(float(os.getenv("TRADING_METRICS_FLUSH_SECONDS", 60)), 1.0)
        self._last_persist_ts = 0.0

        self.ticks_total = 0
        self.signals_total = 0
        self.orders_sent_total = 0
        self.orders_filled_total = 0
        self.rejections_total = 0
        self.qty_rejected_total = 0.0
        self.sl_triggered_total = 0
        self.tp_triggered_total = 0
        self.pnl_realized = 0.0
        self.pnl_unrealized = 0.0
        self.latency_exchange_ms = 0.0
        self.loop_latency_ms = 0.0

        self._prom: Optional[_PromHandles] = None
        self._prom_last: Dict[str, float] = {}
        self._setup_prometheus()
        self._ensure_csv_header()

    # ------------------------------------------------------------------
    # Prometheus helpers
    # ------------------------------------------------------------------
    def _setup_prometheus(self) -> None:
        if Counter is None or Gauge is None:
            return
        port_env = os.getenv("PROMETHEUS_PORT") or os.getenv("TRADING_PROM_PORT")
        try:
            prom_port = int(port_env) if port_env else None
        except (TypeError, ValueError):
            prom_port = None
        _start_prometheus_if_needed(prom_port)

        counters = {
            "ticks_total": Counter("bot_ticks_total", "Ticks procesados por el motor"),
            "signals_total": Counter("bot_signals_total", "Señales válidas detectadas"),
            "orders_sent_total": Counter("bot_orders_sent_total", "Órdenes enviadas"),
            "orders_filled_total": Counter("bot_orders_filled_total", "Órdenes con fill"),
            "rejections_total": Counter("bot_rejections_total", "Rechazos totales"),
            "qty_rejected_total": Counter("bot_qty_rejected_total", "Cantidad rechazada acumulada"),
            "sl_triggered_total": Counter("bot_sl_triggered_total", "Stops Loss ejecutados"),
            "tp_triggered_total": Counter("bot_tp_triggered_total", "Take Profits ejecutados"),
        }
        gauges = {
            "pnl_realized": Gauge("bot_pnl_realized", "PnL realizado acumulado"),
            "pnl_unrealized": Gauge("bot_pnl_unrealized", "PnL no realizado actual"),
            "latency_exchange_ms": Gauge("bot_latency_exchange_ms", "Latencia de consultas al exchange (ms)"),
            "loop_latency_ms": Gauge("bot_loop_latency_ms", "Duración del ciclo de trading (ms)"),
        }
        self._prom = _PromHandles(counters=counters, gauges=gauges)

    def _update_prometheus_locked(self) -> None:
        if self._prom is None:
            return
        counters = self._prom.counters
        gauges = self._prom.gauges
        current_values = {
            "ticks_total": float(self.ticks_total),
            "signals_total": float(self.signals_total),
            "orders_sent_total": float(self.orders_sent_total),
            "orders_filled_total": float(self.orders_filled_total),
            "rejections_total": float(self.rejections_total),
            "qty_rejected_total": float(self.qty_rejected_total),
            "sl_triggered_total": float(self.sl_triggered_total),
            "tp_triggered_total": float(self.tp_triggered_total),
        }
        for key, counter in counters.items():
            value = current_values[key]
            prev = self._prom_last.get(key, 0.0)
            delta = value - prev
            if delta > 0:
                counter.inc(delta)
            self._prom_last[key] = value

        gauges["pnl_realized"].set(self.pnl_realized)
        gauges["pnl_unrealized"].set(self.pnl_unrealized)
        gauges["latency_exchange_ms"].set(self.latency_exchange_ms)
        gauges["loop_latency_ms"].set(self.loop_latency_ms)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------
    def _ensure_csv_header(self) -> None:
        path = self._csv_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        if path.exists():
            return
        header = [
            "timestamp",
            "ticks_total",
            "signals_total",
            "orders_sent_total",
            "orders_filled_total",
            "rejections_total",
            "qty_rejected_total",
            "sl_triggered_total",
            "tp_triggered_total",
            "pnl_realized",
            "pnl_unrealized",
            "latency_exchange_ms",
            "loop_latency_ms",
        ]
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
        except Exception:
            pass

    def _persist_locked(self) -> None:
        now = time.time()
        if (now - self._last_persist_ts) < self._persist_interval:
            return
        self._last_persist_ts = now
        row = [
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            self.ticks_total,
            self.signals_total,
            self.orders_sent_total,
            self.orders_filled_total,
            self.rejections_total,
            f"{self.qty_rejected_total:.6f}",
            self.sl_triggered_total,
            self.tp_triggered_total,
            f"{self.pnl_realized:.2f}",
            f"{self.pnl_unrealized:.2f}",
            f"{self.latency_exchange_ms:.2f}",
            f"{self.loop_latency_ms:.2f}",
        ]
        try:
            with self._csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(row)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_tick(self, *, latency_ms: Optional[float] = None, exchange_latency_ms: Optional[float] = None) -> None:
        with self._lock:
            self.ticks_total += 1
            if latency_ms is not None:
                self.loop_latency_ms = float(latency_ms)
            if exchange_latency_ms is not None:
                self.latency_exchange_ms = float(exchange_latency_ms)
            self._update_prometheus_locked()
            self._persist_locked()

    def record_signal(self, _side: Optional[str] = None) -> None:
        with self._lock:
            self.signals_total += 1
            self._update_prometheus_locked()

    def record_order_sent(self) -> None:
        with self._lock:
            self.orders_sent_total += 1
            self._update_prometheus_locked()

    def record_order_filled(self, qty: Optional[float] = None) -> None:
        with self._lock:
            self.orders_filled_total += 1
            if qty is not None and qty > 0:
                # qty informativa en Prometheus -> sumamos a gauge aunque sea counter
                self.qty_rejected_total += 0.0  # noop para consistencia
            self._update_prometheus_locked()

    def record_rejection(self, reason: str, qty: Optional[float] = None) -> None:
        with self._lock:
            self.rejections_total += 1
            if qty is not None and qty > 0:
                self.qty_rejected_total += float(qty)
            self._update_prometheus_locked()
            self._persist_locked()

    def record_sl_triggered(self) -> None:
        with self._lock:
            self.sl_triggered_total += 1
            self._update_prometheus_locked()

    def record_tp_triggered(self) -> None:
        with self._lock:
            self.tp_triggered_total += 1
            self._update_prometheus_locked()

    def add_realized(self, value: float) -> None:
        if value is None:
            return
        with self._lock:
            self.pnl_realized += float(value)
            self._update_prometheus_locked()
            self._persist_locked()

    def set_unrealized(self, value: float) -> None:
        if value is None:
            return
        with self._lock:
            self.pnl_unrealized = float(value)
            self._update_prometheus_locked()

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "ticks_total": float(self.ticks_total),
                "signals_total": float(self.signals_total),
                "orders_sent_total": float(self.orders_sent_total),
                "orders_filled_total": float(self.orders_filled_total),
                "rejections_total": float(self.rejections_total),
                "qty_rejected_total": float(self.qty_rejected_total),
                "sl_triggered_total": float(self.sl_triggered_total),
                "tp_triggered_total": float(self.tp_triggered_total),
                "pnl_realized": float(self.pnl_realized),
                "pnl_unrealized": float(self.pnl_unrealized),
                "latency_exchange_ms": float(self.latency_exchange_ms),
                "loop_latency_ms": float(self.loop_latency_ms),
            }


_METRICS: Optional[MetricsRecorder] = None
_METRICS_LOCK = threading.Lock()


def get_metrics() -> MetricsRecorder:
    global _METRICS
    if _METRICS is not None:
        return _METRICS
    with _METRICS_LOCK:
        if _METRICS is None:
            _METRICS = MetricsRecorder()
    return _METRICS


__all__ = ["get_metrics", "MetricsRecorder"]
