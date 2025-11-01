"""Simple runtime metrics registry.

This module centralises the counters requested for observability.  It
keeps the data in memory and periodically flushes a CSV snapshot to the
``data/metrics`` directory so it can be scraped by external tools.  The
implementation is deliberately lightweight so it works even when the
``prometheus_client`` dependency is not available.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass(slots=True)
class _MetricSnapshot:
    ticks_total: int = 0
    signals_total: int = 0
    orders_sent_total: int = 0
    orders_filled_total: int = 0
    rejections_total: int = 0
    qty_rejected_total: float = 0.0
    sl_triggered_total: int = 0
    tp_triggered_total: int = 0
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    latency_exchange_ms: float = 0.0


class MetricsManager:
    """Thread-safe metrics collector with CSV export support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = _MetricSnapshot()
        self._latency_sum = 0.0
        self._latency_count = 0
        self._latency_last = 0.0
        self._dirty = False
        self._last_export = 0.0
        base_dir = Path(os.getenv("BOT_METRICS_DIR", "data/metrics"))
        base_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = base_dir / "runtime_metrics.csv"
        self._export_interval = float(os.getenv("BOT_METRICS_EXPORT_INTERVAL", "5"))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _mark_dirty(self) -> None:
        self._dirty = True

    def _snapshot_locked(self) -> Dict[str, float | int]:
        data = asdict(self._snapshot)
        latency_avg = 0.0
        if self._latency_count > 0:
            latency_avg = self._latency_sum / float(self._latency_count)
        elif self._latency_last > 0:
            latency_avg = self._latency_last
        data["latency_exchange_ms"] = round(latency_avg, 6)
        return data

    def _write_csv(self, payload: Dict[str, float | int]) -> None:
        header = ["timestamp"] + sorted(payload.keys())
        row = [time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())]
        for key in sorted(payload.keys()):
            row.append(payload[key])

        path = self._csv_path
        write_header = not path.exists()
        try:
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
        except OSError:
            # Best effort: if writing fails we keep the metrics in memory.
            pass

    def _maybe_export(self, *, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not self._dirty and not force:
                return
            if not force and (now - self._last_export) < self._export_interval:
                return
            payload = self._snapshot_locked()
            self._dirty = False
            self._last_export = now
        self._write_csv(payload)

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------
    def tick(self) -> None:
        with self._lock:
            self._snapshot.ticks_total += 1
            self._mark_dirty()
        self._maybe_export()

    def record_signal(self) -> None:
        with self._lock:
            self._snapshot.signals_total += 1
            self._mark_dirty()

    def record_order_sent(self) -> None:
        with self._lock:
            self._snapshot.orders_sent_total += 1
            self._mark_dirty()

    def record_order_filled(self) -> None:
        with self._lock:
            self._snapshot.orders_filled_total += 1
            self._mark_dirty()

    def record_rejection(self, reason: Optional[str] = None, *, qty: float | None = None) -> None:
        with self._lock:
            self._snapshot.rejections_total += 1
            if qty is not None:
                try:
                    qty_val = float(qty)
                except (TypeError, ValueError):
                    qty_val = 0.0
                else:
                    if qty_val > 0:
                        self._snapshot.qty_rejected_total += qty_val
            self._mark_dirty()

    def observe_latency(self, latency_ms: float) -> None:
        try:
            value = float(latency_ms)
        except (TypeError, ValueError):
            return
        if value <= 0:
            return
        with self._lock:
            self._latency_sum += value
            self._latency_count += 1
            self._latency_last = value
            self._mark_dirty()

    def update_unrealized(self, pnl_value: float) -> None:
        try:
            pnl = float(pnl_value)
        except (TypeError, ValueError):
            pnl = 0.0
        with self._lock:
            self._snapshot.pnl_unrealized = pnl
            self._mark_dirty()

    def add_realized(self, pnl_value: float) -> None:
        try:
            pnl = float(pnl_value)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._snapshot.pnl_realized += pnl
            self._mark_dirty()

    def record_sl_triggered(self) -> None:
        with self._lock:
            self._snapshot.sl_triggered_total += 1
            self._mark_dirty()

    def record_tp_triggered(self) -> None:
        with self._lock:
            self._snapshot.tp_triggered_total += 1
            self._mark_dirty()

    def record_close(
        self,
        *,
        summary: Dict[str, object] | None,
        snapshot_before: Dict[str, object] | None = None,
        status_before: Dict[str, object] | None = None,
    ) -> None:
        """Update realised PnL and stop/take-profit counters."""

        summary = summary or {}

        def _first_float(*values: object) -> Optional[float]:
            for candidate in values:
                if candidate in (None, ""):
                    continue
                try:
                    val = float(candidate)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if not (val is None or (val != val)):
                    return val
            return None

        realized = _first_float(summary.get("realized_pnl"), summary.get("pnl_balance_delta"))
        if realized is not None:
            self.add_realized(realized)

        exit_price = _first_float(summary.get("exit_price"))
        entry_price = _first_float(summary.get("entry_price"))
        side = str(summary.get("side") or "").upper() or "LONG"

        sl_candidates = []
        tp_candidates = []
        for container in (summary, snapshot_before or {}, status_before or {}):
            sl_candidates.append(container.get("sl"))
            sl_candidates.append(container.get("stop_loss"))
            tp_candidates.append(container.get("tp"))
            tp_candidates.append(container.get("tp_price"))
        sl_price = _first_float(*sl_candidates)
        tp_price = _first_float(*tp_candidates)

        if exit_price is not None:
            tolerance = 0.001  # 0.1%
            if sl_price is not None and entry_price is not None:
                if side == "LONG" and exit_price <= sl_price * (1.0 + tolerance):
                    self.record_sl_triggered()
                elif side == "SHORT" and exit_price >= sl_price * (1.0 - tolerance):
                    self.record_sl_triggered()
            if tp_price is not None and entry_price is not None:
                if side == "LONG" and exit_price >= tp_price * (1.0 - tolerance):
                    self.record_tp_triggered()
                elif side == "SHORT" and exit_price <= tp_price * (1.0 + tolerance):
                    self.record_tp_triggered()

        self.update_unrealized(0.0)

    def flush(self, *, force: bool = False) -> None:
        self._maybe_export(force=force)


METRICS = MetricsManager()


__all__ = ["METRICS", "MetricsManager"]

