"""Dependency-light observability facade; exports OTel when installed."""

from app.observability.tracing import record_event, record_measurement, snapshot_metrics, span

__all__ = ["record_event", "record_measurement", "snapshot_metrics", "span"]
