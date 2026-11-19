"""Alert Agent consumes verified anomaly facts and emits deduplicated events."""

from __future__ import annotations

from app.analysis.contracts import AlertEvent, AnomalyFinding
from app.analysis.events import EventOutbox, anomaly_alert_payload
from app.analysis.contracts import PeriodSpec
from app.analysis.periods import query_metric_series
from app.analysis.statistical_anomaly import detect_and_emit_anomalies
from app.core.config import get_settings


class AlertAgent:
    def __init__(self, outbox: EventOutbox) -> None:
        self.outbox = outbox

    def handle(self, finding: AnomalyFinding) -> tuple[str, AlertEvent]:
        key, payload, severity = anomaly_alert_payload(finding)
        direction = "上升" if finding.direction == "up" else "下降"
        summary = (
            f"{finding.metric_name} 于 {finding.metric_date} {direction}至 {finding.value:.2f}，"
            f"基线 {finding.baseline:.2f}，{finding.method} score={finding.score:.2f}。"
        )
        event = self.outbox.emit_anomaly(
            payload | {"summary": summary}, key, severity, get_settings().anomaly_cooldown_hours,
        )
        return summary, event


class MetricWindowAgent:
    """Consume a closed-window event and produce verified anomaly events."""

    def __init__(self, outbox: EventOutbox) -> None:
        self.outbox = outbox

    def handle(self, event: AlertEvent) -> int:
        if event.event_type != "MetricWindowClosed":
            raise ValueError("MetricWindowAgent only accepts MetricWindowClosed events")
        period = PeriodSpec.model_validate({
            "start_date": event.payload["start_date"], "end_date": event.payload["end_date"], "granularity": "day",
        })
        metrics = event.payload.get("metrics", [])
        if not isinstance(metrics, list):
            raise ValueError("MetricWindowClosed.metrics must be a list")
        return sum(
            len(detect_and_emit_anomalies(query_metric_series(str(metric), period), self.outbox))
            for metric in metrics
        )
