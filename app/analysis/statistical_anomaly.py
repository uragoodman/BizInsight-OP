"""Deterministic, explainable anomaly detection for metric series."""

from __future__ import annotations

from hashlib import sha256
from statistics import median
from typing import Literal, Protocol

from app.analysis.contracts import AnomalyFinding, MetricSeries


class AnomalyOutbox(Protocol):
    def emit(self, event_type: Literal["AnomalyDetected"], payload: dict[str, object], idempotency_key: str, severity: Literal["info", "warning", "critical"] = "info") -> object: ...

    def emit_anomaly(self, payload: dict[str, object], idempotency_key: str, severity: Literal["info", "warning", "critical"] = "warning", cooldown_hours: int = 24) -> object: ...


def _evidence_id(series: MetricSeries, bucket: str) -> str:
    point = next((item for item in series.points if item.bucket.isoformat() == bucket), None)
    raw = f"{series.metric_name}|{bucket}|{getattr(point, 'value', None)}|{series.definition_version}|{','.join(series.source_refs)}"
    return f"ev_{sha256(raw.encode()).hexdigest()[:16]}"


def detect_series_anomalies(
    series: MetricSeries,
    *,
    lookback: int = 14,
    min_samples: int = 7,
    sensitivity: float = 3.5,
    fallback_change_ratio: float = 0.3,
) -> list[AnomalyFinding]:
    """Detect anomalies using rolling median/MAD with an explicit fallback.

    Missing values are skipped, never coerced to zero. Each point only sees prior
    observations, which avoids leaking the point being classified into its baseline.
    """

    findings: list[AnomalyFinding] = []
    history: list[float] = []
    for point in series.points:
        if point.value is None:
            continue
        window = history[-lookback:]
        if len(window) >= min_samples:
            baseline = float(median(window))
            deviations = [abs(value - baseline) for value in window]
            mad = float(median(deviations))
            if mad > 0:
                score = 0.6745 * (point.value - baseline) / mad
                anomalous = abs(score) >= sensitivity
                method: Literal["mad", "iqr", "change_threshold"] = "mad"
                threshold = sensitivity
            else:
                ordered = sorted(window)
                q1 = ordered[len(ordered) // 4]
                q3 = ordered[(len(ordered) * 3) // 4]
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    score = (point.value - baseline) / iqr
                    anomalous = point.value < lower or point.value > upper
                    method = "iqr"
                    threshold = 1.5
                else:
                    ratio = (point.value - baseline) / abs(baseline) if baseline else 0.0
                    score = ratio
                    anomalous = abs(ratio) >= fallback_change_ratio
                    method = "change_threshold"
                    threshold = fallback_change_ratio
            if anomalous:
                duration_days = 1
                if (
                    findings and findings[-1].metric_date.toordinal() + 1 == point.bucket.toordinal()
                    and findings[-1].direction == ("up" if point.value > baseline else "down")
                ):
                    duration_days = findings[-1].duration_days + 1
                findings.append(AnomalyFinding(
                    metric_name=series.metric_name,
                    metric_date=point.bucket,
                    value=point.value,
                    baseline=round(baseline, 4),
                    score=round(score, 4),
                    threshold=threshold,
                    direction="up" if point.value > baseline else "down",
                    method=method,
                    sufficient_data=True,
                    duration_days=duration_days,
                    evidence_ids=[_evidence_id(series, point.bucket.isoformat())],
                ))
        history.append(point.value)
    return findings


def detect_and_emit_anomalies(
    series: MetricSeries,
    outbox: AnomalyOutbox,
    *,
    lookback: int = 14,
    min_samples: int = 7,
    sensitivity: float = 3.5,
    fallback_change_ratio: float = 0.3,
) -> list[AnomalyFinding]:
    """Event-driven boundary: persist deduplicated events for detected facts."""

    from app.analysis.events import anomaly_alert_payload
    from app.core.config import get_settings

    findings = detect_series_anomalies(
        series, lookback=lookback, min_samples=min_samples, sensitivity=sensitivity,
        fallback_change_ratio=fallback_change_ratio,
    )
    for finding in findings:
        key, payload, severity = anomaly_alert_payload(finding)
        outbox.emit_anomaly(payload, key, severity, get_settings().anomaly_cooldown_hours)
    return findings
