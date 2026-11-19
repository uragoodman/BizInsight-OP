"""Optional headless chart rendering for periodic reports."""

from __future__ import annotations

from hashlib import sha256
from app.analysis.contracts import ChartArtifact, ContributionFinding, MetricSeries


def render_trend_chart(series: MetricSeries, output_dir: str = "data/reports/charts") -> ChartArtifact:
    evidence = [f"ev_series_{sha256(series.model_dump_json().encode()).hexdigest()[:16]}"]
    valid = [(point.bucket, point.value) for point in series.points if point.value is not None]
    if not valid:
        return ChartArtifact(chart_type="trend", status="chart_unavailable", reason="series has no values", evidence_ids=evidence)
    digest = sha256(series.model_dump_json().encode()).hexdigest()
    return ChartArtifact(
        chart_type="trend", metric_name=series.metric_name, unit=series.unit,
        data_points=[{"bucket": str(bucket), "value": value} for bucket, value in valid],
        content_hash=digest, status="ok", evidence_ids=evidence,
    )


def render_contribution_chart(finding: ContributionFinding, output_dir: str = "data/reports/charts") -> ChartArtifact:
    visible = finding.items[:10]
    if not visible:
        return ChartArtifact(chart_type="contribution", status="chart_unavailable", reason="contribution has no items", evidence_ids=finding.evidence_ids)
    digest = sha256(finding.model_dump_json().encode()).hexdigest()
    return ChartArtifact(
        chart_type="contribution", metric_name=finding.metric_name,
        data_points=[{"dimension": item.dimension_value, "value": item.delta_value} for item in visible],
        content_hash=digest, status="ok", evidence_ids=finding.evidence_ids,
    )
