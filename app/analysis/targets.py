"""Versioned target lookup and variance calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import sqlite3
from contextlib import closing
from pathlib import Path

from app.analysis.contracts import PeriodComparison, PeriodSpec
from app.analysis.semantic import get_metric_definition


@dataclass(frozen=True)
class MetricTarget:
    metric_name: str
    period_start: date
    period_end: date
    target_value: float
    version: str
    effective_at: date
    created_by: str
    scope: str = "global"
    seller_id: str | None = None


class TargetStore:
    def __init__(self, path: str = "data/phase6.sqlite") -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS metric_targets (
                    metric_name TEXT NOT NULL, period_start TEXT NOT NULL, period_end TEXT NOT NULL,
                    scope TEXT NOT NULL, target_value REAL NOT NULL, version TEXT NOT NULL,
                    effective_at TEXT NOT NULL, created_by TEXT NOT NULL,
                    UNIQUE(metric_name,period_start,period_end,scope,version)
                )
            """)

    def put(self, target: MetricTarget) -> MetricTarget:
        get_metric_definition(target.metric_name)
        if target.period_end < target.period_start:
            raise ValueError("target period_end must be on or after period_start")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            existing = connection.execute(
                """SELECT target_value,effective_at,created_by FROM metric_targets
                   WHERE metric_name=? AND period_start=? AND period_end=? AND scope=? AND version=?""",
                (target.metric_name, target.period_start.isoformat(), target.period_end.isoformat(), target.scope, target.version),
            ).fetchone()
            if existing is not None:
                if (float(existing[0]), existing[1], existing[2]) == (
                    target.target_value, target.effective_at.isoformat(), target.created_by,
                ):
                    return target
                raise ValueError("Metric target versions are immutable; create a new version")
            connection.execute(
                """INSERT INTO metric_targets
                   (metric_name,period_start,period_end,scope,target_value,version,effective_at,created_by)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (target.metric_name, target.period_start.isoformat(), target.period_end.isoformat(),
                 target.scope, target.target_value, target.version, target.effective_at.isoformat(), target.created_by),
            )
        return target

    def list(self, *, metric_name: str | None = None, seller_id: str | None = None,
             period_start: date | None = None, period_end: date | None = None,
             version: str | None = None, offset: int = 0, limit: int = 20) -> tuple[list[MetricTarget], int]:
        scope = f"seller:{seller_id}" if seller_id else "global"
        clauses = ["scope = ?"]
        params: list[object] = [scope]
        if metric_name: clauses.append("metric_name = ?"); params.append(metric_name)
        if period_start: clauses.append("period_start >= ?"); params.append(period_start.isoformat())
        if period_end: clauses.append("period_end <= ?"); params.append(period_end.isoformat())
        if version: clauses.append("version = ?"); params.append(version)
        where = " AND ".join(clauses)
        with closing(sqlite3.connect(self.path)) as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM metric_targets WHERE {where}", params).fetchone()[0])
            connection.row_factory = sqlite3.Row
            rows = connection.execute(f"SELECT * FROM metric_targets WHERE {where} ORDER BY period_start DESC, effective_at DESC, version DESC LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        return [MetricTarget(metric_name=row["metric_name"], period_start=date.fromisoformat(row["period_start"]), period_end=date.fromisoformat(row["period_end"]), target_value=float(row["target_value"]), version=row["version"], effective_at=date.fromisoformat(row["effective_at"]), created_by=row["created_by"], scope=row["scope"], seller_id=seller_id) for row in rows], total

    def delete(self, *, metric_name: str, period_start: date, period_end: date, version: str, seller_id: str | None = None) -> bool:
        scope = f"seller:{seller_id}" if seller_id else "global"
        with closing(sqlite3.connect(self.path)) as connection, connection:
            result = connection.execute("DELETE FROM metric_targets WHERE metric_name=? AND period_start=? AND period_end=? AND scope=? AND version=?", (metric_name, period_start.isoformat(), period_end.isoformat(), scope, version))
        return result.rowcount > 0

    def get(self, metric_name: str, period: PeriodSpec, scope: str = "global", seller_id: str | None = None) -> MetricTarget | None:
        scope = f"seller:{seller_id}" if seller_id else scope
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """SELECT * FROM metric_targets
                   WHERE metric_name=? AND period_start=? AND period_end=? AND scope=? AND effective_at<=?
                   ORDER BY effective_at DESC, version DESC LIMIT 1""",
                (metric_name, period.start_date.isoformat(), period.end_date.isoformat(), scope, period.end_date.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return MetricTarget(
            metric_name=row["metric_name"], period_start=date.fromisoformat(row["period_start"]),
            period_end=date.fromisoformat(row["period_end"]), target_value=float(row["target_value"]),
            version=row["version"], effective_at=date.fromisoformat(row["effective_at"]),
            created_by=row["created_by"], scope=row["scope"], seller_id=seller_id,
        )


def compare_to_target(
    metric_name: str, actual: float | None, period: PeriodSpec, target: MetricTarget | None,
    *, scope: str = "platform",
) -> PeriodComparison:
    if target is None or actual is None:
        return PeriodComparison(
            comparison_type="target", current_period=period, current_value=actual, baseline_value=None,
            delta_value=None, delta_ratio=None, status="not_found",
        )
    definition = get_metric_definition(metric_name, scope)
    raw_delta = actual - target.target_value
    # Favorable variance is always positive, regardless of metric direction.
    delta = -raw_delta if definition.direction == "lower_is_better" else raw_delta
    ratio = None if target.target_value == 0 else delta * 100 / abs(target.target_value)
    if target.target_value == 0 or actual is None:
        attainment = None
    elif definition.direction == "lower_is_better":
        attainment = target.target_value * 100 / actual if actual != 0 else None
    else:
        attainment = actual * 100 / target.target_value
    return PeriodComparison(
        comparison_type="target", current_period=period, current_value=actual,
        baseline_value=target.target_value, delta_value=round(delta, 2),
        delta_ratio=None if ratio is None else round(ratio, 2),
        attainment_rate=None if attainment is None else round(attainment, 2),
        target_version=target.version, target_effective_at=target.effective_at,
        status="zero_baseline" if target.target_value == 0 else "ok",
    )
