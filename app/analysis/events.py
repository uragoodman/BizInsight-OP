"""Durable local outbox for Phase 6 domain events."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Literal, Protocol
from uuid import uuid4

from app.analysis.contracts import AlertEvent, AnomalyFinding

EventType = Literal["MetricWindowClosed", "AnomalyDetected", "ReportDrafted", "ReportApproved", "ReportPublished", "DiagnosisProposed"]
Severity = Literal["info", "warning", "critical"]


class NotificationAdapter(Protocol):
    def deliver(self, event: AlertEvent) -> None: ...


class EventOutbox:
    def __init__(self, path: str = "data/phase6.sqlite") -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS event_outbox (
                    event_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    processed_at TEXT,
                    claimed_at TEXT,
                    last_error TEXT
                )
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(event_outbox)").fetchall()}
            if "claimed_at" not in columns:
                connection.execute("ALTER TABLE event_outbox ADD COLUMN claimed_at TEXT")
            if "last_error" not in columns:
                connection.execute("ALTER TABLE event_outbox ADD COLUMN last_error TEXT")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS diagnosis_attempts (
                    tenant_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    attempted_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, metric_name)
                )
            """)

    def claim_diagnosis_attempt(self, tenant_id: str, metric_name: str, cooldown_hours: int) -> bool:
        """Atomically reserve one diagnosis attempt per tenant/metric cooldown.

        This is policy state, not an outbox event: failed or abstained runs
        must still be throttled without being sent to notification consumers.
        """
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=max(0, cooldown_hours))).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempted_at FROM diagnosis_attempts WHERE tenant_id=? AND metric_name=?",
                (tenant_id, metric_name),
            ).fetchone()
            if cooldown_hours > 0 and row is not None and str(row["attempted_at"]) >= cutoff:
                return False
            connection.execute(
                """INSERT INTO diagnosis_attempts (tenant_id, metric_name, attempted_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(tenant_id, metric_name) DO UPDATE SET attempted_at=excluded.attempted_at""",
                (tenant_id, metric_name, now.isoformat()),
            )
            return True

    def _insert(self, connection: sqlite3.Connection, event: AlertEvent) -> AlertEvent:
        connection.execute(
            """INSERT OR IGNORE INTO event_outbox
               (event_id,idempotency_key,event_type,severity,payload_json,occurred_at)
               VALUES (?,?,?,?,?,?)""",
            (event.event_id, event.idempotency_key, event.event_type, event.severity,
             json.dumps(event.payload, ensure_ascii=False), event.occurred_at.isoformat()),
        )
        row = connection.execute("SELECT * FROM event_outbox WHERE idempotency_key=?", (event.idempotency_key,)).fetchone()
        return self._event_from_row(row)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> AlertEvent:
        return AlertEvent(
            event_id=row["event_id"], idempotency_key=row["idempotency_key"], event_type=row["event_type"],
            severity=row["severity"], payload=json.loads(row["payload_json"]), occurred_at=row["occurred_at"],
        )

    def emit_in_transaction(
        self, connection: sqlite3.Connection, event_type: EventType, payload: dict[str, object],
        idempotency_key: str, severity: Severity = "info",
    ) -> AlertEvent:
        event = AlertEvent(
            event_id=f"evt_{uuid4().hex}", idempotency_key=idempotency_key,
            event_type=event_type, severity=severity, payload=payload,
        )
        return self._insert(connection, event)

    def emit(self, event_type: EventType, payload: dict[str, object], idempotency_key: str, severity: Severity = "info") -> AlertEvent:
        with self._connection() as connection:
            return self.emit_in_transaction(connection, event_type, payload, idempotency_key, severity)

    def emit_anomaly(
        self, payload: dict[str, object], idempotency_key: str, severity: Severity = "warning",
        cooldown_hours: int = 24,
    ) -> AlertEvent:
        """Emit an anomaly once per metric/direction during a configurable cooldown."""

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max(0, cooldown_hours))).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT * FROM event_outbox
                   WHERE event_type='AnomalyDetected' AND occurred_at >= ?
                   ORDER BY occurred_at DESC""",
                (cutoff,),
            ).fetchall()
            for row in rows:
                existing = json.loads(row["payload_json"])
                if (
                    existing.get("metric_name") == payload.get("metric_name")
                    and existing.get("direction") == payload.get("direction")
                ):
                    return self._event_from_row(row)
            return self.emit_in_transaction(
                connection, "AnomalyDetected", payload, idempotency_key, severity,
            )

    def pending(self, limit: int = 100) -> list[AlertEvent]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM event_outbox WHERE status='pending' ORDER BY occurred_at LIMIT ?", (limit,)
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def claim_next(self) -> AlertEvent | None:
        """Atomically lease one event so concurrent workers cannot double-deliver it."""

        stale_before = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM event_outbox
                   WHERE status='pending' OR (status='processing' AND claimed_at < ?)
                   ORDER BY occurred_at LIMIT 1""",
                (stale_before,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE event_outbox SET status='processing', claimed_at=? WHERE event_id=?",
                (datetime.now(timezone.utc).isoformat(), row["event_id"]),
            )
            return self._event_from_row(row)

    def mark_processed(self, event_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE event_outbox SET status='processed', attempts=attempts+1, processed_at=?, claimed_at=NULL WHERE event_id=?",
                (datetime.now(timezone.utc).isoformat(), event_id),
            )

    def mark_failed(self, event_id: str, error: str, max_attempts: int = 3) -> None:
        with self._connection() as connection:
            row = connection.execute("SELECT attempts FROM event_outbox WHERE event_id=?", (event_id,)).fetchone()
            if row is None:
                return
            attempts = int(row["attempts"]) + 1
            status = "dead_letter" if attempts >= max_attempts else "pending"
            connection.execute(
                "UPDATE event_outbox SET status=?, attempts=?, last_error=?, claimed_at=NULL WHERE event_id=?",
                (status, attempts, error[:500], event_id),
            )


def anomaly_alert_payload(finding: AnomalyFinding) -> tuple[str, dict[str, object], Severity]:
    data = finding.model_dump(mode="json")
    impact_value = finding.value - finding.baseline
    impact_ratio = impact_value / abs(finding.baseline) if finding.baseline else None
    data.update({
        "impact_value": round(impact_value, 4), "impact_ratio": impact_ratio,
        "duration_days": finding.duration_days,
    })
    severity: Severity = (
        "critical" if abs(float(data["score"])) >= float(data["threshold"]) * 2
        or (impact_ratio is not None and abs(impact_ratio) >= 0.5)
        or finding.duration_days >= 3 else "warning"
    )
    key = f"anomaly:{data['metric_name']}:{data['metric_date']}:{data['method']}"
    return key, data, severity


class LocalNotificationAdapter:
    """Deterministic first-party adapter; external channels implement the same method."""

    def __init__(self) -> None:
        self.delivered: list[AlertEvent] = []

    def deliver(self, event: AlertEvent) -> None:
        self.delivered.append(event)


def process_pending_events(outbox: EventOutbox, adapter: NotificationAdapter, limit: int = 100) -> int:
    processed = 0
    for _ in range(limit):
        event = outbox.claim_next()
        if event is None:
            break
        try:
            adapter.deliver(event)
        except Exception as exc:
            outbox.mark_failed(event.event_id, str(exc))
        else:
            outbox.mark_processed(event.event_id)
            processed += 1
    return processed
