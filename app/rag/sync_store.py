from __future__ import annotations

"""Durable queue and per-source state for knowledge index synchronization."""

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
from typing import Any
import uuid

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine


_SQLITE_LOCKS: dict[str, threading.Lock] = {}
_SQLITE_LOCKS_GUARD = threading.Lock()


def _tables(schema: str | None):
    metadata = MetaData()
    sources = Table(
        "rag_document_sources", metadata,
        Column("collection_name", String(160), primary_key=True),
        Column("source_id", Text, primary_key=True),
        Column("source_revision", String(128), nullable=False),
        Column("content_hash", String(64)),
        Column("indexed_revision", String(128)),
        Column("indexed_hash", String(64)),
        Column("status", String(16), nullable=False),
        Column("last_synced_at", DateTime(timezone=True)),
        Column("last_error", Text),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        schema=schema,
    )
    jobs = Table(
        "rag_document_sync_jobs", metadata,
        Column("job_id", String(64), primary_key=True),
        Column("collection_name", String(160), nullable=False),
        Column("corpus_hash", String(64), nullable=False),
        Column("source_revision", String(128), nullable=False),
        Column("status", String(16), nullable=False),
        Column("attempts", Integer, nullable=False, default=0),
        Column("max_attempts", Integer, nullable=False, default=8),
        Column("available_at", DateTime(timezone=True), nullable=False),
        Column("lease_owner", String(128)),
        Column("lease_expires_at", DateTime(timezone=True)),
        Column("last_error", Text),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("completed_at", DateTime(timezone=True)),
        UniqueConstraint("collection_name", "corpus_hash", name="uq_rag_sync_collection_corpus"),
        schema=schema,
    )
    return metadata, sources, jobs


class RagDocumentSyncStore:
    """Database-backed, at-least-once queue with idempotent corpus snapshots.

    PostgreSQL tables are installed by Alembic. SQLite is supported for local
    development and tests using the same repository behavior.
    """

    def __init__(self, engine: Engine | None = None, *, create_schema: bool = False):
        if engine is None:
            from app.data.database import engine as app_engine

            engine = app_engine
        self.engine = engine
        schema = "governance" if engine.dialect.name == "postgresql" else None
        self.metadata, self.sources, self.jobs = _tables(schema)
        if create_schema:
            self.metadata.create_all(engine)

    def enqueue(
        self, *, collection_name: str, corpus_hash: str, source_revision: str,
        retry_failed: bool = False,
    ) -> str:
        """Queue one reconciliation per distinct corpus snapshot."""
        job_id = hashlib.sha256(
            f"{collection_name}\0{source_revision}\0{corpus_hash}".encode("utf-8")
        ).hexdigest()
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            existing_row = conn.execute(
                select(self.jobs.c.job_id, self.jobs.c.status).where(
                    self.jobs.c.corpus_hash == corpus_hash,
                    self.jobs.c.collection_name == collection_name,
                )
            ).first()
            existing = existing_row[0] if existing_row else None
            if existing:
                if existing_row[1] == "failed" and retry_failed:
                    conn.execute(update(self.jobs).where(
                        self.jobs.c.job_id == existing
                    ).values(status="queued", attempts=0, available_at=now,
                             updated_at=now, completed_at=None, last_error=None))
                return str(existing)
            # An unclaimed job for an older snapshot is no longer useful: the
            # worker always indexes the latest complete corpus snapshot.
            conn.execute(update(self.jobs).where(and_(
                self.jobs.c.collection_name == collection_name,
                self.jobs.c.status.in_(["queued", "retry"]),
                self.jobs.c.corpus_hash != corpus_hash,
            )).values(status="superseded", updated_at=now))
            values = dict(
                job_id=job_id,
                collection_name=collection_name,
                corpus_hash=corpus_hash,
                source_revision=source_revision,
                status="queued",
                attempts=0,
                max_attempts=8,
                available_at=now,
                created_at=now,
                updated_at=now,
            )
            insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
            conn.execute(insert(self.jobs).values(**values).on_conflict_do_nothing(
                index_elements=[self.jobs.c.collection_name, self.jobs.c.corpus_hash]
            ))
            actual = conn.execute(select(self.jobs.c.job_id).where(
                self.jobs.c.collection_name == collection_name,
                self.jobs.c.corpus_hash == corpus_hash,
            )).scalar_one()
            return str(actual)

    def claim(self, *, worker_id: str, lease_seconds: int = 300) -> dict[str, Any] | None:
        """Claim the oldest available job; expired leases are recoverable."""
        now = datetime.now(timezone.utc)
        eligible = or_(
            and_(self.jobs.c.status.in_(["queued", "retry"]), self.jobs.c.available_at <= now),
            and_(self.jobs.c.status == "running", self.jobs.c.lease_expires_at <= now),
        )
        with self.engine.begin() as conn:
            # A worker may die while holding the final permitted attempt. Once
            # that lease expires, make the job terminal instead of leaving it
            # stranded as `running` forever.
            conn.execute(update(self.jobs).where(and_(
                self.jobs.c.status == "running",
                self.jobs.c.lease_expires_at <= now,
                self.jobs.c.attempts >= self.jobs.c.max_attempts,
            )).values(
                status="failed",
                lease_owner=None,
                lease_expires_at=None,
                last_error="worker lease expired after maximum attempts",
                updated_at=now,
            ))
            query = select(self.jobs).where(
                eligible, self.jobs.c.attempts < self.jobs.c.max_attempts
            ).order_by(self.jobs.c.created_at).limit(1)
            if self.engine.dialect.name == "postgresql":
                query = query.with_for_update(skip_locked=True)
            row = conn.execute(query).mappings().first()
            if row is None:
                return None
            token = uuid.uuid4().hex
            claimed = conn.execute(
                update(self.jobs).where(self.jobs.c.job_id == row["job_id"]).values(
                    status="running",
                    attempts=self.jobs.c.attempts + 1,
                    lease_owner=worker_id,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                ).returning(*self.jobs.c)
            ).mappings().first()
            return dict(claimed) if claimed else None

    def complete(self, job_id: str) -> None:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            conn.execute(update(self.jobs).where(self.jobs.c.job_id == job_id).values(
                status="completed", completed_at=now, updated_at=now,
                lease_owner=None, lease_expires_at=None, last_error=None,
            ))

    def renew_lease(self, job_id: str, worker_id: str, *, lease_seconds: int = 300) -> bool:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            result = conn.execute(update(self.jobs).where(and_(
                self.jobs.c.job_id == job_id,
                self.jobs.c.status == "running",
                self.jobs.c.lease_owner == worker_id,
            )).values(lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now))
            return result.rowcount == 1

    def supersede(self, job_id: str) -> None:
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            conn.execute(update(self.jobs).where(self.jobs.c.job_id == job_id).values(
                status="superseded", updated_at=now, lease_owner=None,
                lease_expires_at=None,
            ))

    def fail(
        self, job: dict[str, Any], error: str, *, retry_delay_seconds: int = 15,
        permanent: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc)
        terminal = permanent or int(job["attempts"]) >= int(job["max_attempts"])
        with self.engine.begin() as conn:
            conn.execute(update(self.jobs).where(self.jobs.c.job_id == job["job_id"]).values(
                status="failed" if terminal else "retry",
                available_at=now + timedelta(seconds=retry_delay_seconds),
                updated_at=now,
                lease_owner=None,
                lease_expires_at=None,
                last_error=error[:4000],
            ))

    def record_snapshot(
        self,
        *,
        collection_name: str,
        source_revision: str,
        documents: dict[str, str],
        succeeded: bool,
        error: str = "",
    ) -> None:
        """Persist observed/indexed hashes, including deletion tombstones."""
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            old_ids = set(conn.execute(select(self.sources.c.source_id).where(
                self.sources.c.collection_name == collection_name
            )).scalars())
            for source_id, content_hash in documents.items():
                values = {
                    "source_revision": source_revision,
                    "content_hash": content_hash,
                    "status": "indexed" if succeeded else "failed",
                    "last_error": None if succeeded else error[:4000],
                    "updated_at": now,
                }
                if succeeded:
                    values.update(indexed_revision=source_revision, indexed_hash=content_hash,
                                  last_synced_at=now)
                stmt = self.sources.select().where(
                    and_(self.sources.c.collection_name == collection_name,
                         self.sources.c.source_id == source_id)
                )
                exists = conn.execute(stmt).first()
                if exists:
                    conn.execute(update(self.sources).where(
                        and_(self.sources.c.collection_name == collection_name,
                             self.sources.c.source_id == source_id)
                    ).values(**values))
                else:
                    conn.execute(self.sources.insert().values(
                        collection_name=collection_name, source_id=source_id, **values
                    ))
            removed = old_ids - set(documents)
            if removed:
                conn.execute(update(self.sources).where(
                    and_(self.sources.c.collection_name == collection_name,
                         self.sources.c.source_id.in_(removed))
                ).values(
                    status="deleted" if succeeded else "failed",
                    content_hash=None,
                    indexed_hash=None if succeeded else self.sources.c.indexed_hash,
                    indexed_revision=source_revision if succeeded else self.sources.c.indexed_revision,
                    last_synced_at=now if succeeded else self.sources.c.last_synced_at,
                    last_error=None if succeeded else error[:4000],
                    updated_at=now,
                ))

    def list_sources(self, collection_name: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(select(self.sources).where(
                self.sources.c.collection_name == collection_name
            ).order_by(self.sources.c.source_id)).mappings()]

    @contextmanager
    def collection_lock(self, collection_name: str):
        """Serialize Qdrant writes and manifest commits per logical collection."""
        if self.engine.dialect.name == "postgresql":
            digest = hashlib.sha256(collection_name.encode("utf-8")).digest()
            key1 = int.from_bytes(digest[:4], "big", signed=True)
            key2 = int.from_bytes(digest[4:8], "big", signed=True)
            conn = self.engine.connect()
            acquired = bool(conn.execute(text(
                "SELECT pg_try_advisory_lock(:key1, :key2)"
            ), {"key1": key1, "key2": key2}).scalar())
            try:
                yield acquired
            finally:
                if acquired:
                    conn.execute(text("SELECT pg_advisory_unlock(:key1, :key2)"), {
                        "key1": key1, "key2": key2,
                    })
                conn.close()
            return

        lock_key = f"{self.engine.url}:{collection_name}"
        with _SQLITE_LOCKS_GUARD:
            lock = _SQLITE_LOCKS.setdefault(lock_key, threading.Lock())
        acquired = lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()


def corpus_snapshot(root: str) -> tuple[str, dict[str, str]]:
    """Return deterministic source hashes and corpus digest for a directory."""
    from app.rag.indexing import discover_knowledge_files

    documents: dict[str, str] = {}
    for filename in discover_knowledge_files(root):
        path = Path(filename)
        if not path.is_file():
            continue
        # Match KnowledgeDocument.doc_id so the queue, manifest and Qdrant
        # payload all use exactly the same stable source identity.
        source_id = path.resolve().relative_to(Path(root).resolve()).as_posix()
        documents[source_id] = hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
    encoded = json.dumps(documents, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), documents
