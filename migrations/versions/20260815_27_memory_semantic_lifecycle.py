"""Memory semantic recall + lifecycle columns (P3.1/P3.2).

- analysis_memories 增加 float8[] 嵌入向量（RLS 留在 PostgreSQL 内做余弦；pgvector
  HNSW 作为未来数据量扩大时的索引升级路径，不引入扩展依赖）。
- 状态枚举从 (active, revoked) 扩展为 (active, superseded, stale, revoked)，
  支持结论被取代、数据版本失效等生命周期；新增 valid_from/valid_to、
  superseded_by、source_data_version、metric_definition_version 与审核字段。
"""
from alembic import op

revision = "20260815_27"
down_revision = "20260814_26"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE governance.analysis_memories DROP CONSTRAINT IF EXISTS analysis_memories_status_check")
    op.execute("""ALTER TABLE governance.analysis_memories ADD CONSTRAINT analysis_memories_status_check
        CHECK (status IN ('active','superseded','stale','revoked'))""")
    # 语义召回：归一化嵌入（unit-length → 余弦 = 点积），pgvector 索引是未来扩容路径。
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS embedding float8[]")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS embedding_model varchar(64)")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS embedding_version varchar(64)")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS embedded_at timestamptz")
    # 生命周期治理。
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS valid_from date")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS valid_to date")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS superseded_by varchar(128)")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS source_data_version varchar(64)")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS metric_definition_version varchar(64)")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS reviewed_by varchar(128)")
    op.execute("ALTER TABLE governance.analysis_memories ADD COLUMN IF NOT EXISTS reviewed_at timestamptz")


def downgrade() -> None:
    for column in (
        "reviewed_at", "reviewed_by", "metric_definition_version", "source_data_version",
        "superseded_by", "valid_to", "valid_from", "embedded_at", "embedding_version",
        "embedding_model", "embedding",
    ):
        op.execute(f"ALTER TABLE governance.analysis_memories DROP COLUMN IF EXISTS {column}")
    op.execute("ALTER TABLE governance.analysis_memories DROP CONSTRAINT IF EXISTS analysis_memories_status_check")
    op.execute("""ALTER TABLE governance.analysis_memories ADD CONSTRAINT analysis_memories_status_check
        CHECK (status IN ('active','revoked'))""")
