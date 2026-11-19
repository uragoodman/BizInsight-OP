"""Add password login, sessions and membership-level seller grants."""

from alembic import op


revision = "20260806_18"
down_revision = "20260805_17"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing Phase 7 identities had display names only.  Give legacy rows a
    # deterministic non-login placeholder before making email mandatory.
    op.execute("ALTER TABLE iam.users ADD COLUMN IF NOT EXISTS email varchar(320)")
    op.execute("ALTER TABLE iam.users ADD COLUMN IF NOT EXISTS password_hash text")
    op.execute("ALTER TABLE iam.users ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now()")
    op.execute("UPDATE iam.users SET email=lower(user_id) || '@legacy.local' WHERE email IS NULL")
    op.execute("ALTER TABLE iam.users ALTER COLUMN email SET NOT NULL")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_iam_users_email_ci ON iam.users(lower(email))")
    op.execute("ALTER TABLE iam.tenant_memberships DROP CONSTRAINT IF EXISTS tenant_memberships_role_check")
    op.execute("ALTER TABLE iam.tenant_memberships ADD CONSTRAINT tenant_memberships_role_check CHECK (role IN ('owner','admin','analyst','reviewer','viewer'))")
    op.execute("ALTER TABLE iam.tenant_memberships ADD COLUMN IF NOT EXISTS granted_by varchar(128)")
    op.execute("""
      CREATE TABLE IF NOT EXISTS iam.membership_seller_grants (
        tenant_id varchar(128) NOT NULL,
        user_id varchar(128) NOT NULL,
        seller_id varchar(64) NOT NULL,
        access_start_date date,
        access_end_date date,
        granted_by varchar(128),
        revoked_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (tenant_id,user_id,seller_id),
        FOREIGN KEY (tenant_id,user_id) REFERENCES iam.tenant_memberships(tenant_id,user_id),
        CHECK (access_end_date IS NULL OR access_start_date IS NULL OR access_end_date >= access_start_date)
      )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_membership_seller_grants_active ON iam.membership_seller_grants(tenant_id,user_id) WHERE revoked_at IS NULL")
    # Preserve the former one-seller tenant behaviour for all active members.
    op.execute("""
      INSERT INTO iam.membership_seller_grants(tenant_id,user_id,seller_id)
      SELECT m.tenant_id,m.user_id,a.seller_id
      FROM iam.tenant_memberships m
      JOIN iam.tenant_seller_accounts a ON a.tenant_id=m.tenant_id
      WHERE m.revoked_at IS NULL
      ON CONFLICT (tenant_id,user_id,seller_id) DO NOTHING
    """)
    op.execute("""
      CREATE TABLE IF NOT EXISTS iam.auth_sessions (
        session_id varchar(64) PRIMARY KEY,
        tenant_id varchar(128) NOT NULL REFERENCES iam.tenants(tenant_id),
        user_id varchar(128) NOT NULL REFERENCES iam.users(user_id),
        expires_at timestamptz NOT NULL,
        revoked_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT now(),
        last_seen_at timestamptz NOT NULL DEFAULT now()
      )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_active ON iam.auth_sessions(user_id,tenant_id,expires_at) WHERE revoked_at IS NULL")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS iam.auth_sessions")
    op.execute("DROP TABLE IF EXISTS iam.membership_seller_grants")
    op.execute("ALTER TABLE iam.tenant_memberships DROP COLUMN IF EXISTS granted_by")
    op.execute("ALTER TABLE iam.tenant_memberships DROP CONSTRAINT IF EXISTS tenant_memberships_role_check")
    op.execute("ALTER TABLE iam.tenant_memberships ADD CONSTRAINT tenant_memberships_role_check CHECK (role IN ('owner','analyst','reviewer','admin'))")
    op.execute("DROP INDEX IF EXISTS iam.uq_iam_users_email_ci")
    op.execute("ALTER TABLE iam.users DROP COLUMN IF EXISTS updated_at")
    op.execute("ALTER TABLE iam.users DROP COLUMN IF EXISTS password_hash")
    op.execute("ALTER TABLE iam.users DROP COLUMN IF EXISTS email")
