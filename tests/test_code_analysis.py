"""Phase 11 policy and tenant-scoped job lifecycle tests."""

from __future__ import annotations

import os
import secrets

import pytest

from app.code_analysis.policy import (
    validate_candidate_code,
    validate_candidate_requirements,
)
from app.code_analysis.providers import (
    _extract_json_payload,
    _normalize_explanation,
    _normalize_requirements,
    _response_text,
)
from app.code_analysis.service import (
    cancel_job,
    confirm_job,
    create_job,
    get_job,
    revise_job,
    upload_file,
)
from app.data.database import engine
from app.tenancy import TenantContext
from sqlalchemy import text


pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_URL"), reason="requires PostgreSQL"
)


def _context(user: str) -> TenantContext:
    return TenantContext(
        tenant_id="phase11-test",
        user_id=f"{user}-{secrets.token_hex(4)}",
        role="admin",
        scope_type="platform",
    )


def test_candidate_policy_rejects_execution_and_network_primitives():
    unsafe = (
        "import subprocess\n\ndef analyze(input_path, output_dir):\n return eval('1')"
    )
    result = validate_candidate_code(unsafe)
    assert not result.valid
    assert any("subprocess" in error for error in result.errors)
    assert any("eval" in error for error in result.errors)


def test_candidate_contract_and_requirements_are_strict():
    payload = _extract_json_payload(
        '```json\n{"analysis.py":"def analyze(input_path, output_dir):\\n return {}",'
        '"explanation.json":{},"requirements.json":["pandas"]}\n```'
    )
    assert payload["requirements.json"] == ["pandas"]
    preamble_payload = _extract_json_payload(
        'Here is the artifact:\n{"analysis.py":"def analyze(input_path, output_dir):\\n return {}",'
        '"explanation.json":{},"requirements.json":[]} '
    )
    assert "analysis.py" in preamble_payload
    brace_preamble_payload = _extract_json_payload(
        'The result shape is {not-json}; final artifact follows: '
        '{"analysis.py":"def analyze(input_path, output_dir):\\n return {}",'
        '"explanation.json":{},"requirements.json":[]}'
    )
    assert "analysis.py" in brace_preamble_payload
    assert (
        _response_text(
            {
                "parts": [
                    {"type": "reasoning", "text": "hidden"},
                    {"type": "text", "text": "{}"},
                ]
            }
        )
        == "{}"
    )
    assert _normalize_explanation('{"method":"trend"}') == {"method": "trend"}
    assert _normalize_requirements('{"dependencies":["pandas"]}') == ["pandas"]
    assert validate_candidate_requirements(["pandas", "scikit-learn"]).valid
    assert not validate_candidate_requirements(["requests>=2"]).valid


def test_upload_create_confirm_and_tenant_isolation():
    owner = _context("phase11-owner")
    uploaded = upload_file(
        owner, "sales.csv", "text/csv", b"month,gmv,email\n2024-01,12,a@example.com\n"
    )
    assert uploaded["file_type"] == "csv"
    assert uploaded["data_contract"]["columns"][2]["sensitive"] is True
    job = create_job(owner, uploaded["file_id"], "Analyze GMV trend")
    assert job["status"] == "awaiting_confirmation"
    # The application connection used by local tests is PostgreSQL superuser,
    # which bypasses RLS. Prove isolation under the production-style non-
    # BYPASSRLS role instead of asserting a misleading superuser behaviour.
    with engine.begin() as conn:
        conn.execute(
            text(
                "DO $$ BEGIN CREATE ROLE phase11_reader NOINHERIT; EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )
        conn.execute(text("GRANT USAGE ON SCHEMA governance TO phase11_reader"))
        conn.execute(
            text("GRANT SELECT ON governance.code_analysis_jobs TO phase11_reader")
        )
        conn.execute(
            text("GRANT SELECT ON governance.code_analysis_access_grants TO phase11_reader")
        )
        conn.execute(text("SET LOCAL ROLE phase11_reader"))
        conn.execute(text("SELECT set_config('app.tenant_id', 'phase11-test', true)"))
        conn.execute(text("SELECT set_config('app.user_id', 'phase11-other', true)"))
        assert (
            conn.execute(
                text(
                    "SELECT count(*) FROM governance.code_analysis_jobs WHERE job_id=:id"
                ),
                {"id": job["job_id"]},
            ).scalar_one()
            == 0
        )
    revised = revise_job(owner, job["job_id"], "Analyze GMV trend and anomalies")
    assert get_job(owner, job["job_id"])["status"] == "superseded"
    assert revised["parent_job_id"] == job["job_id"]
    assert revised["revision"] == 2
    assert revised["plan_hash"] != job["plan_hash"]
    confirmed = confirm_job(owner, revised["job_id"], revised["plan_hash"])
    assert confirmed["status"] == "queued"
    assert cancel_job(owner, revised["job_id"])
    assert get_job(owner, revised["job_id"])["status"] == "cancelled"


def test_confirm_rejects_tampered_plan_hash():
    context = _context("phase11-hash")
    uploaded = upload_file(context, "simple.csv", "text/csv", b"x\n1\n")
    job = create_job(context, uploaded["file_id"], "Summarize the file")
    with pytest.raises(ValueError, match="changed"):
        confirm_job(context, job["job_id"], "0" * 64)


def test_expired_confirmation_persists_expired_status():
    context = _context("phase11-expiry")
    uploaded = upload_file(context, "expiry.csv", "text/csv", b"x\n1\n")
    job = create_job(context, uploaded["file_id"], "Summarize before expiry")
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE governance.code_analysis_jobs SET expires_at=now()-interval '1 second' WHERE job_id=:id"
            ),
            {"id": job["job_id"]},
        )
    with pytest.raises(ValueError, match="expired"):
        confirm_job(context, job["job_id"], job["plan_hash"])
    assert get_job(context, job["job_id"])["status"] == "expired"
