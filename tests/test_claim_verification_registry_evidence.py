from types import SimpleNamespace
from evaluation.claim_verification.registry_evidence import unit_evidence


def test_unit_requires_matching_authorized_catalog_and_trusted_evidence():
    registry = SimpleNamespace(catalog_digest="abc", get=lambda _: SimpleNamespace(unit="%"))
    bundle = {"trusted": True, "content": {"metric_id": "m", "result": {"parsed": {"metric_name": "m", "authorized_metric": {"metric_id": "m", "catalog_digest": "abc"}}}}}
    case = {"request_contract": {"metric_id": "m"}, "evidence_bundles": [bundle]}
    assert unit_evidence(case, registry)["unit"] == "%"
    bundle["content"]["result"]["parsed"]["authorized_metric"]["catalog_digest"] = "stale"
    assert unit_evidence(case, registry) is None
    bundle["trusted"] = False
    assert unit_evidence(case, registry) is None
