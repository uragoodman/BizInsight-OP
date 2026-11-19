"""Project unit metadata only when a trusted authorization pins this catalog."""
from __future__ import annotations
from typing import Any


def unit_evidence(case: dict[str, Any], registry: Any) -> dict[str, Any] | None:
    metric = (case.get("request_contract") or {}).get("metric_id")
    if not metric:
        return None
    digests = set()
    for bundle in case.get("evidence_bundles") or []:
        if not isinstance(bundle, dict) or bundle.get("trusted") is not True:
            continue
        content = bundle.get("content") or {}
        result = content.get("result") or {}
        parsed = result.get("parsed") or {}
        auth = parsed.get("authorized_metric") or {}
        if content.get("metric_id") == metric and parsed.get("metric_name") == metric and auth.get("metric_id") == metric and auth.get("catalog_digest"):
            digests.add(auth["catalog_digest"])
    if digests != {registry.catalog_digest}:
        return None
    try:
        definition = registry.get(metric)
    except (KeyError, ValueError):
        return None
    unit = getattr(definition, "unit", None)
    if not isinstance(unit, str) or not unit:
        return None
    return {"source": "authorized_registry_catalog", "metric_id": metric, "unit": unit, "catalog_digest": registry.catalog_digest,
            "role": "metric_definition_not_business_numeric_evidence"}
