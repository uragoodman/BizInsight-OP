"""Offline B/C comparison and constrained Hybrid arbitration for P3b.

The comparison layer has no provider client and never compiles a contract or
executes SQL. Predictors are injected by the caller. Prediction rows are bound
to the case's existing L1 mention IDs, canonical dimensions, and exact spans.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

ALLOWED_ROLES = frozenset({"grouping", "filter", "reference", "comparison", "unclear"})
RULE_ROLES = ALLOWED_ROLES | {"conflict", "unresolved"}
ELIGIBLE_RISK_TIERS = frozenset({"open", "elevated", "conflict"})
RoleRecord = Mapping[str, Any]
PredictionRows = Sequence[RoleRecord]
RolePredictor = Callable[[dict[str, Any]], Any]


def _mentions(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    source_text = str(case.get("input") or "")
    for item in case.get("mentions") or []:
        if not isinstance(item, Mapping):
            raise ValueError("annotated mention must be an object")
        if item.get("note") == "metric":
            continue
        mention_id = item.get("id")
        canonical = item.get("canonical")
        if not isinstance(mention_id, str) or not mention_id:
            raise ValueError("annotated mention id must be a non-empty string")
        if not isinstance(canonical, str) or not canonical:
            raise ValueError("annotated mention canonical must be a non-empty string")
        start, end = item.get("start"), item.get("end")
        raw_text = item.get("raw_text")
        if (
            type(start) is not int or type(end) is not int or start < 0 or end <= start
            or not isinstance(raw_text, str) or source_text[start:end] != raw_text
        ):
            raise ValueError(f"annotated mention has invalid source span: {mention_id}")
        if mention_id in result:
            raise ValueError(f"duplicate annotated mention id: {mention_id}")
        result[mention_id] = dict(item)
    return result


def _unpack_prediction(value: Any) -> tuple[list[dict[str, Any]], bool]:
    """Accept rows or a verifier envelope; malformed structures are invalid."""
    abstain = False
    if isinstance(value, Mapping):
        if set(value) != {"roles", "abstain"} or not isinstance(value.get("abstain"), bool):
            raise ValueError("prediction envelope must contain roles and abstain only")
        abstain = bool(value["abstain"])
        value = value["roles"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("predictions must be a sequence of mention role records")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("prediction row must be an object")
        rows.append(dict(item))
    if abstain and rows:
        raise ValueError("abstain output must not contain roles")
    return rows, abstain


def _validate_rows(
    rows: PredictionRows,
    mentions: Mapping[str, Mapping[str, Any]],
    *,
    allow_unresolved: bool,
) -> dict[str, dict[str, Any]]:
    """Validate id/span/canonical binding; return an ID-indexed copy."""
    checked: dict[str, dict[str, Any]] = {}
    allowed = RULE_ROLES if allow_unresolved else ALLOWED_ROLES
    for raw in rows:
        row = dict(raw)
        mention_id = row.get("id")
        if not isinstance(mention_id, str) or not mention_id:
            raise ValueError("mention id must be a non-empty string")
        if not mention_id or mention_id in checked:
            raise ValueError("missing or duplicate mention id")
        source = mentions.get(mention_id)
        if source is None:
            raise ValueError("prediction introduced an unknown mention id")
        if type(row.get("start")) is not int or type(row.get("end")) is not int:
            raise ValueError("mention offsets must be integers")
        if not isinstance(row.get("canonical"), str):
            raise ValueError("mention canonical must be a string")
        if not isinstance(row.get("role"), str):
            raise ValueError("role must be a string")
        for field in ("start", "end", "canonical"):
            if row.get(field) != source.get(field):
                raise ValueError(f"prediction changed mention {field}")
        role = row.get("role")
        if role not in allowed:
            raise ValueError("prediction role is outside the closed label set")
        # Reject extra keys so the comparison interface cannot smuggle metric,
        # seller, scope, date, SQL, or free-text values into an output record.
        if set(row) != {"id", "start", "end", "canonical", "role"}:
            raise ValueError("prediction row has unexpected fields")
        checked[mention_id] = row
    return checked


def _expected_roles(case: dict[str, Any]) -> dict[str, str]:
    expected = dict(case.get("roles") or (case.get("expected") or {}).get("roles") or {})
    return {str(key): str(value) for key, value in expected.items() if value != "unresolved"}


def _score_rows(
    rows: PredictionRows,
    *,
    mentions: Mapping[str, Mapping[str, Any]],
    source_text: str,
    allow_unmatched_mentions: bool,
) -> dict[str, dict[str, Any]]:
    if not allow_unmatched_mentions:
        return _validate_rows(rows, mentions, allow_unresolved=True)

    checked: dict[str, dict[str, Any]] = {}
    raw_ids: set[str] = set()
    for raw in rows:
        row = dict(raw)
        mention_id = row.get("id")
        start, end = row.get("start"), row.get("end")
        canonical, role = row.get("canonical"), row.get("role")
        if not isinstance(mention_id, str) or not mention_id or mention_id in raw_ids:
            raise ValueError("missing or duplicate predicted mention id")
        raw_ids.add(mention_id)
        if type(start) is not int or type(end) is not int or start < 0 or end <= start or end > len(source_text):
            raise ValueError("predicted mention offsets are outside the input")
        if not isinstance(canonical, str) or not canonical or not isinstance(role, str) or role not in RULE_ROLES:
            raise ValueError("invalid predicted canonical dimension or role")
        if set(row) != {"id", "start", "end", "canonical", "role"}:
            raise ValueError("prediction row has unexpected fields")

        source = mentions.get(mention_id)
        exact = bool(source) and all(
            row.get(field) == source.get(field)
            for field in ("start", "end", "canonical")
        )
        key = mention_id if exact else f"fp:{mention_id}:{start}:{end}:{canonical}"
        if key in checked:
            raise ValueError("duplicate predicted mention span")
        checked[key] = row
    return checked


def score_roles(
    cases: Sequence[dict[str, Any]],
    predictions_by_case: Mapping[str, Any] | RolePredictor,
    *,
    allow_unmatched_mentions: bool = False,
) -> dict[str, Any]:
    """Score mention/span-bound predictions against annotation only.

    ``predictions_by_case`` may be a predictor callable or a mapping from
    ``case_id`` to predictions. The optional unmatched mode is intended only
    for the deterministic rules baseline: extra valid spans count as FP and
    missing gold spans count as FN. Verifier/B/C predictions stay strict.
    """
    mention_tp = mention_fp = mention_fn = 0
    role_tp = role_fp = role_fn = 0
    invalid_predictions = 0

    for case in cases:
        annotated = _mentions(case)
        try:
            if callable(predictions_by_case):
                prediction = predictions_by_case(case)
            else:
                prediction = predictions_by_case.get(str(case.get("case_id") or ""), [])
            raw_rows, abstain = _unpack_prediction(prediction)
            predicted = {} if abstain else _score_rows(
                raw_rows,
                mentions=annotated,
                source_text=str(case.get("input") or ""),
                allow_unmatched_mentions=allow_unmatched_mentions,
            )
        except (TypeError, ValueError):
            # Invalid B labels are a failed prediction, not a relaxed contract.
            predicted = {}
            invalid_predictions += 1

        predicted_ids = set(predicted)
        expected_ids = set(annotated)
        mention_tp += len(expected_ids & predicted_ids)
        mention_fp += len(predicted_ids - expected_ids)
        mention_fn += len(expected_ids - predicted_ids)

        expected_roles = _expected_roles(case)
        for mention_id, mention in annotated.items():
            canonical = str(mention.get("canonical") or "")
            gold_role = expected_roles.get(canonical)
            if gold_role is None:
                continue
            predicted_role = predicted.get(mention_id, {}).get("role")
            if predicted_role == gold_role:
                role_tp += 1
            else:
                role_fn += 1
                if predicted_role not in {None, "unresolved"}:
                    role_fp += 1
        if allow_unmatched_mentions:
            role_fp += sum(
                1 for mention_id in set(predicted) - set(annotated)
                if predicted[mention_id].get("role") != "unresolved"
            )

    mention_denom = 2 * mention_tp + mention_fp + mention_fn
    role_denom = 2 * role_tp + role_fp + role_fn
    return {
        "case_count": len(cases),
        "mention_precision": round(mention_tp / (mention_tp + mention_fp), 4)
        if mention_tp + mention_fp else None,
        "mention_recall": round(mention_tp / (mention_tp + mention_fn), 4)
        if mention_tp + mention_fn else None,
        "mention_f1": round(2 * mention_tp / mention_denom, 4) if mention_denom else 0.0,
        "role_precision": round(role_tp / (role_tp + role_fp), 4)
        if role_tp + role_fp else None,
        "role_recall": round(role_tp / (role_tp + role_fn), 4)
        if role_tp + role_fn else None,
        "role_f1": round(2 * role_tp / role_denom, 4) if role_denom else 0.0,
        "false_positive_roles": role_fp,
        "false_negative_roles": role_fn,
        "invalid_prediction_cases": invalid_predictions,
        "coverage": None,
        "wrong_contract_rate": None,
        "wrong_execution_rate": None,
        "metrics_scope": "role_and_mention_only",
    }


def _metrics(cases: Sequence[dict[str, Any]], predictor: RolePredictor) -> dict[str, Any]:
    """Strict alias retained for comparison callers and P3b tests."""
    return score_roles(cases, predictor, allow_unmatched_mentions=False)


def hybrid_roles(
    rule_roles: PredictionRows,
    verifier_roles: Any,
    *,
    mentions: Mapping[str, Mapping[str, Any]],
    risk_tier: str,
    eligible_mentions: Sequence[str],
    blocked: bool = True,
) -> list[dict[str, Any]]:
    """Apply a closed, mention-bound arbitration policy.

    Any denied/high/date/scope/support preflight must be passed as ``blocked``
    by the caller. Abstention or any invalid verifier set leaves rule rows
    untouched. The result contains the original complete rule mention set.
    """
    rules = _validate_rows(rule_roles, mentions, allow_unresolved=True)
    original = {key: dict(value) for key, value in rules.items()}
    if blocked or risk_tier not in ELIGIBLE_RISK_TIERS:
        return list(original.values())

    try:
        raw_verifier, abstain = _unpack_prediction(verifier_roles)
        if abstain:
            return list(original.values())
        verifier = _validate_rows(raw_verifier, mentions, allow_unresolved=False)
    except (TypeError, ValueError):
        return list(original.values())

    eligible = {str(item) for item in eligible_mentions}
    for mention_id, candidate in verifier.items():
        if mention_id not in rules or mention_id not in eligible:
            continue
        # Unclear is an abstention for arbitration, even though it is a valid
        # verifier label. All other labels only replace this existing mention.
        if candidate["role"] == "unclear":
            continue
        original[mention_id] = dict(rules[mention_id]) | {"role": candidate["role"]}
    return list(original.values())


def compare(
    cases: Sequence[dict[str, Any]],
    *,
    rules_predictor: RolePredictor,
    llm_predictor: RolePredictor,
    verifier_predictor: RolePredictor,
    risk_by_case: Callable[[dict[str, Any]], str] | None = None,
    blocked_by_case: Callable[[dict[str, Any]], bool] | None = None,
    eligible_by_case: Callable[[dict[str, Any]], Sequence[str]] | None = None,
    post_hybrid_preflight: Callable[[dict[str, Any], PredictionRows], bool] | None = None,
    result_provenance: str = "synthetic_test_only",
) -> dict[str, Any]:
    """Compare A/B/C predictions; this function never creates a provider client."""
    if result_provenance not in {"synthetic_test_only", "adapter_predictions"}:
        raise ValueError("unsupported result provenance")

    gates: dict[str, tuple[str, bool, tuple[str, ...]]] = {}
    evaluated_cases: list[dict[str, Any]] = []
    blocked_cases = 0
    for case in cases:
        try:
            risk_tier = risk_by_case(case) if risk_by_case else "high"
            # Missing a deterministic authorization/scope/date/support
            # preflight is fail-closed before any B/C predictor may run.
            blocked = blocked_by_case(case) if blocked_by_case else True
            eligible = tuple(
                str(item) for item in (
                    eligible_by_case(case) if eligible_by_case else tuple(_mentions(case))
                )
            )
        except Exception:  # noqa: BLE001 - preflight errors are denied
            risk_tier, blocked, eligible = "high", True, ()
        allowed = risk_tier in ELIGIBLE_RISK_TIERS and not blocked
        gates[id(case)] = (risk_tier, blocked, eligible)
        if allowed:
            evaluated_cases.append(case)
        else:
            blocked_cases += 1

    def hybrid_predictor(case: dict[str, Any]) -> list[dict[str, Any]]:
        annotated = _mentions(case)
        rules, _ = _unpack_prediction(rules_predictor(case))
        risk_tier, blocked, eligible = gates[id(case)]
        if blocked or risk_tier not in ELIGIBLE_RISK_TIERS:
            # Do not call the verifier for high/denied/scope/date/support
            # failures; keep the complete deterministic role map.
            return list(_validate_rows(rules, annotated, allow_unresolved=True).values())
        if not eligible:
            return list(_validate_rows(rules, annotated, allow_unresolved=True).values())
        raw_verifier = verifier_predictor(case)
        candidate = hybrid_roles(
            rules,
            raw_verifier,
            mentions=annotated,
            risk_tier=risk_tier,
            eligible_mentions=eligible,
            blocked=blocked,
        )
        if candidate != rules:
            # A relabel remains diagnostic until deterministic support and
            # authorization accept it. A missing/failed recheck keeps rules.
            if post_hybrid_preflight is None:
                return list(rules)
            try:
                if not post_hybrid_preflight(case, candidate):
                    return list(rules)
            except Exception:  # noqa: BLE001 - failed recheck is denied
                return list(rules)
        return candidate

    def safe_llm_predictor(case: dict[str, Any]) -> Any:
        risk_tier, blocked, _eligible = gates[id(case)]
        if blocked or risk_tier not in ELIGIBLE_RISK_TIERS:
            # Keep the full dataset denominator comparable with A/C while
            # representing cases where a model call was prohibited as missing
            # predictions. Never invoke an adapter for those cases.
            return []
        return llm_predictor(case)

    result: dict[str, Any] = {
        "dataset_count": len(cases),
        "A_rules_only": score_roles(cases, rules_predictor, allow_unmatched_mentions=True),
        # Keep A/B/C on the same gold dataset and denominator. B has no rules
        # fallback: when deterministic preflight blocks a call, its prediction
        # is empty and scores as missing rather than borrowing the rule output.
        "B_llm_only": _metrics(cases, safe_llm_predictor),
        "C_hybrid": _metrics(cases, hybrid_predictor),
        "evaluated_case_count": len(evaluated_cases),
        "blocked_case_count": blocked_cases,
        "result_provenance": result_provenance,
        "synthetic_test_only": result_provenance == "synthetic_test_only",
        "metrics_scope": "role_and_mention_only",
        "coverage": None,
        "wrong_contract_rate": None,
        "wrong_execution_rate": None,
        "note": (
            "Role/mention-only offline comparison. This module does not call a "
            "provider, compile contracts, or execute SQL. Contract and execution "
            "metrics are intentionally null."
        ),
    }
    return result
