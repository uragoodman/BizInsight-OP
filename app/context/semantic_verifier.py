"""Span-bound semantic role verification for P3.

The verifier is an observer in this release. It may inspect only detected
dimension mentions and their deterministic role proposals. It cannot change a
contract, scope, metric, date, or execution decision.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import date
from time import perf_counter
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

RULES_VERSION = "dimension-role-p2"
PROMPT_VERSION = "semantic-verifier-p3-v2"
ALLOWED_ROLES = ("grouping", "filter", "reference", "comparison", "unclear")
REASON_CODES = (
    "explicit_grouping",
    "filter_condition",
    "scope_reference",
    "comparison_target",
    "insufficient_context",
    "ambiguous_cue",
)
EVIDENCE_CODES = (
    "group_prefix",
    "group_suffix",
    "filter_prefix",
    "region_filter",
    "reference_prefix",
    "reference_suffix",
    "possessive",
    "comparison_context",
    "conflict",
    "coordination",
    "region_token",
)
RiskTier = Literal["high", "conflict", "elevated", "open", "low"]
RoleLabel = Literal["grouping", "filter", "reference", "comparison", "unclear"]
ReasonCode = Literal[
    "explicit_grouping",
    "filter_condition",
    "scope_reference",
    "comparison_target",
    "insufficient_context",
    "ambiguous_cue",
]


class MentionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, max_length=40)
    raw_text: str = Field(min_length=1, max_length=120)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    canonical: str = Field(min_length=1, max_length=80)


class ProposedRole(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, max_length=40)
    role: Literal["grouping", "filter", "reference", "comparison", "conflict", "unresolved"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    # Coarse rule evidence codes are safe to send; raw question context is not.
    evidence: tuple[Literal[
        "group_prefix", "group_suffix", "filter_prefix", "region_filter",
        "reference_prefix", "reference_suffix", "possessive",
        "comparison_context", "conflict", "coordination", "region_token",
    ], ...] = Field(default=(), max_length=12)


class VerifierRole(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, max_length=40)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    role: RoleLabel
    reason_code: ReasonCode


class VerifierOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    roles: list[VerifierRole]
    abstain: bool


InvokeModel = Callable[[str, dict[str, Any], str], Any]
_CACHE: dict[str, VerifierOutput] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_LIMIT = 512

SYSTEM_PROMPT = """You are a semantic role verifier. Return one JSON object only.
You may classify only the provided eligible mention IDs. Do not add mentions,
dimensions, or facts. Copy each mention's id/start/end exactly. Choose one role
from grouping, filter, reference, comparison, unclear. Use only a listed
reason_code. If uncertain, set abstain=true and return roles=[]. Never output
SQL, a metric, a date, a seller, a scope, or natural-language query text.
"""


class VerifierUnavailable(RuntimeError):
    def __init__(self, reason_code: str = "verifier_call_failed") -> None:
        self.reason_code = reason_code if reason_code in {"timeout", "provider_error", "parse_failure", "quota_blocked"} else "provider_error"
        super().__init__(self.reason_code)


def _digest(value: Any) -> str:
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def mention_id(index: int) -> str:
    return f"m{index + 1}"


def coarse_evidence_codes(values: Any) -> tuple[str, ...]:
    """Keep rule evidence useful while excluding arbitrary source fragments."""
    result: list[str] = []
    for value in values or ():
        raw = str(value)
        if raw.startswith("group_prefix:"):
            code = "group_prefix"
        elif raw.startswith("group_suffix:"):
            code = "group_suffix"
        elif raw.startswith("filter_prefix:"):
            code = "filter_prefix"
        elif raw.startswith("reference_prefix:"):
            code = "reference_prefix"
        elif raw.startswith("reference_suffix:"):
            code = "reference_suffix"
        elif raw in EVIDENCE_CODES:
            code = raw
        else:
            continue
        if code not in result:
            result.append(code)
    return tuple(result)


def _key(
    *, question: str, mentions: list[MentionInput], proposals: list[ProposedRole],
    eligible_ids: list[str], model: str, rules_version: str, prompt_version: str,
) -> str:
    # The question hash is required because roles depend on surrounding syntax;
    # the raw question itself is never stored in the cache key or cache value.
    return _digest({
        "question_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "mentions_hash": _digest([item.model_dump() for item in mentions]),
        "proposed_roles_hash": _digest([item.model_dump() for item in proposals]),
        "eligible_ids": sorted(eligible_ids),
        "model": model,
        "rules_version": rules_version,
        "prompt_version": prompt_version,
    })


def _default_invoke(system_prompt: str, request: dict[str, Any], model: str) -> tuple[Any, dict[str, Any]]:
    from app.core.execution_context import active_state
    from app.core.config import get_settings
    from app.llm.client import llm_extract
    from app.llm.interaction_policy import InteractionPolicy

    active = active_state()
    calls_before = len(active.get("llm_calls") or []) if isinstance(active, dict) else 0
    settings = get_settings()
    interaction_policy = InteractionPolicy(
        name="semantic_verifier",
        transport="nonstream",
        thinking="provider_default",
        connect_timeout_seconds=10.0,
        idle_timeout_seconds=None,
        total_timeout_seconds=float(settings.semantic_verifier_timeout_seconds),
        provider_fallback_enabled=False,
    )
    result = llm_extract(
        system_prompt,
        json.dumps(request, ensure_ascii=False, separators=(",", ":")),
        model=model or None,
        temperature=0.0,
        operation="extract.semantic_verifier",
        interaction_policy=interaction_policy,
    )
    usage: dict[str, Any] = {}
    if isinstance(active, dict):
        calls = active.get("llm_calls") or []
        if len(calls) > calls_before:
            call = calls[-1]
            if result is None:
                outcome = str(call.get("outcome") or "provider_error")
                if outcome in {"timeout", "provider_timeout"}:
                    raise VerifierUnavailable("timeout")
                if outcome in {"parse_failure", "empty_response"}:
                    raise VerifierUnavailable("parse_failure")
                if outcome == "not_sent" and call.get("not_sent_reason") == "quota_blocked":
                    raise VerifierUnavailable("quota_blocked")
                raise VerifierUnavailable("provider_error")
            usage = {
                "input_tokens": call.get("input_tokens_subtotal"),
                "output_tokens": call.get("output_tokens_subtotal"),
                "model": next((
                    attempt.get("model")
                    for attempt in reversed(call.get("attempts") or [])
                    if attempt.get("model")
                ), model or None),
            }
    if result is None:
        raise VerifierUnavailable("provider_error")
    return result, usage


def _safe_event(record: dict[str, Any]) -> None:
    try:
        from app.observability import record_event

        # Do not add question text, mention text, or arbitrary model content.
        record_event("semantic.verifier.shadow", {
            "status": record["status"],
            "mode": record["mode"],
            "risk_tier": record["risk_tier"],
            "prompt_version": record["prompt_version"],
            "rules_version": record["rules_version"],
            "model": record.get("model"),
            "span_ids": record.get("span_ids", []),
            "rule_roles": record.get("rule_roles", {}),
            "verifier_roles": record.get("verifier_roles", {}),
            "verifier_reason_codes": record.get("verifier_reason_codes", {}),
            "disagreement_span_ids": record.get("disagreement_span_ids", []),
            "latency_ms": record.get("latency_ms"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "estimated_cost": record.get("estimated_cost"),
            "cache_hit": bool(record.get("cache_hit")),
            "reason_code": record.get("reason_code"),
        })
    except Exception:  # noqa: BLE001 - observer must never affect compilation
        pass


def verify_roles(
    *,
    question: str,
    mentions: list[dict[str, Any]],
    proposed_roles: list[dict[str, Any]],
    eligible_ids: list[str],
    risk_tier: str,
    mode: str,
    model: str = "",
    allow_raw_question: bool = False,
    invoke_model: InvokeModel | None = None,
    rules_version: str = RULES_VERSION,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    """Call and validate the verifier; always return a safe, non-throwing record."""
    started = perf_counter()
    record: dict[str, Any] = {
        "status": "skipped",
        "mode": mode,
        "risk_tier": risk_tier,
        "rules_version": rules_version,
        "prompt_version": prompt_version,
        "model": model or None,
        "span_ids": sorted(set(eligible_ids)),
        "rule_roles": {},
        "verifier_roles": {},
        "verifier_reason_codes": {},
        "disagreement_span_ids": [],
        "latency_ms": 0.0,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost": None,
        "cache_hit": False,
        "reason_code": None,
        # Output is observation-only. Callers must never apply it in P3.
        "applied": False,
    }
    try:
        mention_models = [MentionInput.model_validate(item) for item in mentions]
        proposal_models = [ProposedRole.model_validate(item) for item in proposed_roles]
        if mode not in {"off", "shadow", "open_only"}:
            record["status"] = "invalid_configuration"
            record["reason_code"] = "invalid_mode"
            return record
        if mode == "off":
            record["reason_code"] = "mode_off"
            return record
        if risk_tier not in {"elevated", "open", "conflict"}:
            record["reason_code"] = "risk_not_eligible"
            return record
        if mode == "open_only" and risk_tier != "open":
            record["reason_code"] = "open_only_route_mismatch"
            return record
        if not allow_raw_question:
            record["status"] = "blocked_by_privacy"
            record["reason_code"] = "raw_question_not_permitted"
            return record

        mention_by_id = {item.id: item for item in mention_models}
        proposal_by_id = {item.id: item for item in proposal_models}
        if len(mention_by_id) != len(mention_models) or len(proposal_by_id) != len(proposal_models):
            raise ValueError("duplicate mention or proposal ID")
        if not set(proposal_by_id).issubset(mention_by_id):
            raise ValueError("proposal ID is not bound to a known mention")
        if len(eligible_ids) != len(set(eligible_ids)):
            raise ValueError("duplicate eligible mention ID")
        eligible = sorted(eligible_ids)
        if not eligible:
            record["reason_code"] = "no_eligible_mentions"
            return record
        if any(item_id not in mention_by_id or item_id not in proposal_by_id for item_id in eligible):
            raise ValueError("eligible mention is missing a bound proposal")
        for item_id, mention in mention_by_id.items():
            if mention.end <= mention.start or question[mention.start:mention.end] != mention.raw_text:
                raise ValueError("mention span does not match source text")
            proposal = proposal_by_id.get(item_id)
            if proposal is not None and (proposal.start, proposal.end) != (mention.start, mention.end):
                raise ValueError("proposal span differs from mention")
        record["rule_roles"] = {
            item_id: proposal_by_id[item_id].role for item_id in eligible
        }
        cache_key = _key(
            question=question,
            mentions=mention_models,
            proposals=proposal_models,
            eligible_ids=eligible,
            model=model,
            rules_version=rules_version,
            prompt_version=prompt_version,
        )
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            parsed = cached
            record["cache_hit"] = True
            usage: dict[str, Any] = {}
            record["input_tokens"] = 0
            record["output_tokens"] = 0
            record["estimated_cost"] = 0.0
        else:
            request = {
                "mentions": [item.model_dump() for item in mention_models],
                "proposed_roles": [item.model_dump() for item in proposal_models],
                "eligible_mention_ids": eligible,
                "allowed_roles": list(ALLOWED_ROLES),
                "reason_codes": list(REASON_CODES),
            }
            result = (invoke_model or _default_invoke)(SYSTEM_PROMPT, request, model)
            if isinstance(result, tuple) and len(result) == 2:
                raw_output, usage = result
            else:
                raw_output, usage = result, {}
            if not isinstance(raw_output, dict):
                raise ValueError("verifier response is not a JSON object")
            parsed = VerifierOutput.model_validate(raw_output)
            output_ids = [item.id for item in parsed.roles]
            if len(output_ids) != len(set(output_ids)) or any(item_id not in eligible for item_id in output_ids):
                raise ValueError("verifier returned unknown or duplicate mention IDs")
            if parsed.abstain:
                if parsed.roles:
                    raise ValueError("abstaining verifier must return no roles")
            elif set(output_ids) != set(eligible):
                raise ValueError("verifier must classify every eligible mention")
            for item in parsed.roles:
                source = mention_by_id[item.id]
                if (item.start, item.end) != (source.start, source.end):
                    raise ValueError("verifier changed a span")
            with _CACHE_LOCK:
                if len(_CACHE) >= _CACHE_LIMIT:
                    _CACHE.pop(next(iter(_CACHE)))
                _CACHE[cache_key] = parsed
            record["input_tokens"] = usage.get("input_tokens")
            record["output_tokens"] = usage.get("output_tokens")
            record["model"] = usage.get("model") or record["model"]
            if isinstance(record["input_tokens"], int) and isinstance(record["output_tokens"], int):
                from app.core.config import get_settings

                configured = get_settings()
                record["estimated_cost"] = round(
                    record["input_tokens"] * float(configured.llm_input_cost_per_1k) / 1000
                    + record["output_tokens"] * float(configured.llm_output_cost_per_1k) / 1000,
                    8,
                )

        if parsed.abstain:
            record["status"] = "abstained"
            record["reason_code"] = "verifier_abstained"
            return record
        roles = {item.id: item.role for item in parsed.roles}
        record["verifier_roles"] = roles
        record["verifier_reason_codes"] = {
            item.id: item.reason_code for item in parsed.roles
        }
        record["disagreement_span_ids"] = sorted(
            item_id for item_id, role in roles.items()
            if role != record["rule_roles"].get(item_id)
            and not (role == "unclear" and record["rule_roles"].get(item_id) == "unresolved")
        )
        record["status"] = "success"
        record["reason_code"] = "validated_output"
    except (ValidationError, ValueError, TypeError, KeyError):
        record["status"] = "invalid_output"
        record["reason_code"] = "output_contract_rejected"
    except VerifierUnavailable as exc:
        record["status"] = "timeout" if exc.reason_code == "timeout" else "provider_error"
        record["reason_code"] = exc.reason_code
    except Exception:  # noqa: BLE001 - failure is observational and fail closed
        record["status"] = "provider_error"
        record["reason_code"] = "verifier_call_failed"
    finally:
        record["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        _safe_event(record)
    return record


def risk_tier_for_compilation(compiled: dict[str, Any], state: dict[str, Any]) -> str:
    """Conservatively classify deterministic blockers before any verifier call."""
    blocked_reasons = {
        "date_outside_locked_scope", "region_filter_not_supported",
        "dimension_outside_locked_scope", "dimension_not_permitted",
        "dimension_not_supported_by_metric", "capability_not_enabled",
        "support_unknown", "authorization_denied", "seller_scope_denied",
        "seller_date_out_of_scope", "scope_outside_authorization",
    }
    reason = str(compiled.get("clarification_reason") or "")
    if reason in blocked_reasons or state.get("authorization_denied") is True:
        return "high"
    scope = state.get("controlled_analysis_scope") or {}
    allowed_dimensions = scope.get("allowed_dimensions") if isinstance(scope, dict) else getattr(scope, "allowed_dimensions", ())
    if allowed_dimensions is not None:
        allowed = {str(item) for item in allowed_dimensions}
        for item in compiled.get("dimension_roles") or []:
            dimension = str(item.get("dimension") or "")
            if dimension and dimension not in allowed:
                return "high"
    preflight = trusted_authorization_preflight(compiled, state)
    if not preflight[0]:
        return "high"

    risk_rank = {"low": 0, "open": 1, "elevated": 2, "conflict": 3, "high": 4}
    candidates = ["low"]
    explicit = str(state.get("semantic_verifier_risk_tier") or "")
    if explicit in risk_rank:
        candidates.append(explicit)
    assignments = compiled.get("dimension_roles") or []
    candidates.extend(
        str(item.get("risk_tier")) for item in assignments
        if str(item.get("risk_tier")) in risk_rank
    )
    scope_type = str(scope.get("scope_type") or ("seller" if scope.get("seller_id") else "platform")) if isinstance(scope, dict) else "platform"
    if scope_type == "seller" and any(
        item.get("dimension") == "seller" and item.get("role") == "grouping"
        for item in assignments
    ):
        candidates.append("elevated")
    if any(bool(item.get("conflict")) for item in assignments):
        candidates.append("conflict")
    if any(item.get("confidence_class") in {"open", "unresolved"} for item in assignments):
        candidates.append("open")
    return max(candidates, key=risk_rank.__getitem__)


def _as_scope_mapping(scope: Any) -> dict[str, Any]:
    if isinstance(scope, dict):
        return scope
    model_dump = getattr(scope, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
        return value if isinstance(value, dict) else {}
    return {}


def _date_bounds(value: Any) -> tuple[date, date] | None:
    matches = re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", str(value or ""))
    if not matches:
        return None
    try:
        parsed = [date.fromisoformat(item) for item in matches]
    except ValueError:
        return None
    start, end = min(parsed), max(parsed)
    return (start, end) if start <= end else None


def trusted_authorization_preflight(
    compiled: dict[str, Any],
    state: dict[str, Any],
) -> tuple[bool, str | None, Any | None]:
    """Recheck trusted identity, dimension, seller and date grants before P3 IO.

    The function reads tenant identity only from the active server context and
    accepts an AuthorizationContext in graph state only when it is the real
    immutable authorization snapshot bound to that tenant.
    """
    from app.auth.context import AuthorizationContext, build_authorization_context
    from app.tenancy import TenantContext, current_tenant_context

    tenant = current_tenant_context()
    if not isinstance(tenant, TenantContext):
        return False, "trusted_tenant_missing", None
    authz = state.get("authorization_context")
    if authz is not None and not isinstance(authz, AuthorizationContext):
        return False, "trusted_authorization_missing", None
    try:
        if authz is None:
            authz = build_authorization_context(tenant)
        if (
            authz.tenant_id != tenant.tenant_id
            or authz.user_id != tenant.user_id
            or authz.scope_type != tenant.scope_type
        ):
            return False, "authorization_identity_mismatch", None
        tenant_grants = {
            item.seller_id: (item.start_date, item.end_date)
            for item in tenant.seller_grants
        }
        authz_grants = {
            item.seller_id: (item.start_date, item.end_date)
            for item in authz.seller_grants
        }
        if (
            set(authz.allowed_seller_ids) != set(tenant.allowed_seller_ids)
            or authz_grants != tenant_grants
        ):
            return False, "seller_grant_snapshot_mismatch", None

        dimensions = [
            str(item.get("dimension") or "")
            for item in compiled.get("dimension_roles") or []
            if item.get("dimension")
        ]
        authz.require_dimensions(dimensions)

        scope = _as_scope_mapping(state.get("controlled_analysis_scope"))
        payload = state.get("seller_scope_payload")
        seller_ids: tuple[str, ...] = ()
        start: date | None = None
        end: date | None = None

        if payload:
            from app.scope.seller_scope import resolve_seller_scope, verify_seller_scope_payload

            intent = verify_seller_scope_payload(payload, tenant=tenant)
            resolve_seller_scope(intent, tenant=tenant, authz=authz)
            seller_ids = tuple(intent.seller_ids)
            start, end = intent.start_date, intent.end_date
            date_slot = ((compiled.get("known_slots") or {}).get("date") or {}).get("value")
            requested_dates = _date_bounds(date_slot)
            if date_slot and requested_dates is None:
                return False, "seller_date_invalid", None
            if requested_dates:
                if requested_dates[0] < start or requested_dates[1] > end:
                    return False, "seller_date_out_of_scope", None
                start, end = requested_dates

            # A second, legacy Explorer scope may only narrow the signed scope.
            if scope:
                legacy_ids = {
                    str(item) for item in (scope.get("seller_ids") or ()) if str(item)
                }
                if scope.get("seller_id"):
                    legacy_ids.add(str(scope["seller_id"]))
                if intent.seller_scope_type == "all" and legacy_ids:
                    return False, "seller_scope_mismatch", None
                if legacy_ids and not legacy_ids.issubset(set(seller_ids)):
                    return False, "seller_scope_mismatch", None
                locked = _date_bounds(
                    f"{scope.get('default_start_date', '')} {scope.get('default_end_date', '')}"
                )
                if locked and (start < locked[0] or end > locked[1]):
                    return False, "seller_date_out_of_scope", None
        else:
            raw_ids = scope.get("seller_ids") or ()
            seller_ids = tuple(sorted({str(item) for item in raw_ids if str(item)}))
            if scope.get("seller_id"):
                seller_id = str(scope["seller_id"])
                if seller_ids and seller_id not in seller_ids:
                    return False, "seller_scope_mismatch", None
                seller_ids = (seller_id,)
            elif not seller_ids and tenant.scope_type == "seller":
                seller_ids = tuple(authz.allowed_seller_ids)

            date_slot = ((compiled.get("known_slots") or {}).get("date") or {}).get("value")
            requested_dates = _date_bounds(date_slot)
            if date_slot and requested_dates is None and seller_ids:
                return False, "seller_date_invalid", None
            if requested_dates:
                start, end = requested_dates
            elif scope.get("default_start_date") and scope.get("default_end_date"):
                locked = _date_bounds(
                    f"{scope['default_start_date']} {scope['default_end_date']}"
                )
                if locked is None:
                    return False, "seller_date_invalid", None
                start, end = locked
            elif scope and seller_ids:
                return False, "seller_date_missing", None

        if seller_ids:
            authz.require_sellers(seller_ids)
            if start is not None and end is not None:
                for seller_id in seller_ids:
                    authz.require_date_range(seller_id, start, end)
            if scope and scope.get("default_start_date") and scope.get("default_end_date"):
                locked = _date_bounds(
                    f"{scope['default_start_date']} {scope['default_end_date']}"
                )
                if locked and start is not None and end is not None:
                    if start < locked[0] or end > locked[1]:
                        return False, "seller_date_out_of_scope", None
        return True, None, authz
    except Exception:  # noqa: BLE001 - any malformed/denied scope fails closed
        return False, "seller_or_date_authorization_denied", None


def observe_compilation(state: dict[str, Any], compiled: dict[str, Any], question: str) -> dict[str, Any] | None:
    """Observe compiler roles without mutating compiled output or contract."""
    from app.core.config import get_settings

    settings = get_settings()
    mode = str(settings.semantic_verifier_mode or "off")
    if mode == "off":
        return None
    risk_tier = risk_tier_for_compilation(compiled, state)
    assignments = compiled.get("dimension_roles") or []
    # The compiler role list is the authoritative L1 span inventory. Keep stable
    # IDs aligned with it and include only eligible spans in verifier output.
    mentions = []
    proposed = []
    eligible_ids = []
    for index, item in enumerate(assignments):
        start, end = int(item.get("start") or 0), int(item.get("end") or 0)
        if end <= start:
            continue
        item_id = mention_id(len(mentions))
        raw_text = question[start:end]
        mentions.append({
            "id": item_id, "raw_text": raw_text, "start": start, "end": end,
            "canonical": str(item.get("dimension") or ""),
        })
        proposed.append({
            "id": item_id, "role": str(item.get("role") or "unresolved"),
            "start": start, "end": end,
            "evidence": coarse_evidence_codes(item.get("evidence")),
        })
        if (
            risk_tier in {"elevated", "conflict"}
            or item.get("confidence_class") in {"open", "unresolved"}
            or item.get("conflict")
        ):
            eligible_ids.append(item_id)
    if risk_tier in {"elevated", "open", "conflict"} and eligible_ids:
        try:
            preflight_ok, _preflight_reason, authz = trusted_authorization_preflight(compiled, state)
            if not preflight_ok or authz is None:
                risk_tier = "high"
            else:
                mention_by_id = {item["id"]: item for item in mentions}
                metric = ((compiled.get("known_slots") or {}).get("metric") or {}).get("value")
                operation_intent = str((compiled.get("operation_intent") or {}).get("kind") or "")
                operation = "contribution" if operation_intent == "dimensional_statistical_contribution" else "breakdown"
                from app.context.support_resolver import resolve_support

                for item_id in eligible_ids:
                    dimension = str(mention_by_id[item_id]["canonical"])
                    authz.require_dimensions([dimension])
                    support = resolve_support(
                        metric_id=str(metric) if metric else None,
                        dimension=dimension,
                        operation=operation,
                        scope=state.get("controlled_analysis_scope"),
                        authz=authz,
                    )
                    if not support.executable or support.user_permitted != "ok":
                        risk_tier = "high"
                        break
        except Exception:  # noqa: BLE001 - missing/denied support prevents external call
            risk_tier = "high"
    record = verify_roles(
        question=question,
        mentions=mentions,
        proposed_roles=proposed,
        eligible_ids=eligible_ids,
        risk_tier=risk_tier,
        mode=mode,
        model=str(settings.semantic_verifier_model or settings.llm_model),
        allow_raw_question=bool(settings.semantic_verifier_send_raw_question),
    )
    return record if mode != "off" else None
