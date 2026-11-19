"""Compile a user turn with narrowly-scoped, traceable prior-turn context.

The compiler resolves an unambiguous metric reference from same-scope chat
history, or falls back to the server-validated Explorer metric when no chat
history exists. Dates, sellers, permissions, and query scope are never derived
from conversation text; those remain owned by the governed scope and planning
layers.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from app.graph.state import AgentState
from app.observability import record_event, record_measurement
from app.core.time_expression import extract_iso_dates
from app.context.request_contract import seal_request_contract
from app.context.negative_intent import (
    default_query_aliases,
    mask_spans,
    parse_negative_intents,
    span_is_negated,
)
from app.context.dimension_intent import (
    RoleAssignment,
    aggregate_roles,
    classify_dimension_mentions,
    conflict_dimensions,
)
from app.context.operation_intent import is_explicit_compare_request
from app.context.result_context import detect_result_followup


_REFERENCE_PATTERNS = (
    re.compile(
        r"上述指标|前面那个指标|这个指标|该指标|这项指标|这个口径|那项指标|同一天|当天|当日|这一天|那天|同一日期|"
        r"(?:和|与|跟)\s*20\d{2}\s*年\s*\d{1,2}\s*月\s*(?:相比|对比|相较)|"
        r"刚才(?:查询的|的)?|上一轮(?:的)?(?:查询|结果|回答)|上轮(?:的)?(?:查询|结果|回答)|"
        r"^那(?=按|拆|再|继续|还有|呢)|^那(?=.{1,8}呢)|^(?=(?:按|拆开|拆分|继续看|继续|再看|换成))|"
        r"这个(?!月|周|季度|年|店|平台|商品|类目|日期|范围|时间|订单)|"
        r"那个(?!月|周|季度|年)|它|其(?!他)|\bit\b|\bthat\b",
        re.IGNORECASE,
    ),
)
_COMPARISON_CHANGE_REFERENCE = re.compile(
    r"(?:这个变化|该变化|上述变化|刚才的变化|上一轮(?:的)?(?:比较|对比|变化|结果)?|"
    r"上轮(?:的)?(?:比较|对比|变化|结果)?)"
)
_EXPLICIT_RELATIVE_PERIOD = re.compile(
    r"(?:今天|昨天|前天|明天|当天|当日|本周|这周|上周|上星期|本月|这个月|上月|上个月|"
    r"本季度|这个季度|上季度|今年|去年|前年|最近\s*\d+\s*(?:天|周|星期|个月|月|季度|年)|"
    r"近\s*\d+\s*(?:天|周|星期|个月|月|季度|年))"
)

_EXPLICIT_DEFINITION_MARKERS = ("定义", "怎么算", "计算方式", "计算公式", "限制", "含义", "是什么", "how is", "definition", "limitations")
_GENERIC_DEFINITION_MARKERS = ("口径",)
_DATA_MARKERS = ("趋势", "下降", "上升", "拆开", "拆分", "按", "对比", "变化", "多少", "走势", "breakdown", "trend", "compare")


def _contract_dimension_ids(contract: Mapping[str, object]) -> set[str]:
    """Return dimensions represented by either the legacy or multi-axis contract shape.

    Contract v1 carries one ``dimension``.  Accepting a future ``dimensions``
    list here keeps the comparison-consumption guard reusable if a later
    contract version adds multiple authorized axes.
    """
    result: set[str] = set()
    single = contract.get("dimension")
    if isinstance(single, str) and single.strip():
        result.add(single.strip())

    dimensions = contract.get("dimensions")
    if isinstance(dimensions, (list, tuple)):
        for item in dimensions:
            if isinstance(item, str) and item.strip():
                result.add(item.strip())
            elif isinstance(item, Mapping):
                for key in ("dimension_id", "dimension", "id"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        result.add(value.strip())
                        break
    return result


def _unrepresented_comparison_dimensions(
    assignments: Iterable[RoleAssignment],
    contract: Mapping[str, object],
) -> tuple[str, ...]:
    """Find comparison-role dimensions omitted from the executable contract."""
    represented = _contract_dimension_ids(contract)
    mentioned = {
        assignment.dimension
        for assignment in assignments
        if assignment.role == "comparison"
    }
    return tuple(sorted(mentioned - represented))


def _comparison_date_roles(question: str) -> tuple[str, str] | None:
    """Resolve baseline/current once, at the user-input compilation boundary.

    Explicitly labeled baseline/current dates take precedence. An unlabeled
    two-date comparison uses chronological roles (earlier=baseline,
    later=current). Downstream layers consume these roles, not token order or
    legacy left/right fields.
    """
    raw_dates = extract_iso_dates(question)
    if len(raw_dates) != 2 or raw_dates[0] == raw_dates[1]:
        return None
    try:
        parsed = [(date.fromisoformat(value), value) for value in raw_dates]
    except ValueError:
        return None

    explicit: dict[str, str] = {}
    role_conflict = False
    for role, pattern in (
        ("baseline_date", r"(?:baseline(?:\s+date|\s+period)?|基线(?:日期|期))\s*[:：]?\s*(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)"),
        ("current_date", r"(?:current(?:\s+date|\s+period)?|当前(?:日期|期))\s*[:：]?\s*(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)"),
    ):
        matches = [match.group(1) for match in re.finditer(pattern, question, flags=re.IGNORECASE)]
        matches = [value for value in matches if value in raw_dates]
        if len(matches) > 1:
            role_conflict = True
        elif matches:
            explicit[role] = matches[0]
    if role_conflict:
        return None
    if len(explicit) == 2 and explicit["baseline_date"] != explicit["current_date"]:
        return explicit["baseline_date"], explicit["current_date"]
    if len(explicit) == 2:
        # Contradictory explicit labels are not overridden by chronological
        # sorting; leave the roles unresolved for a safe clarification path.
        return None
    if len(explicit) == 1:
        assigned_role, assigned_date = next(iter(explicit.items()))
        other_date = next((value for value in raw_dates if value != assigned_date), None)
        if other_date is None:
            return None
        if assigned_role == "baseline_date":
            return assigned_date, other_date
        return other_date, assigned_date

    ordered = sorted(parsed, key=lambda item: item[0])
    if ordered[0][0] == ordered[1][0]:
        return None
    return ordered[0][1], ordered[1][1]


def _month_tokens_with_inherited_year(question: str) -> list[tuple[int, int]]:
    """Resolve month mentions in text order, inheriting a nearby explicit year.

    ``2018 年 7 月和 8 月`` names one year once and then repeats only the month.
    The previous parser required every month to carry its own year and therefore
    left the second period incomplete.
    """

    explicit = [
        (match.start(), match.end(), int(match.group(1)), int(match.group(2)))
        for match in re.finditer(
            r"(?<!\d)(20\d{2})\s*年\s*(\d{1,2})\s*月", question,
        )
    ]
    year_anchors = [
        (match.start(), int(match.group(1)))
        for match in re.finditer(r"(?<!\d)(20\d{2})\s*年", question)
    ]
    resolved: list[tuple[int, int, int]] = [
        (start, year, month) for start, _end, year, month in explicit
    ]
    for match in re.finditer(r"(?<!\d)(\d{1,2})\s*月", question):
        if any(start <= match.start() < end for start, end, _year, _month in explicit):
            continue
        preceding = [item for item in year_anchors if item[0] < match.start()]
        if not preceding:
            continue
        resolved.append((match.start(), preceding[-1][1], int(match.group(1))))
    resolved.sort(key=lambda item: item[0])
    return [(year, month) for _position, year, month in resolved]


_CAUSAL_MARKER_NEGATION_RE = re.compile(
    r"(?:不要|别|勿|无需|不必|不用|不再)"
    r"[^，。；！？,.;!?]{0,8}$|"
    r"不(?:再)?\s*(?:推断|分析|解释|讨论|说明|涉及|进行|做|谈|下结论|判断|归因|"
    r"考虑|看|讲|提|说|把|将)[^，。；！？,.;!?]{0,8}$"
)


def _marker_is_locally_negated(text: str, start: int) -> bool:
    """Whether a marker at ``start`` sits inside an explicit do-not clause."""

    return bool(_CAUSAL_MARKER_NEGATION_RE.search(text[max(0, start - 16):start]))


def _has_unnegated_marker(text: str, markers: tuple[str, ...]) -> bool:
    folded = text.casefold()
    for marker in markers:
        needle = marker.casefold()
        cursor = 0
        while True:
            index = folded.find(needle, cursor)
            if index < 0:
                break
            if not _marker_is_locally_negated(folded, index):
                return True
            cursor = index + len(needle)
    return False


def _comparison_resolution(question: str) -> tuple[dict[str, object], str, str | None, list[str]]:
    """Resolve the comparison shape once, using the shared ISO token parser."""
    tokens = extract_iso_dates(question)
    invalid = []
    for token in tokens:
        try:
            date.fromisoformat(token)
        except ValueError:
            invalid.append(token)
    if invalid:
        return ({"kind": "day_pair", "baseline_date": None, "current_date": None},
                "clarification_required", "invalid_comparison_date", tokens)

    month_tokens = _month_tokens_with_inherited_year(question)
    if any(month < 1 or month > 12 for _, month in month_tokens):
        return ({"kind": "period_pair", "granularity": "month"},
                "clarification_required", "invalid_comparison_period", tokens)

    if month_tokens and tokens:
        return ({"kind": "unresolved", "date_tokens": tokens},
                "clarification_required", "comparison_shape_conflict", tokens)
    month_unique = list(dict.fromkeys(month_tokens))
    if len(month_unique) == 2 and len(month_tokens) == 2:
        from calendar import monthrange

        # Explicit baseline/current labels take precedence over text order.
        month_values = [f"{year:04d}-{month:02d}" for year, month in month_tokens]
        explicit_roles: dict[str, str] = {}
        role_patterns = (
            ("baseline", r"(?:baseline(?:\s+period)?|基线(?:日期|期间|期))"),
            ("current", r"(?:current(?:\s+period)?|当前(?:日期|期间|期))"),
        )
        for role, label_pattern in role_patterns:
            for match in re.finditer(label_pattern, question, flags=re.IGNORECASE):
                tail = question[match.end():match.end() + 36]
                nearby = re.search(r"(?<!\d)(20\d{2})\s*年\s*(\d{1,2})\s*月", tail)
                if nearby:
                    explicit_roles[role] = f"{int(nearby.group(1)):04d}-{int(nearby.group(2)):02d}"
                    break
        if len(explicit_roles) == 2:
            baseline, current = explicit_roles["baseline"], explicit_roles["current"]
            if baseline == current or baseline not in month_values or current not in month_values:
                return ({"kind": "period_pair", "granularity": "month"},
                        "clarification_required", "comparison_period_labels_conflict", tokens)
        elif len(explicit_roles) == 1:
            assigned = next(iter(explicit_roles.values()))
            other = next((value for value in month_values if value != assigned), None)
            if other is None:
                return ({"kind": "period_pair", "granularity": "month"},
                        "clarification_required", "comparison_periods_incomplete", tokens)
            if "baseline" in explicit_roles:
                baseline, current = assigned, other
            else:
                baseline, current = other, assigned
        else:
            baseline, current = sorted(month_values)

        def period(month_text: str) -> dict[str, str]:
            year, month = (int(part) for part in month_text.split("-"))
            return {
                "start_date": f"{year:04d}-{month:02d}-01",
                "end_date": f"{year:04d}-{month:02d}-{monthrange(year, month)[1]:02d}",
            }

        return ({"kind": "period_pair", "baseline": period(baseline),
                 "current": period(current), "granularity": "month"},
                "resolved", None, tokens)
    if len(month_tokens) > 0 and len(month_unique) == 1:
        return ({"kind": "period_pair", "granularity": "month"},
                "clarification_required", "comparison_periods_incomplete", tokens)
    if len(month_tokens) > 0 and len(month_tokens) != 2:
        return ({"kind": "period_pair", "granularity": "month"},
                "clarification_required", "comparison_periods_ambiguous", tokens)

    if len(tokens) == 2 and tokens[0] != tokens[1]:
        roles = _comparison_date_roles(question)
        if roles:
            return ({"kind": "day_pair", "baseline_date": roles[0], "current_date": roles[1]},
                    "resolved", None, tokens)
        return ({"kind": "day_pair", "baseline_date": None, "current_date": None},
                "clarification_required", "comparison_date_labels_conflict", tokens)
    reason = "comparison_dates_incomplete" if len(tokens) < 2 else "comparison_dates_ambiguous"
    return ({"kind": "day_pair", "baseline_date": None, "current_date": None},
            "clarification_required", reason, tokens)


def _single_explicit_month_period(question: str) -> dict[str, str] | None:
    """Return a complete calendar month only for one explicit month mention."""
    tokens = re.findall(r"(?<!\d)(20\d{2})\s*年\s*(\d{1,2})\s*月", question)
    if len(tokens) != 1 or extract_iso_dates(question):
        return None
    year, month = (int(value) for value in tokens[0])
    if not 1 <= month <= 12:
        return None
    from calendar import monthrange

    return {
        "start_date": f"{year:04d}-{month:02d}-01",
        "end_date": f"{year:04d}-{month:02d}-{monthrange(year, month)[1]:02d}",
    }


def _has_explicit_current_period(
    question: str,
    *,
    iso_tokens: list[str],
    explicit_month_period: dict[str, str] | None,
) -> bool:
    """Do not inherit a prior comparison when the current turn names time."""
    return bool(
        iso_tokens
        or explicit_month_period
        or re.search(r"(?<!\d)20\d{2}\s*年", question)
        or _EXPLICIT_RELATIVE_PERIOD.search(question)
    )


def _is_period_comparison_followup(question: str) -> bool:
    """Recognize a comparison follow-up naming just one explicit month."""
    month = r"(?:20\d{2}\s*年\s*\d{1,2}\s*月)"
    patterns = (
        rf"(?:和|与|跟)\s*{month}\s*(?:相比|对比|相较)",
        rf"(?:对比|比较)\s*{month}",
        rf"比\s*{month}\s*(?:呢|如何|怎么样|相比)?",
    )
    return any(re.search(pattern, question) for pattern in patterns)


def _requires_exact_date_anchor(question: str) -> bool:
    """Whether a day-reference must resolve to one exact prior/current date."""
    return any(token in question for token in ("当天", "当日", "同一天", "这一天", "那天"))


def _dimension_alias_matches(
    question: str, *, include_unregistered: bool = False,
) -> list[tuple[str, str, int, int]]:
    """Find canonical dimensions, preferring a complete longer alias phrase."""
    aliases = {
        "product_category": ("商品品类", "商品类别", "产品品类", "品类", "类目", "product_category", "product category", "category"),
        "product": ("商品", "产品", "product"),
        "seller": ("店铺", "商家", "seller"),
        # The Registry has no generic `region` dimension: these aliases map
        # to the existing customer-state grouping contract, which is checked
        # against the selected metric/scope before any tool call.
        "customer_state": ("客户州", "顾客州", "州", "省份", "地区", "区域", "region", "state", "customer_state", "customer state"),
        "payment_type": ("支付方式", "payment_type", "payment type"),
        "month_position": ("month_position", "month position", "月初", "月中", "月末"),
        "customer_city": ("客户城市", "顾客城市", "customer_city", "customer city"),
        "seller_state": ("卖家州", "商家州", "seller_state", "seller state"),
    }
    registered: set[str] = set()
    try:
        from app.metrics.registry import get_registry
        registry = get_registry()
        for definition in get_registry().list():
            for implementation in definition.implementations.values():
                registered.update(
                    str(value) for value in implementation.dimensions
                    if registry.dimension_enabled(str(value))
                )
            for capability in definition.attribution.scopes.values():
                registered.update(str(value) for value in capability.dimensions)
    except Exception:
        registered = set(aliases)
    folded = question.casefold()
    matches: list[tuple[str, str, int, int]] = []
    for dimension, terms in aliases.items():
        if dimension not in registered and not include_unregistered:
            continue
        for alias in terms:
            start = folded.find(alias.casefold())
            while start >= 0:
                matches.append((dimension, alias, start, start + len(alias)))
                start = folded.find(alias.casefold(), start + 1)
    # A shorter alias contained in a more specific phrase is not a separate
    # request ("商品品类" should not also resolve as "商品").  Independent
    # mentions remain visible and can trigger clarification.
    return [
        item for item in matches
        if not any(
            other[2] <= item[2] and other[3] >= item[3]
            and other[3] - other[2] > item[3] - item[2]
            for other in matches
        )
    ]


def _metric_mention_spans(question: str) -> list[tuple[int, int]]:
    """Return exact registry metric spans for metric-noun role suppression."""

    try:
        from app.metrics.authorization import server_exact_mentions
        from app.metrics.registry import get_registry

        catalog = get_registry().planning_catalog(scope="platform")
        return [
            (int(item["start"]), int(item["end"]))
            for item in server_exact_mentions(question, catalog=catalog)
        ]
    except Exception:  # noqa: BLE001 - role classification still works locally
        return []


def _metric_labels() -> dict[str, str]:
    try:
        from app.metrics.registry import get_registry

        return {
            definition.metric_id: str(definition.display_name or definition.metric_id)
            for definition in get_registry().list()
        }
    except Exception:
        return {}


def _previous_metrics(state: AgentState, labels: dict[str, str]) -> list[dict[str, object]]:
    """Return deduplicated metric candidates from authorized context.

    Conversation State is preferred; a selected Task State's last structured
    slots are a fallback.  Both remain interpretation hints: data-query
    planning still validates the metric against the current registry and
    authorization boundary.
    """
    previous = _trusted_previous_turn_context(state)
    task_progress = _task_progress_for_slots(state)
    rows: list[tuple[dict, str, int | None]] = []
    for item in (previous.get("authorized_metrics") or []):
        if isinstance(item, dict):
            rows.append((item, "previous_turn_context", None))
        elif isinstance(item, str):
            rows.append(({"metric_id": item}, "previous_turn_context", None))
    prior_slot = (previous.get("context_slots") or {}).get("metric")
    if isinstance(prior_slot, dict) and prior_slot.get("value"):
        rows.append(({"metric_id": prior_slot["value"]}, "previous_turn_context", None))
    task_slot = (task_progress.get("context_slots") or {}).get("metric")
    if isinstance(task_slot, dict) and task_slot.get("value"):
        rows.append((
            {
                "metric_id": task_slot["value"],
                # Do not substitute the task's latest turn here.  An old
                # snapshot without a slot-level source Turn has unknown
                # provenance; the Task State version remains the provenance.
                "source_turn_id": task_slot.get("source_turn_id"),
            },
            "task_state",
            task_progress.get("task_state_version"),
        ))
    decision = previous.get("metric_planning_decision") or {}
    for metric_id in decision.get("confirmed_metrics") or []:
        rows.append(({"metric_id": metric_id}, "previous_turn_context", None))

    candidates: dict[str, dict[str, object]] = {}
    for row, source, task_version in rows:
        if isinstance(row, str):
            metric_id = row
            label = labels.get(metric_id, metric_id)
        elif isinstance(row, dict):
            metric_id = str(row.get("metric_id") or "").strip()
            label = str(row.get("display_name") or row.get("metric_name") or labels.get(metric_id) or metric_id).strip()
        else:
            continue
        if metric_id:
            candidates[metric_id] = {
                "metric_id": metric_id,
                "label": label or metric_id,
                "source": source,
                "source_turn_id": row.get("source_turn_id"),
                "source_task_state_version": task_version,
            }
    return list(candidates.values())


def _trusted_previous_turn_context(state: AgentState) -> dict[str, object]:
    """Return only the server-loaded predecessor for the current query scope.

    The chat workflow loads this object from the current tenant/user/session and
    filters it by the same scope fingerprint.  Re-check that binding here before
    using its slots.  Direct compiler callers without a scope fingerprint retain
    the legacy in-process behavior only when the supplied context has no scope
    identity; the authenticated workflow always supplies a fingerprint.
    """
    previous = state.get("previous_turn_context") or {}
    if not isinstance(previous, dict) or not previous:
        return {}

    boundary = state.get("authorization_boundary") or {}
    current_scope = str(
        state.get("current_query_scope_fingerprint")
        or (boundary.get("query_scope_fingerprint") if isinstance(boundary, dict) else "")
        or ""
    )
    previous_scope = str(previous.get("scope_fingerprint") or "")
    source_turn_id = str(previous.get("source_turn_id") or "")

    if current_scope:
        if not source_turn_id or previous_scope != current_scope:
            return {}
    elif previous_scope or source_turn_id:
        # A context that claims a persisted identity cannot be verified when
        # the current request has no corresponding scope identity.
        return {}

    snapshot = previous.get("source_context_snapshot") or {}
    turn_resolution = snapshot.get("turn_resolution") if isinstance(snapshot, dict) else None
    snapshot_turn_id = str(
        turn_resolution.get("turn_id") or ""
    ) if isinstance(turn_resolution, dict) else ""
    if snapshot_turn_id and source_turn_id and snapshot_turn_id != source_turn_id:
        return {}
    return previous


def _current_explorer_metric(state: AgentState, labels: dict[str, str]) -> dict[str, str] | None:
    """Use the server-validated Explorer metric when chat history has no metric.

    The raw browser payload is deliberately ignored. ``controlled_analysis_scope``
    is populated only after the resolver revalidates the Explorer request, and
    the selected metric must also be present in that scope's allowed registry
    metrics before it can resolve an elliptical reference.
    """
    if not state.get("use_current_explorer_context"):
        return None
    scope = state.get("controlled_analysis_scope") or {}
    if not isinstance(scope, dict):
        return None
    metric_id = str(scope.get("metric_name") or "").strip()
    allowed_metrics = {str(value).strip() for value in (scope.get("allowed_metrics") or [])}
    if not metric_id or metric_id not in labels or metric_id not in allowed_metrics:
        return None
    return {
        "metric_id": metric_id,
        "label": labels.get(metric_id, metric_id),
        "source": "current_explorer_scope",
    }


def _previous_explorer_metric(state: AgentState) -> str | None:
    """Return the selected metric stored with the prior turn's page scope."""
    previous = _trusted_previous_turn_context(state)
    snapshot = previous.get("source_context_snapshot") or {}
    scope = snapshot.get("scope") or {}
    # Lifecycle snapshots keep the selected UI metric next to the canonical
    # scope (whose metric_ids intentionally represent the full allowed set).
    # Older snapshots may place it inside ``scope``.
    metric_id = str(snapshot.get("metric_name") or scope.get("metric_name") or "").strip()
    return metric_id or None


def _explicit_metric(question: str, labels: dict[str, str]) -> dict[str, object] | None:
    """Find a registered metric explicitly named in the current message."""
    folded = question.casefold()
    candidates: list[tuple[int, int, str]] = []
    try:
        from app.metrics.registry import get_registry

        for definition in get_registry().list():
            metric_id = str(definition.metric_id)
            # Planning phrases and semantic terms support retrieval, but only
            # registered exact aliases can bind a deterministic metric.
            terms = (
                metric_id,
                definition.display_name,
                *(definition.planning.exact_aliases or definition.aliases),
            )
            for raw_term in terms:
                term = str(raw_term or "").strip()
                if not term:
                    continue
                needle = term.casefold()
                start = folded.find(needle)
                while start >= 0:
                    candidates.append((start, start + len(needle), metric_id))
                    start = folded.find(needle, start + 1)
    except Exception:
        for metric_id, label in labels.items():
            for term in (metric_id, label):
                if term and term.casefold() in folded:
                    start = folded.find(term.casefold())
                    candidates.append((start, start + len(term), metric_id))
    if not candidates:
        return None
    # Nested aliases describe one phrase. Distinct non-nested mentions are
    # ambiguous; never select one by dictionary order or alias length.
    salient = [
        item for item in candidates
        if not any(
            other[0] <= item[0] and other[1] >= item[1]
            and other[1] - other[0] > item[1] - item[0]
            for other in candidates
        )
    ]
    strongest = sorted({item[2] for item in salient})
    if len(strongest) != 1:
        return {
            "ambiguous_candidates": [
                {"metric_id": metric_id, "label": labels.get(metric_id, metric_id)}
                for metric_id in strongest
            ],
            "source": "current_turn",
        }
    metric_id = strongest[0]
    return {"metric_id": metric_id, "label": labels.get(metric_id, metric_id), "source": "current_turn"}


def _intent(question: str) -> str:
    folded = question.casefold()
    # Explicit data actions take precedence over the generic word "口径".
    # A phrase such as "按注册口径比较两期" asks for data, while "GMV口径是什么"
    # asks for a definition. Strong definition wording still wins when present.
    if any(marker in folded for marker in _EXPLICIT_DEFINITION_MARKERS):
        return "retrieve_metric_definition"
    if any(marker in folded for marker in _DATA_MARKERS):
        return "analyze_business_data"
    if any(marker in folded for marker in _GENERIC_DEFINITION_MARKERS):
        return "retrieve_metric_definition"
    return "follow_up"


def _is_personalisation_request(question: str) -> bool:
    folded = question.casefold()
    return any(marker in folded for marker in (
        "按我平时", "我平时的", "我的偏好", "我的习惯", "按我的习惯", "按我习惯",
        "我通常", "按我通常", "老规矩", "上次约定", "按上次约定",
        "my preference", "my usual", "as usual", "my habit",
    ))


def _replace_reference(question: str, label: str) -> tuple[str, str] | None:
    for pattern in _REFERENCE_PATTERNS:
        match = pattern.search(question)
        if match:
            return question[:match.start()] + label + question[match.end():], match.group(0)
    return None


_CORRECTION_MARKERS = (
    "换成", "改为", "改成", "调整为", "转到", "切换到", "纠正", "更正", "不再",
    "instead", "change to", "switch to",
)
_MONTH_REPLACEMENT = re.compile(
    r"(?:换成|改为|改成|调整为|转到|切换到|change\s+to|switch\s+to)\s*"
    r"(?:(20\d{2})\s*年)?\s*(\d{1,2})\s*月",
    re.IGNORECASE,
)
_REGION_SCOPE_TOKENS = (
    "华东", "华南", "华北", "华中", "西南", "西北", "东北",
    "港澳台", "香港", "澳门", "台湾",
)


def _task_progress_for_slots(state: AgentState) -> dict:
    """Return Task State slots only when their scope matches the current turn.

    Task identity may survive a scope change, but inherited values may not.  A
    mismatched or unversioned snapshot is useful only as historical context;
    treating its slots as the current request would silently switch scope.
    """
    progress = state.get("task_progress_context") or {}
    if not isinstance(progress, dict):
        return {}
    current_scope = str(state.get("current_query_scope_fingerprint") or "")
    task_scope = str(progress.get("scope_fingerprint") or "")
    if not current_scope or not task_scope or current_scope != task_scope:
        return {}
    current_version = str(state.get("data_version_manifest_hash") or "")
    task_version = str(progress.get("data_version_manifest_hash") or "")
    if not current_version or not task_version or current_version != task_version:
        return {}
    return progress


def _prior_slot_value(state: AgentState, name: str) -> object:
    for container in (
        _trusted_previous_turn_context(state),
        _task_progress_for_slots(state),
    ):
        slot = (container.get("context_slots") or {}).get(name)
        if isinstance(slot, dict) and slot.get("value") not in (None, ""):
            return slot["value"]
    return None


def _month_replacement(question: str, prior_date: object) -> str | None:
    match = _MONTH_REPLACEMENT.search(question)
    if not match:
        return None
    year = match.group(1)
    if not year:
        prior_match = re.match(r"(?<!\d)(20\d{2})-\d{2}-\d{2}", str(prior_date or ""))
        year = prior_match.group(1) if prior_match else None
    if not year:
        return None
    from calendar import monthrange

    month = int(match.group(2))
    if month < 1 or month > 12:
        return None
    year_int = int(year)
    return (
        f"{year_int:04d}-{month:02d}-01 至 "
        f"{year_int:04d}-{month:02d}-{monthrange(year_int, month)[1]:02d}"
    )


def _date_bounds(value: object) -> tuple[date, date] | None:
    tokens = extract_iso_dates(str(value or ""))
    if not tokens:
        return None
    try:
        parsed = [date.fromisoformat(token) for token in tokens]
    except ValueError:
        return None
    return min(parsed), max(parsed)


def _date_outside_scope(value: object, state: AgentState) -> bool:
    scope = state.get("controlled_analysis_scope") or {}
    if not isinstance(scope, dict):
        return False
    start = str(scope.get("default_start_date") or "")
    end = str(scope.get("default_end_date") or "")
    if not start or not end:
        return False
    requested = _date_bounds(value)
    if requested is None:
        return False
    try:
        scope_start, scope_end = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError:
        return False
    return requested[0] < scope_start or requested[1] > scope_end


def _comparison_outside_scope(comparison: dict, state: AgentState) -> bool:
    """Check explicit or inherited comparison endpoints against locked dates."""
    scope = state.get("controlled_analysis_scope") or {}
    if not isinstance(scope, dict):
        return False
    try:
        scope_start = date.fromisoformat(str(scope.get("default_start_date") or ""))
        scope_end = date.fromisoformat(str(scope.get("default_end_date") or ""))
    except ValueError:
        return False
    if scope_end < scope_start:
        return False

    endpoints: list[date] = []
    kind = comparison.get("kind")
    if kind == "day_pair":
        for key in ("baseline_date", "current_date"):
            raw = comparison.get(key)
            if not isinstance(raw, str):
                continue
            try:
                endpoints.append(date.fromisoformat(raw))
            except ValueError:
                continue
    elif kind == "period_pair":
        for role in ("baseline", "current"):
            period = comparison.get(role)
            if not isinstance(period, dict):
                continue
            for key in ("start_date", "end_date"):
                raw = period.get(key)
                if not isinstance(raw, str):
                    continue
                try:
                    endpoints.append(date.fromisoformat(raw))
                except ValueError:
                    continue
    return bool(endpoints and (min(endpoints) < scope_start or max(endpoints) > scope_end))


def _append_inherited_date(question: str, value: object) -> str:
    date_text = str(value or "").strip()
    if not date_text or date_text in question:
        return question
    if question.rstrip().endswith(("？", "?", "。", "！", "!")):
        return f"{question.rstrip()}（日期 {date_text}）"
    return f"{question}，日期 {date_text}"


def _reject_inherited_slot(
    compiled: dict[str, object], *, name: str, value: object, source: str, reason: str,
) -> None:
    compiled["scope_candidates"].append({
        "field": name,
        "value": value,
        "source": source,
        "status": "rejected",
        "reason": reason,
    })
    compiled["needs_clarification"] = True
    compiled["clarification_reason"] = reason
    missing_slots = compiled["missing_slots"]
    if name not in missing_slots:
        missing_slots.append(name)


def _detect_region_scope_candidates(question: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for token in _REGION_SCOPE_TOKENS:
        if token in question:
            candidates.append({
                "field": "region",
                "value": token,
                "source": "current_turn",
                "status": "unsupported",
                "reason": "region_filter_not_supported_by_current_scope",
            })
    return candidates


def _user_changes(
    question: str,
    compiled_slots: dict[str, dict],
    state: AgentState,
) -> list[dict[str, object]]:
    if not any(marker.casefold() in question.casefold() for marker in _CORRECTION_MARKERS):
        return []
    changes: list[dict[str, object]] = []
    for name in ("metric", "date", "dimension", "operation"):
        slot = compiled_slots.get(name) or {}
        if slot.get("source") not in {"current_turn", "current_turn_correction"}:
            continue
        prior = _prior_slot_value(state, name)
        if prior in (None, ""):
            continue
        if str(prior) == str(slot.get("value")):
            continue
        changes.append({
            "field": name,
            "from": prior,
            "to": slot.get("value"),
            "source": slot.get("source"),
            "reason": "explicit_user_correction",
        })
    return changes


def turn_resolution_payload(state: AgentState, compilation: dict) -> dict:
    """Build the WP1 TurnResolution persisted in chat_turns.context_snapshot.

    This is a projection of the existing compiler output, not a second
    resolver.  It is the object referenced by ContextBundle.resolved_request.
    """
    from app.context.turn_resolver import client_turn_id, durable_turn_id

    known_slots = compilation.get("known_slots") or {}

    explicit_fields: list[str] = []
    inherited_fields: list[dict] = []
    for name, slot in known_slots.items():
        if not isinstance(slot, dict) or not slot.get("value"):
            continue
        source = str(slot.get("source") or "")
        if source in {"current_turn", "current_turn_correction", "current_explorer_scope"}:
            explicit_fields.append(str(name))
        elif source in {"previous_turn_context", "task_state"}:
            inherited = {
                "field": str(name),
                "value": slot.get("value"),
                "source": source,
                "source_turn_id": slot.get("source_turn_id"),
                "source_turn_id_status": (
                    "known" if slot.get("source_turn_id") else "unknown"
                ),
                "source_task_state_version": slot.get("source_task_state_version"),
                "confidence": slot.get("confidence"),
            }
            inherited_fields.append(inherited)
    confidences = [
        float(slot.get("confidence") or 0.0)
        for slot in known_slots.values()
        if isinstance(slot, dict) and slot.get("confidence") is not None
    ]
    return {
        "schema_version": "turn-resolution-v1",
        "turn_id": durable_turn_id(state),
        "client_turn_id": client_turn_id(state),
        "session_id": str(state.get("chat_session_id") or state.get("memory_session_id") or ""),
        "task_id": str(state["task_id"]) if state.get("task_id") else None,
        "original_text": str(state.get("user_message") or ""),
        "resolved_text": str(compilation.get("standalone_question") or state.get("user_message") or ""),
        "relation": (
            "clarify" if compilation.get("needs_clarification")
            else "continue" if compilation.get("relation") == "continue"
            else "new"
        ),
        "intent": str(compilation.get("intent") or ""),
        "explicit_fields": explicit_fields,
        "inherited_fields": inherited_fields,
        "user_changes": list(compilation.get("user_changes") or []),
        "unresolved_fields": list(compilation.get("missing_slots") or []),
        "known_slots": known_slots,
        "missing_slots": list(compilation.get("missing_slots") or []),
        "confidence": min(confidences) if confidences else 1.0,
        "needs_clarification": bool(compilation.get("needs_clarification")),
        "clarification_reason": compilation.get("clarification_reason"),
        "source_turn_id": compilation.get("source_turn_id"),
        "scope_fingerprint": str(
            compilation.get("scope_fingerprint")
            or state.get("current_query_scope_fingerprint")
            or ""
        ),
        "request_contract_hash": str(compilation.get("request_contract_hash") or ""),
        "resolver_version": "context-compiler-v1",
        "policy_version": "wp1",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_p0_audit(
    *,
    original: str,
    negative_query_spans: dict,
    dimension_matches: list[tuple[str, str, int, int]],
    explicit: dict | None,
    comparison_shape: dict,
    operation_intent: dict,
    contribution_requested_textually: bool,
    compiled: dict,
    state: AgentState,
) -> dict:
    """Record P0 observations without feeding them back into business parsing.

    The audit deliberately re-scans the original text with unregistered
    dimensions enabled.  Its output is stored next to ``context_compilation``
    only; it is never read by the routing, contract or support layers.
    """

    started = time.perf_counter()
    audit_matches = _dimension_alias_matches(original, include_unregistered=True)
    registered_matches = _dimension_alias_matches(original)
    registered_dims = sorted({item[0] for item in registered_matches})
    all_dims = sorted({item[0] for item in audit_matches})
    unregistered_dims = sorted(set(all_dims) - set(registered_dims))
    business_dims = sorted({item[0] for item in dimension_matches})

    dimension_mentions: list[dict[str, object]] = []
    for dimension, _alias, start, end in audit_matches:
        dimension_mentions.append({
            "kind": "dimension",
            "canonical": dimension,
            "start": start,
            "end": end,
            "source": "alias",
            "role": "unclassified",
            "negated": span_is_negated(start, end, negative_query_spans.get("dimension", ())),
        })
    mentions: list[dict[str, object]] = list(dimension_mentions)

    metric_mentions: list[dict[str, object]] = []
    if isinstance(explicit, dict):
        if explicit.get("metric_id"):
            metric_mentions.append({
                "kind": "metric", "canonical": str(explicit["metric_id"]),
                "source": "registry_exact",
            })
        for candidate in explicit.get("ambiguous_candidates") or []:
            if isinstance(candidate, dict) and candidate.get("metric_id"):
                metric_mentions.append({
                    "kind": "metric", "canonical": str(candidate["metric_id"]),
                    "source": "registry_exact",
                })
    mentions.extend(metric_mentions)

    comparison_kind = str(comparison_shape.get("kind") or "none")
    mentions.append({
        "kind": "date", "canonical": comparison_kind, "source": "comparison_parser",
    })
    operation_kind = str((operation_intent or {}).get("kind") or "ordinary_metric_query")
    mentions.append({
        "kind": "operation", "canonical": operation_kind, "source": "operation_lexicon",
    })

    scope = state.get("controlled_analysis_scope") or {}
    scope_allowed = sorted(
        str(item) for item in (scope.get("allowed_dimensions") or []) if item
    )
    routing = {
        "outcome": "clarify" if compiled.get("needs_clarification") else "resolved",
        "reason": compiled.get("clarification_reason"),
        "candidates": list(compiled.get("dimension_candidates") or []),
        "dimension_slot": (compiled.get("known_slots", {}).get("dimension") or {}).get("value"),
        "dimensions_slot": (compiled.get("known_slots", {}).get("dimensions") or {}).get("value"),
        "metric_slot": (compiled.get("known_slots", {}).get("metric") or {}).get("value"),
    }
    audit = {
        "version": "p0-1",
        "mentions": mentions,
        "dimension_detection": {
            "candidates_all": all_dims,
            "business_candidates": business_dims,
            "registered": registered_dims,
            "unregistered": unregistered_dims,
            "include_unregistered": contribution_requested_textually,
            "operation_hint": operation_kind,
        },
        "support_snapshot": {
            "status": "not_observed",
            "source": "p0_audit_no_support_resolver",
            "scope_allowed_dimensions": scope_allowed,
            "registered_dimensions": registered_dims,
            "unregistered_dimensions": unregistered_dims,
            "note": "P0 records existing signals only; SupportResolver is P1.",
        },
        "routing_observed": routing,
    }

    trace_id = state.get("trace_id")
    try:
        record_event("semantic.dimension.audit", {
            "trace_id": trace_id,
            "mention_count": len(mentions),
            "dimension_mention_count": len(dimension_mentions),
            "registered_dimension_count": len(registered_dims),
            "unregistered_dimension_count": len(unregistered_dims),
            "operation_hint": operation_kind,
            "outcome": routing["outcome"],
            "reason": routing["reason"],
        })
        record_event("semantic.support.audit", {
            "trace_id": trace_id,
            "status": "not_observed",
            "scope_allowed_count": len(scope_allowed),
            "registered_count": len(registered_dims),
            "unregistered_count": len(unregistered_dims),
            "reason_code": routing["reason"],
        })
    except Exception:  # noqa: BLE001 - audit must never break compilation
        pass

    try:
        elapsed_ms = (time.perf_counter() - started) * 1000
        dimension_count = len(dimension_mentions)
        if dimension_count:
            record_measurement(
                "dimension_mentions_total", dimension_count, labels={"source": "alias"},
            )
        if metric_mentions:
            record_measurement(
                "dimension_mentions_total", len(metric_mentions),
                labels={"source": "registry_exact"},
            )
        if comparison_kind != "none":
            record_measurement(
                "dimension_mentions_total", 1, labels={"source": "comparison_parser"},
            )
        record_measurement(
            "dimension_mentions_total", 1, labels={"source": "operation_lexicon"},
        )
        decision = (
            "resolved" if not compiled.get("needs_clarification")
            else "unsupported" if compiled.get("clarification_reason") in {
                "capability_gap", "region_filter_not_supported",
                "dimension_not_supported_by_metric",
            }
            else "clarify"
        )
        record_measurement(
            "dimension_decisions_total", 1, labels={"decision": decision},
        )
        record_measurement(
            "dimension_support_total", 1, labels={"status": "not_observed"},
        )
        record_measurement(
            "semantic_audit_latency_ms", elapsed_ms, labels={"stage": "audit"},
        )
    except Exception:  # noqa: BLE001 - metrics are diagnostic only
        pass
    return audit


def compile_turn_context(state: AgentState) -> AgentState:
    """Write a validated context-compilation record for the current turn."""
    original = str(state.get("user_message") or "").strip()
    metric_aliases = {metric_id: [metric_id, label] for metric_id, label in _metric_labels().items()}
    try:
        from app.metrics.registry import get_registry

        for definition in get_registry().list():
            metric_aliases[str(definition.metric_id)] = [
                str(term) for term in (
                    definition.metric_id,
                    definition.display_name,
                    *definition.aliases,
                ) if str(term or "").strip()
            ]
    except Exception:
        pass
    negative_analysis = parse_negative_intents(
        original,
        query_aliases=default_query_aliases(metric_aliases=metric_aliases),
    )
    negative_query_spans = negative_analysis["negative_query_spans"]
    query_text = mask_spans(original, negative_query_spans)
    for start, end in negative_analysis["unsupported_spans"]:
        query_text = mask_spans(query_text, {"unsupported": [(start, end)]})
    task_progress = state.get("task_progress_context") or {}
    current_scope_fp = str(state.get("current_query_scope_fingerprint") or "")
    previous_context = _trusted_previous_turn_context(state)
    current_data_version = str(state.get("data_version_manifest_hash") or "")
    task_scope_fp = str(task_progress.get("scope_fingerprint") or "")
    task_data_version = str(task_progress.get("data_version_manifest_hash") or "")
    task_context_valid = bool(
        task_progress
        and current_scope_fp
        and current_data_version
        and task_scope_fp
        and task_data_version
        and current_scope_fp == task_scope_fp
        and current_data_version == task_data_version
    )
    if task_context_valid:
        historical_claims = list(state.get("historical_claims_context") or [])
        known_claims = {str(item.get("value")) for item in historical_claims if isinstance(item, dict)}
        historical_claims.extend(
            item for item in (task_progress.get("approved_claims") or [])
            if isinstance(item, dict) and item.get("value") and str(item["value"]) not in known_claims
        )
        known_claims.update(str(item.get("value")) for item in historical_claims if isinstance(item, dict))
        state["historical_claims_context"] = historical_claims[-20:]
        historical_evidence = list(state.get("historical_evidence_context") or [])
        known_evidence = {str(item.get("evidence_id")) for item in historical_evidence if isinstance(item, dict)}
        historical_evidence.extend(
            item for item in (task_progress.get("evidence_refs") or [])
            if isinstance(item, dict) and item.get("evidence_id")
            and str(item["evidence_id"]) not in known_evidence
        )
        state["historical_evidence_context"] = historical_evidence[-40:]
        hypotheses = dict(state.get("historical_hypothesis_context") or {})
        hypotheses.update(task_progress.get("hypothesis_chain_summary") or {})
        state["historical_hypothesis_context"] = hypotheses

    labels = _metric_labels()
    explicit = _explicit_metric(query_text, labels)
    reference = _REFERENCE_PATTERNS[0].search(query_text)
    candidates = _previous_metrics(state, labels) if reference else []
    explorer_metric = _current_explorer_metric(state, labels) if reference else None
    previous_scope_metric = _previous_explorer_metric(state) if reference else None
    if reference and explorer_metric:
        # The selected Explorer metric is the current page context. If the
        # selection changed since the prior turn, do not let a broad query
        # authorization fingerprint carry the old default metric forward.
        # When the selection is unchanged, a more specific metric asked in
        # chat still takes precedence over the page's default selection.
        if previous_scope_metric and previous_scope_metric != explorer_metric["metric_id"]:
            candidates = [explorer_metric]
        elif not candidates:
            candidates = [explorer_metric]

    compiled = {
        # An explicit slot can replace one value while the utterance still
        # continues the prior request (for example, “那订单量呢？”).  Keep
        # the relation about discourse continuity; explicit current-turn slots
        # win below, and only missing slots may be inherited.
        "relation": "continue" if reference else "independent",
        "original_question": original,
        "standalone_question": original,
        "resolved_references": [],
        "intent": _intent(query_text),
        "known_slots": {},
        "missing_slots": [],
        "needs_clarification": False,
        "clarification_reason": None,
        "context_sources": [],
        "user_changes": [],
        "scope_candidates": [],
        "source_turn_id": previous_context.get("source_turn_id"),
        "scope_fingerprint": (
            previous_context.get("scope_fingerprint")
            or state.get("current_query_scope_fingerprint")
            or (state.get("authorization_boundary") or {}).get("query_scope_fingerprint")
        ),
        "task_progress_context_used": task_context_valid,
        "unsupported_negative_instruction": bool(negative_analysis["unsupported_double_negation"]),
        "negative_query_constraints": {
            "version": "negative-query-constraints-v1",
            "params": negative_analysis["negative_query_params"],
        },
    }
    result_followup_spec, result_followup_context, result_followup_reason = (
        detect_result_followup(query_text, previous_context, state)
    )
    if result_followup_spec:
        # A result follow-up is answered from the same-scope result snapshot,
        # never by reinterpreting a generic phrase as a fresh data request.
        compiled["intent"] = "follow_up"
        compiled["result_followup"] = {
            **result_followup_spec,
            "source_turn_id": (
                (result_followup_context or {}).get("source_turn_id")
                or previous_context.get("source_turn_id")
            ),
            "source_trace_id": (result_followup_context or {}).get("source_trace_id"),
            **({"unavailable_reason": result_followup_reason} if result_followup_reason else {}),
        }
        if result_followup_reason:
            compiled["needs_clarification"] = True
            compiled["clarification_reason"] = result_followup_reason
            if "result_context" not in compiled["missing_slots"]:
                compiled["missing_slots"].append("result_context")
        else:
            compiled["relation"] = "continue"

    def add_slot(
        name: str,
        value: object,
        source: str,
        confidence: float,
        *,
        source_task_state_version: object = None,
        source_turn_id: object = None,
        candidate_id: str | None = None,
    ) -> None:
        if value is not None and value != "":
            if source in {"current_turn", "current_turn_correction", "current_explorer_scope"}:
                resolved_source_turn = None
            elif source == "task_state":
                # A Task State slot may predate per-slot provenance.  Keep
                # it unknown rather than borrowing the latest conversation or
                # task turn as if it were the slot's origin.
                resolved_source_turn = source_turn_id
            else:
                resolved_source_turn = source_turn_id or compiled["source_turn_id"]
            compiled["known_slots"][name] = {
                "value": value, "source": source, "confidence": confidence,
                "source_turn_id": resolved_source_turn,
                **({"source_task_state_version": source_task_state_version}
                   if source_task_state_version is not None else {}),
                **({"candidate_id": candidate_id} if candidate_id else {}),
            }

    comparison_shape, comparison_status, comparison_reason, iso_tokens = _comparison_resolution(query_text)
    if result_followup_spec and result_followup_context and not result_followup_reason:
        add_slot(
            "metric",
            result_followup_context.get("metric_id"),
            "previous_result_context",
            1.0,
            source_turn_id=result_followup_context.get("source_turn_id"),
        )
        if result_followup_context.get("dimension"):
            add_slot(
                "dimension",
                result_followup_context.get("dimension"),
                "previous_result_context",
                1.0,
                source_turn_id=result_followup_context.get("source_turn_id"),
            )
    if compiled["unsupported_negative_instruction"]:
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "unsupported_negative_instruction"
    comparison_roles = (
        (comparison_shape.get("baseline_date"), comparison_shape.get("current_date"))
        if comparison_shape.get("kind") == "day_pair"
        and comparison_shape.get("baseline_date") and comparison_shape.get("current_date")
        else None
    )
    if comparison_roles:
        baseline_date, current_date = comparison_roles
        add_slot("baseline_date", baseline_date, "current_turn", 1.0)
        add_slot("current_date", current_date, "current_turn", 1.0)

    if explicit and explicit.get("ambiguous_candidates"):
        compiled["needs_clarification"] = True
        compiled["missing_slots"].append("metric")
        compiled["clarification_reason"] = "ambiguous_metric"
        compiled["metric_candidates"] = explicit["ambiguous_candidates"]
    elif explicit:
        add_slot("metric", explicit["metric_id"], "current_turn", 1.0)
    elif reference:
        if len(candidates) == 1:
            candidate = candidates[0]
            replacement = _replace_reference(original, candidate["label"])
            if replacement:
                standalone, reference_text = replacement
                compiled["standalone_question"] = standalone
                compiled["resolved_references"].append({
                    "text": reference_text,
                    "value": candidate["metric_id"],
                    "label": candidate["label"],
                    "source": candidate["source"],
                    "resolution": "single_same_scope_authorized_metric",
                })
                confidence = 0.99 if candidate["source"] == "current_explorer_scope" else 0.96
                if candidate["source"] == "task_state":
                    confidence = 0.90
                add_slot(
                    "metric", candidate["metric_id"], candidate["source"], confidence,
                    source_task_state_version=candidate.get("source_task_state_version"),
                    source_turn_id=candidate.get("source_turn_id"),
                )
                compiled["context_sources"].append(candidate["source"])
            else:
                compiled["needs_clarification"] = True
                compiled["clarification_reason"] = "reference_not_resolved"
        else:
            if not _is_personalisation_request(original):
                compiled["needs_clarification"] = True
                compiled["missing_slots"].append("metric")
                compiled["clarification_reason"] = (
                    "multiple_prior_metrics" if len(candidates) > 1 else "no_same_scope_metric_context"
                )

    # Extract explicit slots first. Same-scope date continuity is allowed only
    # for a data follow-up; authorization and seller/filter scope are never
    # inherited from conversational text.
    if len(iso_tokens) == 1:
        add_slot("date", iso_tokens[0], "current_turn", 1.0)
    elif not iso_tokens:
        replacement_date = _month_replacement(query_text, _prior_slot_value(state, "date"))
        if replacement_date:
            add_slot("date", replacement_date, "current_turn_correction", 0.98)

    explicit_month_period = _single_explicit_month_period(query_text)
    requires_exact_date_anchor = _requires_exact_date_anchor(query_text)
    period_comparison_followup = bool(
        explicit_month_period
        and comparison_status != "resolved"
        and _is_period_comparison_followup(query_text)
    )
    prior_slots = previous_context.get("context_slots") or {}
    if period_comparison_followup:
        prior_period_slot = prior_slots.get("period") or {}
        prior_period = prior_period_slot.get("value") if isinstance(prior_period_slot, dict) else None
        prior_metric_slot = prior_slots.get("metric") or {}
        prior_metric_id = (
            str(prior_metric_slot.get("value") or "")
            if isinstance(prior_metric_slot, dict) else ""
        )
        if not prior_metric_id:
            prior_metric_ids = {
                str(item.get("metric_id") or "")
                for item in (previous_context.get("authorized_metrics") or [])
                if isinstance(item, dict) and item.get("metric_id")
            }
            if len(prior_metric_ids) == 1:
                prior_metric_id = next(iter(prior_metric_ids))
        current_metric_id = str(
            (compiled["known_slots"].get("metric") or {}).get("value") or ""
        )
        valid_prior_period: dict[str, str] | None = None
        if isinstance(prior_period, dict):
            start = str(prior_period.get("start_date") or "")
            end = str(prior_period.get("end_date") or "")
            try:
                start_day, end_day = date.fromisoformat(start), date.fromisoformat(end)
                from calendar import monthrange

                if (
                    start_day.day == 1
                    and end_day.year == start_day.year
                    and end_day.month == start_day.month
                    and end_day.day == monthrange(start_day.year, start_day.month)[1]
                ):
                    valid_prior_period = {"start_date": start, "end_date": end}
            except ValueError:
                pass
        if (
            previous_context.get("source_turn_id")
            and valid_prior_period
            and prior_metric_id
            and current_metric_id == prior_metric_id
            and valid_prior_period["start_date"] != explicit_month_period["start_date"]
        ):
            periods = [valid_prior_period, explicit_month_period]
            periods.sort(key=lambda item: item["start_date"])
            comparison_shape = {
                "kind": "period_pair",
                "baseline": periods[0],
                "current": periods[1],
                "granularity": "month",
            }
            comparison_status = "resolved"
            comparison_reason = None
            add_slot("period", periods[1], "current_query_evidence", 1.0)
        else:
            compiled["needs_clarification"] = True
            compiled["clarification_reason"] = "comparison_period_context_unavailable"
            if "comparison_period" not in compiled["missing_slots"]:
                compiled["missing_slots"].append("comparison_period")
    elif explicit_month_period:
        # A complete month is a period fact, never a fabricated point-date slot.
        add_slot("period", explicit_month_period, "current_turn", 1.0)

    # The current locked Explorer scope cannot be widened by chat text.  Keep
    # the explicit date visible, but stop before any tool call when it falls
    # outside the server-issued range.
    date_slot = compiled["known_slots"].get("date") or {}
    if (
        date_slot.get("source") in {"current_turn", "current_turn_correction"}
        and _date_outside_scope(date_slot.get("value"), state)
    ):
        compiled["scope_candidates"].append({
            "field": "date",
            "value": date_slot.get("value"),
            "source": date_slot.get("source"),
            "status": "rejected",
            "reason": "date_outside_locked_scope",
        })
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "date_outside_locked_scope"
        if "date" not in compiled["missing_slots"]:
            compiled["missing_slots"].append("date")

    region_candidates = _detect_region_scope_candidates(query_text)
    if region_candidates:
        compiled["scope_candidates"].extend(region_candidates)
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "region_filter_not_supported"
        if "scope_filter" not in compiled["missing_slots"]:
            compiled["missing_slots"].append("scope_filter")

    contribution_markers = ("统计贡献", "贡献", "contribution")
    split_contribution_requested = (
        any(marker in query_text for marker in ("拆解", "拆分"))
        and (
            (comparison_status == "resolved"
             and comparison_shape.get("kind") in {"day_pair", "period_pair"})
            or "变化" in query_text
        )
    )
    contribution_requested_textually = (
        any(marker.casefold() in query_text.casefold() for marker in contribution_markers)
        or split_contribution_requested
    )
    # P2: classify every mention into grouping/filter/reference/comparison/
    # conflict.  Only grouping dimensions become business candidates; support
    # validation and contract sealing remain downstream.
    all_dimension_matches = _dimension_alias_matches(
        original, include_unregistered=True,
    )
    role_assignments = classify_dimension_mentions(
        original,
        all_dimension_matches,
        intent=compiled["intent"],
        metric_spans=_metric_mention_spans(original),
        region_tokens=_REGION_SCOPE_TOKENS,
        negated_spans=negative_query_spans.get("dimension", ()),
    )
    role_map = aggregate_roles(role_assignments)
    compiled["dimension_roles"] = [item.as_dict() for item in role_assignments]
    conflict_dims = conflict_dimensions(role_assignments)
    dimension_candidates = sorted(
        dimension for dimension, role in role_map.items() if role == "grouping"
    )
    # A resolved period/day comparison grouped by one valid dimension, asking
    # how each segment changed, is the statistical-contribution shape even when
    # the wording says “变化” instead of “贡献” or “拆解”.
    if (
        not contribution_requested_textually
        and dimension_candidates
        and not conflict_dims
        and comparison_status == "resolved"
        and comparison_shape.get("kind") in {"day_pair", "period_pair"}
        and "变化" in query_text
    ):
        contribution_requested_textually = True
    if (
        not dimension_candidates
        and compiled.get("relation") == "continue"
        and (
            contribution_requested_textually
            or any(
                marker in query_text.casefold()
                for marker in ("拆开", "拆分", "拆解", "breakdown", "分布")
            )
        )
    ):
        previous_slots = (
            state.get("previous_turn_context") or {}
        ).get("context_slots") or {}
        prior_dimension = (previous_slots.get("dimension") or {}).get("value")
        if prior_dimension:
            dimension_candidates = [str(prior_dimension)]
            role_assignments.append(RoleAssignment(
                dimension=str(prior_dimension),
                start=0,
                end=0,
                role="grouping",
                evidence=("previous_turn_context",),
                confidence_class="closed",
            ))
            role_map[str(prior_dimension)] = "grouping"
            compiled["dimension_roles"] = [
                item.as_dict() for item in role_assignments
            ]
    registered_dimension_names = {
        item[0] for item in _dimension_alias_matches(original)
        if not span_is_negated(item[2], item[3], negative_query_spans.get("dimension", ()))
    }
    dimension_matches = [
        item for item in all_dimension_matches
        if item[0] in set(dimension_candidates)
    ]
    if conflict_dims:
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "role_conflict"
        if "dimension" not in compiled["missing_slots"]:
            compiled["missing_slots"].append("dimension")
        compiled["dimension_candidates"] = dimension_candidates
        compiled["dimension_conflicts"] = conflict_dims
    elif len(dimension_candidates) == 1:
        candidate = dimension_candidates[0]
        # Contribution requests must pass the metric/attribution capability
        # check even when the dimension is registered globally: payment_type,
        # for example, exists in the catalog but is not a GMV attribution axis.
        if candidate in registered_dimension_names and not contribution_requested_textually:
            add_slot("dimension", candidate, "current_turn", 1.0)
        else:
            from app.context.support_resolver import resolve_support

            support = resolve_support(
                metric_id=(explicit or {}).get("metric_id") if isinstance(explicit, dict) else None,
                dimension=candidate,
                operation="contribution" if contribution_requested_textually else "breakdown",
                scope=state.get("controlled_analysis_scope"),
            )
            if support.executable:
                add_slot("dimension", candidate, "current_turn", 1.0)
            else:
                compiled["needs_clarification"] = True
                compiled["clarification_reason"] = (
                    support.reason_code or "dimension_not_registered"
                )
                if "dimension" not in compiled["missing_slots"]:
                    compiled["missing_slots"].append("dimension")
                compiled["dimension_candidates"] = dimension_candidates
    elif len(dimension_candidates) > 1:
        compiled["dimension_candidates"] = dimension_candidates
        multi_dimension_scope_blocked = bool(
            state.get("controlled_analysis_scope")
            or state.get("seller_scope_payload")
            or state.get("scope_type") == "seller"
        )
        if (
            2 <= len(dimension_candidates) <= 4
            and not conflict_dims
            and not contribution_requested_textually
            and not multi_dimension_scope_blocked
            and all(item in registered_dimension_names for item in dimension_candidates)
        ):
            from app.context.support_resolver import resolve_support

            metric_slot = compiled["known_slots"].get("metric") or {}
            metric_id = str(metric_slot.get("value") or "")
            support_results = [
                resolve_support(
                    metric_id=metric_id or None,
                    dimension=candidate,
                    operation="breakdown",
                    scope=state.get("controlled_analysis_scope"),
                )
                for candidate in dimension_candidates
            ]
            unsupported = next(
                (item for item in support_results if not item.executable), None,
            )
            if unsupported is None:
                # An inherited/result-follow-up singular dimension must not
                # survive beside the current turn's explicit multi-axis set.
                compiled["known_slots"].pop("dimension", None)
                add_slot("dimensions", list(dimension_candidates), "current_turn", 1.0)
            else:
                compiled["needs_clarification"] = True
                compiled["missing_slots"].append("dimensions")
                compiled["clarification_reason"] = (
                    unsupported.reason_code or "dimension_not_supported_by_metric"
                )
        else:
            compiled["needs_clarification"] = True
            compiled["missing_slots"].append("dimension")
            compiled["clarification_reason"] = (
                "multi_dimension_scope_not_supported"
                if multi_dimension_scope_blocked
                else "too_many_dimensions"
                if len(dimension_candidates) > 4
                else "ambiguous_dimension"
            )

    # A comparison-result follow-up may reuse the immediately preceding
    # server-owned period pair, but only when the current turn explicitly
    # refers to that change and contributes a newly named dimension. This is
    # deliberately before month_comparison and operation_intent are derived so
    # those mirrors and the sealed request contract cannot drift apart.
    period_inherited_from_result = False
    dimension_slot = compiled["known_slots"].get("dimension") or {}
    if (
        compiled.get("relation") == "continue"
        and contribution_requested_textually
        and _COMPARISON_CHANGE_REFERENCE.search(query_text)
        and not _has_explicit_current_period(
            query_text, iso_tokens=iso_tokens,
            explicit_month_period=explicit_month_period,
        )
        and len(dimension_candidates) == 1
        and isinstance(dimension_slot, dict)
        and dimension_slot.get("source") == "current_turn"
        and not (explicit and explicit.get("ambiguous_candidates"))
        and not compiled.get("unsupported_negative_instruction")
    ):
        from app.context.result_context import trusted_result_context

        trusted_comparison = trusted_result_context(previous_context, state)
        if (
            trusted_comparison
            and trusted_comparison.get("result_status") == "completed"
            and trusted_comparison.get("query_type") == "metric_compare"
        ):
            prior_metric_id = str(trusted_comparison.get("metric_id") or "")
            current_metric_slot = compiled["known_slots"].get("metric") or {}
            current_metric_id = str(
                current_metric_slot.get("value") or ""
            ) if isinstance(current_metric_slot, dict) else ""
            # The prior result is an exact metric source only if the current
            # request has not explicitly selected a different metric.
            metric_labels = _metric_labels()
            metric_matches = bool(
                prior_metric_id
                and prior_metric_id in metric_labels
                and (not current_metric_id or current_metric_id == prior_metric_id)
            )
            query_period = trusted_comparison.get("query_period") or {}
            try:
                baseline_start = date.fromisoformat(str(query_period["baseline_start"]))
                baseline_end = date.fromisoformat(str(query_period["baseline_end"]))
                current_start = date.fromisoformat(str(query_period["current_start"]))
                current_end = date.fromisoformat(str(query_period["current_end"]))
                from calendar import monthrange

                complete_natural_months = all((
                    baseline_start.day == 1,
                    baseline_end.year == baseline_start.year,
                    baseline_end.month == baseline_start.month,
                    baseline_end.day == monthrange(baseline_start.year, baseline_start.month)[1],
                    current_start.day == 1,
                    current_end.year == current_start.year,
                    current_end.month == current_start.month,
                    current_end.day == monthrange(current_start.year, current_start.month)[1],
                ))
                valid_period_pair = (
                    baseline_start <= baseline_end
                    and current_start <= current_end
                    and baseline_end < current_start
                    and (baseline_start, baseline_end) != (current_start, current_end)
                    and complete_natural_months
                )
            except (KeyError, TypeError, ValueError):
                valid_period_pair = False

            if metric_matches and valid_period_pair:
                source_turn_id = str(
                    trusted_comparison.get("source_turn_id")
                    or previous_context.get("source_turn_id")
                    or ""
                )
                if not current_metric_id:
                    add_slot(
                        "metric", prior_metric_id, "previous_result_context", 1.0,
                        source_turn_id=source_turn_id,
                    )
                comparison_shape = {
                    "kind": "period_pair",
                    "baseline": {
                        "start_date": baseline_start.isoformat(),
                        "end_date": baseline_end.isoformat(),
                    },
                    "current": {
                        "start_date": current_start.isoformat(),
                        "end_date": current_end.isoformat(),
                    },
                    "granularity": "month",
                }
                comparison_status = "resolved"
                comparison_reason = None
                add_slot(
                    "comparison_period",
                    {key: value for key, value in comparison_shape.items() if key != "kind"},
                    "previous_result_context", 1.0,
                    source_turn_id=source_turn_id,
                )
                compiled["context_sources"].append("previous_result_context")
                compiled["inherited_comparison_source_turn_id"] = source_turn_id
                # A precise result snapshot resolves only metric ambiguity
                # caused by missing/multiple generic history candidates. Other
                # clarification conditions remain intact.
                if compiled.get("clarification_reason") in {
                    "no_same_scope_metric_context", "multiple_prior_metrics",
                }:
                    compiled["needs_clarification"] = False
                    compiled["clarification_reason"] = None
                    compiled["missing_slots"] = [
                        value for value in compiled["missing_slots"]
                        if value not in {"metric", "no_same_scope_metric_context", "multiple_prior_metrics"}
                    ]
                period_inherited_from_result = True
            elif current_metric_id and current_metric_id != prior_metric_id:
                compiled["needs_clarification"] = True
                compiled["clarification_reason"] = "previous_comparison_metric_conflict"
                if "metric" not in compiled["missing_slots"]:
                    compiled["missing_slots"].append("metric")

    # Current Explorer bounds apply to both user-named and inherited comparison
    # endpoints. This is an early refusal only; the planner/MCP still performs
    # the authoritative date authorization before any query executes.
    if _comparison_outside_scope(comparison_shape, state):
        compiled["scope_candidates"].append({
            "field": "comparison_period",
            "value": {
                key: comparison_shape.get(key)
                for key in ("baseline", "current", "baseline_date", "current_date")
                if comparison_shape.get(key) is not None
            },
            "source": "previous_result_context" if period_inherited_from_result else "current_turn",
            "status": "rejected",
            "reason": "date_outside_locked_scope",
        })
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "date_outside_locked_scope"
        if "comparison_period" not in compiled["missing_slots"]:
            compiled["missing_slots"].append("comparison_period")

    operation_aliases = {
        "breakdown": ("拆开", "拆分", "拆解", "breakdown", "分布"),
        "trend": ("趋势", "走势", "trend"),
        "compare": ("对比", "compare", "同比", "环比"),
        "value": ("多少", "总额", "查询", "value"),
    }
    for operation, aliases in operation_aliases.items():
        if any(alias.casefold() in query_text.casefold() for alias in aliases):
            add_slot("operation", operation, "current_turn", 1.0)
            break
    else:
        if is_explicit_compare_request(query_text):
            add_slot("operation", "compare", "current_turn", 1.0)

    # Keep registered factor decomposition distinct from generic compare and
    # dimension distribution.  This is an intent hint only: data_query still
    # resolves the exact metric, Registry method, dependencies, dates and
    # authorization before it can enter the governed diagnostic executor.
    folded_question = query_text.casefold()
    dimension_requested = bool(
        "dimension" in compiled["known_slots"]
        or "dimensions" in compiled["known_slots"]
    )
    dimensions_slot_value = (
        (compiled["known_slots"].get("dimensions") or {}).get("value")
    )
    month_comparison = (
        {key: value for key, value in comparison_shape.items() if key != "kind"}
        if comparison_shape.get("kind") == "period_pair"
        and comparison_shape.get("baseline") and comparison_shape.get("current")
        else None
    )
    explicit_causal_request = _has_unnegated_marker(
        query_text, ("\u4e3a\u4ec0\u4e48", "\u539f\u56e0\u662f\u4ec0\u4e48", "\u5bfc\u81f4", "\u9020\u6210", "\u6839\u56e0", "\u4e1a\u52a1\u673a\u5236"),
    ) or _has_unnegated_marker(
        folded_question, ("why did", "root cause", "caused by", "business cause"),
    )
    explicit_registered_decomposition = (
        any(marker in query_text for marker in (
            "\u6ce8\u518c\u5206\u89e3\u65b9\u6cd5", "\u6ce8\u518c\u6570\u503c\u5206\u89e3", "\u6570\u503c\u5206\u89e3", "\u6570\u503c\u53d8\u5316\u8d21\u732e",
        ))
        or any(marker in folded_question for marker in (
            "registered factor decomposition", "registered numeric decomposition", "numeric decomposition",
        ))
    )
    # A request that explicitly orders the two evidence steps is a bounded
    # statistical workflow, not an ambiguous choice between trend and
    # contribution.  The sequence is carried in the sealed request contract;
    # it is not reconstructed by the planner from the original wording.
    trend_markers = ("趋势", "走势", "trend")
    contribution_markers = ("统计贡献", "贡献", "contribution") + (
        ("拆解", "拆分") if split_contribution_requested else ()
    )
    trend_positions = [folded_question.find(marker.casefold()) for marker in trend_markers
                       if folded_question.find(marker.casefold()) >= 0]
    contribution_positions = [folded_question.find(marker.casefold()) for marker in contribution_markers
                              if folded_question.find(marker.casefold()) >= 0]
    ordered_statistical_steps = False
    if contribution_requested_textually and trend_positions and contribution_positions:
        trend_position = min(trend_positions)
        contribution_position = min(contribution_positions)
        prefix = folded_question[:trend_position]
        between = folded_question[trend_position:contribution_position]
        ordered_statistical_steps = bool(
            trend_position < contribution_position
            and any(marker in prefix for marker in ("先", "首先", "first"))
            and any(marker in between for marker in ("再", "然后", "随后", "then", "next"))
        )
    if explicit_registered_decomposition and not dimension_requested and not explicit_causal_request:
        target_slot = compiled["known_slots"].get("metric") or {}
        operation_intent = {
            "kind": "registered_numeric_decomposition",
            "target_metric_id": target_slot.get("value"),
            "source": "current_turn",
            "causal_boundary": "numeric_explanation_only",
            **({"comparison_periods": month_comparison} if month_comparison else {}),
        }
    elif explicit_causal_request:
        operation_intent = {
            "kind": "business_causal_investigation",
            "source": "current_turn",
            **({"comparison_periods": month_comparison} if month_comparison else {}),
        }
    elif contribution_requested_textually and any(marker in folded_question for marker in ("趋势", "trend")) and not ordered_statistical_steps:
        operation_intent = {
            "kind": "dimensional_statistical_contribution",
            "source": "current_turn",
            "dimension": (compiled["known_slots"].get("dimension") or {}).get("value"),
        }
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "ambiguous_operation"
        compiled["missing_slots"].append("operation")
    elif contribution_requested_textually:
        operation_intent = {
            "kind": "dimensional_statistical_contribution",
            "source": "current_turn",
            "dimension": (compiled["known_slots"].get("dimension") or {}).get("value"),
            **({"comparison_periods": month_comparison} if month_comparison else {}),
            **({"required_evidence_sequence": ["trend", "dimension_contribution"]}
               if ordered_statistical_steps else {}),
        }
    else:
        operation_intent = {
            "kind": "ordinary_metric_query",
            "source": "current_turn",
            **({"dimensions": list(dimensions_slot_value)} if dimensions_slot_value else {}),
            **({"comparison_periods": month_comparison or {"granularity": "month"}}
               if month_comparison or period_comparison_followup else {}),
        }
    compiled["operation_intent"] = operation_intent
    if operation_intent["kind"] != "ordinary_metric_query":
        add_slot("operation_intent", operation_intent["kind"], "current_turn", 1.0)

    contribution_requested = operation_intent["kind"] == "dimensional_statistical_contribution"
    if dimensions_slot_value and operation_intent["kind"] != "ordinary_metric_query":
        # The new tool is an exact aggregate cross-tab, not a comparison,
        # contribution, causal investigation, or registered decomposition.
        # Do not let a multi-axis phrase silently change those workflows.
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "multi_dimension_operation_not_supported"
        if "operation" not in compiled["missing_slots"]:
            compiled["missing_slots"].append("operation")
    explicit_multi_date_comparison = bool(
        dimensions_slot_value
        and len(iso_tokens) == 2
        and is_explicit_compare_request(query_text)
    )
    if (
        dimensions_slot_value
        and (
            (
                comparison_status == "resolved"
                and comparison_shape.get("kind") in {"day_pair", "period_pair"}
            )
            or explicit_multi_date_comparison
        )
        and not compiled.get("needs_clarification")
    ):
        # The exact cross-tab tool has one aggregate date range. It cannot
        # compare the same cells across two dates/periods; keep that distinct
        # from a plain multi-axis breakdown until a comparison contract exists.
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "multi_dimension_comparison_not_supported"
        if "comparison_period" not in compiled["missing_slots"]:
            compiled["missing_slots"].append("comparison_period")
    contract_status = "resolved"
    contract_reason = None
    if contribution_requested:
        if compiled.get("clarification_reason") in {
            "ambiguous_operation",
            "role_conflict",
            "dimension_not_registered",
            "dimension_outside_locked_scope",
            "dimension_not_supported_by_metric",
            "dimension_not_permitted",
            "contribution_distinct_not_additive",
            "contribution_not_additive",
            "capability_not_enabled",
            "contribution_not_supported",
            "support_unknown",
            "date_outside_locked_scope",
        }:
            contract_status, contract_reason = (
                "clarification_required", compiled["clarification_reason"],
            )
        elif len(dimension_candidates) > 1:
            contract_status, contract_reason = "clarification_required", "ambiguous_dimension"
        elif not dimension_requested:
            contract_status, contract_reason = "clarification_required", "dimension_required"
        elif dimension_candidates[0] not in {
            dimension for item in _dimension_alias_matches(query_text)
            for dimension in [item[0]]
        }:
            # A recognized phrase that is not a Registry dimension is kept
            # visible in the contract, but cannot be silently mapped to a
            # different supported dimension.
            contract_status, contract_reason = "clarification_required", "capability_gap"
        elif explicit and explicit.get("ambiguous_candidates"):
            contract_status, contract_reason = "clarification_required", "ambiguous_metric"
        elif not (compiled["known_slots"].get("metric") or {}).get("value"):
            contract_status, contract_reason = "clarification_required", "metric_required"
        elif comparison_status != "resolved":
            contract_status, contract_reason = "clarification_required", comparison_reason or "comparison_dates_incomplete"
        if contract_status != "resolved":
            compiled["needs_clarification"] = True
            compiled["clarification_reason"] = contract_reason
            if contract_reason not in compiled["missing_slots"]:
                compiled["missing_slots"].append(contract_reason)

    contract_intent = str(operation_intent["kind"])
    contract_comparison: dict[str, object]
    if contribution_requested:
        contract_comparison = dict(comparison_shape)
    elif comparison_roles:
        contract_comparison = dict(comparison_shape)
    elif month_comparison:
        contract_comparison = {"kind": "period_pair", **month_comparison}
    elif period_comparison_followup and comparison_shape.get("kind") == "period_pair" and comparison_status != "resolved":
        contract_comparison = dict(comparison_shape)
    elif iso_tokens:
        contract_comparison = {"kind": "single_date", "date": iso_tokens[0]} if len(iso_tokens) == 1 else {
            "kind": "unresolved", "date_tokens": iso_tokens,
        }
    else:
        contract_comparison = {"kind": "none"}
    required_tool = (
        "compare_breakdown_by_dimension_period"
        if contract_comparison.get("kind") == "period_pair"
        else "compare_breakdown_by_dimension"
    ) if contribution_requested else None
    if dimensions_slot_value and contract_intent == "ordinary_metric_query":
        required_tool = "query_metric_by_dimensions"
    request_contract = seal_request_contract({
        "metric_id": (compiled["known_slots"].get("metric") or {}).get("value"),
        "operation_intent": contract_intent,
        "dimension": (compiled["known_slots"].get("dimension") or {}).get("value"),
        **({"dimensions": list(dimensions_slot_value)} if dimensions_slot_value else {}),
        "comparison": contract_comparison,
        "causal_boundary": (
            "statistical_contribution_not_business_causation"
            if contribution_requested else
            "numeric_explanation_only"
            if contract_intent == "registered_numeric_decomposition" else
            "causal_investigation_requested"
            if contract_intent == "business_causal_investigation" else
            "no_causal_claim"
        ),
        "required_capability": "compare" if contribution_requested else (
            "breakdown" if dimensions_slot_value and contract_intent == "ordinary_metric_query" else None
        ),
        "required_tool": required_tool,
        "required_evidence_sequence": (
            ["trend", "dimension_contribution"] if ordered_statistical_steps else None
        ),
        "source": "current_user_message",
        "parse_status": contract_status,
        "uncertainty_reason": contract_reason,
        "candidate_metrics": list(compiled.get("metric_candidates") or []),
        "candidate_dimensions": list(compiled.get("dimension_candidates") or []),
    })
    compiled["request_contract"] = request_contract
    compiled["request_contract_hash"] = request_contract["contract_hash"]

    if compiled["relation"] == "continue" and compiled["intent"] != "retrieve_metric_definition":
        prior_slots = previous_context.get("context_slots") or {}
        task_progress = _task_progress_for_slots(state)
        task_slots = task_progress.get("context_slots") or {}
        for slot_name in ("date", "dimension", "operation"):
            if slot_name == "dimension" and "dimensions" in compiled["known_slots"]:
                continue
            if slot_name == "date" and (
                (period_comparison_followup and comparison_status == "resolved")
                or period_inherited_from_result
            ):
                continue
            if slot_name in compiled["known_slots"]:
                continue
            prior = prior_slots.get(slot_name)
            source = "previous_turn_context"
            inherited = prior
            if not (isinstance(inherited, dict) and inherited.get("value")):
                inherited = task_slots.get(slot_name)
                source = "task_state"
            if isinstance(inherited, dict) and inherited.get("value"):
                inherited_value = inherited["value"]
                if slot_name == "date" and requires_exact_date_anchor:
                    try:
                        exact_day = date.fromisoformat(str(inherited_value))
                        inherited_value = exact_day.isoformat()
                    except (TypeError, ValueError):
                        _reject_inherited_slot(
                            compiled, name="date", value=inherited_value,
                            source=source, reason="exact_date_context_unavailable",
                        )
                        continue
                if slot_name == "date" and _date_outside_scope(inherited_value, state):
                    _reject_inherited_slot(
                        compiled, name="date", value=inherited_value,
                        source=source, reason="date_outside_locked_scope",
                    )
                    continue

                if slot_name in {"dimension", "operation"}:
                    metric_slot = compiled["known_slots"].get("metric") or {}
                    metric_id = str(metric_slot.get("value") or "")
                    if not metric_id:
                        _reject_inherited_slot(
                            compiled, name=slot_name, value=inherited_value,
                            source=source, reason="metric_required_for_inherited_support",
                        )
                        continue
                    if slot_name == "dimension":
                        prior_operation = prior_slots.get("operation") or task_slots.get("operation") or {}
                        operation_slot = compiled["known_slots"].get("operation") or prior_operation
                        operation = str((operation_slot or {}).get("value") or "breakdown")
                        support_kwargs = {
                            "metric_id": metric_id,
                            "dimension": str(inherited_value),
                            "operation": operation,
                            "scope": state.get("controlled_analysis_scope"),
                        }
                    else:
                        support_kwargs = {
                            "metric_id": metric_id,
                            "dimension": None,
                            "operation": str(inherited_value),
                            "scope": state.get("controlled_analysis_scope"),
                        }
                    try:
                        from app.context.support_resolver import resolve_support

                        support = resolve_support(**support_kwargs)
                        support_reason = support.reason_code or "capability_not_enabled"
                        support_executable = support.executable
                    except Exception:  # noqa: BLE001 - missing support metadata fails closed
                        support_reason = "support_unknown"
                        support_executable = False
                    if not support_executable:
                        _reject_inherited_slot(
                            compiled, name=slot_name, value=inherited_value,
                            source=source, reason=support_reason,
                        )
                        continue

                add_slot(
                    slot_name, inherited_value, source, 0.90,
                    source_task_state_version=(
                        task_progress.get("task_state_version") if source == "task_state" else None
                    ),
                    source_turn_id=(inherited.get("source_turn_id") if source == "task_state" else None),
                )
                compiled["context_sources"].append(source)
                if slot_name == "date":
                    compiled["standalone_question"] = _append_inherited_date(
                        compiled["standalone_question"], inherited_value,
                    )

        if (
            compiled["intent"] == "analyze_business_data"
            and "date" not in compiled["known_slots"]
            and compiled.get("clarification_reason") != "date_outside_locked_scope"
            and not requires_exact_date_anchor
            and state.get("use_current_explorer_context")
        ):
            scope = state.get("controlled_analysis_scope") or {}
            start = str(scope.get("default_start_date") or "").strip()
            end = str(scope.get("default_end_date") or "").strip()
            if start and end:
                date_range = f"{start} 至 {end}"
                add_slot("date", date_range, "current_explorer_scope", 0.99)
                compiled["context_sources"].append("current_explorer_scope")
                compiled["standalone_question"] += f"，日期 {date_range}"

        if (
            requires_exact_date_anchor
            and "date" not in compiled["known_slots"]
            and compiled.get("clarification_reason") != "date_outside_locked_scope"
        ):
            compiled["needs_clarification"] = True
            compiled["clarification_reason"] = "exact_date_context_unavailable"
            if "date" not in compiled["missing_slots"]:
                compiled["missing_slots"].append("date")

    compiled["user_changes"] = _user_changes(original, compiled["known_slots"], state)

    compiled["intent"] = (
        "retrieve_metric_definition" if compiled["intent"] == "retrieve_metric_definition"
        else "follow_up" if result_followup_spec
        else "analyze_business_data" if compiled["intent"] == "analyze_business_data"
        else "follow_up" if reference else "general"
    )

    # Refresh the sealed snapshot after all trusted same-scope continuation
    # slots have been resolved. No downstream node is allowed to repeat this
    # interpretation or overwrite the snapshot from raw text.
    request_contract = dict(compiled.get("request_contract") or {})
    final_slots = compiled["known_slots"]
    request_contract.update({
        "metric_id": (final_slots.get("metric") or {}).get("value"),
        "dimension": (final_slots.get("dimension") or {}).get("value"),
        "operation_intent": str((compiled.get("operation_intent") or {}).get("kind") or "ordinary_metric_query"),
    })
    final_dimensions = (final_slots.get("dimensions") or {}).get("value")
    if final_dimensions:
        request_contract["dimensions"] = list(final_dimensions)
    else:
        request_contract.pop("dimensions", None)
        if request_contract.get("required_tool") == "query_metric_by_dimensions":
            request_contract.pop("required_tool", None)
            request_contract.pop("required_capability", None)
    unrepresented_comparison_dimensions = _unrepresented_comparison_dimensions(
        role_assignments,
        request_contract,
    )
    if unrepresented_comparison_dimensions and not compiled.get("needs_clarification"):
        # Fail closed when a recognized comparison axis is not present in the
        # executable contract.  Keep this dimension-agnostic: a future
        # multi-axis contract can satisfy the same check through `dimensions`.
        compiled["needs_clarification"] = True
        compiled["clarification_reason"] = "comparison_dimension_not_supported"
        if "comparison_dimension" not in compiled["missing_slots"]:
            compiled["missing_slots"].append("comparison_dimension")
    if compiled.get("needs_clarification") and compiled.get("clarification_reason"):
        request_contract["parse_status"] = "clarification_required"
        request_contract["uncertainty_reason"] = compiled.get("clarification_reason")
    if request_contract.get("operation_intent") == "dimensional_statistical_contribution":
        # Keep date roles from the single parser result. Ambiguous/incomplete
        # cases retain the unresolved comparison object and cannot be upgraded.
        if comparison_roles:
            request_contract["comparison"] = dict(comparison_shape)
        elif month_comparison:
            request_contract["comparison"] = {"kind": "period_pair", **month_comparison}
    compiled["request_contract"] = seal_request_contract(request_contract)
    compiled["request_contract_hash"] = compiled["request_contract"]["contract_hash"]

    # P0 audit is observe-only.  It is deliberately built after the contract
    # is sealed and is never read by routing, support or contract validation.
    compiled["p0_audit"] = _build_p0_audit(
        original=original,
        negative_query_spans=negative_query_spans,
        dimension_matches=dimension_matches,
        explicit=explicit,
        comparison_shape=comparison_shape,
        operation_intent=operation_intent,
        contribution_requested_textually=contribution_requested_textually,
        compiled=compiled,
        state=state,
    )

    # P3 verifier is shadow-only: its result is stored outside the sealed
    # compilation and is never read by role routing, support, or execution.
    try:
        from app.context.semantic_verifier import observe_compilation

        verifier_observation = observe_compilation(state, compiled, original)
        if verifier_observation is not None:
            state["semantic_verifier_observation"] = verifier_observation
    except Exception:  # noqa: BLE001 - verifier failures must not affect P2
        state["semantic_verifier_observation"] = {
            "status": "provider_error",
            "mode": "shadow",
            "applied": False,
            "reason_code": "observer_failed",
        }

    state["context_compilation"] = compiled
    from app.context.turn_resolver import (
        apply_candidate_analysis_plan,
        enrich_turn_resolution,
        resolve_preference_context,
    )

    if not isinstance(state.get("preference_context"), dict) or not state.get("preference_context"):
        provider = state.get("preference_provider")
        if provider is None:
            from app.memory.preference_provider import provider_for_context

            provider = provider_for_context()
        context = resolve_preference_context(
            state, provider=provider, compilation=compiled,
        )
    else:
        context = state["preference_context"]
    if isinstance(context, dict):
        context = apply_candidate_analysis_plan(state, compiled, context)
    state["preference_context"] = context
    state["turn_resolution"] = enrich_turn_resolution(
        state, compiled, turn_resolution_payload(state, compiled),
    )
    # Pin the compiler's final contract as the runtime anchor. This also
    # supports governed node-level execution paths that begin after context
    # compilation and do not pass through triage first.
    state["request_contract_hash"] = compiled["request_contract_hash"]
    from app.context.request_contract import record_contract_observation
    record_contract_observation(state, "context_compiler")
    try:
        from app.observability import record_measurement

        record_measurement(
            "context_resolver_decisions",
            labels={"outcome": str(state["turn_resolution"].get("relation") or "invalid")},
        )
        record_event("context.compile", {
            "trace_id": state.get("trace_id"),
            "relation": compiled["relation"],
            "intent": compiled["intent"],
            "needs_clarification": compiled["needs_clarification"],
            "resolved_reference_count": len(compiled["resolved_references"]),
            "context_sources": compiled["context_sources"],
            "known_slot_names": sorted(compiled["known_slots"]),
            "preference_trigger_ids": [
                trigger
                for need in (state["turn_resolution"].get("source_needs") or [])
                if isinstance(need, dict)
                for trigger in (need.get("trigger_ids") or [])
            ],
            "task_relation_action": (
                state["turn_resolution"].get("task_relation") or {}
            ).get("action"),
        })
    except Exception:
        pass
    state.setdefault("node_history", []).append({
        "node": "context_compiler",
        "summary": (
            f"relation={compiled['relation']}; resolved={len(compiled['resolved_references'])}; "
            f"clarification={compiled['needs_clarification']}"
        ),
    })
    return state


def context_clarification(state: AgentState) -> AgentState:
    """Return an explicit clarification without entering a retrieval/tool path."""
    compiled = state.get("context_compilation") or {}
    state["task_type"] = "clarification"
    clarification_messages = {
        "ambiguous_metric": "请求中有多个可能的指标匹配。请明确选择一个指标。",
        "metric_required": "请明确要分析的指标。",
        "ambiguous_dimension": "请求中有多个可能的维度匹配。请明确选择一个维度。",
        "role_conflict": "该维度同时出现分组和范围线索，无法唯一确定语义；请明确是要按该维度分组，还是在当前范围内分析。",
        "dimension_required": "请明确要按哪个维度展示统计贡献。",
        "too_many_dimensions": "一次最多支持四个分组维度，请减少分组维度后重试。",
        "multi_dimension_scope_not_supported": "多维交叉查询当前不支持已锁定的店铺/浏览范围，请先切换到平台分析范围。",
        "multi_dimension_comparison_not_supported": "当前多维交叉工具只支持一个日期范围的精确汇总，不能同时比较两个日期或期间；请先分别查询，或改用单维度比较。",
        "multi_dimension_operation_not_supported": "当前多维交叉工具只支持普通指标汇总，不能直接执行贡献、根因或数值分解；请改为普通交叉汇总，或使用已支持的单维度分析路径。",
        "dimension_not_registered": "当前指标体系未注册该维度，已停止分析；请选择已支持的维度。",
        "dimension_not_supported_by_metric": (
            "当前指标未声明该维度的统计贡献能力，已停止分析；请选择该指标支持的维度。"
        ),
        "dimension_outside_locked_scope": (
            "该维度超出当前已锁定的分析范围，请重新选择范围或改用已支持的维度。"
        ),
        "capability_not_enabled": "当前指标或范围不支持该操作或维度，已停止分析。",
        "contribution_distinct_not_additive": (
            "该指标是去重计数类指标，所选维度无法得到可加总的贡献分解，已停止分析。"
        ),
        "contribution_not_supported": "当前指标未声明可加总贡献能力，已停止分析。",
        "contribution_not_additive": "该指标是比率/均值类指标，不能使用加法贡献方法。",
        "dimension_degenerate_in_scope": "当前授权范围只包含一家店铺，无法形成有意义的店铺维度对比；请调整范围或选择其他维度。",
        "dimension_not_permitted": "当前账号无权按该维度查看，已停止分析。",
        "support_unknown": "暂时无法验证该维度的支持性，已停止分析；请稍后重试或选择其他维度。",
        "comparison_dates_incomplete": "统计贡献需要明确的基线日期和当前日期，请补充两个不同日期。",
        "comparison_dates_ambiguous": "比较日期不唯一，请明确一组基线日期和当前日期。",
        "comparison_date_labels_conflict": "基线/当前日期标签存在冲突，请明确每个日期的角色。",
        "comparison_periods_incomplete": "期间比较需要明确基线期间和当前期间，请补充完整起止日期。",
        "comparison_periods_ambiguous": "期间不唯一，请明确一组基线期间和当前期间。",
        "comparison_period_labels_conflict": "基线/当前期间标签存在冲突，请明确期间角色。",
        "comparison_shape_conflict": "请求同时包含日比较和月比较日期，无法唯一确定比较范围，请澄清。",
        "comparison_dimension_not_supported": (
            "当前问题同时要求按商品品类分组并比较支付方式；系统暂不支持一次查询完成这项复合比较。"
            "请改为单独查询，或先在数据浏览中选择一个维度。"
        ),
        "task_type_unresolved": (
            "目前无法可靠判断要执行哪类分析。请说明你要查询指标、查看指标定义、分析变化原因，还是生成报告。"
        ),
        "invalid_comparison_date": "日期格式无效，请检查并重新提供基线日期和当前日期。",
        "invalid_comparison_period": "月份无效，请检查并明确基线月份和当前月份。",
        "date_outside_locked_scope": "该日期超出当前已锁定的分析范围，请重新选择数据范围后再分析。",
        "region_filter_not_supported": "当前受控分析范围不支持地区筛选（如华东）。请先在数据浏览器中重新选择服务端支持的范围。",
    }
    reason_code = str(compiled.get("clarification_reason") or "ambiguous_context_reference")
    state["answer"] = clarification_messages.get(
        reason_code, "你提到的指标可能不止一个，或当前范围里没有可沿用的指标。请说明具体指标或条件。",
    )
    state["final_answer"] = state["answer"]
    state["metric_planning_decision"] = {
        "status": "clarification_required",
        "reason": reason_code,
        "reason_code": reason_code,
        "missing_slots": compiled.get("missing_slots") or ["metric"],
        "dimension_candidates": list(compiled.get("dimension_candidates") or []),
    }
    metric_candidates = compiled.get("metric_candidates") or []
    if reason_code == "ambiguous_metric" and isinstance(metric_candidates, list):
        state["metric_planning_decision"]["ambiguous_candidates"] = [
            {
                "metric_id": str(item.get("metric_id") or ""),
                "display_name": str(item.get("label") or item.get("metric_id") or ""),
            }
            for item in metric_candidates[:12]
            if isinstance(item, dict) and item.get("metric_id")
        ]
    state.setdefault("node_history", []).append({
        "node": "context_clarification", "summary": "Asked user to resolve an ambiguous contextual reference.",
    })
    return state
