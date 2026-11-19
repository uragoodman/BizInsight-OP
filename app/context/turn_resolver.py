"""WP2 Turn Resolver: one governed interpretation for a chat turn.

The resolver does not replace the existing compiler or task service.  It
enriches their outputs with the WP2 contracts:

- explicit / inherited / corrected fields are kept distinguishable;
- task relation and clarification decisions are represented once;
- Mem0 personalisation is requested as an explicit source need, not queried
  on every turn;
- preference candidates are typed and may only affect presentation, a
  candidate analysis plan, or candidate scope defaults after validation.

This module never reads or writes Mem0.  WP3 supplies a provider.  When no
provider is configured the need is still observable and the turn continues
without personalisation.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

from app.observability import record_event
from app.context.operation_intent import is_explicit_compare_request
from app.context.negative_intent import default_query_aliases, parse_negative_intents


TURN_RESOLVER_VERSION = "turn-resolver-v2"
TURN_RESOLUTION_POLICY_VERSION = "wp2"
PREFERENCE_TRIGGER_VERSION = "pref-trigger-v1"
PREFERENCE_INTENT_RULES_VERSION = "preference-intent-rules-v4"
PREFERENCE_TOP_K_PER_TYPE = 5
PREFERENCE_TOP_K_TOTAL = 10
PREFERENCE_RELEVANCE_THRESHOLD = 0.55


def _preference_relevance_threshold() -> float:
    """Use the server-configured threshold for both retrieval and acceptance."""
    try:
        from app.core.config import get_settings

        value = float(get_settings().mem0_relevance_threshold)
    except Exception:  # noqa: BLE001 - fall back to the frozen default
        value = PREFERENCE_RELEVANCE_THRESHOLD
    return min(1.0, max(0.0, value))

PREFERENCE_PRESENTATION = "presentation"
PREFERENCE_ANALYSIS_METHOD = "analysis_method"
PREFERENCE_SCOPE_DEFAULT = "scope_default"
PREFERENCE_AUTHORITY_SECURITY = "authority_or_security"
ALLOWED_PREFERENCE_TYPES = frozenset({
    PREFERENCE_PRESENTATION,
    PREFERENCE_ANALYSIS_METHOD,
    PREFERENCE_SCOPE_DEFAULT,
})

_PERSONALISATION_MARKERS = (
    "按我平时", "我平时的", "按我平常", "我平常的", "我常用", "常用的回答方式",
    "我的常用", "我的默认", "用我的默认",
    "我的偏好", "我的习惯", "我的分析习惯", "按我的习惯", "按我习惯",
    "我通常", "按我通常", "老规矩", "上次约定", "按上次约定", "个性化",
    "我的分析方式", "我的报告格式", "my preference", "my usual", "as usual",
    "my habit", "how i usually",
)
_PREFERENCE_MANAGEMENT_MARKERS = (
    "记得我", "我的偏好是什么", "有哪些偏好", "查看我的偏好", "列出我的偏好",
    "忘记我的", "删除我的偏好", "修改我的偏好", "更新我的偏好",
    "what do you remember", "show my preferences", "forget my preference",
    "delete my preference", "update my preference",
)
_P1_ANALYSIS_MARKERS = (
    "分析", "步骤", "维度", "对比", "趋势", "拆解", "分解", "同比", "环比",
    "analysis", "dimension", "trend", "compare", "comparison", "breakdown",
)
_P1_SCOPE_MARKERS = (
    "默认时间范围", "常用时间范围", "统计周期", "默认周期", "默认日期", "常用日期",
    "时间范围", "最近一个月", "最近三个月", "本月", "上个月", "最近一周",
    "default period", "default date", "date range", "time range",
)
_PERSONALISED_ANSWER_RE = re.compile(
    r"(?:按|用|保持)[^，。；,;]{0,10}(?:我平时|我平常|我常用|我通常|我的习惯|我的方式|老规矩)[^，。；,;]{0,12}(?:回答|答复|回复|表达|输出)|"
    r"(?:my usual|as usual)[^,.;]{0,24}(?:answer|respond|reply)",
    re.IGNORECASE,
)
_REPORT_OUTPUT_MARKERS = (
    "报告", "汇报", "周报", "日报", "月报", "总结", "摘要", "report", "weekly report",
    "daily report", "summary",
)
_STYLE_VALUE_MARKERS = {
    "concise": ("简洁", "简短", "精简", "concise"),
    "detailed": ("详细", "展开说明", "detailed", "verbose"),
    "balanced": ("平衡", "适中", "balanced"),
}
_FORMAT_VALUE_MARKERS = {
    "table": ("表格", "table"),
    "bullets": ("项目符号", "项目列表", "bullet"),
    "narrative": ("叙述", "段落形式", "narrative"),
}
_LANGUAGE_VALUE_MARKERS = {
    "en-US": ("英文", "英语", "english"),
    "zh-TW": ("繁体中文", "繁體中文", "traditional chinese"),
    "zh-CN": ("中文", "简体中文", "简體中文", "chinese"),
}
_OPERATION_VALUE_MARKERS = {
    "compare": ("同比", "环比", "对比", "比较", "compare"),
    "trend": ("趋势", "走势", "trend"),
    "breakdown": ("breakdown",),
    "statistics": ("统计特征", "分布特征", "statistics"),
}
_PREFERENCE_FIELDS_BY_KEY: dict[str, tuple[str, ...]] = {
    "response_style": ("style",),
    "report_style": ("format",),
    "response_language": ("language",),
    "analysis_method": ("metric", "dimension", "operation", "date"),
    "analysis_steps": ("metric", "dimension", "operation", "date"),
    "default_period": ("start_date", "end_date"),
    "default_date": ("date",),
}
_PRESENTATION_MARKERS = (
    "报告", "周报", "日报", "总结", "摘要", "表格", "详细", "简洁",
    "report", "summary", "table", "verbose", "concise",
)
_SCOPE_FIELD_ALIASES = {
    "metric": ("metric", "metrics", "metric_id", "metric_ids"),
    "dimension": ("dimension", "dimensions"),
    "operation": ("operation", "analysis_operation"),
    "date": ("date", "dates", "date_range", "period", "periods"),
}
_FORBIDDEN_SCOPE_FIELDS = frozenset({
    "seller_id", "seller_ids", "tenant_id", "user_id", "permissions",
    "permission_version", "policy_set_version", "allowed_metrics",
    "allowed_dimensions", "authorization_fingerprint",
})


class PreferenceProvider(Protocol):
    """WP3 adapter boundary.

    Implementations must return only server-scoped candidates.  The resolver
    still re-validates every candidate and never treats provider output as an
    authorization credential.
    """

    def search(self, *, need: dict[str, Any], state: dict[str, Any]) -> list[Any]:
        ...


@dataclass(frozen=True)
class PreferenceCandidate:
    memory_key: str
    memory_type: str
    value: Any
    source_ref: str = ""
    confidence: float = 0.0
    suggested_fields: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)


def _fold(value: object) -> str:
    return str(value or "").casefold()


def _contains_marker(text: str, marker: str) -> bool:
    """Match English marker phrases on token boundaries; Chinese cues are substrings."""
    if marker.isascii() and re.fullmatch(r"[a-z0-9 ]+", marker):
        return re.search(rf"(?<![a-z0-9_]){re.escape(marker)}(?![a-z0-9_])", text) is not None
    return marker in text


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


def _canonical_preference_value(field_name: str, value: Any) -> str:
    """Normalize supported aliases before applying value-level exclusions."""
    folded = _fold(value).strip()
    markers_by_field = {
        "style": _STYLE_VALUE_MARKERS,
        "format": _FORMAT_VALUE_MARKERS,
        "language": _LANGUAGE_VALUE_MARKERS,
        "operation": _OPERATION_VALUE_MARKERS,
    }
    markers = markers_by_field.get(field_name)
    if markers:
        for canonical, aliases in markers.items():
            if folded == canonical.casefold() or any(_fold(alias) == folded for alias in aliases):
                return canonical
    aliases = default_query_aliases()
    parameter = {
        "dimension": "dimension",
        "metric": "metric",
        "operation": "operation",
        "date": "date",
        "start_date": "date",
        "end_date": "date",
        "period": "date",
        "date_range": "date",
    }.get(field_name)
    for canonical, values in (aliases.get(parameter or "") or {}).items():
        if folded == canonical.casefold() or any(_fold(alias) == folded for alias in values):
            return canonical
    return folded


def _preference_query_aliases() -> dict[str, dict[str, tuple[str, ...]]]:
    """Build the same metric aliases for Resolver and Compiler negation parsing."""
    metric_aliases: dict[str, tuple[str, ...]] = {}
    try:
        from app.metrics.registry import get_registry

        for definition in get_registry().list():
            metric_aliases[str(definition.metric_id)] = tuple(
                str(term)
                for term in (
                    definition.display_name,
                    *definition.aliases,
                    *definition.planning.phrases,
                )
                if str(term or "").strip()
            )
    except Exception:  # noqa: BLE001 - fail closed when registry is unavailable
        metric_aliases = {}
    return default_query_aliases(metric_aliases=metric_aliases)


def _period_range_for_alias(value: str, reference_date: date) -> tuple[date, date] | None:
    """Resolve a supported relative period alias for candidate comparison."""
    if value == "last_month":
        end = reference_date.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end
    if value == "last_30_days":
        return reference_date - timedelta(days=30), reference_date
    if value == "last_three_months":
        end = reference_date.replace(day=1) - timedelta(days=1)
        start_month = end.month - 2
        start_year = end.year
        while start_month <= 0:
            start_month += 12
            start_year -= 1
        return date(start_year, start_month, 1), end
    if value == "this_month":
        return reference_date.replace(day=1), reference_date
    if value == "last_year":
        return date(reference_date.year - 1, 1, 1), date(reference_date.year - 1, 12, 31)
    if value == "this_year":
        return date(reference_date.year, 1, 1), reference_date
    if value == "last_week":
        this_monday = reference_date - timedelta(days=reference_date.weekday())
        end = this_monday - timedelta(days=1)
        return end - timedelta(days=6), end
    return None


def _candidate_period_range(values: dict[str, Any]) -> tuple[date, date] | None:
    if values.get("date") not in (None, ""):
        candidate_date = _parse_date_range(values.get("date"))
        if candidate_date is not None:
            return candidate_date
    start = _parse_date_range(values.get("start_date"))
    end = _parse_date_range(values.get("end_date"))
    if start is None or end is None:
        # MemoryPolicy allows YYYY-MM as well as YYYY-MM-DD.
        try:
            start_raw, end_raw = str(values.get("start_date") or ""), str(values.get("end_date") or "")
            if re.fullmatch(r"\d{4}-\d{2}", start_raw) and re.fullmatch(r"\d{4}-\d{2}", end_raw):
                start_date = date.fromisoformat(f"{start_raw}-01")
                end_month = date.fromisoformat(f"{end_raw}-01")
                if end_month.month == 12:
                    end_date = date(end_month.year, 12, 31)
                else:
                    end_date = date(end_month.year, end_month.month + 1, 1) - timedelta(days=1)
                return start_date, end_date
        except ValueError:
            return None
        return None
    return start[0], end[1]


def _scope_reference_date(state: dict[str, Any]) -> date:
    reference_raw = state.get("scope_reference_date") or state.get("current_date")
    try:
        return date.fromisoformat(str(reference_raw)) if reference_raw else date.today()
    except ValueError:
        return date.today()


def _date_value_is_excluded(value: Any, exclusions: list[Any], state: dict[str, Any]) -> bool:
    reference_date = _scope_reference_date(state)
    candidate_canonical = _canonical_preference_value("date", value)
    candidate_range = (
        _period_range_for_alias(candidate_canonical, reference_date)
        or _parse_date_range(value)
    )
    for excluded in exclusions:
        excluded_canonical = _canonical_preference_value("date", excluded)
        excluded_range = (
            _period_range_for_alias(excluded_canonical, reference_date)
            or _parse_date_range(excluded)
        )
        if candidate_canonical == excluded_canonical:
            return True
        if candidate_range is not None and excluded_range is not None and not (
            candidate_range[1] < excluded_range[0]
            or candidate_range[0] > excluded_range[1]
        ):
            return True
    return False


def _is_metric_preference_phrase(text: str) -> bool:
    return bool(
        re.search(r"(?:默认|常用|常看)(?:的)?(?:分析)?指标|指标(?:偏好|习惯)", text)
        or _contains_any_marker(text, ("default metric", "usual metric", "preferred metric", "metric preference"))
    )


def _is_analysis_date_preference_phrase(text: str) -> bool:
    return bool(
        re.search(r"分析(?:时)?(?:(?:默认|常用|常看)(?:的)?)?(?:的)?日期(?:习惯|偏好|方式)?", text)
        or _contains_any_marker(text, ("analysis date preference", "usual analysis date"))
    )


def _is_scope_default_date_phrase(text: str) -> bool:
    return bool(
        not _is_analysis_date_preference_phrase(text)
        and (
            re.search(r"(?:默认|常用)(?:的)?(?:日期|时间范围|统计周期|周期)", text)
            or _contains_any_marker(text, ("default date", "usual date", "default time range"))
        )
    )


def _is_scope_default_date_only_phrase(text: str) -> bool:
    """Whether the user specifically asks for a default point date."""
    return bool(
        re.search(r"(?:默认|常用)(?:的)?日期", text)
        or _contains_any_marker(text, ("default date", "usual date"))
    )


def _operation_value_for_text(text: str) -> str | None:
    """Return a clearly named operation, ignoring adjectival 比较 usage."""
    for operation, markers in _OPERATION_VALUE_MARKERS.items():
        for marker in markers:
            if marker == "比较":
                continue
            if _contains_marker(text, marker):
                return operation
    if is_explicit_compare_request(text):
        return "compare"
    if re.search(r"比较(?:喜欢|容易|好|差|高|低|多|少|方便|简单|清楚|明白)", text):
        return None
    return None


_NEGATED_CATEGORY_MARKERS = {
    PREFERENCE_PRESENTATION: ("回答方式", "方式回答", "回答风格", "回复方式", "方式回复", "答复方式", "呈现偏好", "presentation preference", "answer style", "response style"),
    PREFERENCE_ANALYSIS_METHOD: ("分析", "分析习惯", "分析方式", "分析方法", "分析步骤", "analysis preference", "analysis habit", "analysis method"),
    PREFERENCE_SCOPE_DEFAULT: ("默认周期", "默认日期", "默认时间范围", "统计周期", "default period", "default date", "default time range"),
}


def _preference_instruction_constraints(
    text: str,
    *,
    compiled_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse deterministic negative constraints, ignoring quoted meta text.

    Negation is scoped to a punctuation/conjunction-delimited clause.  A later
    positive clause can express a separate intent.  This is deliberately a
    small rules parser, not a general natural-language negation system.
    """
    parsed = parse_negative_intents(text, query_aliases=_preference_query_aliases())
    clauses = parsed["clauses"]
    positive_clauses: list[str] = []
    negated_categories: set[str] = set()
    positive_categories: set[str] = set()
    excluded_values: dict[str, set[str]] = {}
    conflicts: set[str] = set()
    positive_values: dict[str, set[str]] = {}
    unsupported_double_negation = bool(parsed["unsupported_double_negation"])

    def add_exclusion(path: str, value: str) -> None:
        excluded_values.setdefault(path, set()).add(value)

    for clause_info in clauses:
        clause = str(clause_info["text"])
        if not clause_info["negative"]:
            positive_clauses.append(clause)
            folded_clause = _fold(clause)
            if _PERSONALISED_ANSWER_RE.search(clause) or _contains_any_marker(folded_clause, ("回答方式", "回答风格", "回复方式", "answer style", "response style")):
                positive_categories.add(PREFERENCE_PRESENTATION)
            if _contains_any_marker(folded_clause, ("分析习惯", "分析方式", "分析方法", "analysis habit", "analysis method")):
                positive_categories.add(PREFERENCE_ANALYSIS_METHOD)
            if _contains_any_marker(folded_clause, ("默认周期", "默认日期", "默认时间范围", "统计周期", "default period", "default date")):
                positive_categories.add(PREFERENCE_SCOPE_DEFAULT)
            for path, markers in (
                ("presentation.style", _STYLE_VALUE_MARKERS),
                ("presentation.format", _FORMAT_VALUE_MARKERS),
                ("presentation.language", _LANGUAGE_VALUE_MARKERS),
            ):
                for canonical, aliases in markers.items():
                    if _contains_any_marker(_fold(clause), aliases):
                        positive_values.setdefault(path, set()).add(canonical)
            continue

        # Double negation has no reliable v1/v2 interpretation.  Fail closed
        # for preference lookup in this clause rather than guessing intent.
        if clause_info["unsupported_double_negation"]:
            unsupported_double_negation = True
            continue
        negation_start, negation_end = clause_info["negation_spans"][0]
        local_negation_end = negation_end - int(clause_info["start"])
        tail = clause[local_negation_end:]
        folded_tail = _fold(tail)
        for category, markers in _NEGATED_CATEGORY_MARKERS.items():
            if _contains_any_marker(folded_tail, markers):
                negated_categories.add(category)

        # A value-level negation excludes only that canonical value.  This
        # allows (for example) a saved bullets preference after “不要用表格”.
        for path, markers in (
            ("presentation.style", _STYLE_VALUE_MARKERS),
            ("presentation.format", _FORMAT_VALUE_MARKERS),
            ("presentation.language", _LANGUAGE_VALUE_MARKERS),
        ):
            for canonical, aliases in markers.items():
                if _contains_any_marker(folded_tail, aliases):
                    add_exclusion(path, canonical)
    # A positive instruction in a separate clause re-enables that category;
    # clause-local negation does not suppress a later explicit intent.
    negated_categories.difference_update(positive_categories)
    if unsupported_double_negation:
        # The turn-level fail-closed decision is represented separately; do
        # not claim a particular category/value interpretation for it.
        negated_categories.clear()
        excluded_values.clear()
        conflicts.clear()

    # A later explicit positive value conflicts with a value exclusion.  The
    # resolver does not choose one silently; it suppresses the affected field.
    for path, excluded in excluded_values.items():
        if positive_values.get(path, set()) & excluded:
            conflicts.add(path)

    negative_query_params = {
        key: set(values)
        for key, values in parsed["negative_query_params"].items()
    }
    compiled_negative = (compiled_constraints or {}).get("negative_query_constraints") or {}
    if compiled_negative.get("version") == "negative-query-constraints-v1":
        for key, values in (compiled_negative.get("params") or {}).items():
            negative_query_params.setdefault(str(key), set()).update(str(value) for value in values)

    return {
        "unquoted_text": parsed["text"],
        "query_text": parsed["query_text"],
        "positive_text": " ".join(positive_clauses),
        "blocked_categories": {
            category: "explicit_negative_instruction"
            for category in sorted(negated_categories)
        },
        "excluded_values": {path: sorted(values) for path, values in sorted(excluded_values.items())},
        "conflicts": sorted(conflicts),
        "unsupported_double_negation": unsupported_double_negation,
        "negative_query_params": {
            key: sorted(values) for key, values in sorted(negative_query_params.items())
        },
    }


def _negated_preference_categories(text: str) -> set[str]:
    """Compatibility helper returning categories blocked by v2 negation rules."""
    return set(_preference_instruction_constraints(text)["blocked_categories"])


def _preference_lookup_request(text: str) -> dict[str, Any] | None:
    """Describe read-only preference lookup separately from apply retrieval."""
    # A request explicitly asking to *use* a habit remains an apply/retrieval
    # request even if the wording also contains "what is" or "tell me".
    if re.search(r"(?:按|用|遵循).{0,16}(?:我平时|我平常|我常用|我的默认|my usual)", text, re.IGNORECASE):
        return None
    if re.search(r"(?:有哪些偏好|查看我的偏好|列出我的偏好|what do you remember|show my preferences)", text, re.IGNORECASE):
        return None
    lookup_verb = re.search(r"(?:是什么|有哪些|查看|告诉我|what is|show me|list)", text, re.IGNORECASE)
    metric_phrase = _is_metric_preference_phrase(text) or bool(
        re.search(r"(?:我的|我平时的|我常用的)?(?:默认|常用|常看的)?(?:分析)?指标(?:偏好)?(?:是什么|有哪些)?", text)
    )
    date_phrase = _is_analysis_date_preference_phrase(text) or bool(
        re.search(r"(?:我的|我平时的|我常用的)?分析(?:时)?(?:常用的?)?日期(?:习惯|偏好)?(?:是什么|有哪些)?", text)
    )
    if not lookup_verb:
        return None
    if metric_phrase:
        return {"source": "preference_registry", "mode": "lookup", "memory_type": PREFERENCE_ANALYSIS_METHOD, "memory_keys": ["analysis_method", "analysis_steps"], "fields": ["metric"]}
    if date_phrase:
        return {"source": "preference_registry", "mode": "lookup", "memory_type": PREFERENCE_ANALYSIS_METHOD, "memory_keys": ["analysis_method", "analysis_steps"], "fields": ["date"]}
    if _is_scope_default_date_only_phrase(text):
        return {"source": "preference_registry", "mode": "lookup", "memory_type": PREFERENCE_SCOPE_DEFAULT, "memory_keys": ["default_date"], "fields": ["date"]}
    if _is_scope_default_date_phrase(text):
        return {"source": "preference_registry", "mode": "lookup", "memory_type": PREFERENCE_SCOPE_DEFAULT, "memory_keys": ["default_period"], "fields": ["start_date", "end_date"]}
    if re.search(r"(?:回答风格|回答方式|语言偏好|报告格式|response style|preferred language)", text, re.IGNORECASE):
        return {"source": "preference_registry", "mode": "lookup", "memory_type": PREFERENCE_PRESENTATION, "memory_keys": ["response_style", "response_language", "report_style"], "fields": ["style", "language", "format"]}
    return None


def _query_hash(text: str, scope: str) -> str:
    return hashlib.sha256(f"{scope}\n{text}".encode("utf-8")).hexdigest()[:16]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_id(memory_key: str, memory_type: str, source_ref: str) -> str:
    raw = f"{memory_type}\n{memory_key}\n{source_ref}".encode("utf-8")
    return "pref_" + hashlib.sha256(raw).hexdigest()[:16]


def durable_turn_id(state: dict[str, Any]) -> str:
    """Return the canonical durable Turn ID used by resolution and persistence.

    ``persistence_turn_id`` is the server-issued chat turn.  ``chat_turn_id``
    is the client idempotency key and must not silently replace it when both
    exist.  Legacy/direct callers without a lifecycle binding fall back to the
    client turn and finally the trace ID.
    """
    return str(
        state.get("persistence_turn_id")
        or state.get("chat_turn_id")
        or state.get("trace_id")
        or ""
    )


def client_turn_id(state: dict[str, Any]) -> str | None:
    value = str(state.get("chat_turn_id") or "").strip()
    return value or None


def detect_preference_source_needs(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return explicit preference source needs for this turn.

    The trigger is intentionally narrow: short references, data queries and
    task recovery do not call Mem0. P1 gathers all explicitly named preference
    intents. P2 currently covers report/summary output without an explicit
    format. P3 is routed to the Registry management API, never semantic search.
    """
    if not state.get("chat_session_id") and not state.get("memory_enabled", True):
        # Durable chat ignores the deprecated per-request switch; only legacy
        # memory-only callers may disable preference retrieval this way.
        return []
    text = str(state.get("user_message") or "").strip()
    if not text:
        return []
    compilation = state.get("context_compilation")
    constraints = _preference_instruction_constraints(
        text,
        compiled_constraints=compilation if isinstance(compilation, dict) else None,
    )
    if constraints["unsupported_double_negation"] or constraints["conflicts"]:
        # Ambiguous negative/positive combinations fail closed. The decisions
        # are retained by resolve_preference_context for privacy-safe tracing.
        return []
    folded = _fold(constraints["positive_text"])
    if _preference_lookup_request(folded):
        # Read-only profile lookup belongs to the Registry management path,
        # never to semantic retrieval used to apply a preference.
        return []
    # A clarification turn must not perform preference retrieval. The user
    # needs to resolve the ambiguity before historical defaults are relevant.
    if isinstance(compilation, dict) and compilation.get("needs_clarification") is True:
        return []
    known_slots = compilation.get("known_slots") if isinstance(compilation, dict) else {}
    p1 = _contains_any_marker(folded, _PERSONALISATION_MARKERS)
    negated_categories = set(constraints["blocked_categories"])
    p3 = _contains_any_marker(folded, _PREFERENCE_MANAGEMENT_MARKERS)
    report_requested = _contains_any_marker(folded, _REPORT_OUTPUT_MARKERS)
    personalized_answer = bool(_PERSONALISED_ANSWER_RE.search(text))
    intents: set[str] = set()
    requested_keys: dict[str, set[str]] = {}
    allowed_values: dict[str, list[str]] = {}
    excluded_values: dict[str, list[str]] = dict(constraints["excluded_values"])

    def add_keys(memory_type: str, *keys: str) -> None:
        intents.add(memory_type)
        requested_keys.setdefault(memory_type, set()).update(keys)

    if p1:
        if personalized_answer and PREFERENCE_PRESENTATION not in negated_categories:
            add_keys(PREFERENCE_PRESENTATION, "response_style", "response_language")
        personalized_format = _contains_any_marker(folded, (
            "我平时的格式", "我平常的格式", "我常用的格式", "我的常用格式", "my usual format",
        ))
        if (report_requested or personalized_format) and _contains_any_marker(
            folded,
            ("我平时", "我平常", "我常用", "我通常", "我的习惯", "老规矩", "as usual", "my usual"),
        ) and PREFERENCE_PRESENTATION not in negated_categories:
            add_keys(PREFERENCE_PRESENTATION, "report_style")
        analysis_intent = _contains_any_marker(folded, _P1_ANALYSIS_MARKERS) or (
            _is_metric_preference_phrase(folded)
            or _is_analysis_date_preference_phrase(folded)
        ) or _operation_value_for_text(folded) is not None
        if analysis_intent and PREFERENCE_ANALYSIS_METHOD not in negated_categories:
            add_keys(PREFERENCE_ANALYSIS_METHOD, "analysis_method", "analysis_steps")
        if PREFERENCE_SCOPE_DEFAULT not in negated_categories and not _is_analysis_date_preference_phrase(folded) and (
            _contains_any_marker(folded, _P1_SCOPE_MARKERS)
            or _is_scope_default_date_phrase(folded)
        ):
            add_keys(PREFERENCE_SCOPE_DEFAULT, "default_period", "default_date")

    # P2 is limited to report-like requests. It must not make ordinary metric
    # queries retrieve analysis or scope preferences.
    explicit_format = any(
        _contains_marker(folded, marker)
        for markers in _FORMAT_VALUE_MARKERS.values()
        for marker in markers
    )
    p2 = report_requested and not explicit_format
    if p2:
        add_keys(PREFERENCE_PRESENTATION, "report_style")

    # P3 is handled by the Registry management API, never by semantic search.
    # A co-occurring P1/P2 remains narrow and does not expand to all types.
    if not requested_keys:
        return []

    trigger_ids = (["P1"] if p1 else []) + (["P2"] if p2 else [])
    if not trigger_ids:
        return []

    explicit_fields: set[str] = set()
    blocked_fields: dict[str, str] = {}
    negative_query_params = constraints.get("negative_query_params") or {}
    # A negated explicit query value excludes that value from preference
    # candidates, while leaving other values in the same field eligible.
    # This keeps “don't use GMV; use my usual metric” able to select another
    # permitted metric without reintroducing GMV.
    for field_name in ("metric", "dimension", "operation", "date"):
        values = negative_query_params.get(field_name) or []
        if not values:
            continue
        if field_name == "date":
            excluded_values.setdefault("analysis_method.date", []).extend(values)
            excluded_values.setdefault("scope_default.period", []).extend(values)
        else:
            excluded_values.setdefault(f"analysis_method.{field_name}", []).extend(values)
    excluded_values = {
        path: sorted(set(values)) for path, values in excluded_values.items()
    }
    for canonical, markers in _STYLE_VALUE_MARKERS.items():
        if _contains_any_marker(folded, markers):
            explicit_fields.add("presentation.style")
    for canonical, markers in _FORMAT_VALUE_MARKERS.items():
        if _contains_any_marker(folded, markers):
            explicit_fields.add("presentation.format")
    for canonical, markers in _LANGUAGE_VALUE_MARKERS.items():
        if _contains_any_marker(folded, markers):
            explicit_fields.add("presentation.language")
    explicit_operation = _operation_value_for_text(
        _fold(constraints.get("query_text") or folded)
    )
    if negative_query_params.get("operation"):
        explicit_operation = None
    if explicit_operation:
        explicit_fields.add("analysis_method.operation")
        allowed_values["analysis_method.operation"] = [explicit_operation]

    for conflict_path in constraints["conflicts"]:
        explicit_fields.add(conflict_path)

    explicit_sources = {"current_turn", "current_turn_correction", "current_explorer_scope"}
    inherited_sources = {"previous_turn_context", "task_state"}
    if isinstance(known_slots, dict):
        for slot_name, slot in known_slots.items():
            if not isinstance(slot, dict) or not slot.get("value"):
                continue
            source = str(slot.get("source") or "")
            if slot_name in {"metric", "dimension", "operation", "date"}:
                path = f"analysis_method.{slot_name}"
                if source in explicit_sources:
                    explicit_fields.add(path)
                elif source in inherited_sources:
                    blocked_fields[path] = "task_state_value_present"
            if slot_name == "date":
                date_paths = (
                    "scope_default.start_date", "scope_default.end_date", "scope_default.date",
                )
                if source in explicit_sources:
                    explicit_fields.update(date_paths)
                elif source in inherited_sources:
                    blocked_fields.update({path: "task_state_value_present" for path in date_paths})

    # An explicit period in the current wording is authoritative even when the
    # compiler has not populated slots (e.g. direct resolver callers/tests).
    explicit_period_markers = (
        "最近一个月", "最近三个月", "本月", "上个月", "去年", "今年", "最近一周",
        "last month", "this month", "last week", "this year", "last year",
    )
    if _contains_any_marker(folded, explicit_period_markers):
        explicit_fields.update({
            "scope_default.start_date", "scope_default.end_date", "scope_default.date",
        })
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", folded):
        explicit_fields.update({
            "scope_default.start_date", "scope_default.end_date", "scope_default.date",
        })

    # Restrict analysis fields to the facet named by this turn. A generic
    # "usual analysis method" can consider operation/dimension; metric/date
    # require an explicit request and therefore are not implicit defaults.
    requested_fields: dict[str, set[str]] = {}
    if PREFERENCE_ANALYSIS_METHOD in requested_keys:
        analysis_fields: set[str] = set()
        if _is_metric_preference_phrase(folded):
            analysis_fields.add("metric")
        elif _is_analysis_date_preference_phrase(folded):
            analysis_fields.add("date")
        operation_named = explicit_operation is not None
        dimension_named = any(
            _contains_marker(folded, marker)
            for marker in ("维度", "拆解", "分解", "按品类", "按类别", "按地区", "by category", "by dimension")
        )
        if operation_named:
            analysis_fields.add("operation")
        if dimension_named:
            analysis_fields.add("dimension")
        if not analysis_fields:
            analysis_fields.update({"operation", "dimension"})
        requested_fields[PREFERENCE_ANALYSIS_METHOD] = analysis_fields
    if PREFERENCE_PRESENTATION in requested_keys:
        requested_fields[PREFERENCE_PRESENTATION] = {
            field
            for key in requested_keys[PREFERENCE_PRESENTATION]
            for field in _PREFERENCE_FIELDS_BY_KEY.get(key, ())
        }
    if PREFERENCE_SCOPE_DEFAULT in requested_keys:
        requested_fields[PREFERENCE_SCOPE_DEFAULT] = (
            {"date"} if _is_scope_default_date_only_phrase(folded)
            else {"start_date", "end_date"}
        )

    effective_keys: dict[str, set[str]] = {}
    allowed_fields: dict[str, set[str]] = {}
    for memory_type, keys in requested_keys.items():
        fields = requested_fields.get(memory_type)
        if fields is None:
            fields = {
                field for key in keys
                for field in _PREFERENCE_FIELDS_BY_KEY.get(key, ())
            }
        fields = set(fields)
        for field_name in tuple(fields):
            path = f"{memory_type}.{field_name}"
            if path in explicit_fields or path in blocked_fields:
                fields.discard(field_name)
        allowed_fields[memory_type] = fields
        remaining_keys = {
            key for key in keys
            if set(_PREFERENCE_FIELDS_BY_KEY.get(key, ())) & fields
        }
        if remaining_keys:
            effective_keys[memory_type] = remaining_keys

    # No effective field means there is nothing to read from Mem0. P3-only and
    # fully overridden requests therefore never fall through to all types.
    if not effective_keys:
        return []

    scope = str(
        state.get("current_query_scope_fingerprint")
        or (state.get("authorization_boundary") or {}).get("query_scope_fingerprint")
        or ""
    )
    return [{
        "source": "mem0",
        "reason": "personalization_trigger" if p1 else "presentation_default",
        "rules_version": PREFERENCE_INTENT_RULES_VERSION,
        "intent_ids": sorted(intents),
        "trigger_ids": list(dict.fromkeys(trigger_ids)),
        "memory_types": sorted(effective_keys),
        "allowed_keys": {
            kind: sorted(keys) for kind, keys in sorted(effective_keys.items())
        },
        "allowed_fields": {
            kind: sorted(allowed_fields[kind]) for kind in sorted(effective_keys)
        },
        "allowed_values": {
            path: values for path, values in sorted(allowed_values.items())
        },
        "blocked_categories": dict(constraints["blocked_categories"]),
        "excluded_values": excluded_values,
        "negative_query_params": negative_query_params,
        "overridden_fields": sorted(explicit_fields),
        "blocked_fields": dict(sorted(blocked_fields.items())),
        "other_intents": ["preference_management"] if p3 else [],
        "top_k_per_type": PREFERENCE_TOP_K_PER_TYPE,
        "top_k_total": PREFERENCE_TOP_K_TOTAL,
        "relevance_threshold": _preference_relevance_threshold(),
        "query_hash": _query_hash(text, scope),
        "scope_fingerprint": scope,
        "required": False,
        "status": "requested",
        "policy_version": PREFERENCE_TRIGGER_VERSION,
    }]


def _normalise_candidate(raw: Any) -> PreferenceCandidate | None:
    if isinstance(raw, PreferenceCandidate):
        return raw
    if not isinstance(raw, dict):
        return None
    memory_key = str(raw.get("memory_key") or raw.get("key") or "").strip()
    memory_type = str(raw.get("memory_type") or raw.get("type") or "").strip()
    if not memory_key or not memory_type:
        return None
    value = raw.get("value")
    suggested = raw.get("suggested_fields")
    if not isinstance(suggested, dict):
        suggested = value if isinstance(value, dict) else {}
    return PreferenceCandidate(
        memory_key=memory_key,
        memory_type=memory_type,
        value=value,
        source_ref=str(raw.get("source_ref") or raw.get("memory_id") or memory_key),
        confidence=_as_float(raw.get("confidence"), 0.0),
        suggested_fields=dict(suggested),
        scope=dict(raw.get("scope") or {}),
    )


def _explicit_slots(compilation: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    slots = (compilation or {}).get("known_slots") or {}
    return {
        str(name): slot for name, slot in slots.items()
        if isinstance(slot, dict) and str(slot.get("source") or "") in {
            "current_turn", "current_turn_correction", "current_explorer_scope",
        }
    }


def _inherited_slots(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return prior-task/conversation slots used for conflict checks."""
    inherited: dict[str, dict[str, Any]] = {}
    task = state.get("task_progress_context") or {}
    current_scope = str(state.get("current_query_scope_fingerprint") or "")
    task_scope = str(task.get("scope_fingerprint") or "")
    current_version = str(state.get("data_version_manifest_hash") or "")
    task_version = str(task.get("data_version_manifest_hash") or "")
    if (
        not current_scope or not task_scope or current_scope != task_scope
        or not current_version or not task_version or current_version != task_version
    ):
        task = {}
    for name, slot in (task.get("context_slots") or {}).items():
        if isinstance(slot, dict) and slot.get("value") not in (None, ""):
            inherited[str(name)] = {**slot, "source": "task_state"}
    previous = state.get("previous_turn_context") or {}
    for name, slot in (previous.get("context_slots") or {}).items():
        if isinstance(slot, dict) and slot.get("value") not in (None, ""):
            inherited.setdefault(str(name), {**slot, "source": "previous_turn_context"})
    return inherited


def _controlled_scope(state: dict[str, Any]) -> dict[str, Any]:
    scope = state.get("controlled_analysis_scope")
    return scope if isinstance(scope, dict) else {}


def _allowed_metrics(state: dict[str, Any]) -> set[str]:
    return {str(item) for item in (_controlled_scope(state).get("allowed_metrics") or []) if item}


def _metric_registered(metric_id: str) -> bool:
    try:
        from app.metrics.registry import get_registry

        get_registry().get(metric_id)
        return True
    except Exception:  # noqa: BLE001 - unknown metric is a rejection, not a crash
        return False


def _metric_allowed(metric_id: str, state: dict[str, Any]) -> tuple[bool, str | None]:
    if not _metric_registered(metric_id):
        return False, "metric_not_registered"
    allowed = _allowed_metrics(state)
    if allowed and metric_id not in allowed:
        return False, "metric_outside_locked_scope"
    return True, None


def _dimension_allowed(dimension: str, metric_ids: set[str], state: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate a dimension through the unified SupportResolver.

    Kept as a thin wrapper so existing callers and reason codes stay stable
    while all dimension entries converge on one five-status decision.
    """

    from app.context.support_resolver import resolve_support

    scope = _controlled_scope(state)
    if not scope and str(state.get("scope_type") or "") == "seller":
        # Preserve the old seller-scope fallback when a caller has no locked
        # scope payload but the tenant identity is seller-scoped.
        scope = {"seller_id": "__seller_scope__"}
    for metric_id in (metric_ids or {None}):
        support = resolve_support(
            metric_id=metric_id,
            dimension=dimension,
            operation="breakdown",
            scope=scope,
        )
        if not support.executable:
            if support.reason_code == "support_unknown":
                return False, "metric_semantics_unavailable"
            return False, support.reason_code or "dimension_not_supported_by_metric"
    return True, None


def _parse_date_range(value: Any) -> tuple[date, date] | None:
    text = str(value or "")
    matches = re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", text)
    if len(matches) >= 2:
        try:
            start, end = date.fromisoformat(matches[0]), date.fromisoformat(matches[1])
        except ValueError:
            return None
        return (start, end) if start <= end else None
    if len(matches) == 1:
        try:
            value_date = date.fromisoformat(matches[0])
        except ValueError:
            return None
        return value_date, value_date
    return None


def _scope_contains_dates(state: dict[str, Any], value: Any) -> tuple[bool, str | None]:
    scope = _controlled_scope(state)
    start_raw, end_raw = scope.get("default_start_date"), scope.get("default_end_date")
    if not start_raw or not end_raw:
        return False, "scope_issuance_required"
    requested = _parse_date_range(value)
    if requested is None:
        return False, "date_not_parseable"
    try:
        start, end = date.fromisoformat(str(start_raw)), date.fromisoformat(str(end_raw))
    except ValueError:
        return False, "scope_issuance_required"
    if requested[0] < start or requested[1] > end:
        return False, "date_outside_locked_scope"
    return True, None


def _date_values_equal(left: Any, right: Any) -> bool:
    left_range, right_range = _parse_date_range(left), _parse_date_range(right)
    if left_range is not None and right_range is not None:
        return left_range == right_range
    return str(left or "").strip() == str(right or "").strip()


def _explicit_field_values(compilation: dict[str, Any] | None) -> dict[str, Any]:
    slots = _explicit_slots(compilation)
    return {
        name: (slot or {}).get("value")
        for name, slot in slots.items()
        if (slot or {}).get("value") not in (None, "")
    }


def _presentation_explicitly_overridden(need: dict[str, Any], field_name: str) -> bool:
    return f"presentation.{field_name}" in set(need.get("overridden_fields") or [])


def _contains_forbidden_scope_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_SCOPE_FIELDS:
                return True
            if _contains_forbidden_scope_field(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_scope_field(item) for item in value)
    return False


def _field_values(candidate: PreferenceCandidate) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, value in (candidate.suggested_fields or {}).items():
        canonical = None
        for name, aliases in _SCOPE_FIELD_ALIASES.items():
            if str(key).casefold() in aliases:
                canonical = name
                break
        if canonical:
            fields[canonical] = value
    return fields


def _validate_analysis_field(
    name: str, value: Any, candidate: PreferenceCandidate,
    state: dict[str, Any], explicit: dict[str, Any], inherited: dict[str, Any],
) -> tuple[bool, str | None, Any]:
    if name == "metric":
        metric_ids = value if isinstance(value, (list, tuple)) else [value]
        checks = [str(item) for item in metric_ids if item]
        if not checks:
            return False, "empty_metric_candidate", None
        if explicit.get("metric") and str(explicit["metric"]) not in checks:
            return False, "explicit_input_wins", None
        if inherited.get("metric") and str(inherited["metric"]) not in checks:
            return False, "task_state_conflict", None
        rejected: list[str] = []
        reasons: list[str] = []
        for metric_id in checks:
            ok, reason = _metric_allowed(metric_id, state)
            if not ok:
                rejected.append(metric_id)
                reasons.append(reason or "metric_rejected")
        if rejected and len(rejected) == len(checks):
            return False, reasons[0] if reasons else "metric_rejected", None
        return True, None, [item for item in checks if item not in rejected]
    if name == "dimension":
        dim = str(value or "")
        if not dim:
            return False, "empty_dimension_candidate", None
        if explicit.get("dimension") and str(explicit["dimension"]) != dim:
            return False, "explicit_input_wins", None
        if inherited.get("dimension") and str(inherited["dimension"]) != dim:
            return False, "task_state_conflict", None
        metric_ids = set()
        metric_value = explicit.get("metric") or inherited.get("metric")
        if isinstance(metric_value, (list, tuple)):
            metric_ids = {str(item) for item in metric_value}
        elif metric_value:
            metric_ids = {str(metric_value)}
        ok, reason = _dimension_allowed(dim, metric_ids, state)
        if not ok:
            return False, reason, None
        return True, None, dim
    if name == "operation":
        op = str(value or "")
        if not op:
            return False, "empty_operation_candidate", None
        if explicit.get("operation") and str(explicit["operation"]) != op:
            return False, "explicit_input_wins", None
        if inherited.get("operation") and str(inherited["operation"]) != op:
            return False, "task_state_conflict", None
        return True, None, op
    if name == "date":
        if explicit.get("date") and str(explicit["date"]) != str(value):
            return False, "explicit_input_wins", None
        if inherited.get("date") and str(inherited["date"]) != str(value):
            return False, "task_state_conflict", None
        ok, reason = _scope_contains_dates(state, value)
        if not ok:
            return False, reason, None
        return True, None, value
    return False, "unsupported_preference_field", None


def _evaluate_scope_fields(
    candidate: PreferenceCandidate,
    state: dict[str, Any],
    explicit: dict[str, Any],
    inherited: dict[str, Any],
    excluded_values: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    effects: dict[str, Any] = {}
    rejections: list[dict[str, Any]] = []
    values = (
        candidate.suggested_fields
        if isinstance(candidate.suggested_fields, dict) and candidate.suggested_fields
        else candidate.value
    )
    if not isinstance(values, dict):
        return effects, [{"field": "*", "reason": "no_scope_default_value"}]
    period_exclusions = (excluded_values or {}).get("scope_default.period") or []
    if period_exclusions:
        reference_date = _scope_reference_date(state)
        candidate_range = _candidate_period_range(values)
        for exclusion in period_exclusions:
            canonical = _canonical_preference_value("date", exclusion)
            excluded_range = _period_range_for_alias(canonical, reference_date)
            if excluded_range is None:
                try:
                    excluded_date = date.fromisoformat(str(exclusion))
                    excluded_range = (excluded_date, excluded_date)
                except ValueError:
                    continue
            if candidate_range is not None and not (
                candidate_range[1] < excluded_range[0]
                or candidate_range[0] > excluded_range[1]
            ):
                return effects, [
                    {"field": "start_date", "reason": "source_need_value_excluded"},
                    {"field": "end_date", "reason": "source_need_value_excluded"},
                ]
    for key, value in values.items():
        field_name = str(key)
        if field_name.casefold() in _FORBIDDEN_SCOPE_FIELDS:
            rejections.append({"field": field_name, "reason": "authorization_scope_cannot_be_modified"})
            continue
        if field_name.casefold() in {"region", "regions", "area", "areas"}:
            # The current server scope has no authorized region filter.  Do
            # not map it to a different dimension or silently drop the filter.
            rejections.append({"field": field_name, "reason": "region_filter_not_supported"})
            continue
        if field_name.casefold() in {"start_date", "end_date", "date_range", "period"}:
            exclusion_path = f"scope_default.{field_name}"
            excluded = (excluded_values or {}).get(exclusion_path) or []
            if _canonical_preference_value(field_name, value) in {
                _canonical_preference_value(field_name, item) for item in excluded
            }:
                rejections.append({"field": field_name, "reason": "source_need_value_excluded"})
                continue
            if explicit.get("date") and not _date_values_equal(explicit["date"], value):
                rejections.append({"field": field_name, "reason": "explicit_input_wins"})
                continue
            if inherited.get("date") and not _date_values_equal(inherited["date"], value):
                rejections.append({"field": field_name, "reason": "task_state_conflict"})
                continue
            ok, reason = _scope_contains_dates(state, value)
            if not ok:
                rejections.append({"field": field_name, "reason": reason or "scope_rejected"})
            else:
                effects["date"] = value
            continue
        rejections.append({"field": field_name, "reason": "unsupported_scope_preference"})
    return effects, rejections


def _candidate_analysis_plan(effects: list[dict[str, Any]]) -> dict[str, Any]:
    steps = []
    candidate_ids = []
    for effect in effects:
        if not isinstance(effect, dict) or effect.get("kind") != "analysis_method":
            continue
        candidate_ids.append(effect.get("candidate_id"))
        for field_name, value in (effect.get("fields") or {}).items():
            steps.append({
                "source": "mem0_preference",
                "kind": "analysis_field",
                "field": field_name,
                "value": value,
                "confidence": effect.get("confidence"),
                "candidate_id": effect.get("candidate_id"),
                "memory_key": effect.get("memory_key"),
                "memory_type": effect.get("memory_type"),
                "rules_version": effect.get("source_need_rules_version"),
                "requires_server_application": True,
            })
    return {
        "schema_version": "preference-candidate-plan-v1",
        "status": "validated_candidate" if steps else "not_available",
        "source": "mem0_preference",
        "steps": steps,
        "candidate_ids": [item for item in candidate_ids if item],
        "applied": False,
        "requires_server_application": True,
        "not_authorization": True,
    }


def _match_candidate_to_source_need(
    memory_type: str,
    memory_key: str,
    source_needs: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not source_needs:
        return None, "source_need_missing"
    matching_type = False
    stale_rules = False
    for need in source_needs:
        if not isinstance(need, dict):
            continue
        if memory_type not in (need.get("memory_types") or []):
            continue
        matching_type = True
        if need.get("rules_version") != PREFERENCE_INTENT_RULES_VERSION:
            stale_rules = True
            continue
        keys = (need.get("allowed_keys") or {}).get(memory_type) or []
        if memory_key in keys:
            return need, None
    if stale_rules:
        return None, "source_need_rules_version_mismatch"
    if matching_type:
        return None, "source_need_key_not_allowed"
    return None, "source_need_type_not_allowed"


def evaluate_preference_candidates(
    state: dict[str, Any],
    raw_candidates: list[Any] | None,
    compilation: dict[str, Any] | None = None,
    *,
    source_needs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Type and validate preference candidates before they can affect a plan.

    Validation order is explicit current input -> inherited Task State conflict
    -> Registry semantics -> locked scope/authorization.  A candidate is never
    a scope or permission grant.  Even an accepted analysis/scope candidate is
    only a candidate; this function does not mutate known slots or execute it.
    """
    candidates = [
        candidate for candidate in (
            _normalise_candidate(raw) for raw in (raw_candidates or [])
        ) if candidate is not None
    ]
    if not candidates:
        return {
            "schema_version": "preference-context-v1",
            "status": "no_candidates",
            "accepted": [],
            "rejected": [],
            "items": [],
            "effects": [],
            "candidate_analysis_plan": _candidate_analysis_plan([]),
        }

    explicit = _explicit_field_values(compilation)
    inherited_raw = _inherited_slots(state)
    inherited = {
        name: slot.get("value") for name, slot in inherited_raw.items()
        if slot.get("value") not in (None, "")
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_id = _candidate_id(candidate.memory_key, candidate.memory_type, candidate.source_ref)
        metadata = {
            "candidate_id": candidate_id,
            "memory_key": candidate.memory_key,
            "memory_type": candidate.memory_type,
            "source_ref": candidate.source_ref,
            "confidence": candidate.confidence,
        }
        if candidate.memory_type == PREFERENCE_AUTHORITY_SECURITY:
            rejected.append({**metadata, "reason": "authority_or_security_preference"})
            continue
        matching_need, need_rejection = _match_candidate_to_source_need(
            candidate.memory_type, candidate.memory_key, source_needs,
        )
        if matching_need is None:
            rejected.append({**metadata, "reason": need_rejection or "source_need_missing"})
            continue
        allowed_fields = set(
            (matching_need.get("allowed_fields") or {}).get(candidate.memory_type) or []
        )
        allowed_values = matching_need.get("allowed_values") or {}
        if candidate.memory_type not in ALLOWED_PREFERENCE_TYPES:
            rejected.append({**metadata, "reason": "unknown_preference_type"})
            continue
        # WP0 pref-trigger-v1: a candidate without provider confidence is
        # indistinguishable from a zero-confidence hit.  Fail closed rather
        # than letting a missing field bypass the relevance threshold.
        if candidate.confidence < _preference_relevance_threshold():
            rejected.append({**metadata, "reason": "below_relevance_threshold"})
            continue

        if candidate.memory_type == PREFERENCE_PRESENTATION:
            presentation_field = {
                "response_style": "style",
                "report_style": "format",
                "response_language": "language",
            }.get(candidate.memory_key)
            if presentation_field not in allowed_fields:
                rejected.append({**metadata, "reason": "source_need_field_not_allowed"})
                continue
            if _contains_forbidden_scope_field(candidate.value):
                rejected.append({**metadata, "reason": "authorization_scope_cannot_be_modified"})
                continue
            if _presentation_explicitly_overridden(matching_need, presentation_field):
                rejected.append({**metadata, "reason": "explicit_input_wins"})
                continue
            candidate_value = candidate.value
            if isinstance(candidate_value, dict):
                candidate_value = candidate_value.get(presentation_field)
            if candidate_value in (None, ""):
                rejected.append({**metadata, "reason": "preference_value_missing"})
                continue
            exclusions = (matching_need.get("excluded_values") or {}).get(
                f"presentation.{presentation_field}", [],
            )
            if _canonical_preference_value(presentation_field, candidate_value) in {
                _canonical_preference_value(presentation_field, item) for item in exclusions
            }:
                rejected.append({**metadata, "reason": "source_need_value_excluded"})
                continue
            effect = {
                **metadata,
                "source_need_rules_version": matching_need.get("rules_version"),
                "kind": "presentation",
                "applies_to": "answer",
                "eligible_for_plan": False,
                "fields": {},
            }
            accepted.append({**metadata, "status": "accepted", "eligible_for_plan": False})
            effects.append(effect)
            value = {presentation_field: candidate_value}
            items.append({**metadata, "value": value, "effective_fields": value, "status": "accepted"})
            continue

        if candidate.memory_type == PREFERENCE_ANALYSIS_METHOD:
            field_values = _field_values(candidate)
            if not field_values:
                rejected.append({**metadata, "reason": "no_valid_analysis_fields"})
                continue
            validated: dict[str, Any] = {}
            field_rejections: list[dict[str, Any]] = [
                {"field": name, "reason": "outside_source_need"}
                for name in field_values if name not in allowed_fields
            ]
            for name, value in field_values.items():
                if name not in allowed_fields:
                    continue
                allowed = allowed_values.get(f"{candidate.memory_type}.{name}")
                if allowed and str(value) not in {str(item) for item in allowed}:
                    field_rejections.append({"field": name, "reason": "outside_source_need_allowed_values"})
                    continue
                excluded = (matching_need.get("excluded_values") or {}).get(
                    f"{candidate.memory_type}.{name}", [],
                )
                excluded_canonical = {
                    _canonical_preference_value(name, excluded_item)
                    for excluded_item in excluded
                }
                if excluded and isinstance(value, (list, tuple)):
                    value = [
                        item for item in value
                        if (
                            _date_value_is_excluded(item, excluded, state)
                            if name == "date"
                            else _canonical_preference_value(name, item) in excluded_canonical
                        ) is False
                    ]
                    if not value:
                        field_rejections.append({"field": name, "reason": "source_need_value_excluded"})
                        continue
                elif excluded and (
                    _date_value_is_excluded(value, excluded, state)
                    if name == "date"
                    else _canonical_preference_value(name, value) in excluded_canonical
                ):
                    field_rejections.append({"field": name, "reason": "source_need_value_excluded"})
                    continue
                ok, reason, normalised = _validate_analysis_field(
                    name, value, candidate, state, explicit, inherited,
                )
                if ok:
                    validated[name] = normalised
                else:
                    field_rejections.append({"field": name, "reason": reason})
            if not validated:
                rejected.append({
                    **metadata,
                    "reason": field_rejections[0]["reason"] if field_rejections else "candidate_rejected",
                    "field_rejections": field_rejections,
                })
                continue
            effect = {
                **metadata,
                "source_need_rules_version": matching_need.get("rules_version"),
                "kind": "analysis_method",
                "applies_to": "candidate_plan",
                "eligible_for_plan": True,
                "applied": False,
                "requires_server_plan_validation": True,
                "fields": validated,
            }
            accepted.append({
                **metadata,
                "status": "partially_accepted" if field_rejections else "accepted",
                "eligible_for_plan": True,
                "field_rejections": field_rejections,
            })
            effects.append(effect)
            items.append({
                **metadata,
                # Only the fields that passed explicit/Task/Registry/scope
                # validation may reach a downstream prompt.  Never copy the
                # raw provider value, because it may also contain a rejected
                # seller/region/authority field.
                "value": dict(validated),
                "effective_fields": dict(validated),
                "status": "accepted",
            })
            continue

        # scope_default
        scope_values = (
            candidate.suggested_fields
            if isinstance(candidate.suggested_fields, dict) and candidate.suggested_fields
            else candidate.value
        )
        if not isinstance(scope_values, dict) or not scope_values:
            rejected.append({**metadata, "reason": "no_scope_default_value"})
            continue
        forbidden_scope_fields = {
            str(key).casefold() for key in scope_values
            if str(key).casefold() in _FORBIDDEN_SCOPE_FIELDS
        }
        unsupported_region_fields = {
            str(key).casefold() for key in scope_values
            if str(key).casefold() in {"region", "regions", "area", "areas"}
        }
        if forbidden_scope_fields:
            rejected.append({**metadata, "reason": "authorization_scope_cannot_be_modified"})
            continue
        if unsupported_region_fields:
            rejected.append({**metadata, "reason": "region_filter_not_supported"})
            continue
        projected_scope_values = {
            str(key): value for key, value in scope_values.items()
            if (
                str(key) in allowed_fields
                or (str(key).casefold() in {"date_range", "period"}
                    and {"start_date", "end_date"}.issubset(allowed_fields))
            )
        }
        if not projected_scope_values:
            rejected.append({**metadata, "reason": "source_need_field_not_allowed"})
            continue
        scope_candidate = PreferenceCandidate(
            memory_key=candidate.memory_key,
            memory_type=candidate.memory_type,
            value=candidate.value,
            source_ref=candidate.source_ref,
            confidence=candidate.confidence,
            suggested_fields=projected_scope_values,
            scope=candidate.scope,
        )
        validated_scope, scope_rejections = _evaluate_scope_fields(
            scope_candidate, state, explicit, inherited,
            matching_need.get("excluded_values") or {},
        )
        if not validated_scope:
            rejected.append({
                **metadata,
                "reason": (scope_rejections[0]["reason"] if scope_rejections else "scope_candidate_rejected"),
                "field_rejections": scope_rejections,
            })
            continue
        effect = {
            **metadata,
            "source_need_rules_version": matching_need.get("rules_version"),
            "kind": "scope_default",
            "applies_to": "candidate_scope",
            "eligible_for_plan": True,
            "applied": False,
            "requires_scope_issuance": True,
            "fields": validated_scope,
        }
        accepted.append({
            **metadata,
            "status": "accepted",
            "eligible_for_plan": True,
            "field_rejections": scope_rejections,
        })
        effects.append(effect)
        items.append({
            **metadata,
            "value": dict(validated_scope),
            "effective_fields": dict(validated_scope),
            "status": "accepted",
        })

    status = (
        "accepted" if accepted and not rejected
        else "partially_accepted" if accepted
        else "rejected"
    )
    return {
        "schema_version": "preference-context-v1",
        "status": status,
        "accepted": accepted,
        "rejected": rejected,
        "items": items,
        "effects": effects,
        "candidate_analysis_plan": _candidate_analysis_plan(effects),
    }


def resolve_preference_context(
    state: dict[str, Any],
    *,
    provider: PreferenceProvider | None = None,
    compilation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve source needs and optional provider candidates for one turn."""
    resolved_compilation = compilation or state.get("context_compilation")
    instruction_constraints = _preference_instruction_constraints(
        str(state.get("user_message") or ""),
        compiled_constraints=resolved_compilation if isinstance(resolved_compilation, dict) else None,
    )
    needs = detect_preference_source_needs(state)
    lookup_request = _preference_lookup_request(_fold(instruction_constraints["positive_text"]))
    traced_exclusions: dict[str, set[str]] = {}
    for need in needs:
        for path, values in (need.get("excluded_values") or {}).items():
            traced_exclusions.setdefault(str(path), set()).update(str(value) for value in values)
    for path, values in instruction_constraints["excluded_values"].items():
        traced_exclusions.setdefault(str(path), set()).update(str(value) for value in values)
    intent_decision = {
        "rules_version": PREFERENCE_INTENT_RULES_VERSION,
        "blocked_categories": dict(instruction_constraints["blocked_categories"]),
        "excluded_value_fields": sorted(traced_exclusions),
        "excluded_value_counts": {
            path: len(values)
            for path, values in sorted(traced_exclusions.items())
        },
        "conflicts": list(instruction_constraints["conflicts"]),
        "unsupported_double_negation": bool(instruction_constraints["unsupported_double_negation"]),
        "other_intents": (
            ["preference_management"]
            if _contains_any_marker(
                _fold(str(state.get("user_message") or "")),
                _PREFERENCE_MANAGEMENT_MARKERS,
            )
            else []
        ),
        "negative_query_param_fields": sorted(instruction_constraints["negative_query_params"]),
        "negative_query_param_counts": {
            key: len(values)
            for key, values in instruction_constraints["negative_query_params"].items()
        },
    }
    retrieval = {
        "trigger_ids": [
            trigger for need in needs for trigger in need.get("trigger_ids") or []
        ],
        "top_k_per_type": PREFERENCE_TOP_K_PER_TYPE,
        "top_k_total": PREFERENCE_TOP_K_TOTAL,
        "threshold": _preference_relevance_threshold(),
        "hit_count": 0,
        "elapsed_ms": 0.0,
        "provider_calls": 0,
    }
    if not needs:
        context = {
            "schema_version": "preference-context-v1",
            "status": "registry_lookup_required" if lookup_request else "not_requested",
            "availability": "registry_lookup_required" if lookup_request else "not_requested",
            "lookup_request": lookup_request,
            "intent_decision": intent_decision,
            "source_needs": [],
            "accepted": [],
            "rejected": [],
            "items": [],
            "effects": [],
            "candidate_analysis_plan": _candidate_analysis_plan([]),
            "retrieval": retrieval,
        }
    elif provider is None:
        context = {
            "schema_version": "preference-context-v1",
            "status": "provider_unavailable",
            "availability": "provider_unavailable",
            "source_needs": [{**need, "status": "unavailable"} for need in needs],
            "intent_decision": intent_decision,
            "accepted": [],
            "rejected": [],
            "items": [],
            "effects": [],
            "candidate_analysis_plan": _candidate_analysis_plan([]),
            "retrieval": retrieval,
        }
    else:
        raw: list[Any] = []
        errors: list[dict[str, Any]] = []
        started = time.perf_counter()
        for need in needs:
            retrieval["provider_calls"] += 1
            try:
                found = provider.search(need=need, state=state)
                if isinstance(found, list):
                    raw.extend(found[:PREFERENCE_TOP_K_TOTAL])
            except Exception as exc:  # noqa: BLE001 - personalisation is optional
                errors.append({
                    "trigger_ids": need.get("trigger_ids"),
                    "error": str(getattr(exc, "reason", "") or type(exc).__name__)[:80],
                })
        retrieval["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        retrieval["hit_count"] = len(raw)
        evaluated = evaluate_preference_candidates(
            state, raw, compilation, source_needs=needs,
        )
        context = {
            **evaluated,
            "availability": "available" if not errors else "partial_provider_error",
            "source_needs": needs,
            "intent_decision": intent_decision,
            "provider_errors": errors,
            "retrieval": retrieval,
        }
    state["preference_context"] = context
    accepted_ids = [item.get("candidate_id") for item in context.get("accepted") or []]
    rejected = context.get("rejected") or []
    try:
        record_event("context.preference_source_decision", {
            "trace_id": state.get("trace_id"),
            "preference.status": context.get("status"),
            "preference.availability": context.get("availability"),
            "preference.trigger_ids": [
                trigger for need in needs for trigger in need.get("trigger_ids") or []
            ],
            "preference.rules_versions": ",".join(sorted({
                str(need.get("rules_version")) for need in needs if need.get("rules_version")
            })),
            "preference.intent_ids": ",".join(sorted({
                str(intent) for need in needs for intent in need.get("intent_ids") or []
            })),
            "preference.allowed_keys": ";".join(sorted({
                f"{memory_type}:{key}"
                for need in needs
                for memory_type, keys in (need.get("allowed_keys") or {}).items()
                for key in keys
            })),
            "preference.allowed_fields": ";".join(sorted({
                f"{memory_type}:{field_name}"
                for need in needs
                for memory_type, fields in (need.get("allowed_fields") or {}).items()
                for field_name in fields
            })),
            "preference.overridden_fields": ",".join(sorted({
                str(field_name)
                for need in needs for field_name in need.get("overridden_fields") or []
            })),
            "preference.blocked_categories": ",".join(sorted(
                intent_decision["blocked_categories"],
            )),
            "preference.excluded_value_fields": ",".join(
                intent_decision["excluded_value_fields"],
            ),
            "preference.conflict_fields": ",".join(intent_decision["conflicts"]),
            "preference.double_negation_fail_closed": intent_decision["unsupported_double_negation"],
            "preference.accepted_count": len(accepted_ids),
            "preference.rejected_count": len(rejected),
            "preference.top_k_total": retrieval["top_k_total"],
            "preference.threshold": retrieval["threshold"],
            "preference.hit_count": retrieval["hit_count"],
            "preference.elapsed_ms": retrieval["elapsed_ms"],
            "preference.provider_calls": retrieval["provider_calls"],
            "preference.rejection_reasons": sorted({
                str(item.get("reason")) for item in rejected if item.get("reason")
            }),
        })
    except Exception:  # noqa: BLE001 - observability must not break the turn
        pass
    return context


def apply_candidate_analysis_plan(
    state: dict[str, Any],
    compilation: dict[str, Any],
    preference_context: dict[str, Any],
) -> dict[str, Any]:
    """Apply only a server-validated, non-authorizing analysis preference.

    The first supported application is a single breakdown dimension. It can
    only fill a dimension slot that is otherwise absent, and only after the
    metric is fixed by the current request or trusted same-scope context. The
    downstream request-contract and query authorization checks still run.
    Other candidate fields remain proposals until their own execution
    semantics are implemented.
    """
    plan = preference_context.get("candidate_analysis_plan")
    if not isinstance(plan, dict) or not plan.get("steps"):
        return preference_context

    steps = [dict(item) for item in plan.get("steps") or [] if isinstance(item, dict)]
    for step in steps:
        step["applied"] = False
        step["application_reason"] = "unsupported_preference_application_field"

    if compilation.get("needs_clarification"):
        reason = str(compilation.get("clarification_reason") or "request_requires_clarification")
        for step in steps:
            step["application_reason"] = reason
        plan.update({"steps": steps, "status": "not_applied_requires_clarification", "applied": False})
        return preference_context

    contract = compilation.get("request_contract")
    if not isinstance(contract, dict):
        for step in steps:
            step["application_reason"] = "request_contract_missing"
        plan.update({"steps": steps, "status": "not_applied_contract_missing", "applied": False})
        return preference_context

    metric_id = str(contract.get("metric_id") or "").strip()
    if not metric_id:
        for step in steps:
            step["application_reason"] = "metric_required_before_preference_application"
        plan.update({"steps": steps, "status": "not_applied_metric_required", "applied": False})
        return preference_context
    metric_ok, metric_reason = _metric_allowed(metric_id, state)
    if not metric_ok:
        for step in steps:
            step["application_reason"] = metric_reason or "metric_rejected"
        plan.update({"steps": steps, "status": "not_applied_metric_rejected", "applied": False})
        return preference_context

    slots = compilation.setdefault("known_slots", {})
    applied_candidate_ids: list[str] = []
    applied_fields: list[str] = []
    for step in steps:
        if step.get("field") != "dimension":
            continue
        candidate_type = str(step.get("memory_type") or "")
        candidate_key = str(step.get("memory_key") or "")
        matching_need, need_rejection = _match_candidate_to_source_need(
            candidate_type, candidate_key,
            preference_context.get("source_needs") if isinstance(preference_context.get("source_needs"), list) else None,
        )
        if matching_need is None:
            step["application_reason"] = need_rejection or "source_need_missing"
            continue
        if step.get("rules_version") != PREFERENCE_INTENT_RULES_VERSION:
            step["application_reason"] = "source_need_rules_version_mismatch"
            continue
        allowed_fields = set(
            (matching_need.get("allowed_fields") or {}).get(candidate_type) or []
        )
        if "dimension" not in allowed_fields:
            step["application_reason"] = "source_need_field_not_allowed"
            continue
        excluded_dimension_values = (matching_need.get("excluded_values") or {}).get(
            "analysis_method.dimension", [],
        )
        if _canonical_preference_value("dimension", step.get("value")) in {
            _canonical_preference_value("dimension", item)
            for item in excluded_dimension_values
        }:
            step["application_reason"] = "source_need_value_excluded"
            continue
        candidate_id = str(step.get("candidate_id") or "") or None
        if not candidate_id:
            step["application_reason"] = "candidate_id_missing"
            continue
        existing_dimension = slots.get("dimension")
        if isinstance(existing_dimension, dict) and existing_dimension.get("source") == "validated_preference_candidate":
            if existing_dimension.get("candidate_id") == candidate_id:
                step["applied"] = True
                step["application_reason"] = "already_applied_to_request_contract"
                applied_fields.append("dimension")
                if candidate_id:
                    applied_candidate_ids.append(candidate_id)
            else:
                step["application_reason"] = "another_preference_candidate_already_applied"
            continue
        if existing_dimension:
            step["application_reason"] = "explicit_or_inherited_dimension_wins"
            continue
        dimension = str(step.get("value") or "").strip()
        if not dimension:
            step["application_reason"] = "empty_dimension_candidate"
            continue
        dimension_ok, dimension_reason = _dimension_allowed(dimension, {metric_id}, state)
        if not dimension_ok:
            step["application_reason"] = dimension_reason or "dimension_rejected"
            continue
        slots["dimension"] = {
            "value": dimension,
            "source": "validated_preference_candidate",
            "confidence": max(0.0, min(1.0, _as_float(step.get("confidence")))),
            "source_turn_id": None,
            **({"candidate_id": candidate_id} if candidate_id else {}),
        }
        compilation.setdefault("context_sources", []).append("mem0")
        contract["dimension"] = dimension
        step["applied"] = True
        step["application_reason"] = "applied_to_sealed_request_contract"
        applied_fields.append("dimension")
        if candidate_id:
            applied_candidate_ids.append(candidate_id)

    if applied_candidate_ids:
        from app.context.request_contract import seal_request_contract

        contract["preference_application"] = {
            "source": "validated_mem0_candidate",
            "candidate_ids": list(dict.fromkeys(applied_candidate_ids)),
            "fields": list(dict.fromkeys(applied_fields)),
        }
        contract.pop("contract_hash", None)
        compilation["request_contract"] = seal_request_contract(contract)
        compilation["request_contract_hash"] = compilation["request_contract"]["contract_hash"]
        state["request_contract_hash"] = compilation["request_contract_hash"]

    all_applied = bool(steps) and all(bool(step.get("applied")) for step in steps)
    any_applied = any(bool(step.get("applied")) for step in steps)
    plan.update({
        "steps": steps,
        "applied": any_applied,
        "requires_server_application": not all_applied,
        "status": "applied" if all_applied else "partially_applied" if any_applied else "not_applied",
        "applied_candidate_ids": list(dict.fromkeys(applied_candidate_ids)),
    })
    return preference_context


def _normalise_relation(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return {
        "continue": "continue",
        "independent": "new",
        "new": "new",
        "new_task": "new",
        "parallel_new": "parallel_new",
        "parallel_new_task": "parallel_new",
        "clarify": "clarify",
        "clarification": "clarify",
    }.get(raw, "new")


def _task_relation_metadata(state: dict[str, Any], compilation: dict[str, Any]) -> dict[str, Any]:
    decision = state.get("turn_relation_decision") or {}
    if not isinstance(decision, dict):
        decision = {}
    action = _normalise_relation(decision.get("action") or decision.get("relation"))
    if (
        action == "new"
        and str(decision.get("decision") or "").upper() == "PARALLEL_NEW_TASK"
    ):
        action = "parallel_new"
    if not decision and compilation.get("relation") == "continue":
        action = "continue"
    candidates = decision.get("task_candidates") or state.get("task_relation_candidates") or []
    nested = (decision.get("context_resolution") or {}).get("task_relation") or {}
    if not candidates and isinstance(nested, dict):
        candidates = nested.get("task_candidates") or []
    candidate_ids = [
        str(item.get("task_id")) for item in candidates
        if isinstance(item, dict) and item.get("task_id")
    ]
    return {
        "action": action,
        "task_id": decision.get("task_id") or state.get("task_id"),
        "reason": decision.get("reason"),
        "decision_driver": (
            decision.get("decision_driver") or nested.get("decision_driver")
            or ("workflow_fallback" if decision else "resolver")
        ),
        "candidate_task_ids": candidate_ids or list(nested.get("candidate_task_ids") or []),
        "scope_conflict": bool(decision.get("scope_conflict") or nested.get("scope_conflict")),
        "ambiguous": bool(decision.get("ambiguous") or nested.get("ambiguous") or action == "clarify"),
    }


def _preference_metadata(context: dict[str, Any]) -> dict[str, Any]:
    effects = []
    for effect in context.get("effects") or []:
        if not isinstance(effect, dict):
            continue
        effects.append({
            key: effect.get(key)
            for key in (
                "candidate_id", "memory_key", "memory_type", "kind", "applies_to",
                "eligible_for_plan", "applied", "requires_server_plan_validation",
                "requires_scope_issuance",
            )
            if effect.get(key) is not None
        })
    plan = context.get("candidate_analysis_plan") or {}
    source_need_metadata = []
    for raw_need in context.get("source_needs") or []:
        if not isinstance(raw_need, dict):
            continue
        need = {key: value for key, value in raw_need.items() if key != "excluded_values"}
        exclusions = raw_need.get("excluded_values") or {}
        need["excluded_value_fields"] = sorted(exclusions)
        need["excluded_value_counts"] = {
            str(path): len(values) if isinstance(values, (list, tuple, set)) else 0
            for path, values in exclusions.items()
        }
        source_need_metadata.append(need)
    intent_decision = context.get("intent_decision") or {}
    safe_intent_decision = {
        "rules_version": intent_decision.get("rules_version"),
        "blocked_categories": dict(intent_decision.get("blocked_categories") or {}),
        "excluded_value_fields": list(intent_decision.get("excluded_value_fields") or []),
        "excluded_value_counts": dict(intent_decision.get("excluded_value_counts") or {}),
        "conflicts": list(intent_decision.get("conflicts") or []),
        "unsupported_double_negation": bool(intent_decision.get("unsupported_double_negation")),
        "other_intents": list(intent_decision.get("other_intents") or []),
        "negative_query_param_fields": list(intent_decision.get("negative_query_param_fields") or []),
        "negative_query_param_counts": dict(intent_decision.get("negative_query_param_counts") or {}),
    }
    return {
        "schema_version": context.get("schema_version"),
        "status": context.get("status"),
        "availability": context.get("availability"),
        "lookup_request": dict(context.get("lookup_request") or {}),
        "source_needs": source_need_metadata,
        "intent_decision": safe_intent_decision,
        "accepted": list(context.get("accepted") or []),
        "rejected": list(context.get("rejected") or []),
        "provider_errors": [
            {
                "trigger_ids": list(item.get("trigger_ids") or []),
                "reason": str(item.get("error") or "")[:80],
            }
            for item in (context.get("provider_errors") or [])
            if isinstance(item, dict)
        ],
        "effects": effects,
        "candidate_analysis_plan": {
            "schema_version": plan.get("schema_version"),
            "status": plan.get("status"),
            "candidate_ids": list(plan.get("candidate_ids") or []),
            "applied_candidate_ids": list(plan.get("applied_candidate_ids") or []),
            "step_fields": [
                item.get("field") for item in (plan.get("steps") or [])
                if isinstance(item, dict) and item.get("field")
            ],
            "application_reasons": [
                str(item.get("application_reason"))
                for item in (plan.get("steps") or [])
                if isinstance(item, dict) and item.get("application_reason")
            ],
            "applied": bool(plan.get("applied")),
            "requires_server_application": bool(plan.get("requires_server_application")),
        },
        "retrieval": dict(context.get("retrieval") or {}),
    }


def _candidate_fields(compilation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "field": name,
            "source": slot.get("source"),
            "candidate_id": slot.get("candidate_id"),
        }
        for name, slot in (compilation.get("known_slots") or {}).items()
        if isinstance(slot, dict) and slot.get("source") in {
            "preference_candidate", "validated_preference_candidate",
        }
    ]


def enrich_turn_resolution(
    state: dict[str, Any],
    compilation: dict[str, Any],
    base_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge WP1 TurnResolution with WP2 task/preference/correction metadata."""
    resolution = dict(base_resolution or state.get("turn_resolution") or {})
    task_relation = _task_relation_metadata(state, compilation)
    relation = _normalise_relation(resolution.get("relation"))
    if task_relation["action"] == "clarify":
        relation = "clarify"
    elif task_relation["action"] == "parallel_new":
        relation = "parallel_new"
    elif task_relation["action"] in {"continue", "new"} and task_relation.get("decision_driver") != "resolver":
        relation = task_relation["action"]
    if compilation.get("needs_clarification"):
        relation = "clarify"

    preference = state.get("preference_context")
    if not isinstance(preference, dict) or not preference:
        preference = resolve_preference_context(state, compilation=compilation)
    preference_meta = _preference_metadata(preference)
    needs = list(preference_meta.get("source_needs") or [])
    if not needs:
        needs = detect_preference_source_needs(state)
        preference_meta["source_needs"] = needs

    task_version = (
        (state.get("task_progress_context") or {}).get("task_state_version")
        or state.get("task_state_version")
    )
    inherited = []
    for item in resolution.get("inherited_fields") or []:
        if not isinstance(item, dict):
            continue
        updated = dict(item)
        if (
            updated.get("source") == "task_state"
            and updated.get("source_task_state_version") is None
            and task_version is not None
        ):
            updated["source_task_state_version"] = _as_int(task_version)
        inherited.append(updated)

    explicit_fields = list(resolution.get("explicit_fields") or [])
    for field_name in compilation.get("known_slots") or {}:
        slot = (compilation.get("known_slots") or {}).get(field_name) or {}
        if slot.get("source") in {"preference_candidate", "validated_preference_candidate"} and field_name not in explicit_fields:
            # Candidate fields are deliberately not explicit.
            continue
        if slot.get("source") == "current_turn_correction" and field_name not in explicit_fields:
            explicit_fields.append(field_name)

    confidence = _as_float(resolution.get("confidence"), 1.0)
    if task_relation["action"] == "clarify" or compilation.get("needs_clarification"):
        confidence = min(confidence, 0.5)
    if task_relation.get("ambiguous"):
        confidence = min(confidence, 0.5)
    if any(
        isinstance(item, dict) and item.get("status") in {"unsupported", "rejected"}
        for item in (compilation.get("scope_candidates") or [])
    ):
        confidence = min(confidence, 0.4)

    resolution.update({
        "schema_version": "turn-resolution-v1",
        "resolver_version": TURN_RESOLVER_VERSION,
        "policy_version": TURN_RESOLUTION_POLICY_VERSION,
        "relation": relation,
        "task_relation": task_relation,
        "explicit_fields": list(dict.fromkeys(explicit_fields)),
        "inherited_fields": inherited,
        "user_changes": list(compilation.get("user_changes") or resolution.get("user_changes") or []),
        "unresolved_fields": list(
            compilation.get("missing_slots") or resolution.get("unresolved_fields") or []
        ),
        "known_slots": dict(compilation.get("known_slots") or resolution.get("known_slots") or {}),
        "candidate_fields": _candidate_fields(compilation),
        "scope_candidates": list(compilation.get("scope_candidates") or []),
        "source_needs": needs,
        "preference": preference_meta,
        "needs_clarification": bool(
            resolution.get("needs_clarification") or compilation.get("needs_clarification")
            or task_relation["action"] == "clarify"
        ),
        "clarification_reason": (
            task_relation.get("reason") if task_relation["action"] == "clarify"
            else compilation.get("clarification_reason") or resolution.get("clarification_reason")
        ),
        "confidence": confidence,
        "created_at": resolution.get("created_at") or "",
    })
    return resolution
