"""Server-owned result context for same-scope follow-up questions.

A follow-up such as “哪个品类最高？” or “这个数字的证据是什么？” must answer
from the result the user already saw.  This module builds a minimal, bounded
projection of a completed query result, validates it before reuse, and emits a
deterministic presentation without issuing another business query.

It never stores the raw question, SQL, or free-form model text.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

RESULT_CONTEXT_VERSION = "result-context-v2"
MAX_ROWS = 20
MAX_FACTS = 5
MAX_EVIDENCE_IDS = 40

_EXTREMUM = re.compile(
    r"(?:哪个|哪一个|哪一项|哪一类|哪类|哪款|哪种)[^。！？\n]{0,12}(最高|最大|最低|最小)"
)
_TOP_N = re.compile(
    r"(?:只看|列出|取|保留|显示)\s*(?:前|top)\s*([0-9一二三四五六七八九十两]+)"
    r"|\btop\s*([0-9一二三四五六七八九十两]+)",
    re.IGNORECASE,
)
_COMPARISON_DIRECTION = re.compile(
    r"(?:哪个月|哪一期|哪期|哪个|哪边|谁)[^。！？\n]{0,8}(更高|更低|高|低)"
)
_PROVENANCE_MARKERS = ("证据", "依据", "来源", "出处", "怎么算", "哪里来的", "可信")
_PROVENANCE_REFERENCES = (
    "这个数字", "这个值", "这个结果", "该数字", "该值", "刚才", "上一轮", "前面",
)
_PREVIOUS_RESULT_REFERENCES = ("上一轮", "刚才", "上次", "此前", "前面查询")
_RESULT_VALUE_REQUESTS = ("数值", "是多少", "具体值", "金额")
_NO_DATA_NEXT_STEP_MARKERS = ("下一步", "怎么办", "接下来", "怎么处理")
_REPORT_ACTION_MARKERS = (
    "简短报告", "经营报告", "经营摘要", "行动建议", "下一步建议",
)
_NO_DATA_STATUSES = {"not_found", "insufficient_data"}
_QUERY_PERIOD_FIELDS = (
    "metric_date", "start_date", "end_date", "baseline_date", "current_date",
    "baseline_start", "baseline_end", "current_start", "current_end",
)
_QUERY_TYPES = {
    "metric_value", "metric_trend", "metric_breakdown", "metric_compare",
    "metric_contribution",
}
_TOP_N_WORDS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_DIMENSION_LABELS = {
    "product_category": "商品品类",
    "payment_type": "支付方式",
    "seller_state": "卖家州",
    "customer_state": "客户州",
    "customer_city": "客户城市",
    "month_position": "月内位置",
}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").strip()[:limit]


def _display_name(metric_id: str) -> str:
    try:
        from app.metrics.registry import get_registry

        return str(get_registry().get(metric_id).display_name)
    except Exception:  # noqa: BLE001 - display fallback must not break the answer
        return metric_id


def _dimension_label(dimension: Any) -> str:
    value = _clean_text(dimension, limit=80)
    return _DIMENSION_LABELS.get(value, value or "分组")


def _metric_unit(metric_id: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    try:
        from app.metrics.registry import get_registry

        return str(get_registry().get(metric_id).unit or "")
    except Exception:  # noqa: BLE001
        return ""


def _query_metric_id(item: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    breakdown = result.get("breakdown_result") if isinstance(result.get("breakdown_result"), Mapping) else {}
    result_parsed = result.get("parsed") if isinstance(result.get("parsed"), Mapping) else {}
    item_parsed = item.get("parsed") if isinstance(item.get("parsed"), Mapping) else {}
    arguments = item.get("arguments") if isinstance(item.get("arguments"), Mapping) else {}
    for source in (breakdown, result_parsed, item_parsed, arguments):
        metric_id = _clean_text(source.get("metric_id") or source.get("metric_name"), limit=80)
        if metric_id:
            return metric_id
    return ""


def _source_evidence_ids(item: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    raw_values: list[Any] = [item.get("evidence_id")]
    nested = result.get("evidence_ids")
    if isinstance(nested, list):
        raw_values.extend(
            value.get("evidence_id") if isinstance(value, Mapping) else value
            for value in nested
        )
    for raw in raw_values:
        value = _clean_text(raw, limit=120) if isinstance(raw, str) else ""
        if value and value not in found:
            found.append(value)
    return found[:MAX_EVIDENCE_IDS]


def _extract_rows(
    query_results: Any, *, fallback_metric_id: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    dimensions: set[str] = set()
    if not isinstance(query_results, list):
        return rows, None
    for item in query_results:
        if not isinstance(item, Mapping):
            continue
        result = item.get("result") if isinstance(item.get("result"), Mapping) else {}
        breakdown = result.get("breakdown_result") if isinstance(result, Mapping) else None
        if not isinstance(breakdown, Mapping):
            continue
        dimension = _clean_text(breakdown.get("dimension"), limit=80)
        metric_id = _query_metric_id(item, result) or _clean_text(fallback_metric_id, limit=80)
        # A row without a resolvable metric cannot be safely ranked or unit-labelled.
        if not metric_id:
            continue
        unit = _clean_text(breakdown.get("unit"), limit=40) or _metric_unit(metric_id)
        period = _safe_query_period({
            **(item.get("arguments") if isinstance(item.get("arguments"), Mapping) else {}),
            **(item.get("parsed") if isinstance(item.get("parsed"), Mapping) else {}),
            **(result.get("parsed") if isinstance(result.get("parsed"), Mapping) else {}),
        })
        source_ids = _source_evidence_ids(item, result)
        if dimension:
            dimensions.add(dimension)
        for raw in breakdown.get("items") or []:
            if not isinstance(raw, Mapping):
                continue
            key = _clean_text(
                raw.get("dimension_value") or raw.get("key") or raw.get("label"),
                limit=120,
            )
            value = _finite_number(raw.get("value"))
            if key and value is not None:
                rows.append({
                    "key": key,
                    "value": value,
                    "metric_id": metric_id,
                    "dimension": dimension or None,
                    "unit": _clean_text(raw.get("unit"), limit=40) or unit,
                    "query_period": period,
                    "source_evidence_ids": source_ids,
                })
            if len(rows) >= MAX_ROWS:
                return rows, next(iter(dimensions)) if len(dimensions) == 1 else None
    return rows, next(iter(dimensions)) if len(dimensions) == 1 else None


def _extract_facts(presentation: Mapping[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for raw in presentation.get("facts") or []:
        if not isinstance(raw, Mapping):
            continue
        metric_id = _clean_text(raw.get("metric_name"), limit=80)
        value = _finite_number(raw.get("value"))
        if not metric_id or value is None:
            continue
        facts.append({
            "metric_name": metric_id,
            "display_name": _clean_text(raw.get("display_name"), limit=80),
            "value": value,
            "unit": _clean_text(raw.get("unit"), limit=40),
            "date_or_range": _clean_text(raw.get("date_or_range"), limit=120),
            "evidence_ids": [
                _clean_text(item, limit=120) for item in (raw.get("evidence_ids") or [])
                if isinstance(item, str) and item
            ][:MAX_EVIDENCE_IDS],
        })
        if len(facts) >= MAX_FACTS:
            break
    return facts


def _extract_comparison(
    presentation: Mapping[str, Any],
    query_results: Any,
) -> dict[str, Any] | None:
    changes = presentation.get("changes") or []
    raw = changes[0] if isinstance(changes, list) and changes and isinstance(changes[0], Mapping) else None
    if raw is not None:
        metric_id = _clean_text(raw.get("metric_name"), limit=80)
        baseline = _finite_number(raw.get("baseline_value"))
        current = _finite_number(raw.get("current_value"))
        if metric_id and baseline is not None and current is not None:
            return {
                "metric_name": metric_id,
                "baseline_value": baseline,
                "current_value": current,
                "delta_value": _finite_number(raw.get("delta_value")),
                "delta_ratio": _finite_number(raw.get("delta_ratio")),
                "unit": _clean_text(raw.get("unit"), limit=40),
                "baseline_period": _clean_text(raw.get("baseline_period"), limit=120),
                "current_period": _clean_text(raw.get("current_period"), limit=120),
                "evidence_ids": [
                    _clean_text(item, limit=120) for item in (raw.get("evidence_ids") or [])
                    if isinstance(item, str) and item
                ][:MAX_EVIDENCE_IDS],
            }
    if not isinstance(query_results, list):
        return None
    for item in query_results:
        if not isinstance(item, Mapping):
            continue
        result = item.get("result") if isinstance(item.get("result"), Mapping) else {}
        compare = result.get("compare_result") if isinstance(result, Mapping) else None
        if not isinstance(compare, Mapping):
            continue
        metric_id = _clean_text(compare.get("metric_name"), limit=80)
        baseline = _finite_number(compare.get("baseline_value", compare.get("right_value")))
        current = _finite_number(compare.get("current_value", compare.get("left_value")))
        if not metric_id or baseline is None or current is None:
            continue
        return {
            "metric_name": metric_id,
            "baseline_value": baseline,
            "current_value": current,
            "delta_value": _finite_number(compare.get("delta_value")),
            "delta_ratio": _finite_number(compare.get("delta_ratio")),
            "unit": _clean_text(compare.get("unit"), limit=40),
            "baseline_period": _clean_text(
                compare.get("baseline_period") or compare.get("right_date"), limit=120
            ),
            "current_period": _clean_text(
                compare.get("current_period") or compare.get("left_date"), limit=120
            ),
            "evidence_ids": [
                _clean_text(item.get("evidence_id"), limit=120)
                for item in (result.get("evidence_ids") or []) if isinstance(item, Mapping)
                and item.get("evidence_id")
            ][:MAX_EVIDENCE_IDS],
        }
    return None


def _extract_scope(result: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("explorer_context", "controlled_analysis_scope"):
        raw = result.get(key)
        if not isinstance(raw, Mapping):
            continue
        scope = {
            field: _clean_text(raw.get(field), limit=120)
            for field in ("seller_id", "start_date", "end_date")
            if raw.get(field)
        }
        if scope:
            return scope
    return None


def _safe_query_period(value: Any) -> dict[str, str]:
    """Keep only ISO calendar dates from server-owned query arguments."""

    if not isinstance(value, Mapping):
        return {}
    from datetime import date

    period: dict[str, str] = {}
    for field in _QUERY_PERIOD_FIELDS:
        raw = value.get(field)
        if not isinstance(raw, str):
            continue
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            continue
        if parsed.isoformat() == raw:
            period[field] = raw
    return period


def _extract_no_data_result(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read a typed no-data status without carrying arbitrary response text."""

    for item in result.get("query_results") or []:
        if not isinstance(item, Mapping):
            continue
        envelope = item.get("result") if isinstance(item.get("result"), Mapping) else {}
        status = _clean_text(envelope.get("status"), limit=40)
        if status not in _NO_DATA_STATUSES:
            continue
        parsed = envelope.get("parsed") if isinstance(envelope.get("parsed"), Mapping) else {}
        arguments = item.get("arguments") if isinstance(item.get("arguments"), Mapping) else {}
        metric_id = _clean_text(
            parsed.get("metric_name") or parsed.get("metric_id")
            or arguments.get("metric_name") or arguments.get("metric_id"),
            limit=80,
        )
        query_type = _clean_text(
            parsed.get("query_type") or arguments.get("query_type"), limit=40,
        )
        if query_type not in _QUERY_TYPES:
            query_type = "metric_value"
        period = _safe_query_period(parsed) or _safe_query_period(arguments)
        reason = "data_not_found" if status == "not_found" else "insufficient_data"
        evidence_id = _clean_text(item.get("evidence_id"), limit=120)
        return {
            "metric_id": metric_id,
            "query_type": query_type,
            "result_status": status,
            "result_reason": reason,
            "query_period": period,
            "evidence_ids": [evidence_id] if evidence_id else [],
        }
    return None


def _extract_query_period(query_results: Any) -> dict[str, str]:
    """Keep the first server-planned ISO period for a completed query result."""

    if not isinstance(query_results, list):
        return {}
    for item in query_results:
        if not isinstance(item, Mapping):
            continue
        parsed = item.get("parsed") if isinstance(item.get("parsed"), Mapping) else {}
        envelope = item.get("result") if isinstance(item.get("result"), Mapping) else {}
        result_parsed = envelope.get("parsed") if isinstance(envelope.get("parsed"), Mapping) else {}
        arguments = item.get("arguments") if isinstance(item.get("arguments"), Mapping) else {}
        period = _safe_query_period({**arguments, **parsed, **result_parsed})
        if period:
            return period
    return {}


def _context_evidence_record(
    *,
    source_turn_id: str,
    source_trace_id: str,
    metric_id: str,
    query_type: str,
    source_evidence_ids: list[str],
    payload: Mapping[str, Any],
    dimension: str | None = None,
    period: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create an explicitly derived evidence record for a persisted result snapshot."""

    from app.agents.contracts import stable_evidence_id

    source_ref = f"result_context:{source_turn_id or 'unknown'}:{source_trace_id or 'unknown'}"
    content = {
        "summary": "上一轮查询结果快照",
        "metric_id": metric_id,
        "query_type": query_type,
        "dimension": dimension,
        "period": dict(period or {}),
        "source_turn_id": source_turn_id,
        "source_trace_id": source_trace_id,
        "source_evidence_ids": list(dict.fromkeys(source_evidence_ids))[:MAX_EVIDENCE_IDS],
        **dict(payload),
    }
    return {
        "evidence_id": stable_evidence_id("result_context", source_ref, content),
        "source_ref": source_ref,
        "content": content,
    }


def _normalize_metric_phrase(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _explicit_metric_ids(question: str) -> set[str]:
    """Resolve only registered exact aliases explicitly present in a follow-up."""

    normalized = _normalize_metric_phrase(question)
    if not normalized:
        return set()
    try:
        from app.metrics.registry import get_registry

        metrics = get_registry().list()
    except Exception:  # noqa: BLE001 - inability to resolve stays ambiguous/fail-closed
        return set()
    matches: set[str] = set()
    for metric in metrics:
        phrases = {
            str(metric.metric_id), str(metric.display_name),
            *(str(alias) for alias in metric.aliases),
            *(str(alias) for alias in metric.planning.exact_aliases),
        }
        if any(
            phrase and _normalize_metric_phrase(phrase) in normalized
            for phrase in phrases
        ):
            matches.add(str(metric.metric_id))
    return matches


def _select_metric_context(
    context: Mapping[str, Any], metric_id: str,
) -> dict[str, Any]:
    selected = dict(context)
    selected["metric_id"] = metric_id
    selected["metric_ids"] = [metric_id]
    selected["rows"] = [
        row for row in (context.get("rows") or [])
        if isinstance(row, Mapping) and str(row.get("metric_id") or "") == metric_id
    ]
    selected["facts"] = [
        fact for fact in (context.get("facts") or [])
        if isinstance(fact, Mapping) and str(fact.get("metric_name") or "") == metric_id
    ]
    comparison = context.get("comparison")
    selected["comparison"] = (
        comparison if isinstance(comparison, Mapping)
        and str(comparison.get("metric_name") or "") == metric_id else None
    )
    records = [
        record for record in (context.get("evidence_records") or [])
        if isinstance(record, Mapping)
        and str((record.get("content") or {}).get("metric_id") or "") == metric_id
    ]
    selected["evidence_records"] = records
    selected["evidence_ids"] = [str(record["evidence_id"]) for record in records]
    selected["query_type"] = (
        "metric_compare" if selected["comparison"]
        else "metric_breakdown" if selected["rows"] else "metric_value"
    )
    dimensions = {
        str(row.get("dimension")) for row in selected["rows"] if row.get("dimension")
    }
    selected["dimension"] = next(iter(dimensions)) if len(dimensions) == 1 else None
    periods = {
        tuple(sorted((row.get("query_period") or {}).items()))
        for row in selected["rows"] if isinstance(row.get("query_period"), Mapping)
    }
    if len(periods) == 1:
        selected["query_period"] = dict(next(iter(periods)))
    return selected


def build_result_context(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build the minimal same-scope result projection persisted with a turn."""

    if not isinstance(result, Mapping):
        return None
    presentation = result.get("analysis_presentation")
    if not isinstance(presentation, Mapping):
        return None
    status = _clean_text(presentation.get("status"), limit=40)
    if status not in {"completed", "partial"}:
        return None
    facts = _extract_facts(presentation)
    comparison = _extract_comparison(presentation, result.get("query_results"))
    no_data = _extract_no_data_result(result)
    semantic = presentation.get("semantic_interpretation")
    semantic = semantic if isinstance(semantic, Mapping) else {}
    query_results = result.get("query_results")
    query_metrics = {
        _query_metric_id(item, item.get("result") if isinstance(item.get("result"), Mapping) else {})
        for item in query_results if isinstance(item, Mapping)
    } if isinstance(query_results, list) else set()
    query_metrics.discard("")
    known_metrics = {
        *query_metrics,
        *(str(fact["metric_name"]) for fact in facts),
        *([str(comparison["metric_name"])] if comparison else []),
        *([str(no_data["metric_id"])] if no_data and no_data.get("metric_id") else []),
        *([_clean_text(semantic.get("metric_id"), limit=80)] if semantic.get("metric_id") else []),
    }
    known_metrics.discard("")
    fallback_metric_id = next(iter(known_metrics)) if len(known_metrics) == 1 else None
    rows, dimension = _extract_rows(query_results, fallback_metric_id=fallback_metric_id)
    metric_ids = sorted(known_metrics | {str(row["metric_id"]) for row in rows})
    if not metric_ids or not (rows or comparison or facts or no_data):
        return None
    metric_id = metric_ids[0] if len(metric_ids) == 1 else ""
    source_turn_id = _clean_text(
        result.get("persistence_turn_id") or result.get("chat_turn_id"), limit=120
    )
    source_trace_id = _clean_text(result.get("trace_id"), limit=120)
    evidence_records: list[dict[str, Any]] = []
    source_ids_by_metric: dict[str, list[str]] = {}
    if isinstance(query_results, list):
        for query_result in query_results:
            if not isinstance(query_result, Mapping):
                continue
            query_payload = query_result.get("result") if isinstance(query_result.get("result"), Mapping) else {}
            query_metric = _query_metric_id(query_result, query_payload)
            if not query_metric:
                continue
            source_ids_by_metric.setdefault(query_metric, [])
            for source_id in _source_evidence_ids(query_result, query_payload):
                if source_id not in source_ids_by_metric[query_metric]:
                    source_ids_by_metric[query_metric].append(source_id)
    for row in rows:
        source_ids = list(row.get("source_evidence_ids") or [])
        record = _context_evidence_record(
            source_turn_id=source_turn_id,
            source_trace_id=source_trace_id,
            metric_id=str(row["metric_id"]),
            query_type="metric_breakdown",
            source_evidence_ids=source_ids,
            dimension=row.get("dimension"),
            period=row.get("query_period"),
            payload={"row": {key: row.get(key) for key in ("key", "value", "unit")}},
        )
        row["evidence_ids"] = [record["evidence_id"]]
        evidence_records.append(record)
    for fact in facts:
        source_ids = list(fact.get("evidence_ids") or source_ids_by_metric.get(str(fact["metric_name"]), []))
        record = _context_evidence_record(
            source_turn_id=source_turn_id,
            source_trace_id=source_trace_id,
            metric_id=str(fact["metric_name"]),
            query_type="metric_value",
            source_evidence_ids=source_ids,
            period={},
            payload={"fact": {
                key: fact.get(key)
                for key in ("display_name", "value", "unit", "date_or_range")
            }},
        )
        fact["source_evidence_ids"] = source_ids
        fact["evidence_ids"] = [record["evidence_id"]]
        evidence_records.append(record)
    if comparison:
        source_ids = list(comparison.get("evidence_ids") or source_ids_by_metric.get(str(comparison["metric_name"]), []))
        record = _context_evidence_record(
            source_turn_id=source_turn_id,
            source_trace_id=source_trace_id,
            metric_id=str(comparison["metric_name"]),
            query_type="metric_compare",
            source_evidence_ids=source_ids,
            period={},
            payload={"comparison": {
                key: comparison.get(key)
                for key in (
                    "baseline_value", "current_value", "delta_value", "delta_ratio",
                    "unit", "baseline_period", "current_period",
                )
            }},
        )
        comparison["source_evidence_ids"] = source_ids
        comparison["evidence_ids"] = [record["evidence_id"]]
        evidence_records.append(record)
    evidence_records = evidence_records[:MAX_EVIDENCE_IDS]
    evidence_ids = [record["evidence_id"] for record in evidence_records]
    return {
        "version": RESULT_CONTEXT_VERSION,
        "source_turn_id": source_turn_id,
        "source_trace_id": source_trace_id,
        "scope_fingerprint": _clean_text(
            result.get("current_query_scope_fingerprint"), limit=200
        ),
        "data_version_manifest_hash": _clean_text(
            result.get("data_version_manifest_hash"), limit=200
        ),
        "metric_id": metric_id,
        "metric_ids": metric_ids,
        "dimension": dimension,
        "operation": "result_followup_source",
        "rows": rows[:MAX_ROWS],
        "facts": facts[:MAX_FACTS],
        "comparison": comparison,
        "evidence_ids": evidence_ids[:MAX_EVIDENCE_IDS],
        "evidence_records": evidence_records,
        "result_status": no_data.get("result_status") if no_data else status,
        "result_reason": no_data.get("result_reason") if no_data else None,
        "query_period": (
            no_data.get("query_period") if no_data
            else _extract_query_period(result.get("query_results"))
        ),
        "query_type": no_data.get("query_type") if no_data else (
            "metric_compare" if comparison else "metric_breakdown" if rows else "metric_value"
        ),
        "scope": _extract_scope(result),
    }


def safe_result_context(value: Any) -> dict[str, Any] | None:
    """Return a bounded, type-checked projection of a persisted result context."""

    if not isinstance(value, Mapping):
        return None
    if value.get("version") != RESULT_CONTEXT_VERSION:
        return None
    raw_metric_ids = value.get("metric_ids")
    if not isinstance(raw_metric_ids, list):
        return None
    metric_ids = sorted({
        _clean_text(metric, limit=80)
        for metric in raw_metric_ids if isinstance(metric, str) and metric
    })
    if not metric_ids:
        return None
    metric_id = _clean_text(value.get("metric_id"), limit=80)
    if (len(metric_ids) == 1 and metric_id != metric_ids[0]) or (len(metric_ids) > 1 and metric_id):
        return None
    source_turn_id = _clean_text(value.get("source_turn_id"), limit=120)
    source_trace_id = _clean_text(value.get("source_trace_id"), limit=120)
    raw_records = value.get("evidence_records")
    if not isinstance(raw_records, list):
        return None
    safe_records: list[dict[str, Any]] = []
    record_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_records[:MAX_EVIDENCE_IDS]:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("content"), Mapping):
            return None
        content = raw["content"]
        record_metric = _clean_text(content.get("metric_id"), limit=80)
        query_type = _clean_text(content.get("query_type"), limit=40)
        dimension = _clean_text(content.get("dimension"), limit=80) or None
        period = _safe_query_period(content.get("period"))
        source_ids = [
            _clean_text(item, limit=120) for item in content.get("source_evidence_ids", [])
            if isinstance(item, str) and item
        ][:MAX_EVIDENCE_IDS]
        payload_fields = [key for key in ("row", "fact", "comparison") if key in content]
        if (
            record_metric not in metric_ids or query_type not in _QUERY_TYPES
            or len(payload_fields) != 1
            or _clean_text(content.get("source_turn_id"), limit=120) != source_turn_id
            or _clean_text(content.get("source_trace_id"), limit=120) != source_trace_id
        ):
            return None
        payload_key = payload_fields[0]
        payload = content.get(payload_key)
        if not isinstance(payload, Mapping):
            return None
        if payload_key == "row":
            key = _clean_text(payload.get("key"), limit=120)
            number = _finite_number(payload.get("value"))
            if not key or number is None or query_type != "metric_breakdown":
                return None
            safe_payload = {"row": {
                "key": key, "value": number, "unit": _clean_text(payload.get("unit"), limit=40),
            }}
        elif payload_key == "fact":
            number = _finite_number(payload.get("value"))
            if number is None or query_type != "metric_value":
                return None
            safe_payload = {"fact": {
                "display_name": _clean_text(payload.get("display_name"), limit=80),
                "value": number,
                "unit": _clean_text(payload.get("unit"), limit=40),
                "date_or_range": _clean_text(payload.get("date_or_range"), limit=120),
            }}
        else:
            baseline = _finite_number(payload.get("baseline_value"))
            current = _finite_number(payload.get("current_value"))
            if baseline is None or current is None or query_type != "metric_compare":
                return None
            safe_payload = {"comparison": {
                "baseline_value": baseline,
                "current_value": current,
                "delta_value": _finite_number(payload.get("delta_value")),
                "delta_ratio": _finite_number(payload.get("delta_ratio")),
                "unit": _clean_text(payload.get("unit"), limit=40),
                "baseline_period": _clean_text(payload.get("baseline_period"), limit=120),
                "current_period": _clean_text(payload.get("current_period"), limit=120),
            }}
        rebuilt = _context_evidence_record(
            source_turn_id=source_turn_id,
            source_trace_id=source_trace_id,
            metric_id=record_metric,
            query_type=query_type,
            source_evidence_ids=source_ids,
            dimension=dimension,
            period=period,
            payload=safe_payload,
        )
        if raw.get("evidence_id") != rebuilt["evidence_id"] or raw.get("source_ref") != rebuilt["source_ref"]:
            return None
        safe_records.append(rebuilt)
        record_by_id[rebuilt["evidence_id"]] = rebuilt

    def safe_evidence_ids(raw: Mapping[str, Any], *, metric: str, kind: str) -> list[str] | None:
        ids = [
            _clean_text(item, limit=120) for item in raw.get("evidence_ids", [])
            if isinstance(item, str) and item
        ][:MAX_EVIDENCE_IDS]
        if not ids:
            return None
        for evidence_id in ids:
            record = record_by_id.get(evidence_id)
            if not record or record["content"].get("metric_id") != metric:
                return None
            payload_key = next(
                (key for key in ("row", "fact", "comparison") if key in record["content"]),
                None,
            )
            if payload_key != kind:
                return None
        return ids

    rows: list[dict[str, Any]] = []
    for raw in value.get("rows") or []:
        if not isinstance(raw, Mapping):
            return None
        key = _clean_text(raw.get("key"), limit=120)
        number = _finite_number(raw.get("value"))
        row_metric = _clean_text(raw.get("metric_id"), limit=80)
        evidence_ids = safe_evidence_ids(raw, metric=row_metric, kind="row")
        if not key or number is None or row_metric not in metric_ids or evidence_ids is None:
            return None
        row_dimension = _clean_text(raw.get("dimension"), limit=80) or None
        row_unit = _clean_text(raw.get("unit"), limit=40)
        row_period = _safe_query_period(raw.get("query_period"))
        if any(
            record_by_id[evidence_id]["content"].get("dimension") != row_dimension
            or record_by_id[evidence_id]["content"].get("period") != row_period
            or record_by_id[evidence_id]["content"].get("row") != {
                "key": key, "value": number, "unit": row_unit,
            }
            for evidence_id in evidence_ids
        ):
            return None
        rows.append({
            "key": key, "value": number, "metric_id": row_metric,
            "dimension": row_dimension,
            "unit": row_unit,
            "query_period": row_period,
            "source_evidence_ids": [
                _clean_text(item, limit=120) for item in raw.get("source_evidence_ids", [])
                if isinstance(item, str) and item
            ][:MAX_EVIDENCE_IDS],
            "evidence_ids": evidence_ids,
        })
        if len(rows) >= MAX_ROWS:
            break
    facts: list[dict[str, Any]] = []
    for raw in value.get("facts") or []:
        if not isinstance(raw, Mapping):
            return None
        fact_metric = _clean_text(raw.get("metric_name"), limit=80)
        number = _finite_number(raw.get("value"))
        evidence_ids = safe_evidence_ids(raw, metric=fact_metric, kind="fact")
        if fact_metric not in metric_ids or number is None or evidence_ids is None:
            return None
        fact_payload = {
            "display_name": _clean_text(raw.get("display_name"), limit=80),
            "value": number,
            "unit": _clean_text(raw.get("unit"), limit=40),
            "date_or_range": _clean_text(raw.get("date_or_range"), limit=120),
        }
        if any(record_by_id[evidence_id]["content"].get("fact") != fact_payload for evidence_id in evidence_ids):
            return None
        facts.append({
            "metric_name": fact_metric,
            "display_name": fact_payload["display_name"],
            "value": number,
            "unit": fact_payload["unit"],
            "date_or_range": fact_payload["date_or_range"],
            "source_evidence_ids": [
                _clean_text(item, limit=120) for item in raw.get("source_evidence_ids", [])
                if isinstance(item, str) and item
            ][:MAX_EVIDENCE_IDS],
            "evidence_ids": evidence_ids,
        })
        if len(facts) >= MAX_FACTS:
            break
    comparison = value.get("comparison")
    safe_comparison = None
    if isinstance(comparison, Mapping):
        baseline = _finite_number(comparison.get("baseline_value"))
        current = _finite_number(comparison.get("current_value"))
        comp_metric = _clean_text(comparison.get("metric_name"), limit=80)
        if comp_metric and baseline is not None and current is not None:
            evidence_ids = safe_evidence_ids(comparison, metric=comp_metric, kind="comparison")
            if comp_metric not in metric_ids or evidence_ids is None:
                return None
            comparison_payload = {
                "baseline_value": baseline,
                "current_value": current,
                "delta_value": _finite_number(comparison.get("delta_value")),
                "delta_ratio": _finite_number(comparison.get("delta_ratio")),
                "unit": _clean_text(comparison.get("unit"), limit=40),
                "baseline_period": _clean_text(comparison.get("baseline_period"), limit=120),
                "current_period": _clean_text(comparison.get("current_period"), limit=120),
            }
            if any(
                record_by_id[evidence_id]["content"].get("comparison") != comparison_payload
                for evidence_id in evidence_ids
            ):
                return None
            safe_comparison = {
                "metric_name": comp_metric,
                "baseline_value": baseline,
                "current_value": current,
                "delta_value": comparison_payload["delta_value"],
                "delta_ratio": comparison_payload["delta_ratio"],
                "unit": comparison_payload["unit"],
                "baseline_period": comparison_payload["baseline_period"],
                "current_period": comparison_payload["current_period"],
                "source_evidence_ids": [
                    _clean_text(item, limit=120) for item in comparison.get("source_evidence_ids", [])
                    if isinstance(item, str) and item
                ][:MAX_EVIDENCE_IDS],
                "evidence_ids": evidence_ids,
            }
    result_status = _clean_text(value.get("result_status"), limit=40)
    if result_status not in {"completed", "partial", *_NO_DATA_STATUSES}:
        return None
    safe_query_period = _safe_query_period(value.get("query_period"))
    if not (rows or facts or safe_comparison) and result_status not in _NO_DATA_STATUSES:
        return None
    safe_scope = None
    if isinstance(value.get("scope"), Mapping):
        safe_scope = {
            field: _clean_text(value["scope"].get(field), limit=120)
            for field in ("seller_id", "start_date", "end_date")
            if value["scope"].get(field)
        } or None
    return {
        "version": RESULT_CONTEXT_VERSION,
        "source_turn_id": _clean_text(value.get("source_turn_id"), limit=120),
        "source_trace_id": _clean_text(value.get("source_trace_id"), limit=120),
        "scope_fingerprint": _clean_text(value.get("scope_fingerprint"), limit=200),
        "data_version_manifest_hash": _clean_text(
            value.get("data_version_manifest_hash"), limit=200
        ),
        "metric_id": metric_id,
        "metric_ids": metric_ids,
        "query_type": (
            _clean_text(value.get("query_type"), limit=40)
            if _clean_text(value.get("query_type"), limit=40) in _QUERY_TYPES else "metric_value"
        ),
        "dimension": _clean_text(value.get("dimension"), limit=80) or None,
        "operation": _clean_text(value.get("operation"), limit=80),
        "rows": rows,
        "facts": facts,
        "comparison": safe_comparison,
        "evidence_ids": [record["evidence_id"] for record in safe_records],
        "evidence_records": safe_records,
        "result_status": result_status,
        "result_reason": (
            _clean_text(value.get("result_reason"), limit=40)
            if value.get("result_reason") in {"data_not_found", "insufficient_data"} else None
        ),
        "query_period": safe_query_period,
        "scope": safe_scope,
    }


def trusted_result_context(
    previous_context: Mapping[str, Any] | None,
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the previous result only when it is still in the same scope."""

    if not isinstance(previous_context, Mapping):
        return None
    context = safe_result_context(previous_context.get("result_context"))
    if context is None:
        return None
    current_scope = _clean_text(
        state.get("current_query_scope_fingerprint")
        or previous_context.get("scope_fingerprint"),
        limit=200,
    )
    result_scope = _clean_text(context.get("scope_fingerprint"), limit=200)
    if current_scope and result_scope and current_scope != result_scope:
        return None
    if context.get("result_status") in _NO_DATA_STATUSES and (
        not current_scope or not result_scope or current_scope != result_scope
    ):
        return None
    current_version = _clean_text(state.get("data_version_manifest_hash"), limit=200)
    result_version = _clean_text(context.get("data_version_manifest_hash"), limit=200)
    if current_version and result_version and current_version != result_version:
        return None
    return context


def _parse_top_n(value: str) -> int | None:
    value = value.strip()
    if value.isdigit():
        number = int(value)
    else:
        number = _TOP_N_WORDS.get(value)
    if number is None:
        return None
    return max(1, min(number, 10))


def _match_followup(question: str) -> dict[str, Any] | None:
    if any(marker in question for marker in _REPORT_ACTION_MARKERS):
        return {"kind": "report_action"}
    extremum = _EXTREMUM.search(question)
    if extremum:
        return {"kind": "extremum", "direction": "max" if extremum.group(1) in {"最高", "最大"} else "min"}
    top_n = _TOP_N.search(question)
    if top_n:
        number = _parse_top_n(top_n.group(1) or top_n.group(2) or "")
        if number is not None:
            return {"kind": "top_n", "n": number}
    direction = _COMPARISON_DIRECTION.search(question)
    if direction:
        return {"kind": "comparison_direction"}
    if (
        any(marker in question for marker in _PROVENANCE_MARKERS)
        and any(reference in question for reference in _PROVENANCE_REFERENCES)
    ):
        return {"kind": "provenance"}
    if (
        any(reference in question for reference in _PREVIOUS_RESULT_REFERENCES)
        and any(marker in question for marker in _RESULT_VALUE_REQUESTS)
    ):
        return {"kind": "value"}
    return None


def detect_result_followup(
    question: str,
    previous_context: Mapping[str, Any] | None,
    state: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Detect a deterministic result follow-up and validate its source context."""

    spec = _match_followup(question)
    if spec is None:
        if not any(marker in question for marker in _NO_DATA_NEXT_STEP_MARKERS):
            return None, None, None
        context = trusted_result_context(previous_context, state)
        if context is not None and context.get("result_status") not in _NO_DATA_STATUSES:
            return None, None, None
        spec = {"kind": "no_data_next_step"}
        if context is None:
            return spec, None, "previous_no_data_context_unavailable"
        return spec, context, None
    context = trusted_result_context(previous_context, state)
    if context is None:
        return spec, None, "previous_result_context_unavailable"
    kind = spec["kind"]
    context_metric_ids = {
        str(item) for item in (context.get("metric_ids") or [context.get("metric_id")]) if item
    }
    explicit_metric_ids = _explicit_metric_ids(question)
    if len(explicit_metric_ids) > 1:
        return spec, None, "previous_result_metric_ambiguous"
    if explicit_metric_ids and not explicit_metric_ids.issubset(context_metric_ids):
        return spec, None, "previous_result_metric_unavailable"
    if len(context_metric_ids) > 1 and not explicit_metric_ids:
        return spec, None, "previous_result_metric_ambiguous"
    if len(context_metric_ids) == 1:
        context = _select_metric_context(context, next(iter(context_metric_ids)))
    elif explicit_metric_ids:
        context = _select_metric_context(context, next(iter(explicit_metric_ids)))
    if kind in {"extremum", "top_n", "report_action"}:
        row_groups = {
            (
                str(row.get("dimension") or ""),
                tuple(sorted((row.get("query_period") or {}).items())),
            )
            for row in (context.get("rows") or []) if isinstance(row, Mapping)
        }
        if len(row_groups) > 1:
            return spec, None, "previous_result_breakdown_scope_ambiguous"
    if kind == "report_action" and context.get("result_status") in _NO_DATA_STATUSES:
        if any(marker in question for marker in _NO_DATA_NEXT_STEP_MARKERS):
            spec = {"kind": "no_data_next_step"}
        else:
            return spec, None, "previous_result_has_no_data"
    if kind in {"extremum", "top_n"} and not context.get("rows"):
        return spec, None, "previous_result_rows_unavailable"
    if kind == "value" and not (context.get("facts") or context.get("comparison")):
        return spec, None, "previous_result_value_unavailable"
    if kind == "comparison_direction" and not context.get("comparison"):
        return spec, None, "previous_result_comparison_unavailable"
    if kind == "provenance" and not (
        context.get("evidence_ids")
        and any(
            isinstance(fact, Mapping) and fact.get("evidence_ids")
            for fact in (context.get("facts") or [])
        )
    ):
        return spec, None, "previous_result_evidence_unavailable"
    if spec["kind"] == "report_action" and (
        not context.get("evidence_ids")
        or not (context.get("facts") or context.get("rows") or context.get("comparison"))
    ):
        return spec, None, "previous_result_evidence_unavailable"
    return spec, context, None


def _evidence_bundles(context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Materialize derived snapshot evidence without labelling it as a new MCP call."""

    records = [
        item for item in (context.get("evidence_records") or [])
        if isinstance(item, Mapping) and item.get("evidence_id")
    ][:MAX_EVIDENCE_IDS]
    bundles = [{
        "evidence_id": str(item["evidence_id"]),
        "source_type": "result_context",
        "source_ref": str(item.get("source_ref") or "result_context"),
        "content": dict(item.get("content") or {}),
        "trusted": True,
    } for item in records]
    return bundles, [str(item["evidence_id"]) for item in records]


def _period_label(context: Mapping[str, Any]) -> str:
    comparison = context.get("comparison") if isinstance(context.get("comparison"), Mapping) else {}
    baseline = _clean_text(comparison.get("baseline_period"), limit=120)
    current = _clean_text(comparison.get("current_period"), limit=120)
    if baseline and current:
        return f"{baseline} 至 {current}"
    query_period = context.get("query_period") if isinstance(context.get("query_period"), Mapping) else {}
    metric_date = _clean_text(query_period.get("metric_date"), limit=40)
    if metric_date:
        return metric_date
    start = _clean_text(query_period.get("start_date") or query_period.get("current_start"), limit=40)
    end = _clean_text(query_period.get("end_date") or query_period.get("current_end"), limit=40)
    if start and end:
        return start if start == end else f"{start} 至 {end}"
    scope = context.get("scope") if isinstance(context.get("scope"), Mapping) else {}
    start = _clean_text(scope.get("start_date"), limit=40)
    end = _clean_text(scope.get("end_date"), limit=40)
    if start and end:
        return start if start == end else f"{start} 至 {end}"
    facts = context.get("facts") or []
    if facts and isinstance(facts[0], Mapping):
        fact_period = _clean_text(facts[0].get("date_or_range"), limit=120)
        if fact_period:
            return fact_period
    return "上一轮查询范围"


def build_result_followup_answer(state: dict[str, Any]) -> dict[str, Any]:
    """Populate answer/presentation/claims for a validated result follow-up."""

    compilation = state.get("context_compilation") or {}
    followup = compilation.get("result_followup") or {}
    context = state.get("result_followup_context") or {}
    kind = str(followup.get("kind") or "")
    metric_id = str(context.get("metric_id") or "")
    display = _display_name(metric_id)
    if kind == "no_data_next_step":
        if context.get("result_status") not in _NO_DATA_STATUSES:
            raise ValueError("no_data_result_context_required")
        period = context.get("query_period") if isinstance(context.get("query_period"), Mapping) else {}
        date_label = (
            str(period.get("metric_date") or "")
            or (
                f"{period.get('start_date')} 至 {period.get('end_date')}"
                if period.get("start_date") and period.get("end_date") else "上一轮查询范围"
            )
        )
        reason = "未找到数据" if context.get("result_status") == "not_found" else "可用样本不足"
        recommendations = [
            {"text": "扩大日期范围后重试", "evidence_ids": []},
            {"text": "确认数据源覆盖的日期", "evidence_ids": []},
            {"text": "检查指标和筛选范围", "evidence_ids": []},
        ]
        answer = (
            f"上一轮查询的 {date_label} {display}{reason}。可以扩大日期范围，确认数据覆盖日期，"
            "并检查当前指标和筛选范围。此建议没有触发新查询，也没有生成数值。"
        )
        state["answer"] = answer
        state["final_answer"] = answer
        state["claims"] = []
        state["evidence_bundles"] = []
        state["analysis_presentation"] = {
            "status": "completed",
            "scope": context.get("scope"),
            "semantic_interpretation": {
                "source_text": "",
                "metric_id": metric_id,
                "display_name": display,
                "dimension": None,
                "dimension_label": "",
                "interpretation": "对上一轮无数据结果提供确定性下一步建议",
                "confidence": 1.0,
                "reason": "no_data_result_context",
                "requires_confirmation": False,
            },
            "facts": [],
            "changes": [],
            "contributions": [],
            "limitations": [
                f"上一轮结果状态：{context.get('result_reason') or 'data_not_found'}。",
                "本轮复用了无数据状态，未执行新的业务查询，也未生成数值。",
            ],
            "recommendations": recommendations,
            "evidence": [],
            "trace_id": str(state.get("trace_id") or ""),
        }
        state["node_history"] = list(state.get("node_history") or []) + [{
            "node": "generate_answer",
            "summary": "Produced deterministic no-data next steps without a business query.",
        }]
        return state
    unit = _metric_unit(metric_id, str(
        (context.get("comparison") or {}).get("unit")
        or (context.get("facts") or [{}])[0].get("unit")
        or (context.get("rows") or [{}])[0].get("unit")
        or ""
    ))
    rows = sorted(
        [row for row in (context.get("rows") or []) if isinstance(row, Mapping)],
        key=lambda row: _finite_number(row.get("value")) or float("-inf"),
        reverse=True,
    )
    comparison = context.get("comparison") if isinstance(context.get("comparison"), Mapping) else None
    bundles, evidence_ids = _evidence_bundles(context)
    evidence = [
        {
            "evidence_id": bundle["evidence_id"],
            "source_type": "result_context",
            "summary": "上一轮同范围查询结果快照",
        }
        for bundle in bundles
    ]
    facts: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    limitations = ["本回答复用上一轮同范围结果，未执行新的业务查询。"]

    if kind == "extremum":
        selected = rows[:1]
        if not selected:
            raise ValueError("result_followup_rows_missing")
        row = selected[0]
        value = _finite_number(row.get("value"))
        facts.append({
            "metric_name": metric_id,
            "display_name": display,
            "value": value,
            "unit": _metric_unit(metric_id, str(row.get("unit") or unit)),
            "date_or_range": _period_label(context),
            "evidence_ids": list(row.get("evidence_ids") or []),
        })
        direction = "最高" if followup.get("direction") == "max" else "最低"
        answer = (
            f"上一轮按{_dimension_label(context.get('dimension'))}拆解的{display}中，"
            f"{direction}的是 {row.get('key')}（{value:g} {facts[0]['unit']}）。"
            f"该结论来自上一轮同范围查询结果，未执行新的数据查询。"
        )
    elif kind == "top_n":
        limit = max(1, min(int(followup.get("n") or 3), len(rows) or 1))
        selected = rows[:limit]
        if not selected:
            raise ValueError("result_followup_rows_missing")
        for row in selected:
            facts.append({
                "metric_name": metric_id,
                "display_name": display,
                "value": _finite_number(row.get("value")),
                "unit": _metric_unit(metric_id, str(row.get("unit") or unit)),
                "date_or_range": _period_label(context),
                "evidence_ids": list(row.get("evidence_ids") or []),
            })
        preview = "、".join(
            f"{row.get('key')}（{_finite_number(row.get('value')):g} "
            f"{_metric_unit(metric_id, str(row.get('unit') or unit))}）"
            for row in selected
        )
        answer = (
            f"上一轮结果的前 {len(selected)} 项为：{preview}。"
            f"仅基于上一轮同范围结果，未执行新的数据查询。"
        )
    elif kind == "comparison_direction":
        if comparison is None:
            raise ValueError("result_followup_comparison_missing")
        baseline = _finite_number(comparison.get("baseline_value"))
        current = _finite_number(comparison.get("current_value"))
        higher = (current or 0) >= (baseline or 0)
        changes.append({
            "metric_name": metric_id,
            "display_name": display,
            "baseline_value": baseline,
            "current_value": current,
            "delta_value": _finite_number(comparison.get("delta_value")),
            "delta_ratio": _finite_number(comparison.get("delta_ratio")),
            "unit": unit,
            "change_unit": unit,
            "delta_ratio_unit": "%",
            "baseline_period": _clean_text(comparison.get("baseline_period"), limit=120),
            "current_period": _clean_text(comparison.get("current_period"), limit=120),
            "evidence_ids": list(comparison.get("evidence_ids") or []),
        })
        current_period = changes[0]["current_period"] or "当前期"
        baseline_period = changes[0]["baseline_period"] or "基线期"
        answer = (
            f"上一轮比较中，{current_period} 的{display}为 {current:g} {unit}，"
            f"{baseline_period} 为 {baseline:g} {unit}；"
            f"{'前者更高' if higher else '前者更低'}。"
            f"该结论来自上一轮同范围比较结果。"
        )
    elif kind == "provenance":
        source_facts = [item for item in (context.get("facts") or []) if isinstance(item, Mapping)]
        if not source_facts:
            raise ValueError("result_followup_facts_missing")
        fact = source_facts[0]
        value = _finite_number(fact.get("value"))
        date_label = _clean_text(fact.get("date_or_range"), limit=120) or _period_label(context)
        facts.append({
            "metric_name": metric_id,
            "display_name": _clean_text(fact.get("display_name"), limit=80) or display,
            "value": value,
            "unit": _clean_text(fact.get("unit"), limit=40) or unit,
            "date_or_range": date_label,
            "evidence_ids": list(fact.get("evidence_ids") or []),
        })
        answer = (
            f"这个数字来自上一轮查询结果：{date_label} 的{display}为 "
            f"{value:g} {_clean_text(fact.get('unit'), limit=40) or unit}。"
            f"结果快照证据 ID：{evidence_ids[0]}，来源 trace "
            f"{context.get('source_trace_id') or '未记录'}；本轮未重新查询或计算。"
        )
    elif kind == "value":
        source_facts = [
            item for item in (context.get("facts") or [])
            if (
                isinstance(item, Mapping)
                and str(item.get("metric_name") or "") == metric_id
                and _finite_number(item.get("value")) is not None
            )
        ]
        if source_facts:
            for source_fact in source_facts[:4]:
                value = _finite_number(source_fact.get("value"))
                fact_unit = _clean_text(source_fact.get("unit"), limit=40) or unit
                facts.append({
                    "metric_name": _clean_text(source_fact.get("metric_name"), limit=80) or metric_id,
                    "display_name": _clean_text(source_fact.get("display_name"), limit=80) or display,
                    "value": value,
                    "unit": fact_unit,
                    "date_or_range": _clean_text(source_fact.get("date_or_range"), limit=120) or _period_label(context),
                    "evidence_ids": [
                        str(item) for item in (source_fact.get("evidence_ids") or [])
                        if str(item) in evidence_ids
                    ],
                })
            answer = "上一轮查询结果：" + "；".join(
                f"{item['date_or_range']} 的{item['display_name']}为 {item['value']:g} {item['unit']}"
                for item in facts
            ) + "。该回答复用上一轮同范围证据，未重新查询。"
        elif comparison and str(comparison.get("metric_name") or "") == metric_id:
            baseline = _finite_number(comparison.get("baseline_value"))
            current = _finite_number(comparison.get("current_value"))
            if baseline is None or current is None:
                raise ValueError("result_followup_comparison_values_missing")
            for value, period_name in (
                (baseline, "baseline_period"), (current, "current_period"),
            ):
                facts.append({
                    "metric_name": metric_id,
                    "display_name": display,
                    "value": value,
                    "unit": unit,
                    "date_or_range": _clean_text(comparison.get(period_name), limit=120),
                    "evidence_ids": list(comparison.get("evidence_ids") or []),
                })
            answer = (
                f"上一轮比较结果：{facts[1]['date_or_range'] or '当前期'} 的{display}为 {current:g} {unit}，"
                f"{facts[0]['date_or_range'] or '基线期'} 为 {baseline:g} {unit}。"
                "该回答复用上一轮同范围证据，未重新查询。"
            )
        else:
            raise ValueError("result_followup_value_missing")
    elif kind == "report_action":
        if not evidence_ids:
            raise ValueError("result_followup_evidence_missing")
        period = _period_label(context)
        source_facts = [item for item in (context.get("facts") or []) if isinstance(item, Mapping)]
        report_facts: list[dict[str, Any]] = []
        detail_lines: list[str] = []
        for source_fact in source_facts[:1]:
            value = _finite_number(source_fact.get("value"))
            if value is None:
                continue
            fact_evidence = [
                str(item) for item in (source_fact.get("evidence_ids") or [])
                if str(item) in evidence_ids
            ]
            fact = {
                "metric_name": _clean_text(source_fact.get("metric_name"), limit=80) or metric_id,
                "display_name": _clean_text(source_fact.get("display_name"), limit=80) or display,
                "value": value,
                "unit": _clean_text(source_fact.get("unit"), limit=40) or unit,
                "date_or_range": _clean_text(source_fact.get("date_or_range"), limit=120) or period,
                "evidence_ids": fact_evidence,
            }
            report_facts.append(fact)
            detail_lines.append(f"{fact['display_name']}：{value:g} {fact['unit']}")

        selected_rows = rows[:3]
        for row in selected_rows:
            value = _finite_number(row.get("value"))
            if value is None:
                continue
            report_facts.append({
                "metric_name": metric_id,
                "display_name": display,
                "value": value,
                "unit": _metric_unit(metric_id, str(row.get("unit") or unit)),
                "date_or_range": period,
                "evidence_ids": list(row.get("evidence_ids") or []),
            })
            detail_lines.append(f"{row.get('key')}：{value:g} {report_facts[-1]['unit']}")

        report_changes: list[dict[str, Any]] = []
        if comparison:
            baseline_value = _finite_number(comparison.get("baseline_value"))
            current_value = _finite_number(comparison.get("current_value"))
            if baseline_value is not None and current_value is not None:
                report_changes.append({
                    "metric_name": metric_id,
                    "display_name": display,
                    "baseline_value": baseline_value,
                    "current_value": current_value,
                    "delta_value": _finite_number(comparison.get("delta_value")),
                    "delta_ratio": _finite_number(comparison.get("delta_ratio")),
                    "unit": unit,
                    "change_unit": unit,
                    "delta_ratio_unit": "%",
                    "baseline_period": _clean_text(comparison.get("baseline_period"), limit=120),
                    "current_period": _clean_text(comparison.get("current_period"), limit=120),
                    "evidence_ids": list(comparison.get("evidence_ids") or []),
                })
                if not detail_lines:
                    detail_lines.append(
                        f"{comparison.get('current_period') or '当前期'}：{current_value:g} {unit}；"
                        f"{comparison.get('baseline_period') or '基线期'}：{baseline_value:g} {unit}"
                    )
        if not detail_lines:
            raise ValueError("result_followup_report_facts_missing")

        if selected_rows:
            recommendation_text = (
                f"下一步可核对“{selected_rows[0].get('key')}”对应的订单明细，验证该项与上一轮"
                f"{display}结果的一致性；这是一项待验证动作，不构成因果结论。"
            )
        else:
            recommendation_text = (
                f"下一步可用同一口径复查{display}的明细来源与后续周期变化；该建议不代表已确认业务原因。"
            )
        recommendations = [{
            "text": recommendation_text,
            "evidence_ids": list(selected_rows[0].get("evidence_ids") or []) if selected_rows
            else list((source_facts[0].get("evidence_ids") or []) if source_facts else evidence_ids[:1]),
        }]
        answer = (
            f"简短经营摘要\n指标：{display}\n期间：{period}\n关键结果："
            f"{'；'.join(detail_lines[:4])}\n行动建议：{recommendation_text}\n"
            "本摘要复用上一轮同范围证据，不是完整经营报告，也不证明业务因果；未重新查询数据。"
        )
        state["answer"] = answer
        state["final_answer"] = answer
        state["claims"] = [
            {
                "claim_type": "metric",
                "value": f"{item['display_name']} {item['value']:g} {item['unit']}",
                "source_evidence_ids": list(item["evidence_ids"]),
            }
            for item in report_facts
        ] + [{
            "claim_type": "recommendation",
            "value": recommendation_text,
            "source_evidence_ids": recommendations[0]["evidence_ids"],
        }]
        state["evidence_bundles"] = bundles
        state["analysis_presentation"] = {
            "status": "completed",
            "scope": context.get("scope"),
            "semantic_interpretation": {
                "source_text": "",
                "metric_id": metric_id,
                "display_name": display,
                "dimension": context.get("dimension"),
                "dimension_label": _dimension_label(context.get("dimension")) if context.get("dimension") else "",
                "interpretation": "基于上一轮同范围结果整理简短经营摘要",
                "confidence": 1.0,
                "reason": "same_scope_result_context_report_action",
                "requires_confirmation": False,
            },
            "facts": report_facts,
            "changes": report_changes,
            "contributions": [],
            "limitations": [
                "这是基于上一轮结果的简短摘要，不是完整经营报告。",
                "建议是待验证动作；当前证据不证明业务因果。",
                "本轮未执行新的业务查询。",
            ],
            "recommendations": recommendations,
            "evidence": evidence,
            "trace_id": str(state.get("trace_id") or ""),
        }
        state["node_history"] = list(state.get("node_history") or []) + [{
            "node": "generate_answer",
            "summary": "Produced deterministic report action from same-scope result context without a new business query or LLM report call.",
        }]
        return state
    else:
        raise ValueError("result_followup_kind_unsupported")

    claims: list[dict[str, Any]] = []
    for fact in facts:
        claims.append({
            "claim_type": "metric",
            "value": str(fact.get("value")),
            "source_evidence_ids": fact.get("evidence_ids") or evidence_ids[:1],
        })
    for change in changes:
        claims.append({
            "claim_type": "analysis",
            "value": f"{change.get('metric_name')} 变化 {change.get('delta_value')}",
            "source_evidence_ids": change.get("evidence_ids") or evidence_ids[:1],
        })

    state["answer"] = answer
    state["final_answer"] = answer
    state["claims"] = claims
    state["evidence_bundles"] = bundles
    state["analysis_presentation"] = {
        "status": "completed",
        "scope": context.get("scope"),
        "semantic_interpretation": {
            "source_text": "",
            "metric_id": metric_id,
            "display_name": display,
            "dimension": context.get("dimension"),
            "dimension_label": context.get("dimension") or "",
            "interpretation": "复用上一轮同范围结果",
            "confidence": 1.0,
            "reason": "result_context",
            "requires_confirmation": False,
        },
        "facts": facts,
        "changes": changes,
        "contributions": [],
        "limitations": limitations,
        "recommendations": [],
        "evidence": evidence,
        "trace_id": str(state.get("trace_id") or ""),
    }
    state["node_history"] = list(state.get("node_history") or []) + [{
        "node": "generate_answer",
        "summary": f"Answered from result context ({kind}); no business query executed.",
    }]
    return state
