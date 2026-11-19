from __future__ import annotations

import copy
import json

from app.context.semantic_catalog_v2 import DEFAULT_SEMANTIC_CATALOG
from app.context.semantic_contract_v2 import SemanticContractCandidateV21
from app.context.semantic_contract_validator import validate_semantic_contract
from app.context.semantic_interpreter import semantic_candidate_response_format
from app.llm.semantic_interpreter_prompt import (
    EXAMPLE_CASES,
    FIELD_RULES,
    FRAME_ROLE_RELATION_RULES,
    INTENT_CONTEXT_RULES,
    KNOWN_GAP_RULES,
    MENTION_OPERATION_RULES,
    PROMPT_VERSION,
    REPAIR_PROMPT_VERSION,
    RFM_RULES,
    SPAN_CLAUSE_RULES,
    WIRE_SCHEMA,
    build_system_prompt,
    build_repair_system_prompt,
    build_request,
)


def _materialize_example(item: dict) -> tuple[str, dict]:
    source_text = item["source_text"]
    candidate = copy.deepcopy(item["candidate"])
    candidate["turn_id"] = "prompt-example-turn"
    candidate["catalog_snapshot_hash"] = DEFAULT_SEMANTIC_CATALOG.snapshot_hash()
    for mention in candidate.get("mentions", []):
        mention["source_turn_id"] = candidate["turn_id"]
    for operation in candidate.get("operations", []):
        for negation in operation.get("argument_negations", []):
            negation["source_turn_id"] = candidate["turn_id"]
    return source_text, candidate


def _keys_recursive(value):
    if isinstance(value, dict):
        yield from value.keys()
        for child in value.values():
            yield from _keys_recursive(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys_recursive(child)


def test_prompt_v3_10_has_wire_contract_enums_and_safety_rules():
    prompt = build_system_prompt()

    assert PROMPT_VERSION == "semantic-interpreter-v3.10"
    for field in (
        "contract_version", "turn_id", "parse_status", "domain", "clauses",
        "mentions", "operations", "request_frame", "roles", "relations",
        "intent_candidates", "primary_intent", "ambiguities", "context_sources",
        "catalog_snapshot_hash",
    ):
        assert f'"{field}"' in WIRE_SCHEMA
        assert field in FIELD_RULES
    for enum_value in (
        "PARSED", "AMBIGUOUS", "UNSUPPORTED", "INCOMPLETE", "GENERAL_CONVERSATION",
        "BUSINESS_ANALYSIS", "BUSINESS_QUERY", "BUSINESS_REPORT", "SEGMENT",
        "ANALYZE", "FILTER", "SORT", "AGGREGATE", "QUERY", "EXPLAIN",
        "EXPLAIN_CAUSE", "COMPARE", "TREND", "CONTRIBUTION", "BREAKDOWN",
        "ENUMERATE", "ADOPTED", "MENTIONED_ONLY", "EXCLUDED", "exact_alias",
        "catalog_synonym", "llm_inferred", "inherited", "EXECUTE", "ASK",
        "POSITIVE", "NEGATIVE", "SAME_CLAUSE", "REQUEST", "NEW", "CONTINUE",
        "SWITCH", "FACTUAL", "HYPOTHETICAL", "basis", "topic", "metric",
        "dimension", "field", "value", "key", "direction", "group_by",
        "comparison_target", "time_grain", "outcome", "entity", "filter",
        "analysis_subject", "segment_basis", "filter_field", "reference",
        "time_axis", "GROUP_BY", "FILTER_BY", "COMPARE_WITH", "AGGREGATE_BY",
        "REFERS_TO", "INHERITS_FROM", "CAUSE_OF", "BREAKS_DOWN_BY", "mention_id",
        "concept_id", "role_id", "HIGH", "MEDIUM", "LOW", "EXPLICIT_REFERENCE",
        "INHERITED_CONTEXT", "method",
    ):
        assert enum_value in prompt

    assert '"start"' in WIRE_SCHEMA and '"end"' in WIRE_SCHEMA
    assert "Python Unicode code point" in SPAN_CLAUSE_RULES
    assert "右开区间" in SPAN_CLAUSE_RULES
    assert "mentions[].surface 必须是 JSON 字符串" in SPAN_CLAUSE_RULES
    assert '主键统一叫 id' in FIELD_RULES
    assert "semantic_status" in prompt
    assert "effective_arguments" in prompt
    assert "template_id" in prompt and "route_id" in prompt
    assert "SQL" in prompt and "permission" in prompt and "CONFIRMED" in prompt
    assert "capability_id" in prompt
    assert "ANALYSIS_METHOD" in prompt
    assert "RFM 是 ANALYSIS_METHOD" in prompt
    assert "method_concept_ids" in prompt
    assert "negative RFM" in prompt or "负向 RFM" in prompt
    assert "每个 argument_negation 的 target_value_mention_id 必须同时出现在同一 operation 的 mentioned_arguments[argument_path] 中" in prompt
    assert "不能只写 argument_negations" in prompt
    assert "roles[].mention_id" in FRAME_ROLE_RELATION_RULES
    assert "不是 null、对象、数组或 endpoint" in FRAME_ROLE_RELATION_RULES
    assert '"roles": []' in FRAME_ROLE_RELATION_RULES
    assert '"id":"r1","clause_id":"c1","mention_id":"m1","concept_id":"CUSTOMER","name":"analysis_subject"' in FRAME_ROLE_RELATION_RULES
    assert "RelationEndpoint" in FRAME_ROLE_RELATION_RULES
    assert "但改/改成/改用/改为" in MENTION_OPERATION_RULES
    assert "同轮可跨 clause" in MENTION_OPERATION_RULES
    assert "SORT target 是排序键" in MENTION_OPERATION_RULES
    assert "不是被排序实体" in MENTION_OPERATION_RULES
    assert "SORT_ASCENDING" in prompt and "SORT_DESCENDING" in prompt
    assert "SORT 的 `direction` 必须引用 catalog path domain" in MENTION_OPERATION_RULES
    assert "原文没有方向短语时省略 direction" in MENTION_OPERATION_RULES
    assert "方向含糊时设 parse_status=AMBIGUOUS" in MENTION_OPERATION_RULES
    assert '"path_concept_domains"' in prompt
    assert "concept_id 只能属于 concepts，intent_id 只能属于 intent_ids" in FIELD_RULES
    assert "仅属于 intent_ids 的 CUSTOMER_SEGMENTATION 是分析意图标签，不是可提及的实体概念" in MENTION_OPERATION_RULES
    assert "“分层”“排序”“比较”等动作词表达 operation，不是被分析实体 mention" in MENTION_OPERATION_RULES
    assert "按金额排序客户，不是按金额分层" in KNOWN_GAP_RULES
    assert "CUSTOMER_RETENTION 表示宽泛目标" in KNOWN_GAP_RULES
    assert "只保留所选的具体目标 operation" in KNOWN_GAP_RULES
    assert "设置 clause_scope=REQUEST" in MENTION_OPERATION_RULES
    assert "缺少 basis 本身不把执行请求改成 ASK" in RFM_RULES
    assert "NEEDS_CONTEXT" in RFM_RULES
    assert "CONTINUE 指代/撤回无可信历史" in INTENT_CONTEXT_RULES
    assert "这个不要做" in INTENT_CONTEXT_RULES
    assert "不得归入 GENERAL_CONVERSATION" in INTENT_CONTEXT_RULES
    assert "编造 mention/operation/context_source" in INTENT_CONTEXT_RULES
    assert len(prompt) < 22_000
    assert prompt == build_system_prompt()


def test_v39_prompt_names_reviewed_semantic_boundaries_and_new_regressions():
    expected_fragments = (
        "后置否定只回指紧邻的前一个 operation",
        "跨轮否定不得静默折叠为本轮 EXCLUDED",
        "排序否定是独立 operation",
        "句首否定作用于完整 operation",
        "跨子句替代只保留最终 operation",
        "“RFM 客户分群先不要做” -> SEGMENT/RFM/NEGATIVE",
        "不支持动作或限定词不能被吞掉",
        "多义指标应追问",
        "不发明 catalog 概念",
    )
    assert all(fragment in KNOWN_GAP_RULES for fragment in expected_fragments)
    assert "concept 与 intent 字段按各自 catalog 集合取值" in KNOWN_GAP_RULES
    assert "仅否定金额等 basis" in KNOWN_GAP_RULES
    assert "后置否定跨 clause 用 REQUEST" in KNOWN_GAP_RULES
    for example in (
        "做客户生命周期分析，不要做",
        "上一轮说不要用金额，这次按购买频次给客户分层",
        "不按金额排序，按购买频次给客户分层",
        "不要按金额降序给客户分层",
        "RFM 分群不做，改用 Cohort 留存分析",
        "RFM 客户分群先不要做",
        "提醒我不要按金额排序客户",
        "做生命周期成本分析",
        "说明一下销量是多少",
    ):
        assert example in KNOWN_GAP_RULES


def test_only_the_eleven_committed_review_decisions_are_marked_reviewed():
    from pathlib import Path

    root = Path(__file__).parents[1]
    dataset_paths = (
        root / "evaluation/datasets/complex_semantic_v341_adversarial.jsonl",
        root / "evaluation/datasets/complex_semantic_v341_labeled.jsonl",
    )
    cases = {
        case["case_id"]: case
        for path in dataset_paths
        for case in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    decisions_path = root / "docs/design/evidence/semantic-interpreter-v3.5-review-decisions.jsonl"
    decisions = {
        item["case_id"]: item
        for item in (
            json.loads(line)
            for line in decisions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if item["case_id"] in {
            "v341-neg-postposed-01", "v341-required-011-04", "v341-required-011-07",
            "v341-required-011-08", "v341-required-011-09", "v341-required-011-11",
            "v341-required-011-14", "v341-labeled-neg-reminder-01",
            "v341-labeled-rfm-neg-01", "v341-labeled-cross-turn-01",
            "v341-labeled-cross-turn-02",
        }
    }
    assert len(cases) == 70
    assert len(decisions) == 11
    assert set(cases) >= set(decisions)
    postposed = cases["v341-neg-postposed-01"]["expected_contract"]["operations"][0]["candidate"]
    assert postposed["clause_scope"] == "REQUEST"
    assert sum(case.get("label_status") == "reviewed" for case in cases.values()) == 11
    assert sum(case.get("label_status", "draft") == "draft" for case in cases.values()) == 59
    for case_id, decision in decisions.items():
        case = cases[case_id]
        assert case["label_status"] == "reviewed"
        assert case["review_source"] == "codex-ai-20261114"
        assert case["expected_status"] == decision["reviewed_status"]
        candidate_status = case["expected_contract"].get("candidate", {}).get("parse_status")
        expected_candidate_status = {
            "CONFIRMED": "PARSED",
            "AMBIGUOUS": "AMBIGUOUS",
            "UNSUPPORTED": "UNSUPPORTED",
        }[decision["reviewed_status"]]
        assert candidate_status == expected_candidate_status
        expected_operations = []
        for operation in case["expected_contract"].get("operations", []):
            candidate = operation.get("candidate") or {}
            expected_operations.append({
                "kind": operation["kind"],
                "target_concept_ids": sorted(operation["target_concept_ids"]),
                "polarity": candidate.get("polarity", operation.get("polarity", "POSITIVE")),
            })
        normalized_decision_operations = [
            {**operation, "target_concept_ids": sorted(operation["target_concept_ids"])}
            for operation in decision["reviewed_operations"]
        ]
        assert expected_operations == normalized_decision_operations


def test_validation_repair_prompt_and_feedback_are_bounded_and_untrusted():
    prompt = build_repair_system_prompt()
    request = json.loads(build_request(
        "给客户分层",
        "turn-1",
        repair_feedback={
            "previous_status": "schema_invalid",
            "reason_code": "candidate_schema_invalid",
            "schema_errors": [{"loc": ["operations", 1], "type": "value_error", "msg": "bad polarity"}],
            "validator_diagnostics": [{
                "code": "missing_target_reference",
                "location": "operations[0].target_mention_ids",
                "severity": "MISSING",
            }],
            "issue_codes": ["missing_target_reference"],
            "previous_candidate": {"operations": []},
            "previous_output_excerpt": "previous model output",
            "repair_guidance": [{
                "code": "invalid_argument_path",
                "operation_index": 0,
                "operation_kind": "QUERY",
                "invalid_path": "time_grain",
                "allowed_paths": ["metric", "group_by"],
                "instruction": "QUERY only accepts metric and group_by; preserve the user meaning.",
            }],
        },
    ))

    assert REPAIR_PROMPT_VERSION == "semantic-interpreter-repair-v1"
    assert "不可信数据" in prompt
    assert "question 是唯一语义来源" in prompt
    assert request["repair_feedback"]["previous_status"] == "schema_invalid"
    assert request["repair_feedback"]["validator_diagnostics"][0]["code"] == "missing_target_reference"
    assert request["repair_feedback"]["previous_candidate"] == {"operations": []}
    assert request["repair_feedback"]["repair_guidance"][0]["allowed_paths"] == ["metric", "group_by"]
    assert "只用于修正对应字段" in prompt


def test_validation_repair_feedback_omits_oversized_candidate():
    request = json.loads(build_request(
        "问题",
        "turn-1",
        repair_feedback={
            "previous_status": "schema_invalid",
            "previous_candidate": {"large": "x" * 12_001},
        },
    ))

    assert "previous_candidate" not in request["repair_feedback"]


def test_semantic_candidate_schema_is_strict_and_catalog_scoped_by_operation():
    response_format = semantic_candidate_response_format()
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    operation_schema = schema["$defs"]["OperationCandidate"]

    query_rule = next(
        item for item in operation_schema["allOf"]
        if item["if"]["properties"]["kind"].get("const") == "QUERY"
    )
    trend_rule = next(
        item for item in operation_schema["allOf"]
        if item["if"]["properties"]["kind"].get("const") == "TREND"
    )
    query_paths = set(query_rule["then"]["properties"]["mentioned_arguments"]["properties"])
    trend_paths = set(trend_rule["then"]["properties"]["mentioned_arguments"]["properties"])
    assert query_paths == {"metric", "group_by"}
    assert "time_grain" in trend_paths
    assert "time_grain" not in query_paths

    def schema_keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from schema_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from schema_keys(child)

    keys = set(schema_keys(schema))
    assert "propertyNames" not in keys
    assert "discriminator" not in keys


def test_prompt_examples_are_schema_valid_and_have_no_validator_issues():
    assert EXAMPLE_CASES
    for item in EXAMPLE_CASES:
        source_text, candidate_data = _materialize_example(item)
        candidate = SemanticContractCandidateV21.model_validate_json(
            json.dumps(candidate_data, ensure_ascii=False)
        )
        validation = validate_semantic_contract(
            candidate,
            source_text=source_text,
            catalog=DEFAULT_SEMANTIC_CATALOG,
        )

        assert not validation.issues
        assert validation.contract.semantic_status not in {"CONFLICT", "MISSING", "AMBIGUOUS"}
        forbidden = {
            "semantic_status", "effective_arguments", "template_id", "route_id",
            "sql", "scope", "permission", "capability_id",
        }
        assert not (forbidden & {str(key).casefold() for key in _keys_recursive(candidate_data)})
        assert "CONFIRMED" not in json.dumps(candidate_data, ensure_ascii=False)


def test_positive_rfm_example_uses_customer_target_and_method_argument():
    positive, negative, no_basis = EXAMPLE_CASES[:3]
    positive_operation = positive["candidate"]["operations"][0]
    negative_operation = negative["candidate"]["operations"][0]
    no_basis_operation = no_basis["candidate"]["operations"][0]

    assert positive_operation["target_concept_ids"] == ["CUSTOMER"]
    assert positive_operation["mentioned_arguments"]["method"][0]["concept_id"] == "RFM"
    assert negative_operation["target_concept_ids"] == ["RFM"]
    assert negative_operation["polarity"] == "NEGATIVE"
    assert negative_operation["mentioned_arguments"] == {}
    assert no_basis_operation["target_mention_ids"] == []
    assert no_basis_operation["target_concept_ids"] == ["CUSTOMER"]
    assert no_basis_operation["mentioned_arguments"]["basis"] == []
    assert no_basis_operation["mentioned_arguments"]["method"][0]["concept_id"] == "RFM"


def test_argument_negation_example_excludes_the_referenced_basis_value():
    example = next(
        item for item in EXAMPLE_CASES
        if item["source_text"] == "不要按金额给客户分层"
    )
    operation = example["candidate"]["operations"][0]
    excluded_basis = operation["mentioned_arguments"]["basis"][0]
    negation = operation["argument_negations"][0]

    assert excluded_basis["concept_id"] == "MONETARY"
    assert excluded_basis["stance"] == "EXCLUDED"
    assert negation["argument_path"] == "basis"
    assert negation["target_value_mention_id"] == excluded_basis["mention_id"]


def test_sort_direction_example_uses_grounded_descending_concept():
    assert "按金额从高到低排序" in MENTION_OPERATION_RULES
    assert 'surface="从高到低" 创建 SORT_DESCENDING mention' in MENTION_OPERATION_RULES
    assert "由 direction ArgumentValue 引用它" in MENTION_OPERATION_RULES


def test_catalog_and_datasets_have_no_positive_segment_target_rfm():
    from pathlib import Path

    root = Path(__file__).parents[1]
    dataset_paths = (
        root / "evaluation/datasets/complex_semantic_v341_adversarial.jsonl",
        root / "evaluation/datasets/complex_semantic_v341_labeled.jsonl",
    )
    dataset_cases = [
        json.loads(line)
        for path in dataset_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    operation_sets = [
        [operation for example in EXAMPLE_CASES for operation in example["candidate"]["operations"]],
        [
            operation
            for case in dataset_cases
            for operation in case["expected_contract"].get("operations", [])
        ],
    ]
    for operations in operation_sets:
        for operation in operations:
            candidate = operation.get("candidate") or {}
            polarity = candidate.get("polarity", operation.get("polarity", "POSITIVE"))
            if operation.get("kind") == "SEGMENT" and "RFM" in operation.get("target_concept_ids", ()):
                assert polarity == "NEGATIVE"

    rfm_capability = next(
        item for item in DEFAULT_SEMANTIC_CATALOG.capabilities
        if item.capability_id == "customer_rfm_segmentation"
    )
    assert "RFM" not in rfm_capability.concept_ids
    assert rfm_capability.method_concept_ids == ("RFM",)

