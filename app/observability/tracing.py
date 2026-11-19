"""可观测性追踪层：OpenTelemetry Span + 内存事件 + Prometheus 指标。

本模块是项目所有可观测性的中枢，提供三个对外原语：
- `span(name, attributes)`：上下文管理器，创建一个 OpenTelemetry Span
- `record_event(name, attributes)`：记录一次带属性的业务事件（同时推送到 OTel 和本地）
- `snapshot_metrics()` / `prometheus_payload()`：导出累积指标快照

架构：
┌──────────────────────────────────────────────────────┐
│                    调用方                              │
│  (middleware, workflow, runtime, LLM client, RAG)     │
├──────────────────────────────────────────────────────┤
│              span() / record_event()                  │ ← 本模块
├──────────────────┬───────────────────────────────────┤
│  OpenTelemetry   │  内存事件列表 + Counter + Prometheus │
│  (OTLP Exporter) │  (_EVENTS, _METRICS, _PROM_*)      │
└──────────────────┴───────────────────────────────────┘

关键设计：
- 所有属性值在写入 Span 前都经过 `redact_sensitive()` 脱敏
- Prometheus 是可选依赖，不存在时降级到基于 dict 的轻量计数器
- OTLP Exporter 仅在有 endpoint 配置时启用
"""

from __future__ import annotations

from collections import Counter as CollectionCounter
from contextlib import contextmanager
from typing import Any

from app.security.governance import redact_sensitive
from app.core.config import get_settings

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ============================================================
# OpenTelemetry 初始化
# ============================================================

# 创建 TracerProvider，service.name 用于在 Jaeger/Tempo 中区分服务
_provider = TracerProvider(
    resource=Resource.create({"service.name": "bizinsight-agent"})
)
# 如果配置了 OTLP endpoint（如 Jaeger、Grafana Tempo），启用批量导出
if get_settings().otel_exporter_otlp_endpoint:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=get_settings().otel_exporter_otlp_endpoint)
        )
    )
trace.set_tracer_provider(_provider)
_TRACER = trace.get_tracer("bizinsight.agent")
_METER = metrics.get_meter("bizinsight.agent")


# ============================================================
# span() — 上下文管理器风格的可观测 Span
# ============================================================

@contextmanager
def span(name: str, attributes: dict[str, Any]):
    """创建一个 OpenTelemetry Span，所有属性值经过脱敏处理。

    用法：
        with span("graph.run", {"trace_id": tid, "thread_id": thid}):
            result = graph.invoke(state)

    Span 的生命周期由上下文管理器控制：进入 with 时创建，退出时自动结束。
    支持嵌套 —— 内层 span 会自动成为外层 span 的子 span。

    参数：
    - name：Span 名称（如 "graph.run"、"chat.request"）
    - attributes：属性字典，写入前自动调用 redact_sensitive() 脱敏
    """
    with _TRACER.start_as_current_span(name) as active_span:
        for key, value in redact_sensitive(attributes).items():
            if value is not None and isinstance(value, (str, bool, int, float)):
                active_span.set_attribute(key, value)
        yield active_span


# ============================================================
# Prometheus 指标（可选依赖）
# ============================================================

try:
    from prometheus_client import (
        Counter as PrometheusCounter,
        Histogram as PrometheusHistogram,
        generate_latest,
    )
except ImportError:  # pragma: no cover - 部署镜像中会安装此依赖
    PrometheusCounter = None
    PrometheusHistogram = None
    generate_latest = None

# ============================================================
# 本地内存存储（Prometheus 不可用时的轻量降级方案）
# ============================================================

# 事件日志：记录每一次 record_event 调用的完整信息
_EVENTS: list[dict[str, Any]] = []
# 累计计数器：按事件名统计调用次数
_METRICS: CollectionCounter[str] = CollectionCounter()
# Prometheus Counter 缓存（按事件名索引）
_PROM_COUNTERS: dict[str, Any] = {}
# Prometheus Histogram 缓存（按事件名索引，记录耗时分布）
_PROM_HISTOGRAMS: dict[str, Any] = {}
_PROM_MEASUREMENTS: dict[str, Any] = {}
_MEASUREMENTS: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}

_MEASUREMENT_SPECS = {
    "context_bundle_tokens": ("histogram", ("agent",), (0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)),
    "context_bundle_omitted_items": ("histogram", ("agent",), (0, 1, 2, 4, 8, 16, 32, 64)),
    "context_resolver_decisions": ("counter", ("outcome",), ()),
    "context_focus_cas_conflicts": ("counter", (), ()),
    "dimension_mentions_total": ("counter", ("source",), ()),
    "dimension_decisions_total": ("counter", ("decision",), ()),
    "dimension_support_total": ("counter", ("status",), ()),
    "semantic_audit_latency_ms": ("histogram", ("stage",), (0, 1, 2, 5, 10, 25, 50, 100)),
    "semantic_route_canary_actions": ("counter", ("action",), ()),
    "semantic_route_canary_preflight_outcomes": ("counter", ("outcome",), ()),
    "semantic_route_canary_guardrails": ("counter", ("reason",), ()),
    "semantic_route_canary_latency_ms": ("histogram", (), (0, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 180000)),
    "semantic_route_canary_input_tokens": ("counter", (), ()),
    "semantic_route_canary_output_tokens": ("counter", (), ()),
    "semantic_route_canary_estimated_cost_usd": ("counter", (), ()),
}
_MEASUREMENT_LABEL_VALUES = {
    "agent": {"triage", "metric_definition", "data_query", "diagnostic", "report", "answer", "critic"},
    "outcome": {"continue", "new", "parallel_new", "clarify", "invalid", "plan_created"},
    "source": {"alias", "registry_exact", "comparison_parser", "operation_lexicon"},
    "decision": {"resolved", "clarify", "unsupported", "not_applicable"},
    "status": {"not_observed", "ok", "missing", "denied", "unknown"},
    "stage": {"mention", "support", "decision", "audit"},
    "action": {"complex_plan", "clarify", "legacy_triage"},
    "reason": {
        "interpreter_or_validator_failure",
        "router_validation_failure",
        "sealed_decision_rejected",
        "canary_authorization_failure",
        "scope_preflight_failure",
        "plan_creation_failure",
    },
}


def record_measurement(name: str, value: float = 1, *, labels: dict[str, str] | None = None) -> None:
    """Record a bounded, low-cardinality metric for architecture-level signals."""
    spec = _MEASUREMENT_SPECS.get(name)
    if spec is None:
        raise ValueError("unknown_observability_measurement")
    kind, label_names, buckets = spec
    supplied = labels or {}
    if set(supplied) != set(label_names):
        raise ValueError("invalid_observability_labels")
    safe_labels: dict[str, str] = {}
    for label_name in label_names:
        label_value = str(supplied[label_name])
        if label_value not in _MEASUREMENT_LABEL_VALUES.get(label_name, set()):
            label_value = "invalid"
        safe_labels[label_name] = label_value
    numeric = max(0.0, float(value))
    key = (name, tuple(sorted(safe_labels.items())))
    values = _MEASUREMENTS.setdefault(key, [0.0, 0.0])
    values[0] += numeric
    values[1] += 1
    if PrometheusCounter is None:
        return
    metric = _PROM_MEASUREMENTS.get(name)
    if metric is None:
        if kind == "counter":
            metric = PrometheusCounter(
                f"bizinsight_{name}_total", "BizInsight context architecture signal",
                labelnames=list(label_names),
            )
        else:
            metric = PrometheusHistogram(
                f"bizinsight_{name}", "BizInsight context architecture measurement",
                labelnames=list(label_names), buckets=buckets,
            )
        _PROM_MEASUREMENTS[name] = metric
    child = metric.labels(**safe_labels) if label_names else metric
    if kind == "counter":
        child.inc(numeric)
    else:
        child.observe(numeric)


# ============================================================
# record_event() — 统一的事件记录入口
# ============================================================

def record_event(name: str, attributes: dict[str, Any]) -> None:
    """记录一次业务事件，同时推送到 OTel Span、内存日志和 Prometheus。

    这是项目最重要的可观测性原语之一。所有关键通路
    （LLM 调用、MCP 工具、RAG 检索、agent 执行）都通过此函数记录。

    执行路径：
    1. 属性脱敏（redact_sensitive）
    2. 追加到内存事件列表（供电梯调试 / 本地开发查看）
    3. 创建 OpenTelemetry Span（供 Jaeger/Tempo 追踪）
    4. 递增本地 Counter
    5. 如果 Prometheus 可用，递增对应的 Counter + 记录 Histogram
    6. 如果有 cost 属性，累加到 LLM 成本指标

    参数：
    - name：事件名（如 "llm.call"、"agent.run"、"rag.retrieve"、"mcp.call"）
    - attributes：属性字典，支持的特殊 key：
        - duration_ms：操作耗时（毫秒），会记录到 Prometheus Histogram
        - llm.estimated_cost：LLM 估算成本（美元），会累加到成本指标
    """
    event = {"name": name, "attributes": redact_sensitive(attributes)}
    # 1. 内存事件列表
    _EVENTS.append(event)
    # 2. OpenTelemetry Span
    with _TRACER.start_as_current_span(name) as span:
        for key, value in event["attributes"].items():
            if value is not None and isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
    # 3. 本地 Counter
    _METRICS[f"{name}_total"] += 1
    # 4. Prometheus Counter（如果可用）
    if PrometheusCounter is not None:
        counter = _PROM_COUNTERS.get(name)
        if counter is None:
            counter = PrometheusCounter(
                f"bizinsight_{name.replace('.', '_')}_total",
                "BizInsight runtime events",
            )
            _PROM_COUNTERS[name] = counter
        counter.inc()
    # 5. Prometheus Histogram — 耗时分布（如果提供了 duration_ms）
    duration_ms = attributes.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and PrometheusHistogram is not None:
        histogram = _PROM_HISTOGRAMS.get(name)
        if histogram is None:
            histogram = PrometheusHistogram(
                f"bizinsight_{name.replace('.', '_')}_duration_seconds",
                "BizInsight operation duration",
                buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30),
            )
            _PROM_HISTOGRAMS[name] = histogram
        histogram.observe(duration_ms / 1000)
    # 6. LLM 成本累计
    cost = attributes.get("llm.estimated_cost")
    if isinstance(cost, (int, float)) and PrometheusCounter is not None:
        counter = _PROM_COUNTERS.get("llm_cost")
        if counter is None:
            counter = PrometheusCounter(
                "bizinsight_llm_estimated_cost_total",
                "Estimated LLM cost",
            )
            _PROM_COUNTERS["llm_cost"] = counter
        counter.inc(cost)


# ============================================================
# 指标导出
# ============================================================

def snapshot_metrics() -> dict[str, int]:
    """返回内存中累积的指标快照。

    供 /metrics 端点在 Prometheus 不可用时使用。
    """
    snapshot = dict(_METRICS)
    for (name, labels), values in _MEASUREMENTS.items():
        label_suffix = ",".join(f"{key}={value}" for key, value in labels)
        key = f"{name}{{{label_suffix}}}" if label_suffix else name
        kind = _MEASUREMENT_SPECS[name][0]
        snapshot[f"{key}_count" if kind == "histogram" else key] = int(
            values[1] if kind == "histogram" else values[0]
        )
    return snapshot


def prometheus_payload() -> bytes:
    """导出 Prometheus 文本格式的指标数据。

    如果 prometheus_client 已安装，使用其标准 generate_latest()；
    否则用内存 Counter 组装简单的文本格式。

    返回：
    - bytes：Prometheus 文本格式的指标数据
    """
    if generate_latest is None:
        lines = [
            f"bizinsight_{key.replace('.', '_')} {value}"
            for key, value in _METRICS.items()
        ]
        for (name, labels), values in _MEASUREMENTS.items():
            label_text = (
                "{" + ",".join(f'{key}="{value}"' for key, value in labels) + "}"
                if labels else ""
            )
            kind = _MEASUREMENT_SPECS[name][0]
            if kind == "histogram":
                lines.extend([
                    f"bizinsight_{name}_count{label_text} {int(values[1])}",
                    f"bizinsight_{name}_sum{label_text} {values[0]}",
                ])
            else:
                lines.append(f"bizinsight_{name}_total{label_text} {values[0]}")
        return ("\n".join(lines) + "\n").encode("utf-8")
    return generate_latest()
