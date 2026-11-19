"""Runtime contracts for the metric registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

MetricCapability = Literal["value", "trend", "compare", "breakdown", "period", "statistics"]
DisplayFormat = Literal["currency", "percentage", "hours", "days", "score", "number"]
MetricScope = Literal["platform", "seller"]
MetricOperation = Literal["value", "trend", "compare", "breakdown", "period", "statistics"]
Surface = Literal["explorer", "report", "target", "chat"]
PeriodAggregationKind = Literal["weighted_ratio"]

ALL_CAPABILITIES: tuple[MetricCapability, ...] = (
    "value", "trend", "compare", "breakdown", "period", "statistics",
)
ALL_DISPLAY_FORMATS: tuple[DisplayFormat, ...] = (
    "currency", "percentage", "hours", "days", "score", "number",
)
ALL_SURFACES: tuple[Surface, ...] = (
    "explorer", "report", "target", "chat",
)


@dataclass(frozen=True)
class ScopeImplementation:
    """Per-scope (platform/seller) implementation of a metric.

    A metric registered only for ``platform`` has no seller entry;
    attempting to query it in seller scope is a ``ScopeNotAllowed`` error.
    """

    handler_key: str
    capabilities: frozenset[MetricCapability]
    dimensions: tuple[str, ...]
    lineage_source: str
    lineage_description: str = ""
    dimension_semantics: dict[str, "DimensionMetricSemantics"] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensionMetricSemantics:
    additive: bool
    semantics: str


@dataclass(frozen=True)
class DimensionDefinition:
    dimension_id: str
    source: str
    value_type: str


@dataclass(frozen=True)
class MetricPlanningMetadata:
    """Non-executable business language exposed to the planning LLM.

    This deliberately contains no SQL, handler key, schema name, or trusted
    scope value.  It is the only metric metadata allowed into the planner
    prompt.
    """

    phrases: tuple[str, ...]
    examples: tuple[str, ...]
    intents: tuple[str, ...]
    limitations: str = ""
    # Only these phrases may enter the deterministic exact-match path.
    exact_aliases: tuple[str, ...] = ()
    # Natural-language hints are retrieval/LLM context only.
    semantic_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class PeriodAggregation:
    """Declarative metadata for exact period aggregation of composite metrics.

    The registry stores the component metric IDs, while the period service
    remains responsible for fetching and calculating the result.  YAML can
    therefore describe business semantics without containing executable code.
    """

    kind: PeriodAggregationKind
    numerator_metric: str
    denominator_metric: str
    multiplier: float = 1.0


@dataclass(frozen=True)
class AttributionFactorSpec:
    """Declarative source contract for one product-decomposition factor."""

    factor_id: str
    kind: Literal["sum", "ratio_of_sums"]
    metric_id: str | None = None
    numerator_metric: str | None = None
    denominator_metric: str | None = None

    @property
    def source_metric_ids(self) -> tuple[str, ...]:
        if self.kind == "sum":
            return (self.metric_id,) if self.metric_id else ()
        return tuple(item for item in (self.numerator_metric, self.denominator_metric) if item)


@dataclass(frozen=True)
class AttributionScopeCapability:
    method_ids: tuple[str, ...]
    dimensions: tuple[str, ...] = ()
    required_data: dict[str, tuple[str, ...]] = field(default_factory=dict)
    factor_specs: dict[str, tuple[AttributionFactorSpec, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AttributionCapability:
    """Explicit, conservative attribution support declared per metric/scope."""

    scopes: dict[str, AttributionScopeCapability]
    limitation: str = ""

    def for_scope(self, scope: str) -> AttributionScopeCapability:
        return self.scopes.get(scope, AttributionScopeCapability(method_ids=()))

    @property
    def supported(self) -> bool:
        return any(item.method_ids for item in self.scopes.values())


@dataclass(frozen=True)
class MetricDefinition:
    """Canonical runtime metric definition.

    Built from YAML metadata + handler catalog.  YAML never contains
    SQL, Python module paths, or executable expressions.
    """

    metric_id: str
    display_name: str
    aliases: tuple[str, ...]
    unit: str
    display_format: DisplayFormat
    base_grain: Literal["order", "customer", "review", "day"]
    aggregation: Literal["sum", "weighted_ratio", "average", "distinct_count"]
    direction: Literal["higher_is_better", "lower_is_better", "neutral"]
    version: str
    documentation: str
    display_order: int
    surfaces: tuple[Surface, ...]
    implementations: dict[MetricScope, ScopeImplementation]
    data_dependencies: tuple[str, ...] = ()
    planning: MetricPlanningMetadata = MetricPlanningMetadata(
        phrases=(), examples=(), intents=(), limitations="",
    )
    period_aggregation: PeriodAggregation | None = None
    attribution: AttributionCapability = AttributionCapability(
        scopes={}, limitation="归因能力尚未声明。",
    )


@dataclass(frozen=True)
class MetricQueryRequest:
    """Immutable request object passed to every handler.

    **platform** queries receive a ``PlatformQueryContext``.
    **seller** queries receive a ``ControlledAnalysisScope``
    (from ``app.schemas.explorer``) as context.

    Exactly one of ``metric_date`` | (``start_date`` + ``end_date``) is set
    depending on ``operation``.
    """

    operation: MetricOperation
    context: PlatformQueryContext | ControlledAnalysisScope  # noqa: F821
    metric_date: date | None = None
    start_date: date | None = None
    end_date: date | None = None
    left_date: date | None = None
    right_date: date | None = None
    granularity: str = "day"
    dimension: str | None = None
    limit: int = 10
    statistics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlatformQueryContext:
    """Trusted context for platform-scope metric queries.

    Carries the authenticated identity.  Created by the API layer,
    never by client input.
    """

    tenant_id: str = "platform"
    user_id: str = "system"


# Import needed for the type annotation above.
from app.schemas.explorer import ControlledAnalysisScope  # noqa: E402, F401
