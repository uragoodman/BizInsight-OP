"""Metric registry — single source of truth for metric metadata.

YAML driven, startup-validated, lazy-loaded singleton.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from app.analysis.attribution_methods import (
    ATTRIBUTION_REQUIRED_DATA_FIELDS,
    SUPPORTED_ATTRIBUTION_METHODS,
)
from app.metrics.contracts import (
    ALL_CAPABILITIES,
    ALL_DISPLAY_FORMATS,
    ALL_SURFACES,
    ControlledAnalysisScope,
    MetricDefinition,
    AttributionCapability,
    AttributionFactorSpec,
    AttributionScopeCapability,
    DimensionDefinition,
    DimensionMetricSemantics,
    PeriodAggregation,
    MetricPlanningMetadata,
    MetricQueryRequest,
    MetricScope,
    PlatformQueryContext,
    ScopeImplementation,
)
from app.metrics.exceptions import (
    CapabilityNotSupported,
    DimensionNotAllowed,
    MetricNotFound,
    RegistryValidationError,
    ScopeNotAllowed,
)
from app.metrics.handler_catalog import get_handler

_YAML_DIR = os.getenv(
    "METRICS_YAML_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "config" / "metrics"),
)


class MetricRegistry:
    """Lazy-loaded, validated metric registry.

    Usage::

        registry = MetricRegistry()
        registry.load()            # scan + parse + validate
        gmv = registry.get("gmv")  # → MetricDefinition
    """

    def __init__(self, yaml_dir: str = "") -> None:
        self._yaml_dir = yaml_dir or _YAML_DIR
        self._metrics: dict[str, MetricDefinition] = {}
        self._dimensions: dict[str, DimensionDefinition] = {}
        self._alias_index: dict[str, str] = {}
        self._catalog_digest = ""
        self._catalog_digest_seller_enabled: bool | None = None
        self._attribution_catalog: dict[str, dict[str, Any]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Scan the YAML directory, parse every file, build indexes."""
        self._metrics.clear()
        self._dimensions.clear()
        self._alias_index.clear()

        yaml_path = Path(self._yaml_dir)
        if not yaml_path.is_dir():
            raise RegistryValidationError(
                f"Metrics YAML directory not found: {yaml_path}"
            )

        seen_ids: set[str] = set()
        seen_aliases: dict[str, str] = {}

        attribution_file = yaml_path / "attribution.yaml"
        if attribution_file.exists():
            with open(attribution_file, encoding="utf-8") as f:
                attribution_data = yaml.safe_load(f) or {}
            raw_catalog = attribution_data.get("metrics", {})
            if not isinstance(raw_catalog, dict):
                raise RegistryValidationError("attribution.yaml metrics must be a mapping")
            self._attribution_catalog = raw_catalog

        for yaml_file in sorted(path for path in yaml_path.glob("*.yaml") if path.name != "attribution.yaml"):
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            planning_defaults = (data or {}).get("planning_defaults", {})
            planning_by_metric = (data or {}).get("planning_by_metric", {})
            for dimension_raw in (data or {}).get("dimensions", []):
                definition = DimensionDefinition(
                    dimension_id=str(dimension_raw.get("dimension_id", "")).strip(),
                    source=str(dimension_raw.get("source", "")).strip(),
                    value_type=str(dimension_raw.get("type", "")).strip(),
                )
                if not definition.dimension_id or not definition.source or definition.value_type != "categorical":
                    raise RegistryValidationError("Dimension definitions require id, source, and categorical type")
                previous = self._dimensions.get(definition.dimension_id)
                if previous is not None and previous != definition:
                    raise RegistryValidationError(f"Conflicting dimension definition: {definition.dimension_id}")
                self._dimensions[definition.dimension_id] = definition
            for raw in (data or {}).get("metrics", []):
                raw = dict(raw)
                # YAML may declare category defaults and per-metric overrides;
                # the resulting planning metadata remains explicit config, not
                # executable Python behavior.
                raw["planning"] = {
                    **planning_defaults,
                    **raw.get("planning", {}),
                    **planning_by_metric.get(raw["metric_id"], {}),
                }
                defn = self._parse_metric(raw)
                if defn.metric_id in seen_ids:
                    raise RegistryValidationError(
                        f"Duplicate metric_id {defn.metric_id!r} "
                        f"in {yaml_file.name}"
                    )
                seen_ids.add(defn.metric_id)
                # ``exact_aliases`` is the authoritative deterministic
                # vocabulary.  Include it in both uniqueness validation and
                # server-side alias resolution even when a YAML file uses the
                # new split fields without the legacy ``aliases`` key.
                exact_phrases = (
                    defn.display_name,
                    *defn.aliases,
                    *defn.planning.exact_aliases,
                )
                for alias in exact_phrases:
                    normalized_alias = self._normalize_planning_phrase(alias)
                    if not normalized_alias:
                        continue
                    previous_metric = seen_aliases.get(normalized_alias)
                    if previous_metric is not None and previous_metric != defn.metric_id:
                        raise RegistryValidationError(
                            f"Duplicate alias/exact alias {alias!r}: "
                            f"{previous_metric!r} and {defn.metric_id!r}"
                        )
                    seen_aliases[normalized_alias] = defn.metric_id
                    self._alias_index[normalized_alias] = defn.metric_id
                self._metrics[defn.metric_id] = defn

        if attribution_file.exists():
            unknown_attribution = set(self._attribution_catalog) - seen_ids
            missing_attribution = seen_ids - set(self._attribution_catalog)
            if unknown_attribution or missing_attribution:
                raise RegistryValidationError(
                    "Attribution catalog must cover every registered metric exactly; "
                    f"unknown={sorted(unknown_attribution)}, missing={sorted(missing_attribution)}"
                )
            for metric_id, definition in self._metrics.items():
                for scope_name, scope_capability in definition.attribution.scopes.items():
                    for method_id, factors in scope_capability.factor_specs.items():
                        for factor in factors:
                            if factor.factor_id not in self._metrics:
                                raise RegistryValidationError(
                                    f"{metric_id}[{scope_name}]: factor {factor.factor_id!r} is not a registered metric"
                                )
                            if scope_name not in self._metrics[factor.factor_id].implementations:
                                raise RegistryValidationError(
                                    f"{metric_id}[{scope_name}]: factor {factor.factor_id!r} has no {scope_name} implementation"
                                )
                            for source_metric_id in factor.source_metric_ids:
                                source = self._metrics.get(source_metric_id)
                                if source is None:
                                    raise RegistryValidationError(
                                        f"{metric_id}[{scope_name}]: factor source {source_metric_id!r} is not registered"
                                    )
                                if scope_name not in source.implementations:
                                    raise RegistryValidationError(
                                        f"{metric_id}[{scope_name}]: factor source {source_metric_id!r} has no {scope_name} implementation"
                                    )
                                if source.aggregation != "sum":
                                    raise RegistryValidationError(
                                        f"{metric_id}[{scope_name}]: factor source {source_metric_id!r} must be additive"
                                    )
        self._catalog_digest = self._compute_catalog_digest()
        from app.core.config import get_settings
        self._catalog_digest_seller_enabled = bool(get_settings().seller_dimension_enabled)
        self._loaded = True

    def validate(self) -> None:
        """Fail-fast: check every handler, doc, dimension, and capability."""
        if not self._loaded:
            self.load()

        errors: list[str] = []
        for mid, defn in self._metrics.items():
            # Handler existence
            for scope_key, impl in defn.implementations.items():
                try:
                    get_handler(impl.handler_key)
                except KeyError:
                    errors.append(
                        f"{mid}[{scope_key}]: handler "
                        f"{impl.handler_key!r} not in HANDLER_REGISTRY"
                    )
            # Documentation file
            if not defn.documentation:
                errors.append(f"{mid}: documentation is required")
            else:
                doc_path = Path(
                    Path(__file__).resolve().parent.parent.parent,
                    defn.documentation,
                )
                if not doc_path.exists():
                    errors.append(
                        f"{mid}: doc not found at {defn.documentation}"
                    )
            invalid_surfaces = set(defn.surfaces) - set(ALL_SURFACES)
            if invalid_surfaces:
                errors.append(
                    f"{mid}: invalid surface(s): {sorted(invalid_surfaces)}"
                )
            # Breakdown capability must have dimensions
            for scope_key, impl in defn.implementations.items():
                if "breakdown" in impl.capabilities and not impl.dimensions:
                    errors.append(
                        f"{mid}[{scope_key}]: breakdown capability "
                        "requires at least one dimension"
                    )
            if not defn.planning.phrases or not defn.planning.examples:
                errors.append(f"{mid}: planning requires phrases and examples")
            period = defn.period_aggregation
            if period is not None:
                if defn.aggregation != "weighted_ratio":
                    errors.append(
                        f"{mid}: period_aggregation requires weighted_ratio aggregation"
                    )
                for component in (period.numerator_metric, period.denominator_metric):
                    if component not in self._metrics:
                        errors.append(
                            f"{mid}: period aggregation references unknown metric {component!r}"
                        )
            if period is not None:
                common_scopes = set(defn.implementations)
                for component in (period.numerator_metric, period.denominator_metric):
                    if component in self._metrics:
                        common_scopes &= set(self._metrics[component].implementations)
                if not common_scopes:
                    errors.append(
                        f"{mid}: period aggregation components have no common scope"
                    )
                numerator = self._metrics.get(period.numerator_metric)
                denominator = self._metrics.get(period.denominator_metric)
                if numerator is not None and denominator is not None:
                    numerator_family = self._unit_family(numerator.unit)
                    denominator_family = self._unit_family(denominator.unit)
                    if (
                        numerator_family == "percentage"
                        or denominator_family == "percentage"
                        or (
                            numerator_family != denominator_family
                            and denominator_family not in {"orders", "customers", "items"}
                        )
                    ):
                        errors.append(
                            f"{mid}: incompatible period aggregation units "
                            f"{numerator.unit!r}/{denominator.unit!r}"
                        )

        # A composite metric must not depend on itself, directly or through
        # another composite metric.  Detecting this at startup avoids a
        # recursive aggregation failure in the request path.
        graph = {
            metric_id: {
                definition.period_aggregation.numerator_metric,
                definition.period_aggregation.denominator_metric,
            }
            for metric_id, definition in self._metrics.items()
            if definition.period_aggregation is not None
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(metric_id: str) -> bool:
            if metric_id in visiting:
                return True
            if metric_id in visited:
                return False
            visiting.add(metric_id)
            cycle = any(visit(child) for child in graph.get(metric_id, set()))
            visiting.remove(metric_id)
            visited.add(metric_id)
            return cycle

        for metric_id in graph:
            if visit(metric_id):
                errors.append(f"{metric_id}: cyclic period aggregation dependency")
        if errors:
            raise RegistryValidationError(
                "Metric registry validation failed:\n  "
                + "\n  ".join(errors)
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get(self, metric_id: str) -> MetricDefinition:
        """Look up a canonical metric by ID."""
        self._require_loaded()
        try:
            return self._metrics[metric_id]
        except KeyError as exc:
            raise MetricNotFound(f"Unknown metric: {metric_id!r}") from exc

    def resolve(self, name_or_alias: str) -> str:
        """Resolve a metric ID or alias to a canonical metric ID."""
        self._require_loaded()
        if name_or_alias in self._metrics:
            return name_or_alias
        canonical = self._alias_index.get(self._normalize_planning_phrase(name_or_alias))
        if canonical is not None:
            return canonical
        raise MetricNotFound(f"Unknown metric or alias: {name_or_alias!r}")

    def list(self, scope: MetricScope | None = None) -> list[MetricDefinition]:
        """List all metrics, optionally filtered by scope availability."""
        self._require_loaded()
        result = list(self._metrics.values())
        if scope is not None:
            result = [m for m in result if scope in m.implementations]
        return result

    def dimensions(self) -> tuple[DimensionDefinition, ...]:
        self._require_loaded()
        return tuple(self._dimensions[key] for key in sorted(self._dimensions))

    def dimension(self, dimension_id: str) -> DimensionDefinition | None:
        self._require_loaded()
        return self._dimensions.get(dimension_id)

    @staticmethod
    def dimension_enabled(dimension_id: str) -> bool:
        if dimension_id != "seller":
            return True
        from app.core.config import get_settings
        return bool(get_settings().seller_dimension_enabled)

    def planning_catalog(
        self,
        *,
        scope: MetricScope,
        allowed_metrics: set[str] | None = None,
        surface: str = "chat",
    ) -> list[dict[str, object]]:
        """Return the safe, non-executable catalog for semantic planning."""
        catalog: list[dict[str, object]] = []
        for metric in self.list(scope):
            if surface not in metric.surfaces:
                continue
            if allowed_metrics is not None and metric.metric_id not in allowed_metrics:
                continue
            impl = metric.implementations[scope]
            capabilities = set(impl.capabilities)
            dimensions = set(impl.dimensions)
            if (
                scope == "seller"
                and metric.metric_id in {"gmv", "order_count"}
                and self.supports(metric.metric_id, "breakdown", scope="platform")
                and self.is_dimension_allowed(metric.metric_id, "seller", scope="platform")
            ):
                # This is a deliberate single-seller degenerate grouping
                # capability; it does not make other seller-scope dimensions
                # inherit platform grouping capabilities.
                capabilities.add("breakdown")
                dimensions.add("seller")
            catalog.append({
                "metric_id": metric.metric_id,
                "display_name": metric.display_name,
                "aliases": list(metric.aliases),
                "exact_aliases": list(metric.planning.exact_aliases or metric.aliases),
                "semantic_terms": list(metric.planning.semantic_terms),
                "phrases": list(metric.planning.phrases),
                "examples": list(metric.planning.examples),
                "intents": list(metric.planning.intents),
                # Safe planning metadata only.  This is an allow-list of
                # operations, not an execution detail or handler reference.
                "capabilities": sorted(capabilities),
                "dimensions": [dimension for dimension in sorted(dimensions) if self.dimension_enabled(dimension)],
                "limitations": metric.planning.limitations or impl.lineage_description,
            })
        return sorted(catalog, key=lambda item: str(item["metric_id"]))

    @property
    def catalog_digest(self) -> str:
        """Digest of execution-relevant and semantic metric metadata."""

        self._require_loaded()
        # Feature flags can be overridden in tests and are operator runtime
        # configuration. Recompute only when the exposed seller catalog state
        # changes, not on every metric authorization check.
        from app.core.config import get_settings
        seller_enabled = bool(get_settings().seller_dimension_enabled)
        if (
            not self._catalog_digest
            or seller_enabled != self._catalog_digest_seller_enabled
        ):
            self._catalog_digest = self._compute_catalog_digest()
            self._catalog_digest_seller_enabled = seller_enabled
        return self._catalog_digest

    def _compute_catalog_digest(self) -> str:
        from app.core.config import get_settings

        seller_enabled = bool(get_settings().seller_dimension_enabled)
        payload: list[dict[str, object]] = []
        visible_dimensions = [
            {"dimension_id": item.dimension_id, "source": item.source, "type": item.value_type}
            for item in (self._dimensions[key] for key in sorted(self._dimensions))
            if seller_enabled or item.dimension_id != "seller"
        ]
        # The pre-seller serializer had no dimension registry field. Omitting
        # this key when all registered dimensions are feature-gated preserves
        # the exact historical payload (and digest) with the flag off.
        if visible_dimensions:
            payload.append({"dimension_registry": visible_dimensions})
        for metric_id in sorted(self._metrics):
            metric = self._metrics[metric_id]
            implementation_payload: dict[str, dict[str, object]] = {}
            for scope, impl in sorted(metric.implementations.items()):
                visible_impl_dimensions = [
                    dimension for dimension in impl.dimensions
                    if seller_enabled or dimension != "seller"
                ]
                implementation: dict[str, object] = {
                    "handler_key": impl.handler_key,
                    "capabilities": sorted(impl.capabilities),
                    "dimensions": visible_impl_dimensions,
                    "lineage_source": impl.lineage_source,
                }
                visible_semantics = {
                    key: {"additive": value.additive, "semantics": value.semantics}
                    for key, value in sorted(impl.dimension_semantics.items())
                    if seller_enabled or key != "seller"
                }
                # Old catalog payloads did not contain this field. Emit it
                # only if a currently visible dimension has declared semantics.
                if visible_semantics:
                    implementation["dimension_semantics"] = visible_semantics
                implementation_payload[scope] = implementation
            payload.append({
                "metric_id": metric.metric_id,
                "display_name": metric.display_name,
                "aliases": list(metric.aliases),
                "unit": metric.unit,
                "definition": metric.documentation,
                "version": metric.version,
                "surfaces": list(metric.surfaces),
                "planning": {
                    "exact_aliases": list(metric.planning.exact_aliases),
                    "semantic_terms": list(metric.planning.semantic_terms),
                    "phrases": list(metric.planning.phrases),
                    "examples": list(metric.planning.examples),
                    "intents": list(metric.planning.intents),
                    "limitations": metric.planning.limitations,
                },
                "implementations": implementation_payload,
                "attribution": {
                    "scopes": {
                        scope: {
                            "methods": list(capability.method_ids),
                            "dimensions": list(capability.dimensions),
                            "required_data": {
                                method: list(fields)
                                for method, fields in sorted(capability.required_data.items())
                            },
                            "factor_specs": {
                                method: [
                                    {
                                        "factor_id": factor.factor_id,
                                        "kind": factor.kind,
                                        "metric_id": factor.metric_id,
                                        "numerator_metric": factor.numerator_metric,
                                        "denominator_metric": factor.denominator_metric,
                                    }
                                    for factor in factors
                                ]
                                for method, factors in sorted(capability.factor_specs.items())
                            },
                        }
                        for scope, capability in sorted(metric.attribution.scopes.items())
                    },
                    "limitation": metric.attribution.limitation,
                },
            })
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def supports(
        self, metric_id: str, capability: str, scope: MetricScope = "platform",
    ) -> bool:
        """Check whether a metric supports a capability in the given scope."""
        defn = self.get(metric_id)
        impl = defn.implementations.get(scope)
        if impl is None:
            return False
        return capability in impl.capabilities

    def is_dimension_allowed(
        self, metric_id: str, dimension: str, scope: MetricScope = "platform",
    ) -> bool:
        """Check whether a dimension is allowed for this metric + scope."""
        defn = self.get(metric_id)
        impl = defn.implementations.get(scope)
        if impl is None:
            raise ScopeNotAllowed(
                f"{metric_id} has no {scope} implementation"
            )
        return dimension in impl.dimensions and self.dimension_enabled(dimension)

    def get_implementation(
        self, metric_id: str, scope: MetricScope,
    ) -> ScopeImplementation:
        """Get the scope-specific implementation."""
        defn = self.get(metric_id)
        impl = defn.implementations.get(scope)
        if impl is None:
            raise ScopeNotAllowed(
                f"{metric_id} has no {scope} implementation"
            )
        return impl

    def get_period_aggregation(self, metric_id: str) -> PeriodAggregation | None:
        """Return declarative exact-period aggregation metadata, if present."""

        return self.get(self.resolve(metric_id)).period_aggregation

    def execute(
        self,
        metric_id: str,
        operation: str = "value",
        *,
        scope: MetricScope = "platform",
        metric_date: date | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        left_date: date | None = None,
        right_date: date | None = None,
        granularity: str = "day",
        dimension: str | None = None,
        limit: int = 10,
        statistics: tuple[str, ...] | list[str] | None = None,
        seller_scope: ControlledAnalysisScope | None = None,
    ) -> dict:
        """Execute a metric query through the registry handler.

        Platform queries use ``PlatformQueryContext``.
        Seller queries use the provided ``seller_scope``.
        """
        # Resolve alias to canonical ID first
        canonical = self.resolve(metric_id)

        defn = self.get(canonical)
        impl = self.get_implementation(canonical, scope)
        if operation not in impl.capabilities:
            raise CapabilityNotSupported(
                f"{canonical}: {operation!r} not supported in {scope} scope"
            )

        # Seller queries require a trusted ControlledAnalysisScope — fail closed
        if scope == "seller" and seller_scope is None:
            raise ScopeNotAllowed(
                f"{canonical}: seller scope query requires a trusted "
                f"ControlledAnalysisScope (seller_scope parameter)"
            )

        # Dimension validation for breakdown operations
        if operation == "breakdown" and dimension is not None:
            if not self.is_dimension_allowed(canonical, dimension, scope):
                raise DimensionNotAllowed(
                    f"{canonical}: dimension {dimension!r} is not allowed "
                    f"in {scope} scope"
                )

        context: PlatformQueryContext | ControlledAnalysisScope = (
            seller_scope if scope == "seller"
            else PlatformQueryContext()
        )
        request = MetricQueryRequest(
            operation=operation,  # type: ignore[arg-type]
            context=context,
            metric_date=metric_date,
            start_date=start_date,
            end_date=end_date,
            left_date=left_date,
            right_date=right_date,
            granularity=granularity,
            dimension=dimension,
            limit=limit,
            statistics=tuple(statistics or ()),
        )
        # This metric's seller implementation is customer-grain but requires
        # a same-seller historical purchase test. It cannot use explorer's
        # generic item-metric expression map; keep the narrow dispatch here
        # until a dedicated handler-catalog entry is introduced.
        if canonical == "repeat_purchase_rate" and scope == "seller" and operation == "value":
            from app.data.repositories.daily_metrics import query_seller_repeat_purchase_rate

            return query_seller_repeat_purchase_rate(seller_scope, metric_date)
        handler = get_handler(impl.handler_key)
        return handler(request)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_metric(self, raw: dict[str, Any]) -> MetricDefinition:
        mid = raw["metric_id"]

        # Parse implementations
        impls: dict[MetricScope, ScopeImplementation] = {}
        for scope_key in ("platform", "seller"):
            scope_raw = raw.get("implementations", {}).get(scope_key)
            if scope_raw is None:
                continue
            self._validate_capabilities(mid, scope_key, scope_raw)
            self._validate_dimensions(mid, scope_key, scope_raw)
            dimension_semantics_raw = scope_raw.get("dimension_semantics") or {}
            if set(dimension_semantics_raw) - set(scope_raw.get("dimensions", [])):
                raise RegistryValidationError(f"{mid}[{scope_key}]: dimension semantics reference an undeclared dimension")
            for dimension_id, semantic_raw in dimension_semantics_raw.items():
                if not isinstance(semantic_raw, dict) or not isinstance(semantic_raw.get("additive"), bool) or not semantic_raw.get("semantics"):
                    raise RegistryValidationError(f"{mid}[{scope_key}]: invalid semantics for dimension {dimension_id!r}")
            capabilities = set(scope_raw.get("capabilities", []))
            # A trend implementation can summarize the same scoped points.
            # Derive this capability centrally instead of duplicating it in
            # every existing YAML metric entry.
            if "trend" in capabilities:
                capabilities.add("statistics")
            impls[scope_key] = ScopeImplementation(
                handler_key=scope_raw["handler_key"],
                capabilities=frozenset(capabilities),
                dimensions=tuple(scope_raw.get("dimensions", [])),
                dimension_semantics={
                    str(key): DimensionMetricSemantics(
                        additive=bool(value.get("additive")),
                        semantics=str(value.get("semantics", "")),
                    )
                    for key, value in (scope_raw.get("dimension_semantics") or {}).items()
                },
                lineage_source=scope_raw.get("lineage_source", ""),
                lineage_description=scope_raw.get("lineage_description", ""),
            )

        planning_raw = raw.get("planning") or {}
        display_name = raw.get("display_name", mid)
        # ``aliases`` remains a compatibility input for existing metric files.
        # New files should use exact_aliases/semantic_terms explicitly.  Until
        # migrated, legacy aliases are treated as exact aliases and are still
        # protected by the global uniqueness check above.
        # ``aliases`` is the legacy exact-alias field.  Merge it with the new
        # explicit field during migration so adding semantic metadata cannot
        # accidentally remove an existing deterministic alias.
        exact_aliases = tuple(dict.fromkeys([
            *(raw.get("aliases") or []),
            *(raw.get("exact_aliases") or []),
            *(planning_raw.get("exact_aliases") or []),
        ]))
        semantic_terms = tuple(
            planning_raw.get("semantic_terms")
            or raw.get("semantic_terms")
            or planning_raw.get("intents", [])
        )
        phrases = tuple(
            planning_raw.get("phrases")
            or [display_name, *exact_aliases, *semantic_terms]
        )
        examples = tuple(str(value).format(display_name=display_name, metric_id=mid) for value in (
            planning_raw.get("examples") or [f"查询{display_name}"]
        ))
        intents = tuple(planning_raw.get("intents") or self._derived_intents(impls))
        planning = MetricPlanningMetadata(
            phrases=phrases,
            examples=examples,
            intents=intents,
            limitations=str(planning_raw.get("limitations", "")),
            exact_aliases=exact_aliases,
            semantic_terms=semantic_terms,
        )

        period_raw = raw.get("period_aggregation")
        period_aggregation = None
        if period_raw is not None:
            if not isinstance(period_raw, dict):
                raise RegistryValidationError(
                    f"{mid}: period_aggregation must be a mapping"
                )
            kind = period_raw.get("kind", "weighted_ratio")
            if kind != "weighted_ratio":
                raise RegistryValidationError(
                    f"{mid}: unsupported period_aggregation kind {kind!r}"
                )
            numerator = str(period_raw.get("numerator_metric", "")).strip()
            denominator = str(period_raw.get("denominator_metric", "")).strip()
            if not numerator or not denominator:
                raise RegistryValidationError(
                    f"{mid}: period_aggregation requires numerator_metric and denominator_metric"
                )
            try:
                multiplier = float(period_raw.get("multiplier", 1.0))
            except (TypeError, ValueError) as exc:
                raise RegistryValidationError(
                    f"{mid}: period_aggregation multiplier must be numeric"
                ) from exc
            period_aggregation = PeriodAggregation(
                kind="weighted_ratio",
                numerator_metric=numerator,
                denominator_metric=denominator,
                multiplier=multiplier,
            )

        attribution_raw = self._attribution_catalog.get(mid)
        if attribution_raw is None:
            attribution = AttributionCapability(scopes={}, limitation="归因能力尚未声明。")
        else:
            if not isinstance(attribution_raw, dict):
                raise RegistryValidationError(f"{mid}: attribution entry must be a mapping")
            limitation = str(attribution_raw.get("limitation", ""))
            scopes_raw = attribution_raw.get("scopes", {})
            if not isinstance(scopes_raw, dict):
                raise RegistryValidationError(f"{mid}: attribution scopes must be a mapping")
            scopes: dict[str, AttributionScopeCapability] = {}
            allowed_methods = SUPPORTED_ATTRIBUTION_METHODS
            for scope_key, scope_capability in scopes_raw.items():
                if scope_key not in impls or not isinstance(scope_capability, dict):
                    raise RegistryValidationError(f"{mid}: invalid attribution scope {scope_key!r}")
                methods = tuple(str(item) for item in scope_capability.get("methods", []))
                dimensions = tuple(str(item) for item in scope_capability.get("dimensions", []))
                required_data_raw = scope_capability.get("required_data", {})
                if not isinstance(required_data_raw, dict):
                    raise RegistryValidationError(f"{mid}[{scope_key}]: attribution required_data must be a mapping")
                required_data: dict[str, tuple[str, ...]] = {}
                for method_id, fields_raw in required_data_raw.items():
                    if not isinstance(fields_raw, list) or not fields_raw:
                        raise RegistryValidationError(
                            f"{mid}[{scope_key}]: required_data for {method_id!r} must be a nonempty list"
                        )
                    fields = tuple(str(item).strip() for item in fields_raw)
                    if any(not item for item in fields) or len(set(fields)) != len(fields):
                        raise RegistryValidationError(
                            f"{mid}[{scope_key}]: required_data for {method_id!r} contains empty/duplicate fields"
                        )
                    expected_fields = ATTRIBUTION_REQUIRED_DATA_FIELDS.get(str(method_id))
                    if expected_fields is not None and set(fields) != expected_fields:
                        missing = sorted(expected_fields - set(fields))
                        unknown = sorted(set(fields) - expected_fields)
                        raise RegistryValidationError(
                            f"{mid}[{scope_key}]: required_data for {method_id!r} does not match "
                            f"the deterministic method contract; missing={missing}, unknown={unknown}"
                        )
                    required_data[str(method_id)] = fields
                factor_specs_raw = scope_capability.get("factor_specs", {})
                if not isinstance(factor_specs_raw, dict):
                    raise RegistryValidationError(f"{mid}[{scope_key}]: attribution factor_specs must be a mapping")
                factor_specs: dict[str, tuple[AttributionFactorSpec, ...]] = {}
                for method_id, raw_specs in factor_specs_raw.items():
                    method_id = str(method_id)
                    if method_id != "product_shapley_v1" or method_id not in methods:
                        raise RegistryValidationError(
                            f"{mid}[{scope_key}]: factor_specs are only valid for declared product_shapley_v1"
                        )
                    if not isinstance(raw_specs, list) or len(raw_specs) != 2:
                        raise RegistryValidationError(
                            f"{mid}[{scope_key}]: product_shapley_v1 requires exactly two factor specs"
                        )
                    parsed_specs: list[AttributionFactorSpec] = []
                    seen_factor_ids: set[str] = set()
                    for item in raw_specs:
                        if not isinstance(item, dict):
                            raise RegistryValidationError(f"{mid}[{scope_key}]: factor spec must be a mapping")
                        factor_id = str(item.get("factor_id") or "").strip()
                        kind = item.get("kind")
                        if not factor_id or factor_id in seen_factor_ids:
                            raise RegistryValidationError(f"{mid}[{scope_key}]: factor IDs must be nonempty and unique")
                        seen_factor_ids.add(factor_id)
                        if kind == "sum":
                            metric_id = str(item.get("metric_id") or "").strip()
                            if not metric_id or any(item.get(key) is not None for key in ("numerator_metric", "denominator_metric")):
                                raise RegistryValidationError(f"{mid}[{scope_key}]: sum factor requires only metric_id")
                            parsed_specs.append(AttributionFactorSpec(
                                factor_id=factor_id, kind="sum", metric_id=metric_id,
                            ))
                        elif kind == "ratio_of_sums":
                            numerator = str(item.get("numerator_metric") or "").strip()
                            denominator = str(item.get("denominator_metric") or "").strip()
                            if not numerator or not denominator or item.get("metric_id") is not None:
                                raise RegistryValidationError(
                                    f"{mid}[{scope_key}]: ratio_of_sums factor requires numerator_metric and denominator_metric"
                                )
                            parsed_specs.append(AttributionFactorSpec(
                                factor_id=factor_id, kind="ratio_of_sums",
                                numerator_metric=numerator, denominator_metric=denominator,
                            ))
                        else:
                            raise RegistryValidationError(
                                f"{mid}[{scope_key}]: unsupported factor kind {kind!r}"
                            )
                    factor_specs[method_id] = tuple(parsed_specs)
                unknown_methods = set(methods) - allowed_methods
                if unknown_methods:
                    raise RegistryValidationError(f"{mid}: unknown attribution method(s): {sorted(unknown_methods)}")
                if set(required_data) != set(methods):
                    raise RegistryValidationError(
                        f"{mid}[{scope_key}]: every declared method requires exactly one required_data contract"
                    )
                if ("product_shapley_v1" in methods) != ("product_shapley_v1" in factor_specs):
                    raise RegistryValidationError(
                        f"{mid}[{scope_key}]: product_shapley_v1 requires an explicit factor_specs contract"
                    )
                if methods and not (impls[scope_key].capabilities & {"breakdown", "compare", "period", "trend", "statistics"}):
                    raise RegistryValidationError(f"{mid}[{scope_key}]: attribution method declared without query capability")
                known_attribution_dimensions = {
                    "customer_state", "customer_city", "seller_state", "product_category",
                    "payment_type", "month_position", "day_of_week", "hour_of_day",
                    "quarter_of_year", "half_of_year",
                }
                if set(dimensions) - known_attribution_dimensions:
                    raise RegistryValidationError(f"{mid}[{scope_key}]: unknown attribution dimension")
                scopes[scope_key] = AttributionScopeCapability(
                    method_ids=methods, dimensions=dimensions, required_data=required_data,
                    factor_specs=factor_specs,
                )
            if not any(cap.method_ids for cap in scopes.values()) and not limitation.strip():
                raise RegistryValidationError(f"{mid}: unsupported attribution must include a limitation")
            attribution = AttributionCapability(scopes=scopes, limitation=limitation)

        return MetricDefinition(
            metric_id=mid,
            display_name=raw.get("display_name", mid),
            aliases=tuple(raw.get("aliases", [])),
            unit=raw.get("unit", ""),
            display_format=self._validate_display_format(mid, raw),
            base_grain=raw.get("base_grain", "order"),
            aggregation=raw.get("aggregation", "sum"),
            direction=raw.get("direction", "neutral"),
            version=raw.get("version", "1.0"),
            documentation=raw.get("documentation", ""),
            display_order=raw.get("display_order", 99),
            surfaces=tuple(raw.get("surfaces", [])),
            implementations=impls,
            data_dependencies=("tenant_gmv_items",) if mid == "gmv" else (),
            planning=planning,
            period_aggregation=period_aggregation,
            attribution=attribution,
        )

    @staticmethod
    def _normalize_planning_phrase(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(value)).casefold()
        return "".join(normalized.split())

    @staticmethod
    def _unit_family(unit: str) -> str:
        """Map configured units to conservative compatibility families."""

        value = str(unit).casefold().replace(" ", "")
        if "%" in value or "percent" in value:
            return "percentage"
        if "brl" in value or "usd" in value or "eur" in value or "currency" in value:
            return "currency"
        if "order" in value:
            return "orders"
        if "customer" in value:
            return "customers"
        if "item" in value:
            return "items"
        if "hour" in value:
            return "hours"
        if "day" in value:
            return "days"
        return value

    @staticmethod
    def _derived_intents(impls: dict[MetricScope, ScopeImplementation]) -> tuple[str, ...]:
        operations = {
            "value": "metric_value", "trend": "metric_trend",
            "compare": "metric_compare", "breakdown": "metric_breakdown",
            "period": "metric_contribution", "statistics": "metric_statistics",
        }
        capabilities = set().union(*(impl.capabilities for impl in impls.values())) if impls else set()
        return tuple(operations[item] for item in operations if item in capabilities)

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_display_format(
        metric_id: str, raw: dict[str, Any],
    ) -> str:
        fmt = raw.get("display_format", "number")
        if fmt not in ALL_DISPLAY_FORMATS:
            raise RegistryValidationError(
                f"{metric_id}: invalid display_format {fmt!r}"
            )
        return fmt

    @staticmethod
    def _validate_capabilities(
        metric_id: str, scope_key: str, scope_raw: dict[str, Any],
    ) -> None:
        for cap in scope_raw.get("capabilities", []):
            if cap not in ALL_CAPABILITIES:
                raise RegistryValidationError(
                    f"{metric_id}[{scope_key}]: unknown capability {cap!r}"
                )

    def _validate_dimensions(
        self, metric_id: str, scope_key: str, scope_raw: dict[str, Any],
    ) -> None:
        KNOWN = {
            "customer_state", "customer_city", "seller_state",
            "product_category", "payment_type",
            "month_position", "day_of_week", "hour_of_day",
            "quarter_of_year", "half_of_year",
        }
        KNOWN.update(self._dimensions)
        for dim in scope_raw.get("dimensions", []):
            if dim not in KNOWN:
                raise RegistryValidationError(
                    f"{metric_id}[{scope_key}]: unknown dimension {dim!r}"
                )


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_REGISTRY: MetricRegistry | None = None


def get_registry() -> MetricRegistry:
    """Lazy singleton accessor — safe at import time."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = MetricRegistry()
    return _REGISTRY
