"""Closed semantic catalog for v3.4.1 Phase 0.

The catalog contains identifiers and grounding evidence only. Capability lookup
uses IDs and structured operation requirements; it has no utterance or display
name fields and does not inspect user text.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal

from pydantic import Field, model_validator

from app.context.semantic_contract_v2 import (
    ArgumentPath,
    CapabilityDecisionBasis,
    CapabilityReference,
    ClauseScope,
    ContractModel,
    OperationKind,
    RelationPredicate,
    RequestDomain,
    RoleName,
)


ConceptType = Literal[
    "BUSINESS_ENTITY", "METRIC", "DIMENSION", "ANALYSIS_TARGET", "ANALYSIS_METHOD",
    "TIME_CONCEPT", "ANALYSIS_PARAMETER",
]
RouteDecisionStatus = Literal[
    "SELECTED",
    "NO_MATCH",
    "UNSUPPORTED",
    "NEEDS_CONTEXT",
    "GENERAL_CONVERSATION",
    "AMBIGUOUS",
    "CONFLICT",
]
ROUTE_DECISION_STATUSES: tuple[RouteDecisionStatus, ...] = (
    "SELECTED",
    "NO_MATCH",
    "UNSUPPORTED",
    "NEEDS_CONTEXT",
    "GENERAL_CONVERSATION",
    "AMBIGUOUS",
    "CONFLICT",
)


class ConceptSpec(ContractModel):
    concept_id: str = Field(min_length=1)
    concept_type: ConceptType
    exact_aliases: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_aliases(self) -> ConceptSpec:
        if any(not alias for alias in (*self.exact_aliases, *self.synonyms)):
            raise ValueError("catalog aliases must be non-empty")
        if set(self.exact_aliases) & set(self.synonyms):
            raise ValueError("exact aliases and synonyms must be disjoint")
        return self


class OperationArgumentSpec(ContractModel):
    kind: OperationKind
    allowed_paths: tuple[ArgumentPath, ...]
    path_concept_domains: dict[ArgumentPath, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_path_concept_domains(self) -> OperationArgumentSpec:
        allowed_paths = set(self.allowed_paths)
        if not set(self.path_concept_domains) <= allowed_paths:
            raise ValueError("path concept domains must use allowed argument paths")
        for path, concept_ids in self.path_concept_domains.items():
            if not concept_ids:
                raise ValueError(f"path concept domain for {path!r} must not be empty")
            if len(concept_ids) != len(set(concept_ids)):
                raise ValueError(f"path concept domain for {path!r} must be unique")
        return self


class OperationRequirement(ContractModel):
    kind: OperationKind
    target_concept_ids: tuple[str, ...] = Field(min_length=1)
    required_argument_paths: tuple[ArgumentPath, ...] = ()


class VetoRequirement(ContractModel):
    kind: OperationKind
    target_concept_id: str = Field(min_length=1)
    clause_scope: ClauseScope


class RelationRequirement(ContractModel):
    subject_role: RoleName
    predicate: RelationPredicate
    object_role: RoleName


class BasisRequirement(ContractModel):
    concept_ids: tuple[str, ...] = Field(min_length=1)
    min_basis_matches: int | None = Field(default=None, ge=1)
    decision_status: Literal["OPEN", "FROZEN"]

    @model_validator(mode="after")
    def validate_open_decision(self) -> BasisRequirement:
        if self.decision_status == "OPEN" and self.min_basis_matches is not None:
            raise ValueError("OPEN basis requirements cannot choose a minimum")
        if self.decision_status == "FROZEN" and self.min_basis_matches is None:
            raise ValueError("FROZEN basis requirements require min_basis_matches")
        return self

    def is_satisfied_by(self, effective_basis_concept_ids: Iterable[str]) -> bool:
        """Return whether validated, distinct basis concepts meet the frozen minimum."""
        if self.decision_status != "FROZEN" or self.min_basis_matches is None:
            return False
        matched = set(self.concept_ids).intersection(effective_basis_concept_ids)
        return len(matched) >= self.min_basis_matches


class PrimaryIntentRequirement(ContractModel):
    """Consistency rule for an audit intent label and structured operations."""

    intent_id: str = Field(min_length=1)
    operation_kinds: tuple[OperationKind, ...] = Field(min_length=1)
    target_concept_ids: tuple[str, ...] = Field(min_length=1)


class CapabilitySpec(ContractModel):
    """A semantic ability, explicitly mapped to route and optional implementation."""

    capability_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    template_id: str | None = None
    concept_ids: tuple[str, ...] = Field(min_length=1)
    method_concept_ids: tuple[str, ...] = ()
    operation_requirements: tuple[OperationRequirement, ...] = Field(min_length=1)
    veto_requirements: tuple[VetoRequirement, ...] = ()
    required_roles: tuple[RoleName, ...] = ()
    required_relations: tuple[RelationRequirement, ...] = ()
    basis_requirement: BasisRequirement | None = None


class RouteSpec(ContractModel):
    """Route identity and safety envelope; no utterances or ranking scores."""

    route_id: str = Field(min_length=1)
    domain: RequestDomain
    capability_ids: tuple[CapabilityReference, ...] = ()
    tool_ids: tuple[str, ...] = ()
    creates_plan: bool = False
    database_access: bool = False

    @model_validator(mode="after")
    def general_domain_is_inert(self) -> RouteSpec:
        if self.domain == "GENERAL_CONVERSATION" and (
            self.capability_ids or self.tool_ids or self.creates_plan or self.database_access
        ):
            raise ValueError(
                "GENERAL_CONVERSATION cannot have capabilities, tools, plans, or database access"
            )
        return self


class RouteOutcome(ContractModel):
    """Status protocol; not a Router implementation."""

    status: RouteDecisionStatus
    domain: RequestDomain
    route_id: str | None = None
    capability_ids: tuple[CapabilityReference, ...] = ()
    # Explicit, deterministic choices for an otherwise valid ambiguous route.
    # These are never executable until the user clarifies and routing runs again.
    candidate_capability_ids: tuple[CapabilityReference, ...] = ()
    # Catalog-declared capability set for route_id. This lets the model reject
    # a selected capability that is not a member of the selected route.
    route_capability_ids: tuple[CapabilityReference, ...] = ()
    clarification_required: bool = False
    decision_basis: CapabilityDecisionBasis

    @model_validator(mode="after")
    def validate_status_boundary(self) -> RouteOutcome:
        route_capability_ids = [item.capability_id for item in self.route_capability_ids]
        if len(route_capability_ids) != len(set(route_capability_ids)):
            raise ValueError("route capability membership references must be unique")
        candidate_capability_ids = [item.capability_id for item in self.candidate_capability_ids]
        if len(candidate_capability_ids) != len(set(candidate_capability_ids)):
            raise ValueError("candidate capability references must be unique")
        if self.candidate_capability_ids and (
            self.status != "AMBIGUOUS"
            or not set(candidate_capability_ids)
            <= {item.capability_id for item in self.decision_basis.capability_ids}
        ):
            raise ValueError("route candidates must be a subset of an ambiguous decision basis")
        if self.status == "SELECTED" and self.candidate_capability_ids:
            raise ValueError("selected outcomes cannot also expose clarification candidates")
        if self.status != "SELECTED" and self.route_capability_ids:
            raise ValueError("only SELECTED outcomes may expose route capability membership")
        if self.domain == "GENERAL_CONVERSATION" and self.status != "GENERAL_CONVERSATION":
            raise ValueError("GENERAL_CONVERSATION domain must use its no-tool outcome status")
        if self.status == "GENERAL_CONVERSATION":
            if (
                self.domain != "GENERAL_CONVERSATION"
                or self.route_id not in {None, "general_conversation"}
                or self.capability_ids
                or self.decision_basis.capability_ids
                or self.clarification_required
            ):
                raise ValueError("GENERAL_CONVERSATION is a no-tool, no-plan domain outcome")
        elif self.status == "NO_MATCH":
            if (
                self.domain == "GENERAL_CONVERSATION"
                or self.route_id is not None
                or self.capability_ids
                or self.decision_basis.capability_ids
            ):
                raise ValueError("NO_MATCH applies to business domains without a capability")
        elif self.status == "UNSUPPORTED":
            if (
                self.domain == "GENERAL_CONVERSATION"
                or self.route_id is not None
                or self.capability_ids
                or self.decision_basis.capability_ids
            ):
                raise ValueError("UNSUPPORTED has no selected business route or capability")
        elif self.status == "NEEDS_CONTEXT":
            if self.domain == "GENERAL_CONVERSATION" or not self.clarification_required:
                raise ValueError("NEEDS_CONTEXT is a business outcome that requires clarification")
        elif self.status in {"AMBIGUOUS", "CONFLICT"} and not self.clarification_required:
            raise ValueError("AMBIGUOUS and CONFLICT require clarification")
        elif self.status == "SELECTED":
            if self.domain == "GENERAL_CONVERSATION" or not self.route_id or not self.capability_ids:
                raise ValueError("SELECTED requires a business route and capability reference")
            route_capability_ids = {item.capability_id for item in self.route_capability_ids}
            if not route_capability_ids:
                raise ValueError("SELECTED requires the catalog-declared route capabilities")
            if not {item.capability_id for item in self.capability_ids} <= route_capability_ids:
                raise ValueError("selected capability must belong to the selected route")
            if self.decision_basis.capability_ids != self.capability_ids:
                raise ValueError("SELECTED decision_basis must cite the selected capability references")
            if self.clarification_required:
                raise ValueError("SELECTED cannot require clarification")
            if (
                not self.decision_basis.operation_ids
                or not self.decision_basis.clause_ids
                or not self.decision_basis.effective_arguments_digest
            ):
                raise ValueError(
                    "SELECTED requires non-empty operation_ids, clause_ids, and effective_arguments_digest"
                )
        return self


class SemanticCatalog(ContractModel):
    version: str = Field(min_length=1)
    concepts: tuple[ConceptSpec, ...]
    intent_ids: tuple[str, ...]
    primary_intent_requirements: tuple[PrimaryIntentRequirement, ...]
    operation_arguments: tuple[OperationArgumentSpec, ...]
    capabilities: tuple[CapabilitySpec, ...]
    routes: tuple[RouteSpec, ...]
    role_names: tuple[RoleName, ...]
    relation_predicates: tuple[RelationPredicate, ...]

    @model_validator(mode="after")
    def validate_registry_references(self) -> SemanticCatalog:
        concept_ids = [item.concept_id for item in self.concepts]
        capability_ids = [item.capability_id for item in self.capabilities]
        route_ids = [item.route_id for item in self.routes]
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("concept_id values must be unique")
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capability_id values must be unique")
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route_id values must be unique")
        if len(self.intent_ids) != len(set(self.intent_ids)):
            raise ValueError("intent_id values must be unique")
        requirement_ids = [item.intent_id for item in self.primary_intent_requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("primary intent requirements must be unique")
        if set(requirement_ids) != set(self.intent_ids):
            raise ValueError("every registered intent_id must have one operation consistency rule")

        concept_id_set = set(concept_ids)
        concept_by_id = {item.concept_id: item for item in self.concepts}
        capability_id_set = set(capability_ids)
        route_by_id = {item.route_id: item for item in self.routes}
        operation_kinds = [item.kind for item in self.operation_arguments]
        if set(operation_kinds) != set(_ALL_OPERATION_KINDS) or len(operation_kinds) != len(
            _ALL_OPERATION_KINDS
        ):
            raise ValueError("operation argument schema must define every OperationKind exactly once")
        paths_by_kind = {item.kind: set(item.allowed_paths) for item in self.operation_arguments}
        if any(len(paths_by_kind[item.kind]) != len(item.allowed_paths) for item in self.operation_arguments):
            raise ValueError("operation argument paths must be unique per OperationKind")
        for operation_spec in self.operation_arguments:
            for path, domain in operation_spec.path_concept_domains.items():
                for concept_id in domain:
                    concept = concept_by_id.get(concept_id)
                    if concept is None:
                        raise ValueError(
                            f"operation {operation_spec.kind!r} path {path!r} references an unknown concept_id"
                        )
                    if concept.concept_type != "ANALYSIS_PARAMETER":
                        raise ValueError(
                            f"operation {operation_spec.kind!r} path {path!r} domains must reference ANALYSIS_PARAMETER concepts"
                        )

        for capability in self.capabilities:
            if capability.route_id not in route_by_id:
                raise ValueError(f"capability {capability.capability_id!r} references an unknown route_id")
            route = route_by_id[capability.route_id]
            if route.domain == "GENERAL_CONVERSATION":
                raise ValueError("business capabilities cannot belong to GENERAL_CONVERSATION")
            if capability.capability_id not in {item.capability_id for item in route.capability_ids}:
                raise ValueError("route/capability relationship must be declared on both records")
            method_concept_ids = set(capability.method_concept_ids)
            if len(method_concept_ids) != len(capability.method_concept_ids):
                raise ValueError(f"capability {capability.capability_id!r} method concept ids must be unique")
            if not method_concept_ids <= concept_id_set:
                raise ValueError(f"capability {capability.capability_id!r} references an unknown method concept_id")
            if any(concept_by_id[item].concept_type != "ANALYSIS_METHOD" for item in method_concept_ids):
                raise ValueError(f"capability {capability.capability_id!r} method ids must reference ANALYSIS_METHOD concepts")
            referenced_concepts = set(capability.concept_ids)
            referenced_concepts.update(method_concept_ids)
            for requirement in capability.operation_requirements:
                referenced_concepts.update(requirement.target_concept_ids)
            referenced_concepts.update(item.target_concept_id for item in capability.veto_requirements)
            if capability.basis_requirement is not None:
                referenced_concepts.update(capability.basis_requirement.concept_ids)
            for requirement in capability.operation_requirements:
                if not set(requirement.required_argument_paths) <= paths_by_kind[requirement.kind]:
                    raise ValueError(
                        f"capability {capability.capability_id!r} requires an illegal argument path"
                    )
            if not referenced_concepts <= concept_id_set:
                raise ValueError(f"capability {capability.capability_id!r} references an unknown concept_id")

        if any(
            not set(requirement.target_concept_ids) <= concept_id_set
            for requirement in self.primary_intent_requirements
        ):
            raise ValueError("primary intent requirements reference an unknown concept_id")

        for route in self.routes:
            route_capability_ids = {item.capability_id for item in route.capability_ids}
            unknown = route_capability_ids - capability_id_set
            if unknown:
                raise ValueError(f"route {route.route_id!r} references unknown capability_id values")
            for capability_id in route_capability_ids:
                capability = next(item for item in self.capabilities if item.capability_id == capability_id)
                if capability.route_id != route.route_id:
                    raise ValueError("route/capability relationship must point to the same route_id")
        return self

    def concept(self, concept_id: str) -> ConceptSpec | None:
        return next((item for item in self.concepts if item.concept_id == concept_id), None)

    def operation_paths(self, kind: OperationKind) -> tuple[ArgumentPath, ...]:
        spec = next((item for item in self.operation_arguments if item.kind == kind), None)
        return spec.allowed_paths if spec is not None else ()

    def operation_path_concept_domain(
        self, kind: OperationKind, path: ArgumentPath
    ) -> tuple[str, ...] | None:
        spec = next((item for item in self.operation_arguments if item.kind == kind), None)
        return spec.path_concept_domains.get(path) if spec is not None else None

    def primary_intent_requirement(self, intent_id: str) -> PrimaryIntentRequirement | None:
        return next(
            (item for item in self.primary_intent_requirements if item.intent_id == intent_id),
            None,
        )

    def snapshot_hash(self) -> str:
        serialized = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


_ALL_OPERATION_KINDS: tuple[OperationKind, ...] = (
    "SEGMENT",
    "ANALYZE",
    "FILTER",
    "SORT",
    "AGGREGATE",
    "QUERY",
    "EXPLAIN",
    "EXPLAIN_CAUSE",
    "COMPARE",
    "TREND",
    "CONTRIBUTION",
    "BREAKDOWN",
    "ENUMERATE",
)

_OPERATION_ARGUMENTS = (
    OperationArgumentSpec(kind="SEGMENT", allowed_paths=("basis", "method")),
    OperationArgumentSpec(kind="ANALYZE", allowed_paths=("topic", "metric", "dimension")),
    OperationArgumentSpec(kind="FILTER", allowed_paths=("field", "value")),
    OperationArgumentSpec(
        kind="SORT",
        allowed_paths=("key", "direction"),
        path_concept_domains={
            "direction": ("SORT_ASCENDING", "SORT_DESCENDING"),
        },
    ),
    OperationArgumentSpec(kind="AGGREGATE", allowed_paths=("metric", "group_by")),
    OperationArgumentSpec(kind="QUERY", allowed_paths=("metric", "group_by")),
    OperationArgumentSpec(kind="EXPLAIN", allowed_paths=("topic", "metric")),
    OperationArgumentSpec(kind="EXPLAIN_CAUSE", allowed_paths=("outcome", "metric", "dimension")),
    OperationArgumentSpec(kind="COMPARE", allowed_paths=("metric", "comparison_target")),
    OperationArgumentSpec(kind="TREND", allowed_paths=("metric", "time_grain")),
    OperationArgumentSpec(kind="CONTRIBUTION", allowed_paths=("metric", "dimension")),
    OperationArgumentSpec(kind="BREAKDOWN", allowed_paths=("metric", "dimension")),
    OperationArgumentSpec(kind="ENUMERATE", allowed_paths=("entity", "filter")),
)


def build_default_semantic_catalog() -> SemanticCatalog:
    concepts = (
        ConceptSpec(concept_id="CUSTOMER", concept_type="BUSINESS_ENTITY", exact_aliases=("客户",)),
        ConceptSpec(
            concept_id="PURCHASE_FREQUENCY",
            concept_type="METRIC",
            exact_aliases=("购买频次",),
            synonyms=("复购频次",),
        ),
        ConceptSpec(concept_id="MONETARY", concept_type="METRIC", exact_aliases=("金额",)),
        ConceptSpec(concept_id="RECENCY", concept_type="METRIC", exact_aliases=("最近购买",)),
        ConceptSpec(
            concept_id="RFM",
            concept_type="ANALYSIS_METHOD",
            exact_aliases=("RFM",),
            synonyms=("RFM 分群", "RFM 客户分群"),
        ),
        ConceptSpec(
            concept_id="COHORT_RETENTION",
            concept_type="ANALYSIS_TARGET",
            exact_aliases=("Cohort 留存分析",),
            synonyms=("Cohort 留存",),
        ),
        ConceptSpec(
            concept_id="CUSTOMER_RETENTION",
            concept_type="ANALYSIS_TARGET",
            exact_aliases=("客户留存分析",),
            synonyms=("客户留存",),
        ),
        ConceptSpec(
            concept_id="GMV_GROWTH_DIAGNOSTICS",
            concept_type="ANALYSIS_TARGET",
            exact_aliases=("GMV 诊断", "GMV 异动诊断", "GMV 异动诊断包"),
        ),
        ConceptSpec(
            concept_id="LIFECYCLE",
            concept_type="ANALYSIS_TARGET",
            exact_aliases=("生命周期",),
        ),
        ConceptSpec(concept_id="OTHER", concept_type="ANALYSIS_TARGET", exact_aliases=("别的分析",)),
        ConceptSpec(concept_id="GMV", concept_type="METRIC", exact_aliases=("GMV", "成交总额")),
        ConceptSpec(concept_id="PRODUCT_CATEGORY", concept_type="DIMENSION", exact_aliases=("品类",)),
        ConceptSpec(concept_id="ORDER", concept_type="BUSINESS_ENTITY", exact_aliases=("订单",)),
        ConceptSpec(concept_id="TIME", concept_type="TIME_CONCEPT", exact_aliases=("时间",)),
        ConceptSpec(
            concept_id="SORT_ASCENDING",
            concept_type="ANALYSIS_PARAMETER",
            exact_aliases=("升序", "从低到高", "由低到高", "正序", "从小到大"),
        ),
        ConceptSpec(
            concept_id="SORT_DESCENDING",
            concept_type="ANALYSIS_PARAMETER",
            exact_aliases=("降序", "从高到低", "由高到低", "倒序", "从大到小"),
        ),
    )
    routes = (
        RouteSpec(
            route_id="customer_analysis",
            domain="BUSINESS_ANALYSIS",
            capability_ids=(
                CapabilityReference(capability_id="customer_rfm_segmentation"),
                CapabilityReference(capability_id="customer_cohort_retention"),
                CapabilityReference(capability_id="customer_lifecycle"),
                CapabilityReference(capability_id="customer_retention_cohort_choice"),
                CapabilityReference(capability_id="customer_retention_lifecycle_choice"),
            ),
            tool_ids=("complex_analysis",),
            creates_plan=True,
            database_access=True,
        ),
        RouteSpec(
            route_id="growth_analysis",
            domain="BUSINESS_ANALYSIS",
            capability_ids=(CapabilityReference(capability_id="gmv_growth_diagnostics"),),
            tool_ids=("complex_analysis",),
            creates_plan=True,
            database_access=True,
        ),
        RouteSpec(
            route_id="general_conversation",
            domain="GENERAL_CONVERSATION",
            capability_ids=(),
            tool_ids=(),
            creates_plan=False,
            database_access=False,
        ),
    )
    capabilities = (
        CapabilitySpec(
            capability_id="customer_rfm_segmentation",
            route_id="customer_analysis",
            template_id="rfm_segmentation",
            concept_ids=("CUSTOMER", "PURCHASE_FREQUENCY", "MONETARY", "RECENCY"),
            method_concept_ids=("RFM",),
            operation_requirements=(
                OperationRequirement(
                    kind="SEGMENT",
                    target_concept_ids=("CUSTOMER",),
                    required_argument_paths=("basis",),
                ),
            ),
            # Negative operations cannot carry arguments; represent a negative
            # RFM method request as a target RFM veto.
            veto_requirements=(
                VetoRequirement(kind="SEGMENT", target_concept_id="RFM", clause_scope="SAME_CLAUSE"),
                VetoRequirement(kind="SEGMENT", target_concept_id="RFM", clause_scope="REQUEST"),
            ),
            # The SEGMENT target and basis arguments already encode the
            # customer-by-basis relationship. CUSTOMER may be implicit, so
            # this capability must not require a mention-backed subject role.
            required_roles=("segment_basis",),
            required_relations=(),
            basis_requirement=BasisRequirement(
                concept_ids=("PURCHASE_FREQUENCY", "MONETARY", "RECENCY"),
                min_basis_matches=2,
                decision_status="FROZEN",
            ),
        ),
        CapabilitySpec(
            capability_id="customer_cohort_retention",
            route_id="customer_analysis",
            template_id="cohort_retention",
            concept_ids=("CUSTOMER", "COHORT_RETENTION", "TIME"),
            operation_requirements=(
                OperationRequirement(kind="TREND", target_concept_ids=("COHORT_RETENTION",)),
            ),
            veto_requirements=(
                VetoRequirement(
                    kind="TREND", target_concept_id="COHORT_RETENTION", clause_scope="SAME_CLAUSE"
                ),
                VetoRequirement(
                    kind="TREND", target_concept_id="COHORT_RETENTION", clause_scope="REQUEST"
                ),
            ),
        ),
        CapabilitySpec(
            capability_id="customer_lifecycle",
            route_id="customer_analysis",
            template_id="customer_lifecycle",
            concept_ids=("CUSTOMER", "LIFECYCLE"),
            operation_requirements=(
                OperationRequirement(kind="ANALYZE", target_concept_ids=("LIFECYCLE",)),
            ),
            veto_requirements=(
                VetoRequirement(kind="ANALYZE", target_concept_id="LIFECYCLE", clause_scope="SAME_CLAUSE"),
                VetoRequirement(kind="ANALYZE", target_concept_id="LIFECYCLE", clause_scope="REQUEST"),
            ),
        ),
        # A generic retention request intentionally matches both registered
        # implementations. The router must ask the user to choose; it must not
        # infer a Cohort or lifecycle method from the broad wording.
        CapabilitySpec(
            capability_id="customer_retention_cohort_choice",
            route_id="customer_analysis",
            template_id="cohort_retention",
            concept_ids=("CUSTOMER_RETENTION",),
            operation_requirements=(
                OperationRequirement(kind="ANALYZE", target_concept_ids=("CUSTOMER_RETENTION",)),
            ),
        ),
        CapabilitySpec(
            capability_id="customer_retention_lifecycle_choice",
            route_id="customer_analysis",
            template_id="customer_lifecycle",
            concept_ids=("CUSTOMER_RETENTION",),
            operation_requirements=(
                OperationRequirement(kind="ANALYZE", target_concept_ids=("CUSTOMER_RETENTION",)),
            ),
        ),
        CapabilitySpec(
            capability_id="gmv_growth_diagnostics",
            route_id="growth_analysis",
            template_id="growth_diagnostics",
            concept_ids=("GMV_GROWTH_DIAGNOSTICS", "GMV"),
            operation_requirements=(
                OperationRequirement(
                    kind="ANALYZE",
                    target_concept_ids=("GMV_GROWTH_DIAGNOSTICS",),
                ),
            ),
        ),
    )
    return SemanticCatalog(
        version="semantic-catalog-v2.2-phase0.3",
        concepts=concepts,
        intent_ids=(
            "CUSTOMER_SEGMENTATION", "COHORT_RETENTION", "CUSTOMER_LIFECYCLE",
            "CUSTOMER_RETENTION", "GMV_GROWTH_DIAGNOSTICS",
        ),
        primary_intent_requirements=(
            PrimaryIntentRequirement(
                intent_id="CUSTOMER_SEGMENTATION",
                operation_kinds=("SEGMENT",),
                target_concept_ids=("CUSTOMER",),
            ),
            PrimaryIntentRequirement(
                intent_id="COHORT_RETENTION",
                operation_kinds=("TREND",),
                target_concept_ids=("COHORT_RETENTION",),
            ),
            PrimaryIntentRequirement(
                intent_id="CUSTOMER_LIFECYCLE",
                operation_kinds=("ANALYZE",),
                target_concept_ids=("LIFECYCLE",),
            ),
            PrimaryIntentRequirement(
                intent_id="CUSTOMER_RETENTION",
                operation_kinds=("ANALYZE",),
                target_concept_ids=("CUSTOMER_RETENTION",),
            ),
            PrimaryIntentRequirement(
                intent_id="GMV_GROWTH_DIAGNOSTICS",
                operation_kinds=("ANALYZE",),
                target_concept_ids=("GMV_GROWTH_DIAGNOSTICS",),
            ),
        ),
        operation_arguments=_OPERATION_ARGUMENTS,
        capabilities=capabilities,
        routes=routes,
        role_names=(
            "analysis_subject",
            "segment_basis",
            "filter_field",
            "comparison_target",
            "reference",
            "time_axis",
            "outcome",
        ),
        relation_predicates=(
            "GROUP_BY",
            "FILTER_BY",
            "COMPARE_WITH",
            "AGGREGATE_BY",
            "REFERS_TO",
            "INHERITS_FROM",
            "CAUSE_OF",
            "BREAKS_DOWN_BY",
        ),
    )


DEFAULT_SEMANTIC_CATALOG = build_default_semantic_catalog()
