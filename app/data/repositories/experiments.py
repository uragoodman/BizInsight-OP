"""Fail-closed access to registered, verified randomized experiments.

Experiment observations are executable causal evidence only when a trusted
governance record is closed, randomized, and independently verified. This
repository deliberately has no public write path for that verification.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import text

from app.analysis.causal_estimation import (
    ExperimentObservation,
    VerifiedRandomizedExperiment,
    estimate_randomized_itt,
)
from app.data.database import engine
from app.data.rls import set_local_context
from app.schemas.explorer import ControlledAnalysisScope
from app.tenancy import current_tenant_context


_SOURCE = "governance.experiment_registry+analytics.experiment_observations"
_MAX_EXPERIMENT_UNITS = 100_000


def _authorized_experiment_context(scope: ControlledAnalysisScope):
    tenant = current_tenant_context()
    if tenant is None:
        raise PermissionError("Trusted tenant context is required for experiment analysis")

    from app.auth.context import AuthorizationDenied, build_authorization_context

    authz = build_authorization_context(
        tenant,
        allowed_metric_ids=scope.allowed_metrics,
        metric_scope_type="seller",
    )
    try:
        authz.require_sellers([scope.seller_id])
        authz.require_date_range(
            scope.seller_id, scope.default_start_date, scope.default_end_date,
        )
        authz.require_tool("estimate_scoped_randomized_experiment")
    except AuthorizationDenied as exc:
        raise PermissionError(str(exc)) from exc
    return tenant, authz


def list_scoped_randomized_experiments(
    scope: ControlledAnalysisScope,
    *,
    limit: int = 50,
) -> dict[str, object]:
    """List only independently verified experiments visible in this scope."""
    if not 1 <= limit <= 50:
        raise ValueError("Experiment list limit must be between 1 and 50")
    tenant, authz = _authorized_experiment_context(scope)
    statement = text("""
        SELECT r.experiment_id, r.seller_id, r.outcome_metric_id,
               r.assignment_unit, r.starts_at, r.ends_at,
               COUNT(o.randomization_unit_hash)::int AS observation_count
        FROM governance.experiment_registry r
        JOIN analytics.experiment_observations o
          ON o.tenant_id = r.tenant_id AND o.experiment_id = r.experiment_id
         AND o.seller_id = r.seller_id
         AND o.outcome_metric_id = r.outcome_metric_id
        WHERE r.tenant_id = :tenant_id
          AND r.seller_id = :seller_id
          AND r.status = 'closed'
          AND r.assignment_method = 'randomized'
          AND r.randomization_verified IS TRUE
          AND r.verification_ref IS NOT NULL
          AND r.verified_by IS NOT NULL
          AND r.verified_at IS NOT NULL
          AND r.outcome_metric_id = ANY(:allowed_metric_ids)
          AND r.starts_at::date >= :scope_start
          AND r.ends_at::date <= :scope_end
          AND o.observed_at >= r.starts_at
          AND o.observed_at <= r.ends_at
        GROUP BY r.experiment_id, r.seller_id, r.outcome_metric_id,
                 r.assignment_unit, r.starts_at, r.ends_at
        ORDER BY r.starts_at DESC, r.experiment_id
        LIMIT :row_limit
    """)
    with engine.begin() as connection:
        set_local_context(
            connection, tenant, scope_type="seller",
            seller_access={scope.seller_id: (scope.default_start_date, scope.default_end_date)},
        )
        rows = connection.execute(statement, {
            "tenant_id": tenant.tenant_id,
            "seller_id": scope.seller_id,
            "allowed_metric_ids": list(authz.allowed_metric_ids),
            "scope_start": scope.default_start_date,
            "scope_end": scope.default_end_date,
            "row_limit": limit + 1,
        }).mappings().all()

    from app.auth.context import AuthorizationDenied
    from app.metrics.registry import get_registry

    items = []
    registry = get_registry()
    for row in rows[:limit]:
        metric_id = str(row["outcome_metric_id"])
        try:
            authz.require_metrics([metric_id])
            if registry.resolve(metric_id) != metric_id:
                continue
            definition = registry.get(metric_id)
        except (AuthorizationDenied, KeyError, ValueError):
            # Do not disclose the existence of experiments on hidden metrics.
            continue
        items.append({
            "experiment_id": str(row["experiment_id"]),
            "outcome_metric_id": metric_id,
            "outcome_metric_name": getattr(definition, "display_name", metric_id),
            "assignment_unit": str(row["assignment_unit"]),
            "starts_at": row["starts_at"].isoformat(),
            "ends_at": row["ends_at"].isoformat(),
            "observation_count": int(row["observation_count"]),
        })
    return {
        "items": items,
        "truncated": len(rows) > limit,
        "scope": {
            "seller_id": scope.seller_id,
            "start_date": scope.default_start_date.isoformat(),
            "end_date": scope.default_end_date.isoformat(),
        },
    }


def estimate_scoped_randomized_experiment(
    scope: ControlledAnalysisScope,
    *,
    experiment_id: str,
    minimum_per_arm: int = 30,
) -> dict[str, object]:
    """Estimate ITT for one server-registered experiment inside the locked scope.

    The caller supplies only an opaque experiment handle. Tenant, actor, seller,
    metric authorization and scope dates are re-derived/revalidated here. Raw
    pseudonymous unit keys are consumed in memory and never returned.
    """
    if not experiment_id or len(experiment_id) > 128:
        raise ValueError("A valid registered experiment ID is required")
    tenant, authz = _authorized_experiment_context(scope)
    from app.auth.context import AuthorizationDenied
    from app.metrics.registry import get_registry

    statement = text("""
        SELECT r.experiment_id, r.seller_id, r.outcome_metric_id,
               r.assignment_unit, r.assignment_method, r.status,
               r.starts_at, r.ends_at, r.randomization_verified,
               r.verification_ref, r.verified_by, r.verified_at,
               o.randomization_unit_hash, o.assigned_group, o.outcome_value
        FROM governance.experiment_registry r
        JOIN analytics.experiment_observations o
          ON o.tenant_id = r.tenant_id AND o.experiment_id = r.experiment_id
         AND o.seller_id = r.seller_id
         AND o.outcome_metric_id = r.outcome_metric_id
        WHERE r.tenant_id = :tenant_id
          AND r.seller_id = :seller_id
          AND r.experiment_id = :experiment_id
          AND r.status = 'closed'
          AND r.assignment_method = 'randomized'
          AND r.randomization_verified IS TRUE
          AND r.verification_ref IS NOT NULL
          AND r.verified_by IS NOT NULL
          AND r.verified_at IS NOT NULL
          AND r.starts_at::date >= :scope_start
          AND r.ends_at::date <= :scope_end
          AND o.observed_at >= r.starts_at
          AND o.observed_at <= r.ends_at
        ORDER BY o.randomization_unit_hash
        LIMIT :row_limit
    """)
    with engine.begin() as connection:
        set_local_context(
            connection, tenant, scope_type="seller",
            seller_access={scope.seller_id: (scope.default_start_date, scope.default_end_date)},
        )
        rows = connection.execute(statement, {
            "tenant_id": tenant.tenant_id,
            "seller_id": scope.seller_id,
            "experiment_id": experiment_id,
            "scope_start": scope.default_start_date,
            "scope_end": scope.default_end_date,
            "row_limit": _MAX_EXPERIMENT_UNITS + 1,
        }).mappings().all()

    if len(rows) > _MAX_EXPERIMENT_UNITS:
        raise ValueError("Experiment exceeds the bounded unit limit")
    if not rows:
        # Do not reveal whether the ID was absent, cross-seller, unverified,
        # outside scope, or simply had no observations.
        return {
            "status": "not_available",
            "reason": "No closed, verified randomized experiment is available within the authorized seller and date scope.",
            "causal_effect_estimated": False,
            "source_table": _SOURCE,
        }

    first = rows[0]
    metric_id = str(first["outcome_metric_id"])
    try:
        authz.require_metrics([metric_id])
        canonical_metric = get_registry().resolve(metric_id)
        metric_definition = get_registry().get(canonical_metric)
    except (AuthorizationDenied, KeyError, ValueError) as exc:
        raise PermissionError("Experiment outcome metric is not authorized or registered") from exc
    if canonical_metric != metric_id:
        raise ValueError("Experiment outcome metric must use its canonical registry ID")

    identity = (
        str(first["experiment_id"]), str(first["seller_id"]), metric_id,
        str(first["assignment_unit"]), str(first["assignment_method"]),
        first["starts_at"], first["ends_at"], bool(first["randomization_verified"]),
        first["verification_ref"], first["verified_by"], first["verified_at"],
    )
    for row in rows[1:]:
        candidate = (
            str(row["experiment_id"]), str(row["seller_id"]), str(row["outcome_metric_id"]),
            str(row["assignment_unit"]), str(row["assignment_method"]),
            row["starts_at"], row["ends_at"], bool(row["randomization_verified"]),
            row["verification_ref"], row["verified_by"], row["verified_at"],
        )
        if candidate != identity:
            raise ValueError("Experiment rows do not share one verified registry definition")

    experiment = VerifiedRandomizedExperiment(
        experiment_id=experiment_id,
        metric_id=canonical_metric,
        outcome_unit=str(getattr(metric_definition, "unit", "registered_metric_unit")),
        assignment_unit=str(first["assignment_unit"]),
        assignment_method="randomized",
        randomization_verified=True,
        assignment_source=_SOURCE,
        observations=tuple(
            ExperimentObservation(
                unit_id=str(row["randomization_unit_hash"]),
                assigned_group=str(row["assigned_group"]),
                outcome=float(row["outcome_value"]),
            )
            for row in rows
        ),
    )
    result = estimate_randomized_itt(
        experiment, minimum_per_arm=minimum_per_arm,
    )
    result["source_table"] = _SOURCE
    result["seller_id"] = scope.seller_id
    result["analysis_scope"] = {
        "start_date": scope.default_start_date.isoformat(),
        "end_date": scope.default_end_date.isoformat(),
    }
    result["unit_identifiers_returned"] = False
    return result
