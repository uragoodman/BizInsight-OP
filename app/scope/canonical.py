"""CanonicalScope and QueryScope. A scope can only become narrower."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal, Mapping

from app.auth.context import AuthorizationContext, AuthorizationDenied


class ScopeError(ValueError):
    """A malformed or unauthorized scope."""


def _sorted_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values}))


def _date_or_none(value: date | str | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _fingerprint(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalScope:
    tenant_id: str
    user_id: str
    seller_ids: tuple[str, ...]
    start_date: date
    end_date: date
    metric_ids: tuple[str, ...]
    dimensions: tuple[str, ...] = ()
    filters: tuple[tuple[str, str], ...] = ()
    timezone: str = "UTC"
    granularity: str = "day"
    order_status: str | None = None
    currency: str | None = None
    authorization_fingerprint: str = ""
    permission_version: str = ""
    policy_set_version: str = ""
    metric_definition_versions: tuple[tuple[str, str], ...] = ()
    actions: tuple[str, ...] = ()
    seller_scope_type: Literal["single", "selected", "all"] = "single"
    include_unattributed: bool = False

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.user_id:
            raise ScopeError("tenant_id and user_id are required")
        if self.end_date < self.start_date:
            raise ScopeError("end_date must not precede start_date")
        if self.seller_scope_type == "all":
            if self.seller_ids:
                raise ScopeError("ALL scope must use an explicit all marker, not seller_ids")
        elif not self.seller_ids:
            raise ScopeError("at least one seller is required")
        elif self.seller_scope_type == "single" and len(self.seller_ids) != 1:
            raise ScopeError("single seller scope requires exactly one seller")
        elif self.seller_scope_type == "selected" and len(self.seller_ids) < 1:
            raise ScopeError("selected seller scope requires at least one seller")
        if self.include_unattributed and self.seller_scope_type != "all":
            raise ScopeError("include_unattributed is only valid for ALL scope")
        if not self.metric_ids:
            raise ScopeError("at least one metric is required")

    @classmethod
    def resolve(
        cls,
        authz: AuthorizationContext,
        *,
        seller_ids: Iterable[str] = (),
        seller_scope_type: Literal["single", "selected", "all"] | None = None,
        include_unattributed: bool = False,
        start_date: date | str,
        end_date: date | str,
        metric_ids: Iterable[str],
        dimensions: Iterable[str] = (),
        filters: Mapping[str, object] | None = None,
        timezone: str = "UTC",
        granularity: str = "day",
        order_status: str | None = None,
        currency: str | None = None,
        actions: Iterable[str] = ("analyze",),
        metric_definition_versions: Mapping[str, str] | None = None,
    ) -> "CanonicalScope":
        sellers = _sorted_strings(seller_ids)
        resolved_seller_scope_type = seller_scope_type or ("single" if len(sellers) == 1 else "selected")
        if resolved_seller_scope_type == "all" and not (
            authz.scope_type == "platform" and authz.role == "admin"
        ):
            raise ScopeError("ALL seller scope requires a platform administrator")
        if resolved_seller_scope_type == "selected" and not (
            authz.scope_type == "platform" and authz.role == "admin"
        ):
            raise ScopeError("Selected seller scope requires a platform administrator")
        if include_unattributed and not (
            resolved_seller_scope_type == "all"
            and authz.scope_type == "platform" and authz.role == "admin"
        ):
            raise ScopeError("include_unattributed requires ALL platform administrator scope")
        metrics = _sorted_strings(metric_ids)
        dims = _sorted_strings(dimensions)
        start = _date_or_none(start_date)
        end = _date_or_none(end_date)
        if start is None or end is None:
            raise ScopeError("start_date and end_date are required")

        try:
            if resolved_seller_scope_type != "all":
                authz.require_sellers(sellers)
            authz.require_metrics(metrics)
            authz.require_dimensions(dims)
            requested_actions = _sorted_strings(actions)
            authz.require_action("read")
            for action in requested_actions:
                authz.require_action(action)
            for seller in sellers:
                authz.require_date_range(seller, start, end)
        except AuthorizationDenied as exc:
            raise ScopeError(str(exc)) from exc

        normalized_filters = tuple(sorted(
            (str(key), json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
            for key, value in (filters or {}).items()
        ))
        if metric_definition_versions is None:
            # Definition versions are part of the server-owned scope, not a
            # client-supplied cache hint.  Resolve them from the Registry so
            # a catalog change invalidates a previously authorized scope.
            from app.metrics.registry import get_registry

            metric_definition_versions = {
                metric_id: get_registry().get(metric_id).version
                for metric_id in metrics
            }
        return cls(
            tenant_id=authz.tenant_id,
            user_id=authz.user_id,
            seller_ids=sellers,
            start_date=start,
            end_date=end,
            metric_ids=metrics,
            dimensions=dims,
            filters=normalized_filters,
            timezone=timezone,
            granularity=granularity,
            order_status=order_status,
            currency=currency,
            authorization_fingerprint=authz.authz_digest,
            permission_version=authz.permission_version,
            policy_set_version=authz.policy_set_version,
            metric_definition_versions=tuple(sorted(
                (str(metric_id), str(version))
                for metric_id, version in (metric_definition_versions or {}).items()
            )),
            actions=requested_actions,
            seller_scope_type=resolved_seller_scope_type,
            include_unattributed=include_unattributed,
        )

    @property
    def query_scope_fingerprint(self) -> str:
        return _fingerprint(self.as_query_dict())

    def as_query_dict(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "seller_ids": self.seller_ids,
            "seller_scope_type": self.seller_scope_type,
            "include_unattributed": self.include_unattributed,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "metric_ids": self.metric_ids,
            "dimensions": self.dimensions,
            "filters": self.filters,
            "timezone": self.timezone,
            "granularity": self.granularity,
            "order_status": self.order_status,
            "currency": self.currency,
            "permission_version": self.permission_version,
            "policy_set_version": self.policy_set_version,
            "metric_definition_versions": self.metric_definition_versions,
            "actions": self.actions,
        }

    def narrow(
        self,
        *,
        seller_ids: Iterable[str] | None = None,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        metric_ids: Iterable[str] | None = None,
        dimensions: Iterable[str] | None = None,
        filters: Mapping[str, object] | None = None,
        timezone: str | None = None,
        granularity: str | None = None,
        actions: Iterable[str] | None = None,
        metric_definition_versions: Mapping[str, str] | None = None,
    ) -> "QueryScope":
        if self.seller_scope_type == "all" and seller_ids is not None:
            raise ScopeError("ALL scope cannot be narrowed without a finite authorized seller set")
        sellers = _sorted_strings(seller_ids if seller_ids is not None else self.seller_ids)
        if self.seller_scope_type in {"single", "selected"} and not sellers:
            raise ScopeError(f"{self.seller_scope_type} seller scope cannot be narrowed to an empty seller set")
        metrics = _sorted_strings(metric_ids if metric_ids is not None else self.metric_ids)
        dims = _sorted_strings(dimensions if dimensions is not None else self.dimensions)
        start = _date_or_none(start_date) or self.start_date
        end = _date_or_none(end_date) or self.end_date
        if not set(sellers).issubset(self.seller_ids):
            raise ScopeError("QueryScope cannot expand seller scope")
        if not set(metrics).issubset(self.metric_ids):
            raise ScopeError("QueryScope cannot expand metric scope")
        if not set(dims).issubset(self.dimensions):
            raise ScopeError("QueryScope cannot expand dimension scope")
        if start < self.start_date or end > self.end_date:
            raise ScopeError("QueryScope cannot expand date range")
        requested_actions = _sorted_strings(actions if actions is not None else self.actions)
        if not set(requested_actions).issubset(self.actions):
            raise ScopeError("QueryScope cannot expand actions")
        requested_versions = tuple(sorted(
            (str(metric_id), str(version))
            for metric_id, version in (metric_definition_versions or dict(self.metric_definition_versions)).items()
        ))
        if requested_versions != self.metric_definition_versions:
            raise ScopeError("QueryScope cannot change metric definition versions")
        normalized_filters = self.filters if filters is None else tuple(sorted(
            (str(key), json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
            for key, value in filters.items()
        ))
        return QueryScope(
            parent_fingerprint=self.query_scope_fingerprint,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            seller_ids=sellers,
            start_date=start,
            end_date=end,
            metric_ids=metrics,
            dimensions=dims,
            filters=normalized_filters,
            timezone=timezone or self.timezone,
            granularity=granularity or self.granularity,
            order_status=self.order_status,
            currency=self.currency,
            authorization_fingerprint=self.authorization_fingerprint,
            permission_version=self.permission_version,
            policy_set_version=self.policy_set_version,
            metric_definition_versions=self.metric_definition_versions,
            actions=requested_actions,
            seller_scope_type=self.seller_scope_type,
            include_unattributed=self.include_unattributed,
        )


@dataclass(frozen=True, slots=True)
class QueryScope:
    parent_fingerprint: str
    tenant_id: str
    user_id: str
    seller_ids: tuple[str, ...]
    start_date: date
    end_date: date
    metric_ids: tuple[str, ...]
    dimensions: tuple[str, ...]
    filters: tuple[tuple[str, str], ...]
    timezone: str
    granularity: str
    order_status: str | None
    currency: str | None
    authorization_fingerprint: str
    permission_version: str = ""
    policy_set_version: str = ""
    metric_definition_versions: tuple[tuple[str, str], ...] = ()
    actions: tuple[str, ...] = ()
    seller_scope_type: Literal["single", "selected", "all"] = "single"
    include_unattributed: bool = False

    def __post_init__(self) -> None:
        if self.seller_scope_type == "all":
            if self.seller_ids:
                raise ScopeError("ALL QueryScope must use an explicit all marker, not seller_ids")
        elif not self.seller_ids:
            raise ScopeError(f"{self.seller_scope_type} QueryScope requires seller_ids")
        elif self.seller_scope_type == "single" and len(self.seller_ids) != 1:
            raise ScopeError("single QueryScope requires exactly one seller")
        if self.include_unattributed and self.seller_scope_type != "all":
            raise ScopeError("include_unattributed is only valid for ALL QueryScope")

    @property
    def query_scope_fingerprint(self) -> str:
        return _fingerprint({
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "seller_ids": self.seller_ids,
            "seller_scope_type": self.seller_scope_type,
            "include_unattributed": self.include_unattributed,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "metric_ids": self.metric_ids,
            "dimensions": self.dimensions,
            "filters": self.filters,
            "timezone": self.timezone,
            "granularity": self.granularity,
            "order_status": self.order_status,
            "currency": self.currency,
            "permission_version": self.permission_version,
            "policy_set_version": self.policy_set_version,
            "metric_definition_versions": self.metric_definition_versions,
            "actions": self.actions,
        })

    def as_dict(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "seller_ids": list(self.seller_ids),
            "seller_scope_type": self.seller_scope_type,
            "include_unattributed": self.include_unattributed,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "metric_ids": list(self.metric_ids),
            "dimensions": list(self.dimensions),
            "filters": dict(self.filters),
            "timezone": self.timezone,
            "granularity": self.granularity,
            "order_status": self.order_status,
            "currency": self.currency,
            "authorization_fingerprint": self.authorization_fingerprint,
            "permission_version": self.permission_version,
            "policy_set_version": self.policy_set_version,
            "metric_definition_versions": dict(self.metric_definition_versions),
            "actions": list(self.actions),
            "query_scope_fingerprint": self.query_scope_fingerprint,
            "parent_fingerprint": self.parent_fingerprint,
        }
