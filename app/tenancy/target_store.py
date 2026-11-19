"""RLS-backed metric targets for seller tenants."""
from datetime import date
from sqlalchemy import text

from app.analysis.contracts import PeriodSpec
from app.analysis.targets import MetricTarget
from app.analysis.semantic import get_metric_definition
from app.data.database import engine
from app.tenancy.context import TenantContext


class TenantTargetStore:
    def __init__(self, context: TenantContext): self.context = context
    def _scope(self, conn):
        conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": self.context.tenant_id})
        conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": self.context.scope_type})
    def put(self, target: MetricTarget) -> MetricTarget:
        if target.scope != "tenant": raise ValueError("seller targets must use tenant scope")
        if target.period_end < target.period_start: raise ValueError("target period_end must be on or after period_start")
        if not target.seller_id or target.seller_id not in self.context.allowed_seller_ids:
            raise ValueError("You can only manage targets for an authorized seller")
        get_metric_definition(target.metric_name, "seller")
        with engine.begin() as conn:
            self._scope(conn)
            conn.execute(text("""INSERT INTO analytics.tenant_metric_targets(tenant_id,seller_id,metric_name,period_start,period_end,target_value,version,effective_at,created_by)
              VALUES(:tenant,:seller,:metric,:start,:end,:value,:version,:effective,:actor) ON CONFLICT DO NOTHING"""), {"tenant": self.context.tenant_id,"seller":target.seller_id,"metric":target.metric_name,"start":target.period_start,"end":target.period_end,"value":target.target_value,"version":target.version,"effective":target.effective_at,"actor":target.created_by})
        return target
    def get(self, metric_name: str, period: PeriodSpec, seller_id: str | None = None) -> MetricTarget | None:
        seller_id = seller_id or self.context.seller_id
        if not seller_id or seller_id not in self.context.allowed_seller_ids:
            return None
        with engine.begin() as conn:
            self._scope(conn)
            row=conn.execute(text("""SELECT * FROM analytics.tenant_metric_targets WHERE seller_id=:seller AND metric_name=:metric AND period_start=:start AND period_end=:end AND effective_at<=:end ORDER BY effective_at DESC,version DESC LIMIT 1"""), {"seller":seller_id,"metric":metric_name,"start":period.start_date,"end":period.end_date}).mappings().first()
        return MetricTarget(metric_name=row["metric_name"],period_start=row["period_start"],period_end=row["period_end"],target_value=float(row["target_value"]),version=row["version"],effective_at=row["effective_at"],created_by=row["created_by"],scope="tenant",seller_id=row["seller_id"]) if row else None

    def list(self, *, seller_id: str | None = None, metric_name: str | None = None, version: str | None = None, period_start: date | None = None, period_end: date | None = None, offset: int = 0, limit: int = 20) -> tuple[list[MetricTarget], int]:
        if seller_id and seller_id not in self.context.allowed_seller_ids: raise ValueError("You can only view targets for an authorized seller")
        with engine.begin() as conn:
            self._scope(conn)
            filters = ["tenant_id=:tenant"]; params: dict[str, object] = {"tenant": self.context.tenant_id, "offset": offset, "limit": limit}
            if seller_id: filters.append("seller_id=:seller"); params["seller"] = seller_id
            if metric_name: filters.append("metric_name=:metric"); params["metric"] = metric_name
            if version: filters.append("version=:version"); params["version"] = version
            if period_start: filters.append("period_start>=:period_start"); params["period_start"] = period_start
            if period_end: filters.append("period_end<=:period_end"); params["period_end"] = period_end
            where = " AND ".join(filters)
            total = conn.execute(text(f"SELECT count(*) FROM analytics.tenant_metric_targets WHERE {where}"), params).scalar_one()
            rows = conn.execute(text(f"SELECT * FROM analytics.tenant_metric_targets WHERE {where} ORDER BY period_start DESC,effective_at DESC,version DESC LIMIT :limit OFFSET :offset"), params).mappings().all()
        return [MetricTarget(metric_name=r["metric_name"], period_start=r["period_start"], period_end=r["period_end"], target_value=float(r["target_value"]), version=r["version"], effective_at=r["effective_at"], created_by=r["created_by"], scope="tenant", seller_id=r["seller_id"]) for r in rows], int(total)

    def delete(self, *, seller_id: str, metric_name: str, period_start: date, period_end: date, version: str) -> bool:
        if seller_id not in self.context.allowed_seller_ids: raise ValueError("You can only delete targets for an authorized seller")
        with engine.begin() as conn:
            self._scope(conn)
            result = conn.execute(text("DELETE FROM analytics.tenant_metric_targets WHERE tenant_id=:tenant AND seller_id=:seller AND metric_name=:metric AND period_start=:start AND period_end=:end AND version=:version"), {"tenant":self.context.tenant_id,"seller":seller_id,"metric":metric_name,"start":period_start,"end":period_end,"version":version})
        return result.rowcount > 0


class PlatformTargetStore:
    """平台管理员配置的店铺目标，与卖家租户共用同一张 RLS 目标表。

    写入、列表、删除必须落在同一存储，否则管理员写入的行在列表和删除里
    永远看不到。RLS 的 WITH CHECK 要求 tenant_id 等于会话变量，因此写入前
    先解析店铺归属的租户并逐租户落行。
    """

    def __init__(self, context: TenantContext): self.context = context

    def _scope(self, conn):
        conn.execute(text("SELECT set_config('app.scope_type', :scope, true)"), {"scope": "platform"})

    def put(self, target: MetricTarget) -> MetricTarget:
        if target.scope != "tenant": raise ValueError("seller targets must use tenant scope")
        if not target.seller_id: raise ValueError("seller_id is required for seller targets")
        if target.period_end < target.period_start: raise ValueError("target period_end must be on or after period_start")
        get_metric_definition(target.metric_name, "seller")
        with engine.begin() as conn:
            self._scope(conn)
            tenant_ids = [row["tenant_id"] for row in conn.execute(
                text("SELECT DISTINCT tenant_id FROM iam.tenant_seller_accounts WHERE seller_id=:seller ORDER BY tenant_id"),
                {"seller": target.seller_id}).mappings()]
            if not tenant_ids:
                raise ValueError("Seller is not mapped to any tenant")
            for tenant_id in tenant_ids:
                conn.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_id})
                conn.execute(text("""INSERT INTO analytics.tenant_metric_targets(tenant_id,seller_id,metric_name,period_start,period_end,target_value,version,effective_at,created_by)
                  VALUES(:tenant,:seller,:metric,:start,:end,:value,:version,:effective,:actor) ON CONFLICT DO NOTHING"""), {"tenant":tenant_id,"seller":target.seller_id,"metric":target.metric_name,"start":target.period_start,"end":target.period_end,"value":target.target_value,"version":target.version,"effective":target.effective_at,"actor":target.created_by})
        return target

    def get(self, metric_name: str, period: PeriodSpec, seller_id: str | None = None) -> MetricTarget | None:
        if not seller_id: raise ValueError("seller_id is required to read a seller target")
        with engine.begin() as conn:
            self._scope(conn)
            row=conn.execute(text("""SELECT * FROM analytics.tenant_metric_targets WHERE seller_id=:seller AND metric_name=:metric AND period_start=:start AND period_end=:end AND effective_at<=:end ORDER BY effective_at DESC,version DESC LIMIT 1"""), {"seller":seller_id,"metric":metric_name,"start":period.start_date,"end":period.end_date}).mappings().first()
        return MetricTarget(metric_name=row["metric_name"],period_start=row["period_start"],period_end=row["period_end"],target_value=float(row["target_value"]),version=row["version"],effective_at=row["effective_at"],created_by=row["created_by"],scope="tenant",seller_id=row["seller_id"]) if row else None
