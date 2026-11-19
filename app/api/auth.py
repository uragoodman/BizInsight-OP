"""租户认证边界：JWT 解析、API Key 验证、角色权限校验。

本模块实现 Phase 7 三层认证模型，每个接口按需组合依赖注入：

认证函数（FastAPI Depends，按优先级组合使用）：
1. require_tenant_context()         — 必选。解析 Bearer JWT 或 x-api-key 返回 TenantContext。
                                     开发环境无凭据时自动降级为平台管理员。
2. require_analysis_write()         — 可选，加在写接口上。验证 x-analysis-token 并返回 actor_id。
3. require_actor_match()            — 可选，加在审核接口上。确保认证身份与 payload 声明的身份一致。

执行流程：
    请求入站
      ↓
    require_tenant_context()     ← 解析身份（JWT / API Key / 降级）
      ↓
    require_analysis_write()     ← 验证写权限（仅 report review/publish 等接口）
      ↓
    require_actor_match()        ← 校验审核人一致性（仅 human_review 接口）
      ↓
    路由处理
"""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from hmac import compare_digest

from fastapi import Header, HTTPException, Request
from sqlalchemy import text

from app.core.config import get_settings
from app.data.database import engine
from app.tenancy import PLATFORM_SCOPE, TenantContext
from app.iam.service import authenticate_api_credential, authenticate_session


def _decode_base64url(value: str) -> bytes:
    """Base64 URL-safe 解码（自动补全 padding）。"""
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _decode_hs256_jwt(token: str, secret: str) -> dict[str, object]:
    """解码并验证 HS256 JWT，返回 payload。

    安全措施：
    - 固定 alg 为 HS256：忽略 token 里的 alg 声明，防止 alg=none 或 RS→HS 降级攻击
    - HMAC 恒定时间比较：用 compare_digest 避免 timing side-channel
    - 过期检查：对比 exp claim 与当前 UTC 时间

    参数：
    - token：紧凑格式 JWT（header.payload.signature）
    - secret：签名密钥

    返回：
    - dict：JWT payload（tenant_id, sub, scope_type, role, …）

    抛出：
    - HTTPException(401)：格式错误、签名不匹配、已过期
    """
    try:
        header_part, payload_part, signature_part = token.split(".")
        header = json.loads(_decode_base64url(header_part))
        payload = json.loads(_decode_base64url(payload_part))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Malformed bearer token") from exc
    # 安全要求：拒绝非 HS256 的 token（如 alg:none 或 alg:RS256）
    if header != {"alg": "HS256", "typ": "JWT"}:
        raise HTTPException(status_code=401, detail="Unsupported bearer token")
    # 用服务端密钥重新计算 signature，与 token 携带的 signature 对比
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{header_part}.{payload_part}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        supplied = _decode_base64url(signature_part)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Malformed bearer token") from exc
    # 恒定时间比较：防止攻击者通过响应耗时差异猜测签名
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    # 过期检查：拒绝 exp（Unix 时间戳）已小于当前时间的 token
    if not isinstance(payload, dict) or int(payload.get("exp", 0)) < int(
        datetime.now(timezone.utc).timestamp()
    ):
        raise HTTPException(status_code=401, detail="Expired bearer token")
    return payload


def _context_from_claims(claims: dict[str, object]) -> TenantContext:
    """从 JWT claims 字典构造 TenantContext 实例。

    参数：
    - claims：JWT payload，必须包含 tenant_id, sub, scope_type, role

    返回：
    - TenantContext：冻结的租户上下文

    抛出：
    - HTTPException(401)：缺少必要字段或字段值非法
    """
    # 密码登录签发的 JWT 只保存 session 指针，不把完整权限永久写死在 token 中。
    # 每次请求都根据 sid 重新加载成员关系和授权信息，因此撤销成员资格或权限后
    # 可以立即生效，不必等到 JWT 自然过期。
    if isinstance(claims.get("sid"), str):
        context = authenticate_session(
            tenant_id=str(claims.get("tenant_id", "")),
            user_id=str(claims.get("sub", "")),
            session_id=str(claims["sid"]),
        )
        if context is None:
            raise HTTPException(status_code=401, detail="Expired or revoked session")
        return context
    try:
        scope = str(claims["scope_type"])
        role = str(claims["role"])
        return TenantContext(
            tenant_id=str(claims["tenant_id"]),
            user_id=str(claims["sub"]),
            role=role,
            scope_type=scope,
            seller_id=(
                str(claims["seller_id"]) if claims.get("seller_id") else None
            ),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid tenant claims") from exc


def _audit_denial(request: Request, reason: str) -> None:
    """在 governance.audit_events 表中记录认证拒绝事件。

    仅用于审计溯源，写入失败不能影响主流程（401 该返回还是得返回），
    所以用 try/except 静默吞掉所有异常。——best-effort"""

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO governance.audit_events(event_type, outcome, detail, request_path)
                    VALUES ('tenant_auth', 'denied', :detail, :path)"""
                ),
                {"detail": reason, "path": request.url.path},
            )
    except Exception:
        pass


def require_tenant_context(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_analysis_token: str | None = Header(default=None),
    x_actor_id: str | None = Header(default=None),
) -> TenantContext:
    """从请求头解析租户身份，按优先级依次尝试 4 种认证方式。

    认证链（优先级从高到低）：
    1. Bearer JWT（HS256）：调用 _decode_hs256_jwt 验证签名+过期 → 从 claims 构造 TenantContext
    2. x-api-key：对 key 做 SHA256 → 查 iam.api_credentials 表 → 联表拿到角色/scope/seller 信息
    3. x-analysis-token（Phase 6 遗留）：仅限非生产且未启用租户认证时使用
    4. 开发降级（TENANT_AUTH_REQUIRED=false）：直接返回平台管理员上下文，用于本地开发

    任一方式成功后立即返回，不再尝试后续方式。
    全部失败后写入审计日志并抛 401。

    返回：
    - TenantContext：包含 tenant_id、user_id、role、scope_type、seller_id

    抛出：
    - HTTPException(401)：所有认证方式均失败
    """
    settings = get_settings()
    try:
        # ── 认证方式 1：Bearer JWT（HS256）────────────────────────────
        # 从 authorization 头提取 Bearer token，用 _decode_hs256_jwt 验证
        # 签名完整性 + 过期时间，最后从 claims 构造 TenantContext。
        # 生产环境推荐方式，依赖 tenant_auth_secret 配置项。
        if authorization and authorization.startswith("Bearer ") and settings.tenant_auth_secret:
            return _context_from_claims(
                _decode_hs256_jwt(
                    authorization.removeprefix("Bearer ").strip(),
                    settings.tenant_auth_secret,
                )
            )
        # ── 认证方式 2：x-api-key（数据库 lookup）────────────────────
        # 对明文 key 做 SHA256 → 查 iam.api_credentials 表匹配 key_hash
        # → 使用凭证引用重新加载当前成员角色和全部有效 seller grants。
        # 只返回未吊销、未过期、租户状态为 active 的记录。
        if x_api_key:
            key_hash = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
            with engine.connect() as conn:
                row = (
                    conn.execute(
                        text(
                    """SELECT c.credential_id, c.tenant_id, c.user_id
                    FROM iam.api_credentials c
                    JOIN iam.tenants t ON t.tenant_id = c.tenant_id
                    JOIN iam.tenant_memberships m
                        ON m.tenant_id = c.tenant_id AND m.user_id = c.user_id
                    WHERE c.key_hash = :key_hash
                      AND c.revoked_at IS NULL
                      AND (c.expires_at IS NULL OR c.expires_at > now())
                      AND m.revoked_at IS NULL
                      AND t.status = 'active'"""
                        ),
                        {"key_hash": key_hash},
                    )
                    .mappings()
                    .first()
                )
            if row:
                context = authenticate_api_credential(
                    tenant_id=str(row["tenant_id"]),
                    user_id=str(row["user_id"]),
                    credential_id=str(row["credential_id"]),
                )
                if context is not None:
                    return context
        # ── 认证方式 3：x-analysis-token（Phase 6 遗留）─────────────
        # 恒定时间对比硬编码的 analysis_write_token，只返回平台管理员 scope。
        # 仅在非生产且 TENANT_AUTH_REQUIRED=false 时保留为开发期兼容入口；
        # 生产环境和启用租户认证的环境必须通过 JWT 或 API Key 建立身份。
        if (
            settings.environment.lower() not in {"production", "prod"}
            and not settings.tenant_auth_required
            and settings.analysis_write_token
            and x_analysis_token
            and compare_digest(x_analysis_token, settings.analysis_write_token)
            and x_actor_id
        ):
            return TenantContext(
                tenant_id="platform",
                user_id=x_actor_id,
                role="admin",
                scope_type=PLATFORM_SCOPE,
            )
        # ── 认证方式 4：开发环境降级 ────────────────────────────────
        # 当 TENANT_AUTH_REQUIRED=false 且非生产环境时，直接返回平台管理员上下文。
        # 这样本地开发/测试不需要真实凭据也能调试接口。（安全要求：生产环境必须关掉此降级）
        if (
            settings.environment.lower() not in {"production", "prod"}
            and not settings.tenant_auth_required
        ):
            return TenantContext(
                tenant_id="platform",
                user_id=settings.user_id,
                role="admin",
                scope_type=PLATFORM_SCOPE,
            )
    except HTTPException:
        _audit_denial(request, "invalid_credential")
        raise
    _audit_denial(request, "missing_or_invalid_credential")
    raise HTTPException(status_code=401, detail="Tenant authentication required")


def require_analysis_write(
    x_analysis_token: str | None = Header(default=None),
    x_actor_id: str | None = Header(default=None),
) -> str:
    """验证 x-analysis-token，确保调用方有报告写入权限。

    逻辑：
    - 未配置 analysis_write_token：
        → 生产环境抛 503（缺少关键配置，无法提供服务）
        → 非生产环境直接放行，返回 x_actor_id 或默认 user_id
    - 配置了 analysis_write_token：
        → 恒定时间对比 token，不匹配抛 403
        → 生产环境：必须同时提供 x_actor_id，否则抛 403
        → 非生产环境：x_actor_id 可选，缺省用 settings.user_id

    返回：
    - str：认证后的 actor ID

    抛出：
    - HTTPException(503)：生产环境未配置 write token（配置缺失）
    - HTTPException(403)：令牌不匹配或缺少 actor ID
    """
    settings = get_settings()
    if not settings.analysis_write_token:
        if settings.environment.lower() in {"production", "prod"}:
            raise HTTPException(
                status_code=503,
                detail="Analysis write authentication is not configured",
            )
        return x_actor_id or settings.user_id
    if x_analysis_token is None or not compare_digest(
        x_analysis_token, settings.analysis_write_token
    ):
        raise HTTPException(
            status_code=403, detail="Analysis write permission denied"
        )
    if (
        settings.environment.lower() in {"production", "prod"}
        and not x_actor_id
    ):
        raise HTTPException(
            status_code=403,
            detail="Authenticated actor identity is required",
        )
    return x_actor_id or settings.user_id


def require_actor_match(authenticated_actor: str, claimed_actor: str) -> None:
    """校验令牌认证的身份与请求 payload 声明的身份是否一致。

    用途：防止 A 用自己的 token 但冒充 B 提交审核（越权审批）。

    规则：
    - 生产环境：强制校验，authenticated_actor != claimed_actor 时抛 403
    - 非生产环境：跳过校验，方便 curl/Postman 调试

    参数：
    - authenticated_actor：从令牌解析的 actor 身份
    - claimed_actor：请求 payload 中的 actor

    抛出：
    - HTTPException(403)：身份不一致
    """
    if (
        get_settings().environment.lower() in {"production", "prod"}
        and authenticated_actor != claimed_actor
    ):
        raise HTTPException(
            status_code=403,
            detail="Request actor does not match authenticated identity",
        )
