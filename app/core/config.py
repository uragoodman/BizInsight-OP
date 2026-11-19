from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "BizInsight Agent"
    app_version: str = "0.1.0"
    # Keep the seller breakdown registered but closed until an operator opts in.
    seller_dimension_enabled: bool = False
    seller_multi_scope_enabled: bool = False
    seller_scope_max_selected_sellers: int = 100
    # Align seller scopes with the existing Explorer three-year range cap.
    seller_scope_max_date_span_days: int = 366 * 3
    # P3 verifier is observational only in this release. Raw user text may be
    # sent to the configured LLM provider only after an explicit opt-in.
    semantic_verifier_mode: Literal["off", "shadow", "open_only"] = "off"
    semantic_verifier_send_raw_question: bool = False
    semantic_verifier_model: str = ""
    semantic_verifier_timeout_seconds: float = Field(default=25.0, gt=0, le=60)
    # Empty keeps every registered template visible to chat entry points.
    complex_chat_visible_templates: list[str] = Field(default_factory=list)
    # Owner decision (2026-11-15): active semantic routing is the default for
    # every authenticated tenant; environment variables can still override it.
    complex_route_mode: Literal["off", "shadow", "active"] = "active"
    complex_route_canary_enabled: bool = True
    complex_route_canary_tenants: list[str] = Field(default_factory=lambda: ["*"])
    # Keep rollout bounded to plan templates declared by the semantic catalog.
    # New catalog templates require an explicit config update.
    complex_route_canary_templates: list[str] = Field(default_factory=lambda: [
        "rfm_segmentation", "cohort_retention", "customer_lifecycle", "growth_diagnostics",
    ])
    complex_route_canary_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    complex_route_canary_salt: str = ""
    # Active interpretation is enabled by default for authenticated requests.
    semantic_interpreter_mode: Literal["off", "shadow", "active"] = "active"
    # Owner-approved default: raw questions may be sent to the configured provider.
    semantic_interpreter_model: str = ""
    # Retained for older callers; interaction-policy calls use the preset's
    # connect/idle/total deadlines instead of this legacy total timeout.
    semantic_interpreter_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    semantic_interpreter_interaction_policy: Literal[
        "stream_fast", "stream_thinking", "nonstream_fast"
    ] = "stream_fast"
    semantic_interpreter_max_tokens: int = Field(default=2048, ge=256, le=8192)
    semantic_interpreter_validation_repair_enabled: bool = True
    semantic_interpreter_send_raw_question: bool = True
    semantic_interpreter_shadow_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Synchronous inline shadow can delay user requests; keep it offline-only
    # until a non-blocking execution path exists.
    semantic_interpreter_inline_shadow_enabled: bool = False
    # Repair deterministic spans and catalog grounding metadata before the
    # Validator sees a schema-valid Candidate; unresolved evidence still fails closed.
    semantic_interpreter_span_resolution_enabled: bool = True
    # Convert only uniquely grounded current-turn cross-clause negative targets
    # from SAME_CLAUSE to REQUEST after coordinate normalization, before Validator.
    semantic_interpreter_request_scope_normalization_enabled: bool = True
    # Raw model output is sensitive diagnostic material and stays disabled by default.
    semantic_interpreter_capture_raw_output: bool = False
    # Kept as inert compatibility configuration until the next stable release.
    # Phase 4 removes selector observations and routing; capability routing is deterministic.
    complex_selector_mode: Literal["off", "shadow", "active"] = "off"
    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    llm_provider: str = "opencode_go_openai_compatible"
    llm_model: str = "hy3"
    llm_input_cost_per_1k: float = 0.0
    llm_output_cost_per_1k: float = 0.0
    llm_fast_model: str = ""
    llm_standard_model: str = ""
    llm_quality_model: str = ""
    llm_fast_input_cost_per_1k: float = 0.0
    llm_fast_output_cost_per_1k: float = 0.0
    llm_quality_input_cost_per_1k: float = 0.0
    llm_quality_output_cost_per_1k: float = 0.0
    # Reasoning-capable providers may spend completion tokens before visible
    # JSON/text appears. Keep explicit per-call ceilings instead of relying on
    # provider defaults, which are often much smaller.
    llm_extract_max_tokens: int = 8192
    # Critic verdicts are compact JSON; keep their output budget independent
    # from general-purpose extraction calls.
    critic_max_output_tokens: int = 1024
    # Reserve room for tokenizer overhead before sending a model request. The
    # output cap is reserved in full so the request cannot exceed the turn cap.
    llm_budget_preflight_margin_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    llm_chat_max_tokens: int = 4096
    llm_price_version: str = "configured-v1"
    llm_user_daily_token_limit: int = 0
    llm_user_monthly_token_limit: int = 0
    llm_user_daily_cost_limit: float = 0.0
    llm_user_monthly_cost_limit: float = 0.0
    # 备用 LLM 提供商列表（JSON 数组）：主 provider 之外的可恢复故障回退端点。
    # 每项：{"id","base_url","api_key_env","default_model","models":{"fast":..,"standard":..,"quality":..}}
    llm_providers_json: str = "[]"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "bizinsight_knowledge"
    # Optional stable query/write alias. Keep the physical collection in
    # QDRANT_COLLECTION only when no versioned alias is configured.
    rag_qdrant_active_alias: str = ""
    metric_semantic_qdrant_collection: str = "metric_semantic_catalog_v1"
    metric_semantic_top_k: int = 5
    rag_knowledge_root: str = "docs/knowledge"
    rag_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rag_embedding_model_revision: str = ""
    rag_top_k: int = 5
    rag_backend: str = "local_hybrid"
    rag_cache_dir: str = "data/cache/llama_index"
    rag_offline_embeddings: bool = True
    # P3.3 cross-encoder reranking; keep the model revision pinned for
    # reproducible prefetches and offline deployments.
    rag_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rag_reranker_model_revision: str = "1427fd652930e4ba29e8149678df786c240d8825"
    rag_reranker_enabled: bool = True
    rag_reranker_version: str = "v1"
    rag_reranker_strategy: str = "cross_encoder"
    rag_reranker_top_k: int = 0
    rag_reranker_timeout_ms: int = 5000
    rag_index_version: str = "v1"
    rag_chunk_pipeline_version: str = "section-v1"
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 80
    rag_manifest_path: str = "data/rag/index_manifest.json"
    rag_release_manifest_path: str = "data/rag/release_manifest.json"
    rag_candidate_k: int = 12
    rag_vector_weight: float = 0.40
    rag_sparse_weight: float = 0.45
    rag_semantic_weight: float = 0.35
    rag_metadata_weight: float = 0.20
    rag_release_controller_enabled: bool = False
    rag_shadow_enabled: bool = False
    rag_shadow_sample_rate: float = 0.0
    # Semantic Compiler v2 shadow sampling. P0 keeps both switches off; P4
    # turns on a deterministic, salted sample of auto-executed turns.
    semantic_shadow_enabled: bool = False
    semantic_shadow_sample_rate: float = 0.0
    semantic_shadow_salt: str = ""
    rag_canary_enabled: bool = False
    rag_canary_percent: int = 0
    rag_ragas_enabled: bool = False
    rag_phoenix_enabled: bool = False
    rag_shadow_lease_seconds: int = 60
    # An empty endpoint means no remote provider: progressive delivery then
    # fails closed to stable retrieval.
    rag_unleash_url: str = ""
    rag_unleash_api_token: str = ""
    rag_unleash_app_name: str = "bizinsight"
    rag_unleash_refresh_interval_seconds: int = 15
    # Result safety is data-driven: a result is only suppressed when its
    # producer supplies an explicit sample size/classification. This keeps
    # legacy metric payloads compatible while allowing governed sources to
    # enable small-cell and field-level protections.
    result_policy_min_cell_size: int = 5
    result_policy_allowed_classifications_json: str = '["public","internal"]'
    result_policy_mask_fields_json: str = "{}"
    result_policy_export_approval_required: bool = True
    result_policy_max_derived_dimensions: int = 3
    # Deterministic privacy risk controls.  A producer can provide explicit
    # quasi-identifier metadata; the output boundary never guesses business
    # meaning from free-form prose.
    result_policy_reidentification_threshold: int = 3
    result_policy_reidentification_max_quasi_identifiers: int = 2
    result_policy_database_masking_enabled: bool = True
    chat_execution_stale_seconds: int = 600
    lifecycle_reconcile_db_password: str = ""
    postgres_url: str = "postgresql+psycopg://localhost:5432/bizinsight"
    trace_dir: str = "data/traces"
    trace_backend: Literal["file", "postgres"] = "file"
    checkpoint_path: str = "data/checkpoints/langgraph.sqlite"
    checkpoint_backend: Literal["sqlite", "postgres"] = "sqlite"
    phase6_store_path: str = "data/phase6.sqlite"
    quality_data_dir: str = "data/quality"
    report_chart_dir: str = "data/reports/charts"
    analysis_write_token: str = ""
    anomaly_cooldown_hours: int = 24
    # Server-enforced related-metric set for the required diagnostic skeleton
    # (per target metric).  Factor metrics from the metric registry are merged
    # in automatically; this map only adds cross-metric operating signals.
    diagnostic_related_metrics_json: str = '{"gmv": ["order_count", "avg_order_value", "delivery_delay_rate"]}'
    # AGENT_PLAN Phase 6: open business-diagnosis agent.  Auto-run is disabled
    # by default — without the explicit switch the agent refuses before any
    # scan or LLM call, and findings only ever reach the pending-review channel.
    open_diagnosis_enabled: bool = False
    open_diagnosis_max_problems: int = 3
    open_diagnosis_cooldown_hours: int = 24
    open_diagnosis_max_tool_calls_per_problem: int = 6
    open_diagnosis_metrics_whitelist_json: str = '["gmv", "order_count", "avg_order_value", "delivery_delay_rate"]'
    # AGENT_PLAN Phase 9: alert triage.  Detection stays deterministic; only
    # tenants listed here may have an anomaly escalated into a budgeted
    # diagnostic run, and results still go to pending review, never publish.
    alert_auto_diagnosis_enabled: bool = False
    alert_auto_diagnosis_tenants_json: str = "[]"
    alert_auto_diagnosis_metrics_json: str = '["gmv"]'
    alert_auto_diagnosis_cooldown_hours: int = 24
    # AGENT_PLAN Phase 7: conversation relation resolver.  The LLM decision is
    # advisory; the deterministic keyword heuristic remains the degradation path.
    relation_resolver_llm_enabled: bool = True
    # Cross-turn task snapshots are persisted separately from short session memory.
    # This kill switch disables snapshot loading/writes without affecting context handoff.
    task_continuity_enabled: bool = True
    # AGENT_PLAN Phase 8: insight evidence agent.  When enabled it replaces the
    # fixed metric/dimension selection; any LLM failure degrades to the fixed
    # selection with decision_driver=workflow_fallback.
    insight_agent_enabled: bool = True
    insight_agent_max_queries: int = 6
    insight_agent_max_queries_per_round: int = 3
    # Persist a completed graph once, instead of writing every node transition.
    # Interrupted human-review runs are still checkpointed so they can resume.
    checkpoint_durability: Literal["async", "sync", "exit"] = "exit"
    environment: str = "development"
    # Leave enough orchestration headroom for the bounded six-action chat
    # investigation loop plus normal workflow/finalization nodes. The separate
    # max_tool_calls/action budgets remain the actual data-access limits.
    max_graph_steps: int = 48
    max_tool_calls: int = 16
    # A real workflow can make several sequential LLM calls (triage, query
    # planning, and critic review).  Thirty seconds can expire after the
    # answer is generated but before trace persistence runs.
    max_latency_ms: int = 600000
    # Request-wide token ceiling; larger multi-step diagnostic flows can use
    # several calls before producing a grounded answer.
    max_tokens: int = 60000
    max_cost: float = 5.0
    # Input context budgets are separate from the request-wide LLM usage cap.
    # They control deterministic packing before each model call.
    context_default_input_budget: int = 6000
    context_triage_input_budget: int = 2500
    context_data_query_input_budget: int = 5000
    context_critic_input_budget: int = 6000
    context_answer_input_budget: int = 4000
    context_insight_input_budget: int = 8000
    context_model_context_window: int = 32768
    context_model_context_windows_json: str = "{}"
    context_reserved_output_tokens: int = 4096
    # Covers fixed instructions added inside llm_client (JSON-only clauses,
    # allowed-class labels, and message framing) that are not caller content.
    context_system_overhead_tokens: int = 128
    context_default_tokenizer_encoding: str = "cl100k_base"
    context_model_tokenizers_json: str = '{"hy3":"cl100k_base","mimo-v2.5":"cl100k_base"}'
    context_required_overflow_policy: Literal["reject", "allow_required"] = "reject"
    # Phase B: disabled by default.  Consolidation is an operator-triggered
    # proposal only; it can never activate or supersede a memory automatically.
    memory_llm_consolidation_enabled: bool = False
    memory_llm_consolidation_max_claims: int = 5
    memory_llm_consolidation_model: str = ""
    memory_consolidation_database_url: str = ""
    memory_consolidation_job_lease_seconds: int = 90
    memory_consolidation_max_attempts: int = 3
    # Phase C source snapshots are RLS-protected, bounded query/evidence
    # payloads.  They are a replay artifact, not a substitute for upstream CDC.
    memory_source_snapshot_max_bytes: int = 512 * 1024
    memory_auto_stale_on_data_version: bool = True
    # WP3 Mem0 OSS preference memory. Disabled by default so deployments
    # without a registered preference store keep the non-personalised path.
    mem0_enabled: bool = False
    mem0_qdrant_collection: str = "bizinsight_user_preferences_v1"
    mem0_qdrant_url: str = ""
    mem0_qdrant_api_key: str = ""
    mem0_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    mem0_embedding_dims: int = 512
    mem0_hmac_secret: str = ""
    mem0_scope_version: str = "v1"
    mem0_history_db_path: str = ""
    mem0_search_timeout_seconds: float = 1.5
    mem0_write_timeout_seconds: float = 5.0
    mem0_breaker_failures: int = 5
    mem0_breaker_window_seconds: float = 30.0
    mem0_top_k_per_type: int = 5
    mem0_top_k_total: int = 10
    # Calibrated on the v4 production-shaped real Mem0/Qdrant corpus: 0.55
    # reached T4 precision/recall 1.0 on 10 Resolver-generated queries and
    # worker-rendered preferences. Recalibrate with a larger paraphrase set.
    mem0_relevance_threshold: float = 0.55
    # Mem0's optional infer=True path uses an OpenAI-compatible model. The
    # adapter defaults to the application's configured provider only when an
    # explicit model/key is available; typed add(infer=False) remains the
    # production write path.
    mem0_llm_model: str = ""
    mem0_llm_api_key: str = ""
    mem0_llm_base_url: str = ""
    # Dedicated database URL for the cross-tenant preference outbox worker.
    # Use a narrowly scoped worker role that can lease rows and then apply
    # explicit tenant/user predicates; never use the web app's RLS role here.
    memory_preference_database_url: str = ""
    memory_preference_worker_lease_seconds: int = 60
    # 32-byte hex AES-GCM key; separate from the Mem0 entity HMAC secret.
    memory_preference_payload_key: str = ""
    # JSON map of retained encryption keys, e.g. {"v1":"<hex>","v2":"<hex>"}.
    # Keep old versions until every Outbox row encrypted with them has drained.
    memory_preference_payload_keyring: str = ""
    memory_preference_payload_current_key_version: str = "v1"
    data_versioning_enabled: bool = False
    max_revisions: int = 1
    critic_limit_policy: str = "human_review"
    # Keep the optional Critic policy explicit in Settings: critic.py reads
    # both switches on every graph run, including active routing fallback.
    critic_review_stop_on_failure_enabled: bool = True
    critic_default_non_thinking: bool = True
    # Observe the deterministic optional-review cascade before enabling any
    # call reduction. Active mode can skip only advisory reviews for a fully
    # validated registered numeric decomposition; groundedness remains required.
    critic_optional_review_cascade_mode: Literal["off", "shadow", "active"] = "shadow"
    mcp_timeout_seconds: float = 10.0
    mcp_transport: str = "stdio"
    remote_mcp_url: str = ""
    remote_mcp_oauth_token: str = ""
    remote_mcp_oauth_scope: str = "bizinsight.read"
    remote_mcp_audience: str = ""
    tool_permission_policy_path: str = "config/governance/tool_permissions.json"
    abac_database_enabled: bool = False
    permission_event_webhook_url: str = ""
    permission_event_webhook_secret: str = ""
    permission_event_database_url: str = ""
    permission_event_worker_interval_seconds: int = 30
    # Slow reasoning providers may legitimately take several minutes. Keep
    # this separate from the graph wall-clock budget so a single slow call is
    # not misclassified as provider unavailability.
    llm_timeout_seconds: float = 300.0
    llm_resilience_attempts: int = 1
    # Loading the local embedding model on the first RAG request can take more
    # than ten seconds even when Qdrant is healthy.
    rag_timeout_seconds: float = 30.0
    database_timeout_seconds: float = 10.0
    # When enabled, every new connection runs `SET ROLE bizinsight_app` so the
    # configured PostgreSQL RLS policies actually enforce tenant isolation
    # (superuser connections otherwise bypass RLS). Requires the role to exist
    # and the business schemas to be granted; see scripts/setup_rls_role.py.
    db_use_rls_role: bool = False
    # Role assumed on pooled connections. Application processes default to
    # bizinsight_app; narrowly scoped workers may select a dedicated role.
    db_rls_role: str = "bizinsight_app"
    otel_exporter_otlp_endpoint: str = ""
    tenant_id: str = "public"
    user_id: str = "anonymous"
    user_role: str = "analyst"
    tenant_auth_secret: str = ""
    tenant_auth_required: bool = False
    auth_session_hours: int = 8
    auth_bootstrap_token: str = ""
    tenant_checkpoint_schema: str = "governance"
    tenant_rag_collection_suffix: str = "tenant"
    delegation_signing_secret: str = ""
    # Phase 11: Code Agent jobs are intentionally separated from the API
    # process.  Local storage is only a development fallback; deployments use
    # the S3-compatible configuration below (MinIO works as well).
    # A separate URL lets the HTTP API use an RLS-bound login while the queue
    # worker uses a narrowly scoped BYPASSRLS login for cross-tenant leasing.
    # It falls back to POSTGRES_URL so existing local development keeps working.
    code_analysis_database_url: str = ""
    code_analysis_storage_backend: Literal["local", "s3"] = "local"
    code_analysis_storage_dir: str = "data/code_analysis"
    code_analysis_s3_endpoint: str = ""
    code_analysis_s3_bucket: str = "bizinsight-code-analysis"
    code_analysis_s3_access_key: str = ""
    code_analysis_s3_secret_key: str = ""
    code_analysis_s3_server_side_encryption: Literal["", "AES256", "aws:kms"] = ""
    code_analysis_s3_kms_key_id: str = ""
    code_analysis_provider: Literal["fake", "opencode"] = "fake"
    code_analysis_opencode_url: str = "http://opencode-generator:4096"
    code_analysis_opencode_username: str = "opencode"
    code_analysis_opencode_password: str = ""
    # Per-attempt guard. The overall job budget is configured separately so a
    # complex task is not mistaken for a hung HTTP request.
    code_analysis_opencode_timeout_seconds: float = 600.0
    code_analysis_opencode_protocol_retries: int = 1
    code_analysis_opencode_version: str = "1.18.5"
    code_analysis_model_proxy_url: str = ""
    code_analysis_model_proxy_token: str = ""
    code_analysis_model_proxy_max_requests_per_minute: int = 20
    code_analysis_model_proxy_max_concurrent_requests: int = 2
    code_analysis_model_proxy_upstream_max_retries: int = 3
    code_analysis_model_proxy_upstream_backoff_seconds: float = 2.0
    code_analysis_model_proxy_upstream_timeout_seconds: float = 600.0
    code_analysis_model_proxy_max_output_tokens: int = 16000
    # Code Agent uses the same canonical model setting as the rest of the app:
    # LLM_MODEL. Keeping the model in one environment-backed setting prevents
    # local Debug and Docker from silently selecting different models.
    code_analysis_upstream_api_key: str = ""
    code_analysis_upstream_base_url: str = ""
    code_analysis_max_upload_bytes: int = 50 * 1024 * 1024
    code_analysis_max_uncompressed_bytes: int = 256 * 1024 * 1024
    code_analysis_max_rows: int = 1_000_000
    code_analysis_max_columns: int = 200
    code_analysis_max_stored_upload_bytes_per_user: int = 200 * 1024 * 1024
    code_analysis_max_active_jobs_per_user: int = 3
    code_analysis_max_jobs_per_user_per_day: int = 20
    code_analysis_retention_hours: int = 24
    code_analysis_job_lease_seconds: int = 90
    code_analysis_job_deadline_seconds: int = 1800
    code_analysis_max_job_attempts: int = 3
    code_analysis_execution_timeout_seconds: int = 120
    # One initial candidate plus at most two bounded code-repair candidates.
    # Security review has its own tighter limit in the worker.
    code_analysis_max_repair_attempts: int = 2
    code_analysis_max_security_repair_attempts: int = 1
    code_analysis_security_review_enabled: bool = True
    code_analysis_max_output_bytes: int = 20 * 1024 * 1024
    code_analysis_max_result_bytes: int = 2 * 1024 * 1024
    code_analysis_runner_memory_mb: int = 1024
    code_analysis_runner_cpu_cores: float = 0.5
    code_analysis_runner_pids: int = 64
    code_analysis_runner_image: str = "bizinsight-analysis-runner:phase11"
    # 从环境变量中获取
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache  
def get_settings() -> Settings:
    return Settings()
