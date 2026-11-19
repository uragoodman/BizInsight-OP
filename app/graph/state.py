"""工作流的共享状态（AgentState）。

可以把 AgentState 理解成一次用户请求在图中流转时携带的“工作包”。所有节点收到的
都是同一个逻辑 state：上游节点把中间结果写进去，下游节点读取自己需要的字段并追加
自己的结果。它不是数据库模型，也不是 API 返回模型；它只在一次图执行（或 checkpoint
恢复）期间充当 Agent 间通信协议。

最值得优先理解的主链字段是：

``user_message → task_type / next_agent → 专项结果 → answer → critic_verdict → trace_id``

- ``user_message``：用户问了什么；
- ``task_type`` 与 ``next_agent``：triage 如何决定走哪条业务路径；
- ``metric_context`` / ``tool_result`` / ``analysis_result``：专项节点的核心产出；
- ``answer``：准备返回前端的文本；
- ``errors`` 与 ``critic_verdict``：是否发生错误、是否能正常结束或需要返工；
- ``trace_id``：一次运行的定位键，可用它读取最终 trace。

``total=False`` 的意思是 TypedDict 不要求每个字段一开始就存在。``init_state`` 会先补齐
列表、预算等公共默认值；其他字段只在对应节点成功运行后才出现。因此读取可选字段时应
使用 ``state.get(...)``，而不是假定它一定存在。

MECE 边界说明（每个节点只写自己负责的 key）：
  triage              → task_type, current_agent, next_agent, handoffs
  metric_definition   → metric_context, metric_authoritative_definition, rag_result, rag_citations, current_agent, node_history
  data_query          → tool_result, tool_calls, current_agent, node_history
  critic              → critic_result, critic_verdict, critic_decisions,
                        critic_revision_target, current_agent, node_history
  root_cause / report → analysis_result, current_agent, node_history
  diagnostic_*       → diagnostic plan/action/observations and termination metadata
  generate_answer     → answer / final_answer / answer_revision, current_agent, node_history
  init_state          → 全字段设默认值
  persist_trace       → 不新增字段，只读取并落盘
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from app.agents.contracts import CriticRevisionTarget


class MetricBatchExecutionState(TypedDict):
    """Terminal status for a planned multi-metric execution batch."""

    status: Literal["all_success", "partial_unavailable", "execution_failed"]


class AgentState(TypedDict, total=False):
    # ============================================================
    # 一、请求身份与主路由【最重要：理解图从哪里开始、要往哪里走】
    # ============================================================
    user_message: str
    # 输入：用户的原始问题。由 API 写入；triage、RAG、数据查询等几乎所有业务节点都会读取。

    trace_id: str
    # 本次“运行”的唯一 ID。由 API 生成；persist_trace 用它命名 trace，前端用它请求 /traces/{id}。

    evaluation_run_id: str
    # 可选的评测主记录 ID；评测脚本用它把 case、Trace 和用量关联到同一 run。

    thread_id: str
    # 可恢复“会话”的 ID。通常等于 trace_id；HITL 暂停/恢复时 LangGraph 以它定位 checkpoint。
    chat_session_id: Optional[str]
    chat_turn_id: Optional[str]
    task_id: Optional[str]
    # Durable conversation/task identity.  These are API-provided provenance
    # keys consumed by TurnResolution and ContextBundle; they are not authority
    # to widen scope.

    task_type: str
    # triage 的分类结果，包括指标/报告任务及 seller_catalog 目录查询。
    # 它决定主路径，例如 anomaly_analysis 会在 data_query 后进入 root_cause。
    triage_unresolved: bool
    triage_unresolved_reason: str

    current_agent: Optional[str]
    # 最近一个正在/刚刚执行的节点名，主要用于日志、错误归属和调试展示，不直接决定路由。

    next_agent: Optional[str]
    # 由 triage 或 root_cause 写入的“下一站”提示。条件边会读取它；例如 root_cause 补证后
    # 令 data_query 返回 root_cause，避免被 task_type 的通用规则分流走。

    handoffs: List[Dict[str, Any]]
    # Agent 间交接记录，如 {from, to, reason}。用于解释“为什么从 triage 到 data_query”。

    # ============================================================
    # 二、业务专项结果【最重要：这些是生成 answer 的事实来源】
    # ============================================================
    metric_context: str
    # metric_definition 从 RAG 拼出的可读上下文（指标定义、口径、SOP）。定义类问题常直接用它回答。

    metric_authoritative_definition: Dict[str, Any]
    # 指标注册表给出的权威结构化口径；公式、单位、源表和允许维度优先于 RAG 文本。

    rag_result: Dict[str, Any]
    # RAG 的原始结构化检索结果。供 Critic 审核、排查检索质量和 trace 回放；通常不直接展示给用户。

    rag_citations: List[Dict[str, Any]]
    # RAG 文档引用列表。与 metric_context 配套，标明指标口径/知识来自哪些可信文档。

    tool_result: Dict[str, Any]
    # data_query 最近一次主 MCP 查询的原始结果。普通查数、报告、异常分析都以它作为数据事实来源。

    seller_catalog: Dict[str, Any]
    # seller_catalog 节点按当前 TenantContext 返回的授权店铺清单；不修改分析范围。

    primary_tool_result: Dict[str, Any]
    # 指标规划器（registry planner）的输出 + 每个指标的结果；``tool_result`` 仍保留为
    # 第一个结果，以便兼容已有单指标节点。
    # 异常归因进入多轮补充查询前保存的首轮查询结果，避免后续补充查询覆盖掉关键基线数据。
    metric_confirmation: List[str]
    semantic_mapping_ids: List[str]
    skipped_semantic_mapping_ids: List[str]
    # 用户/调用方确认要查询的指标列表（由 API 从 payload 传入，并非 Agent 自行决定）。
    metric_planning_decision: Dict[str, Any]
    # 指标查询的规划决策；status 非 confirmed 时，workflow 会把它作为澄清问题返回给前端。
    evaluation_tool_surface: Dict[str, Any]
    # Internal formal-evaluation contract shared by B/G; never accepted from public request payloads.
    metric_query_batch_plan: Dict[str, Any]
    # Server-resolved registered factor decomposition intent. Target metric,
    # method and dependency inputs remain separate; this is not a causal task.
    registered_decomposition_request: Dict[str, Any]
    # data_query sets the execution outcome; Critic may raise its severity
    # after validating every result/evidence binding. Later nodes must retain it.
    metric_batch_execution: MetricBatchExecutionState
    authorized_metrics: List[Dict[str, Any]]
    # 多指标批量查询计划（结构化的 decision + plans），供 data_query 分批执行。
    query_results: List[Dict[str, Any]]
    # 批量查询返回的结果列表，按计划顺序与指标一一对应。

    analysis_result: Dict[str, Any]
    # Deterministically extracted user intent.  These fields constrain
    # anomaly follow-up dimensions; they are not model-authorized scope.
    requested_dimensions: List[str]
    unavailable_requested_dimensions: List[str]
    result_completeness: Dict[str, Any]
    premise_check: Dict[str, Any]
    awaiting_user_confirmation: bool
    # root_cause 或 report 的结构化业务产出，至少通常含 summary；generate_answer 优先从这里取答案。

    periodic_report: Dict[str, Any]
    # 周报/周期报告的完整结构化制品，供 Critic 做数值、图表、时效性检查，不等同于普通 answer。

    report_id: str
    # 已保存报告的业务 ID，供审核、发布和后续查询使用。

    report_version: int
    # 报告版本号。人工编辑、重新生成或发布流程需要它防止覆盖错误版本。

    report_risks: List[str]
    # 报告识别出的发布风险清单，例如数据不完整；human_review 可据此要求人工确认。

    # ============================================================
    # 三、证据链与最终回答【最重要：回答为什么可信、最终给用户什么】
    # ============================================================
    evidence_bundles: List[Dict[str, Any]]
    # 统一证据仓：RAG/MCP/归因/报告结果都带 stable evidence_id 写入这里，供后续声明溯源。

    explorer_context: Dict[str, Any]
    # 数据浏览器选中的结构化筛选条件。仅在用户明确启用“基于当前浏览数据分析”时存在。

    seller_scope_payload: Dict[str, Any]
    # Server-signed seller scope from the Explorer selector. It is forwarded
    # unchanged to the controlled MCP boundary and never sourced from the LLM.

    seller_authorization_boundary: Dict[str, Any]
    # Boundary returned by the controlled seller MCP handler for lifecycle handoff.

    seller_scope_fingerprint: str
    # Fingerprint of the canonical seller scope returned by the seller handler.

    seller_scope_terminal: bool
    # True when the graph stopped because the requested operation has no
    # seller-scoped handler; terminal routing must not enter platform paths.

    use_current_explorer_context: bool
    # 是否启用当前会话锁定的数据浏览范围。开关为 True 时，explorer_context 才会被
    # resolve_explorer_context 节点服务端重验并生成 controlled_analysis_scope。
    # 不允许由后续 Agent 修改为扩大范围——仅由初始请求设置。

    explorer_snapshot: Dict[str, Any]
    # 后端重新按 explorer_context 查询得到的可信预览快照；不会信任浏览器传来的数据行。
    controlled_analysis_scope: Dict[str, Any]
    # Server-issued identifier for a verified explorer selection.  The client
    # never supplies this as an execution credential.
    explorer_selection_id: str
    # 受控分析范围（seller/日期/指标边界）；启用浏览上下文时，data_query 据此锁定查询。
    data_version_manifest_id: str
    data_version_manifest_hash: str

    # Lifecycle + authorization foundation.  These are server-created values;
    # graph nodes may validate/consume them but must not manufacture them.
    authorization_boundary: Dict[str, Any]
    current_query_scope_fingerprint: str
    lifecycle_execution_id: str
    persistence_turn_id: str

    claims: List[Dict[str, Any]]
    # generate_answer 输出的“可追溯声明”。每条声明应通过 source_evidence_ids 指向 evidence_bundles。

    pre_critic_answer_snapshots: List[Dict[str, Any]]
    # 每次进入 Critic 前保留的答案输入快照。它不会被返工后的 answer 覆盖，供离线审核基准复现。

    answer: str
    # 【API 主要输出】最终展示给用户的文本。generate_answer 写入，/chat 与 /chat/stream 返回它。

    final_answer: Optional[str]
    # 内部最终答案副本；用于逐步把内部组合结果与对外 API 的 answer 字段解耦。

    # P0 Memory 与 Trace/Checkpoint 分工：它只提供压缩的受审计上下文，不存放原始数据或可执行 Scope。
    memory_session_id: str
    memory_enabled: bool
    memory_context: str
    session_memory_context: str
    memory_session_loaded: bool
    analysis_memories: List[Dict[str, Any]]
    memory_write_status: str
    memory_errors: List[str]
    # Phase 7: 多轮历史上下文（只读展示，不作为本轮事实依据）
    historical_evidence_context: List[Dict[str, Any]]
    historical_claims_context: List[Dict[str, Any]]
    historical_hypothesis_context: Dict[str, Any]
    previous_turn_context: Dict[str, Any]
    context_compilation: Dict[str, Any]
    # Phase 1 semantic contract and graph-level Phase 2 route observation.
    # The contract is template-independent; route decisions remain separate.
    semantic_contract: Dict[str, Any]
    dialogue_context_resolution: Dict[str, Any]
    routing_context: Dict[str, Any]
    complex_route_canary_selected: bool
    complex_route_canary_observation: Dict[str, Any]
    semantic_parse_status: str
    semantic_parse_observation: Dict[str, Any]
    complex_route_decision: Any
    route_decision_status: str
    semantic_interpreter_observation: Dict[str, Any]
    complex_plan: Dict[str, Any]
    complex_clarification: Dict[str, Any]
    # Same-scope result follow-up: compiler intent plus the validated projection
    # of the previous turn's result. Both are server-built and never derived
    # from free-form answer text.
    result_followup: Dict[str, Any]
    result_followup_context: Dict[str, Any]
    # Immutable-by-contract user request snapshot plus layer observations.
    # The hash is checked at planning/execution/review/answer boundaries.
    semantic_verifier_observation: Dict[str, Any]
    # P3 observation-only role verifier result. Kept in the typed graph state
    # so the terminal P4 sidecar can classify the completed request; routing,
    # contract construction, and execution must not consume this field.
    request_contract_hash: str
    metric_planning_request_contract_hash: str
    request_contract_observations: List[Dict[str, Any]]
    request_contract_failures: List[Dict[str, Any]]
    request_contract_query_reservations: List[Dict[str, Any]]
    # Bounded, versioned cross-turn task summary. It is only eligible as
    # historical context when its scope fingerprint matches the current turn.
    task_progress_context: Dict[str, Any]
    rag_query_trace: Dict[str, Any]
    answer_grounding: Dict[str, Any]
    # 上轮根因假设链的历史上下文（只读展示，不作为本轮事实依据）。

    # ============================================================
    # 四、Critic、返工与错误【最重要：解释为什么流程转向/终止】
    # ============================================================
    critic_result: Dict[str, Any]
    terminal_code: str
    decision_records: List[Dict[str, Any]]
    # Non-query control outcomes (for example, a registered capability gap).
    # These are deliberately separate from tool_calls so operational metrics do
    # not mistake a safe preflight stop for a failed data invocation.
    control_events: List[Dict[str, Any]]
    # Deterministic record of how generate_answer consumed the latest Critic
    # feedback. Feedback guides wording only; facts remain bound to evidence.
    answer_revision: Dict[str, Any]
    # Present only when adaptive mode produced the final answer from a
    # validated registered-decomposition card with no free-form additions.
    # Critic verifies it before it may skip optional LLM review.
    adaptive_answer_contract: Dict[str, Any]
    critic_llm_review_mode: str
    critic_llm_thinking_config: Dict[str, Any]
    # Critic 的完整审核结果，含各类结构化检查的明细；比 verdict 更适合排查“审核为何失败”。

    critic_decisions: List[Dict[str, Any]]
    # 每一轮 Critic 的裁决历史，返工后不会覆盖旧决定，方便 trace 看见完整审核过程。

    critic_verdict: Optional[str]
    # 【关键路由信号】approved / needs_revision / rejected / human_review。
    # _route_from_critic 根据它决定落盘、人工审核或回到上游节点返工。

    critic_revision_target: Optional[CriticRevisionTarget]
    # verdict=needs_revision 时指定返工节点，如 report 或 root_cause；只接受 workflow 注册过的节点名。

    errors: List[Dict[str, Any]]
    # 【关键故障信号】结构化错误列表。runtime、业务节点、预算和契约校验都会写入；多数 _route_* 发现
    # 它非空会优先去 critic。调试时首看 errors[-1]["code"]，不要只看 errors=True。

    _revision_counts: Dict[str, int]
    # 每个 Specialist 已被执行的次数，用于限制 Critic 返工次数，防止“生成→打回”无限循环。

    human_decision: Dict[str, Any]
    # human_review 从人工中断恢复后记录的批准/编辑/驳回决定及操作者信息。

    # ============================================================
    # 五、Root Cause 的多轮证据循环【只在 anomaly_analysis 路径重点关注】
    # ============================================================
    pending_query: Optional[Dict[str, Any]]
    # root_cause 请求 data_query 执行的“下一条”补充计划。有值时路由回 data_query；查完会清为 None。

    root_cause_queue: List[Dict[str, str]]
    # 尚未执行的补充查询队列，例如按客户地区、品类继续拆解异常贡献。

    root_cause_evidence: List[Dict[str, Any]]
    # data_query 针对 pending_query 返回的证据集合，root_cause 聚合它们再形成结论。

    branch_results: Dict[str, Dict[str, Any]]
    # P1.1 受控并行：每个独立事实分支（趋势/维度贡献等）只写自己的
    # branch_results.<purpose>，汇聚节点按稳定顺序合并，不共享 dict 原地覆盖。

    branch_timings: Dict[str, Any]
    # P1.1 分支并发耗时（total_ms + 各分支 duration_ms），供 trace 评估并行收益。

    # ============================================================
    # 六、工具调用、执行计划与节点历史【调试时最常看】
    # ============================================================
    tool_calls: List[Dict[str, Any]]
    # 每次 MCP 调用的标准化记录（server、tool_name、arguments、status、request_id）。
    mcp_call_attempts: List[Dict[str, Any]]
    # Internal provider experiment only: per-call local MCP resilience attempts, without raw errors.

    execution_plan: Dict[str, Any]
    # triage/planner 生成的高层执行计划，说明理论上有哪些 Agent 及依赖关系。

    node_history: List[Dict[str, Any]]
    # 【调试重点】业务节点写入的简短历史（node、summary 等）。trace 中的 graph_path 即从它提取。

    messages: List[Dict[str, Any]]
    # 预留的 Agent 间显式消息通道；当前主要使用共享 state，未来复杂协作可使用此字段。

    # ============================================================
    # 七、运行时预算与观测【排查超时、循环、成本时重点关注】
    # ============================================================
    runtime_steps: int
    # 已进入 instrument_node 外壳的节点总数。超过 max_graph_steps 会写 graph_step_budget_exceeded。

    runtime_budget: Dict[str, Any]
    # Turn 级累计预算；max_tokens 是当前有效 hard limit，global_max_tokens 保存服务端全局上限，
    # soft_tokens/token_budget_profile 是服务端按已确认 task_type 解析的任务 profile。
    # context packing 和 diagnostic plan 的轮数/工具限制是不同层，不代表累计 LLM 用量。

    runtime_started_at: float
    # 第一节点记录的 perf_counter 起点。后续节点用它计算“整个图”累计耗时；不是自然时间戳。

    llm_usage: Dict[str, Any]
    # LLM 客户端累积的 token 与成本状态；未知价格时 estimated_cost=null/cost_status=unknown。
    budget_warnings: List[Dict[str, Any]]
    # 成本未知、reservation 超额等预算观测，不表示预算被有效执行。
    llm_calls: List[Dict[str, Any]]
    # 本请求的逻辑模型调用及其应用层 SDK 尝试；不包含提示词或认证信息。
    llm_token_summary: Dict[str, Any]
    llm_call_config: Dict[str, Any]
    llm_call_config_summary: Dict[str, Any]
    # 基于 provider usage 的已知小计与完整性；未知 usage 不以 0 代替。

    node_timings: List[Dict[str, Any]]
    # 每个 wrapped 节点的 duration_ms。即便节点失败或预算提前返回，也会在 finally 中留下记录。

    # ============================================================
    # Deterministic context-packing audit events.  They retain inclusion and
    # clipping reasons but never raw prompt content.
    context_packing_events: List[Dict[str, Any]]
    context_packing_summary: Dict[str, Any]
    # WP1 Context Orchestrator: per-agent bundle metadata (no item content),
    # TriageContext route snapshot, and the evidence-refresh marker.
    triage_context: Dict[str, Any]
    context_bundles: Dict[str, Any]
    context_source_availability: Dict[str, List[Dict[str, Any]]]
    context_orchestration_events: List[Dict[str, Any]]
    context_orchestration_summary: Dict[str, Any]
    evidence_refresh: Dict[str, Any]
    turn_resolution: Dict[str, Any]
    # WP2: task relation decision passed from the API boundary and typed
    # preference source needs/candidates (metadata only in trace).
    turn_relation_decision: Dict[str, Any]
    preference_context: Dict[str, Any]
    task_state_version: int
    rag_review: Dict[str, Any]
    critic_investigation_context: Any
    # ============================================================
    # 八、租户、权限与版本元数据【通常由 API/init_state 写入，不由业务节点改动】
    # ============================================================
    tenant_id: str
    # 当前租户 ID；用于数据、checkpoint、trace 的租户隔离。

    user_id: str
    # 当前调用用户的 ID；用于审计和权限策略。

    user_role: str
    # 用户角色，如 analyst/admin；工具调用可据此决定是否允许访问特定数据。

    scope_type: str
    # 数据范围级别（如 platform / seller）；与 tenant_id 一起防止跨范围读取。

    seller_id: Optional[str]
    # 当范围限定到商家时的商家 ID；为空代表不按单一 seller 限制。

    audit_actor: str
    # 审计日志中记录“谁发起了这次动作”的身份，通常取 user_id。

    rag_release_version: str
    # 本次检索使用的已发布 RAG 知识版本，使旧 trace 能被解释和复现。

    permission_policy_version: str
    # 本次运行适用的工具/数据权限策略版本，便于审计当时为何允许或拒绝调用。

    trace_generation: str
    # 可比较的 Trace 分代；质量仪表盘只在相同 generation 内计算指标。

    trace_schema_version: str
    # Trace 外层存储结构的版本，便于兼容和识别 legacy trace。

    workflow_version: str
    # 当前编排、路由与节点契约组合的版本。

    prompt_version: str
    # 当前系统 Prompt/任务 Prompt 合同的版本，不记录实际 Prompt 原文。

    trace_kind: str
    # 运行入口类型，如 agent_chat；不同入口不能混用 Critic/延迟指标。

    _root_cause_iterations: int
    # root_cause → data_query 证据循环已执行的迭代次数。

    analysis_presentation: Dict[str, Any]
    # generate_answer 构建的结构化展示结论，被 SseCompletedEvent 和 ChatResponse
    # 的 presentation 字段引用。包含 facts / changes / contributions / recommendations
    # 等卡片级数据。不代替 answer 文本，两者共存。

    hypothesis_chain: List[Dict[str, Any]]
    # 单指标根因假设验证链 List[HypothesisStep]，root_cause node 写入

    hypothesis_chains_by_metric: Dict[str, List[Dict[str, Any]]]
    # 多指标根因假设验证链 metric_id → List[HypothesisStep]

    _evaluated_query_ids: Any
    # 已评价的 query_id 集合，防重复评价

    _prev_keys: Dict[str, Any]
    # instrument_node 内部调试辅助：记录上一节点的 key 集合。

    # Phase 1-4: private Root Cause diagnostic-agent protocol.
    diagnostic_plan: Dict[str, Any]
    diagnostic_step_state: Dict[str, str]
    diagnostic_action: Dict[str, Any]
    diagnostic_tool_results: List[Dict[str, Any]]
    diagnostic_candidate_data_gaps: List[Dict[str, Any]]
    # Transient server-computed attribution evidence passed from policy_execute
    # to observe across the compiled LangGraph node boundary. Keep it in the
    # graph schema; undeclared keys are dropped by graph state updates.
    diagnostic_local_attribution_evidence: Any
    diagnostic_decision_source: str
    diagnostic_fallback_reason: str
    diagnostic_backend_terminal: str
    diagnostic_round: int
    policy_decision: Dict[str, Any]
    hypotheses: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    finish_decision: Dict[str, Any]
    diagnostic_fallback: bool
    decision_driver: str
    termination_initiator: str
    # Server-enforced diagnostic skeleton (required evidence categories).
    diagnostic_target_metric: str
    investigation_state: Dict[str, Any]
    diagnostic_selected_candidate_id: str
    diagnostic_no_progress_rounds: int
    diagnostic_required_action: bool
    diagnostic_denied_metrics: List[str]
    diagnostic_investigation_action_count: int
    diagnostic_capability_gaps: List[Dict[str, Any]]
    diagnostic_data_reconciliation_failures: List[Dict[str, Any]]
    causal_readiness: Dict[str, Any]
    # Failure taxonomy for LLM plan/judgment calls (review fix #5): separates
    # "model cannot do the task" from "model formatted the output wrong".
    diagnostic_llm_failures: List[Dict[str, Any]]

    # Phase 10: shared runtime protocol record for the executed agent loop.
    agent_run: Dict[str, Any]


# 所有 AgentState 声明字段的白名单，用于 instrument_node 未知 key 检测。
# 从 TypedDict 的 __annotations__ 自动派生，无需手动维护。
AGENTSTATE_KEYS: frozenset[str] = frozenset(AgentState.__annotations__)
