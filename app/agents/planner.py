"""Controlled planner: only emits pre-registered, permission-safe paths."""

from app.agents.contracts import AgentName, ExecutionPlan, TaskType

_PLANS = {
    TaskType.METRIC_DEFINITION: [AgentName.METRIC_DEFINITION],
    TaskType.METRIC_QUERY: [AgentName.DATA_QUERY],
    TaskType.ANOMALY_ANALYSIS: [AgentName.DATA_QUERY, AgentName.ROOT_CAUSE],
    TaskType.REPORT_GENERATION: [AgentName.METRIC_DEFINITION, AgentName.DATA_QUERY, AgentName.REPORT],
    TaskType.SELLER_CATALOG: [AgentName.SELLER_CATALOG],
}


def build_execution_plan(task_type: str) -> ExecutionPlan:
    """Return an allowlisted plan; unknown tasks fall back to definition lookup."""
    parsed = TaskType(task_type) if task_type in TaskType._value2member_map_ else TaskType.METRIC_DEFINITION
    agents = _PLANS[parsed]
    dependencies = {agent: [agents[index - 1]] for index, agent in enumerate(agents) if index}
    return ExecutionPlan(agents=agents, dependencies=dependencies)


def validate_or_fallback(payload: dict[str, object], task_type: str) -> ExecutionPlan:
    """Reject unregistered topology and fall back to the audited static route."""
    try:
        plan = ExecutionPlan.model_validate(payload)
        if any(agent not in AgentName for agent in plan.agents):
            raise ValueError("unregistered agent")
        return plan
    except Exception:
        return build_execution_plan(task_type)
