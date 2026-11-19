"""Agent 层模块。

这个目录用于放置各类 Agent 的职责实现。当前项目里的 Agent 还属于轻量版：
- `data_query_agent`：负责把自然语言问题解析成结构化查询计划
- `critic_agent`：负责对工具结果做第一层结构化审核

后续如果继续向多 Agent 演进，这里还会逐步加入更完整的：
- Triage Agent
- Root Cause Agent
- Report Agent
等角色实现。
"""
