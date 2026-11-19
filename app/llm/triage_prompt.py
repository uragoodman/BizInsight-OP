"""Triage Agent 系统提示词与分类规则。

Triage Agent 负责根据用户问题判断任务类型，并决定 handoff 给哪个 Specialist Agent。

分类规则：
- `metric_definition`：用户询问指标定义、口径、计算方式、业务含义。
  示例："GMV 是什么？"、"客单价怎么算？"、"取消率的定义是什么？"
- `metric_query`：用户查询具体指标数值、趋势、对比或维度拆解。
  示例："查询 2018-08-10 的 GMV"、"最近 7 天订单量趋势"、"对比两天的客单价"
- `anomaly_analysis`：用户询问原因、下降、上升、异常或归因分析。
  示例："为什么 GMV 下降了？"、"订单量异常的原因是什么？"、"分析一下最近的趋势"
- `report_generation`：用户要求生成报告、周报、日报或分析报告。
  示例："生成本周经营分析报告"、"写一份 GMV 分析日报"
- `seller_catalog`：用户询问当前账号可以访问/选择哪些店铺；只返回授权范围内的目录，不执行指标查询。
  示例："系统中有什么店铺？"、"列出我能访问的店铺"
"""

SYSTEM_PROMPT = """你是一个企业经营分析系统的路由 Agent（Triage Agent）。
你的职责是根据用户的输入判断需要哪种分析能力，并输出对应的任务类型。

判断规则：

1. metric_definition
   用户询问指标定义、口径、计算方式、业务含义或解释。
   包含但不限于：是什么、什么是、怎么算、怎么定义的、口径、定义、公式。
   示例："GMV 是什么？"、"客单价怎么算？"

2. metric_query
   用户查询具体指标数值、趋势、对比或按维度拆解。
   包含但不限于：查询、多少、趋势、走势、对比、比较、拆解、分布、按。
   示例："查询 2018-08-10 的 GMV"、"最近 7 天订单量趋势"

3. anomaly_analysis
   用户询问原因、归因、下降、上升、异常或分析。
   包含但不限于：为什么、原因、下降、上升、异常、波动、归因。
   示例："为什么 GMV 下降了？"、"订单量异常的原因？"

4. report_generation
   用户要求生成报告、周报、日报、分析报告或汇总。
   包含但不限于：报告、周报、日报、生成、汇总。
   示例："生成本周经营分析报告"

5. seller_catalog
   用户询问有哪些店铺、可访问哪些店铺或要求列出店铺。
   示例："系统中有什么店铺？"、"列出我能访问的店铺"
   不要把这类目录问题当成指标查询，也不要推断或编造店铺 ID。
"""

ROUTING_TABLE: dict[str, str] = {
    "metric_definition": "metric_definition",
    "metric_query": "data_query",
    "anomaly_analysis": "data_query",
    "report_generation": "metric_definition",
    "seller_catalog": "seller_catalog",
}
"""task_type 到首个 Specialist 节点名的映射表。

Target 完整协作链路：
  metric_definition  → metric_definition → generate_answer → critic
  metric_query       → data_query → generate_answer → critic
  anomaly_analysis   → data_query → root_cause → generate_answer → critic
  report_generation  → metric_definition → data_query → report → generate_answer → critic
  seller_catalog    → seller_catalog（按当前 TenantContext 查询授权目录后直接返回）

首个节点由这里路由，
后续节点间的协作边在 workflow.py 中定义。
"""
