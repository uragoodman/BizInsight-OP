# 新增指标工作流

新增一个标准指标只需以下步骤。不需要修改 MCP Schema、LLM 提示词、前端选择器或 Scope 白名单。

## 检查清单

```
□ 1. 在 config/metrics/ 对应域 yaml 中添加指标配置
□ 2. 若需要新的查询逻辑：
   · 在 app/metrics/handlers/ 对应域中添加 handler 函数
   · 在 app/metrics/handler_catalog.py 中注册 handler_key
□ 3. 在 docs/knowledge/metrics/ 中添加口径文档（markdown）
□ 4. 添加测试：
   · handler 单元测试
   · 口径回归测试（与独立 SQL 对比）
   · Scope 越界与能力缺失测试
□ 5. 启动 registry.validate() 通过
□ 6. 前端 Explorer 下拉会自动出现（已从 API 获取）
```

## YAML 配置模板

```yaml
# config/metrics/<domain>.yaml
metrics:
  - metric_id: <canonical_name>
    display_name: <中文展示名>
    aliases: [别名1, 别名2]
    unit: <单位>
    display_format: currency|percentage|hours|days|score|number
    base_grain: order|customer|review|day
    aggregation: sum|weighted_ratio|average|distinct_count
    direction: higher_is_better|lower_is_better|neutral
    version: "1.0"
    documentation: docs/knowledge/metrics/<id>.md
    display_order: <排序位置>
    surfaces: [explorer, report, target, chat]
    implementations:
      platform:
        handler_key: <domain>.<handler_name>
        capabilities: [value, trend, compare, breakdown, period]
        dimensions: [customer_state, customer_city, ...]
        lineage_source: <schema.table_or_view>
        lineage_description: <口径说明>
      # seller 实现可选
      seller:
        handler_key: <domain>.<handler_name>
        capabilities: [value, trend, compare]
        dimensions: []
        lineage_source: <schema.table_or_view>
        lineage_description: <口径说明>
```

## Handler 函数签名

```python
# app/metrics/handlers/<domain>.py
from app.metrics.contracts import MetricQueryRequest

def my_handler(request: MetricQueryRequest) -> dict:
    # request.operation: "value" | "trend" | "compare" | "breakdown" | "period"
    # request.context: PlatformQueryContext | ControlledAnalysisScope
    # request.metric_date, request.start_date, request.end_date
    # request.dimension, request.limit, request.granularity
    ...
```

## 注册 Handler

```python
# app/metrics/handler_catalog.py
from app.metrics.handlers.<domain> import my_handler

HANDLER_REGISTRY = {
    ...
    "<domain>.<handler_name>": my_handler,
}
```

## 关键原则

- **YAML 不含 SQL**：只描述元数据、能力和来源。
- **Handler 是白名单注册**：禁止 YAML 写 Python 模块路径动态 import。
- **Scope 由实现分离**：platform 和 seller 的 handler、能力、维度分别在 YAML 的 `implementations` 中声明。
- **不支持的能力返回结构化错误**，不降级为任意查询。
- **启动时 fail-fast**：配置存在但 handler/文档缺失时应用拒绝启动。

## 参考

- 当前已有指标配置：`config/metrics/*.yaml`
- 当前已有 handler 注册：`app/metrics/handler_catalog.py`
- 语义 API：`GET /analysis/semantic/metrics`
