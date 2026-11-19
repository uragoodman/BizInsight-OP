// ============================================================
// 前端路由与导航定义
// 作用：集中定义侧边栏导航（navigation 数组）与 Vue Router 配置。
// 说明：navigation 每条含 path/name/label/icon/help/component——label 与 help
//       用于侧边栏与页面帮助，component 为按需加载的视图组件。
// 兜底：根路径 "/" 重定向到 /chat；未匹配路径统一重定向回 /chat。
// ============================================================
import { createRouter, createWebHistory } from "vue-router";

declare module "vue-router" {
  interface RouteMeta {
    adminOnly?: boolean;
    requiredPermission?: string;
    title?: string;
  }
}

export const navigation = [
  { path: "/code-analysis", name: "code-analysis", label: "数据分析任务", icon: "Cpu", help: "上传 CSV、Excel 或 Parquet 文件后，系统会先展示数据范围和分析计划。确认后才在隔离环境中生成并运行分析代码；原始文件默认保留 24 小时。", component: () => import("./views/CodeAnalysisView.vue") },
  { path: "/action-plans", name: "action-plans", label: "经营建议", icon: "List", help: "查看复杂分析生成的、带有证据和限制条件的经营建议草案。支持人工编辑和留档，系统不会自动执行业务动作。", component: () => import("./views/ActionPlansView.vue") },
  { path: "/chat", name: "chat", label: "Agent 问答", icon: "ChatDotRound", help: "像聊天一样提出经营问题。普通问题会直接回答；复杂分析会要求你先确认店铺、时间范围和执行计划。", component: () => import("./views/ChatView.vue") },
  { path: "/preferences", name: "preferences", label: "记忆与偏好", icon: "Collection", help: "查看和管理明确保存的个人偏好。新增、修改和删除均需确认；普通聊天不会自动写入长期记忆。", component: () => import("./views/PreferencesView.vue") },
  { path: "/explorer", name: "explorer", label: "数据浏览", icon: "DataAnalysis", help: "先选择店铺、日期和指标，再查看可信数据快照。确认后的范围会带入当前聊天，让 Agent 只分析这部分数据。", component: () => import("./views/ExplorerView.vue") },
  { path: "/reports", name: "reports", label: "周期报告", icon: "Document", help: "按指定周期和指标生成经营报告草稿，并在发布前完成审核。", component: () => import("./views/ReportsView.vue") },
  { path: "/metrics", name: "metrics", label: "指标与目标", icon: "TrendCharts", help: "查看每个经营指标的定义和单位，并设置用于报告比较的目标值。", component: () => import("./views/MetricsView.vue") },
  { path: "/quality", name: "quality", label: "质量仪表盘", icon: "DataLine", help: "查看版本化的 Golden、性能基准、Trace、Critic 与模型用量证据。仅授权管理员可访问。", component: () => import("./views/QualityView.vue"), meta: { title: "质量仪表盘", requiredPermission: "manage_usage" } },
  { path: "/knowledge", name: "knowledge", label: "知识库", icon: "Collection", help: "管理供 Agent 检索的业务文档。更新索引后，新的文档内容才能用于回答。", component: () => import("./views/KnowledgeView.vue") },
  { path: "/settings", name: "settings", label: "连接与权限", icon: "Setting", help: "配置本地开发身份和可选访问令牌，并测试 LLM 服务连接。令牌不会保存到浏览器本地存储。", component: () => import("./views/SettingsView.vue") }
];

export default createRouter({ history: createWebHistory(), routes: [
  { path: "/", redirect: "/chat" },
  ...navigation,
  { path: "/abac-policies", name: "abac-policies", component: () => import("./views/AbacPoliciesView.vue"), meta: { title: "ABAC 策略", adminOnly: true } },
  { path: "/:pathMatch(.*)*", redirect: "/chat" }
] });
