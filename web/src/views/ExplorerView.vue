<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { api } from "@/api";
import TrendChart from "@/components/TrendChart.vue";
import ScopedChatPanel from "@/components/ScopedChatPanel.vue";
import { useAppStore } from "@/stores/app";
import type { ExplorerQuery, InventorySnapshotBatchInput, InventorySnapshotBatchResult, InventorySnapshotResult, SellerOption, SellerScopeType } from "@/types";
import { DEFAULT_ANALYSIS_RANGE, validateExplorerQuery } from "@/utils/scope";
import { useSellerScopeCredential } from "@/utils/sellerScopeCredential";
import { formatMetricValue } from "@/utils/format";

const store = useAppStore();
const loading = ref(false);
const inventoryLoading = ref(false);
const sellers = ref<SellerOption[]>([]);
const metricOptions = ref<Array<{ value: string; label: string }>>([]);
const sellerScopeType = ref<SellerScopeType>("single");
const selectedSellerIds = ref<string[]>([]);
const sellerMetricIds = ref<string[]>(["gmv"]);
const includeUnattributed = ref(false);
const allScopeConfirmed = ref(false);
// form must exist before the credential watcher registers its scope getter.
const existingScope = store.activeSession?.scope;
const form = ref<ExplorerQuery>(existingScope ? { ...existingScope } : {
  seller_id: null,
  metric_name: "gmv",
  ...DEFAULT_ANALYSIS_RANGE,
  granularity: "day",
});
const sellerCredential = useSellerScopeCredential(
  () => [sellerScopeType.value, form.value.seller_id, form.value.start_date, form.value.end_date, form.value.metric_name,
    form.value.granularity, selectedSellerIds.value.join("\u001f"), sellerMetricIds.value.join("\u001f"),
    includeUnattributed.value, allScopeConfirmed.value],
  computed(() => store.activeSessionId),
);
const sellerScopePayload = sellerCredential.payload;
const preset = ref("month");
const presetYear = ref<number | undefined>(undefined);
const presetValue = ref(1);
const bounds = ref({ minYear: 2016, maxYear: 2018 });
const validationError = computed(() => validateExplorerQuery(form.value));
const sellerDateSpanDays = computed(() => {
  if (!form.value.start_date || !form.value.end_date) return -1;
  return (Date.parse(`${form.value.end_date}T00:00:00Z`) - Date.parse(`${form.value.start_date}T00:00:00Z`)) / 86_400_000;
});
const sellerScopeDisplay = computed(() => {
  const intent = sellerScopePayload.value?.seller_scope_intent;
  if (!intent) return "";
  const kind = intent.seller_scope_type === "all" ? "全平台（ALL）" : intent.seller_scope_type === "selected" ? "指定商户子集" : "单店";
  const sellersLabel = intent.seller_scope_type === "all" ? "全部已授权商户" : intent.seller_ids.join("、");
  return `${kind} · ${sellersLabel} · ${intent.start_date} 至 ${intent.end_date}`;
});
let debounce: number | undefined;

const inventoryAsOfDate = ref(form.value.end_date);
const inventoryBasis = ref<"on_hand" | "available">("available");
const inventoryWarehouse = ref("");
const inventoryProduct = ref("");
const inventoryResult = ref<InventorySnapshotResult | null>(null);
const inventoryBatchJson = ref("");
const inventoryBatchLoading = ref(false);
const inventoryBatchResult = ref<InventorySnapshotBatchResult | null>(null);
const canManageInventory = computed(() => store.account?.role === "owner" && store.account.permissions.includes("manage_sellers"));

watch(
  () => [form.value.seller_id, form.value.start_date, form.value.end_date, inventoryAsOfDate.value, inventoryBasis.value, inventoryWarehouse.value, inventoryProduct.value],
  () => { inventoryResult.value = null; },
);

watch(() => sellerScopeType.value, (scopeType) => {
  allScopeConfirmed.value = false;
  if (scopeType !== "all") includeUnattributed.value = false;
});

function searchSellers(keyword: string) {
  window.clearTimeout(debounce);
  debounce = window.setTimeout(async () => {
    try { sellers.value = (await api.sellers(store.auth, keyword)).items; }
    catch (error) { ElMessage.error(error instanceof Error ? error.message : "店铺搜索失败"); }
  }, 250);
}

function presetChanged() { presetValue.value = preset.value === "recent_months" ? 3 : 1; }

onMounted(async () => {
  try {
    const [result, metrics] = await Promise.all([
      api.timeBounds(store.auth),
      api.metrics(store.auth),
    ]);
    bounds.value = { minYear: Number(result.min_date.slice(0, 4)), maxYear: Number(result.max_date.slice(0, 4)) };
    presetYear.value = bounds.value.maxYear;
    metricOptions.value = metrics
      .filter((metric: Record<string, any>) => (metric.surfaces || []).includes("explorer"))
      .map((metric: Record<string, any>) => ({ value: metric.metric_name, label: metric.display_name || metric.metric_name }));
    // 异步加载目录不能覆盖用户已选或会话已锁定的指标。
    if (metricOptions.value.length && !metricOptions.value.some((item) => item.value === form.value.metric_name)) {
      form.value.metric_name = metricOptions.value[0].value;
    }
  } catch { /* keep the current form defaults */ }
});

async function applyPreset() {
  try {
    const payload: { kind: string; year?: number; value?: number } = { kind: preset.value };
    if (["year", "half", "quarter"].includes(preset.value) && presetYear.value) payload.year = presetYear.value;
    if (["recent_months", "half", "quarter"].includes(preset.value)) payload.value = presetValue.value;
    const resolved = await api.resolveTimeRange(store.auth, payload);
    form.value.start_date = resolved.start_date;
    form.value.end_date = resolved.end_date;
    inventoryAsOfDate.value = resolved.end_date;
    if (resolved.truncated) ElMessage.info(`已按可用数据截断，数据截至 ${resolved.data_max_date}`);
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : "时间范围解析失败"); }
}

async function preview() {
  if (sellerScopeType.value !== "single") {
    ElMessage.info("范围预览目前只支持单店；指定商户和全平台范围可签发受控 seller scope 后向 Agent 提问");
    return;
  }
  if (validationError.value) return ElMessage.warning(validationError.value);
  loading.value = true;
  store.clearPreview();
  try {
    const result = await api.preview(store.auth, form.value);
    const validatedScope = {
      seller_id: result.context.seller_id,
      metric_name: result.context.metric_name,
      start_date: result.context.start_date,
      end_date: result.context.end_date,
      granularity: result.context.granularity,
    };
    if (result.summary.status === "ok") {
      await store.saveActiveSessionScope(validatedScope);
      ElMessage.success("已加载数据，可直接在右侧向 Agent 提问");
    } else {
      await store.saveActiveSessionScope(null);
      ElMessage.warning("当前范围没有数据");
    }
    store.preview = result;
    form.value = validatedScope;
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : "数据预览失败"); }
  finally { loading.value = false; }
}

async function applySellerScope() {
  if (!form.value.start_date || !form.value.end_date || form.value.end_date < form.value.start_date) {
    return ElMessage.warning("请先设置有效的开始和结束日期");
  }
  if (sellerDateSpanDays.value > 1098) return ElMessage.warning("商户分析日期范围不能超过 1098 天");
  if (!sellerMetricIds.value.length) return ElMessage.warning("至少选择一个 seller 分组指标");
  if (sellerScopeType.value === "single" && !form.value.seller_id) return ElMessage.warning("单店范围必须选择一个店铺");
  if (sellerScopeType.value === "selected") {
    if (!selectedSellerIds.value.length) return ElMessage.warning("指定商户子集不能为空");
    if (selectedSellerIds.value.length > 100) return ElMessage.warning("最多选择 100 个商户");
  }
  if (sellerScopeType.value === "all" && !allScopeConfirmed.value) {
    return ElMessage.warning("请明确确认全平台 ALL 范围后再继续");
  }
  if (sellerScopeType.value !== "all" && includeUnattributed.value) {
    return ElMessage.warning("只有全平台 ALL 范围可以把未归属 seller 行计入 total");
  }

  loading.value = true;
  sellerCredential.invalidate();
  store.clearPreview();
  // Bind the short-lived token to the current chat session. Creating the
  // session here prevents first-submit auto-creation from switching the
  // session after the token has been signed.
  if (!store.activeSession) store.createSession();
  sellerCredential.bindToSession(store.activeSessionId || null);
  try {
    const response = await api.resolveSellerScope(store.auth, {
      seller_scope_type: sellerScopeType.value,
      // ALL is explicit and must send an empty list; never use omitted scope as ALL.
      seller_ids: sellerScopeType.value === "all"
        ? []
        : sellerScopeType.value === "single" ? [form.value.seller_id!] : [...selectedSellerIds.value],
      start_date: form.value.start_date,
      end_date: form.value.end_date,
      metric_ids: [...sellerMetricIds.value],
      granularity: form.value.granularity,
      include_unattributed: sellerScopeType.value === "all" && includeUnattributed.value,
    });
    // Keep the signed payload byte-for-byte as returned by the server. It is
    // short-lived and stays in this page's memory; refresh requires re-resolve.
    sellerCredential.setPayload(response.seller_scope_payload);
    ElMessage.success("服务端已验证并签发当前商户范围");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "商户范围验证失败");
  } finally { loading.value = false; }
}

async function queryInventorySnapshot() {
  if (!form.value.seller_id) return ElMessage.warning("请先选择一个店铺");
  if (!form.value.start_date || !form.value.end_date || form.value.end_date < form.value.start_date) {
    return ElMessage.warning("请先设置有效的开始和结束日期");
  }
  if (inventoryAsOfDate.value < form.value.start_date || inventoryAsOfDate.value > form.value.end_date) {
    return ElMessage.warning("库存时点必须落在当前筛选日期范围内");
  }

  inventoryLoading.value = true;
  inventoryResult.value = null;
  try {
    inventoryResult.value = await api.inventorySnapshot(store.auth, {
      seller_id: form.value.seller_id,
      start_date: form.value.start_date,
      end_date: form.value.end_date,
      // 以所选日期 UTC 当天结束为截点，避免漏掉当天较晚的快照。
      as_of: `${inventoryAsOfDate.value}T23:59:59+00:00`,
      quantity_basis: inventoryBasis.value,
      warehouse_id: inventoryWarehouse.value.trim() || undefined,
      product_id: inventoryProduct.value.trim() || undefined,
      limit: 500,
    });
    if (inventoryResult.value.status === "not_found") ElMessage.info("该店铺和时点范围内没有观察到库存快照");
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "库存快照查询失败");
  } finally { inventoryLoading.value = false; }
}

function loadInventoryBatchFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    ElMessage.warning("JSON 文件不能超过 10 MB");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    inventoryBatchJson.value = String(reader.result ?? "");
    inventoryBatchResult.value = null;
  };
  reader.onerror = () => ElMessage.error("无法读取所选 JSON 文件");
  reader.readAsText(file);
}

function downloadInventoryBatchTemplate() {
  const template = {
    source_system: "replace-with-source-system",
    source_batch_id: "replace-with-unique-batch-id",
    rows: [{
      seller_id: form.value.seller_id || "replace-with-authorized-seller-id",
      warehouse_id: "warehouse-001",
      product_id: "product-001",
      sku: "SKU-001",
      snapshot_at: "2026-09-25T23:59:59+00:00",
      snapshot_semantics: "daily_close",
      on_hand_quantity: 12,
      reserved_quantity: 2,
      available_quantity: 10,
      quantity_unit: "unit",
    }],
  };
  const blob = new Blob([JSON.stringify(template, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "inventory-snapshot-batch.example.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

async function ingestInventoryBatch() {
  if (!canManageInventory.value) return ElMessage.error("只有具备管理权限的店铺所有者可以导入");
  if (new TextEncoder().encode(inventoryBatchJson.value).byteLength > 10 * 1024 * 1024) {
    return ElMessage.warning("JSON 批次不能超过 10 MB");
  }
  let batch: InventorySnapshotBatchInput;
  try {
    batch = JSON.parse(inventoryBatchJson.value) as InventorySnapshotBatchInput;
  } catch {
    return ElMessage.warning("JSON 格式无效，请检查导入文件内容");
  }
  if (!batch || typeof batch !== "object" || !Array.isArray(batch.rows) || batch.rows.length < 1 || batch.rows.length > 10_000) {
    return ElMessage.warning("批次必须包含 1 至 10,000 条 rows 记录");
  }
  const sellerIds = [...new Set(batch.rows.map((row) => row?.seller_id).filter((id): id is string => typeof id === "string"))];
  if (sellerIds.length !== 1 || sellerIds[0] !== form.value.seller_id) {
    return ElMessage.warning("批次必须只包含当前选择店铺的记录；服务端还会重新校验店铺授权");
  }
  try {
    await ElMessageBox.confirm(
      `即将向当前租户追加 ${batch.rows.length} 条快照。已有记录不会覆盖；手工导入不代表数据完整或权威。确认继续？`,
      "确认导入库存快照",
      { type: "warning", confirmButtonText: "确认追加", cancelButtonText: "取消" },
    );
  } catch {
    return;
  }
  inventoryBatchLoading.value = true;
  inventoryBatchResult.value = null;
  try {
    inventoryBatchResult.value = await api.ingestInventorySnapshotBatch(store.auth, batch);
    ElMessage.success(`已接收 ${inventoryBatchResult.value.received_rows} 条，新增 ${inventoryBatchResult.value.inserted_rows} 条，重复忽略 ${inventoryBatchResult.value.duplicate_rows_ignored} 条`);
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : "库存快照导入失败");
  } finally { inventoryBatchLoading.value = false; }
}
</script>

<template>
  <div class="page">
    <div class="page-title">
      <div><h1>电商数据浏览</h1><p>从服务端加载可信快照；确认后，筛选范围会在当前聊天会话持续生效。</p></div>
      <el-tag type="info">会话级范围</el-tag>
    </div>

    <el-card shadow="never">
      <div class="time-presets">
        <el-select v-model="preset" style="width:170px" @change="presetChanged">
          <el-option label="本月（数据最新月）" value="month"/><el-option label="上月" value="previous_month"/>
          <el-option label="最近 N 个月" value="recent_months"/><el-option label="全年" value="year"/>
          <el-option label="半年" value="half"/><el-option label="季度" value="quarter"/>
        </el-select>
        <el-select v-if="preset === 'recent_months'" v-model="presetValue" style="width:110px"><el-option v-for="n in [1,3,6,12]" :key="n" :label="`${n} 个月`" :value="n"/></el-select>
        <el-select v-if="preset === 'half'" v-model="presetValue" style="width:110px"><el-option label="上半年" :value="1"/><el-option label="下半年" :value="2"/></el-select>
        <el-select v-if="preset === 'quarter'" v-model="presetValue" style="width:110px"><el-option v-for="n in [1,2,3,4]" :key="n" :label="`Q${n}`" :value="n"/></el-select>
        <el-input-number v-if="['year','half','quarter'].includes(preset)" v-model="presetYear" :min="bounds.minYear" :max="bounds.maxYear" placeholder="年份"/>
        <el-button @click="applyPreset">应用时间预设</el-button>
      </div>
      <el-form :model="form" label-position="top" class="filter-grid" @submit.prevent="preview">
        <el-form-item v-if="sellerScopeType === 'single'" label="店铺"><el-select v-model="form.seller_id" filterable remote reserve-keyword clearable placeholder="搜索 seller_id" :remote-method="searchSellers" @visible-change="(visible:boolean) => visible && searchSellers('')"><el-option v-for="seller in sellers" :key="seller.seller_id" :label="`${seller.label} · ${seller.order_count ?? 0} orders`" :value="seller.seller_id"/></el-select></el-form-item>
        <el-form-item label="开始日期"><el-date-picker v-model="form.start_date" type="date" value-format="YYYY-MM-DD"/></el-form-item>
        <el-form-item label="结束日期"><el-date-picker v-model="form.end_date" type="date" value-format="YYYY-MM-DD"/></el-form-item>
        <el-form-item label="指标"><el-select v-model="form.metric_name"><el-option v-for="item in metricOptions" :key="item.value" :label="item.label" :value="item.value"/></el-select></el-form-item>
        <el-form-item label="粒度"><el-segmented v-model="form.granularity" :options="[{label:'日',value:'day'},{label:'周',value:'week'},{label:'月',value:'month'},{label:'季',value:'quarter'},{label:'年',value:'year'}]"/></el-form-item>
        <el-form-item class="filter-action"><el-button type="primary" :loading="loading" :disabled="sellerScopeType !== 'single'" @click="preview">查询数据</el-button></el-form-item>
      </el-form>
      <el-divider content-position="left">商户分组分析范围</el-divider>
      <div class="seller-scope-controls">
        <el-form-item label="范围类型">
          <el-radio-group v-model="sellerScopeType">
            <el-radio-button value="single">单店</el-radio-button>
            <el-radio-button value="selected">指定商户子集</el-radio-button>
            <el-radio-button value="all">全平台 ALL</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item v-if="sellerScopeType === 'selected'" label="选择商户（最多 100 个）">
          <el-select v-model="selectedSellerIds" multiple filterable remote reserve-keyword collapse-tags collapse-tags-tooltip
            placeholder="搜索并选择一个或多个已授权商户" :remote-method="searchSellers"
            @visible-change="(visible:boolean) => visible && searchSellers('')">
            <el-option v-for="seller in sellers" :key="seller.seller_id" :label="seller.label" :value="seller.seller_id"/>
          </el-select>
          <small>所选 {{ selectedSellerIds.length }} / 100；空选择不能提交。</small>
        </el-form-item>
        <el-alert v-if="sellerScopeType === 'all'" title="ALL 是全平台范围，不会默认启用。请确认你要查询所有已授权商户。" type="warning" :closable="false" show-icon />
        <el-checkbox v-if="sellerScopeType === 'all'" v-model="allScopeConfirmed" class="section">
          我明确确认使用全平台 ALL 范围
        </el-checkbox>
        <el-form-item label="分组指标">
          <el-checkbox-group v-model="sellerMetricIds">
            <el-checkbox value="gmv">GMV</el-checkbox>
            <el-checkbox value="order_count">订单数</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        <el-checkbox v-if="sellerScopeType === 'all'" v-model="includeUnattributed">
          将未归属 seller 行显式计入 ALL total（仅平台管理员可用）
        </el-checkbox>
        <el-alert v-if="sellerScopeType === 'selected'" title="未归属 seller 行统计整个日期范围，单独展示且不计入所选商户 total。" type="info" :closable="false" show-icon />
        <el-alert v-if="sellerScopeType === 'all'" :title="includeUnattributed ? '未归属 seller 行来自整个日期范围；已明确选择纳入 ALL total。' : '未归属 seller 行来自整个日期范围，单独展示且默认不计入 ALL total。'" type="info" :closable="false" show-icon />
        <el-alert title="order_count 按 seller 分组为 distinct orders per seller，商户间订单可能重叠、分组数不可相加；total 由全范围 distinct 订单并集独立计算。" type="info" :closable="false" show-icon />
        <div class="seller-scope-actions">
          <span>范围上限 1098 天；授权、feature flag 与日期范围会由服务端再次校验。</span>
          <el-button type="primary" :loading="loading" :disabled="sellerMetricIds.length === 0" @click="applySellerScope">校验并签发范围</el-button>
        </div>
      </div>
    </el-card>

    <el-card v-if="sellerScopePayload" shadow="never" class="section">
      <template #header><strong>已签发 seller 分析范围</strong></template>
      <el-descriptions :column="2" border>
        <el-descriptions-item label="当前范围">{{ sellerScopeDisplay }}</el-descriptions-item>
        <el-descriptions-item label="范围凭证有效期至">{{ sellerScopePayload.provenance.expires_at }}</el-descriptions-item>
        <el-descriptions-item label="指标">{{ sellerScopePayload.seller_scope_intent.metric_ids.join("、") }}</el-descriptions-item>
        <el-descriptions-item label="粒度">{{ sellerScopePayload.seller_scope_intent.granularity }}</el-descriptions-item>
      </el-descriptions>
      <el-alert v-if="sellerScopePayload.seller_scope_intent.seller_scope_type !== 'single'" class="section"
        :title="sellerScopePayload.seller_scope_intent.seller_scope_type === 'selected' ? '未归属 seller 行是整个日期范围的 NULL seller 行，不属于所选集合，也不进入 selected total。' : sellerScopePayload.seller_scope_intent.include_unattributed ? '未归属 seller 行是整个日期范围的 NULL seller 行；按显式选择计入 ALL total。' : '未归属 seller 行是整个日期范围的 NULL seller 行，单独展示且不进入 ALL total。'"
        type="info" :closable="false" show-icon />
      <el-alert class="section" title="order_count 的 seller 分组是 distinct orders per seller，可能相互重叠；总数是独立计算的 distinct 订单并集，不从分组结果相加。" type="info" :closable="false" show-icon />
      <ScopedChatPanel v-if="!store.preview" :seller-scope-payload="sellerScopePayload" />
    </el-card>

    <el-card v-if="sellerScopeType === 'single'" shadow="never" class="section">
      <template #header><strong>库存快照查询</strong></template>
      <p>按当前店铺和日期范围读取指定时点前最近一次上报的库存观察值。此处不导入库存，也不把未上报记录当作零库存。</p>
      <el-alert title="数据覆盖范围未知：结果只代表已上报的商品/仓库观察记录，不能据此计算全店库存总量或缺货率。" type="warning" :closable="false" show-icon class="section"/>
      <el-form label-position="top" class="filter-grid" @submit.prevent="queryInventorySnapshot">
        <el-form-item label="库存时点（按 UTC 当天结束）"><el-date-picker v-model="inventoryAsOfDate" type="date" value-format="YYYY-MM-DD"/></el-form-item>
        <el-form-item label="数量口径"><el-select v-model="inventoryBasis"><el-option label="可售数量" value="available"/><el-option label="实物数量" value="on_hand"/></el-select></el-form-item>
        <el-form-item label="仓库编号（可选）"><el-input v-model="inventoryWarehouse" maxlength="128" placeholder="留空查询已上报的仓库"/></el-form-item>
        <el-form-item label="商品编号（可选）"><el-input v-model="inventoryProduct" maxlength="128" placeholder="留空查询已上报的商品"/></el-form-item>
        <el-form-item class="filter-action"><el-button type="primary" :loading="inventoryLoading" :disabled="!form.seller_id" @click="queryInventorySnapshot">查询库存快照</el-button></el-form-item>
      </el-form>
      <template v-if="inventoryResult">
        <el-alert v-if="inventoryResult.truncated" title="结果超过 500 条，仅显示前 500 条；请用仓库或商品编号缩小范围。" type="warning" :closable="false" show-icon class="section"/>
        <el-descriptions :column="3" border class="section">
          <el-descriptions-item label="查询店铺">{{ inventoryResult.seller_id }}</el-descriptions-item>
          <el-descriptions-item label="库存时点">{{ inventoryResult.as_of }}</el-descriptions-item>
          <el-descriptions-item label="观察记录数">{{ inventoryResult.observed_product_warehouse_count }}</el-descriptions-item>
        </el-descriptions>
        <el-table :data="inventoryResult.items" stripe empty-text="当前时点没有已上报的库存观察记录">
          <el-table-column prop="product_id" label="商品编号" min-width="150"/>
          <el-table-column prop="warehouse_id" label="仓库编号" min-width="130"/>
          <el-table-column prop="snapshot_at" label="最近上报时间" min-width="190"/>
          <el-table-column label="数量"><template #default="{row}">{{ row.quantity === null ? "缺失" : `${row.quantity} ${row.quantity_unit}` }}</template></el-table-column>
          <el-table-column label="状态"><template #default="{row}"><el-tag :type="row.measurement_status === 'observed' ? 'success' : 'warning'">{{ row.measurement_status === "observed" ? "已观测" : "数量缺失" }}</el-tag></template></el-table-column>
          <el-table-column prop="snapshot_semantics" label="快照口径" min-width="130"/>
          <el-table-column prop="source_system" label="来源" min-width="120"/>
        </el-table>
      </template>
    </el-card>

    <el-card v-if="canManageInventory && sellerScopeType === 'single'" shadow="never" class="section">
      <template #header><strong>库存快照批次导入（店铺所有者）</strong></template>
      <el-alert title="此入口只追加经授权的 JSON 观察记录，不校验来源系统真实性或全量覆盖。导入结果不能直接用于全店库存总量/缺货率；后端会再次检查所有者角色、租户、店铺和日期授权。" type="warning" :closable="false" show-icon />
      <div class="section inventory-import-actions">
        <input type="file" accept=".json,application/json" aria-label="选择库存快照 JSON 批次" @change="loadInventoryBatchFile" />
        <el-button @click="downloadInventoryBatchTemplate">下载 JSON 示例</el-button>
      </div>
      <el-input v-model="inventoryBatchJson" type="textarea" :rows="12" maxlength="10485760" show-word-limit placeholder="选择 JSON 文件，或粘贴包含 source_system、source_batch_id 和 rows 的批次内容" />
      <div class="section"><el-button type="primary" :loading="inventoryBatchLoading" :disabled="!form.seller_id || !inventoryBatchJson.trim()" @click="ingestInventoryBatch">校验并追加库存批次</el-button></div>
      <el-descriptions v-if="inventoryBatchResult" :column="3" border class="section">
        <el-descriptions-item label="店铺">{{ inventoryBatchResult.seller_id }}</el-descriptions-item>
        <el-descriptions-item label="接收 / 新增">{{ inventoryBatchResult.received_rows }} / {{ inventoryBatchResult.inserted_rows }}</el-descriptions-item>
        <el-descriptions-item label="重复忽略">{{ inventoryBatchResult.duplicate_rows_ignored }}</el-descriptions-item>
        <el-descriptions-item label="来源系统">{{ inventoryBatchResult.source_system }}</el-descriptions-item>
        <el-descriptions-item label="批次编号">{{ inventoryBatchResult.source_batch_id }}</el-descriptions-item>
        <el-descriptions-item label="单位">{{ inventoryBatchResult.quantity_units.join("、") }}</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <template v-if="store.preview">
      <el-alert v-if="store.preview.coverage?.truncated" type="warning" :closable="false" :title="`请求周期已截断，数据截至 ${store.preview.coverage.data_max_date}`"/>
      <el-row :gutter="16" class="section">
        <el-col :md="15">
          <el-card shadow="never" class="section"><template #header>指标趋势</template><TrendChart :data="store.preview.series" :metric="store.preview.summary.metric_name" :unit="store.preview.summary.unit"/></el-card>
          <el-card shadow="never" class="section"><template #header>时间明细</template><el-table :data="store.preview.table.rows" stripe><el-table-column prop="bucket" label="时间"/><el-table-column label="指标值"><template #default="{row}">{{formatMetricValue(row.value,store.preview?.summary.unit)}}</template></el-table-column></el-table></el-card>
        </el-col>
        <el-col :md="9"><ScopedChatPanel v-if="store.activeSession?.scope" :seller-scope-payload="sellerScopePayload"/><el-card v-else shadow="never" class="scoped-chat-panel"><el-empty description="请选择一个有数据的范围后再向 Agent 提问。"/></el-card></el-col>
      </el-row>
    </template>
    <el-empty v-else-if="!sellerScopePayload" description="选择筛选条件并查询后，在此查看服务端返回的可信数据快照。"/>
  </div>
</template>

<style scoped>
.inventory-import-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.seller-scope-controls { display: grid; gap: 10px; max-width: 880px; }
.seller-scope-controls :deep(.el-select) { width: 100%; }
.seller-scope-controls small { display: block; margin-top: 5px; color: var(--el-text-color-secondary); }
.seller-scope-actions { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-top: 4px; color: var(--el-text-color-secondary); font-size: 13px; }
</style>
