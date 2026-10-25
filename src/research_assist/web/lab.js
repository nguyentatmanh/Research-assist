import { $, el, icon } from "./dom.js";

export const roleLabels = {
  retrieval: "Retrieval",
  reader: "Reader",
  summarizer: "Summarizer",
  fact_checker: "Fact Checker",
  writer: "Writer",
};

export function renderRoutes(routes) {
  const root = $("#agent-routes");
  root.replaceChildren();
  for (const route of routes) {
    const item = el("div", "route-item");
    item.dataset.agent = route.agent;
    const role = el("strong", "", "");
    const roleIcons = {
      retrieval: "search",
      reader: "book",
      summarizer: "stack",
      fact_checker: "shield",
      writer: "pen",
    };
    role.append(
      icon(roleIcons[route.agent] || "settings"),
      document.createTextNode(roleLabels[route.agent] || route.agent),
    );
    item.append(
      role,
      el("span", "", route.model),
      el(
        "small",
        `execution-badge ${route.execution}`,
        route.execution === "local"
          ? route.planner_execution === "api"
            ? "Local + API lập kế hoạch"
            : "Local"
          : route.execution === "demo"
            ? "Demo"
            : route.provider,
      ),
    );
    root.append(item);
  }
}

export function renderUsage(result) {
  let root = $("#usage-panel");
  if (!root) {
    root = el("details", "usage-panel");
    root.id = "usage-panel";
    $("#metrics-row").after(root);
  }
  root.replaceChildren();
  const summary = el("summary");
  const tokens = (result.input_tokens || 0) + (result.output_tokens || 0);
  const cost =
    result.estimated_cost_usd == null
      ? "Chưa cấu hình đơn giá"
      : `$${result.estimated_cost_usd.toFixed(6)} ước tính`;
  summary.append(
    el("strong", "", "Tài nguyên phiên"),
    el(
      "span",
      "",
      `${tokens.toLocaleString("vi-VN")} token · ${result.cache_hits || 0} cache hit · ${cost}`,
    ),
  );
  root.append(summary);
  const table = el("table", "usage-table");
  const header = el("tr");
  [
    "Agent / Model",
    "Chạy",
    "Request",
    "Cache",
    "Token vào / ra",
    "Thời gian",
  ].forEach((text) => header.append(el("th", "", text)));
  table.append(header);
  for (const usage of result.agent_usage || []) {
    const row = el("tr");
    const model = el("td");
    model.append(
      el("strong", "", roleLabels[usage.agent] || usage.agent),
      el("small", "", usage.model),
    );
    row.append(
      model,
      el("td", "", usage.execution),
      el("td", "", usage.calls),
      el("td", "", usage.cache_hits),
      el("td", "", `${usage.input_tokens} / ${usage.output_tokens}`),
      el("td", "", `${(usage.elapsed_ms / 1000).toFixed(2)}s`),
    );
    table.append(row);
  }
  const wrap = el("div", "table-scroll");
  wrap.append(table);
  root.append(
    wrap,
    el(
      "p",
      "",
      "Request gồm cả thử lại HTTP và sửa JSON; không phải số nhiệm vụ. Token do provider trả về; thời gian gồm xử lý ở từng bước. Đơn giá do người vận hành cấu hình.",
    ),
  );
}

function number(value, min = 0, max = Infinity) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < min ||
    value > max
  )
    throw new Error("Benchmark chứa số liệu không hợp lệ.");
  return value;
}

export function renderBenchmark(data) {
  if (
    data.kind !== "retrieval_benchmark" ||
    data.format_version !== 2 ||
    !Array.isArray(data.variants) ||
    !data.variants.length ||
    data.variants.length > 30
  )
    throw new Error("Chọn JSON benchmark được tạo bởi evaluate_retrieval.py.");
  number(data.eligible_queries, 1);
  number(data.corpus_size, 1);
  const rows = data.variants.map((v) => ({
    name: String(v.name).slice(0, 180),
    recall: number(v.metrics["recall@10"], 0, 1),
    hit: number(v.metrics["hit@10"], 0, 1),
    mrr: number(v.metrics["mrr@10"], 0, 1),
    ndcg: number(v.metrics["ndcg@10"], 0, 1),
    p50: number(v.latency_p50_ms),
    p95: number(v.latency_p95_ms),
  }));
  $("#benchmark-summary").textContent =
    `${data.eligible_queries} câu hỏi · ${data.corpus_size} đoạn · split ${data.split} · ${data.device}. Độ trễ dưới đây chỉ tính truy xuất.`;
  const root = $("#benchmark-results");
  root.replaceChildren();
  const table = el("table", "benchmark-table");
  const header = el("tr");
  ["Cấu hình", "Recall@10", "Hit@10", "MRR@10", "nDCG@10", "p50 / p95"].forEach(
    (text) => header.append(el("th", "", text)),
  );
  table.append(header);
  for (const value of rows) {
    const row = el("tr");
    const name = el("td");
    const bar = el("div", "score-bar");
    const fill = el("span");
    fill.style.width = `${value.recall * 100}%`;
    bar.append(fill);
    name.append(el("strong", "", value.name), bar);
    row.append(
      name,
      ...[value.recall, value.hit, value.mrr, value.ndcg].map((v) =>
        el("td", "", v.toFixed(3)),
      ),
      el("td", "", `${value.p50.toFixed(1)} / ${value.p95.toFixed(1)} ms`),
    );
    table.append(row);
  }
  root.append(table);
}
