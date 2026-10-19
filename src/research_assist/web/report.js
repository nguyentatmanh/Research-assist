import { $, $$, el, icon, locator, statuses, toast } from "./dom.js";
import { renderUsage } from "./lab.js";

const agentLabels = {
  retrieval: "Retrieval",
  reader: "Reader",
  summarizer: "Summarizer",
  fact_checker: "Fact Checker",
  writer: "Writer",
  orchestrator: "Orchestrator",
};
let currentResult = null;
let citations = new Map();

function unescapeMarkdown(text) {
  return text.replace(/\\([\\`*_{}\[\]<>()#!|])/g, "$1");
}

function citationIndex(result) {
  const map = new Map();
  const rows = result.markdown.split("\n");
  for (let index = 0; index < rows.length; index++) {
    const match = rows[index].match(/^\[(\d+)\].*`(ch_[a-z0-9]+)`$/);
    if (!match) continue;
    const quoteLine = rows
      .slice(index + 1, index + 5)
      .find((line) => line.startsWith("> "));
    const quote = quoteLine ? unescapeMarkdown(quoteLine.slice(2)) : null;
    const evidence = result.evidence.find(
      (e) =>
        e.chunk_id === match[2] &&
        (!quote || e.quote.replace(/\n/g, " ") === quote),
    );
    if (evidence) map.set(Number(match[1]), evidence);
  }
  return map;
}

function appendCitations(parent, value) {
  const expression = /(?<!\\)\[(\d+)\]/g;
  let start = 0;
  for (const match of value.matchAll(expression)) {
    parent.append(
      document.createTextNode(
        unescapeMarkdown(value.slice(start, match.index)),
      ),
    );
    const evidence = citations.get(Number(match[1]));
    if (evidence) {
      const button = el("button", "citation-button", match[0]);
      button.title = `${evidence.filename} · ${locator(evidence)}`;
      button.addEventListener("click", () => showEvidence(evidence));
      parent.append(button);
    } else parent.append(document.createTextNode(match[0]));
    start = match.index + match[0].length;
  }
  parent.append(document.createTextNode(unescapeMarkdown(value.slice(start))));
}

// This is a deliberately small, text-only renderer for the writer's known Markdown format.
// No innerHTML, remote links, scripts or raw HTML from documents/models are interpreted.
export function renderMarkdown(markdown) {
  const root = $("#report-content");
  root.replaceChildren();
  let list = null;
  let target = root;
  for (const line of markdown.split("\n")) {
    if (
      [
        "## Sources",
        "## Nguồn trích dẫn",
        "## Question coverage",
        "## Mức đáp ứng câu hỏi",
      ].includes(line)
    ) {
      const details = el("details", "report-appendix");
      details.append(el("summary", "", line.slice(3)));
      root.append(details);
      target = details;
      list = null;
      continue;
    }
    if (line.startsWith("## ")) target = root;
    if (!line.trim()) {
      list = null;
      continue;
    }
    if (line.startsWith("# "))
      target.append(el("h1", "", unescapeMarkdown(line.slice(2))));
    else if (line.startsWith("## "))
      target.append(el("h2", "", unescapeMarkdown(line.slice(3))));
    else if (line.startsWith("- ")) {
      if (!list) {
        list = el("ul");
        target.append(list);
      }
      const item = el("li");
      appendCitations(item, line.slice(2));
      list.append(item);
    } else if (line.startsWith("> "))
      target.append(el("blockquote", "", unescapeMarkdown(line.slice(2))));
    else if (/^\[\d+\]/.test(line)) {
      target.append(
        el(
          "p",
          "report-source-label",
          unescapeMarkdown(line.replace(/`/g, "")),
        ),
      );
    } else if (
      line.startsWith("*") &&
      line.endsWith("*") &&
      !line.startsWith("**")
    ) {
      target.append(
        el("p", "report-note", unescapeMarkdown(line.slice(1, -1))),
      );
    } else {
      const paragraph = el("p");
      appendCitations(paragraph, line.replace(/\*\*/g, ""));
      target.append(paragraph);
    }
  }
}

export function selectTab(name) {
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  ["report", "checks"].forEach((tab) => {
    $(`#${tab}-content`).hidden = tab !== name;
  });
}

export function renderEvents(events, status) {
  const emptyState = $("#activity-empty");
  if (emptyState) emptyState.hidden = events.length > 0;
  const eventCount = $("#trace-count");
  if (eventCount) eventCount.textContent = `${events.length} sự kiện`;
  $("#back-research").textContent =
    status === "completed" ? "Xem kết quả nghiên cứu" : "Về câu hỏi";
  const latest = new Map();
  for (const event of events) latest.set(event.agent, event);
  $$(".agent-node").forEach((node) => {
    const event = latest.get(node.dataset.agent);
    node.classList.remove("running", "completed", "failed");
    const state = event?.status;
    if (state && state !== "cancelled") node.classList.add(state);
    $(".agent-status", node).textContent = state ? statuses[state] : "Đang chờ";
    if (["failed", "cancelled"].includes(status) && state === "running") {
      node.classList.remove("running");
      node.classList.add("failed");
      $(".agent-status", node).textContent = statuses[status];
    }
  });
  const statusLabel = $("#pipeline-status");
  statusLabel.textContent = status
    ? statuses[status].toUpperCase()
    : "SẴN SÀNG";
  statusLabel.dataset.status = status || "ready";
  statusLabel.classList.toggle(
    "running",
    status === "running" || status === "queued",
  );
  const live = $("#pipeline-live");
  live.hidden = events.length === 0;
  if (events.length) {
    const last = events.at(-1);
    live.textContent = `Vòng ${last.round} · ${agentLabels[last.agent] || last.agent} · ${last.message}`;
  }
  const trace = $("#trace-content");
  const key = events[0]?.created_at || "empty";
  if (trace.dataset.eventKey !== key) {
    trace.replaceChildren();
    trace.dataset.eventKey = key;
  }
  const existing = new Set(
    [...trace.children].map((row) => Number(row.dataset.sequence)),
  );
  for (const event of events) {
    if (existing.has(event.sequence)) continue;
    const item = el("div", "trace-item");
    item.dataset.sequence = event.sequence;
    item.dataset.agent = event.agent;
    const filter = $("#trace-filter")?.value || "all";
    item.hidden = filter !== "all" && filter !== event.agent;
    item.append(
      el("span", "trace-sequence", String(event.sequence).padStart(2, "0")),
    );
    const content = el("div");
    content.append(
      el(
        "strong",
        "",
        `${agentLabels[event.agent] || event.agent} · Vòng ${event.round} · ${statuses[event.status]}`,
      ),
    );
    content.append(el("p", "", event.message));
    if (event.details && Object.keys(event.details).length) {
      const facts = el("div", "trace-facts");
      for (const name of [
        "execution",
        "model",
        "calls",
        "input_tokens",
        "output_tokens",
        "cache_hits",
      ]) {
        if (event.details[name] != null)
          facts.append(el("span", "", `${name}: ${event.details[name]}`));
      }
      const detail = el("details");
      detail.append(
        el("summary", "", "Truy vấn, nguồn & quyết định"),
        el("pre", "", JSON.stringify(event.details, null, 2)),
      );
      content.append(facts, detail);
    }
    item.append(
      content,
      el(
        "span",
        "trace-time",
        event.elapsed_ms ? `${(event.elapsed_ms / 1000).toFixed(2)}s` : "—",
      ),
    );
    trace.append(item);
  }
}

export function renderResult(result) {
  currentResult = result;
  citations = citationIndex(result);
  $("#results-section").hidden = false;
  $("#empty-state").hidden = true;
  $("#result-title").textContent = result.title;
  $("#export-md").href = `/api/runs/${result.run_id}/export?format=md`;
  $("#export-json").href = `/api/runs/${result.run_id}/export?format=json`;
  const supported = result.checks.filter(
    (check) => check.verdict === "supported",
  ).length;
  const values = [
    [
      new Set(result.evidence.map((e) => e.document_id)).size,
      "Nguồn có bằng chứng",
    ],
    [`${supported}/${result.claims.length}`, "Nhận định được hỗ trợ"],
    [result.llm_calls, "Lần gọi model"],
    [
      `${(result.elapsed_ms / 1000).toFixed(1)}s`,
      `${result.rounds} vòng nghiên cứu`,
    ],
  ];
  const metrics = $("#metrics-row");
  metrics.replaceChildren();
  values.forEach(([number, label]) => {
    const item = el("div", "metric");
    item.append(el("strong", "", number), el("span", "", label));
    metrics.append(item);
  });
  const warnings = $("#result-warnings");
  renderUsage(result);
  warnings.replaceChildren();
  result.warnings.forEach((message) =>
    warnings.append(el("div", "result-warning", message)),
  );
  $("#checks-count").textContent = String(result.checks.length);
  renderMarkdown(result.markdown);
  renderChecks(result);
  renderCoverage(result);
  renderEvidence(result.evidence);
  renderEvents(result.events, "completed");
  selectTab("report");
  window.dispatchEvent(new Event("research-result-rendered"));
}

function renderChecks(result) {
  const root = $("#checks-content");
  root.replaceChildren();
  const claims = new Map(result.claims.map((claim) => [claim.id, claim]));
  const labels = {
    supported: "Được nguồn hỗ trợ",
    insufficient: "Chưa đủ bằng chứng",
    contradicted: "Nguồn mâu thuẫn",
  };
  if (!result.checks.length)
    root.append(
      el("p", "loading-label", "Chưa có nhận định đủ bằng chứng để kiểm tra."),
    );
  for (const check of result.checks) {
    const item = el("div", "check-card");
    item.append(
      el("span", `check-status ${check.verdict}`, labels[check.verdict]),
    );
    item.append(
      el("p", "", claims.get(check.claim_id)?.text || check.claim_id),
      el("small", "", check.explanation),
    );
    const links = el("div");
    for (const id of check.evidence_ids) {
      const evidence = result.evidence.find((e) => e.id === id);
      if (!evidence) continue;
      const button = el(
        "button",
        "text-button",
        `${evidence.filename} · ${locator(evidence)}`,
      );
      button.addEventListener("click", () => showEvidence(evidence));
      links.append(button);
    }
    item.append(links);
    root.append(item);
  }
}

function renderEvidence(evidence) {
  const root = $("#evidence-list");
  root.replaceChildren();
  $("#evidence-count").textContent = String(evidence.length);
  if (!evidence.length)
    root.append(
      el("p", "evidence-note", "Không có trích dẫn hợp lệ cho câu hỏi này."),
    );
  evidence.forEach((source, index) => {
    const card = el("div", "evidence-card");
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute(
      "aria-label",
      `Mở bằng chứng ${index + 1}: ${source.filename}`,
    );
    const header = el("div", "evidence-card-header");
    header.append(
      el("span", "evidence-label", `E${index + 1}`),
      el("span", "", "TRÍCH NGUYÊN VĂN"),
    );
    card.append(
      header,
      el("h4", "", source.filename),
      el("span", "source-locator", locator(source)),
      el("blockquote", "", source.quote),
    );
    card.addEventListener("click", () => showEvidence(source));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showEvidence(source);
      }
    });
    root.append(card);
  });
  filterEvidence();
}

function filterEvidence() {
  const search = $("#evidence-search");
  if (!search) return;
  const query = search.value.trim().toLocaleLowerCase("vi");
  const cards = $$(".evidence-card");
  cards.forEach((card) => {
    card.hidden = !card.textContent.toLocaleLowerCase("vi").includes(query);
  });
  $("#evidence-no-match").hidden = !query || cards.some((card) => !card.hidden);
}
$("#evidence-search")?.addEventListener("input", filterEvidence);

export function showEvidence(evidence) {
  if (!currentResult) return;
  const dialog = $("#source-dialog");
  $("#source-title").textContent = evidence.filename;
  const root = $("#source-dialog-content");
  root.replaceChildren();
  const meta = el("div", "source-meta");
  meta.append(
    el("span", "", locator(evidence)),
    el("span", "", `Ký tự ${evidence.start}–${evidence.end}`),
  );
  root.append(
    meta,
    el("blockquote", "source-detail-quote", evidence.quote),
    el("p", "source-relevance", evidence.relevance),
  );
  const actions = el("div", "source-actions");
  const link = el("a", "secondary-button", "Mở tài liệu gốc");
  link.href = `/api/documents/${evidence.document_id}/file${evidence.page !== null ? `#page=${evidence.page}` : ""}`;
  link.target = "_blank";
  link.rel = "noopener";
  link.append(icon("external"));
  actions.append(link);
  root.append(actions, el("h3", "", "Ngữ cảnh đoạn truy xuất"));
  const chunk = currentResult.retrieved.find(
    (item) => item.id === evidence.chunk_id,
  );
  if (chunk) {
    const context = el("div", "source-context");
    const start = chunk.text.indexOf(evidence.quote);
    if (start >= 0) {
      context.append(
        document.createTextNode(chunk.text.slice(0, start)),
        el("mark", "", evidence.quote),
        document.createTextNode(
          chunk.text.slice(start + evidence.quote.length),
        ),
      );
    } else context.textContent = chunk.text;
    root.append(context);
  }
  if (!dialog.open) dialog.showModal();
}

export async function copyReport() {
  if (!currentResult) return;
  try {
    await navigator.clipboard.writeText(currentResult.markdown);
    toast("Đã sao chép báo cáo Markdown.");
  } catch {
    toast(
      "Trình duyệt chưa cho phép clipboard. Bạn có thể tải bản Markdown.",
      true,
    );
  }
}

function renderCoverage(result) {
  const panel = $("#coverage-panel");
  panel.replaceChildren();
  panel.hidden = !(result.coverage || []).length;
  if (panel.hidden) return;
  const detail = el("details");
  const covered = result.coverage.filter((r) => r.status === "covered").length;
  detail.append(
    el(
      "summary",
      "",
      `Độ đầy đủ: ${covered}/${result.coverage.length} ý được model đánh giá đã đáp ứng`,
    ),
  );
  panel.append(detail);
  panel.append(
    el(
      "p",
      "coverage-note",
      "Đánh giá của model về độ đầy đủ; bạn có thể đối chiếu từng nhận định với nguồn.",
    ),
  );
  const requirements = new Map(
    (result.requirements || []).map((r) => [r.id, r.text]),
  );
  const labels = {
    covered: "Đủ ý theo model",
    partial: "Một phần",
    missing: "Thiếu bằng chứng",
    unassessed: "Chưa xác định",
  };
  for (const row of result.coverage) {
    const item = el("div", "coverage-item");
    item.append(
      el(
        "span",
        `coverage-status ${row.status}`,
        labels[row.status] || "Chưa xác định",
      ),
    );
    const content = el("div");
    content.append(
      el(
        "strong",
        "",
        requirements.get(row.requirement_id) || row.requirement_id,
      ),
      el("p", "", row.explanation),
    );
    item.append(content);
    detail.append(item);
  }
}
