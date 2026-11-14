import { api } from "./api.js";
import { $, $$, el, locator, statuses, toast } from "./dom.js";
import { renderUsage } from "./lab.js";
import { runPresentation } from "./run_status.js";
import {
  citationIndex,
  presentReport,
  unescapeReport,
} from "./report_content.js?v=verified-report-1";

const agentLabels = {
  supervisor: "Supervisor (phiên cũ)",
  retrieval: "Retrieval",
  reader: "Reader",
  summarizer: "Summarizer",
  fact_checker: "Fact Checker",
  writer: "Writer",
  orchestrator: "Orchestrator",
};
let currentResult = null;
let citations = new Map();

function appendCitations(parent, value) {
  const expression = /(?<!\\)\[(\d+)\]/g;
  let start = 0;
  for (const match of value.matchAll(expression)) {
    parent.append(
      document.createTextNode(unescapeReport(value.slice(start, match.index))),
    );
    const evidence = citations.get(match[0]);
    if (evidence) {
      const button = el("button", "citation-button", match[0]);
      button.title = `${evidence.filename} · ${locator(evidence)}`;
      button.addEventListener("click", () => showEvidence(evidence));
      parent.append(button);
    } else parent.append(document.createTextNode(match[0]));
    start = match.index + match[0].length;
  }
  parent.append(document.createTextNode(unescapeReport(value.slice(start))));
}

// This is a deliberately small, text-only renderer for the writer's known Markdown format.
// No innerHTML, remote links, scripts or raw HTML from documents/models are interpreted.
export function renderMarkdown(markdown) {
  const root = $("#report-content");
  root.replaceChildren();
  let list = null;
  for (const line of presentReport(markdown).split("\n")) {
    if (!line.trim()) {
      list = null;
      continue;
    }
    if (line.startsWith("# ")) {
      const heading = unescapeReport(line.slice(2));
      if (heading !== $("#result-title")?.textContent)
        root.append(el("h1", "", heading));
    } else if (line.startsWith("## "))
      root.append(el("h2", "", unescapeReport(line.slice(3))));
    else if (line.startsWith("- ")) {
      if (!list) {
        list = el("ul");
        root.append(list);
      }
      const item = el("li");
      appendCitations(item, line.slice(2));
      list.append(item);
    } else if (line.startsWith("> "))
      root.append(el("blockquote", "", unescapeReport(line.slice(2))));
    else if (/^\[\d+\]/.test(line)) {
      root.append(
        el("p", "report-source-label", unescapeReport(line.replace(/`/g, ""))),
      );
    } else if (
      line.startsWith("*") &&
      line.endsWith("*") &&
      !line.startsWith("**")
    ) {
      root.append(el("p", "report-note", unescapeReport(line.slice(1, -1))));
    } else {
      const paragraph = el("p");
      appendCitations(paragraph, line.replace(/\*\*/g, ""));
      root.append(paragraph);
    }
  }
}

export function selectTab(name) {
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  ["report", "checks", "methodology"].forEach((tab) => {
    const panel = $(`#${tab}-content`);
    if (panel) panel.hidden = tab !== name;
  });
  $("#coverage-panel").hidden =
    name !== "checks" || !currentResult?.coverage?.length;
}

export function renderEvents(events, status, result = null) {
  const outcome = runPresentation(status, events, result);
  const terminal = ["completed", "failed", "cancelled"].includes(status);
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
    const label = $(".agent-status", node);
    label.textContent = state
      ? statuses[state]
      : terminal
        ? "Chưa thực hiện"
        : "Đang chờ";
    if (state === "failed" && outcome.stopReason === "budget_exhausted")
      label.textContent = "Dừng do hết ngân sách";
    if (
      node.dataset.agent === "writer" &&
      state === "completed" &&
      outcome.partial
    )
      label.textContent =
        outcome.findings > 0 ? "Đã dựng báo cáo một phần" : "Đã dựng thông báo";
    if (["failed", "cancelled"].includes(status) && state === "running") {
      node.classList.remove("running");
      node.classList.add("failed");
      $(".agent-status", node).textContent = statuses[status];
    }
  });
  const statusLabel = $("#pipeline-status");
  statusLabel.textContent = outcome.label.toUpperCase();
  statusLabel.dataset.status = outcome.partial ? "partial" : status || "ready";
  statusLabel.classList.toggle(
    "running",
    status === "running" || status === "queued",
  );
  const live = $("#pipeline-live");
  live.hidden = events.length === 0;
  if (events.length) {
    const last = events.at(-1);
    live.textContent = outcome.partial
      ? `${outcome.label} · ${outcome.findings > 0 ? "Giữ kết quả đã kiểm chứng" : "Chưa có nhận định đã kiểm chứng để trả lời"}`
      : `Vòng ${last.round} · ${agentLabels[last.agent] || last.agent} · ${last.message}`;
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
    const peer = event.details?.peer_message;
    if (peer) {
      content.append(
        el(
          "p",
          "",
          `${agentLabels[peer.sender] || peer.sender} → ${agentLabels[peer.recipient] || peer.recipient}: ${peer.message}`,
        ),
      );
    }
    if (event.details?.phase === "team_planning" && event.details.approach) {
      content.append(el("p", "", event.details.approach));
    }
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

export function renderResult(result, displayQuestion) {
  currentResult = result;
  citations = citationIndex(result);
  $("#results-section").hidden = false;
  $("#empty-state").hidden = true;
  $("#result-title").textContent =
    displayQuestion || result.question || result.title;
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
  renderMethodology(result);
  renderEvents(result.events, "completed", result);
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
  const runId = currentResult.run_id;
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
  const read = el("button", "secondary-button", "Mở tài liệu gốc");
  read.type = "button";
  actions.append(read);
  const original = el("div", "source-previews");
  original.setAttribute("aria-live", "polite");
  root.append(actions, original);
  read.addEventListener("click", async () => {
    read.disabled = true;
    const doc = encodeURIComponent(evidence.document_id);
    const query = new URLSearchParams({
      run_id: runId,
      evidence_id: evidence.id,
    });
    if (evidence.page) {
      const preview = el("img", "source-page-preview");
      preview.alt = `Trang ${evidence.page} · ${evidence.filename}`;
      preview.addEventListener("error", () => {
        original.textContent = "Không tải được trang nguồn của phiên này.";
        read.disabled = false;
      });
      preview.src = `/api/documents/${doc}/pages/${evidence.page}/preview?${query}`;
      original.replaceChildren(preview);
      return;
    }
    original.textContent = "Đang mở tài liệu…";
    try {
      const source = await api(`/documents/${doc}/text?${query}`);
      const content = el("div", "source-context");
      const points = Array.from(source.text);
      const start = source.highlight_start;
      const end = source.highlight_end;
      if (
        Number.isInteger(start) &&
        Number.isInteger(end) &&
        start >= 0 &&
        end <= points.length &&
        start < end
      ) {
        content.append(
          points.slice(0, start).join(""),
          el("mark", "", points.slice(start, end).join("")),
          points.slice(end).join(""),
        );
      } else content.textContent = source.text;
      original.replaceChildren(content);
    } catch (error) {
      original.textContent = error.message;
      read.disabled = false;
    }
  });
  root.append(el("h3", "", "Ngữ cảnh đoạn truy xuất"));
  const chunk = currentResult.retrieved.find(
    (item) => item.id === evidence.chunk_id,
  );
  if (chunk) {
    const context = el("div", "source-context");
    const points = Array.from(chunk.text);
    const start = evidence.start - chunk.start;
    const end = evidence.end - chunk.start;
    if (start >= 0 && points.slice(start, end).join("") === evidence.quote) {
      context.append(
        document.createTextNode(points.slice(0, start).join("")),
        el("mark", "", evidence.quote),
        document.createTextNode(points.slice(end).join("")),
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
      `Đối soát từng ý · ${covered}/${result.coverage.length} ý được đánh giá là đủ`,
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

let methodologyVersion = 0;
async function renderMethodology(result) {
  const version = ++methodologyVersion,
    root = $("#methodology-content");
  if (!root) return;
  root.replaceChildren();
  const audit = result.citation_audit || {},
    card = el("article", `audit-summary ${audit.status || ""}`);
  const labels = {
    passed: "Đạt kiểm tra trích dẫn",
    failed: "Có nhận định bị loại",
    not_applicable: "Chưa áp dụng",
  };
  card.append(
    el("h3", "", labels[audit.status] || "Chưa có kiểm toán"),
    el(
      "p",
      "",
      `${audit.accepted_claim_ids?.length || 0} nhận định được giữ · ${audit.rejected_claim_ids?.length || 0} nhận định bị loại.`,
    ),
  );
  for (const reason of audit.reasons || []) card.append(el("p", "", reason));
  card.append(
    el(
      "small",
      "",
      "Kiểm toán vị trí/quote xác nhận nguồn trích dẫn, không bảo đảm mọi suy luận đều đúng.",
    ),
  );
  root.append(
    card,
    el(
      "p",
      "report-run-meta",
      `Điều phối: ${result.orchestration_mode} · Dừng: ${result.stop_reason}`,
    ),
  );
  const choices = Object.entries(result.agent_targets || {}).map(
    ([role, target]) => `${agentLabels[role]}: ${target}`,
  );
  if (choices.length)
    root.append(
      el("p", "report-run-meta", `Cấu hình đã chốt: ${choices.join(" · ")}`),
    );
  const loading = el("p", "", "Đang đọc checklist phương pháp…");
  root.append(loading);
  try {
    const data = await api(`/runs/${result.run_id}/methodology`);
    if (version !== methodologyVersion) return;
    loading.textContent = data.warning;
    const names = {
      seeds: "Seed / trạng thái ngẫu nhiên",
      uncertainty: "Độ bất định",
      test: "Kiểm định thống kê",
      sample_size: "Cỡ mẫu",
      hardware: "Phần cứng",
      limitations: "Hạn chế",
    };
    for (const item of data.items) {
      const section = el("article", "methodology-item");
      section.append(
        el("h4", "", names[item.criterion] || item.criterion),
        el(
          "p",
          "",
          item.matches.length
            ? `${item.matches.length} bằng chứng có từ khóa liên quan.`
            : "Chưa thấy trong bằng chứng đã đọc.",
        ),
      );
      for (const match of item.matches) {
        const quote = el("blockquote", "", match.quote);
        const button = el(
          "button",
          "text-button",
          `Mở ${match.evidence_id}${match.page ? " · Trang " + match.page : ""}`,
        );
        button.addEventListener("click", () => {
          const source = result.evidence.find(
            (e) => e.id === match.evidence_id,
          );
          if (source) showEvidence(source);
        });
        section.append(quote, button);
      }
      root.append(section);
    }
  } catch (error) {
    if (version === methodologyVersion) loading.textContent = error.message;
  }
}
