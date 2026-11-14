import { api, uploadDocument } from "./api.js";
import {
  expandQuestion,
  mountCommandMenu,
  readCommand,
} from "./commands.js?v=slash-menu-1";
import { $, $$, el } from "./dom.js";
import { recordStatusLabel } from "./run_status.js";
import {
  clearGpuKey,
  initializeGpuSettings,
  loadGpuSettings,
} from "./gpu_settings.js?v=kaggle-chat-1";
import {
  citationIndex,
  presentReport,
  unescapeReport,
} from "./report_content.js?v=source-reader-1";
import { createSourceReader } from "./source_reader.js?v=continuous-pdf-1";
import { createTargetForm } from "./target_editor.js?v=agent-matrix-1";

const byId = (id) => document.getElementById(id);
const click = (id, handler) => byId(id).addEventListener("click", handler);
const prompt = byId("promptInput");
const feed = byId("messageFeed");
const hero = byId("welcomeHero");
const send = byId("sendBtn");
const selected = new Set();
const targets = new Map();
const modelSelects = $$(".agent-model-select");
const modeTargets = {
  gemini: "api",
  cpu_min: "demo",
  cpu_max: "cpu",
  gpu: "gpu",
};
const targetLabels = {
  default: "Mặc định",
  demo: "CPU Min",
  api: "Cloud API",
  cpu: "CPU Max",
  gpu: "GPU Worker",
};
const quickModes = $$(".quick-select-btn");
let docs = [];
let busy = false;
let activeRun = null;
let scout = false;
let loops = 2;
let hiddenRuns = new Set();
let viewedRunId = null;
let historyRequest = 0;
let orchestration = "autonomous";
const targetStorageKey = "fusion-agent-targets-v2";

function notice(message = "") {
  byId("workspaceStatus").textContent = message;
  byId("workspaceStatus").hidden = !message;
}
function task(handler) {
  return async (...args) => {
    notice();
    try {
      await handler(...args);
    } catch (error) {
      notice(error.message);
    }
  };
}
function button(label, classes, handler) {
  const node = el("button", classes, label);
  node.type = "button";
  node.addEventListener("click", task(handler));
  return node;
}
function link(label, classes, href) {
  const node = el("a", classes, label);
  node.href = href;
  return node;
}
function syncSend() {
  const value = prompt.value.trim();
  send.disabled = busy
    ? !activeRun
    : !readCommand(value)?.command && value.length < (scout ? 3 : 8);
  send.classList.toggle("running", busy);
  send.title = busy ? "Dừng phiên nghiên cứu" : "Gửi câu hỏi";
}

function tab(id) {
  for (const node of $$(".tab-btn")) {
    const active = node.dataset.tab === id;
    node.classList.toggle("active", active);
    node.setAttribute("aria-pressed", String(active));
  }
  for (const node of $$(".tab-pane"))
    node.classList.toggle("active", node.id === id);
  syncSidecarHeader();
  if (id === "pdfViewerTab") sourceReader.show();
}
function syncSidecarHeader() {
  const sourceActive = byId("pdfViewerTab").classList.contains("active");
  const doc = sourceReader.document;
  const reportActive = byId("reportViewTab").classList.contains("active");
  byId("sidecarDocTitle").textContent = sourceActive
    ? doc?.filename || "Tài liệu gốc"
    : reportActive
      ? "Báo cáo"
      : "Trích dẫn đối soát";
  byId("downloadDocBtn").hidden = !sourceActive || !doc;
}
function panel(id, toggleId, open) {
  const node = byId(id);
  const visible = open ?? node.classList.contains("collapsed");
  node.classList.toggle("collapsed", !visible);
  byId(toggleId).classList.toggle("active", visible);
  byId(toggleId).setAttribute("aria-expanded", String(visible));
  node.inert = !visible;
}
function showTab(id) {
  panel("sidecar", "toggleSidecarBtn", true);
  tab(id);
}
const sourceReader = createSourceReader(
  () => showTab("pdfViewerTab"),
  syncSidecarHeader,
);
let sidecarRunId = null;
let sidecarDocumentIds = null;
function refreshReaderDocuments() {
  const scope = sidecarDocumentIds;
  sourceReader.setDocuments(
    scope
      ? scope.map((id) => docs.find((doc) => doc.id === id)).filter(Boolean)
      : docs,
  );
}
function preview(doc) {
  return sourceReader.open(doc);
}
function renderDocs() {
  const query = byId("docSearchInput").value.toLocaleLowerCase();
  const list = byId("docList");
  list.replaceChildren();
  for (const doc of docs.filter((d) =>
    d.filename.toLocaleLowerCase().includes(query),
  )) {
    const row = el("div", "doc-item");
    row.classList.toggle("selected", selected.has(doc.id));
    const check = el("input");
    check.type = "checkbox";
    check.checked = selected.has(doc.id);
    check.setAttribute("aria-label", "Chọn " + doc.filename);
    check.addEventListener("change", () => {
      if (check.checked) selected.add(doc.id);
      else selected.delete(doc.id);
      renderDocs();
    });
    const open = button(doc.filename, "doc-name", () => preview(doc));
    open.title = doc.filename;
    row.append(check, open);
    list.append(row);
  }
  if (!list.children.length)
    list.append(
      el(
        "div",
        "doc-empty-state",
        docs.length ? "Không tìm thấy tài liệu" : "Kho tài liệu trống",
      ),
    );
  byId("docCountBadge").textContent = docs.length;
  byId("activeScopeChip").style.display = "";
  byId("activeDocName").textContent = scout
    ? "Tìm bài trên arXiv / Crossref"
    : selected.size + " tài liệu";
  byId("selectAllDocsChk").checked =
    docs.length > 0 && selected.size === docs.length;
  byId("selectAllDocsChk").indeterminate =
    selected.size > 0 && selected.size < docs.length;
}
async function loadDocs() {
  ({ documents: docs } = await api("/documents"));
  const ids = new Set(docs.map((doc) => doc.id));
  for (const id of selected) if (!ids.has(id)) selected.delete(id);
  renderDocs();
  refreshReaderDocuments();
}
async function loadHistory() {
  const { runs } = await api("/runs");
  const list = byId("chatHistoryList");
  list.replaceChildren();
  for (const run of runs.filter((run) => !hiddenRuns.has(run.id))) {
    const entry = button("", "history-item", async () => {
      if (busy) return;
      const request = ++historyRequest;
      const record = await api("/runs/" + encodeURIComponent(run.id));
      if (busy || request !== historyRequest) return;
      resetFeed();
      selectHistoryEntry(run.id);
      appendQuestion(
        record.request.display_question || record.request.question,
      );
      const progress = timeline();
      renderEvents(progress, record);
      if (record.result) report(record);
      else if (record.error) notice(record.error);
    });
    entry.dataset.runId = run.id;
    entry.title = run.question;
    entry.append(el("span", "history-question", run.question));
    list.append(entry);
  }
  byId("historyCountBadge").textContent = list.children.length;
  if (!list.children.length)
    list.append(el("div", "doc-empty-state", "Chưa có lịch sử nghiên cứu"));
  selectHistoryEntry(viewedRunId);
}
function selectHistoryEntry(runId) {
  viewedRunId = runId;
  for (const item of $$("#chatHistoryList .history-item")) {
    const active = item.dataset.runId === runId;
    item.classList.toggle("active", active);
    if (active) item.setAttribute("aria-current", "true");
    else item.removeAttribute("aria-current");
  }
}
function resetFeed() {
  historyRequest++;
  selectHistoryEntry(null);
  feed.replaceChildren(hero);
  hero.style.display = "flex";
  sidecarRunId = sidecarDocumentIds = null;
  sourceReader.clear();
  refreshReaderDocuments();
  byId("sidecarReportPane").replaceChildren(
    el("p", "empty-sidecar", "Chưa có báo cáo trong phiên này."),
  );
  byId("sidecarEvidenceList").replaceChildren(
    el("p", "empty-sidecar", "Chưa có trích dẫn trong phiên này."),
  );
  byId("evidenceCountBadge").textContent = "";
}
function appendQuestion(value) {
  hero.style.display = "none";
  const card = el("div", "message-card user-message-card");
  const row = el("div", "user-query-row");
  row.append(el("div", "user-query-bubble", value));
  card.append(row);
  feed.append(card);
}
function timeline() {
  const card = el("div", "message-card");
  const accordion = el("details", "thinking-accordion open");
  accordion.open = true;
  const title = el("summary", "thinking-header", "Hoạt động agent");
  const body = el("div", "thinking-body thinking-timeline");
  accordion.append(title, body);
  card.append(accordion);
  feed.append(card);
  return { accordion, title, body };
}
const activityRoles = {
  orchestrator: "Điều phối",
  supervisor: "Điều phối",
  retrieval: "Tìm nguồn",
  reader: "Đọc tài liệu",
  summarizer: "Tổng hợp",
  fact_checker: "Kiểm chứng",
  writer: "Biên soạn",
};
function renderEvents(progress, record) {
  const live = ["queued", "running"].includes(record.status);
  const event = record.events.at(-1);
  const state = live && event ? event.message : recordStatusLabel(record);
  progress.title.textContent = "Hoạt động agent · " + state;
  if (!live && progress.lastStatus !== record.status)
    progress.accordion.open = false;
  progress.lastStatus = record.status;
  const follow = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 100;
  progress.body.append(
    ...record.events.slice(progress.body.children.length).map((event) => {
      const row = el("div", "workspace-activity-event");
      row.append(
        el("strong", "", activityRoles[event.agent] || event.agent),
        el("span", "", event.message),
      );
      return row;
    }),
  );
  if (follow) feed.scrollTop = feed.scrollHeight;
}
function markdown(parent, value, sources, followCitation) {
  let list;
  for (const line of value.split("\n")) {
    if (!line.trim()) {
      list = null;
      continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    const bullet = line.startsWith("- ");
    const quoted = line.startsWith("> ");
    const note = /^\*[^*]+\*$/.test(line);
    const row = el(
      heading
        ? "h" + heading[1].length
        : bullet
          ? "li"
          : quoted
            ? "blockquote"
            : "p",
    );
    if (note) row.classList.add("workspace-report-note");
    const raw = (heading ? heading[2] : bullet || quoted ? line.slice(2) : line)
      .replace(/\*\*/g, "")
      .replace(/^\*([^*]+)\*$/, "$1");
    let offset = 0;
    const citations = /\\[\\`*_{}\[\]<>()#!|]|\[\d+\]/g;
    for (const match of raw.matchAll(citations)) {
      const marker = match[0];
      if (marker.startsWith("\\") || quoted || !sources.has(marker)) continue;
      row.append(unescapeReport(raw.slice(offset, match.index)));
      const source = sources.get(marker);
      const citationButton = button(marker, "workspace-citation", () =>
        followCitation(source, marker),
      );
      citationButton.setAttribute(
        "aria-label",
        `Xem trích dẫn ${marker} trong tài liệu`,
      );
      row.append(citationButton);
      offset = match.index + marker.length;
    }
    row.append(unescapeReport(raw.slice(offset)));
    if (bullet) {
      if (!list) {
        list = el("ul");
        parent.append(list);
      }
      list.append(row);
    } else {
      list = null;
      parent.append(row);
    }
  }
}
function report(record) {
  const result = record.result;
  const card = el(
    "article",
    "message-card agent-output-block workspace-report",
  );
  const body = el("div", "workspace-markdown");
  const citations = citationIndex(result);
  const content = presentReport(result.markdown);
  function followCitation(evidence, marker) {
    renderSidecar();
    const doc = docs.find((item) => item.id === evidence.document_id);
    if (!doc) {
      showTab("evidenceTab");
      notice("Tệp gốc đã bị gỡ. Đoạn trích đã lưu vẫn có trong danh sách.");
      return;
    }
    return sourceReader.open(doc, { runId: record.id, marker, evidence });
  }

  markdown(body, content, citations, followCitation);
  function renderSidecar() {
    if (sidecarRunId === record.id) return;
    sidecarRunId = record.id;
    sidecarDocumentIds = record.request.document_ids;
    sourceReader.clear();
    refreshReaderDocuments();
    const pane = byId("sidecarReportPane");
    pane.replaceChildren();
    markdown(pane, content, citations, followCitation);
    const evidenceList = byId("sidecarEvidenceList");
    evidenceList.replaceChildren();
    for (const [marker, evidence] of citations) {
      const source = el("article", "workspace-evidence-card");
      const locator = evidence.page
        ? `Trang ${evidence.page}`
        : evidence.section;
      const open = button(
        `${marker} ${evidence.filename}${locator ? " · " + locator : ""}`,
        "workspace-evidence-link",
        () => followCitation(evidence, marker),
      );
      source.append(open, el("blockquote", "", evidence.quote));
      evidenceList.append(source);
    }
    if (!citations.size)
      evidenceList.append(
        el(
          "p",
          "empty-sidecar",
          "Câu trả lời này chưa có trích dẫn đã xác minh.",
        ),
      );
    byId("evidenceCountBadge").textContent = citations.size
      ? `${citations.size} trích dẫn`
      : "Chưa có trích dẫn";
    syncSidecarHeader();
  }
  const actions = el("div", "scout-card-actions");
  const markdownExport = link(
    "Markdown",
    "scout-action-btn",
    "/api/runs/" + encodeURIComponent(record.id) + "/export?format=md",
  );
  markdownExport.dataset.reportExport = "";
  actions.append(
    button("Mở báo cáo", "scout-action-btn", () => {
      renderSidecar();
      showTab("reportViewTab");
    }),
    button("Tài liệu gốc", "scout-action-btn", () => {
      renderSidecar();
      showTab("pdfViewerTab");
    }),
    button("Trích dẫn đối soát", "scout-action-btn", () => {
      renderSidecar();
      showTab("evidenceTab");
    }),
    markdownExport,
    link(
      "JSON",
      "scout-action-btn",
      "/api/runs/" + encodeURIComponent(record.id) + "/export?format=json",
    ),
    link(
      "Nhật ký agent",
      "scout-action-btn",
      "/api/runs/" + encodeURIComponent(record.id) + "/trace",
    ),
  );
  card.append(body, actions);
  feed.append(card);
  renderSidecar();
  requestAnimationFrame(() => {
    feed.scrollTop = feed.scrollHeight;
  });
}
async function research(question, docIds, displayQuestion = question) {
  const progress = timeline();
  let record = await api("/runs", {
    method: "POST",
    body: {
      question,
      ...(displayQuestion !== question && {
        display_question: displayQuestion,
      }),
      document_ids: docIds,
      scope_mode: "selected",
      orchestration_mode: orchestration,
      max_rounds: loops,
      agent_targets: Object.fromEntries(
        modelSelects.map((node) => [node.dataset.agent, node.value]),
      ),
    },
  });
  activeRun = record.id;
  selectHistoryEntry(record.id);
  syncSend();
  for (;;) {
    renderEvents(progress, record);
    if (!["queued", "running"].includes(record.status)) break;
    await new Promise((resolve) => setTimeout(resolve, 500));
    record = await api("/runs/" + encodeURIComponent(activeRun));
  }
  if (record.result) report(record);
  if (record.error) notice(record.error);
  await loadHistory();
}
async function discover(question) {
  const list = el("div", "agent-output-block scout-results-block");
  list.append(el("h3", "scout-banner-title", "Bài báo trên arXiv / Crossref"));
  feed.append(list);
  const { papers, warnings } = await api("/scout/search", {
    method: "POST",
    body: { query: question.slice(0, 300) },
  });
  for (const warning of warnings) list.append(el("p", "", warning));
  if (!papers.length) list.append(el("p", "", "Không tìm thấy bài phù hợp."));
  for (const paper of papers) {
    const card = el("article", "scout-paper-card");
    const importButton = button(
      "Nhập PDF và phân tích",
      "scout-action-btn primary",
      async () => {
        if (busy) return;
        busy = true;
        syncSend();
        try {
          const { document: doc } = await api("/papers/import", {
            method: "POST",
            body: { paper_id: paper.arxiv_id },
          });
          await loadDocs();
          await research(question, [doc.id]);
        } finally {
          activeRun = null;
          busy = false;
          syncSend();
        }
      },
    );
    card.append(
      el("h4", "scout-card-title", paper.title),
      el("p", "", paper.authors.join(", ")),
      paper.provider === "arxiv"
        ? importButton
        : link("Mở bài qua DOI", "scout-action-btn", paper.url),
    );
    list.append(card);
  }
}
function setScout(enabled) {
  scout = enabled;
  const toggle = byId("autoScoutToggleBtn");
  toggle.classList.toggle("active", scout);
  toggle.setAttribute("aria-pressed", String(scout));
  renderDocs();
  syncSend();
}
const commandMenu = mountCommandMenu({
  input: prompt,
  anchor: $(".input-dock-mono"),
  preferAbove: true,
  onResearch: () => {
    if (scout) setScout(false);
  },
  onAction: task((action) => {
    if (action === "papers") setScout(true);
    else if (action === "export") {
      const latest = $$(".workspace-report [data-report-export]", feed).at(-1);
      if (latest) latest.click();
      else notice("Chưa có báo cáo để xuất.");
    } else if (!targets.get(action)?.configured) {
      notice(
        targetLabels[action] +
          " chưa được cấu hình. Mở menu model để thiết lập.",
      );
    } else {
      setPreset(action);
      notice("Đã chọn " + targetLabels[action] + " cho cả năm agent.");
    }
  }),
});
async function submit() {
  if (busy) {
    if (activeRun)
      await api("/runs/" + encodeURIComponent(activeRun) + "/cancel", {
        method: "POST",
      });
    return;
  }
  const value = prompt.value.trim();
  const parsed = readCommand(value);
  const searchCommand = parsed?.command?.action === "papers";
  if ((!searchCommand || !parsed.argument) && commandMenu.runAction(parsed))
    return;
  if (parsed && !parsed.command) {
    notice("Không tìm thấy lệnh. Gõ / để xem danh sách.");
    return;
  }
  const searching = searchCommand || (scout && !parsed?.command);
  const question = searchCommand ? parsed.argument : expandQuestion(value);
  if (
    question.length < (searching ? 3 : 8) ||
    question.length > (searching ? 300 : 2000)
  ) {
    notice(
      searching
        ? "Từ khóa tìm bài cần từ 3 đến 300 ký tự."
        : "Câu hỏi sau khi mở rộng lệnh cần từ 8 đến 2000 ký tự.",
    );
    return;
  }
  if (!searching && !docs.length) {
    notice("Tải lên tài liệu hoặc nạp tài liệu mẫu trước khi nghiên cứu.");
    return;
  }
  if (!searching && !selected.size) {
    notice("Chọn ít nhất một tài liệu trong kho trước khi gửi câu hỏi.");
    return;
  }
  if (searching !== scout) setScout(searching);
  prompt.value = "";
  commandMenu.close();
  historyRequest++;
  appendQuestion(searchCommand ? "/tim-bai " + question : value);
  busy = true;
  syncSend();
  try {
    if (searching) await discover(question);
    else await research(question, [...selected], value);
  } finally {
    activeRun = null;
    busy = false;
    syncSend();
  }
}
function setPreset(target) {
  for (const node of modelSelects) node.value = target;
  syncModels();
}
function hybridTarget(role) {
  return ["reader", "fact_checker"].includes(role) ? "cpu" : "api";
}

function syncModels() {
  try {
    localStorage.setItem(
      targetStorageKey,
      JSON.stringify(
        Object.fromEntries(
          modelSelects.map((node) => [node.dataset.agent, node.value]),
        ),
      ),
    );
  } catch {
    /* Current choices still apply if storage is unavailable. */
  }
  const values = new Set(modelSelects.map((node) => node.value));
  const hybrid = modelSelects.every(
    (node) => node.value === hybridTarget(node.dataset.agent),
  );
  const label =
    values.size === 1
      ? targetLabels[values.values().next().value]
      : hybrid
        ? "Hybrid"
        : "Tùy biến";
  byId("activeModelLabel").textContent = label;
  byId("quickActivePill").textContent = label;
  for (const node of quickModes) {
    const mode = node.dataset.mode;
    const active =
      mode === "hybrid"
        ? hybrid
        : values.size === 1 && values.has(modeTargets[mode]);
    node.classList.toggle("active", active);
    node.setAttribute("aria-pressed", String(active));
  }
}
async function loadConfig() {
  const config = await api("/config");
  targets.clear();
  orchestration = config.orchestration_mode || "autonomous";
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(targetStorageKey) || "{}");
  } catch {
    /* Use server defaults. */
  }
  for (const target of config.execution_targets) targets.set(target.id, target);
  for (const node of modelSelects) {
    node.replaceChildren();
    for (const target of targets.values()) {
      const option = el(
        "option",
        "",
        targetLabels[target.id] + (target.model ? " · " + target.model : ""),
      );
      option.value = target.id;
      option.disabled = !target.configured;
      node.append(option);
    }
  }
  for (const node of modelSelects) {
    const value = saved?.[node.dataset.agent];
    node.value = targets.get(value)?.configured ? value : "default";
  }
  syncModels();
  for (const node of quickModes) {
    const mode = node.dataset.mode;
    const missing = missingTarget(mode);
    node.classList.toggle("needs-setup", !!missing);
    node.title = missing
      ? "Cấu hình " +
        targetLabels[missing] +
        (mode === "hybrid" ? " để dùng Hybrid" : "")
      : mode === "hybrid"
        ? "Reader và Fact Checker dùng CPU Max, các agent còn lại dùng Cloud API"
        : "Chọn " + targetLabels[modeTargets[mode]] + " cho cả năm agent";
  }
  for (const name of ["api", "cpu", "gpu"]) {
    const target = targets.get(name);
    const badge = byId(name + "TargetStatus");
    badge.textContent = target?.configured ? target.model : "Chưa cấu hình";
    badge.classList.toggle("configured", !!target?.configured);
  }
}
function missingTarget(mode) {
  const required = mode === "hybrid" ? ["api", "cpu"] : [modeTargets[mode]];
  return required.find((name) => !targets.get(name)?.configured);
}

const settingsPanel = byId("modelSettingsPanel");
const settingsRoot = byId("modelSettingsForm");
const settingsStatus = byId("modelSettingsStatus");
const embeddingPanel = byId("workspaceGpuSettings");
const settingsTabs = $$("[data-config-target]");
const settingsClose = byId("closeModelSettings");
const matrix = byId("agentMatrixPopover");
let settingsRequest = 0;

function showSettingsStatus(message = "") {
  settingsStatus.textContent = message;
  settingsStatus.hidden = !message;
}

function renderTargetEditor(name, saved, request) {
  if (request !== settingsRequest || settingsPanel.hidden) return;
  if (saved?.managed) {
    settingsRoot.replaceChildren(
      el(
        "p",
        "",
        "Cấu hình này do máy chủ quản lý. Đổi model hoặc endpoint trong cấu hình khởi động.",
      ),
    );
    return;
  }
  settingsRoot.replaceChildren(
    createTargetForm(name, saved, {
      showProbe: true,
      async onSave(result) {
        await loadConfig();
        if (request !== settingsRequest || settingsPanel.hidden) return;
        renderTargetEditor(
          name,
          result.targets.find((target) => target.id === name),
          request,
        );
        showSettingsStatus(
          "Đã lưu. Kiểm tra kết nối hoặc chọn model cho agent bên dưới.",
        );
      },
    }),
  );
}

async function openModelSettings(name) {
  const request = ++settingsRequest;
  settingsPanel.hidden = false;
  settingsClose.hidden = false;
  settingsRoot.replaceChildren();
  clearGpuKey();
  embeddingPanel.hidden = name !== "embedding";
  for (const button of settingsTabs)
    button.setAttribute(
      "aria-expanded",
      String(button.dataset.configTarget === name),
    );
  if (name === "embedding") {
    showSettingsStatus();
    embeddingPanel.scrollIntoView?.({ block: "nearest" });
    await loadGpuSettings();
    return;
  }
  showSettingsStatus("Đang tải cấu hình…");
  try {
    const { targets: saved } = await api("/config/targets");
    if (request !== settingsRequest || settingsPanel.hidden) return;
    showSettingsStatus();
    renderTargetEditor(
      name,
      saved.find((target) => target.id === name),
      request,
    );
    settingsPanel.scrollIntoView?.({ block: "nearest" });
  } catch (error) {
    if (request === settingsRequest && !settingsPanel.hidden)
      showSettingsStatus(error.message);
  }
}
function closeModelSettings() {
  settingsRequest++;
  settingsPanel.hidden = true;
  settingsClose.hidden = true;
  embeddingPanel.hidden = true;
  clearGpuKey();
  for (const button of settingsTabs)
    button.setAttribute("aria-expanded", "false");
  settingsRoot.replaceChildren();
  showSettingsStatus();
}
click("closeModelSettings", () => {
  const active = settingsTabs.find(
    (button) => button.getAttribute("aria-expanded") === "true",
  );
  closeModelSettings();
  active?.focus();
});
for (const button of settingsTabs)
  button.addEventListener("click", () => {
    if (button.getAttribute("aria-expanded") === "true") closeModelSettings();
    else openModelSettings(button.dataset.configTarget);
  });

function setMatrixOpen(open) {
  matrix.classList.toggle("open", open);
  byId("modelDropdownBtn").setAttribute("aria-expanded", String(open));
  if (!open) closeModelSettings();
}

initializeGpuSettings(loadDocs, () => busy, notice);

function showLogin() {
  if (!byId("workspaceLogin").open) byId("workspaceLogin").showModal();
}
async function initialize() {
  if (!(await api("/auth/status")).authenticated) {
    showLogin();
    return;
  }
  await Promise.all([loadConfig(), loadDocs(), loadHistory()]);
}
window.addEventListener("authentication-required", showLogin);
byId("workspaceLoginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/auth/login", {
      method: "POST",
      body: { token: byId("workspaceAccessToken").value },
    });
    byId("workspaceAccessToken").value = "";
    byId("workspaceLoginError").textContent = "";
    byId("workspaceLogin").close();
    await task(initialize)();
  } catch (error) {
    byId("workspaceLoginError").textContent = error.message;
  }
});
prompt.addEventListener("input", syncSend);
prompt.addEventListener("keydown", (event) => {
  if (
    !event.defaultPrevented &&
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.isComposing &&
    !busy
  ) {
    event.preventDefault();
    task(submit)();
  }
});
for (const node of $$(".suggestion-card"))
  node.addEventListener("click", () => {
    prompt.value = node.dataset.query;
    prompt.dispatchEvent(new Event("input"));
    prompt.focus();
  });
for (const node of $$(".tab-btn"))
  node.addEventListener("click", () => tab(node.dataset.tab));
for (const node of modelSelects) node.addEventListener("change", syncModels);
for (const node of quickModes)
  node.addEventListener("click", () => {
    const missing = missingTarget(node.dataset.mode);
    if (missing) {
      openModelSettings(missing);
    } else if (node.dataset.mode === "hybrid") {
      for (const select of modelSelects)
        select.value = hybridTarget(select.dataset.agent);
      syncModels();
    } else setPreset(modeTargets[node.dataset.mode]);
  });
for (const node of $$(".loop-select-btn")) {
  node.addEventListener("click", () => {
    loops = Number(node.dataset.loops);
    for (const item of $$(".loop-select-btn"))
      item.classList.toggle("active", item === node);
    byId("reflectionLoopsStatus").textContent = loops + " vòng";
  });
}
click("sendBtn", task(submit));
click("newChatBtn", () => {
  if (!busy) {
    resetFeed();
    prompt.focus();
  }
});
click("toggleSidebarBtn", () => panel("sidebar", "toggleSidebarBtn"));
click("toggleSidecarBtn", () => {
  panel("sidecar", "toggleSidecarBtn");
  if (
    !byId("sidecar").inert &&
    byId("pdfViewerTab").classList.contains("active")
  )
    sourceReader.show();
});
click("closeSidecarBtn", () => panel("sidecar", "toggleSidecarBtn", false));
click("autoScoutToggleBtn", () => setScout(!scout));
click("modelDropdownBtn", () =>
  setMatrixOpen(!matrix.classList.contains("open")),
);
click("closePopoverBtn", () => {
  setMatrixOpen(false);
  byId("modelDropdownBtn").focus();
});
click("autoSelectBestBtn", () => setPreset("default"));
click("uploadPdfBtn", () => byId("pdfFileInput").click());
byId("pdfFileInput").accept = ".pdf,.txt,.md,.docx";
byId("pdfFileInput").addEventListener(
  "change",
  task(async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    event.target.value = "";
    notice("Đang nạp tài liệu…");
    const { document: doc } = await uploadDocument(file, {
      onProgress: ({ percent }) =>
        notice("Đang tải tài liệu… " + percent + "%"),
      onTransferred: () => notice("Đang trích xuất và lập chỉ mục tài liệu…"),
    });
    selected.add(doc.id);
    await loadDocs();
    notice();
  }),
);
byId("docSearchInput").addEventListener("input", renderDocs);
click("docMoreBtn", () => byId("docDropdownMenu").classList.toggle("open"));
click("historyMoreBtn", () =>
  byId("historyDropdownMenu").classList.toggle("open"),
);
click("toggleDocSelectModeBtn", () => {
  byId("docBulkBar").style.display = "flex";
  byId("docDropdownMenu").classList.remove("open");
});
byId("selectAllDocsChk").addEventListener("change", (event) => {
  selected.clear();
  if (event.target.checked) for (const doc of docs) selected.add(doc.id);
  renderDocs();
});
click("cancelBulkDelDocsBtn", () => {
  byId("docBulkBar").style.display = "none";
});
const deleteSelected = task(async () => {
  if (!selected.size || !window.confirm("Xóa các tài liệu đã chọn khỏi kho?"))
    return;
  try {
    for (const id of [...selected])
      await api("/documents/" + encodeURIComponent(id), { method: "DELETE" });
  } finally {
    await loadDocs();
  }
});
click("deleteDocsBtn", deleteSelected);
click("confirmBulkDelDocsBtn", deleteSelected);
byId("clearAllHistoryBtn").textContent = "Ẩn lịch sử trong phiên này";
click(
  "clearAllHistoryBtn",
  task(async () => {
    const { runs } = await api("/runs");
    hiddenRuns = new Set(runs.map((run) => run.id));
    byId("historyDropdownMenu").classList.remove("open");
    await loadHistory();
  }),
);
byId("docDropdownMenu").append(
  button("Nạp tài liệu mẫu", "dropdown-item", async () => {
    await api("/documents/demo", { method: "POST" });
    byId("docDropdownMenu").classList.remove("open");
    await loadDocs();
  }),
);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || event.defaultPrevented || event.isComposing)
    return;
  if (matrix.classList.contains("open")) {
    setMatrixOpen(false);
    byId("modelDropdownBtn").focus();
  } else panel("sidecar", "toggleSidecarBtn", false);
});
task(initialize)();

if (window.matchMedia?.("(max-width: 760px)").matches) {
  panel("sidebar", "toggleSidebarBtn", false);
}
panel("sidecar", "toggleSidecarBtn", false);
