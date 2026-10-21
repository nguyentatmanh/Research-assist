import { api, uploadDocument } from "./api.js";
import { $, $$, el, icon, dateLabel, statuses, toast } from "./dom.js";
import { renderEvents, renderResult, selectTab, copyReport } from "./report.js";
import { renderBenchmark, renderRoutes } from "./lab.js";

const state = {
  documents: [],
  selected: new Set(),
  scopeMode: "selected",
  config: null,
  runId: null,
  working: false,
  uploading: false,
  pollVersion: 0,
};

async function initialize() {
  try {
    const auth = await api("/auth/status");
    if (auth.required && !auth.authenticated) {
      showLogin();
      return;
    }
    const [config, documents] = await Promise.all([
      api("/config"),
      api("/documents"),
    ]);
    state.config = config;
    state.documents = documents.documents;
    renderConfig();
    $("#workflow-profile").value = config.workflow_profile || "balanced";
    renderDocuments();
    $("#connection-label").textContent = "Backend đã kết nối";
    const saved = localStorage.getItem("research-active-run");
    if (saved) {
      try {
        await openRun(saved);
      } catch {
        localStorage.removeItem("research-active-run");
      }
    }
  } catch (error) {
    $("#connection-label").textContent = "Chưa kết nối backend";
    $("#provider-pill").textContent = "Mất kết nối";
    $("#provider-pill").classList.add("disconnected");
    toast(error.message || "Không kết nối được backend.", true);
  }
}

function renderConfig() {
  const config = state.config;
  const pill = $("#provider-pill");
  pill.replaceChildren(
    el("span", "status-dot"),
    document.createTextNode(
      config.demo ? "Demo · không cần key" : config.model,
    ),
  );
  pill.classList.toggle("demo", config.demo);
  pill.classList.remove("disconnected");
  $("#mode-notice").hidden = !config.demo;
  $("#logout-button").hidden = !config.auth_required;
  const summary = $("#settings-summary");
  summary.replaceChildren();
  const values = [
    ["Provider", config.provider],
    ["Model", config.model],
    ["Embedding", config.embedding_model || "Chưa bật · dùng BM25"],
    ["Reranker", config.reranker_model || "Chưa bật"],
  ];
  for (const [label, value] of values) {
    const group = el("dl");
    group.append(el("dt", "", label), el("dd", "", value));
    summary.append(group);
  }
  renderRoutes(config.agent_routes || []);
}

function renderDocuments() {
  state.selected = new Set(
    [...state.selected].filter((id) =>
      state.documents.some((d) => d.id === id),
    ),
  );
  $("#nav-doc-count").textContent = String(state.documents.length);
  $("#selected-doc-count").textContent =
    `${state.scopeMode === "all" ? state.documents.length : state.selected.size}/${state.documents.length}`;
  $("#library-total").textContent = String(state.documents.length);
  $("#library-chunks").textContent = state.documents
    .reduce((sum, doc) => sum + doc.chunk_count, 0)
    .toLocaleString("vi-VN");
  $("#getting-started-title").textContent = state.documents.length
    ? "Tài liệu đã sẵn sàng. Bắt đầu khám phá thôi."
    : "Bắt đầu từ tài liệu bạn tin cậy";
  $("#getting-started-note").textContent = state.documents.length
    ? "Chọn nguồn, nhập câu hỏi hoặc dùng một gợi ý ở bên cạnh."
    : "Tải tài liệu của bạn hoặc khám phá bằng bộ ghi chú mẫu.";
  const sidebar = $("#sidebar-documents");
  sidebar.replaceChildren();
  if (!state.documents.length)
    sidebar.append(
      el(
        "p",
        "sidebar-empty",
        "Chưa có tài liệu.\nTải tệp hoặc thử bộ ghi chú mẫu.",
      ),
    );
  for (const document of state.documents) {
    const row = el("label", "sidebar-document");
    const input = el("input");
    input.type = "checkbox";
    input.checked = state.selected.has(document.id);
    input.disabled = state.working;
    input.setAttribute("aria-label", `Chọn ${document.filename}`);
    input.addEventListener("change", () => {
      input.checked
        ? state.selected.add(document.id)
        : state.selected.delete(document.id);
      state.scopeMode = "selected";
      renderDocuments();
    });
    const text = el("div");
    text.append(
      el("strong", "", document.filename),
      el(
        "small",
        "",
        `${document.kind.toUpperCase()} · ${document.chunk_count} đoạn`,
      ),
    );
    row.append(input, text);
    sidebar.append(row);
  }
  renderLibrary();
  renderScopePicker();
  renderScope();
}

function renderLibrary() {
  const body = $("#document-table-body");
  body.replaceChildren();
  const query = $("#library-search").value.trim().toLocaleLowerCase("vi");
  const documents = state.documents.filter((doc) =>
    `${doc.title} ${doc.filename}`.toLocaleLowerCase("vi").includes(query),
  );
  if (!documents.length) {
    const row = el("tr");
    const cell = el(
      "td",
      "table-empty",
      query
        ? "Không tìm thấy tài liệu phù hợp. Thử một từ khóa khác."
        : "Kho tri thức của bạn đang chờ. Tải tài liệu hoặc nạp bộ ghi chú mẫu để bắt đầu.",
    );
    cell.colSpan = 5;
    row.append(cell);
    body.append(row);
  }
  for (const document of documents) {
    const row = el("tr");
    const first = el("td");
    first.append(
      el("strong", "", document.title),
      el("small", "", document.filename),
    );
    const kind = el("td");
    kind.append(el("span", "doc-type", document.kind.toUpperCase()));
    row.append(
      first,
      kind,
      el("td", "", document.page_count || "—"),
      el("td", "", document.chunk_count),
    );
    const actions = el("td");
    const container = el("div", "document-actions");
    const open = el("a", "icon-button");
    open.href = `/api/documents/${document.id}/file`;
    open.target = "_blank";
    open.rel = "noopener";
    open.setAttribute("aria-label", `Mở ${document.filename}`);
    open.append(icon("external"));
    const remove = el("button", "icon-button");
    remove.setAttribute("aria-label", `Xóa ${document.filename}`);
    remove.append(icon("trash"));
    remove.addEventListener("click", () => deleteDocument(document));
    const use = el("button", "text-button", "Hỏi tài liệu này");
    use.disabled = state.working;
    use.addEventListener("click", () => {
      state.selected = new Set([document.id]);
      state.scopeMode = "selected";
      renderDocuments();
      switchView("research");
      $("#question").focus();
    });
    container.append(use, open, remove);
    actions.append(container);
    row.append(actions);
    body.append(row);
  }
}

function renderScope() {
  const selectedCount =
    state.scopeMode === "all" ? state.documents.length : state.selected.size;
  $("#library-selected").textContent = String(selectedCount);
  $("#selected-doc-count").textContent =
    `${selectedCount}/${state.documents.length}`;
  $("#scope-mode").value = state.scopeMode;
  $("#scope-label").textContent =
    state.scopeMode === "all"
      ? `Toàn bộ ${state.documents.length} tài liệu`
      : `${state.selected.size} tài liệu được chọn`;
  const names = state.documents.filter(
    (d) => state.scopeMode === "all" || state.selected.has(d.id),
  );
  $("#scope-filenames").textContent = names.length
    ? names.map((d) => d.filename).join(" · ")
    : "Chưa chọn tài liệu";
  updateResearchSteps();
}

function updateResearchSteps() {
  const hasSources =
    state.scopeMode === "all"
      ? state.documents.length > 0
      : state.selected.size > 0;
  const hasQuestion = $("#question").value.trim().length >= 8;
  const hasReport = !$("#results-section").hidden;
  const steps = [$("#step-sources"), $("#step-question"), $("#step-report")];
  steps.forEach((step, index) => {
    const complete =
      index === 0
        ? hasSources
        : index === 1
          ? hasSources && hasQuestion
          : hasReport;
    const current =
      !hasReport &&
      (index === 0
        ? !hasSources
        : index === 1
          ? hasSources && !hasQuestion
          : hasSources && hasQuestion);
    step.classList.toggle("complete", complete);
    step.classList.toggle("current", current);
    if (current) step.setAttribute("aria-current", "step");
    else step.removeAttribute("aria-current");
  });
}

async function refreshDocuments() {
  state.documents = (await api("/documents")).documents;
  renderDocuments();
}

function setWorking(value) {
  state.working = value;
  $("#run-button").disabled = value;
  $("#cancel-button").hidden = !value;
  $("#question").disabled = value;
  $("#suggestion-area").hidden = value;
  $$(".prompt-chip").forEach((button) => {
    button.disabled = value;
  });
  $("#activity-cancel").hidden = !value;
  $("#scope-mode").disabled = value;
  $("#choose-sources").disabled = value;
  $("#requirements").disabled = value;
  renderDocuments();
  $$("#advanced-panel select, #advanced-panel input").forEach((select) => {
    select.disabled = value;
  });
}

async function startResearch() {
  if (state.working) return;
  const question = $("#question").value.trim();
  if (question.length < 8) {
    toast("Câu hỏi cần ít nhất 8 ký tự.", true);
    $("#question").focus();
    return;
  }
  if (!state.documents.length) {
    toast("Tải tài liệu hoặc nạp bộ mẫu trước khi nghiên cứu.", true);
    return;
  }
  if (state.uploading) {
    toast("Hãy chờ tài liệu được lập chỉ mục.");
    return;
  }
  if (state.scopeMode === "selected" && !state.selected.size) {
    $("#scope-dialog").showModal();
    toast("Chọn ít nhất một tài liệu.", true);
    return;
  }
  const requirements = $("#requirements")
    .value.split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (requirements.length > 12 || requirements.some((s) => s.length > 500)) {
    toast("Tối đa 12 yêu cầu, mỗi yêu cầu tối đa 500 ký tự.", true);
    return;
  }
  setWorking(true);
  $("#results-section").hidden = true;
  $("#empty-state").hidden = true;
  renderEvents([], "queued");
  try {
    const record = await api("/runs", {
      method: "POST",
      body: {
        question,
        requirements: $("#requirements")
          .value.split(/\r?\n/)
          .map((s) => s.trim())
          .filter(Boolean),
        document_ids: state.scopeMode === "all" ? [] : [...state.selected],
        scope_mode: state.scopeMode,
        language: $("#language").value,
        style: $("#report-style").value,
        retrieval_mode: $("#retrieval-mode").value,
        top_k: Number($("#top-k").value),
        max_rounds: Number($("#max-rounds").value),
        profile: $("#workflow-profile").value,
        use_cache: $("#use-cache").checked,
      },
    });
    state.runId = record.id;
    switchView("activity");
    localStorage.setItem("research-active-run", record.id);
    await pollRun(record.id);
  } catch (error) {
    toast(error.message, true);
    setWorking(false);
  }
}

async function pollRun(runId) {
  const version = ++state.pollVersion;
  let failures = 0;
  setWorking(true);
  while (version === state.pollVersion) {
    let record;
    try {
      record = await api(`/runs/${runId}`);
      failures = 0;
    } catch (error) {
      failures++;
      if (failures >= 3 || error.status === 404 || error.status === 401) {
        setWorking(false);
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 1300));
      continue;
    }
    if (version !== state.pollVersion) return;
    activityRecord(record);
    renderEvents(record.events, record.status);
    if (["completed", "failed", "cancelled"].includes(record.status)) {
      setWorking(false);
      if (record.result) {
        renderResult(record.result);
      } else {
        toast(
          record.error || statuses[record.status],
          record.status === "failed",
        );
        $("#empty-state").hidden = false;
      }
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 600));
  }
}

async function openRun(runId) {
  if (state.working && state.runId !== runId) {
    toast("Hãy hoàn thành hoặc dừng phiên hiện tại trước.");
    return;
  }
  const record = await api(`/runs/${runId}`);
  switchView("research");
  state.runId = runId;
  $("#question").value = record.request.question;
  $("#requirements").value = (record.request.requirements || []).join("\n");
  $("#question-count").textContent = `${record.request.question.length} / 2000`;
  $("#language").value = record.request.language;
  $("#report-style").value = record.request.style;
  $("#retrieval-mode").value = record.request.retrieval_mode;
  $("#top-k").value = String(record.request.top_k);
  $("#max-rounds").value = String(record.request.max_rounds);
  $("#workflow-profile").value =
    record.request.profile || state.config.workflow_profile || "academic";
  $("#use-cache").checked = record.request.use_cache !== false;
  state.selected = new Set(record.request.document_ids);
  state.scopeMode = state.selected.size ? "selected" : "all";
  activityRecord(record);
  renderDocuments();
  localStorage.setItem("research-active-run", runId);
  if (record.result) {
    renderResult(record.result);
    setWorking(false);
  } else if (["queued", "running"].includes(record.status))
    await pollRun(runId);
  else {
    renderEvents(record.events, record.status);
    $("#results-section").hidden = true;
    $("#empty-state").hidden = false;
    toast(record.error || statuses[record.status], record.status === "failed");
  }
}

async function cancelResearch() {
  if (!state.runId || !state.working) return;
  $("#cancel-button").disabled = true;
  try {
    await api(`/runs/${state.runId}/cancel`, { method: "POST" });
  } catch (error) {
    toast(error.message, true);
  } finally {
    $("#cancel-button").disabled = false;
  }
}

let currentUpload = null;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const unit = bytes < 1024 * 1024 ? "KB" : "MB";
  const size = unit === "KB" ? bytes / 1024 : bytes / (1024 * 1024);
  return `${new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 }).format(size)} ${unit}`;
}

function formatDuration(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

function formatEmbeddingRate(rate) {
  const amount = rate < 1 ? rate * 60 : rate;
  const unit = rate < 1 ? "đoạn/phút" : "đoạn/giây";
  return `${new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 }).format(amount)} ${unit}`;
}

function newUploadId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = [...bytes]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function renderUploadProgress(view) {
  if (!view) return;
  const panel = $("#upload-status");
  panel.hidden = false;
  panel.dataset.phase = view.phase;
  $("#upload-title").textContent =
    view.phase === "uploading"
      ? "Đang tải tài liệu"
      : view.phase === "processing"
        ? "Đang xử lý tài liệu"
        : view.phase === "error"
          ? "Tải tài liệu chưa hoàn tất"
          : "Đã hoàn tất";
  $("#upload-count").textContent = `${view.index}/${view.total}`;
  $("#upload-filename").textContent = view.file?.name || "";
  $("#upload-size").textContent = view.file
    ? formatBytes(view.file.size)
    : `${view.total} tệp · ${formatBytes(view.batchBytes)}`;
  const server = view.server || {};
  const facts = [];
  if (server.pages) facts.push(`${server.pages} trang PDF`);
  if (server.text_chars)
    facts.push(`${server.text_chars.toLocaleString("vi-VN")} ký tự`);
  if (server.chunks) facts.push(`${server.chunks} đoạn`);
  $("#upload-input-details").textContent = facts.length
    ? facts.join(" · ")
    : view.phase === "uploading"
      ? "Đang nhận tài liệu"
      : view.phase === "processing"
        ? "Đang phân tích đầu vào"
        : "Đã xử lý danh sách tệp";

  const progress = $("#upload-progress");
  progress.parentElement.hidden =
    view.phase === "complete" || view.phase === "error";
  if (view.phase === "uploading") {
    progress.value = view.percent;
    progress.setAttribute("aria-label", "Tiến độ truyền tài liệu");
    $("#upload-percent").textContent = `${view.percent}%`;
    panel.dataset.progressMode = "determinate";
  } else if (
    view.phase === "processing" &&
    server.embedding_percent !== null &&
    server.embedding_percent !== undefined
  ) {
    progress.value = server.embedding_percent;
    progress.setAttribute(
      "aria-label",
      "Tiến độ tạo vector cho các đoạn văn bản",
    );
    $("#upload-percent").textContent = `${server.embedding_percent}%`;
    panel.dataset.progressMode = "determinate";
  } else if (view.phase === "processing") {
    progress.removeAttribute("value");
    $("#upload-percent").textContent = "";
    panel.dataset.progressMode = "indeterminate";
  } else {
    $("#upload-percent").textContent = "";
    panel.dataset.progressMode = "none";
  }
  const stageText = {
    reading: "Đang đọc tệp…",
    waiting: "Đang chờ lượt xử lý tài liệu…",
    extracting: "Đang đọc tài liệu và trích xuất văn bản…",
    chunking: "Đang chia văn bản thành các đoạn…",
    embedding: server.model_loading
      ? "Đang nạp mô hình embedding lần đầu. Thời gian có thể lâu hơn…"
      : server.chunks
        ? `Đang tạo vector ${server.embedded || 0}/${server.chunks} đoạn…`
        : "Đang tạo vector cho tài liệu…",
    saving: "Đang lưu tài liệu và chỉ mục…",
    completed: "Đã lập chỉ mục. Đang nhận kết quả…",
  };
  $("#upload-message").textContent =
    view.message ||
    (view.phase === "uploading"
      ? "Đang truyền tệp lên máy chủ…"
      : stageText[server.stage] || "Đã truyền xong. Đang phân tích tài liệu…");
  $("#upload-transferred").textContent =
    `${formatBytes(view.bytes)} / ${formatBytes(view.file?.size ?? view.batchBytes)}`;
  $("#upload-work").textContent = server.chunks
    ? server.stage === "embedding"
      ? `${server.embedded || 0}/${server.chunks} đoạn`
      : `${server.chunks} đoạn đã tạo`
    : server.pages
      ? `${server.pages} trang đã đọc`
      : view.work || "Chưa xác định";
  const elapsed = Math.max(
    ((view.finishedAt || Date.now()) - view.startedAt) / 1000,
    server.elapsed_seconds || 0,
  );
  $("#upload-elapsed").textContent = formatDuration(elapsed);
  $("#upload-speed").textContent =
    view.phase === "uploading" && view.bytes > 0 && elapsed > 0.5
      ? `Tốc độ truyền: ${formatBytes(view.bytes / elapsed)}/giây`
      : view.phase === "processing" && server.embedding_rate
        ? `Tốc độ lập chỉ mục: ${formatEmbeddingRate(server.embedding_rate)}`
        : "";
  let remaining = "Đang tính";
  if (view.phase === "uploading" && view.bytes > 0 && elapsed > 0.5) {
    const seconds = ((view.file.size - view.bytes) * elapsed) / view.bytes;
    remaining = seconds < 1 ? "< 1 giây" : `~${formatDuration(seconds)}`;
  } else if (
    view.phase === "processing" &&
    server.remaining_seconds !== null &&
    server.remaining_seconds !== undefined
  ) {
    remaining =
      server.remaining_seconds < 1
        ? "< 1 giây"
        : `~${formatDuration(server.remaining_seconds)}`;
  } else if (view.phase === "complete" || view.phase === "error") {
    remaining = "—";
  }
  $("#upload-remaining").textContent = remaining;
  const step =
    view.phase === "uploading"
      ? 0
      : view.phase === "error" && !server.stage
        ? 0
        : ["reading", "waiting", "extracting"].includes(server.stage)
          ? 1
          : server.stage === "chunking"
            ? 2
            : 3;
  ["transfer", "extract", "chunk", "index"].forEach((name, index) => {
    const item = $(`#upload-step-${name}`);
    item.dataset.state =
      view.phase === "complete"
        ? "done"
        : view.phase === "error" && index === step
          ? "error"
          : index < step
            ? "done"
            : index === step
              ? "active"
              : "pending";
  });
  $("#upload-close").hidden =
    view.phase === "uploading" || view.phase === "processing";
}

function monitorUpload(uploadId, view) {
  let stopped = false;
  let timer;
  async function poll() {
    try {
      const progress = await api(`/uploads/${uploadId}`);
      if (!stopped && currentUpload === view) {
        view.server = progress;
        renderUploadProgress(view);
      }
    } catch (error) {
      if (error.status === 401) stopped = true;
      // The first poll can arrive before the upload handler registers its ID.
    }
    if (!stopped) timer = setTimeout(poll, 600);
  }
  poll();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}

function setUploadBusy(value) {
  state.uploading = value;
  for (const id of [
    "sidebar-upload",
    "library-upload",
    "dropzone",
    "seed-button",
    "library-seed",
  ]) {
    $(`#${id}`).disabled = value;
  }
}

async function uploadFiles(files) {
  if (!files.length) return;
  if (state.uploading) {
    toast("Đang tải tài liệu. Hãy chờ thao tác hiện tại hoàn thành.");
    return;
  }
  setUploadBusy(true);
  const accepted = ["pdf", "docx", "txt", "md"];
  let added = 0;
  let duplicates = 0;
  let failed = 0;
  let lastError = "";
  let sentBytes = 0;
  const batchBytes = files.reduce((sum, file) => sum + file.size, 0);
  const batchStartedAt = Date.now();
  const ticker = setInterval(() => renderUploadProgress(currentUpload), 1000);
  $("#connection-label").textContent = "Đang tải tài liệu…";
  try {
    for (const [offset, file] of files.entries()) {
      const view = {
        file,
        index: offset + 1,
        total: files.length,
        batchBytes,
        startedAt: Date.now(),
        bytes: 0,
        percent: 0,
        phase: "uploading",
        server: null,
      };
      currentUpload = view;
      renderUploadProgress(view);
      $("#connection-label").textContent = "Đang tải tài liệu…";
      if (!accepted.includes(file.name.split(".").at(-1).toLowerCase())) {
        failed++;
        lastError = `Không hỗ trợ ${file.name}.`;
        view.phase = "error";
        view.message = lastError;
        view.finishedAt = Date.now();
        renderUploadProgress(view);
        toast(lastError, true);
        continue;
      }
      if (file.size > (state.config?.max_upload_mb || 12) * 1024 * 1024) {
        failed++;
        lastError = `${file.name} vượt giới hạn dung lượng.`;
        view.phase = "error";
        view.message = lastError;
        view.finishedAt = Date.now();
        renderUploadProgress(view);
        toast(lastError, true);
        continue;
      }
      const uploadId = newUploadId();
      let stopMonitor = () => {};
      try {
        const result = await uploadDocument(file, {
          uploadId,
          onProgress: ({ percent, loaded }) => {
            view.percent = percent;
            view.bytes = loaded;
            renderUploadProgress(view);
          },
          onTransferred: () => {
            if (view.phase !== "uploading") return;
            view.bytes = file.size;
            view.phase = "processing";
            renderUploadProgress(view);
            $("#connection-label").textContent = "Đang lập chỉ mục tài liệu…";
            stopMonitor = monitorUpload(uploadId, view);
          },
        });
        view.bytes = file.size;
        try {
          view.server = await api(`/uploads/${uploadId}`);
        } catch {
          view.server = {
            stage: "completed",
            chunks: result.document.chunk_count,
            pages: result.document.page_count,
          };
        }
        view.phase = "complete";
        view.finishedAt = Date.now();
        view.message = result.duplicate
          ? "Tài liệu đã có trong kho."
          : `Đã lập chỉ mục ${result.document.chunk_count} đoạn.`;
        renderUploadProgress(view);
        if (result.duplicate) duplicates++;
        else added++;
        if (result.document.warnings?.length)
          toast(result.document.warnings[0], true);
        if (!state.working) {
          state.selected.add(result.document.id);
          state.scopeMode = "selected";
        }
        try {
          await refreshDocuments();
        } catch (error) {
          toast(
            `Tài liệu đã lưu nhưng chưa làm mới được danh sách: ${error.message}`,
            true,
          );
        }
      } catch (error) {
        failed++;
        lastError = `${file.name}: ${error.message}`;
        view.phase = "error";
        view.message = lastError;
        view.finishedAt = Date.now();
        renderUploadProgress(view);
        toast(lastError, true);
      } finally {
        sentBytes += view.bytes;
        stopMonitor();
      }
    }
  } finally {
    clearInterval(ticker);
    const parts = [];
    if (added) parts.push(`Đã thêm ${added} tài liệu`);
    if (duplicates) parts.push(`${duplicates} tài liệu đã có`);
    if (failed) parts.push(`${failed} tệp lỗi`);
    currentUpload = {
      phase: failed && !added && !duplicates ? "error" : "complete",
      index: files.length,
      total: files.length,
      file: null,
      batchBytes,
      bytes: sentBytes,
      startedAt: batchStartedAt,
      finishedAt: Date.now(),
      work: `${added}/${files.length} tài liệu đã thêm`,
      message:
        failed === 1 && files.length === 1
          ? lastError
          : `${parts.join(" · ")}.`,
    };
    renderUploadProgress(currentUpload);
    setUploadBusy(false);
    $("#connection-label").textContent = "Backend đã kết nối";
    $("#file-input").value = "";
  }
}

async function seedDocuments() {
  if (state.uploading) return;
  setUploadBusy(true);
  $("#connection-label").textContent = "Đang nạp tài liệu mẫu…";
  try {
    const seeded = await api("/documents/demo", { method: "POST" });
    if (!state.working) {
      state.selected = new Set(seeded.documents.map((d) => d.id));
      state.scopeMode = "selected";
    }
    await refreshDocuments();
    toast("Đã nạp ghi chú mẫu. Đây là tài liệu demo do hệ thống biên soạn.");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setUploadBusy(false);
    $("#connection-label").textContent = "Backend đã kết nối";
  }
}

async function deleteDocument(document) {
  if (
    !window.confirm(
      `Xóa "${document.filename}" khỏi kho? Các báo cáo đã lưu giữ bản chụp bằng chứng.`,
    )
  )
    return;
  try {
    await api(`/documents/${document.id}`, { method: "DELETE" });
    await refreshDocuments();
    toast("Đã xóa tài liệu.");
  } catch (error) {
    toast(error.message, true);
  }
}

function switchView(name) {
  $$(".view").forEach((view) => {
    view.hidden = view.id !== `view-${name}`;
  });
  $$(".main-nav .nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === name);
    if (item.dataset.view === name) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  $("#breadcrumb-view").textContent = {
    research: "Nghiên cứu",
    activity: "Hoạt động agent",
    library: "Kho tài liệu",
    history: "Lịch sử phiên",
    experiments: "Thực nghiệm",
  }[name];
  const menuWasOpen = $("#sidebar").classList.contains("open");
  setSidebarOpen(false);
  if (menuWasOpen) $("#main-content").focus({ preventScroll: true });
}

async function showHistory() {
  const root = $("#history-list");
  root.replaceChildren(el("p", "loading-label", "Đang tải lịch sử…"));
  try {
    const records = (await api("/runs")).runs;
    root.replaceChildren();
    if (!records.length)
      root.append(
        el(
          "p",
          "loading-label",
          "Chưa có phiên nghiên cứu. Bắt đầu từ tài liệu trong kho.",
        ),
      );
    for (const record of records) {
      const item = el("button", "history-item");
      const content = el("div");
      content.append(
        el("strong", "", record.question),
        el(
          "small",
          "",
          `${dateLabel(record.created_at)} · ${record.id.slice(0, 16)}…`,
        ),
      );
      item.append(
        icon("file"),
        content,
        el("span", `history-state ${record.status}`, statuses[record.status]),
      );
      item.addEventListener("click", () =>
        openRun(record.id).catch((error) => toast(error.message, true)),
      );
      root.append(item);
    }
  } catch (error) {
    root.replaceChildren(el("p", "loading-label", error.message));
  }
}

function showLogin() {
  if (!$("#login-dialog").open) $("#login-dialog").showModal();
}

$$(".main-nav .nav-item").forEach((button) =>
  button.addEventListener("click", () => {
    switchView(button.dataset.view);
    if (button.dataset.view === "history") showHistory();
    if (button.dataset.view === "library")
      refreshDocuments().catch((error) => toast(error.message, true));
  }),
);
["sidebar-upload", "library-upload", "dropzone"].forEach((id) =>
  $(`#${id}`).addEventListener("click", () => $("#file-input").click()),
);
$("#file-input").addEventListener("change", (event) =>
  uploadFiles([...event.target.files]),
);
$("#upload-close").addEventListener("click", () => {
  $("#upload-status").hidden = true;
});
["seed-button", "library-seed"].forEach((id) =>
  $(`#${id}`).addEventListener("click", seedDocuments),
);
$("#reindex-button").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    const result = await api("/documents/reindex", { method: "POST" });
    toast(`Đã reindex ${result.chunks_updated} đoạn.`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
});
$("#dropzone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("#dropzone").classList.add("dragging");
});
$("#dropzone").addEventListener("dragleave", () =>
  $("#dropzone").classList.remove("dragging"),
);
$("#dropzone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("#dropzone").classList.remove("dragging");
  uploadFiles([...event.dataTransfer.files]);
});
$("#question").addEventListener("input", (event) => {
  $("#question-count").textContent = `${event.target.value.length} / 2000`;
  updateResearchSteps();
});
$("#question").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    startResearch();
  }
});
$$(".prompt-chip").forEach((button) =>
  button.addEventListener("click", () => {
    $("#question").value = button.dataset.question;
    $("#question").dispatchEvent(new Event("input"));
    $("#question").focus();
  }),
);
$("#run-button").addEventListener("click", startResearch);
$("#cancel-button").addEventListener("click", cancelResearch);
$("#advanced-toggle").addEventListener("click", () => {
  const panel = $("#advanced-panel");
  panel.hidden = !panel.hidden;
  $("#advanced-toggle").setAttribute("aria-expanded", String(!panel.hidden));
});
["settings-button", "notice-settings"].forEach((id) =>
  $(`#${id}`).addEventListener("click", async () => {
    $("#settings-dialog").showModal();
    try {
      const info = await api("/diagnostics");
      $("#diagnostics").textContent =
        `${info.documents} tài liệu · ${info.chunks} đoạn · ${info.indexed_chunks} đoạn có vector. ${info.reindex_needed ? "Cần reindex cho model hiện tại." : "Chỉ mục sẵn sàng."} Ngân sách: ${info.budget.calls} lượt gọi/phiên, ${info.budget.requests_per_minute} request/phút.`;
    } catch (error) {
      $("#diagnostics").textContent = error.message;
    }
  }),
);
$("#clear-cache").addEventListener("click", async (event) => {
  event.currentTarget.disabled = true;
  try {
    const result = await api("/cache/clear", { method: "POST" });
    toast(`Đã xóa ${result.deleted} phản hồi đã lưu.`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    event.currentTarget.disabled = false;
  }
});
$("#import-benchmark").addEventListener("click", () =>
  $("#benchmark-input").click(),
);
$("#benchmark-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (file.size > 8 * 1024 * 1024)
      throw new Error("File benchmark vượt quá 8 MB.");
    renderBenchmark(JSON.parse(await file.text()));
    toast("Đã nhập kết quả benchmark.");
  } catch (error) {
    toast(error.message || "JSON không hợp lệ.", true);
  } finally {
    event.target.value = "";
  }
});
$$(".close-dialog").forEach((button) =>
  button.addEventListener("click", () => button.closest("dialog").close()),
);
$$("dialog").forEach((dialog) =>
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog && dialog.id !== "login-dialog") {
      const bounds = dialog.getBoundingClientRect();
      if (
        event.clientX < bounds.left ||
        event.clientX > bounds.right ||
        event.clientY < bounds.top ||
        event.clientY > bounds.bottom
      )
        dialog.close();
    }
  }),
);
const mobileLayout = window.matchMedia?.("(max-width: 800px)");
function setSidebarOpen(open) {
  $("#sidebar").classList.toggle("open", open);
  $("#sidebar-backdrop").hidden = !open;
  $("#sidebar").inert = Boolean(mobileLayout?.matches && !open);
  $(".content-shell").inert = Boolean(mobileLayout?.matches && open);
  document.body.classList.toggle(
    "menu-open",
    Boolean(mobileLayout?.matches && open),
  );
  $("#mobile-menu").setAttribute("aria-expanded", String(open));
  $("#mobile-menu").setAttribute("aria-label", open ? "Đóng menu" : "Mở menu");
  if (open) $(".main-nav .active").focus();
}
mobileLayout?.addEventListener("change", () => setSidebarOpen(false));
setSidebarOpen(false);
$("#mobile-menu").addEventListener("click", () =>
  setSidebarOpen(!$("#sidebar").classList.contains("open")),
);
$("#sidebar-backdrop").addEventListener("click", () => {
  setSidebarOpen(false);
  $("#mobile-menu").focus();
});
document.addEventListener("keydown", (event) => {
  if (
    event.key === "Tab" &&
    mobileLayout?.matches &&
    $("#sidebar").classList.contains("open")
  ) {
    const focusable = $$(
      "#sidebar a, #sidebar button:not(:disabled), #sidebar input:not(:disabled):not([hidden])",
    );
    const first = focusable[0];
    const last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
  if (event.key === "Escape" && $("#sidebar").classList.contains("open")) {
    setSidebarOpen(false);
    $("#mobile-menu").focus();
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (!document.querySelector("dialog[open]")) openLibrarySearch();
  }
});
function openLibrarySearch() {
  switchView("library");
  $("#library-search").focus();
}
$("#workspace-search").addEventListener("click", openLibrarySearch);
$("#library-search").addEventListener("input", renderLibrary);
window.addEventListener("research-result-rendered", updateResearchSteps);
$$(".tab").forEach((button) =>
  button.addEventListener("click", () => selectTab(button.dataset.tab)),
);
$("#copy-button").addEventListener("click", copyReport);
window.addEventListener("authentication-required", showLogin);
$("#login-dialog").addEventListener("cancel", (event) =>
  event.preventDefault(),
);
$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/auth/login", {
      method: "POST",
      body: { token: $("#access-token").value },
    });
    $("#access-token").value = "";
    $("#login-error").textContent = "";
    $("#login-dialog").close();
    await initialize();
  } catch (error) {
    $("#login-error").textContent = error.message;
  }
});
$("#logout-button").addEventListener("click", async () => {
  try {
    await api("/auth/logout", { method: "POST" });
    localStorage.removeItem("research-active-run");
    window.location.reload();
  } catch (error) {
    toast(error.message, true);
  }
});
initialize();

$("#architecture-button").addEventListener("click", async () => {
  const root = $("#architecture-content");
  root.replaceChildren(el("p", "loading-label", "Đang tải luồng agent…"));
  $("#architecture-dialog").showModal();
  try {
    const data = await api("/agents");
    root.replaceChildren();
    data.agents.forEach((agent, index) => {
      const card = el("article", "architecture-card");
      card.append(
        el("small", "", `0${index + 1} · ${agent.role}`),
        el("h3", "", agent.purpose),
      );
      const list = el("ol");
      agent.stages.forEach((stage) => list.append(el("li", "", stage)));
      card.append(list, el("p", "", agent.guard));
      root.append(card);
    });
  } catch (error) {
    root.replaceChildren(el("p", "", error.message));
  }
});

function renderScopePicker() {
  const root = $("#scope-documents");
  const filter = $("#scope-search").value.toLocaleLowerCase("vi");
  root.replaceChildren();
  for (const doc of state.documents.filter((d) =>
    d.filename.toLocaleLowerCase("vi").includes(filter),
  )) {
    const row = el("label", "scope-document");
    const box = el("input");
    box.type = "checkbox";
    box.checked = state.selected.has(doc.id);
    box.disabled = state.working;
    box.setAttribute("aria-label", `Chọn nguồn ${doc.filename}`);
    box.addEventListener("change", () => {
      box.checked ? state.selected.add(doc.id) : state.selected.delete(doc.id);
      state.scopeMode = "selected";
      renderDocuments();
    });
    const text = el("div");
    text.append(
      el("strong", "", doc.filename),
      el(
        "small",
        "",
        `${doc.page_count || "—"} trang · ${doc.chunk_count} đoạn`,
      ),
    );
    row.append(box, text);
    root.append(row);
  }
  if (!root.children.length)
    root.append(el("p", "", "Không có tài liệu phù hợp."));
}
function activityRecord(record) {
  $("#activity-run").textContent =
    `${record.request.question} · ${statuses[record.status]}`;
  $("#export-trace").hidden = false;
  $("#export-trace").href = `/api/runs/${record.id}/trace`;
  const documents = record.scope_documents || [];
  $("#activity-sources").textContent = documents.length
    ? documents.map((doc) => doc.filename).join(" · ")
    : "Phạm vi được lưu trong nhật ký phiên.";
}
$("#scope-mode").addEventListener("change", (e) => {
  state.scopeMode = e.target.value;
  renderScope();
});
$("#choose-sources").addEventListener("click", () =>
  $("#scope-dialog").showModal(),
);
$("#scope-search").addEventListener("input", renderScopePicker);
$("#scope-clear").addEventListener("click", () => {
  if (!state.working) {
    state.selected.clear();
    state.scopeMode = "selected";
    renderDocuments();
  }
});
$("#back-research").addEventListener("click", () => {
  switchView("research");
  if (!$("#results-section").hidden)
    $("#results-section").scrollIntoView?.({ block: "start" });
});
$("#activity-cancel").addEventListener("click", cancelResearch);
$("#trace-filter").addEventListener("change", (e) => {
  $$(".trace-item").forEach(
    (row) =>
      (row.hidden =
        e.target.value !== "all" && row.dataset.agent !== e.target.value),
  );
});
$("#rebuild-button").addEventListener("click", async (e) => {
  if (
    !window.confirm(
      "Chia lại toàn bộ tài liệu bằng cấu hình hiện tại? Báo cáo cũ được giữ lại.",
    )
  )
    return;
  const button = e.currentTarget;
  button.disabled = true;
  try {
    const result = await api("/documents/rebuild", { method: "POST" });
    await refreshDocuments();
    toast(`Đã xây dựng lại ${result.chunks_updated} đoạn.`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
});
