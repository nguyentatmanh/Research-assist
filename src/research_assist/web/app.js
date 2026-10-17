import { api } from "./api.js";
import { $, $$, el, icon, dateLabel, statuses, toast } from "./dom.js";
import { renderEvents, renderResult, selectTab, copyReport } from "./report.js";
import { renderBenchmark, renderRoutes } from "./lab.js";

const state = {
  documents: [],
  selected: new Set(),
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
    input.setAttribute("aria-label", `Chọn ${document.filename}`);
    input.addEventListener("change", () => {
      input.checked
        ? state.selected.add(document.id)
        : state.selected.delete(document.id);
      renderScope();
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
  const body = $("#document-table-body");
  body.replaceChildren();
  if (!state.documents.length) {
    const row = el("tr");
    const cell = el(
      "td",
      "table-empty",
      "Tải tài liệu hoặc nạp bộ ghi chú mẫu để bắt đầu.",
    );
    cell.colSpan = 5;
    row.append(cell);
    body.append(row);
  }
  for (const document of state.documents) {
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
    container.append(open, remove);
    actions.append(container);
    row.append(actions);
    body.append(row);
  }
  renderScope();
}

function renderScope() {
  const label = !state.documents.length
    ? "Chưa có tài liệu"
    : state.selected.size
      ? `${state.selected.size} tài liệu được chọn`
      : `Toàn bộ ${state.documents.length} tài liệu`;
  $("#scope-label").textContent = label;
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
  const requirements = $("#requirements")
    .value.split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (requirements.length > 6 || requirements.some((s) => s.length > 500)) {
    toast("Tối đa 6 yêu cầu, mỗi yêu cầu tối đa 500 ký tự.", true);
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
        document_ids: [...state.selected],
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

async function uploadFiles(files) {
  if (state.uploading) {
    toast("Đang lập chỉ mục. Hãy chờ tải lên hoàn thành.");
    return;
  }
  state.uploading = true;
  const accepted = ["pdf", "docx", "txt", "md"];
  $("#connection-label").textContent = "Đang lập chỉ mục tài liệu…";
  try {
    for (const file of files) {
      if (!accepted.includes(file.name.split(".").at(-1).toLowerCase())) {
        toast(`Không hỗ trợ ${file.name}.`, true);
        continue;
      }
      if (file.size > (state.config?.max_upload_mb || 12) * 1024 * 1024) {
        toast(`${file.name} vượt giới hạn dung lượng.`, true);
        continue;
      }
      const form = new FormData();
      form.append("file", file);
      try {
        const result = await api("/documents", { method: "POST", body: form });
        if (result.document.warnings?.length)
          toast(result.document.warnings[0], true);
        toast(
          result.duplicate
            ? `${file.name}: đã có trong kho.`
            : `${file.name}: đã lập chỉ mục ${result.document.chunk_count} đoạn.`,
        );
        await refreshDocuments();
      } catch (error) {
        toast(`${file.name}: ${error.message}`, true);
      }
    }
  } finally {
    state.uploading = false;
    $("#connection-label").textContent = "Backend đã kết nối";
    $("#file-input").value = "";
  }
}

async function seedDocuments(button) {
  if (state.uploading) return;
  state.uploading = true;
  button.disabled = true;
  $("#connection-label").textContent = "Đang nạp tài liệu mẫu…";
  try {
    await api("/documents/demo", { method: "POST" });
    await refreshDocuments();
    toast("Đã nạp ghi chú mẫu. Đây là tài liệu demo do hệ thống biên soạn.");
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.uploading = false;
    button.disabled = false;
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
  });
  $("#breadcrumb-view").textContent = {
    research: "Nghiên cứu",
    library: "Kho tài liệu",
    history: "Lịch sử phiên",
    experiments: "Thực nghiệm",
  }[name];
  $("#sidebar").classList.remove("open");
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
["seed-button", "library-seed"].forEach((id) =>
  $(`#${id}`).addEventListener("click", (event) =>
    seedDocuments(event.currentTarget),
  ),
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
$("#mobile-menu").addEventListener("click", () =>
  $("#sidebar").classList.toggle("open"),
);
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
