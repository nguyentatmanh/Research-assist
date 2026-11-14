import { api, uploadDocument } from "./api.js";
import { $, $$, el, toast, statuses } from "./dom.js";
import { presentReport } from "./report_content.js?v=verified-report-1";

const roles = [
  ["retrieval", "Retrieval", "Lập truy vấn · tìm nguồn"],
  ["reader", "Reader", "Đọc ngữ cảnh · trích bằng chứng"],
  ["summarizer", "Summarizer", "Tổng hợp · đề xuất nhận định"],
  ["fact_checker", "Fact checker", "Phản biện · kiểm tra hỗ trợ"],
  ["writer", "Writer", "Biên soạn từ kết quả kiểm chứng"],
];
let documents = [],
  selected = new Set(),
  record = null,
  polling = null,
  config = null;
let generation = 0;

function guarded(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (error) {
      toast(error.message, true);
    }
  };
}
function button(text, cls, action) {
  const node = el("button", cls, text);
  node.type = "button";
  node.addEventListener("click", guarded(action));
  return node;
}
function externalLink(text, url) {
  const node = el("a", "source-link", text);
  node.href = url;
  node.target = "_blank";
  node.rel = "noopener noreferrer";
  return node;
}
function scopeLabel() {
  $("#scope-label").textContent = selected.size
    ? `${selected.size} tài liệu trong phạm vi · giới hạn 2 vòng tìm kiếm`
    : "Chọn tài liệu để bắt đầu";
}

function renderDocuments() {
  const root = $("#documents");
  root.replaceChildren();
  $("#doc-count").textContent = documents.length;
  if (!documents.length)
    root.append(el("p", "empty", "Chưa có tài liệu. Tải lên hoặc nạp bộ mẫu."));
  for (const doc of documents) {
    const row = el("div", "document-row");
    const check = el("input");
    check.type = "checkbox";
    check.id = `select-${doc.id}`;
    check.checked = selected.has(doc.id);
    check.addEventListener("change", () => {
      check.checked ? selected.add(doc.id) : selected.delete(doc.id);
      scopeLabel();
    });
    const label = el("label");
    label.htmlFor = check.id;
    label.append(
      el("strong", "", doc.title || doc.filename),
      el("small", "", `${doc.kind.toUpperCase()} · ${doc.chunk_count} đoạn`),
    );
    const remove = button("×", "delete", async () => {
      await api(`/documents/${doc.id}`, { method: "DELETE" });
      selected.delete(doc.id);
      await loadDocuments();
    });
    remove.setAttribute("aria-label", `Xóa ${doc.filename}`);
    row.append(check, label, remove);
    root.append(row);
  }
  scopeLabel();
}
async function loadDocuments(selectNew = false) {
  const known = new Set(documents.map((d) => d.id));
  documents = (await api("/documents")).documents;
  const exists = new Set(documents.map((d) => d.id));
  selected = new Set([...selected].filter((id) => exists.has(id)));
  if (selectNew)
    for (const doc of documents) if (!known.has(doc.id)) selected.add(doc.id);
  renderDocuments();
}
async function upload(files) {
  for (const file of files) {
    $("#upload-status").textContent = `Đang tải ${file.name}…`;
    await uploadDocument(file, {
      onProgress: (p) => {
        $("#upload-status").textContent =
          `${file.name} · tải lên ${p.percent}%`;
      },
      onTransferred: () => {
        $("#upload-status").textContent =
          `${file.name} · đang trích xuất và lập chỉ mục…`;
      },
    });
  }
  $("#upload-status").textContent = "Đã thêm tài liệu.";
  await loadDocuments(true);
  $("#files").value = "";
}
function renderAgents(events = []) {
  const root = $("#agents");
  root.replaceChildren();
  roles.forEach(([id, title, description], index) => {
    const latest = [...events].reverse().find((e) => e.agent === id);
    const status = latest?.status || "";
    const card = el("div", `agent ${status}`);
    card.dataset.role = id;
    const content = el("div");
    content.append(
      el("strong", "", title),
      el("p", "", description),
      el(
        "span",
        "agent-status",
        latest ? statuses[status] || status : "SẴN SÀNG NHẬN NHIỆM VỤ",
      ),
    );
    card.append(
      el("span", "agent-number", String(index + 1).padStart(2, "0")),
      content,
    );
    root.append(card);
  });
}
function renderMarkdown(markdown, root) {
  // Small text-only Markdown subset. Raw HTML/model links are never executed.
  root.replaceChildren();
  let list = null;
  const text = (s) => s.replace(/\\([\\`*_{}\[\]<>()#!|])/g, "$1");
  for (const line of markdown.split("\n")) {
    if (!line.trim()) {
      list = null;
      continue;
    }
    if (line.startsWith("# ")) root.append(el("h1", "", text(line.slice(2))));
    else if (line.startsWith("## "))
      root.append(el("h2", "", text(line.slice(3))));
    else if (line.startsWith("### "))
      root.append(el("h3", "", text(line.slice(4))));
    else if (line.startsWith("> "))
      root.append(el("blockquote", "", text(line.slice(2))));
    else if (line.startsWith("- ")) {
      if (!list) {
        list = el("ul");
        root.append(list);
      }
      list.append(el("li", "", text(line.slice(2))));
    } else {
      list = null;
      root.append(el("p", "", text(line)));
    }
  }
}
function renderLedger(result) {
  const root = $("#ledger-panel");
  root.replaceChildren();
  root.append(
    el(
      "p",
      "small muted",
      "“Được hỗ trợ” là đánh giá của bộ kiểm chứng trên các nguồn bên dưới. Kiểm toán trích dẫn chỉ xác nhận cấu trúc nguồn và vị trí trích dẫn.",
    ),
  );
  if (!result) {
    root.append(el("p", "empty", "Chưa có kết quả kiểm chứng."));
    return;
  }
  const audit = result.citation_audit;
  root.append(
    el("p", "small", `Kiểm toán nguồn: ${audit?.status || "not_applicable"}`),
  );
  for (const claim of result.claims) {
    const check = result.checks.find((c) => c.claim_id === claim.id);
    const card = el("article", "ledger-card");
    const verdict = check?.verdict || "insufficient";
    card.append(
      el(
        "span",
        `verdict ${verdict}`,
        {
          supported: "Được hỗ trợ",
          contradicted: "Mâu thuẫn",
          insufficient: "Chưa đủ bằng chứng",
        }[verdict],
      ),
      el("p", "", claim.text),
      el("p", "small muted", check?.explanation || ""),
    );
    for (const eid of check?.evidence_ids || []) {
      const ev = result.evidence.find((e) => e.id === eid);
      if (!ev) continue;
      card.append(
        el("blockquote", "", ev.quote),
        externalLink(
          `${ev.filename} · ${ev.page ? "trang " + ev.page : ev.section} · ký tự ${ev.start}–${ev.end}`,
          `/api/documents/${encodeURIComponent(ev.document_id)}/file${ev.page ? "#page=" + ev.page : ""}`,
        ),
      );
    }
    root.append(card);
  }
  if (!result.claims.length)
    root.append(el("p", "empty", "Không có nhận định để công bố."));
  root.append(
    button(
      "Kiểm tra dấu hiệu phương pháp thực nghiệm",
      "secondary",
      async () => {
        const report = await api(`/runs/${record.id}/methodology`);
        const section = el("section", "ledger-card");
        section.append(el("p", "small muted", report.warning));
        for (const item of report.items)
          section.append(
            el(
              "p",
              "small",
              `${item.criterion}: ${item.matches.length} trích đoạn có từ khóa.`,
            ),
          );
        root.append(section);
      },
    ),
  );
}
function renderTrace(events) {
  const root = $("#trace-panel");
  root.replaceChildren();
  root.append(
    el(
      "p",
      "small muted",
      "Nhật ký thao tác và kết quả công cụ; không phải chuỗi suy luận riêng tư của mô hình.",
    ),
  );
  for (const event of events) {
    const row = el("article", "trace-event");
    row.append(
      el("span", "", `${event.sequence}. ${event.agent}`),
      el("div", "", event.message),
    );
    if (Object.keys(event.details || {}).length) {
      const details = el("details");
      details.append(
        el("summary", "", "Chi tiết thao tác"),
        el("pre", "", JSON.stringify(event.details, null, 2)),
      );
      row.append(details);
    }
    root.append(row);
  }
}
function renderRun() {
  if (!record) return;
  const running = ["queued", "running"].includes(record.status),
    result = record.result;
  $("#start").disabled = running;
  $("#cancel").hidden = !running;
  $("#run-meta").replaceChildren(
    el(
      "span",
      "",
      `${statuses[record.status] || record.status} · ${record.request.orchestration_mode || config?.orchestration_mode} · ${record.id.slice(-8)}`,
    ),
  );
  if (record.error) $("#run-meta").append(el("p", "warning", record.error));
  if (result) {
    $("#run-meta").append(
      el(
        "p",
        "",
        `${result.report_status} · lý do dừng: ${result.stop_reason}`,
      ),
    );
    for (const warning of result.warnings || [])
      $("#run-meta").append(el("p", "warning", warning));
      renderMarkdown(presentReport(result.markdown), $("#report-panel"));
    const values = [
      result.reported_claim_ids.length,
      result.evidence.length,
      result.llm_calls,
      `${(result.elapsed_ms / 1000).toFixed(1)}s`,
    ];
    $$("#metrics strong").forEach((node, i) => {
      node.textContent = values[i];
    });
  } else if (running) {
    $("#report-panel").replaceChildren(
      el(
        "p",
        "empty",
        "Nhóm nghiên cứu đang làm việc. Theo dõi tab Hoạt động để xem kết quả công cụ.",
      ),
    );
  } else {
    $("#report-panel").replaceChildren(
      el("p", "empty", record.error || "Phiên kết thúc mà chưa có báo cáo."),
    );
  }
  $("#export-md").disabled =
    $("#export-json").disabled =
    $("#print").disabled =
      !result;
  renderLedger(result);
  renderTrace(record.events || []);
  renderAgents(record.events || []);
}
async function watch(id) {
  generation += 1;
  const ticket = generation;
  clearTimeout(polling);
  const refresh = async () => {
    try {
      const next = await api(`/runs/${id}`);
      if (ticket !== generation) return;
      record = next;
      renderRun();
      if (["queued", "running"].includes(next.status))
        polling = setTimeout(refresh, 1200);
      else await history();
    } catch (error) {
      if (ticket !== generation) return;
      toast(error.message, true);
      polling = setTimeout(refresh, 4000);
    }
  };
  await refresh();
}
async function history() {
  const rows = (await api("/runs")).runs;
  const root = $("#history");
  root.replaceChildren();
  for (const row of rows.slice(0, 8))
    root.append(
      button(
        row.question || row.request?.question || row.id,
        "history-item",
        () => watch(row.id),
      ),
    );
}
async function initialize() {
  config = await api("/config");
  $("#connection").textContent = config.demo
    ? "Demo · trích xuất"
    : "Model đã cấu hình";
  $("#demo-notice").hidden = !config.demo;
  $("#mode").value = config.orchestration_mode;
  $("#routes").replaceChildren(el("div", "", "MÔ HÌNH THEO VAI TRÒ"));
  for (const route of config.agent_routes)
    $("#routes").append(
      el("div", "", `${route.agent || route.role}: ${route.model}`),
    );
  $("#logout").hidden = !config.auth_required;
  await loadDocuments(true);
  await history();
}
function showLogin() {
  if (!$("#login-dialog").open) $("#login-dialog").showModal();
}
window.addEventListener("authentication-required", showLogin);
$("#login-dialog").addEventListener("cancel", (event) =>
  event.preventDefault(),
);
$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/auth/login", {
      method: "POST",
      body: { token: $("#token").value },
    });
    $("#token").value = "";
    $("#login-error").textContent = "";
    $("#login-dialog").close();
    await initialize();
  } catch (error) {
    $("#login-error").textContent = error.message;
  }
});
$("#logout").addEventListener(
  "click",
  guarded(async () => {
    generation += 1;
    clearTimeout(polling);
    await api("/auth/logout", { method: "POST" });
    location.reload();
  }),
);
$("#files").addEventListener(
  "change",
  guarded((event) => upload([...event.target.files])),
);
$("#dropzone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("#dropzone").classList.add("dragging");
});
$("#dropzone").addEventListener("dragleave", () =>
  $("#dropzone").classList.remove("dragging"),
);
$("#dropzone").addEventListener(
  "drop",
  guarded(async (event) => {
    event.preventDefault();
    $("#dropzone").classList.remove("dragging");
    await upload([...event.dataTransfer.files]);
  }),
);
$("#seed").addEventListener(
  "click",
  guarded(async () => {
    $("#seed").disabled = true;
    try {
      await api("/documents/demo", { method: "POST" });
      await loadDocuments(true);
    } finally {
      $("#seed").disabled = false;
    }
  }),
);
$("#select-all").addEventListener("click", () => {
  selected = new Set(documents.map((d) => d.id));
  renderDocuments();
});
$("#refresh").addEventListener("click", guarded(history));
$("#example").addEventListener("click", () => {
  $("#question").value = "RAG sử dụng bằng chứng như thế nào?";
  $("#question").focus();
});
$("#research-form").addEventListener(
  "submit",
  guarded(async (event) => {
    event.preventDefault();
    if (!selected.size) {
      toast("Chọn ít nhất một tài liệu.", true);
      return;
    }
    const requirements = $("#requirements")
      .value.split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (requirements.length > 12) {
      toast("Tối đa 12 ý cần trả lời.", true);
      return;
    }
    $("#start").disabled = true;
    try {
      const run = await api("/runs", {
        method: "POST",
        body: {
          question: $("#question").value,
          document_ids: [...selected],
          scope_mode: "selected",
          language: $("#language").value,
          orchestration_mode: $("#mode").value,
          retrieval_mode: $("#retrieval").value,
          requirements,
          profile: "academic",
          use_cache: true,
          max_rounds: 2,
        },
      });
      await watch(run.id);
    } catch (error) {
      $("#start").disabled = false;
      throw error;
    }
  }),
);
$("#cancel").addEventListener(
  "click",
  guarded(async () => {
    if (record) {
      await api(`/runs/${record.id}/cancel`, { method: "POST" });
      await watch(record.id);
    }
  }),
);
$$("[data-tab]").forEach((tab) =>
  tab.addEventListener("click", () => {
    $$("[data-tab]").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", String(t === tab));
    });
    for (const name of ["report", "ledger", "trace"])
      $(`#${name}-panel`).hidden = name !== tab.dataset.tab;
  }),
);
for (const format of ["md", "json"])
  $(`#export-${format}`).addEventListener("click", () => {
    if (record?.result)
      location.href = `/api/runs/${record.id}/export?format=${format}`;
  });
$("#print").addEventListener("click", () => window.print());
$("#scout-form").addEventListener(
  "submit",
  guarded(async (event) => {
    event.preventDefault();
    $("#scout-button").disabled = true;
    const root = $("#scout-results");
    root.replaceChildren(el("p", "small", "Đang tìm…"));
    try {
      const result = await api("/scout/search", {
        method: "POST",
        body: { query: $("#scout-query").value },
      });
      root.replaceChildren();
      for (const warning of result.warnings)
        root.append(el("p", "small muted", warning));
      for (const paper of result.papers) {
        const card = el("div", "paper-card");
        card.append(
          externalLink(paper.title, paper.url),
          el(
            "p",
            "small muted",
            `${paper.provider} · ${paper.year || "chưa rõ năm"}`,
          ),
        );
        if (paper.arxiv_id)
          card.append(
            button("Nhập PDF arXiv", "secondary", async () => {
              await api("/papers/import", {
                method: "POST",
                body: { paper_id: paper.arxiv_id },
              });
              await loadDocuments(true);
              toast("Đã nhập PDF.");
            }),
          );
        else
          card.append(
            el(
              "p",
              "small muted",
              "Mở nguồn và tải toàn văn nếu bạn có quyền truy cập.",
            ),
          );
        root.append(card);
      }
      if (!result.papers.length)
        root.append(el("p", "small muted", "Chưa tìm thấy bài báo."));
    } finally {
      $("#scout-button").disabled = false;
    }
  }),
);
renderAgents();
(async () => {
  try {
    const auth = await api("/auth/status");
    if (auth.required && !auth.authenticated) showLogin();
    else await initialize();
  } catch (error) {
    $("#connection").textContent = "Chưa kết nối";
    toast(error.message, true);
  }
})();
