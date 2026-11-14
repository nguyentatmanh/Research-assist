import { api } from "./api.js";
import { createTargetForm } from "./target_editor.js";
import { $, $$, el, toast } from "./dom.js";

const roles = {
  retrieval: [
    "Retrieval & Router",
    "Phân tích câu hỏi, tìm kiếm BM25/dense và xử lý thiếu nguồn.",
  ],
  reader: ["Reader", "Đọc đoạn gốc, trích nguyên văn và giữ vị trí nguồn."],
  summarizer: ["Summarizer", "Tổng hợp nhận định có liên kết đến bằng chứng."],
  fact_checker: [
    "Fact Checker",
    "Kiểm tra hỗ trợ, số liệu, đơn vị và mâu thuẫn.",
  ],
  writer: ["Writer", "Biên tập báo cáo từ nhận định đã kiểm chứng."],
};
const labels = {
  default: "Cấu hình server",
  demo: "Demo trích xuất",
  api: "Cloud API",
  cpu: "Local CPU",
  gpu: "Kaggle GPU",
};
const key = "fusion-agent-targets-v2";
let config = null,
  selected = {};
try {
  const saved = JSON.parse(localStorage.getItem(key) || "{}");
  if (saved && typeof saved === "object" && !Array.isArray(saved))
    selected = saved;
} catch {
  /* Storage may be unavailable. */
}
export function executionSelection() {
  return { ...selected };
}
function save() {
  try {
    localStorage.setItem(key, JSON.stringify(selected));
  } catch {
    toast(
      "Trình duyệt không cho lưu cấu hình; lựa chọn chỉ giữ trong trang này.",
    );
  }
}
export function selectPreset(name) {
  const wanted = Object.fromEntries(
    Object.keys(roles).map((role) => [
      role,
      name === "hybrid"
        ? ["reader", "fact_checker"].includes(role)
          ? "cpu"
          : "api"
        : name,
    ]),
  );
  const unavailable = Object.values(wanted).find(
    (id) =>
      !config?.execution_targets?.some((t) => t.id === id && t.configured),
  );
  if (unavailable) {
    toast(`${labels[unavailable]} chưa được cấu hình trên server.`, true);
    return false;
  }
  selected = wanted;
  save();
  renderSettings();
  return true;
}
function renderSummary() {
  const effective = Object.keys(roles).map((role) => {
    const target = selected[role] || "default";
    return target === "default"
      ? config?.agent_routes.find((r) => r.agent === role)
      : config?.execution_targets.find((t) => t.id === target);
  });
  const demo = effective.every((route) => route?.provider === "demo");
  $("#mode-notice").hidden = !demo;
  $("#provider-pill").classList.toggle("demo", demo);
  $("#provider-pill").replaceChildren(
    el("span", "status-dot"),
    document.createTextNode(
      demo
        ? "Demo · không cần key"
        : [...new Set(effective.map((r) => r?.model).filter(Boolean))].join(
            " / ",
          ),
    ),
  );
  const names = Object.entries(selected)
    .filter(([, target]) => target !== "default")
    .map(([role, target]) => `${roles[role][0]}: ${labels[target]}`);
  $("#run-execution-summary").textContent = names.length
    ? `Phiên kế tiếp · ${names.join(" · ")}`
    : `Phiên kế tiếp · cấu hình server · ${config?.model || "đang kết nối"}`;
}
export function configureWorkspace(value) {
  config = value;
  const clean = Object.fromEntries(
    Object.entries(selected).filter(
      ([role, target]) =>
        roles[role] &&
        value.execution_targets?.some((t) => t.id === target && t.configured),
    ),
  );
  if (Object.keys(clean).length !== Object.keys(selected).length)
    toast(
      "Một đích đã lưu không còn khả dụng; agent đó trở về cấu hình server.",
      true,
    );
  selected = clean;
  save();
  renderSettings();
  loadTargetEditors();
}
function renderSettings() {
  if (!config) return;
  const root = $("#execution-rows");
  root.replaceChildren();
  Object.entries(roles).forEach(([role, [title, desc]], index) => {
    const row = el("div", "agent-mode-row");
    row.dataset.agent = role;
    const info = el("div", "agent-info");
    const text = el("div");
    const choice = selected[role] || "default";
    const current =
      choice === "default"
        ? config.agent_routes.find((r) => r.agent === role)
        : config.execution_targets.find((t) => t.id === choice);
    text.append(
      el("strong", "", `Agent ${index + 1}: ${title}`),
      el("p", "", desc),
      el(
        "span",
        "agent-active-desc",
        `Phiên kế tiếp: ${current?.provider || "—"} · ${current?.model || "—"}`,
      ),
    );
    info.append(
      el("div", "agent-badge", String(index + 1).padStart(2, "0")),
      text,
    );
    const group = el("div", "agent-toggle-group");
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", `Mô hình ${title}`);
    for (const target of config.execution_targets || []) {
      const button = el(
        "button",
        `btn-agent-mode${choice === target.id ? " active" : ""}`,
        labels[target.id],
      );
      button.type = "button";
      button.dataset.mode = target.id;
      button.dataset.agent = role;
      button.disabled = !target.configured;
      button.setAttribute("aria-pressed", String(choice === target.id));
      button.title = target.configured
        ? `${target.provider} · ${target.model}`
        : "Chưa cấu hình trên server";
      button.addEventListener("click", () => {
        selected[role] = target.id;
        save();
        renderSettings();
        $("#execution-feedback").textContent =
          `Đã chọn ${labels[target.id]} cho ${title}. Áp dụng khi bắt đầu phiên tiếp theo.`;
      });
      group.append(button);
    }
    row.append(info, group);
    root.append(row);
  });
  for (const b of $$("[data-preset]")) {
    const ids =
      b.dataset.preset === "hybrid" ? ["api", "cpu"] : [b.dataset.preset];
    b.disabled = ids.some(
      (id) =>
        !config.execution_targets.some((t) => t.id === id && t.configured),
    );
  }
  const catalog = $("#target-catalog");
  catalog.replaceChildren();
  for (const target of config.execution_targets || []) {
    const row = el("div", "detail-row target-row");
    row.append(
      el("strong", "detail-label", labels[target.id]),
      el(
        "span",
        "detail-val",
        target.configured
          ? `${target.provider} · ${target.model}`
          : "Chưa cấu hình",
      ),
    );
    if (target.configured) {
      const check = el("button", "secondary-button", "Kiểm tra kết nối");
      check.dataset.probe = target.id;
      const status = el("small", "probe-status");
      status.setAttribute("role", "status");
      check.addEventListener("click", async () => {
        check.disabled = true;
        status.textContent = "Đang kiểm tra…";
        try {
          const result = await api(`/config/targets/${target.id}/test`, {
            method: "POST",
          });
          status.textContent = `${result.reachable && result.model_found ? "Kết nối được" : "Chưa xác nhận"} · ${result.note}`;
        } catch (e) {
          status.textContent = e.message;
        } finally {
          check.disabled = false;
        }
      });
      row.append(check, status);
    }
    catalog.append(row);
  }
  renderSummary();
}
export async function refreshDiagnostics() {
  try {
    const d = await api("/diagnostics");
    $("#settings-diag-info").textContent =
      `${d.documents} tài liệu · ${d.chunks} đoạn · ${d.indexed_chunks} vector · ${d.active_runs} phiên đang xử lý. Ngân sách ${d.budget.calls} lượt gọi, ${d.budget.tokens} token/phiên. ${d.reindex_needed ? "Cần reindex cho embedding hiện tại." : "Chỉ mục sẵn sàng."}`;
  } catch (e) {
    $("#settings-diag-info").textContent = e.message;
  }
}
export function initializeWorkspace(refreshDocuments) {
  $$("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => selectPreset(b.dataset.preset)),
  );
  $("#embedding-settings").addEventListener("click", () =>
    $("#settings-button").click(),
  );
  $("#settings-refresh").addEventListener("click", async () => {
    try {
      configureWorkspace(await api("/config"));
      await refreshDiagnostics();
    } catch (e) {
      toast(e.message, true);
    }
  });
  $("#settings-clear-cache").addEventListener("click", () =>
    $("#clear-cache").click(),
  );
  $("#settings-reindex").addEventListener("click", () =>
    $("#reindex-button").click(),
  );
  $("#settings-rebuild").addEventListener("click", () =>
    $("#rebuild-button").click(),
  );
  $("#export-pdf").addEventListener("click", () => window.print());
  $("#scout-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#scout-form button");
    if (button.disabled) return;
    button.disabled = true;
    const status = $("#scout-status"),
      root = $("#scout-results");
    root.replaceChildren();
    status.textContent = "Đang tìm arXiv và Crossref…";
    try {
      const result = await api("/scout/search", {
        method: "POST",
        body: { query: $("#scout-query").value.trim() },
      });
      for (const paper of result.papers) {
        const row = el("li", "scout-result");
        const link = el("a", "", paper.title);
        const url = new URL(paper.url);
        if (url.protocol === "https:") {
          link.href = url.href;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
        }
        row.append(
          link,
          el(
            "small",
            "",
            `${paper.provider} · ${paper.year || "—"} · ${paper.authors.join(", ")}`,
          ),
        );
        if (paper.arxiv_id) {
          const add = el("button", "secondary-button", "Nhập PDF arXiv");
          add.dataset.arxiv = paper.arxiv_id;
          add.addEventListener("click", async () => {
            add.disabled = true;
            try {
              await api("/papers/import", {
                method: "POST",
                body: { paper_id: paper.arxiv_id },
              });
              await refreshDocuments();
              add.textContent = "Đã có trong kho";
            } catch (e) {
              add.disabled = false;
              toast(e.message, true);
            }
          });
          row.append(add);
        }
        root.append(row);
      }
      status.textContent = `${result.papers.length} kết quả. ${(result.warnings || []).join(" ")} ${result.policy}`;
    } catch (e) {
      status.textContent = e.message;
    } finally {
      button.disabled = false;
    }
  });
}

let editorsVersion = 0;
let openApiSetup = window.location.hash === "#settings-api";
async function loadTargetEditors() {
  const version = ++editorsVersion,
    root = $("#target-editors");
  if (!root) return;
  try {
    const data = await api("/config/targets");
    if (version !== editorsVersion) return;
    const expanded = new Set(
      $$("details[open] .target-form", root).map((form) => form.dataset.target),
    );
    root.replaceChildren();
    for (const name of ["api", "cpu", "gpu"]) {
      const saved = data.targets.find((t) => t.id === name),
        details = el("details", "target-editor");
      details.append(
        el(
          "summary",
          "",
          `Cấu hình ${labels[name]}${saved?.managed ? " · do boot quản lý" : ""}`,
        ),
      );
      if (saved?.managed) {
        details.append(
          el(
            "p",
            "input-helper",
            "Đích GPU được boot tạo cùng token phiên. Đổi model qua FUSION_MODEL rồi khởi động lại boot.",
          ),
        );
        root.append(details);
        continue;
      }
      const form = createTargetForm(name, saved, {
        async onSave() {
          configureWorkspace(await api("/config"));
          toast(
            `Đã lưu ${labels[name]}. Chọn đích cho agent rồi kiểm tra kết nối trước khi nghiên cứu.`,
          );
        },
        async onRestore() {
          configureWorkspace(await api("/config"));
          toast(`Đã khôi phục ${labels[name]} theo cấu hình server.`);
        },
      });
      details.append(form);
      details.open = expanded.has(name);
      root.append(details);
      if (name === "api" && openApiSetup) {
        openApiSetup = false;
        details.open = true;
        details.scrollIntoView?.({ block: "start" });
      }
    }
  } catch (error) {
    if (version === editorsVersion)
      root.replaceChildren(el("p", "error-text", error.message));
  }
}
