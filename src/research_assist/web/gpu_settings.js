import { api } from "./api.js";
import { $, toast } from "./dom.js";

let saved = null;
let busy = false;
let generation = 0;
let applied = async () => {};
let working = () => false;

function tokenizerFor(selection) {
  return selection === "qwen3-0.6b"
    ? "Qwen/Qwen3-Embedding-0.6B"
    : selection === "qwen3-4b"
      ? "Qwen/Qwen3-Embedding-4B"
      : "intfloat/multilingual-e5-base";
}

function selectionFor(value) {
  if (value.model_family !== "qwen3") return "e5";
  return value.tokenizer_model?.includes("0.6B") ? "qwen3-0.6b" : "qwen3-4b";
}

function feedback(message, kind = "") {
  const output = $("#gpu-feedback");
  output.textContent = message;
  output.dataset.kind = kind;
}

function setBusy(value) {
  busy = value;
  $("#gpu-fields").disabled = value;
  $("#gpu-config-form").setAttribute("aria-busy", String(value));
  $("#gpu-rebuild").disabled = value || !saved?.document_count;
  $("#gpu-restore").disabled = value || !saved?.saved;
}

function showSaved(value) {
  saved = value;
  $("#gpu-url").value = value.base_url || "";
  $("#gpu-model").value = value.model_id || "";
  $("#gpu-family").value = selectionFor(value);
  $("#gpu-key").value = "";
  $("#gpu-tokenizer").value =
    value.tokenizer_model || tokenizerFor(selectionFor(value));
  $("#gpu-batch").value = value.batch_size || 16;
  $("#gpu-key-note").textContent = value.key_configured
    ? "Đã có key. Để trống để giữ lại khi URL không đổi; nhập lại nếu đổi URL."
    : "Key riêng của notebook, tối thiểu 32 ký tự.";
  const badge = $("#gpu-active-badge");
  badge.textContent = value.active ? "Đang dùng GPU endpoint" : "Chưa bật";
  badge.dataset.active = String(value.active);
  $("#gpu-index-panel").hidden = !value.active || !value.document_count;
  $("#gpu-index-note").textContent = value.reindex_needed
    ? "Tài liệu đang dùng vector của cấu hình cũ. Cập nhật chỉ mục trước khi nghiên cứu với model mới."
    : "Chỉ mục khớp model hiện tại. Nếu đổi tokenizer, hãy cập nhật lại chỉ mục.";
}

function payload() {
  if (!$("#gpu-config-form").reportValidity()) return null;
  return {
    base_url: $("#gpu-url").value.trim(),
    api_key: $("#gpu-key").value,
    model_id: $("#gpu-model").value.trim(),
    model_family: $("#gpu-family").value === "e5" ? "e5" : "qwen3",
    tokenizer_model: $("#gpu-tokenizer").value.trim(),
    batch_size: Number($("#gpu-batch").value),
  };
}

export async function loadGpuSettings() {
  if (busy) return;
  const current = ++generation;
  setBusy(true);
  feedback("Đang đọc cấu hình GPU…");
  try {
    const result = await api("/embedding/remote");
    if (current !== generation) return;
    showSaved(result);
    feedback(
      result.active
        ? "GPU endpoint đã được cấu hình. Bạn có thể kiểm tra lại kết nối khi notebook sẵn sàng."
        : "Dán URL và key từ notebook để kết nối model embedding.",
    );
  } catch (error) {
    if (current === generation) feedback(error.message, "error");
  } finally {
    if (current === generation) setBusy(false);
  }
}

export function initializeGpuSettings(onApplied, isWorking) {
  applied = onApplied;
  working = isWorking;
  $("#settings-dialog").addEventListener("close", () => {
    $("#gpu-key").value = "";
  });
  $("#gpu-family").addEventListener("change", () => {
    $("#gpu-tokenizer").value = tokenizerFor($("#gpu-family").value);
    $("#gpu-model").value = "";
  });
  $("#gpu-config-form").addEventListener("input", () => {
    if (!busy)
      feedback("Có thay đổi chưa lưu. Kiểm tra kết nối hoặc lưu để áp dụng.");
  });
  async function connect(save) {
    if (busy) return;
    if (save && working()) {
      feedback(
        "Hãy chờ nghiên cứu hoặc tải tài liệu hoàn tất trước khi đổi GPU.",
        "error",
      );
      return;
    }
    const body = payload();
    if (!body) return;
    setBusy(true);
    feedback(
      save
        ? "Đang kiểm tra và lưu cấu hình GPU…"
        : "Đang kiểm tra kết nối GPU…",
    );
    try {
      const result = await api(
        save ? "/embedding/remote" : "/embedding/remote/test",
        {
          method: save ? "PUT" : "POST",
          body,
        },
      );
      const connection = save ? result.connection : result;
      if (save) {
        showSaved(result);
        await applied();
        toast("Đã lưu và áp dụng GPU endpoint.");
      } else {
        $("#gpu-model").value = connection.model_id;
        if (connection.tokenizer_model) {
          $("#gpu-tokenizer").value = connection.tokenizer_model;
          $("#gpu-family").value = selectionFor(connection);
        }
      }
      feedback(
        `${save ? "Đã lưu và áp dụng" : "Kết nối thành công"} · ${connection.dimension} chiều · ${connection.latency_ms} ms${save ? "." : ". Chọn Lưu và áp dụng để sử dụng."}`,
        "success",
      );
    } catch (error) {
      feedback(error.message, "error");
    } finally {
      setBusy(false);
    }
  }
  $("#gpu-test").addEventListener("click", () => connect(false));
  $("#gpu-config-form").addEventListener("submit", (event) => {
    event.preventDefault();
    connect(true);
  });
  $("#gpu-restore").addEventListener("click", async () => {
    if (busy) return;
    if (working()) {
      feedback("Hãy chờ phiên làm việc hiện tại kết thúc.", "error");
      return;
    }
    setBusy(true);
    try {
      showSaved(await api("/embedding/remote", { method: "DELETE" }));
      await applied();
      feedback("Đã dùng lại cấu hình embedding gốc của backend.", "success");
    } catch (error) {
      feedback(error.message, "error");
    } finally {
      setBusy(false);
    }
  });
  $("#gpu-rebuild").addEventListener("click", async () => {
    if (busy) return;
    if (working()) {
      feedback("Hãy chờ phiên làm việc hiện tại kết thúc.", "error");
      return;
    }
    setBusy(true);
    feedback(
      "Đang chia lại tài liệu và tạo vector qua GPU. Có thể cần tải tokenizer lần đầu…",
    );
    try {
      const result = await api("/documents/rebuild", { method: "POST" });
      await applied();
      const status = await api("/embedding/remote");
      showSaved(status);
      feedback(
        `Đã cập nhật ${result.chunks_updated} đoạn tài liệu. Chỉ mục đã sẵn sàng.`,
        "success",
      );
    } catch (error) {
      feedback(error.message, "error");
    } finally {
      setBusy(false);
    }
  });
}
