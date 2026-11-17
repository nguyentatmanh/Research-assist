import { api } from "./api.js";
import { el } from "./dom.js";

export function createTargetForm(
  name,
  saved,
  { onSave, onRestore, showProbe = false },
) {
  const form = el("form", "target-form");
  form.dataset.target = name;
  const field = (label, control) => {
    const wrap = el("label", "", label);
    wrap.append(control);
    form.append(wrap);
  };
  const provider = el("select");
  provider.id = `target-${name}-provider`;
  for (const [value, label] of [
    ["gemini", "Gemini"],
    ["openai", "OpenAI-compatible"],
    ["ollama", "Ollama"],
    ["deepseek", "DeepSeek"],
  ]) {
    const opt = el("option", "", label);
    opt.value = value;
    provider.append(opt);
  }
  provider.value =
    saved?.provider ||
    (name === "cpu" ? "ollama" : name === "gpu" ? "openai" : "gemini");
  field("Provider", provider);
  const model = el("input");
  model.id = `target-${name}-model`;
  model.type = "text";
  model.required = true;
  model.maxLength = 200;
  model.value = saved?.model || "";
  model.placeholder = "Tên model đúng như server cung cấp";
  field("Model", model);
  const endpoint = el("input");
  endpoint.id = `target-${name}-url`;
  endpoint.type = "url";
  endpoint.maxLength = 2048;
  endpoint.value =
    saved?.base_url || (name === "cpu" ? "http://127.0.0.1:11434/v1" : "");
  endpoint.placeholder = "https://your-worker.example/v1";
  field("Endpoint /v1", endpoint);
  const keyInput = el("input");
  keyInput.id = `target-${name}-key`;
  keyInput.type = "password";
  keyInput.autocomplete = "new-password";
  keyInput.maxLength = 4096;
  field("API key / worker token", keyInput);
  const note = el(
    "p",
    "input-helper",
    "Localhost là máy đang chạy backend. Đổi endpoint hoặc provider cần nhập token mới. CPU/GPU là nơi triển khai do bạn cấu hình, không tự cài model.",
  );
  const buttons = el("div", "target-actions");
  const save = el("button", "primary-button", "Lưu cấu hình");
  save.type = "submit";
  buttons.append(save);
  const status = el("p", "target-save-status");
  status.setAttribute("role", "status");
  form.append(note, buttons, status);
  const sync = () => {
    endpoint.disabled = provider.value === "gemini";
    endpoint.required = !endpoint.disabled;
    const keepKey =
      saved?.key_configured &&
      saved.provider === provider.value &&
      (saved.base_url || "").replace(/\/+$/, "") ===
        (endpoint.disabled ? "" : endpoint.value.trim().replace(/\/+$/, ""));
    keyInput.required = provider.value !== "ollama" && !keepKey;
    keyInput.placeholder = keepKey
      ? "Đã lưu · để trống để giữ khóa"
      : "Token / API key";
  };
  provider.addEventListener("change", sync);
  endpoint.addEventListener("input", sync);
  sync();
  const restore = el("button", "text-button", "Khôi phục cấu hình server");
  restore.type = "button";
  if (onRestore) buttons.append(restore);

  const check = el("button", "secondary-button", "Kiểm tra kết nối đã lưu");
  check.type = "button";
  if (showProbe && saved) buttons.append(check);

  const perform = async (message, operation) => {
    if (save.disabled) return;
    save.disabled = restore.disabled = check.disabled = true;
    status.textContent = message;
    try {
      await operation();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      save.disabled = restore.disabled = check.disabled = false;
    }
  };
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    perform("Đang lưu…", async () => {
      const payload = {
        provider: provider.value,
        model: model.value.trim(),
        base_url: endpoint.disabled ? "" : endpoint.value.trim(),
        api_key: keyInput.value,
      };
      keyInput.value = "";
      const result = await api(`/config/targets/${name}`, {
        method: "PUT",
        body: payload,
      });
      await onSave(result);
    });
  });
  restore.addEventListener("click", () =>
    perform("Đang khôi phục…", async () => {
      await api(`/config/targets/${name}`, { method: "DELETE" });
      keyInput.value = "";
      await onRestore();
    }),
  );
  check.addEventListener("click", () =>
    perform("Đang kiểm tra…", async () => {
      const result = await api(`/config/targets/${name}/test`, {
        method: "POST",
      });
      const outcome =
        result.reachable && result.model_found
          ? "Kết nối được"
          : "Chưa xác nhận";
      status.textContent = outcome + " · " + result.note;
    }),
  );
  return form;
}
