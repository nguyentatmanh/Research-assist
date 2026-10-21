export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function api(path, { method = "GET", body, ...options } = {}) {
  const headers = {};
  let payload = body;
  if (body && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const response = await fetch(`/api${path}`, {
    method,
    headers,
    body: payload,
    credentials: "same-origin",
    ...options,
  });
  if (!response.ok) {
    let message = `Yêu cầu không hoàn thành (HTTP ${response.status}).`;
    try {
      const data = await response.json();
      message =
        typeof data.detail === "string"
          ? data.detail
          : "Dữ liệu không hợp lệ. Kiểm tra câu hỏi và các tùy chọn.";
    } catch {
      /* A proxy may return a plain-text error. */
    }
    if (response.status === 401)
      window.dispatchEvent(new Event("authentication-required"));
    throw new ApiError(message, response.status);
  }
  return response.status === 204 ? null : response.json();
}

export function uploadDocument(
  file,
  { uploadId, onProgress, onTransferred } = {},
) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/documents");
    request.withCredentials = true;
    request.responseType = "json";
    if (uploadId) request.setRequestHeader("X-Upload-ID", uploadId);
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && event.total > 0)
        onProgress?.({
          percent: Math.min(
            100,
            Math.round((event.loaded / event.total) * 100),
          ),
          loaded: Math.min(
            file.size,
            Math.round((event.loaded / event.total) * file.size),
          ),
        });
    });
    request.upload.addEventListener("load", () => onTransferred?.());
    request.addEventListener("load", () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(request.response);
        return;
      }
      if (request.status === 401)
        window.dispatchEvent(new Event("authentication-required"));
      const detail = request.response?.detail;
      reject(
        new ApiError(
          typeof detail === "string"
            ? detail
            : `Yêu cầu không hoàn thành (HTTP ${request.status}).`,
          request.status,
        ),
      );
    });
    request.addEventListener("error", () =>
      reject(new ApiError("Mất kết nối khi tải tài liệu.", 0)),
    );
    request.addEventListener("abort", () =>
      reject(new ApiError("Đã dừng tải tài liệu.", 0)),
    );
    const form = new FormData();
    form.append("file", file);
    request.send(form);
  });
}
