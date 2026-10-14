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
