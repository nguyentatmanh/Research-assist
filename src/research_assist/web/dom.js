export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [
  ...root.querySelectorAll(selector),
];

export function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

export function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

export function toast(message, error = false) {
  const item = el("div", `toast${error ? " error" : ""}`, message);
  $("#toast-region").append(item);
  setTimeout(() => item.remove(), error ? 8000 : 4500);
}

export function dateLabel(value) {
  return new Intl.DateTimeFormat("vi-VN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function locator(evidence) {
  return evidence.page !== null ? `Trang ${evidence.page}` : evidence.section;
}

export const statuses = {
  queued: "Đang chờ",
  running: "Đang chạy",
  completed: "Hoàn thành",
  failed: "Thất bại",
  cancelled: "Đã dừng",
};
