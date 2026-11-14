import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { startBackend } from "./backend.mjs";

const backend = await startBackend();
const nativeFetch = globalThis.fetch;
const response = await nativeFetch(backend.base);
const html = await response.text();
assert.match(html, /id="antigravity-bg"/);
assert.match(
  response.headers.get("content-security-policy"),
  /frame-src blob:/,
);
assert.match(html, /src="\/assets\/workspace\.js/);
for (const path of ["/demo2", "/demo2/"]) {
  const legacy = await nativeFetch(backend.base + path, { redirect: "manual" });
  assert.equal(legacy.status, 308);
  assert.equal(legacy.headers.get("location"), "/");
}
assert.match(
  await (await nativeFetch(backend.base + "/desk")).text(),
  /id="research-form"|id="question"/,
);
const dom = new JSDOM(html, { url: backend.base });
const w = dom.window;
globalThis.requestAnimationFrame = (callback) => setTimeout(callback, 0);
for (const name of [
  "window",
  "document",
  "Event",
  "FormData",
  "XMLHttpRequest",
  "localStorage",
])
  globalThis[name] = name === "window" ? w : w[name];
globalThis.fetch = (path, options) =>
  nativeFetch(new URL(path, backend.base), options);
const $ = (selector) => w.document.querySelector(selector);
async function wait(condition) {
  for (let i = 0; i < 200; i++) {
    if (condition()) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw Error("Chờ giao diện quá hạn: " + $("#workspaceStatus").textContent);
}
try {
  for (const asset of [
    "/assets/workspace-base.css",
    "/assets/workspace-background.js",
    "/assets/workspace-logo.svg",
    "/assets/workspace.js",
  ])
    assert.equal((await nativeFetch(backend.base + asset)).status, 200);
  await import("../src/research_assist/web/workspace.js");
  await wait(() => $(".agent-model-select").options.length === 5);
  $("#docMoreBtn").click();
  [...w.document.querySelectorAll("#docDropdownMenu button")]
    .find((node) => node.textContent === "Nạp tài liệu mẫu")
    .click();
  await wait(() => Number($("#docCountBadge").textContent) > 0);
  $("#promptInput").value = "Summarize the findings in the selected document.";
  $("#promptInput").dispatchEvent(new w.Event("input"));
  $("#sendBtn").click();
  await wait(() => $("#workspaceStatus").textContent.includes("Chọn ít nhất"));
  assert.equal(
    (await (await nativeFetch(backend.base + "/api/runs")).json()).runs.length,
    0,
  );
  assert.equal(
    $("#promptInput").value,
    "Summarize the findings in the selected document.",
  );
  $(".doc-item input").click();
  const original = "/tom-tat Những kết luận chính của tài liệu này.";
  $("#promptInput").value = original;
  $("#promptInput").dispatchEvent(new w.Event("input"));
  $("#sendBtn").click();
  await wait(
    () =>
      $(".workspace-report") &&
      Number($("#historyCountBadge").textContent) === 1,
  );
  assert.match($(".thinking-accordion").textContent, /Hoạt động agent/);
  assert.equal($(".history-item").textContent, original);
  assert.equal($(".history-item").getAttribute("aria-current"), "true");
  $(".workspace-report .scout-card-actions button").click();
  assert.equal($("#sidecar").classList.contains("collapsed"), false);
  assert.equal($("#reportViewTab").classList.contains("active"), true);
  $(".history-item").click();
  await wait(
    () =>
      $(".workspace-report") &&
      $(".user-query-bubble")?.textContent === original,
  );
  const history = await (await nativeFetch(backend.base + "/api/runs")).json();
  const record = await (
    await nativeFetch(backend.base + "/api/runs/" + history.runs[0].id)
  ).json();
  assert.equal(record.status, "completed");
  assert.equal(history.runs[0].question, original);
  assert.equal(record.request.display_question, original);
  assert.match(record.request.question, /^Tóm tắt/u);
  assert.equal(record.request.orchestration_mode, "autonomous");
  assert.equal(record.request.document_ids.length, 1);
  console.log(
    "ResearchDesk DOM/API: default and alias, assets, source selection, Autonomous run, history, report side panel.",
  );
} finally {
  dom.window.close();
  await backend.close();
}
