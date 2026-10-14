import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { startBackend } from "./backend.mjs";
const require = createRequire(import.meta.url);
const { JSDOM } = require("jsdom");
const backend = await startBackend();
const base = backend.base;
const nativeFetch = globalThis.fetch;
const html = await (await nativeFetch(base)).text();
const dom = new JSDOM(html, { url: base });
const { window } = dom;
globalThis.window = window;
globalThis.document = window.document;
globalThis.localStorage = window.localStorage;
globalThis.Event = window.Event;
globalThis.FormData = window.FormData;
let pendingRequests = 0;
globalThis.fetch = async (path, options) => {
  pendingRequests++;
  try {
    const response = await nativeFetch(new URL(path, base), options);
    await response.clone().arrayBuffer();
    return response;
  } finally {
    pendingRequests--;
  }
};
window.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
window.HTMLDialogElement.prototype.close = function () {
  this.open = false;
};
const errors = [];
window.addEventListener("error", (event) => errors.push(event.message));
const find = (selector) => window.document.querySelector(selector);
async function waitFor(predicate) {
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error("DOM condition timed out");
}
let checks = 0;
try {
  await import("../src/research_assist/web/app.js");
  await waitFor(() =>
    find("#connection-label").textContent.includes("đã kết nối"),
  );
  assert.ok(find("#provider-pill").textContent.includes("Demo"));
  checks++;
  find("#seed-button").click();
  await waitFor(() => find("#nav-doc-count").textContent === "3");
  checks++;
  find(".prompt-chip").click();
  assert.ok(find("#question").value.includes("Self-RAG"));
  checks++;
  find("#run-button").click();
  await waitFor(() => !find("#results-section").hidden);
  assert.equal(
    window.document.querySelectorAll(".agent-node.completed").length,
    5,
  );
  assert.ok(find("#report-content").textContent.includes("RAG"));
  checks++;
  find(".citation-button").click();
  assert.ok(find("#source-dialog").open);
  assert.ok(find(".source-context mark").textContent.length > 20);
  checks++;
  find("#source-dialog .close-dialog").click();
  find('[data-tab="checks"]').click();
  assert.equal(find("#checks-content").hidden, false);
  assert.ok(find(".check-status.supported"));
  checks++;
  find('[data-tab="trace"]').click();
  assert.equal(window.document.querySelectorAll(".trace-item").length, 10);
  checks++;
  find('[data-view="history"]').click();
  await waitFor(() => find(".history-item"));
  find(".history-item").click();
  await waitFor(() => !find("#view-research").hidden);
  checks++;
  find('[data-view="library"]').click();
  await waitFor(
    () =>
      window.document.querySelectorAll("#document-table-body tr").length === 3,
  );
  checks++;
  const report = await import("../src/research_assist/web/report.js");
  report.renderMarkdown(
    '# Safe report\n\n- <img src=x onerror="window.xss=true"> [999]\n',
  );
  assert.equal(find("#report-content img"), null);
  assert.ok(find("#report-content").textContent.includes("<img"));
  assert.equal(window.xss, undefined);
  checks++;
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      status: "passed",
      checks,
      javascriptErrors: errors,
      verification: "DOM/API integration; not visual browser QA",
    }),
  );
} finally {
  await waitFor(() => pendingRequests === 0);
  await new Promise((resolve) => setTimeout(resolve, 50));
  await backend.close();
  dom.window.close();
}
