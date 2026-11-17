import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { startBackend } from "./backend.mjs";
const backend = await startBackend(),
  nativeFetch = globalThis.fetch;
const dom = new JSDOM(
  await (await nativeFetch(backend.base + "/desk")).text(),
  {
    url: backend.base,
  },
);
const w = dom.window;
for (const name of [
  "window",
  "document",
  "localStorage",
  "Event",
  "FormData",
  "XMLHttpRequest",
])
  globalThis[name] =
    name === "window" ? w : name === "document" ? w.document : w[name];
w.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
w.HTMLDialogElement.prototype.close = function () {
  this.open = false;
};
w.confirm = () => true;
let printed = 0;
w.print = () => printed++;
const $ = (s) => w.document.querySelector(s),
  $$ = (s) => [...w.document.querySelectorAll(s)];
const errors = [],
  requests = [];
w.addEventListener("error", (e) => errors.push(e.message));
globalThis.fetch = async (path, options) => {
  const url = new URL(path, backend.base);
  requests.push(url.pathname);
  // Only external discovery is a fixture; all document/run/settings traffic uses real HTTP.
  if (url.pathname === "/api/scout/search")
    return new Response(
      JSON.stringify({
        papers: [
          {
            title: "<img src=x onerror=alert(1)> Literature",
            authors: ["A researcher"],
            year: 2024,
            provider: "crossref",
            url: "https://doi.org/10.1/test",
          },
        ],
        warnings: ["arxiv: fixture unavailable"],
        policy: "Metadata only",
      }),
      { headers: { "Content-Type": "application/json" } },
    );
  return nativeFetch(url, options);
};
async function wait(test) {
  const end = Date.now() + 60000;
  while (Date.now() < end) {
    if (test()) return;
    await new Promise((r) => setTimeout(r, 30));
  }
  throw Error(
    "Timed out: " +
      test.toString() +
      "\nMethodology: " +
      $("#methodology-content").textContent +
      "\nJS errors: " +
      JSON.stringify(errors),
  );
}
try {
  await import("../src/research_assist/web/app.js");
  await wait(() => $("#connection-label").textContent.includes("đã kết nối"));
  assert.match(w.document.title, /Research Assist/);
  assert.equal($$(".main-nav [data-view]").length, 6);
  assert.equal($("#orchestration-mode").value, "autonomous");
  $('[data-view="settings"]').click();
  await wait(() => $("#settings-diag-info").textContent.includes("tài liệu"));
  assert.equal($$(".agent-mode-row").length, 5);
  assert.ok($('[data-preset="gpu"]').disabled);
  await wait(() => $("#target-cpu-model"));
  $("#target-cpu-model").value = "test-local-model";
  $("#target-cpu-url").value = "http://localhost:11434/v1";
  $('.target-form[data-target="cpu"]').dispatchEvent(
    new w.Event("submit", { cancelable: true }),
  );
  await wait(() => !$('[data-preset="cpu"]').disabled);
  assert.equal(
    (await (await nativeFetch(backend.base + "/api/config/targets")).json())
      .targets[0].model,
    "test-local-model",
  );
  await wait(() => $("#target-cpu-model").value === "test-local-model");
  $('.target-form[data-target="cpu"] .text-button').click();
  await wait(() => $('[data-preset="cpu"]').disabled);

  $('[data-probe="demo"]').click();
  await wait(() => $("#target-catalog").textContent.includes("Kết nối được"));
  $('[data-preset="demo"]').click();
  assert.equal(
    Object.keys(JSON.parse(localStorage.getItem("fusion-agent-targets-v2")))
      .length,
    5,
  );
  $('.btn-agent-mode[data-agent="writer"][data-mode="default"]').click();
  assert.equal(
    JSON.parse(localStorage.getItem("fusion-agent-targets-v2")).writer,
    "default",
  );
  $('[data-view="research"]').click();
  $("#question").value = "/so-l";
  $("#question").dispatchEvent(new w.Event("input"));
  assert.equal($("#command-popup").hidden, false);
  assert.equal($$("#command-options [role=option]").length, 1);
  $("#question").dispatchEvent(
    new w.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
  );
  assert.match($("#question").value, /số liệu/);
  assert.equal($("#command-popup").hidden, true);
  $(".quick-chip").click();
  assert.match($("#question").value, /đóng góp/);
  $("#seed-button").click();
  await wait(() => $("#nav-doc-count").textContent === "3");
  Object.defineProperty($("#file-input"), "files", {
    configurable: true,
    value: [
      new w.File(
        [
          "RAG uses retrieved source evidence to support generated answers. Experiments use 50 samples and report limitations. <img src=x onerror=alert(1)> is untrusted data.",
        ],
        "desk-test.txt",
        { type: "text/plain" },
      ),
    ],
  });
  $("#file-input").dispatchEvent(new w.Event("change"));
  await wait(() => $("#nav-doc-count").textContent === "4");
  $("#question").value =
    "How does RAG use retrieved evidence to support generated answers?";
  $("#requirements").value = "Explain evidence retrieval\nExplain verification";
  $("#run-button").click();
  await wait(() => !$("#results-section").hidden);
  const records = await (await nativeFetch(backend.base + "/api/runs")).json();
  const id = records.runs[0].id;
  const result = await (
    await nativeFetch(backend.base + `/api/runs/${id}/export?format=json`)
  ).json();
  assert.equal(result.orchestration_mode, "autonomous");
  assert.equal(result.agent_targets.writer, "default");
  assert.equal(result.agent_targets.reader, "demo");
  assert.ok(result.reported_claim_ids.length > 0);
  $('[data-tab="methodology"]').click();
  await wait(() => $$(".methodology-item").length === 6);
  assert.equal($("#methodology-content").hidden, false);
  assert.ok($(".audit-summary"));
  $('[data-tab="checks"]').click();
  assert.equal($("#checks-content").hidden, false);
  assert.ok($$(".trace-item").length > 10);
  $(".citation-button").click();
  assert.equal($("#source-dialog").open, true);
  assert.ok($(".source-context mark"));
  $("#source-dialog .close-dialog").click();
  $("#export-pdf").click();
  assert.equal(printed, 1);
  assert.equal(
    (
      await nativeFetch(
        new URL($("#export-md").getAttribute("href"), backend.base),
      )
    ).status,
    200,
  );
  assert.equal(
    $$("#report-content img,#checks-content img,#methodology-content img")
      .length,
    0,
  );
  $('[data-view="history"]').click();
  await wait(() => $(".history-item"));
  $(".history-item").click();
  await wait(() => !$("#view-research").hidden);
  $('[data-view="library"]').click();
  $("#scout-query").value = "RAG evidence";
  $("#scout-form").dispatchEvent(new w.Event("submit", { cancelable: true }));
  await wait(() => $(".scout-result"));
  assert.equal($$("#scout-results img").length, 0);
  assert.match($("#scout-results").textContent, /<img/);
  assert.match($("#scout-status").textContent, /fixture unavailable/);
  for (const view of [
    "research",
    "library",
    "activity",
    "experiments",
    "settings",
    "history",
  ]) {
    $(`[data-view="${view}"]`).click();
    assert.equal($(`#view-${view}`).hidden, false);
  }
  for (const node of $$(
    'link[href^="/assets/"],script[src^="/assets/"],img[src^="/assets/"]',
  ))
    assert.equal(
      (
        await nativeFetch(
          new URL(
            node.getAttribute("href") || node.getAttribute("src"),
            backend.base,
          ),
        )
      ).status,
      200,
    );
  const ids = $$("[id]").map((e) => e.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.deepEqual(errors, []);
  console.log(
    "PASS: Research Desk /desk; six views, real config and probes, per-agent selection/persistence, slash commands, quick prompts, demo seed, XHR upload, Autonomous run, frozen selection/export, audit/methodology, source dialog, trace/history, MD/JSON export, print dispatch, assets and unique IDs.",
  );
  console.log(
    "Scout discovery uses an explicit metadata fixture with XSS/partial-failure coverage. CSS layout, print pagination and live GPU are not verified by JSDOM.",
  );
} finally {
  w.close();
  await backend.close();
}
