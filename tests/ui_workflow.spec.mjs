import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { startBackend } from "./backend.mjs";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const backend = await startBackend();
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({
    viewport: { width: 1280, height: 660 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const ids = [];
  for (const name of ["Target paper.txt", "Other paper.txt"]) {
    const response = await page.request.post(backend.base + "/api/documents", {
      multipart: {
        file: {
          name,
          mimeType: "text/plain",
          buffer: Buffer.from(
            `${name} RAG retrieval uses source evidence to support research answers.`,
          ),
        },
      },
    });
    ids.push((await response.json()).document.id);
  }
  await page.goto(backend.base);
  await page.waitForFunction(
    () => document.querySelector("#nav-doc-count").textContent === "2",
  );
  const heights = [];
  for (const viewport of [
    { width: 1280, height: 660 },
    { width: 1366, height: 600 },
    { width: 1440, height: 760 },
  ]) {
    await page.setViewportSize(viewport);
    const height = await page
      .locator("#sidebar-documents")
      .evaluate((e) => e.getBoundingClientRect().height);
    assert.ok(height > 60);
    heights.push({ viewport, height });
  }
  await page.click("#choose-sources");
  await page
    .getByRole("checkbox", { name: "Chọn nguồn Target paper.txt", exact: true })
    .check();
  await page.locator("#scope-dialog .primary-button").click();
  assert.ok(
    (await page.locator("#scope-filenames").textContent()).includes(
      "Target paper",
    ),
  );
  assert.ok(
    !(await page.locator("#scope-filenames").textContent()).includes(
      "Other paper",
    ),
  );
  await page.fill("#question", "Explain RAG retrieval evidence");
  const submitted = page.waitForRequest(
    (r) => r.url().endsWith("/api/runs") && r.method() === "POST",
  );
  await page.click("#run-button");
  const request = await submitted;
  assert.deepEqual(request.postDataJSON().document_ids, [ids[0]]);
  assert.equal(request.postDataJSON().scope_mode, "selected");
  await page.waitForFunction(
    () => !document.querySelector("#results-section").hidden,
  );
  assert.equal(await page.locator("#view-activity").isVisible(), true);
  assert.ok((await page.locator(".trace-item details").count()) > 0);
  const traceHref = await page.locator("#export-trace").getAttribute("href");
  const trace = await (await page.request.get(backend.base + traceHref)).json();
  assert.deepEqual(trace.document_ids, [ids[0]]);
  await page.selectOption("#trace-filter", "reader");
  for (const row of await page.locator(".trace-item:visible").all())
    assert.equal(await row.getAttribute("data-agent"), "reader");
  await page.selectOption("#trace-filter", "all");
  // Controlled event replay checks the UI while a run is still running, including polling stability.
  await page.evaluate(async () => {
    const { renderEvents } = await import("/assets/report.js");
    const events = [
      {
        sequence: 1,
        agent: "reader",
        status: "running",
        round: 1,
        message: "Reading sources",
        created_at: "fixture-live",
        elapsed_ms: 0,
        details: { chunk_ids: ["ch_test"], execution: "api" },
      },
    ];
    renderEvents(events, "running");
    document.querySelector(".trace-item details").open = true;
    renderEvents(events, "running");
  });
  assert.equal(await page.locator("#view-activity").isVisible(), true);
  assert.equal(
    await page.locator(".trace-item details").getAttribute("open"),
    "",
  );
  assert.equal(await page.locator(".agent-node.running").count(), 1);
  await page.click("#back-research");
  // 640 CSS px is the reflow viewport of a 1280px window at 200% browser zoom.
  await page.setViewportSize({ width: 640, height: 360 });
  await page.click("#choose-sources");
  assert.equal(await page.locator("#scope-dialog").isVisible(), true);
  assert.equal(
    await page
      .getByRole("checkbox", {
        name: "Chọn nguồn Target paper.txt",
        exact: true,
      })
      .isVisible(),
    true,
  );
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth + 1,
    ),
    false,
  );
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      status: "passed",
      checks: 12,
      heights,
      javascriptErrors: errors,
      liveRendering:
        "controlled event replay; API scope and trace use real local backend",
    }),
  );
} finally {
  await browser?.close();
  await backend.close();
}
