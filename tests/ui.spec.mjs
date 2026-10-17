import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir } from "node:fs/promises";
import { startBackend } from "./backend.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const backend = await startBackend();
const base = backend.base;
const output = process.env.UI_OUTPUT_DIR || "test-results";
await mkdir(output, { recursive: true });
let browser;
let checks = 0;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
    deviceScaleFactor: 1,
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(base);
  await page.waitForFunction(() =>
    document
      .querySelector("#connection-label")
      .textContent.includes("đã kết nối"),
  );
  await page.screenshot({
    path: `${output}/workspace-desktop.png`,
    fullPage: true,
  });
  checks++;
  await page.click('[data-view="activity"]');
  await page.click("#architecture-button");
  await page.waitForSelector(".architecture-card");
  assert.equal(await page.locator(".architecture-card").count(), 5);
  await page.screenshot({
    path: `${output}/architecture-desktop.png`,
    fullPage: true,
  });
  await page.locator("#architecture-dialog .close-dialog").click();
  checks++;
  await page.click('[data-view="research"]');
  await page.click("#seed-button");
  await page.waitForFunction(
    () => document.querySelector("#nav-doc-count").textContent === "3",
  );
  checks++;
  await page.fill(
    "#question",
    "RAG và Self-RAG khác nhau như thế nào về truy xuất và kiểm tra câu trả lời?",
  );
  await page.locator(".requirement-options summary").click();
  await page.fill(
    "#requirements",
    "Giải thích truy xuất\nGiải thích kiểm chứng",
  );
  await page.click("#run-button");
  await page.waitForFunction(
    () => !document.querySelector("#results-section").hidden,
  );
  await page.click("#back-research");
  assert.ok(
    (await page.locator("#report-content").textContent()).includes("RAG"),
  );
  assert.equal(await page.locator(".agent-node.completed").count(), 5);
  assert.equal(await page.locator("#coverage-panel").isVisible(), true);
  assert.equal(await page.locator(".coverage-item").count(), 2);
  assert.ok(
    (await page.locator("#coverage-panel").textContent()).includes(
      "Chưa xác định",
    ),
  );
  checks++;
  assert.ok((await page.locator(".citation-button").count()) > 0);
  assert.ok(
    (await page.locator("#usage-panel").textContent()).includes("cache hit"),
  );
  checks++;
  await page.screenshot({
    path: `${output}/report-desktop.png`,
    fullPage: true,
  });
  await page.click(".citation-button");
  await page.waitForSelector("#source-dialog[open]");
  assert.ok(
    (await page.locator(".source-detail-quote").textContent()).length > 20,
  );
  assert.equal(await page.locator(".source-context mark").count(), 1);
  await page.locator("#source-dialog .close-dialog").click();
  checks++;
  await page.click('[data-tab="checks"]');
  assert.ok((await page.locator(".check-status.supported").count()) > 0);
  await page.click('[data-view="activity"]');
  assert.ok((await page.locator(".trace-item").count()) >= 14);
  checks++;
  await page.click('[data-view="history"]');
  await page.waitForSelector(".history-item");
  await page.click(".history-item");
  await page.waitForSelector("#results-section:not([hidden])");
  checks++;
  await page.click('[data-view="library"]');
  await page.waitForFunction(
    () => document.querySelectorAll("#document-table-body tr").length === 3,
  );
  const documentPayload = Buffer.from(
    'RAG evidence is displayed safely. <img src=x onerror="window.xss=true"> Research document text is untrusted.',
  );
  await page.locator("#file-input").setInputFiles({
    name: "safe-display.txt",
    mimeType: "text/plain",
    buffer: documentPayload,
  });
  await page.waitForFunction(
    () => document.querySelector("#nav-doc-count").textContent === "4",
  );
  assert.equal(await page.evaluate(() => Boolean(window.xss)), false);
  checks++;
  // Select only the uploaded document and verify source scope in the visible report.
  await page.click('[data-view="research"]');
  await page.click("#choose-sources");
  await page.click("#scope-clear");
  await page
    .getByRole("checkbox", { name: "Chọn nguồn safe-display.txt", exact: true })
    .check();
  await page.locator("#scope-dialog .primary-button").click();
  await page.fill(
    "#question",
    "Explain the RAG evidence in this research document.",
  );
  await page.click("#run-button");
  await page.waitForFunction(
    () => !document.querySelector("#results-section").hidden,
  );
  await page.click("#back-research");
  assert.ok(
    (await page.locator("#report-content").textContent()).includes(
      "displayed safely",
    ),
  );
  assert.equal(await page.evaluate(() => Boolean(window.xss)), false);
  assert.equal(await page.locator("#report-content img").count(), 0);
  checks++;
  const mobile = await browser.newPage({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 1,
  });
  await mobile.goto(base);
  await mobile.waitForFunction(() =>
    document
      .querySelector("#connection-label")
      .textContent.includes("đã kết nối"),
  );
  assert.equal(
    await mobile.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    ),
    false,
  );
  await mobile.click("#mobile-menu");
  assert.ok(
    (await mobile.locator("#sidebar").getAttribute("class")).includes("open"),
  );
  await mobile.click('[data-view="library"]');
  assert.equal(await mobile.locator("#view-library").isVisible(), true);
  await mobile.click("#mobile-menu");
  await mobile.click('[data-view="research"]');
  await mobile.waitForFunction(
    () => document.querySelector("#sidebar").getBoundingClientRect().right < 1,
  );
  await mobile.screenshot({
    path: `${output}/workspace-mobile.png`,
    fullPage: true,
  });
  checks++;
  assert.deepEqual(errors, []);
  await page.click('[data-view="experiments"]');
  assert.equal(await page.locator("#view-experiments").isVisible(), true);
  const benchmark = {
    format_version: 2,
    kind: "retrieval_benchmark",
    split: "test fixture",
    eligible_queries: 2,
    corpus_size: 5,
    device: "cpu",
    variants: [
      {
        name: "Browser fixture (test only)",
        metrics: {
          "recall@10": 0.5,
          "hit@10": 1,
          "mrr@10": 0.5,
          "ndcg@10": 0.6,
        },
        latency_p50_ms: 2,
        latency_p95_ms: 3,
      },
    ],
  };
  await page.locator("#benchmark-input").setInputFiles({
    name: "test.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(benchmark)),
  });
  await page.waitForSelector(".benchmark-table");
  assert.ok(
    (await page.locator("#benchmark-results").textContent()).includes("0.500"),
  );
  await page.click("#settings-button");
  await page.waitForSelector("#settings-dialog[open]");
  assert.equal(await page.locator(".route-item").count(), 5);
  await page.waitForFunction(() =>
    document.querySelector("#diagnostics").textContent.includes("Ngân sách"),
  );
  checks += 3;
  console.log(
    JSON.stringify({
      status: "passed",
      checks,
      javascriptErrors: errors,
      screenshots: output,
    }),
  );
} finally {
  await browser?.close();
  await backend.close();
}
