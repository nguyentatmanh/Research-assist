import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";
import { startBackend } from "./backend.mjs";

const backend = await startBackend();
const output = "test-results/upload";
await mkdir(output, { recursive: true });
let browser;
let checks = 0;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
  });
  const page = await browser.newPage({
    viewport: { width: 1100, height: 800 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(backend.base);
  await page.waitForFunction(() =>
    document
      .querySelector("#connection-label")
      .textContent.includes("đã kết nối"),
  );
  await page.click('[data-view="library"]');

  let releaseResponse;
  const heldResponse = new Promise((resolve) => (releaseResponse = resolve));
  await page.route("**/api/documents", async (route) => {
    if (route.request().method() !== "POST") return route.continue();
    const response = await route.fetch();
    await heldResponse;
    await route.fulfill({ response });
  });
  // Route interception holds the real XHR upload events until fulfillment. Emit
  // controlled transport events while the real backend response is held.
  await page.evaluate(() => {
    const nativeSend = XMLHttpRequest.prototype.send;
    window.restoreXHRSend = () => {
      XMLHttpRequest.prototype.send = nativeSend;
    };
    XMLHttpRequest.prototype.send = function (...args) {
      const result = nativeSend.apply(this, args);
      setTimeout(() => {
        this.upload.dispatchEvent(
          new ProgressEvent("progress", {
            lengthComputable: true,
            loaded: 50,
            total: 100,
          }),
        );
      }, 100);
      setTimeout(() => this.upload.dispatchEvent(new Event("load")), 600);
      return result;
    };
  });
  await page.locator("#file-input").setInputFiles({
    name: "tiếng-Việt.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("Tài liệu nghiên cứu về tìm kiếm và bằng chứng."),
  });
  await page.waitForFunction(
    () => document.querySelector("#upload-percent").textContent === "50%",
  );
  assert.equal(
    await page.locator("#upload-progress").getAttribute("value"),
    "50",
  );
  await page.waitForFunction(
    () =>
      document.querySelector("#upload-status").dataset.phase === "processing",
  );
  assert.equal(await page.locator("#upload-status").isVisible(), true);
  assert.equal(
    await page.locator("#upload-filename").textContent(),
    "tiếng-Việt.txt",
  );
  assert.equal(await page.locator("#upload-count").textContent(), "1/1");
  assert.equal(
    await page.locator("#upload-progress").getAttribute("value"),
    null,
  );
  assert.ok(
    (await page.locator("#upload-message").textContent()).includes(
      "lập chỉ mục",
    ),
  );
  await page.screenshot({ path: `${output}/processing-desktop.png` });
  await page.locator("#upload-status").screenshot({
    path: `${output}/processing-panel.png`,
  });
  checks += 6;

  releaseResponse();
  await page.waitForFunction(
    () =>
      document.querySelector("#upload-status").dataset.phase === "complete" &&
      document
        .querySelector("#upload-message")
        .textContent.includes("Đã thêm 1"),
  );
  assert.ok(
    (await page.locator("#upload-message").textContent()).includes("Đã thêm 1"),
  );
  assert.equal(await page.locator("#nav-doc-count").textContent(), "1");
  assert.equal(await page.locator("#upload-close").isVisible(), true);
  await page.screenshot({ path: `${output}/complete-desktop.png` });
  await page.click("#upload-close");
  assert.equal(await page.locator("#upload-status").isVisible(), false);
  checks += 4;

  await page.evaluate(() => window.restoreXHRSend());
  await page.unroute("**/api/documents");
  let releaseFailure;
  const heldFailure = new Promise((resolve) => (releaseFailure = resolve));
  await page.route("**/api/documents", async (route) => {
    if (route.request().method() !== "POST") return route.continue();
    await heldFailure;
    await route.fulfill({
      status: 400,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Không thể đọc tài liệu thử." }),
    });
  });
  // A controlled progress response verifies the ETA and chunk-count display.
  // The API integration test separately verifies the server's real counts.
  await page.route("**/api/uploads/*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        stage: "embedding",
        size_bytes: 1800,
        pages: 0,
        text_chars: 1750,
        chunks: 8,
        embedded: 4,
        embedding_percent: 50,
        embedding_rate: 0.3,
        elapsed_seconds: 18,
        remaining_seconds: 14,
        model_loading: false,
        error: null,
      }),
    }),
  );
  await page.evaluate(() => {
    const nativeSend = XMLHttpRequest.prototype.send;
    window.restoreXHRSend = () => {
      XMLHttpRequest.prototype.send = nativeSend;
    };
    XMLHttpRequest.prototype.send = function (...args) {
      const result = nativeSend.apply(this, args);
      setTimeout(() => this.upload.dispatchEvent(new Event("load")), 100);
      return result;
    };
  });
  await page.locator("#file-input").setInputFiles({
    name: "du-lieu-minh-hoa.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("Failed document evidence. ".repeat(75)),
  });
  await page.waitForFunction(
    () => document.querySelector("#upload-work").textContent === "4/8 đoạn",
  );
  assert.equal(await page.locator("#upload-percent").textContent(), "50%");
  assert.equal(await page.locator("#upload-remaining").textContent(), "~00:14");
  assert.ok(
    (await page.locator("#upload-speed").textContent()).includes("đoạn/phút"),
  );
  await page.locator("#upload-status").screenshot({
    path: `${output}/estimate-panel.png`,
  });
  checks += 3;
  releaseFailure();
  await page.waitForFunction(
    () => document.querySelector("#upload-status").dataset.phase === "error",
  );
  assert.ok(
    (await page.locator("#upload-message").textContent()).includes(
      "Không thể đọc tài liệu thử",
    ),
  );
  assert.equal(await page.locator("#nav-doc-count").textContent(), "1");
  await page.screenshot({ path: `${output}/error-desktop.png` });
  checks += 2;

  await page.evaluate(() => window.restoreXHRSend());
  await page.unroute("**/api/uploads/*");
  await page.unroute("**/api/documents");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(
    () => document.querySelector("#sidebar").getBoundingClientRect().right <= 1,
  );
  await page.locator("#file-input").setInputFiles([
    {
      name: "tai-lieu-thu-hai.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("Thêm một tài liệu để kiểm tra tải nhiều tệp."),
    },
    {
      name: "khong-ho-tro.exe",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("unsupported"),
    },
  ]);
  await page.waitForFunction(
    () =>
      document.querySelector("#upload-status").dataset.phase === "complete" &&
      document.querySelector("#upload-count").textContent === "2/2",
  );
  assert.ok(
    (await page.locator("#upload-message").textContent()).includes("Đã thêm 1"),
  );
  assert.ok(
    (await page.locator("#upload-message").textContent()).includes("1 tệp lỗi"),
  );
  assert.equal(await page.locator("#nav-doc-count").textContent(), "2");
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
    true,
  );
  await page.screenshot({ path: `${output}/mixed-mobile.png` });
  checks += 4;

  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({ status: "passed", checks, javascriptErrors: errors }),
  );
} finally {
  await browser?.close();
  await backend.close();
}
