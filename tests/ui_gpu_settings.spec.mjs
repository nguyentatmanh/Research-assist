import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { startBackend } from "./backend.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const key = "ui-test-private-" + "x".repeat(32);
const model = "research-e5-" + "a".repeat(32);
const gpu = createServer(async (request, response) => {
  response.setHeader("Content-Type", "application/json");
  if (request.headers.authorization !== `Bearer ${key}`) {
    response.writeHead(401).end(JSON.stringify({ detail: "Wrong key" }));
    return;
  }
  if (request.url === "/v1/info") {
    response.end(
      JSON.stringify({
        model,
        family: "e5",
        protocol: "research-embedding-v1",
        dimension: 2,
        max_seq_length: 512,
        normalized: true,
      }),
    );
    return;
  }
  const parts = [];
  for await (const part of request) parts.push(part);
  const input = JSON.parse(Buffer.concat(parts)).input;
  response.end(
    JSON.stringify({
      model,
      data: input.map((_, index) => ({
        index,
        embedding: [0.6, 0.8],
      })),
    }),
  );
});
await new Promise((resolve) => gpu.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${gpu.address().port}`;
const backend = await startBackend();
const output = process.env.UI_OUTPUT_DIR || "test-results/gpu-settings";
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
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(backend.base + "/desk");
  await page.waitForFunction(() =>
    document
      .querySelector("#connection-label")
      .textContent.includes("đã kết nối"),
  );
  await page.click('[data-view="research"]');
  await page.click("#seed-button");
  await page.waitForFunction(
    () => document.querySelector("#nav-doc-count").textContent === "3",
  );
  await page.click("#settings-button");
  await page.waitForFunction(
    () => !document.querySelector("#gpu-fields").disabled,
  );
  await page.selectOption("#gpu-family", "qwen3-4b");
  assert.equal(
    await page.inputValue("#gpu-tokenizer"),
    "Qwen/Qwen3-Embedding-4B",
  );
  await page.selectOption("#gpu-family", "qwen3-0.6b");
  assert.equal(
    await page.inputValue("#gpu-tokenizer"),
    "Qwen/Qwen3-Embedding-0.6B",
  );
  await page.selectOption("#gpu-family", "e5");
  assert.equal(
    await page.inputValue("#gpu-tokenizer"),
    "intfloat/multilingual-e5-base",
  );
  checks++;
  await page.fill("#gpu-url", url);
  await page.fill("#gpu-key", key);
  await page.click("#gpu-test");
  await page.waitForFunction(() =>
    document
      .querySelector("#gpu-feedback")
      .textContent.includes("Kết nối thành công"),
  );
  assert.equal(await page.inputValue("#gpu-model"), model);
  assert.equal(
    (
      await (
        await page.request.get(backend.base + "/api/embedding/remote")
      ).json()
    ).active,
    false,
  );
  checks++;
  await page.click("#gpu-save");
  await page.waitForFunction(() =>
    document
      .querySelector("#gpu-feedback")
      .textContent.includes("Đã lưu và áp dụng"),
  );
  assert.equal(await page.inputValue("#gpu-key"), "");
  assert.ok(await page.locator("#gpu-index-panel").isVisible());
  assert.ok(
    (await page.locator("#gpu-index-note").textContent()).includes(
      "vector của cấu hình cũ",
    ),
  );
  const persisted = await page.request.get(
    backend.base + "/api/embedding/remote",
  );
  assert.ok(!(await persisted.text()).includes(key));
  assert.equal(
    (await page.request.get(backend.base + "/api/config")).status(),
    200,
  );
  checks++;
  await page
    .locator(".gpu-settings")
    .screenshot({ path: `${output}/gpu-settings-desktop.png` });
  await page.reload();
  await page.waitForFunction(() =>
    document
      .querySelector("#connection-label")
      .textContent.includes("đã kết nối"),
  );
  await page.click("#settings-button");
  await page.waitForFunction(
    () => !document.querySelector("#gpu-fields").disabled,
  );
  assert.equal(await page.inputValue("#gpu-url"), url + "/v1");
  assert.equal(await page.inputValue("#gpu-key"), "");
  assert.ok(
    (await page.locator("#gpu-key-note").textContent()).includes("Đã có key"),
  );
  assert.equal(
    await page.evaluate((k) => JSON.stringify(localStorage).includes(k), key),
    false,
  );
  checks++;
  await page.fill("#gpu-key", "wrong-" + "z".repeat(32));
  await page.click("#gpu-test");
  await page.waitForFunction(
    () => document.querySelector("#gpu-feedback").dataset.kind === "error",
  );
  assert.ok(
    (await page.locator("#gpu-feedback").textContent()).includes(
      "Key truy cập GPU",
    ),
  );
  assert.equal(await page.locator("#gpu-key").getAttribute("type"), "password");
  checks++;
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#gpu-url").scrollIntoViewIfNeeded();
  assert.ok(
    await page.evaluate(() => {
      const d = document.querySelector("#settings-dialog");
      return d.scrollWidth <= d.clientWidth + 1;
    }),
  );
  await page.screenshot({ path: `${output}/gpu-settings-mobile.png` });
  checks++;
  await page.click("#gpu-restore");
  await page.waitForFunction(() =>
    document
      .querySelector("#gpu-feedback")
      .textContent.includes("cấu hình embedding gốc"),
  );
  assert.equal(await page.inputValue("#gpu-key"), "");
  assert.equal(
    (
      await (
        await page.request.get(backend.base + "/api/embedding/remote")
      ).json()
    ).active,
    false,
  );
  await page.goto(backend.base);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.fill("#promptInput", "Câu hỏi đang soạn");
  await page.click("#modelDropdownBtn");
  await page.click('[data-config-target="embedding"]');
  await page.waitForFunction(
    () => !document.querySelector("#gpu-fields").disabled,
  );
  assert.equal(new URL(page.url()).pathname, "/");
  await page.fill("#gpu-url", url);
  await page.fill("#gpu-key", key);
  await page.click("#gpu-save");
  await page.waitForFunction(() =>
    document
      .querySelector("#gpu-feedback")
      .textContent.includes("Đã lưu và áp dụng"),
  );
  assert.equal(await page.inputValue("#gpu-key"), "");
  await page.click("#closeModelSettings");
  assert.equal(await page.inputValue("#promptInput"), "Câu hỏi đang soạn");
  await page.click('[data-config-target="embedding"]');
  await page.waitForFunction(
    () => !document.querySelector("#gpu-fields").disabled,
  );
  assert.equal(await page.inputValue("#gpu-url"), url + "/v1");
  assert.equal(await page.inputValue("#gpu-key"), "");
  await page.click("#gpu-restore");
  await page.waitForFunction(() =>
    document
      .querySelector("#gpu-feedback")
      .textContent.includes("cấu hình embedding gốc"),
  );
  checks++;
  assert.deepEqual(errors, []);
  checks++;
  console.log(
    JSON.stringify({
      status: "passed",
      checks,
      javascriptErrors: errors,
      screenshots: output,
    }),
  );
} finally {
  if (browser) await browser.close();
  await backend.close();
  await new Promise((resolve) => gpu.close(resolve));
}
