import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";
import { startBackend } from "./backend.mjs";
const backend = await startBackend();
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
  });
  const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await mkdir("test-results/desk", { recursive: true });
  await page.goto(backend.base);
  await page.waitForFunction(
    () =>
      document.querySelector("#apiTargetStatus").textContent ===
      "Chưa cấu hình",
  );
  await page.fill("#promptInput", "Bản nháp đang soạn");
  await page.click("#modelDropdownBtn");
  await page.click('[data-mode="gemini"]');
  const settings = page.locator("#agentMatrixPopover");
  const cloud = settings.locator('.target-form[data-target="api"]');
  await cloud.waitFor();
  assert.equal(new URL(page.url()).pathname, "/");
  assert.equal(
    await cloud.locator('input[type="password"]').getAttribute("required"),
    "",
  );
  await cloud.locator("#target-api-model").fill("test-cloud-model");
  await cloud.locator("#target-api-key").fill("test-key");
  await cloud.getByRole("button", { name: "Lưu cấu hình" }).click();
  await page.route("**/api/config/targets/api/test", (route) =>
    route.fulfill({
      json: { reachable: true, model_found: true, note: "Kiểm tra mẫu" },
    }),
  );
  await cloud.getByRole("button", { name: "Kiểm tra kết nối đã lưu" }).click();
  await cloud
    .getByText("Kết nối được · Kiểm tra mẫu", { exact: true })
    .waitFor();
  await page.waitForFunction(
    () =>
      document.querySelector("#apiTargetStatus").textContent ===
      "test-cloud-model",
  );
  await settings.getByRole("button", { name: "Thu gọn", exact: true }).click();
  assert.equal(await page.inputValue("#promptInput"), "Bản nháp đang soạn");
  assert.equal(new URL(page.url()).pathname, "/");
  await page.click('[data-mode="gemini"]');
  assert.deepEqual(
    await page
      .locator(".agent-model-select")
      .evaluateAll((nodes) => nodes.map((node) => node.value)),
    Array(5).fill("api"),
  );
  await settings.getByLabel("Model cho Writer Agent").selectOption("demo");
  assert.equal(
    await page.locator("#quickActivePill").textContent(),
    "Tùy biến",
  );
  const hybrid = settings.getByRole("button", { name: "Hybrid", exact: true });
  await hybrid.click();
  const cpu = settings.locator('.target-form[data-target="cpu"]');
  await cpu.locator("#target-cpu-model").fill("test-cpu-model");
  await cpu.getByRole("button", { name: "Lưu cấu hình" }).click();
  await page.waitForFunction(
    () =>
      document.querySelector("#cpuTargetStatus").textContent ===
      "test-cpu-model",
  );
  await hybrid.click();
  assert.equal(await hybrid.getAttribute("aria-pressed"), "true");
  assert.equal(await page.locator("#quickActivePill").textContent(), "Hybrid");
  assert.deepEqual(
    await page
      .locator(".agent-model-select")
      .evaluateAll((nodes) =>
        Object.fromEntries(
          nodes.map((node) => [node.dataset.agent, node.value]),
        ),
      ),
    {
      retrieval: "api",
      reader: "cpu",
      summarizer: "api",
      fact_checker: "cpu",
      writer: "api",
    },
  );
  await page.screenshot({ path: "test-results/desk/model-matrix.png" });
  await page.setViewportSize({ width: 390, height: 600 });
  await page.click("#toggleSidebarBtn");
  await settings.locator('[data-mode="gpu"]').click();
  await settings.locator("#target-gpu-model").waitFor();
  const bounds = await settings.boundingBox();
  assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= 390);
  assert.ok(bounds.y >= 0 && bounds.y + bounds.height <= 600);
  await page.screenshot({ path: "test-results/desk/model-matrix-mobile.png" });
  await page.keyboard.press("Escape");
  await settings.waitFor({ state: "hidden" });
  assert.equal(await page.inputValue("#promptInput"), "Bản nháp đang soạn");
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(backend.base + "/desk");
  await page.waitForFunction(() =>
    document
      .querySelector("#connection-label")
      .textContent.includes("đã kết nối"),
  );
  await page.screenshot({
    path: "test-results/desk/research.png",
    fullPage: true,
  });
  await page.click('[data-view="settings"]');
  await page.waitForSelector("#target-cpu-model", { state: "attached" });
  assert.equal(await page.locator(".agent-mode-row").count(), 5);
  await page.locator(".target-editor").nth(1).locator("summary").click();
  await page.fill("#target-cpu-model", "test-model");
  await page
    .locator('.target-form[data-target="cpu"] button[type="submit"]')
    .click();
  await page.waitForFunction(
    () => !document.querySelector('[data-preset="cpu"]').disabled,
  );
  await page.click('[data-preset="demo"]');
  await page.screenshot({
    path: "test-results/desk/settings.png",
    fullPage: true,
  });
  await page.click('[data-view="research"]');
  await page.fill("#question", "/so-l");
  await page.keyboard.press("Enter");
  assert.match(await page.inputValue("#question"), /số liệu/);
  await page.click("#seed-button");
  await page.waitForFunction(
    () => document.querySelector("#nav-doc-count").textContent === "3",
  );
  await page.fill(
    "#question",
    "How does RAG use source evidence to support generated answers?",
  );
  await page.click("#run-button");
  await page.waitForFunction(
    () => !document.querySelector("#results-section").hidden,
  );
  await page.click("#back-research");
  await page.click('[data-tab="methodology"]');
  await page.waitForFunction(
    () => document.querySelectorAll(".methodology-item").length === 6,
  );
  await page.screenshot({
    path: "test-results/desk/audit.png",
    fullPage: true,
  });
  await page.click('[data-tab="report"]');
  await page.pdf({
    path: "test-results/desk/report.pdf",
    format: "A4",
    printBackground: true,
  });
  const download = page.waitForEvent("download");
  await page.click("#export-json");
  assert.ok((await download).suggestedFilename().endsWith(".json"));
  await page.setViewportSize({ width: 390, height: 844 });
  for (const view of [
    "research",
    "library",
    "activity",
    "history",
    "experiments",
    "settings",
  ]) {
    await page.click("#mobile-menu");
    await page.click(`[data-view="${view}"]`);
    await page.waitForFunction(
      () =>
        document.querySelector("#sidebar").getBoundingClientRect().right <= 0,
    );
    assert.equal(await page.locator(`#view-${view}`).isVisible(), true);
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
      `horizontal overflow: ${view}`,
    );
  }
  await page.screenshot({
    path: "test-results/desk/settings-mobile.png",
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  console.log(
    "PASS: Research Desk browser, six mobile views, target editor, command picker, Autonomous run, audit, JSON download, PDF render, screenshots, no JS errors.",
  );
} finally {
  if (browser) await browser.close();
  await backend.close();
}
