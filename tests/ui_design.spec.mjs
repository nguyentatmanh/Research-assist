import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";
import { startBackend } from "./backend.mjs";

const backend = await startBackend();
const output = "test-results/redesign";
await mkdir(output, { recursive: true });
let browser;
let checks = 0;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(backend.base);
  await page.waitForFunction(() =>
    document
      .querySelector("#connection-label")
      .textContent.includes("đã kết nối"),
  );
  assert.equal(
    await page.locator('#step-sources[aria-current="step"]').count(),
    1,
  );
  await page.screenshot({
    path: `${output}/research-desktop.png`,
    fullPage: true,
  });
  await page.click("#seed-button");
  await page.waitForFunction(
    () => document.querySelector("#nav-doc-count").textContent === "3",
  );
  assert.equal(
    await page.locator('#step-question[aria-current="step"]').count(),
    1,
  );
  assert.equal(await page.locator("#selected-doc-count").textContent(), "3/3");
  checks += 3;

  await page.keyboard.press("Control+k");
  assert.equal(await page.locator("#view-library").isVisible(), true);
  assert.equal(
    await page
      .locator("#library-search")
      .evaluate((e) => e === document.activeElement),
    true,
  );
  assert.equal(await page.locator("#library-total").textContent(), "3");
  await page.fill("#library-search", "self_rag");
  assert.equal(await page.locator("#document-table-body tr").count(), 1);
  assert.ok(
    (await page.locator("#document-table-body").textContent()).includes(
      "02_self_rag_notes.md",
    ),
  );
  await page.fill("#library-search", "nothing-matches-this-query");
  assert.equal(await page.locator(".table-empty").isVisible(), true);
  await page.fill("#library-search", "");
  assert.equal(await page.locator("#document-table-body tr").count(), 3);
  await page.screenshot({
    path: `${output}/library-desktop.png`,
    fullPage: true,
  });
  checks += 7;

  await page.click('[data-view="research"]');
  await page.locator(".guide-prompt").first().click();
  assert.ok(
    (await page.locator("#question").inputValue()).includes("đóng góp chính"),
  );
  assert.equal(
    await page.locator('#step-report[aria-current="step"]').count(),
    1,
  );
  // A real local demo run supplies evidence; filtering must preserve its citation numbering.
  await page.fill(
    "#question",
    "RAG và Self-RAG khác nhau như thế nào về truy xuất và kiểm tra câu trả lời?",
  );
  await page.click("#run-button");
  await page.waitForFunction(
    () => !document.querySelector("#results-section").hidden,
  );
  assert.equal(await page.locator("#activity-empty").isVisible(), false);
  assert.ok(
    (await page.locator("#activity-sources").textContent()).includes(
      "02_self_rag_notes.md",
    ),
  );
  await page.screenshot({
    path: `${output}/agents-desktop.png`,
    fullPage: true,
  });
  await page.click("#back-research");
  assert.equal(await page.locator("#step-report.complete").count(), 1);
  const count = await page.locator(".evidence-card").count();
  assert.ok(count > 1);
  await page.fill("#evidence-search", "nothing-matches-this-query");
  assert.equal(await page.locator(".evidence-card:visible").count(), 0);
  assert.equal(await page.locator("#evidence-no-match").isVisible(), true);
  await page.fill("#evidence-search", "02_self_rag_notes.md");
  assert.ok((await page.locator(".evidence-card:visible").count()) > 0);
  await page.locator(".evidence-card:visible").first().press("Enter");
  assert.equal(await page.locator("#source-dialog[open]").count(), 1);
  await page.keyboard.press("Escape");
  await page.fill("#evidence-search", "");
  assert.equal(await page.locator(".evidence-card:visible").count(), count);
  await page
    .locator("#results-section")
    .screenshot({ path: `${output}/report-desktop.png` });
  checks += 11;

  await page.click("#settings-button");
  await page.waitForSelector("#settings-dialog[open]");
  assert.equal(await page.locator(".route-item[data-agent] svg").count(), 5);
  await page.screenshot({
    path: `${output}/models-desktop.png`,
    fullPage: true,
  });
  await page.keyboard.press("Escape");
  checks++;

  const widths = [1440, 1280, 1024, 768, 640, 390, 320];
  for (const width of widths) {
    await page.setViewportSize({ width, height: 900 });
    // The document table can scroll within its own container; the page must reflow.
    for (const view of [
      "research",
      "library",
      "activity",
      "history",
      "experiments",
    ]) {
      if (width <= 800) await page.click("#mobile-menu");
      await page.click(`[data-view="${view}"]`);
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
        true,
        `${view} overflows at ${width}px`,
      );
    }
    checks += 5;
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.click("#mobile-menu");
  assert.equal(
    await page.locator("#mobile-menu").getAttribute("aria-expanded"),
    "true",
  );
  assert.equal(
    await page.locator(".content-shell").evaluate((e) => e.inert),
    true,
  );
  await page.keyboard.press("Escape");
  assert.equal(
    await page.locator("#mobile-menu").getAttribute("aria-expanded"),
    "false",
  );
  assert.equal(await page.locator("#sidebar").evaluate((e) => e.inert), true);
  assert.equal(
    await page
      .locator("#mobile-menu")
      .evaluate((e) => e === document.activeElement),
    true,
  );
  await page.click("#mobile-menu");
  await page.click('[data-view="research"]');
  await page.screenshot({
    path: `${output}/research-mobile.png`,
    fullPage: true,
  });
  checks += 5;
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      status: "passed",
      checks,
      widths,
      javascriptErrors: errors,
      screenshots: output,
    }),
  );
} finally {
  await browser?.close();
  await backend.close();
}
