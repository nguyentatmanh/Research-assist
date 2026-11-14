import assert from "node:assert/strict";
import { chromium } from "playwright";
import { startBackend } from "./backend.mjs";

const backend = await startBackend();
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({
    viewport: { width: 1280, height: 800 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let searchMode = "results",
    importMode = "error",
    imports = 0;
  const title = '<img src=x onerror="window.injected=true"> Research evidence';
  const paper = {
    paper_id: "1706.03762v7",
    title,
    authors: ["A. Researcher"],
    published: "2023-01-01",
    url: "https://arxiv.org/abs/1706.03762v7",
  };
  await page.route("**/api/papers/search", async (route) => {
    assert.equal(route.request().method(), "POST");
    assert.equal(route.request().postDataJSON().query, "attention evidence");
    await route.fulfill({
      status: searchMode === "error" ? 429 : 200,
      json:
        searchMode === "error"
          ? { detail: "Vui lòng thử lại sau." }
          : searchMode === "empty"
            ? []
            : [paper],
    });
  });
  await page.route("**/api/papers/import", async (route) => {
    imports++;
    assert.deepEqual(route.request().postDataJSON(), {
      paper_id: paper.paper_id,
    });
    if (importMode === "error") {
      await route.fulfill({
        status: 502,
        json: { detail: "Tải PDF chưa thành công." },
      });
      return;
    }
    // Real library refresh; the network discovery/import responses are controlled fixtures.
    const stored = await page.request.post(backend.base + "/api/documents", {
      multipart: {
        file: {
          name: "Imported research.txt",
          mimeType: "text/plain",
          buffer: Buffer.from(
            "Imported research provides evidence for document retrieval.",
          ),
        },
      },
    });
    assert.equal(stored.status(), 201);
    await route.fulfill({ status: 201, json: await stored.json() });
  });
  await page.goto(backend.base + "/desk");
  await page.locator('[data-view="library"]').click();
  await page
    .locator(".paper-discovery:has(#paper-search-form) summary")
    .click();
  const search = page.locator("#paper-search-form button");
  await page.locator("#paper-query").fill("attention evidence");
  await search.click();
  await page.waitForFunction(
    () =>
      document.querySelector("#paper-status").textContent === "Tìm được 1 bài.",
  );
  assert.equal(await page.locator("#paper-results a").textContent(), title);
  assert.equal(await page.locator("#paper-results img").count(), 0);
  assert.equal(await page.evaluate(() => window.injected), undefined);
  assert.equal(
    await page.locator("#paper-results a").getAttribute("href"),
    paper.url,
  );
  assert.equal(await page.locator("#nav-doc-count").textContent(), "0");
  const add = page.locator("#paper-results button");
  await add.click();
  await page.waitForFunction(
    () =>
      document.querySelector("#paper-status").textContent ===
      "Tải PDF chưa thành công.",
  );
  assert.equal(await add.isEnabled(), true);
  importMode = "success";
  await add.click();
  await page.waitForFunction(
    () => document.querySelector("#nav-doc-count").textContent === "1",
  );
  assert.equal(await add.isDisabled(), true);
  assert.equal(imports, 2);
  assert.ok(
    (await page.locator("#document-table-body").textContent()).includes(
      "Imported research",
    ),
  );
  assert.equal(await page.locator("#library-selected").textContent(), "0");
  searchMode = "empty";
  await search.click();
  await page.waitForFunction(() =>
    document
      .querySelector("#paper-status")
      .textContent.startsWith("Chưa tìm thấy"),
  );
  assert.equal(await page.locator("#paper-results li").count(), 0);
  searchMode = "error";
  await search.click();
  await page.waitForFunction(
    () =>
      document.querySelector("#paper-status").textContent ===
      "Vui lòng thử lại sau.",
  );
  assert.equal(await search.isEnabled(), true);
  searchMode = "results";
  paper.title = "Attention Is All You Need";
  await search.click();
  await page.waitForFunction(
    () =>
      document.querySelector("#paper-status").textContent === "Tìm được 1 bài.",
  );
  for (const width of [1280, 640, 390]) {
    await page.setViewportSize({ width, height: 800 });
    await page.locator("#paper-query").click();
    await search.click({ trial: true });
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth + 1,
      ),
      false,
    );
    assert.equal(await search.isVisible(), true);
  }
  // DOI discovery is a metadata lookup; it does not import unreviewed publisher URLs.
  paper.source = "crossref";
  paper.doi = "10.1234/research.1";
  paper.import_available = false;
  paper.url = "javascript:window.injected=true";
  await page.locator("#paper-source").selectOption("crossref");
  assert.equal(await page.locator("#paper-results li").count(), 0);
  await search.click();
  await page.waitForFunction(
    () =>
      document.querySelector("#paper-status").textContent === "Tìm được 1 bài.",
  );
  assert.equal(await page.locator("#paper-results button").count(), 0);
  assert.equal(
    await page.locator("#paper-results a").getAttribute("href"),
    "https://doi.org/" + encodeURIComponent(paper.doi),
  );
  assert.equal(await page.locator("#nav-doc-count").textContent(), "1");
  assert.equal(imports, 2);
  if (process.env.UI_SCREENSHOT)
    await page.screenshot({ path: process.env.UI_SCREENSHOT, fullPage: true });
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      status: "passed",
      feature: "paper discovery",
      javascriptErrors: errors,
      coverage: [
        "safe text rendering",
        "import retry",
        "library refresh",
        "scope unchanged",
        "empty results",
        "rate limit",
        "responsive widths",
      ],
    }),
  );
} finally {
  await browser?.close();
  await backend.close();
}
