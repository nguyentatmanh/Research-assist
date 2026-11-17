import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
import { startBackend } from "./backend.mjs";

const source = {
  id: "doc_scout",
  title: "Tài liệu nghiên cứu",
  filename: "scout.txt",
  kind: "txt",
  chunk_count: 1,
};
const question = "Tài liệu mô tả retrieval reranking như thế nào?";
const requirement = "Tác động của retrieval reranking trên dữ liệu tiếng Việt";
const expectedResult = {
  run_id: "run_scout",
  question,
  title: question,
  markdown: "Tài liệu có đề cập đến bước xếp hạng lại.",
  claims: [],
  evidence: [],
  checks: [],
  retrieved: [],
  warnings: [],
  events: [],
  rounds: 1,
  elapsed_ms: 16,
  llm_calls: 0,
  report_status: "partial",
  stop_reason: "completed",
  reported_claim_ids: [],
  requirements: [
    {
      id: "rq_1",
      text: requirement.replace("retrieval reranking", "retrieval\n reranking"),
    },
    { id: "rq_2", text: "Chi phí khi huấn luyện" },
    { id: "rq_3", text: "Tốc độ suy luận" },
  ],
  gaps: [requirement, "Chi phí khi huấn luyện", "Tốc độ suy luận"],
  gap_details: [
    { requirement_id: "rq_1", reason: "no_supported_evidence" },
    { requirement_id: "rq_2", reason: "budget_exhausted" },
    { requirement_id: "rq_3", reason: "coverage_unassessed" },
  ].map((gap) => ({ ...gap, scope: "retrieved_context" })),
};

const openAlex = {
  paper_id: "W123456",
  provider: "openalex",
  source: "openalex",
  providers: ["openalex"],
  title: "Evaluation of Vietnamese Retrieval",
  authors: ["A. Researcher"],
  published: "2024",
  year: 2024,
  url: "https://doi.org/10.1234/scout.1",
  doi: "10.1234/scout.1",
  abstract: "Methods and <b>results</b> for Vietnamese retrieval.",
  arxiv_id: null,
  import_available: false,
};
const arXiv = {
  paper_id: "1706.03762v7",
  arxiv_id: "1706.03762v7",
  provider: "arxiv",
  source: "arxiv",
  providers: ["arxiv", "openalex"],
  title: "Attention Is All You Need",
  authors: ["B. Researcher"],
  published: "2017",
  year: 2017,
  url: "https://arxiv.org/abs/1706.03762v7",
  import_available: true,
};

const backend = await startBackend();
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
  });
  for (const path of ["/", "/desk", "/studio"]) {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 960 },
    });
    const errors = [];
    const searches = [];
    const imports = [];
    let record;
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/api/documents", (route) =>
      route.fulfill({ json: { documents: [source] } }),
    );
    await page.route("**/api/runs", (route) => {
      if (route.request().method() !== "POST")
        return route.fulfill({ json: { runs: [] } });
      record = {
        id: expectedResult.run_id,
        status: "completed",
        request: route.request().postDataJSON(),
        scope_documents: [source],
        events: [],
        result: expectedResult,
      };
      return route.fulfill({ json: record });
    });
    await page.route("**/api/runs/run_scout", (route) =>
      route.fulfill({ json: record }),
    );
    await page.route("**/api/runs/run_scout/methodology", (route) =>
      route.fulfill({ json: { items: [] } }),
    );
    await page.route("**/api/scout/search", (route) => {
      searches.push(route.request().postDataJSON());
      return route.fulfill({
        json: {
          papers: [openAlex, arXiv],
          warnings: [],
          policy: "Metadata dùng để tìm bài.",
        },
      });
    });
    await page.route("**/api/papers/import", (route) => {
      imports.push(route.request().postDataJSON());
      return route.fulfill({ status: 201, json: { document: source } });
    });
    await page.goto(backend.base + path);

    if (path === "/") {
      await page.locator('.doc-item [aria-label="Chọn scout.txt"]').click();
      await page.locator("#promptInput").fill(question);
      await page.locator("#sendBtn").click();
    } else if (path === "/desk") {
      await page.waitForFunction(
        () => document.querySelector("#nav-doc-count").textContent === "1",
      );
      await page.locator("#choose-sources").click();
      await page
        .getByRole("checkbox", { name: "Chọn nguồn scout.txt" })
        .check();
      await page.locator("#scope-dialog .primary-button").click();
      await page.locator("#question").fill(question);
      await page.locator("#run-button").click();
      await page
        .locator("#report-content .scout-gap-action")
        .waitFor({ state: "attached" });
      await page.locator('[data-view="research"]').click();
    } else {
      await page.waitForFunction(
        () => document.querySelector("#doc-count").textContent === "1",
      );
      await page.locator("#select-all").click();
      await page.locator("#question").fill(question);
      await page.locator("#start").click();
    }

    const action = page.getByRole("button", { name: /^Tìm tài liệu bổ sung/ });
    await action.waitFor();
    assert.equal(await action.count(), 1, path);
    await action.click();
    const input = page.locator(path === "/" ? "#promptInput" : "#scout-query");
    assert.equal(await input.inputValue(), requirement);
    assert.ok(await input.evaluate((node) => node === document.activeElement));
    if (path === "/")
      assert.equal(
        await page.locator("#autoScoutToggleBtn").getAttribute("aria-pressed"),
        "true",
      );
    const query = requirement + " benchmark";
    await input.fill(query);
    assert.equal(searches.length, 0, path);
    await page
      .locator(path === "/" ? "#sendBtn" : '#scout-form button[type="submit"]')
      .click();

    const cardSelector = {
      "/": ".scout-paper-card",
      "/desk": ".scout-result",
      "/studio": ".paper-card",
    }[path];
    const openAlexCard = page
      .locator(cardSelector)
      .filter({ hasText: openAlex.title });
    await openAlexCard.waitFor();
    assert.deepEqual(searches, [{ query }]);
    await openAlexCard
      .getByText("OpenAlex · 2024 · A. Researcher", { exact: true })
      .waitFor();
    assert.equal(
      await openAlexCard.getByRole("button", { name: /Nhập PDF/ }).count(),
      0,
    );
    await openAlexCard.locator(".scout-abstract summary").click();
    await openAlexCard.getByText(openAlex.abstract, { exact: true }).waitFor();
    assert.equal(await openAlexCard.locator("b").count(), 0);
    assert.ok(await openAlexCard.locator(`a[href="${openAlex.url}"]`).count());
    const openAlexDownload = page.waitForEvent("download");
    await openAlexCard.getByRole("button", { name: /^Tải BibTeX/ }).click();
    const openAlexBib = await openAlexDownload;
    assert.match(openAlexBib.suggestedFilename(), /^scout-[a-z0-9]+\.bib$/);
    const entry = await readFile(await openAlexBib.path(), "utf8");
    assert.match(entry, /^@misc\{/);
    assert.ok(entry.includes("doi = {" + openAlex.doi + "}"));
    assert.ok(entry.includes("title = {" + openAlex.title + "}"));
    assert.ok(!entry.includes("journal ="));

    const arxivCard = page
      .locator(cardSelector)
      .filter({ hasText: arXiv.title });
    const arxivDownload = page.waitForEvent("download");
    await arxivCard.getByRole("button", { name: /^Tải BibTeX/ }).click();
    const preprint = await readFile(await (await arxivDownload).path(), "utf8");
    assert.match(preprint, /^@misc\{/);
    assert.ok(preprint.includes("eprint = {" + arXiv.arxiv_id + "}"));
    assert.ok(preprint.includes("archivePrefix = {arXiv}"));
    const importButton = arxivCard.getByRole("button", { name: /Nhập PDF/ });
    assert.equal(imports.length, 0, path);
    const imported = page.waitForResponse("**/api/papers/import");
    await importButton.click();
    await imported;
    assert.deepEqual(imports, [{ paper_id: arXiv.arxiv_id }]);
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log("OpenAlex Scout browser checks passed on /, /desk and /studio.");
} finally {
  await browser?.close();
  await backend.close();
}
