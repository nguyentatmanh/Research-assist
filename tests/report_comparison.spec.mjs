import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
import { startBackend } from "./backend.mjs";

const documents = [
  {
    id: "doc_transformer",
    title: "Transformer",
    filename: "transformer.txt",
    kind: "txt",
    chunk_count: 1,
  },
  {
    id: "doc_parser",
    title: "Phân tích cú pháp",
    filename: "parser.txt",
    kind: "txt",
    chunk_count: 1,
  },
];
const evidence = documents.map((doc, i) => {
  const quote = [
    "Transformer đạt 28.4 BLEU.",
    "Thí nghiệm này đánh giá độ chính xác phân tích cú pháp.",
  ][i];
  return {
    id: `ev_${i}`,
    chunk_id: `ch_${i}`,
    document_id: doc.id,
    document_title: doc.title,
    filename: doc.filename,
    quote,
    relevance: "Chỉ số được bài báo nêu.",
    page: null,
    section: "Kết quả",
    start: 0,
    end: quote.length,
  };
});
const question = "Có thể lấy điểm BLEU so trực tiếp với độ chính xác không?";
const claims = [
  { id: "claim_1", text: "Transformer đạt 28.4 BLEU.", evidence_ids: ["ev_0"] },
  {
    id: "claim_2",
    text: "Hai bài dùng thước đo khác nhau.",
    evidence_ids: ["ev_0", "ev_1"],
  },
];
const result = {
  run_id: "run_comparison",
  question,
  title: question,
  markdown: [
    "## So sánh tài liệu",
    "",
    "| Tiêu chí | Transformer | Phân tích cú pháp |",
    "| --- | --- | --- |",
    String.raw`| Tên \| chỉ số\\| 28.4 BLEU \[1\] [1] | Chưa đủ bằng chứng |`,
    "",
    "## Nhận định tổng hợp",
    "",
    "Hai bài dùng thước đo khác nhau. [1] [2]",
    "",
    "## Nguồn trích dẫn",
    "",
    "[1] transformer.txt",
    "[2] parser.txt",
  ].join("\n"),
  claims,
  evidence,
  checks: claims.map((claim) => ({
    claim_id: claim.id,
    verdict: "supported",
    explanation: "Có nguồn.",
    evidence_ids: claim.evidence_ids,
  })),
  citation_evidence_ids: evidence.map((item) => item.id),
  retrieved: [],
  warnings: [],
  events: [],
  rounds: 1,
  elapsed_ms: 16,
  llm_calls: 0,
  report_status: "partial",
  stop_reason: "completed",
  reported_claim_ids: claims.map((claim) => claim.id),
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
    await page.addInitScript(() => {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: { writeText: async (text) => (window.copiedTable = text) },
      });
    });
    const errors = [];
    const submissions = [];
    let record;
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/api/documents", (route) =>
      route.fulfill({ json: { documents } }),
    );
    await page.route("**/api/documents/doc_transformer/text?*", (route) => {
      const query = new URL(route.request().url()).searchParams;
      assert.equal(query.get("evidence_id"), evidence[0].id);
      return route.fulfill({
        json: {
          text: evidence[0].quote,
          highlight_start: 0,
          highlight_end: evidence[0].end,
        },
      });
    });
    await page.route("**/api/runs", (route) => {
      if (route.request().method() !== "POST")
        return route.fulfill({ json: { runs: [] } });
      const request = route.request().postDataJSON();
      submissions.push(request);
      record = {
        id: result.run_id,
        status: "completed",
        request,
        events: [],
        result,
        scope_documents: documents,
      };
      return route.fulfill({ status: 201, json: record });
    });
    await page.route("**/api/runs/run_comparison", (route) =>
      route.fulfill({ json: record }),
    );
    await page.route("**/api/runs/run_comparison/methodology", (route) =>
      route.fulfill({ json: { items: [] } }),
    );
    await page.goto(backend.base + path);

    if (path === "/") {
      await page
        .locator('.doc-item [aria-label="Chọn transformer.txt"]')
        .click();
      await page.locator('.doc-item [aria-label="Chọn parser.txt"]').click();
      await page
        .locator("#promptInput")
        .fill("/so-sanh Chỉ số thực nghiệm trong hai tài liệu.");
      await page.locator("#sendBtn").click();
      await page.locator(".workspace-report table").waitFor();
      assert.equal(submissions[0].style, "comparison");
      assert.match(submissions[0].display_question, /^\/so-sanh/);
      await page.locator("#promptInput").fill(question);
      await page.locator("#sendBtn").click();
      await page.waitForFunction(
        () => document.querySelectorAll(".workspace-report table").length === 2,
      );
      assert.equal(submissions[1].style, "comparison");
      assert.equal(submissions[1].question, question);
    } else if (path === "/desk") {
      await page.waitForFunction(
        () => document.querySelector("#nav-doc-count").textContent === "2",
      );
      await page.locator("#choose-sources").click();
      await page
        .getByRole("checkbox", { name: "Chọn nguồn transformer.txt" })
        .check();
      await page
        .getByRole("checkbox", { name: "Chọn nguồn parser.txt" })
        .check();
      await page.locator("#scope-dialog .primary-button").click();
      await page.locator("#question").fill(question);
      await page.locator("#advanced-toggle").click();
      await page.locator("#report-style").selectOption("comparison");
      await page.locator("#run-button").click();
      await page
        .locator("#report-content table")
        .waitFor({ state: "attached" });
      await page.locator('[data-view="research"]').click();
    } else {
      await page.waitForFunction(
        () => document.querySelector("#doc-count").textContent === "2",
      );
      await page.locator("#select-all").click();
      await page
        .locator("#scope-label")
        .filter({ hasText: "2 tài liệu" })
        .waitFor();
      await page.locator("#question").fill(question);
      await page.locator("#research-form summary").click();
      await page.locator("#style").selectOption("comparison");
      await page.locator("#start").click();
    }
    const root = {
      "/": ".workspace-report",
      "/desk": "#report-content",
      "/studio": "#report-panel",
    }[path];
    const report = page.locator(root).last();
    const table = report.getByRole("table");
    await table.waitFor();
    assert.equal(submissions[0].style, "comparison");
    assert.equal(await table.getByRole("columnheader").count(), 3);
    assert.equal(
      await table.getByRole("rowheader").textContent(),
      "Tên | chỉ số\\",
    );
    const first = table.getByRole("cell").first();
    assert.equal(await first.textContent(), "28.4 BLEU [1] [1]");
    assert.equal(await first.getByRole("button").count(), 1);
    assert.equal(
      await table.getByRole("cell").last().textContent(),
      "Chưa đủ bằng chứng",
    );
    assert.equal(
      await report.getByRole("button", { name: /\[2\]/ }).count(),
      1,
      `${path}: ${await report.innerText()}`,
    );

    await page.setViewportSize({ width: 375, height: 780 });
    const dimensions = await report
      .locator(".report-table-scroll")
      .evaluate((node) => ({
        wrapper: node.clientWidth,
        table: node.scrollWidth,
        display: getComputedStyle(node).overflowX,
        viewport: document.documentElement.clientWidth,
      }));
    assert.ok(dimensions.table > dimensions.wrapper);
    assert.ok(dimensions.wrapper <= dimensions.viewport);
    assert.equal(dimensions.display, "auto");
    assert.ok(
      await report
        .locator(".report-table-actions")
        .evaluate((node) => node.scrollWidth <= node.clientWidth),
    );

    await page.setViewportSize({ width: 1440, height: 960 });
    await report
      .getByRole("button", { name: "Sao chép bảng Markdown" })
      .click();
    await report.getByText("Đã sao chép bảng.", { exact: true }).waitFor();
    assert.equal(
      await page.evaluate(() => window.copiedTable),
      result.markdown.split("\n").slice(2, 5).join("\n"),
    );
    const downloaded = page.waitForEvent("download");
    await report.getByRole("button", { name: "Xuất CSV" }).click();
    const csv = await downloaded;
    assert.equal(csv.suggestedFilename(), "bang-so-sanh.csv");
    assert.equal(
      await readFile(await csv.path(), "utf8"),
      '\uFEFF"Tiêu chí","Transformer","Phân tích cú pháp"\r\n' +
        '"Tên | chỉ số\\","28.4 BLEU [1] [1]","Chưa đủ bằng chứng"\r\n',
    );

    await first.getByRole("button").click();
    if (path === "/") {
      assert.equal(
        await page.locator("#sourceContextQuote").textContent(),
        evidence[0].quote,
      );
    } else if (path === "/desk") {
      await page.locator("#source-dialog[open]").waitFor();
      assert.equal(
        await page.locator(".source-detail-quote").textContent(),
        evidence[0].quote,
      );
    } else {
      await page.locator("#ledger-panel").waitFor();
      assert.equal(
        await page.locator(".ledger-evidence:focus blockquote").textContent(),
        evidence[0].quote,
      );
    }
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log(
    "Comparison report browser checks passed on /, /desk and /studio.",
  );
} finally {
  await browser?.close();
  await backend.close();
}
