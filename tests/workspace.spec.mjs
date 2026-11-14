import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";
import { startBackend } from "./backend.mjs";

function samplePdf() {
  const text =
    "BT /F1 14 Tf 50 720 Td (The controlled experiment used exactly 50 samples for the research evaluation.) Tj ET";
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R 6 0 R 7 0 R 8 0 R 9 0 R 10 0 R] /Count 6 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 640 800] >>",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    `<< /Length ${text.length} >>\nstream\n${text}\nendstream`,
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 640 800] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
    ...Array(4).fill("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 640 800] >>"),
  ];
  let pdf = "%PDF-1.4\n";
  const offsets = objects.map((item, index) => {
    const start = pdf.length;
    pdf += `${index + 1} 0 obj\n${item}\nendobj\n`;
    return `${String(start).padStart(10, "0")} 00000 n \n`;
  });
  return Buffer.from(
    pdf +
      `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n` +
      offsets.join("") +
      `trailer\n<< /Root 1 0 R /Size ${objects.length + 1} >>\nstartxref\n${pdf.length}\n%%EOF\n`,
  );
}

const backend = await startBackend();
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 960 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await mkdir("test-results/workspace", { recursive: true });
  await page.goto(backend.base);
  await page.waitForFunction(
    () => document.querySelector(".agent-model-select").options.length === 5,
  );
  await page.screenshot({
    path: "test-results/workspace/desktop.png",
    fullPage: true,
  });
  const composer = page.locator("#promptInput");
  const commandMenu = page.locator("#command-popup");
  const options = commandMenu.getByRole("option");
  await composer.fill("/");
  assert.equal(await options.count(), 16);
  await composer.press("ArrowDown");
  assert.match(await options.nth(1).getAttribute("aria-selected"), /^true$/);
  await composer.press("Tab");
  assert.equal(await composer.inputValue(), "/so-sanh ");
  assert.equal(await commandMenu.isVisible(), false);
  await composer.fill("/khong-co-lenh");
  assert.match(await commandMenu.textContent(), /Không có lệnh phù hợp/);
  await composer.press("Escape");
  assert.equal(await commandMenu.isVisible(), false);
  await composer.fill("/cpu-max");
  await composer.press("Enter");
  assert.match(
    await page.locator("#workspaceStatus").textContent(),
    /chưa được cấu hình/,
  );
  await composer.fill("/cpu-min");
  await composer.press("Enter");
  assert.deepEqual(
    await page
      .locator(".agent-model-select")
      .evaluateAll((nodes) => nodes.map((node) => node.value)),
    Array(5).fill("demo"),
  );
  await composer.fill("/tim-bai");
  await composer.press("Enter");
  assert.equal(
    await page.locator("#autoScoutToggleBtn").getAttribute("aria-pressed"),
    "true",
  );
  await page.route("**/api/scout/search", (route) =>
    route.fulfill({ json: { papers: [], warnings: [] } }),
  );
  await composer.fill("RAG");
  const scoutRequest = page.waitForRequest((request) =>
    request.url().endsWith("/api/scout/search"),
  );
  await composer.press("Enter");
  assert.equal((await scoutRequest).postDataJSON().query, "RAG");
  await page
    .getByText("Không tìm thấy bài phù hợp.", { exact: true })
    .waitFor();
  await composer.fill("/tóm");
  await commandMenu.getByRole("option", { name: /Tóm tắt tài liệu/ }).click();
  assert.equal(await composer.inputValue(), "/tom-tat ");
  assert.equal(
    await page.locator("#autoScoutToggleBtn").getAttribute("aria-pressed"),
    "false",
  );
  await page.locator("#pdfFileInput").setInputFiles({
    name: "evidence.txt",
    mimeType: "text/plain",
    buffer: Buffer.from(
      "RAG uses retrieved source evidence to support generated answers. The experiment used 50 samples. Evidence verification checks claims against source passages.",
    ),
  });
  await page.waitForFunction(
    () => document.querySelector("#docCountBadge").textContent === "1",
  );
  await page.locator(".doc-item input").uncheck();
  await page.fill(
    "#promptInput",
    "How does RAG use source evidence to support answers?",
  );
  await page.click("#sendBtn");
  await page.waitForFunction(() =>
    document
      .querySelector("#workspaceStatus")
      .textContent.includes("Chọn ít nhất"),
  );
  await page.locator(".doc-item input").check();
  await composer.fill("/so-lieu cỡ mẫu của thí nghiệm");
  const runRequest = page.waitForRequest(
    (request) =>
      request.url().endsWith("/api/runs") && request.method() === "POST",
  );
  await composer.press("Enter");
  const payload = (await runRequest).postDataJSON();
  assert.match(
    payload.question,
    /Trích xuất số liệu.*\n\ncỡ mẫu của thí nghiệm/,
  );
  assert.equal(payload.scope_mode, "selected");
  assert.equal(payload.document_ids.length, 1);
  assert.equal(payload.display_question, "/so-lieu cỡ mẫu của thí nghiệm");
  await page.waitForSelector(".workspace-report");
  const markdown = page.waitForEvent("download");
  await composer.fill("/xuat-bao-cao");
  await composer.press("Enter");
  assert.match((await markdown).suggestedFilename(), /\.md$/);
  await page.getByRole("button", { name: "Mở báo cáo", exact: true }).click();
  await page.waitForFunction(
    () => getComputedStyle(document.querySelector("#sidecar")).opacity === "1",
  );
  await page.screenshot({
    path: "test-results/workspace/report.png",
    fullPage: true,
  });
  assert.doesNotMatch(
    await page.locator("#sidecarReportPane").textContent(),
    /Ghi chú nghiên cứu|Mức đáp ứng câu hỏi/u,
  );
  await page.click("#closeSidecarBtn");
  await composer.fill("How does RAG use source evidence to support answers?");
  await composer.press("Enter");
  const textReport = page.locator(".workspace-report").nth(1);
  await textReport
    .getByRole("button", { name: "Trích dẫn đối soát", exact: true })
    .click();
  await page
    .locator("#sidecarEvidenceList .workspace-evidence-link")
    .first()
    .click();
  await page.locator("#docMdViewer mark").waitFor();
  assert.match(await page.locator("#docMdViewer").textContent(), /50 samples/u);
  await page.click("#closeSidecarBtn");
  await page.locator("#pdfFileInput").setInputFiles({
    name: "experiment.pdf",
    mimeType: "application/pdf",
    buffer: samplePdf(),
  });
  await page.getByRole("checkbox", { name: "Chọn experiment.pdf" }).waitFor();
  await page.getByRole("checkbox", { name: "Chọn evidence.txt" }).uncheck();
  await composer.fill(
    "How many samples did the controlled research experiment use?",
  );
  const highlighted = page.waitForResponse(
    (r) =>
      r.url().includes("/pages/2/preview?") && r.url().includes("evidence_id="),
  );
  await composer.press("Enter");
  const pdfReport = page.locator(".workspace-report").nth(2);
  await pdfReport
    .getByRole("button", { name: "Tài liệu gốc", exact: true })
    .click();
  const previewImage = (number) =>
    page.locator(`.source-pdf-page[data-page="${number}"] > img`);
  const pdfViewport = page.locator("#sourceViewport");
  await previewImage(1).waitFor();
  assert.equal(await previewImage(6).count(), 0);
  const firstPageHeight = (await previewImage(1).boundingBox()).height;
  await pdfViewport.hover();
  await page.mouse.wheel(0, firstPageHeight + 40);
  await page.waitForFunction(
    () => document.querySelector("#sourcePageInput").value === "2",
  );
  await previewImage(2).waitFor();
  assert.equal(await previewImage(1).count(), 1);
  await page.click("#tabEvidenceBtn");
  await page.click("#tabPdfBtn");
  assert.equal(await page.inputValue("#sourcePageInput"), "2");
  await page.click("#pdfPrevPageBtn");
  await page.waitForFunction(
    () => document.querySelector("#sourceViewport").scrollTop === 0,
  );
  await page.click("#tabEvidenceBtn");
  await page
    .locator("#sidecarEvidenceList .workspace-evidence-link")
    .first()
    .click();
  assert.equal((await highlighted).headers()["x-source-highlight"], "exact");
  await previewImage(2).waitFor();
  assert.equal(await page.inputValue("#sourcePageInput"), "2");
  assert.ok(await pdfViewport.evaluate((node) => node.scrollTop > 0));
  await page.screenshot({ path: "test-results/workspace/pdf-citation.png" });
  await page.click("#pdfNextPageBtn");
  await previewImage(3).waitFor();
  assert.equal(await page.inputValue("#sourcePageInput"), "3");
  const pageNumber = page.locator("#sourcePageInput");
  await pageNumber.fill("6");
  await pageNumber.press("Tab");
  await previewImage(6).waitFor();
  assert.equal(await pageNumber.inputValue(), "6");
  await page.click("#closeSidecarBtn");
  await page.click("#modelDropdownBtn");
  await page
    .locator('.agent-model-select[data-agent="writer"]')
    .selectOption("demo");
  await page.reload();
  await page.waitForFunction(
    () =>
      document.querySelector('.agent-model-select[data-agent="writer"]')
        .value === "demo",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload();
  await page.waitForFunction(
    () => document.querySelector(".agent-model-select").options.length === 5,
  );
  assert.equal(
    await page.locator("#sidebar").evaluate((node) => node.inert),
    true,
  );
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  );
  await page.screenshot({
    path: "test-results/workspace/mobile.png",
    fullPage: true,
  });
  await composer.fill("/");
  const menuBounds = await commandMenu.boundingBox();
  assert.ok(menuBounds.x >= 0 && menuBounds.x + menuBounds.width <= 390);
  assert.ok(menuBounds.y >= 0 && menuBounds.y + menuBounds.height < 844);
  await page.screenshot({ path: "test-results/workspace/commands-mobile.png" });
  await composer.fill("/clear");
  await commandMenu.getByRole("option", { name: /Xóa ô nhập/ }).click();
  assert.equal(await composer.inputValue(), "");
  assert.equal(await commandMenu.isVisible(), false);
  await page.click("#toggleSidebarBtn");
  await page.getByRole("button", { name: "evidence.txt", exact: true }).click();
  await page.waitForFunction(() =>
    document.querySelector("#docMdViewer").textContent.includes("50 samples"),
  );
  await page.screenshot({
    path: "test-results/workspace/source-mobile.png",
    fullPage: true,
  });
  await page.click("#closeSidecarBtn");
  await page
    .getByRole("button", { name: "experiment.pdf", exact: true })
    .click();
  await previewImage(1).waitFor();
  await pdfViewport.focus();
  await pdfViewport.press("End");
  await previewImage(6).waitFor();
  await page.waitForFunction(
    () => document.querySelector("#sourcePageInput").value === "6",
  );
  await page.click("#closeSidecarBtn");
  await page.click("#toggleSidebarBtn");
  await page.click("#modelDropdownBtn");
  await page.waitForFunction(
    () =>
      getComputedStyle(document.querySelector("#agentMatrixPopover"))
        .opacity === "1",
  );
  const bounds = await page.locator("#agentMatrixPopover").boundingBox();
  assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= 390);
  assert.ok(
    await page
      .locator("#agentMatrixPopover")
      .evaluate((node) => node.scrollWidth <= node.clientWidth),
  );
  await page.screenshot({
    path: "test-results/workspace/models-mobile.png",
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  console.log(
    "PASS: ResearchDesk desktop/mobile, slash commands, search dispatch, scoped research, Markdown download, source preview, persisted models, no JavaScript errors.",
  );
} finally {
  if (browser) await browser.close();
  await backend.close();
}
