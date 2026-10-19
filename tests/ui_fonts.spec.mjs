import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";
import { startBackend } from "./backend.mjs";

const backend = await startBackend();
const output = "test-results/fonts";
await mkdir(output, { recursive: true });
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.UI_CHROMIUM_PATH || undefined,
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  const externalRequests = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (!request.url().startsWith(backend.base))
      externalRequests.push(request.url());
  });
  await page.goto(backend.base);
  await page.waitForFunction(() =>
    document
      .querySelector("#connection-label")
      .textContent.includes("đã kết nối"),
  );
  await page.evaluate(() => document.fonts.ready);

  const cdp = await page.context().newCDPSession(page);
  await cdp.send("DOM.enable");
  await cdp.send("CSS.enable");
  const families = ["Noto Sans", "Newsreader", "JetBrains Mono"];
  // Newsreader's internal TrueType family name includes its default optical size.
  const platformFamilies = { Newsreader: "Newsreader 16pt" };
  const glyphChecks = [];
  // CSS font-family and document.fonts.check alone cannot detect glyph fallback.
  // Inspect the actual fonts used to draw every Vietnamese accent, in NFC and NFD.
  for (const family of families) {
    await page.evaluate(async (family) => {
      const precomposed =
        "Nghiên cứu tài liệu · Cấu hình mô hình · Đối chiếu bằng chứng ĂăÂâĐđÊêÔôƠơƯư " +
        Array.from({ length: 0x1efa - 0x1ea0 }, (_, i) =>
          String.fromCodePoint(0x1ea0 + i),
        ).join("");
      const text = `${precomposed}\n${precomposed.normalize("NFD")}`;
      const probe = document.createElement("p");
      probe.id = "vietnamese-font-probe";
      probe.lang = "vi";
      probe.textContent = text;
      probe.style.fontFamily = `"${family}"`;
      probe.style.fontSize = "28px";
      probe.style.fontWeight = "500";
      probe.style.maxWidth = "700px";
      probe.style.position = "fixed";
      probe.style.top = "80px";
      probe.style.left = "24px";
      probe.style.zIndex = "1000";
      probe.style.background = "white";
      probe.style.padding = "20px";
      probe.style.whiteSpace = "pre-wrap";
      probe.style.overflowWrap = "anywhere";
      await document.fonts.load(`500 28px "${family}"`, text);
      document.body.append(probe);
    }, family);
    await page.locator("#vietnamese-font-probe").screenshot({
      path: `${output}/${family.toLowerCase().replaceAll(" ", "-")}-specimen.png`,
    });
    const { root } = await cdp.send("DOM.getDocument");
    const { nodeId } = await cdp.send("DOM.querySelector", {
      nodeId: root.nodeId,
      selector: "#vietnamese-font-probe",
    });
    const { fonts } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
    const rendered = fonts.filter((font) => font.glyphCount > 0);
    assert.ok(rendered.length > 0, `${family} did not render`);
    assert.ok(
      rendered.every(
        (font) =>
          font.isCustomFont &&
          font.familyName === (platformFamilies[family] || family),
      ),
      `${family} has fallback glyphs: ${JSON.stringify(rendered)}`,
    );
    glyphChecks.push({
      family,
      glyphs: rendered.reduce((sum, font) => sum + font.glyphCount, 0),
      fallbackFonts: 0,
    });
    await page
      .locator("#vietnamese-font-probe")
      .evaluate((element) => element.remove());
  }
  for (const [selector, expected] of [
    ["#view-research h1", "Newsreader 16pt"],
    ["#question-title", "Noto Sans"],
    ["#view-research .page-heading .eyebrow", "JetBrains Mono"],
  ]) {
    await page.locator(selector).scrollIntoViewIfNeeded();
    const { root } = await cdp.send("DOM.getDocument");
    const { nodeId } = await cdp.send("DOM.querySelector", {
      nodeId: root.nodeId,
      selector,
    });
    const { fonts } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
    assert.ok(
      fonts.some(
        (font) =>
          font.familyName === expected &&
          font.isCustomFont &&
          font.glyphCount > 0,
      ),
      `${selector} is not using ${expected}`,
    );
    assert.ok(
      fonts.every((font) => !font.glyphCount || font.familyName === expected),
      `${selector} contains mixed fonts`,
    );
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: `${output}/vietnamese-desktop.png`,
    fullPage: true,
  });
  await page.locator("#view-research .page-heading").screenshot({
    path: `${output}/vietnamese-heading.png`,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(
    () => document.querySelector("#sidebar").getBoundingClientRect().right <= 1,
  );
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
    true,
  );
  await page.screenshot({
    path: `${output}/vietnamese-mobile.png`,
    fullPage: true,
  });
  assert.deepEqual(externalRequests, []);
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      status: "passed",
      glyphChecks,
      externalRequests,
      javascriptErrors: errors,
      screenshots: output,
    }),
  );
} finally {
  await browser?.close();
  await backend.close();
}
