import { chromium } from "playwright";
import { readFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

async function main() {
  console.log("Đọc file HTML nguồn...");
  const htmlPath = resolve("docs", "RESEARCH_ASSIST_ARCHITECTURE_MANUAL.html");
  const html = readFileSync(htmlPath, "utf-8");

  console.log("Khởi động Google Chrome headless...");
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });

  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: "networkidle" });

  const targetPathDocs = resolve("docs", "RESEARCH_ASSIST_ARCHITECTURE_MANUAL.pdf");
  const targetPathRoot = resolve("TAI_LIEU_KIEN_TRUC_HE_THONG_RESEARCH_ASSIST.pdf");

  console.log("Xuất file PDF...");
  await page.pdf({
    path: targetPathDocs,
    format: "A4",
    printBackground: true,
    margin: {
      top: "16mm",
      bottom: "18mm",
      left: "14mm",
      right: "14mm",
    },
    displayHeaderFooter: true,
    headerTemplate: `
      <div style="font-size: 8px; color: #94A3B8; width: 100%; text-align: right; padding-right: 14mm; font-family: -apple-system, sans-serif;">
        ResearchDesk · Cẩm nang Kiến trúc & Mã nguồn Hệ thống
      </div>
    `,
    footerTemplate: `
      <div style="font-size: 8px; color: #94A3B8; width: 100%; display: flex; justify-content: space-between; padding-left: 14mm; padding-right: 14mm; font-family: -apple-system, sans-serif;">
        <span>Dự án AI Research Assist</span>
        <span>Trang <span class="pageNumber"></span> / <span class="totalPages"></span></span>
      </div>
    `,
  });

  // Bản sao ở thư mục gốc
  await page.pdf({
    path: targetPathRoot,
    format: "A4",
    printBackground: true,
    margin: {
      top: "16mm",
      bottom: "18mm",
      left: "14mm",
      right: "14mm",
    },
    displayHeaderFooter: true,
    headerTemplate: `
      <div style="font-size: 8px; color: #94A3B8; width: 100%; text-align: right; padding-right: 14mm; font-family: -apple-system, sans-serif;">
        ResearchDesk · Cẩm nang Kiến trúc & Mã nguồn Hệ thống
      </div>
    `,
    footerTemplate: `
      <div style="font-size: 8px; color: #94A3B8; width: 100%; display: flex; justify-content: space-between; padding-left: 14mm; padding-right: 14mm; font-family: -apple-system, sans-serif;">
        <span>Dự án AI Research Assist</span>
        <span>Trang <span class="pageNumber"></span> / <span class="totalPages"></span></span>
      </div>
    `,
  });

  await browser.close();
  console.log("Xuất file PDF hoàn tất thành công!");
}

main().catch((err) => {
  console.error("Lỗi:", err);
  process.exit(1);
});
