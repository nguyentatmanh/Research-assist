import assert from "node:assert/strict";
import { test } from "node:test";
import {
  reportBlocks,
  reportTableCsv,
} from "../src/research_assist/web/report_content.js";
import { scoutBibtex } from "../src/research_assist/web/scout.js";

test("report table exports preserve the stored Markdown and citation markers", () => {
  const markdown =
    "| Tiêu chí | Dữ liệu |\n| :--- | ---: |\n" +
    '| Tên \\| chỉ số | Độ chính xác, "BLEU" [3] |';
  const blocks = [
    ...reportBlocks("## Bảng\n\n" + markdown + "\n\nBên ngoài bảng"),
  ];
  const table = blocks[2];
  assert.equal(table.markdown, markdown);
  assert.equal(
    reportTableCsv(table),
    '\uFEFF"Tiêu chí","Dữ liệu"\r\n"Tên | chỉ số","Độ chính xác, ""BLEU"" [3]"\r\n',
  );
});

test("CSV protects formula-leading cells and quotes embedded line breaks", () => {
  const table = {
    columns: ["=Phạm vi", "Giá trị"],
    rows: [
      ["  ＋7", "Dòng một\nDòng hai [1]"],
      ["Dữ liệu", "@nguồn"],
    ],
  };
  assert.equal(
    reportTableCsv(table),
    '\uFEFF"\'=Phạm vi","Giá trị"\r\n"\'  ＋7","Dòng một\nDòng hai [1]"\r\n' +
      '"Dữ liệu","\'@nguồn"\r\n',
  );
});

test("BibTeX escapes title and author syntax and preserves DOI punctuation", () => {
  const paper = {
    paper_id: "W123",
    title: "Bảng {A & B}%_v2",
    authors: ["Viện R and D", "Nguyễn Ánh"],
    doi: "10.1234/result_v2",
    published: "2026-11-16",
    url: "https://doi.org/10.1234/result_v2",
  };
  const entry = scoutBibtex(paper);
  assert.match(entry.text, /^@misc\{/);
  assert.ok(
    entry.text.includes(
      "title = {Bảng \\textbraceleft{}A \\& B\\textbraceright{}\\%\\_v2}",
    ),
  );
  assert.ok(entry.text.includes("author = {{Viện R and D} and Nguyễn Ánh}"));
  assert.ok(entry.text.includes("doi = {10.1234/result_v2}"));
  assert.ok(entry.text.includes("year = {2026}"));
  assert.equal(entry.key, scoutBibtex({ ...paper, title: "Tiêu đề sửa" }).key);
});

test("BibTeX omits unavailable fields and accepts versioned arXiv IDs", () => {
  const entry = scoutBibtex({
    source: "arxiv",
    paper_id: "hep-th/9411152v3",
    title: "Mạng truy xuất",
    url: "javascript:discard",
  });
  assert.match(entry.text, /^@misc\{/);
  assert.ok(entry.text.includes("eprint = {hep-th/9411152v3}"));
  assert.ok(!/\bauthor|\byear|\burl|\bdoi/.test(entry.text));
});
