import { el } from "./dom.js";
import { downloadText } from "./download.js";

const appendixHeadings = new Set([
  "sources",
  "nguồn trích dẫn",
  "question coverage",
  "mức đáp ứng câu hỏi",
]);
const genericTitle = /^(research notes|ghi chú nghiên cứu)$/iu;
const genericSection =
  /^(evidence(?: from sources)?|bằng chứng(?: từ tài liệu)?)$/iu;

export function unescapeReport(text) {
  return text.replace(/\\([\\`*_{}\[\]<>()#!|])/g, "$1");
}

export function appendReportCitations(parent, text, createCitation) {
  let offset = 0;
  const inline = /\\[\\`*_{}\[\]<>()#!|]|\[\d+\]/g;
  for (const match of text.matchAll(inline)) {
    const marker = match[0];
    if (marker.startsWith("\\")) continue;
    const citation = createCitation(marker);
    if (!citation) continue;
    parent.append(unescapeReport(text.slice(offset, match.index)), citation);
    offset = match.index + marker.length;
  }
  parent.append(unescapeReport(text.slice(offset)));
}

function tableCells(line) {
  const text = line.trim();
  if (!text.startsWith("|") || !text.endsWith("|")) return null;
  const cells = [];
  let start = 1;
  let escapes = 0;
  for (let i = start; i < text.length; i++) {
    const char = text[i];
    if (char === "|" && escapes % 2 === 0) {
      cells.push(text.slice(start, i).trim());
      start = i + 1;
    }
    escapes = char === "\\" ? escapes + 1 : 0;
  }
  return start === text.length && cells.length >= 2 ? cells : null;
}

// Only the outer-pipe table format emitted by Writer is interpreted.
export function* reportBlocks(markdown) {
  const lines = markdown.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const columns = tableCells(lines[i]);
    const separator = columns && tableCells(lines[i + 1] || "");
    if (
      !columns ||
      separator?.length !== columns.length ||
      !separator.every((cell) => /^:?-{3,}:?$/.test(cell))
    ) {
      yield lines[i];
      continue;
    }
    const rows = [];
    let next = i + 2;
    for (; next < lines.length; next++) {
      const cells = tableCells(lines[next]);
      if (cells?.length !== columns.length) break;
      rows.push(cells);
    }
    if (!rows.length) {
      yield lines[i];
      continue;
    }
    yield { columns, rows, markdown: lines.slice(i, next).join("\n") };
    i = next - 1;
  }
}

export function reportTableCsv({ columns, rows }) {
  function field(markdown) {
    let text = unescapeReport(markdown);
    // Quote every field for CSV syntax, then keep formula-leading cells as text.
    if (/^[\t\r\n]|^[\s\u200B-\u200D]*[=+\-@＝＋－＠]/u.test(text)) {
      text = "'" + text;
    }
    return '"' + text.replaceAll('"', '""') + '"';
  }
  return (
    "\uFEFF" +
    [columns, ...rows].map((line) => line.map(field).join(",")).join("\r\n") +
    "\r\n"
  );
}

export function renderReportTable(block, createCitation, heading) {
  const { columns, rows } = block;
  const container = el("div", "report-table-block");
  const actions = el("div", "report-table-actions");
  const copy = el("button", "report-table-action", "Sao chép bảng Markdown");
  const csv = el("button", "report-table-action", "Xuất CSV");
  const status = el("span", "report-table-status");
  status.setAttribute("role", "status");
  copy.type = csv.type = "button";
  copy.addEventListener("click", async () => {
    if (copy.getAttribute("aria-busy") === "true") return;
    copy.setAttribute("aria-busy", "true");
    try {
      await navigator.clipboard.writeText(block.markdown);
      status.textContent = "Đã sao chép bảng.";
    } catch {
      downloadText(
        "bang-so-sanh.md",
        block.markdown,
        "text/markdown;charset=utf-8",
      );
      status.textContent = "Đã tạo tệp Markdown để tải xuống.";
    } finally {
      copy.removeAttribute("aria-busy");
    }
  });
  csv.addEventListener("click", () => {
    downloadText(
      "bang-so-sanh.csv",
      reportTableCsv(block),
      "text/csv;charset=utf-8",
    );
    status.textContent = "Đã tạo tệp CSV để tải xuống.";
  });
  actions.append(copy, csv, status);

  const wrapper = el("div", "report-table-scroll");
  wrapper.tabIndex = 0;
  wrapper.setAttribute("role", "region");
  wrapper.setAttribute("aria-label", heading || "Bảng so sánh tài liệu");
  const table = el("table");
  const head = el("thead");
  const header = el("tr");
  for (const column of columns) {
    const cell = el("th", "", unescapeReport(column));
    cell.scope = "col";
    header.append(cell);
  }
  head.append(header);
  const body = el("tbody");
  for (const [label, ...values] of rows) {
    const row = el("tr");
    const cell = el("th", "", unescapeReport(label));
    cell.scope = "row";
    row.append(cell);
    for (const value of values) {
      const item = el("td");
      appendReportCitations(item, value, createCitation);
      row.append(item);
    }
    body.append(row);
  }
  table.append(head, body);
  wrapper.append(table);
  container.append(actions, wrapper);
  return container;
}

// Legacy runs include a bibliography and coverage table in the Markdown.
// Keep those in the exported report and display the bibliography as clickable citations.
export function presentReport(markdown) {
  const lines = [];
  let appendix = false;
  for (const line of markdown.split("\n")) {
    const heading = /^(#{1,2})\s+(.*)$/.exec(line);
    if (heading) {
      const [, level, raw] = heading;
      const name = unescapeReport(raw.trim());
      appendix =
        level === "##" && appendixHeadings.has(name.toLocaleLowerCase("vi"));
      if (
        appendix ||
        (level === "#" ? genericTitle : genericSection).test(name)
      )
        continue;
    }
    if (!appendix) lines.push(line);
  }
  return lines.join("\n").trim();
}

export function citationIndex(result) {
  const evidence = new Map(result.evidence.map((item) => [item.id, item]));
  if (result.citation_evidence_ids?.length) {
    const citations = new Map();
    result.citation_evidence_ids.forEach((id, index) => {
      const item = evidence.get(id);
      if (item) citations.set(`[${index + 1}]`, item);
    });
    return citations;
  }
  const citations = new Map();
  const lines = result.markdown.split("\n");
  for (const [index, line] of lines.entries()) {
    const row = /^(\[\d+\]).*`(ch_[\w-]+)`$/.exec(line);
    if (!row) continue;
    const quote = lines
      .slice(index + 1, index + 5)
      .find((text) => text.startsWith("> "));
    const candidates = result.evidence.filter(
      (entry) => entry.chunk_id === row[2],
    );
    const matching = quote
      ? candidates.filter(
          (entry) =>
            entry.quote.replace(/\n/g, " ") === unescapeReport(quote.slice(2)),
        )
      : candidates;
    if (matching.length === 1) citations.set(row[1], matching[0]);
  }
  return citations;
}
