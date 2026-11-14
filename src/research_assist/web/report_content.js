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
