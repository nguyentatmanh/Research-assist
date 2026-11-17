import { el } from "./dom.js";
import { downloadText } from "./download.js";

const sources = new Map([
  ["arxiv", "arXiv"],
  ["crossref", "Crossref"],
  ["openalex", "OpenAlex"],
]);

export function scoutSourceName(source) {
  return sources.get(source);
}

export function scoutPaperMetadata(paper) {
  const providers = Array.isArray(paper.providers)
    ? paper.providers
    : [paper.provider || paper.source];
  const names = [...new Set(providers.map(scoutSourceName).filter(Boolean))];
  return [
    names.join(" / ") || scoutSourceName(paper.provider || paper.source),
    paper.year || paper.published?.slice(0, 4),
    paper.authors?.join(", "),
  ]
    .filter(Boolean)
    .join(" · ");
}

function paperUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password) return null;
    return url.href;
  } catch {
    return null;
  }
}

export function scoutPaperLink(paper, className, label = paper.title) {
  const url = paperUrl(paper.url);
  if (!url) return null;
  const link = el("a", className, label);
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

const bibtexEscapes = {
  "\\": "\\textbackslash{}",
  "{": "\\textbraceleft{}",
  "}": "\\textbraceright{}",
  "#": "\\#",
  $: "\\$",
  "%": "\\%",
  "&": "\\&",
  _: "\\_",
  "~": "\\textasciitilde{}",
  "^": "\\textasciicircum{}",
};

function bibtexField(value) {
  return String(value ?? "")
    .trim()
    .replace(/\s+/gu, " ")
    .replace(/[\\{}#$%&_~^]/g, (symbol) => bibtexEscapes[symbol]);
}

function bibtexIdentifier(value) {
  // Leave URI punctuation intact in verbatim fields and encode delimiters.
  return value?.replace(/[{}\\]/g, encodeURIComponent);
}

export function scoutBibtex(paper) {
  const provider = paper.provider || paper.source;
  const arxivId =
    paper.arxiv_id || (provider === "arxiv" ? paper.paper_id : "");
  const eprint =
    /^(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*\/\d{7})(?:v[1-9]\d{0,3})?$/i.test(arxivId)
      ? arxivId
      : "";
  const doi = paper.doi?.trim().replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, "");
  const url = paperUrl(paper.url);
  const year = /^(\d{4})(?:\D|$)/.exec(
    String(paper.year || paper.published || ""),
  )?.[1];
  const identity =
    doi?.toLowerCase() || eprint || paper.paper_id || url || paper.title;
  let hash = 2166136261;
  for (const character of identity.normalize("NFC")) {
    hash = Math.imul(hash ^ character.codePointAt(0), 16777619);
  }
  const key = "scout-" + (hash >>> 0).toString(36);
  const authors = (paper.authors || [])
    .map(bibtexField)
    .filter(Boolean)
    .map((name) => (/\band\b/i.test(name) ? "{" + name + "}" : name));
  const fields = [
    ["title", bibtexField(paper.title)],
    ["author", authors.join(" and ")],
    ["year", year],
    ["doi", bibtexIdentifier(doi)],
    ["eprint", eprint],
    ["archivePrefix", eprint && "arXiv"],
    ["url", bibtexIdentifier(url)],
  ].filter(([, value]) => value);
  return {
    key,
    text: `@misc{${key},\n${fields.map(([name, value]) => `  ${name} = {${value}}`).join(",\n")}\n}\n`,
  };
}

export function scoutBibtexButton(paper, className) {
  const button = el("button", className, "Tải BibTeX");
  button.type = "button";
  button.setAttribute("aria-label", "Tải BibTeX cho " + paper.title);
  button.addEventListener("click", () => {
    const { key, text } = scoutBibtex(paper);
    downloadText(key + ".bib", text, "application/x-bibtex;charset=utf-8");
  });
  return button;
}

export function appendScoutAbstract(parent, paper) {
  if (typeof paper.abstract !== "string" || !paper.abstract.trim()) return;
  const details = el("details", "scout-abstract");
  const isOpenAlex =
    (paper.provider || paper.source) === "openalex" ||
    paper.providers?.includes("openalex");
  details.append(
    el("summary", "", isOpenAlex ? "Tóm tắt · OpenAlex" : "Tóm tắt"),
    el("p", "", paper.abstract),
  );
  parent.append(details);
}

export function focusScoutInput(input, query) {
  const details = input.closest("details");
  if (details) details.open = true;
  input.value = query;
  input.focus();
}

export function appendScoutGapActions(parent, report, className, openSearch) {
  const requirements = new Map(
    (report.requirements || []).map(({ id, text }) => [id, text]),
  );
  const queries = new Set();
  for (const gap of report.gap_details || []) {
    if (gap.reason !== "no_supported_evidence") continue;
    const requirement = requirements.get(gap.requirement_id);
    if (typeof requirement !== "string") continue;
    const query = requirement
      .normalize("NFC")
      .trim()
      .replace(/\s+/g, " ")
      .slice(0, 300)
      .replace(/[\uD800-\uDBFF]$/, "")
      .trimEnd();
    if (query.length >= 3) queries.add(query);
  }
  if (!queries.size) return;

  const actions = el("div", "scout-gap-actions");
  for (const query of queries) {
    const option = el("div", "scout-gap-option");
    const label = "Tìm tài liệu bổ sung";
    const action = el("button", `${className} scout-gap-action`, label);
    action.type = "button";
    action.setAttribute("aria-label", `${label} cho yêu cầu ${query}`);
    action.addEventListener("click", () => openSearch(query));
    if (queries.size > 1) option.append(el("p", "", query));
    option.append(action);
    actions.append(option);
  }
  parent.append(actions);
}
