import { api } from "./api.js";
import { $, el } from "./dom.js";
import {
  appendScoutAbstract,
  scoutBibtexButton,
  scoutPaperLink,
  scoutSourceName,
} from "./scout.js?v=research-exports-1";

export function initializePaperSearch(refreshDocuments) {
  const form = $("#paper-search-form");
  const input = $("#paper-query");
  const source = $("#paper-source");
  const results = $("#paper-results");
  const status = $("#paper-status");
  let busy = false;
  function setBusy(value) {
    busy = value;
    for (const button of [
      form.querySelector("button"),
      ...results.querySelectorAll("button"),
    ]) {
      button.disabled = busy || button.dataset.imported === "true";
    }
    input.readOnly = busy;
    source.disabled = busy;
  }
  source.addEventListener("change", () => {
    results.replaceChildren();
    status.textContent = "";
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    results.replaceChildren();
    const selectedSource = source.value;
    status.textContent =
      "Đang tìm trên " + (scoutSourceName(selectedSource) || "arXiv") + "…";
    setBusy(true);
    try {
      const papers = await api("/papers/search", {
        method: "POST",
        body: { query: input.value.trim(), source: selectedSource },
      });
      for (const paper of papers) {
        const row = el("li");
        const provider = paper.source || selectedSource;
        if (provider === "openalex") {
          row.append(
            scoutPaperLink(paper, "") || el("strong", "", paper.title),
            el(
              "small",
              "",
              ["OpenAlex", paper.authors.join(", "), paper.published, paper.doi]
                .filter(Boolean)
                .join(" · "),
            ),
          );
          appendScoutAbstract(row, paper);
          row.append(scoutBibtexButton(paper, "secondary-button"));
          results.append(row);
          continue;
        }
        const title = el("a", "", paper.title);
        const isCrossref = provider === "crossref";
        title.href = isCrossref
          ? "https://doi.org/" + encodeURIComponent(paper.doi)
          : "https://arxiv.org/abs/" +
            encodeURIComponent(paper.paper_id).replace(/%2F/g, "/");
        title.target = "_blank";
        title.rel = "noopener noreferrer";
        const add = el("button", "secondary-button", "Thêm vào kho");
        add.type = "button";
        add.setAttribute("aria-label", "Thêm vào kho: " + paper.title);
        add.addEventListener("click", async () => {
          if (busy) return;
          setBusy(true);
          status.textContent = "Đang tải và lập chỉ mục cho " + paper.title;
          try {
            const result = await api("/papers/import", {
              method: "POST",
              body: { paper_id: paper.paper_id },
            });
            add.dataset.imported = "true";
            add.textContent = "Đã có trong kho";
            await refreshDocuments();
            status.textContent = result.duplicate
              ? "Bài báo này đã có trong kho."
              : "Đã thêm " + paper.title;
          } catch (error) {
            status.textContent = error.message;
          } finally {
            setBusy(false);
          }
        });
        row.append(
          title,
          el(
            "small",
            "",
            [paper.authors.join(", "), paper.published, paper.paper_id]
              .filter(Boolean)
              .join(" · "),
          ),
        );
        const actions = el("div", "scout-paper-actions");
        if (isCrossref || paper.import_available === false) {
          row.append(
            el(
              "small",
              "",
              "Mở DOI để xem bài; tải PDF và thêm vào kho để dùng làm bằng chứng.",
            ),
          );
        } else {
          actions.append(add);
        }
        actions.append(scoutBibtexButton(paper, "secondary-button"));
        row.append(actions);
        results.append(row);
      }
      status.textContent = papers.length
        ? "Tìm được " + papers.length + " bài."
        : "Chưa tìm thấy bài. Hãy thử tiêu đề hoặc từ khóa tiếng Anh.";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      setBusy(false);
    }
  });
}
