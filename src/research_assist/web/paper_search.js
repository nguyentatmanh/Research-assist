import { api } from "./api.js";
import { $, el } from "./dom.js";

export function initializePaperSearch(refreshDocuments) {
  const form = $("#paper-search-form");
  const input = $("#paper-query");
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
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    results.replaceChildren();
    status.textContent = "Đang tìm trên arXiv…";
    setBusy(true);
    try {
      const papers = await api("/papers/search", {
        method: "POST",
        body: { query: input.value.trim() },
      });
      for (const paper of papers) {
        const row = el("li");
        const title = el("a", "", paper.title);
        title.href = "https://arxiv.org/abs/" + paper.paper_id;
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
          add,
        );
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
