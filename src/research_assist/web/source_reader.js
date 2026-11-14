import { api } from "./api.js";
import { el } from "./dom.js";

const zoomSteps = [0.75, 1, 1.25, 1.5, 1.75, 2];
const maxPageRequests = 2;

export function createSourceReader(showSource, sourceChanged) {
  const get = (id) => document.getElementById(id);
  const picker = get("sourceDocumentPicker");
  const status = get("sourceStatus");
  const quote = get("sourceContext");
  const viewport = get("sourceViewport");
  const pages = get("sourcePdfPages");
  const text = get("docMdViewer");
  const pageInput = get("sourcePageInput");
  const previous = get("pdfPrevPageBtn");
  const next = get("pdfNextPageBtn");
  const zoomOut = get("pdfZoomOutBtn");
  const zoomIn = get("pdfZoomInBtn");
  let active;
  let sources = [];
  let request;
  let zoom = 1;
  let textFocusPending = false;
  let frame = 0;

  function releaseImage(slot) {
    slot.image?.remove();
    if (slot.objectUrl) URL.revokeObjectURL(slot.objectUrl);
    slot.image = slot.objectUrl = undefined;
  }

  function resetContent() {
    request?.abort();
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    for (const slot of active?.slots || []) releaseImage(slot);
    pages.replaceChildren();
    pages.hidden = true;
    text.replaceChildren();
    text.hidden = true;
    textFocusPending = false;
  }

  function setStatus(message) {
    status.textContent = message;
    status.hidden = !message;
  }

  function clear() {
    resetContent();
    active = undefined;
    quote.hidden = viewport.hidden = true;
    get("sourcePagination").hidden = get("sourceZoom").hidden = true;
    get("downloadDocBtn").hidden = true;
    setStatus(
      "Chọn tên tài liệu trong kho hoặc mở một trích dẫn để đọc nguồn tại đây.",
    );
    sourceChanged();
  }

  function updatePicker() {
    picker.replaceChildren(
      ...sources.map((doc) => {
        const option = el("option", "", doc.filename);
        option.value = doc.id;
        return option;
      }),
    );
    get("sourcePickerLabel").hidden = sources.length < 2;
    if (active) picker.value = active.doc.id;
    else picker.selectedIndex = -1;
  }

  function setDocuments(documents) {
    sources = documents;
    if (active && !sources.some((doc) => doc.id === active.doc.id)) clear();
    updatePicker();
  }

  function updateDownload(url) {
    const button = get("downloadDocBtn");
    button.href = url + "/file";
    button.hidden = !get("pdfViewerTab").classList.contains("active");
  }

  function show() {
    if (!active && sources.length) return open(sources[0]);
    if (active) {
      updateDownload("/api" + active.url);
      if (active.slots) schedulePages();
      else scrollToTextQuote();
    }
    return undefined;
  }

  function scrollToTextQuote() {
    if (!textFocusPending) return;
    const mark = text.querySelector("mark");
    if (mark && mark.getClientRects().length) {
      mark.scrollIntoView?.({ block: "center" });
      textFocusPending = false;
    }
  }

  function setZoom(value) {
    const anchor = active?.slots?.[pageAt(viewport.scrollTop) - 1]?.node;
    const fraction = anchor?.offsetHeight
      ? (viewport.scrollTop - anchor.offsetTop) / anchor.offsetHeight
      : 0;
    zoom = Math.max(0.75, Math.min(zoomSteps.at(-1), value));
    viewport.style.setProperty("--source-zoom", String(zoom));
    zoomOut.disabled = zoom === zoomSteps[0];
    zoomIn.disabled = zoom === zoomSteps.at(-1);
    get("pdfZoomResetBtn").textContent =
      zoom === 1 ? "Vừa khung" : `${zoom * 100}%`;
    if (anchor?.offsetHeight)
      viewport.scrollTop = anchor.offsetTop + fraction * anchor.offsetHeight;
    schedulePages();
  }

  function showQuote(citation) {
    quote.hidden = !citation;
    quote.open = false;
    if (!citation) {
      get("sourceContextQuote").textContent = "";
      return;
    }
    const evidence = citation.evidence;
    get("sourceContextTitle").textContent = evidence.page
      ? `Trích dẫn ${citation.marker} · Trang ${evidence.page}`
      : `Trích dẫn ${citation.marker}`;
    get("sourceContextQuote").textContent = evidence.quote;
  }

  function citationQuery(citation) {
    return citation
      ? "?" +
          new URLSearchParams({
            run_id: citation.runId,
            evidence_id: citation.evidence.id,
          })
      : "";
  }

  async function loadPdfPage(opening, slot) {
    const { doc, citation, url, signal } = opening;
    const { number, node } = slot;
    const match = citation?.evidence.page === number;
    slot.state = "loading";
    opening.inFlight++;
    node.setAttribute("aria-busy", "true");
    try {
      const response = await fetch(
        `/api${url}/pages/${number}/preview${citationQuery(match ? citation : null)}`,
        { signal },
      );
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        if (signal.aborted) return;
        if (response.status === 401)
          window.dispatchEvent(new Event("authentication-required"));
        throw new Error(error?.detail || "Không mở được trang PDF.");
      }
      const blob = await response.blob();
      if (signal.aborted) return;
      const image = el("img");
      image.alt = `Trang ${number} của ${doc.filename}`;
      slot.image = image;
      slot.objectUrl = URL.createObjectURL(blob);
      image.src = slot.objectUrl;
      await image.decode();
      if (signal.aborted) return;
      node.style.setProperty(
        "--page-ratio",
        String(image.naturalWidth / image.naturalHeight),
      );
      node.replaceChildren(image);
      slot.state = "ready";
      if (match) {
        const highlight = response.headers.get("x-source-highlight");
        const focus = response.headers.get("x-source-focus-y");
        const ratio = focus === null ? NaN : Number(focus);
        slot.focus = Number.isFinite(ratio)
          ? Math.min(1, Math.max(0, ratio))
          : 0;
        setStatus(
          highlight === "region"
            ? "Khung vàng đánh dấu vùng nguồn đã lưu. Mở đoạn trích để đối chiếu nội dung."
            : !highlight || highlight === "none"
              ? "Chưa xác định được vị trí tô sáng. Bạn vẫn có thể đối chiếu đoạn trích với trang này."
              : "",
        );
      }
    } catch (error) {
      if (signal.aborted) return;
      releaseImage(slot);
      slot.state = "error";
      const message = el("div", "source-pdf-placeholder");
      const retry = el("button", "source-pdf-retry", "Tải lại trang " + number);
      retry.type = "button";
      retry.addEventListener("click", () => {
        resetSlot(slot);
        schedulePages();
        viewport.focus({ preventScroll: true });
      });
      message.append(
        el("p", "", error.message || "Không hiển thị được trang PDF."),
        retry,
      );
      node.replaceChildren(message);
    } finally {
      opening.inFlight--;
      if (!signal.aborted) {
        node.removeAttribute("aria-busy");
        schedulePages();
      }
    }
  }

  function resetSlot(slot) {
    releaseImage(slot);
    slot.state = "idle";
    slot.node.replaceChildren(
      el("p", "source-pdf-placeholder", `Đang chờ tải trang ${slot.number}…`),
    );
  }

  // Slots preserve the scroll range; only nearby pages hold decoded images.
  function pageAt(offset) {
    const slots = active.slots;
    let low = 0;
    let high = slots.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      const { node } = slots[middle];
      if (node.offsetTop + node.offsetHeight <= offset) low = middle + 1;
      else high = middle;
    }
    return low + 1;
  }

  function updatePageNumber(page) {
    active.page = page;
    if (document.activeElement !== pageInput) pageInput.value = String(page);
    previous.disabled = page <= 1;
    next.disabled = page >= active.doc.page_count;
  }

  function syncPages() {
    const opening = active;
    if (
      !opening?.slots?.length ||
      !viewport.clientHeight ||
      viewport.closest("[inert]")
    )
      return;
    const top = viewport.scrollTop;
    const bottom = top + viewport.clientHeight;
    const first = pageAt(top);
    const last = pageAt(bottom);
    const current =
      top > 0 && bottom >= viewport.scrollHeight - 1 ? last : pageAt(top + 16);
    const target = opening.focusPage;
    if (target && target >= first && target <= last) {
      const slot = opening.slots[target - 1];
      if (slot.state === "ready") {
        opening.focusPage = undefined;
        viewport.scrollTop =
          slot.node.offsetTop +
          Math.max(0, slot.node.offsetHeight * slot.focus - 56);
        schedulePages();
        return;
      }
    } else {
      opening.focusPage = undefined;
    }
    updatePageNumber(current);
    const start = Math.max(1, first - 1);
    const end = Math.min(opening.slots.length, last + 1);
    const candidates = [];
    for (const slot of opening.slots) {
      if (
        slot.state === "ready" &&
        (slot.number < start - 2 || slot.number > end + 2)
      )
        resetSlot(slot);
      if (slot.state === "idle" && slot.number >= start && slot.number <= end)
        candidates.push(slot);
    }
    const priority = (slot) =>
      slot.number === opening.focusPage ? -1 : Math.abs(slot.number - current);
    candidates.sort((a, b) => priority(a) - priority(b));
    for (const slot of candidates.slice(0, maxPageRequests - opening.inFlight))
      void loadPdfPage(opening, slot);
  }

  function schedulePages() {
    if (!active?.slots || frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      syncPages();
    });
  }

  async function loadText({ url, citation, signal }) {
    setStatus("Đang mở tài liệu…");
    try {
      const result = await api(url + "/text" + citationQuery(citation), {
        signal,
      });
      if (signal.aborted) return;
      if (
        citation &&
        result.highlight_start != null &&
        result.highlight_end != null
      ) {
        const points = Array.from(result.text);
        text.append(
          points.slice(0, result.highlight_start).join(""),
          el(
            "mark",
            "",
            points.slice(result.highlight_start, result.highlight_end).join(""),
          ),
          points.slice(result.highlight_end).join(""),
        );
        textFocusPending = true;
        setStatus("");
      } else {
        text.textContent = result.text;
        setStatus(
          citation ? "Chưa tìm thấy vị trí chính xác trong văn bản này." : "",
        );
      }
      text.hidden = false;
      scrollToTextQuote();
    } catch (error) {
      if (!signal.aborted)
        setStatus(error.message || "Không mở được tài liệu nguồn.");
    }
  }

  function jump(page) {
    if (!active?.slots?.length) return;
    const number = Math.max(
      1,
      Math.min(active.doc.page_count, Math.trunc(page) || active.page),
    );
    active.focusPage = undefined;
    pageInput.value = String(number);
    updatePageNumber(number);
    viewport.scrollTop =
      number === 1 ? 0 : active.slots[number - 1].node.offsetTop;
    schedulePages();
  }

  function open(doc, citation, reveal = true) {
    resetContent();
    if (!sources.some((source) => source.id === doc.id))
      sources = [...sources, doc];
    const url = "/documents/" + encodeURIComponent(doc.id);
    request = new AbortController();
    active = { doc, page: 1, url, citation, signal: request.signal };
    updatePicker();
    if (reveal) showSource();
    updateDownload("/api" + url);
    sourceChanged(doc);
    viewport.hidden = false;
    viewport.scrollTop = viewport.scrollLeft = 0;
    showQuote(citation);
    get("sourcePagination").hidden = get("sourceZoom").hidden =
      doc.kind !== "pdf";
    if (doc.kind === "pdf") {
      setStatus("");
      pageInput.max = String(doc.page_count);
      get("pdfPageCount").textContent = `/ ${doc.page_count}`;
      setZoom(1);
      active.inFlight = 0;
      active.slots = Array.from({ length: doc.page_count }, (_, index) => {
        const number = index + 1;
        const node = el("div", "source-pdf-page");
        node.dataset.page = String(number);
        const slot = { number, node };
        resetSlot(slot);
        return slot;
      });
      pages.replaceChildren(...active.slots.map((slot) => slot.node));
      pages.hidden = false;
      jump(citation?.evidence.page || 1);
      if (citation?.evidence.page === active.page)
        active.focusPage = active.page;
      return;
    }
    return loadText(active);
  }

  picker.addEventListener("change", () => {
    const doc = sources.find((item) => item.id === picker.value);
    if (doc) open(doc);
  });
  previous.addEventListener("click", () => jump(active.page - 1));
  next.addEventListener("click", () => jump(active.page + 1));
  pageInput.addEventListener("change", () => jump(Number(pageInput.value)));
  pageInput.addEventListener("blur", () => {
    if (active?.slots) pageInput.value = String(active.page);
  });
  viewport.addEventListener("scroll", schedulePages, { passive: true });
  if (typeof ResizeObserver === "function")
    new ResizeObserver(schedulePages).observe(viewport);
  zoomOut.addEventListener("click", () => setZoom(zoom - 0.25));
  zoomIn.addEventListener("click", () => setZoom(zoom + 0.25));
  get("pdfZoomResetBtn").addEventListener("click", () => setZoom(1));
  window.addEventListener("pagehide", (event) => {
    if (event.persisted) return;
    resetContent();
  });
  return {
    setDocuments,
    show,
    open,
    clear,
    get document() {
      return active?.doc;
    },
  };
}
