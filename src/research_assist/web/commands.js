import { $, $$, el, toast } from "./dom.js";

const groups = [
  [
    "Nghiên cứu tài liệu",
    [
      [
        "/tom-tat",
        "Tóm tắt tài liệu",
        "Tóm tắt mục tiêu, phương pháp, đóng góp và kết quả chính của tài liệu đã chọn.",
      ],
      [
        "/so-sanh",
        "So sánh tài liệu",
        "Đối chiếu phương pháp, dữ liệu, kết quả và hạn chế giữa các tài liệu đã chọn.",
      ],
      [
        "/dong-gop",
        "Đóng góp và điểm mới",
        "Phân tích đóng góp được tác giả nêu và bằng chứng đi kèm.",
      ],
      [
        "/phuong-phap",
        "Phương pháp và kiến trúc",
        "Giải thích phương pháp, kiến trúc và quy trình thực nghiệm.",
      ],
      [
        "/kiem-chung",
        "Kiểm chứng nhận định",
        "Đối chiếu nhận định với tài liệu. Chỉ rõ bằng chứng hỗ trợ, mâu thuẫn hoặc phần còn thiếu căn cứ.",
      ],
      [
        "/so-lieu",
        "Số liệu và benchmark",
        "Trích xuất số liệu thực nghiệm, chỉ số, đơn vị và điều kiện đo kèm vị trí nguồn.",
      ],
      [
        "/thuat-ngu",
        "Giải thích thuật ngữ",
        "Giải thích thuật ngữ và từ viết tắt dựa trên định nghĩa trong tài liệu.",
      ],
      [
        "/trich-dan",
        "Trích dẫn nguyên văn",
        "Tìm đoạn bằng chứng nguyên văn và chỉ rõ trang hoặc mục chứa đoạn trích.",
      ],
      [
        "/khoang-trong",
        "Hạn chế và khoảng trống",
        "Nêu hạn chế và hướng nghiên cứu tiếp theo được tài liệu đề cập.",
      ],
      [
        "/cau-hoi",
        "Câu hỏi phản biện",
        "Đặt câu hỏi về giả định, hạn chế và vấn đề chưa được giải đáp trong tài liệu.",
      ],
      [
        "/scout",
        "Rà nhanh tài liệu",
        "Rà nhanh cấu trúc, mục lục, phạm vi và các phần chính của tài liệu đã chọn.",
      ],
    ],
  ],
  [
    "Tìm nguồn",
    [
      [
        "/tim-bai",
        "Tìm bài báo",
        "Tìm theo từ khóa trên arXiv và Crossref.",
        "papers",
      ],
    ],
  ],
  [
    "Thao tác",
    [
      [
        "/cpu-min",
        "Demo trích xuất",
        "Chọn model Demo cho cả năm agent.",
        "demo",
      ],
      [
        "/cpu-max",
        "Local CPU",
        "Chọn model CPU đã cấu hình cho cả năm agent.",
        "cpu",
      ],
      [
        "/xuat-bao-cao",
        "Xuất báo cáo",
        "Tải báo cáo hiện tại dưới dạng Markdown.",
        "export",
      ],
      ["/clear", "Xóa ô nhập", "Xóa nội dung đang soạn.", "clear"],
    ],
  ],
];
const commands = groups.flatMap(([group, rows]) =>
  rows.map(([name, label, description, action]) => ({
    group,
    name,
    label,
    description,
    action,
  })),
);
const aliases = { "/fact-check": "/kiem-chung" };

function searchable(text) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .replaceAll("đ", "d");
}

export function readCommand(value) {
  const match = /^(\/[\p{L}\p{M}\d_-]+)(?:\s|$)/u.exec(value);
  if (!match) return null;
  const name = searchable(match[1]);
  return {
    command: commands.find((item) => item.name === (aliases[name] ?? name)),
    argument: value.slice(match[1].length).trim(),
  };
}

export function expandQuestion(value) {
  const parsed = readCommand(value);
  if (!parsed?.command || parsed.command.action) return value;
  return [parsed.command.description, parsed.argument]
    .filter(Boolean)
    .join("\n\n");
}

export function mountCommandMenu({
  input,
  anchor,
  preferAbove = false,
  insertPrompt = false,
  onResearch,
  onAction,
}) {
  const popup = el("div", "command-menu");
  popup.id = "command-popup";
  popup.hidden = true;
  const list = el("div", "command-menu-list");
  list.id = "command-options";
  list.setAttribute("role", "listbox");
  list.setAttribute("aria-label", "Lệnh nhanh");
  const empty = el("p", "command-menu-empty", "Không có lệnh phù hợp.");
  empty.setAttribute("role", "status");
  const hints = el("div", "command-menu-hints");
  for (const [keys, label] of [
    ["↑ ↓", "Di chuyển"],
    ["Enter / Tab", "Chọn"],
    ["Esc", "Đóng"],
  ]) {
    const hint = el("span");
    hint.append(el("kbd", "", keys), ` ${label}`);
    hints.append(hint);
  }
  popup.append(list, empty, hints);
  anchor.append(popup);
  input.setAttribute("aria-controls", list.id);
  input.setAttribute("aria-autocomplete", "list");
  input.setAttribute("aria-expanded", "false");
  let matches = [],
    index = 0;

  function position() {
    if (popup.hidden) return;
    const viewport = window.visualViewport;
    const top = viewport?.offsetTop ?? 0;
    const bottom = top + (viewport?.height ?? window.innerHeight);
    const box = anchor.getBoundingClientRect();
    const above = box.top - top - 16;
    const below = bottom - box.bottom - 16;
    const preferred = preferAbove ? above : below;
    const alternative = preferAbove ? below : above;
    const placeAbove =
      preferred >= 280 || preferred >= alternative ? preferAbove : !preferAbove;
    popup.classList.toggle("above", placeAbove);
    popup.style.setProperty(
      "--command-room",
      `${Math.max(0, placeAbove ? above : below)}px`,
    );
  }

  function close() {
    popup.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }

  function choose(command, argument = "") {
    close();
    if (command.action) {
      input.value = command.action === "clear" ? "" : argument;
    } else {
      input.value = insertPrompt ? command.description : `${command.name} `;
    }
    input.dispatchEvent(new Event("input"));
    input.focus();
    if (!command.action) onResearch?.();
    else if (command.action !== "clear") onAction(command.action, argument);
  }

  function highlight() {
    const options = list.querySelectorAll('[role="option"]');
    options.forEach((option, i) =>
      option.setAttribute("aria-selected", String(i === index)),
    );
    const active = options[index];
    if (active) {
      input.setAttribute("aria-activedescendant", active.id);
      active.scrollIntoView?.({ block: "nearest" });
    } else input.removeAttribute("aria-activedescendant");
  }

  function refresh() {
    const match = /^\/([\p{L}\p{M}\d_-]*)$/u.exec(input.value);
    if (!match) return close();
    const query = searchable(match[1]);
    const alias = "fact-check".startsWith(query) ? aliases["/fact-check"] : "";
    matches = commands.filter(
      ({ name, label, description }) =>
        name.includes(query) ||
        name === alias ||
        searchable(`${label} ${description}`).includes(query),
    );
    index = 0;
    list.replaceChildren();
    let group;
    for (const [i, item] of matches.entries()) {
      if (item.group !== group) {
        group = item.group;
        const heading = el("div", "command-menu-group", group);
        heading.setAttribute("aria-hidden", "true");
        list.append(heading);
      }
      const option = el("button", "command-menu-option");
      option.type = "button";
      option.tabIndex = -1;
      option.id = `command-option-${i}`;
      option.setAttribute("role", "option");
      const title = el("span", "command-menu-title");
      title.append(el("strong", "", item.label), el("code", "", item.name));
      option.append(
        title,
        el("span", "command-menu-description", item.description),
      );
      option.addEventListener("mousedown", (event) => event.preventDefault());
      option.addEventListener("click", () => choose(item));
      list.append(option);
    }
    empty.hidden = matches.length > 0;
    popup.hidden = false;
    input.setAttribute("aria-expanded", "true");
    position();
    list.scrollTop = 0;
    highlight();
  }

  input.addEventListener("input", (event) => {
    if (!event.isComposing) refresh();
  });
  input.addEventListener("compositionend", refresh);
  input.addEventListener("keydown", (event) => {
    if (popup.hidden || event.isComposing) return;
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    } else if (["ArrowUp", "ArrowDown"].includes(event.key) && matches.length) {
      event.preventDefault();
      index =
        (index + (event.key === "ArrowDown" ? 1 : -1) + matches.length) %
        matches.length;
      highlight();
    } else if (
      ["Enter", "Tab"].includes(event.key) &&
      matches.length &&
      !event.shiftKey &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey
    ) {
      event.preventDefault();
      choose(matches[index]);
    }
  });
  document.addEventListener("pointerdown", (event) => {
    if (event.target !== input && !popup.contains(event.target)) close();
  });
  document.addEventListener("focusin", (event) => {
    if (event.target !== input && !popup.contains(event.target)) close();
  });
  window.addEventListener("resize", position);
  window.addEventListener("scroll", position, true);
  window.visualViewport?.addEventListener("resize", position);
  window.visualViewport?.addEventListener("scroll", position);

  return {
    close,
    runAction(parsed) {
      if (!parsed?.command?.action) return false;
      choose(parsed.command, parsed.argument);
      return true;
    },
  };
}

export function initializeCommands(selectPreset) {
  const input = $("#question");
  const menu = mountCommandMenu({
    input,
    anchor: $(".question-composer"),
    insertPrompt: true,
    onAction(action, argument) {
      if (action === "export") {
        const link = $("#export-md");
        if (link.getAttribute("href") === "#")
          toast("Chưa có báo cáo để xuất.", true);
        else link.click();
      } else if (action === "papers") {
        $('[data-view="library"]').click();
        $("#scout-form").closest("details").open = true;
        $("#scout-query").value = argument;
        $("#scout-query").focus();
      } else if (selectPreset(action)) {
        toast("Đã chọn cấu hình model cho cả năm agent.");
      }
    },
  });
  $$(".quick-chip").forEach((button) =>
    button.addEventListener("click", () => {
      input.value = button.dataset.prompt;
      input.dispatchEvent(new Event("input"));
      input.focus();
    }),
  );
  return menu;
}
