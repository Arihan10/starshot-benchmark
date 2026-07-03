// A searchable run combobox: a button showing the selected run that opens a
// filter box over a name-sorted, click-to-select list (keyboard-navigable) —
// the replacement for native <select>s whose order couldn't be sorted. Reused
// by the topbar's active-run picker and the run-compare "run B" selector, each
// driven by whatever run set / selection / pick handler the caller supplies.
//
// `container` is a positioned element (given the `run-combo` styling) the
// combobox mounts into. Options:
//   getRuns()      → the candidate run objects [{name, prompt_version?}, …]
//   getSelected()  → the currently-selected run name (or null)
//   onPick(name)   → invoked when a run is chosen
//   buttonLabel(r) → the trigger's text for the selected run (default: its name)
//   placeholder    → the trigger's text when nothing is selected
// Returns { render } to refresh the button label + open list after the run set
// or selection changes.

import { el } from "./ui.js";

export function createRunCombo(
  container,
  { getRuns, getSelected, onPick, buttonLabel = (r) => r.name, placeholder = "select a run" },
) {
  let open = false;
  let activeIdx = -1; // keyboard-highlighted row in the filtered list

  const labelEl = el("span", { class: "run-combo-lab", text: "—" });
  const btn = el("button", {
    class: "run-combo-btn",
    title: "click to search runs by name",
    onclick: () => (open ? close() : openIt()),
  }, labelEl, el("span", { class: "run-combo-caret", text: "▾" }));
  const search = el("input", {
    type: "text", class: "run-combo-search", spellcheck: "false",
    placeholder: "search runs by name…",
  });
  search.addEventListener("input", () => { activeIdx = -1; renderList(); });
  search.addEventListener("keydown", onKey);
  const list = el("div", { class: "run-combo-list" });
  container.append(btn, el("div", { class: "run-combo-pop" }, search, list));
  // Dismiss on any click outside the combobox.
  document.addEventListener("mousedown", (ev) => {
    if (open && !container.contains(ev.target)) close();
  });

  function openIt() {
    open = true;
    container.classList.add("open");
    search.value = "";
    activeIdx = -1;
    renderList();
    search.focus();
  }
  function close() {
    open = false;
    container.classList.remove("open");
  }

  // Candidate runs sorted by name (numeric-aware, case-insensitive) — the stable
  // order a native <select> never gave — then narrowed by the search box.
  function filtered() {
    const q = search.value.trim().toLowerCase();
    const runs = [...getRuns()].sort((a, b) =>
      a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
    return q ? runs.filter((r) => r.name.toLowerCase().includes(q)) : runs;
  }

  function renderList() {
    const runs = filtered();
    list.textContent = "";
    if (!runs.length) {
      list.appendChild(el("div", { class: "run-combo-empty",
        text: getRuns().length ? "no runs match your search" : "no runs yet" }));
      return;
    }
    const sel = getSelected();
    runs.forEach((r, i) => {
      list.appendChild(el("div", {
        class: `run-combo-item${r.name === sel ? " current" : ""}${i === activeIdx ? " active" : ""}`,
        onclick: () => { close(); onPick(r.name); },
      },
        el("span", { class: "run-combo-item-name", text: r.name }),
        el("span", { class: "run-combo-item-ver", text: r.prompt_version ?? "legacy" }),
      ));
    });
    list.querySelector(".run-combo-item.active")?.scrollIntoView({ block: "nearest" });
  }

  function onKey(ev) {
    const runs = filtered();
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      activeIdx = Math.min(activeIdx + 1, runs.length - 1);
      renderList();
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      activeIdx = Math.max(activeIdx - 1, 0);
      renderList();
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      const r = runs[activeIdx] ?? runs[0];
      if (r) { close(); onPick(r.name); }
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      close();
    }
  }

  function render() {
    const sel = getSelected();
    const cur = getRuns().find((r) => r.name === sel);
    labelEl.textContent = cur ? buttonLabel(cur) : (sel || placeholder);
    if (open) renderList();
  }

  render();
  return { render };
}
