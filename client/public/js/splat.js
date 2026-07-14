// The splat screen — Stage 1 (the scene assembler). Selecting a run LISTS every
// convertible cell in it (those with a generated or library build) and their state.
// Clicking a cell starts Stage 1 for THAT cell only: its raw objects-generated/
// + events.jsonl → a validated `splat/scene.json` manifest (see splat/stage1.py),
// with a live progress bar. A done cell can be clicked again to re-convert.
//
// The run picker is a private combobox — picking a run here does NOT switch the
// board's active run; it only lists that run's cells for conversion.

import { api } from "./api.js";
import { state } from "./state.js";
import { el } from "./ui.js";
import { createRunCombo } from "./runcombo.js";
import { openSplatViewer, initSplatViewer } from "./splatviewer.js";

let root = null;
let subEl = null;
let statusEl = null;
let cellsEl = null;
let runCombo = null;

let selectedRun = null;
let cells = []; // last-known cell states for selectedRun
let pollTimer = null;
let loadSeq = 0; // guards against out-of-order run switches

const POLL_MS = 1000;

function runLabel(r) {
    return r.prompt_version
        ? `${r.name} · ${r.prompt_version}`
        : `${r.name} (legacy)`;
}

function isRunning(cell) {
    return cell.status === "running" || cell.status === "pending";
}

function stage2Running(cell) {
    const s = cell.stage2;
    return !!s && (s.status === "running" || s.status === "pending");
}

function anyRunning() {
    return cells.some((c) => isRunning(c) || stage2Running(c));
}

function stopPoll() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function ensurePoll(run) {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
        if (run !== selectedRun) return stopPoll();
        let payload;
        try {
            payload = await api.splatStageCells(run);
        } catch {
            return; // transient — retry next tick
        }
        if (run !== selectedRun) return stopPoll();
        cells = payload.cells ?? [];
        renderCells();
        if (!anyRunning()) stopPoll();
    }, POLL_MS);
}

async function startCell(cell) {
    if (isRunning(cell)) return;
    // Optimistic: flip to pending immediately so the click feels responsive.
    cell.status = "pending";
    cell.done = 0;
    cell.current_id = null;
    cell.error = null;
    renderCells();
    try {
        await api.splatStage1Start(selectedRun, cell.slot, cell.model);
    } catch (e) {
        cell.status = "error";
        cell.error = e.message;
        renderCells();
        return;
    }
    ensurePoll(selectedRun);
}

// Stage 2 (surfel sampling → cloud.ply). Independent of Stage 1 — placement is
// baked into the meshes — so it can run straight off a cell's build.
async function startStage2(cell) {
    if (stage2Running(cell)) return;
    const s = (cell.stage2 = cell.stage2 || {});
    s.status = "pending";
    s.done = 0;
    s.error = null;
    renderCells();
    try {
        await api.splatStage2Start(selectedRun, cell.slot, cell.model);
    } catch (e) {
        s.status = "error";
        s.error = e.message;
        renderCells();
        return;
    }
    ensurePoll(selectedRun);
}

function openView(cell) {
    const s = cell.stage2;
    if (!s || !s.url) return;
    openSplatViewer({
        run: selectedRun,
        slot: cell.slot,
        model: cell.model,
        source: cell.source,
        url: api.absUrl(s.url),
        detailUrl: s.detail_url ? api.absUrl(s.detail_url) : null,
        summary: s.summary,
        label: `${cell.slot} · ${cell.model}${cell.source ? ` · ${cell.source}` : ""}`,
    });
}

// The right-hand Stage-2 control on a cell row: "sample splat" → progress →
// "view splat". Its own button, so clicking it doesn't trigger the row's
// Stage-1 convert.
function stage2Control(cell) {
    const s = cell.stage2 || {};
    const wrap = el("div", { class: "splat-stage2" });
    let btn;
    if (stage2Running(cell)) {
        const prog = s.total ? `${s.done}/${s.total}` : "…";
        btn = el("button", {
            class: "splat-stage2-btn",
            disabled: true,
            text: `sampling ${prog}`,
        });
    } else if (s.status === "done") {
        const n = s.summary && s.summary.splats;
        if (n) {
            wrap.appendChild(
                el("span", { class: "muted", text: n.toLocaleString() }),
            );
        }
        btn = el("button", {
            class: "splat-stage2-btn view",
            text: "view splat",
            title: "open the pre-fine-tuning Gaussian cloud",
            onclick: (ev) => {
                ev.stopPropagation();
                openView(cell);
            },
        });
    } else if (s.status === "error") {
        btn = el("button", {
            class: "splat-stage2-btn err",
            text: "splat failed — retry",
            title: s.error || "",
            onclick: (ev) => {
                ev.stopPropagation();
                startStage2(cell);
            },
        });
    } else {
        btn = el("button", {
            class: "splat-stage2-btn",
            text: "sample splat",
            title: "sample this cell's meshes into a Gaussian cloud (Stage 2)",
            onclick: (ev) => {
                ev.stopPropagation();
                startStage2(cell);
            },
        });
    }
    wrap.appendChild(btn);
    return wrap;
}

function cellRow(cell) {
    const running = isRunning(cell);
    const pct =
        cell.total > 0
            ? Math.round((cell.done / cell.total) * 100)
            : cell.status === "done"
              ? 100
              : 0;
    const fill = el("div", { class: "splat-bar-fill" });
    fill.style.width = `${pct}%`;
    const bar = el("div", { class: "splat-bar" }, fill);

    let right;
    if (cell.status === "error") {
        right = el("span", { class: "muted", text: "error — click to retry" });
    } else if (cell.status === "done") {
        const c = cell.summary && cell.summary.counts;
        right = el("span", {
            class: "muted",
            text: c ? `${c.placed} placed · ${c.missing_holes} holes` : "done",
        });
    } else if (running) {
        right = el("span", { class: "muted", text: `${cell.done}/${cell.total}` });
    } else {
        right = el("span", {
            class: "muted",
            text: `${cell.total} objects${cell.source ? ` · ${cell.source}` : ""} · click to convert`,
        });
    }

    const row = el(
        "div",
        {
            class: `splat-cell ${cell.status}${running ? "" : " clickable"}`,
            title:
                cell.error ||
                (cell.current_id ? `converting ${cell.current_id}…` : cell.model),
        },
        el("span", { class: "splat-cell-model", text: cell.model }),
        bar,
        right,
        stage2Control(cell),
    );
    if (!running) row.addEventListener("click", () => startCell(cell));
    return row;
}

function renderCells() {
    cellsEl.textContent = "";
    if (!cells.length) {
        subEl.textContent = "";
        statusEl.textContent = "no convertible cells";
        statusEl.style.color = "";
        cellsEl.appendChild(
            el("div", {
                class: "splat-empty",
                text: `no convertible cells in “${selectedRun}” — run the pipeline for a slot first`,
            }),
        );
        return;
    }

    const done = cells.filter((c) => c.status === "done").length;
    const running = cells.filter(isRunning).length;
    subEl.textContent = `${cells.length} cell${cells.length === 1 ? "" : "s"} in “${selectedRun}”`;
    if (running) {
        statusEl.textContent = `converting ${running}…`;
        statusEl.style.color = "var(--purple)";
    } else {
        statusEl.textContent = `${done}/${cells.length} converted`;
        statusEl.style.color = done ? "var(--green)" : "";
    }

    // Group by slot, preserving the server's order.
    const bySlot = new Map();
    for (const c of cells) {
        if (!bySlot.has(c.slot)) bySlot.set(c.slot, []);
        bySlot.get(c.slot).push(c);
    }
    for (const [slot, models] of bySlot) {
        cellsEl.appendChild(
            el(
                "div",
                { class: "splat-slot" },
                el("div", { class: "splat-slot-name", text: slot }),
                el("div", { class: "splat-rows" }, models.map(cellRow)),
            ),
        );
    }
}

async function selectRun(run) {
    if (!run) return;
    selectedRun = run;
    runCombo?.render();
    const seq = ++loadSeq;
    stopPoll();
    cells = [];
    subEl.textContent = "";
    statusEl.textContent = "loading cells…";
    statusEl.style.color = "";
    cellsEl.textContent = "";
    cellsEl.appendChild(
        el("div", { class: "splat-empty", text: "loading completed cells…" }),
    );

    let payload;
    try {
        payload = await api.splatStageCells(run);
    } catch (e) {
        if (seq !== loadSeq) return;
        statusEl.textContent = "failed";
        cellsEl.textContent = "";
        cellsEl.appendChild(
            el("div", {
                class: "splat-empty",
                text: `failed to load cells: ${e.message}`,
            }),
        );
        return;
    }
    if (seq !== loadSeq) return; // a later run pick superseded this fetch
    cells = payload.cells ?? [];
    renderCells();
    if (anyRunning()) ensurePoll(run);
}

function openSplat() {
    root.classList.add("open");
    document.getElementById("btn-splat-view")?.classList.add("on");
    runCombo?.render();
    // Pull a fresh run list so runs finished since boot show up, then repaint.
    api.runs()
        .then((p) => {
            state.runs = p.runs;
            runCombo?.render();
        })
        .catch(() => {
            /* keep the last-known run list */
        });
}

function closeSplat() {
    root.classList.remove("open");
    document.getElementById("btn-splat-view")?.classList.remove("on");
    stopPoll();
}

export function initSplat() {
    root = document.getElementById("splat");
    subEl = document.getElementById("splat-sub");
    statusEl = document.getElementById("splat-selection");
    cellsEl = document.getElementById("splat-cells");
    runCombo = createRunCombo(document.getElementById("splat-run-picker"), {
        getRuns: () => state.runs,
        getSelected: () => selectedRun,
        onPick: selectRun,
        buttonLabel: runLabel,
        placeholder: "select a run",
    });
    document
        .getElementById("btn-splat-view")
        .addEventListener("click", () =>
            root.classList.contains("open") ? closeSplat() : openSplat(),
        );
    document.getElementById("splat-close").addEventListener("click", closeSplat);
    initSplatViewer();
    // Esc closes the screen (unless a modal is up), matching the other screens.
    document.addEventListener("keydown", (ev) => {
        if (
            ev.key === "Escape" &&
            root.classList.contains("open") &&
            !document.getElementById("modal-root").firstChild
        ) {
            closeSplat();
        }
    });
    statusEl.textContent = "no run selected";
}
