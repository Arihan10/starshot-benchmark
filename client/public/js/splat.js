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

// Compact images/second for the live capture readout, or null when not meaningful.
function fmtRate(r) {
    if (r == null || !isFinite(r) || r <= 0) return null;
    return r >= 10 ? String(Math.round(r)) : r.toFixed(1);
}

function subRunning(s) {
    return !!s && (s.status === "running" || s.status === "pending");
}

function anyRunning() {
    return cells.some(
        (c) =>
            isRunning(c) ||
            subRunning(c.stage2) ||
            subRunning(c.stage3) ||
            subRunning(c.stage4) ||
            subRunning(c.stage5),
    );
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

// Stage 3 (surfel sampling → cloud.ply). Consumes the Stage-2 free-space grid to
// orient normals + cull hidden faces, so it's gated on Stage 2 being done.
async function startSurfels(cell) {
    if (subRunning(cell.stage3)) return;
    const s = (cell.stage3 = cell.stage3 || {});
    s.status = "pending";
    s.done = 0;
    s.error = null;
    renderCells();
    try {
        await api.splatStage3Start(selectedRun, cell.slot, cell.model);
    } catch (e) {
        s.status = "error";
        s.error = e.message;
        renderCells();
        return;
    }
    ensurePoll(selectedRun);
}

function openView(cell) {
    const s = cell.stage3;
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

// Clicking a cell opens its dedicated pipeline PAGE (the splat viewer): the main
// canvas (empty until surfels run) + the side stepper that drives every stage.
function openCellPage(cell) {
    openSplatViewer({
        run: selectedRun,
        slot: cell.slot,
        model: cell.model,
        source: cell.source,
        label: `${cell.slot} · ${cell.model}${cell.source ? ` · ${cell.source}` : ""}`,
    });
}

// The surfel (Stage 3) control on a cell row: "sample splat" → progress → "view
// splat". Gated on Stage 2 (free-space) being done first.
function surfelControl(cell) {
    const s = cell.stage3 || {};
    const wrap = el("div", { class: "splat-stage2" });
    const gate =
        cell.stage2 && cell.stage2.status === "done"
            ? null
            : "compute free space first (Stage 2)";
    let btn;
    if (subRunning(s)) {
        const prog = s.total ? `${s.done}/${s.total}` : "…";
        btn = el("button", {
            class: "splat-stage2-btn",
            disabled: true,
            text: `sampling ${prog}`,
        });
    } else if (gate) {
        btn = el("button", {
            class: "splat-stage2-btn",
            disabled: true,
            text: "sample splat",
            title: gate,
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
                startSurfels(cell);
            },
        });
    } else {
        btn = el("button", {
            class: "splat-stage2-btn",
            text: "sample splat",
            title: "sample this cell's meshes into a Gaussian cloud (Stage 3)",
            onclick: (ev) => {
                ev.stopPropagation();
                startSurfels(cell);
            },
        });
    }
    wrap.appendChild(btn);
    return wrap;
}

// Stage 2 (free-space grid) control — the shared foundation Stage 3 (surfels) and
// Stage 4 (cameras) consume. Runs straight off a cell's build (no dependency).
function freeSpaceControl(cell) {
    return stageControl(cell, {
        key: "stage2",
        runningVerb: "voxelizing",
        idleLabel: "free space",
        idleTitle: "compute the free-space grid (Stage 2) — needed by surfels + cameras",
        doneLabel: "recompute",
        doneTitle: "recompute the free-space grid",
        errLabel: "free space failed — retry",
        doneText: (s) => {
            const n = s.summary && s.summary.reachable_voxels;
            return n != null ? `${n.toLocaleString()} vox` : null;
        },
        start: (c) => startStage(c, "stage2", api.splatStage2Start),
    });
}

// Generic per-cell stage control (Stage 4 camera plan, Stage 5 references) —
// mirrors the Stage-2 control: trigger → live progress → done (re-runnable) /
// retry. Stage 5 is gated on Stage 4 (it needs the camera plan).
async function startStage(cell, key, apiStart) {
    const s = (cell[key] = cell[key] || {});
    if (subRunning(s)) return;
    s.status = "pending";
    s.done = 0;
    s.error = null;
    renderCells();
    try {
        await apiStart(selectedRun, cell.slot, cell.model);
    } catch (e) {
        s.status = "error";
        s.error = e.message;
        renderCells();
        return;
    }
    ensurePoll(selectedRun);
}

function stageControl(cell, opts) {
    const s = cell[opts.key] || {};
    const wrap = el("div", { class: "splat-stage2" }); // reuse the Stage-2 styling
    const gate = opts.gate ? opts.gate(cell) : null;
    let btn;
    if (subRunning(s)) {
        let label;
        if (s.phase === "deopt") {
            label = "de-optimizing…";
        } else if (s.phase === "tier") {
            label = "building tier…"; // splat asset tier build (shared warm-up)
        } else {
            const prog = s.total ? `${s.done}/${s.total}` : "…";
            // Stage 5 carries a live images/second; other stages don't set `rate`.
            const rate = fmtRate(s.rate);
            label = rate
                ? `${opts.runningVerb} ${prog} · ${rate}/s`
                : `${opts.runningVerb} ${prog}`;
        }
        btn = el("button", { class: "splat-stage2-btn", disabled: true, text: label });
    } else if (gate) {
        btn = el("button", {
            class: "splat-stage2-btn",
            disabled: true,
            text: opts.idleLabel,
            title: gate,
        });
    } else if (s.status === "done") {
        const info = opts.doneText(s);
        if (info) wrap.appendChild(el("span", { class: "muted", text: info }));
        btn = el("button", {
            class: "splat-stage2-btn view",
            text: opts.doneLabel,
            title: opts.doneTitle || "",
            onclick: (ev) => {
                ev.stopPropagation();
                opts.start(cell);
            },
        });
    } else if (s.status === "error") {
        btn = el("button", {
            class: "splat-stage2-btn err",
            text: opts.errLabel,
            title: s.error || "",
            onclick: (ev) => {
                ev.stopPropagation();
                opts.start(cell);
            },
        });
    } else {
        btn = el("button", {
            class: "splat-stage2-btn",
            text: opts.idleLabel,
            title: opts.idleTitle || "",
            onclick: (ev) => {
                ev.stopPropagation();
                opts.start(cell);
            },
        });
    }
    wrap.appendChild(btn);
    return wrap;
}

// Stage 4 — coverage camera plan (cameras.json). Consumes the Stage-2 free-space
// grid + Stage-3 surfel cloud, so it's gated on Stage 3 being done.
function stage4Control(cell) {
    return stageControl(cell, {
        key: "stage4",
        runningVerb: "planning",
        idleLabel: "plan cameras",
        idleTitle: "plan coverage cameras for this cell (Stage 4)",
        doneLabel: "re-plan",
        doneTitle: "re-plan coverage cameras",
        errLabel: "cameras failed — retry",
        doneText: (s) => {
            const n = s.summary && s.summary.cameras;
            return n != null ? `${n} cams` : null;
        },
        gate: (c) =>
            c.stage3 && c.stage3.status === "done"
                ? null
                : "sample splat first (Stage 3)",
        start: (c) => startStage(c, "stage4", api.splatStage4Start),
    });
}

// Stage 5 — unlit reference renders (refs/), gated on the Stage-4 camera plan.
// Rendered by the headless WebGL capture page against the cell's splat tier;
// on failure the job's error carries the capture URL for a manual-browser run.
function stage5Control(cell) {
    return stageControl(cell, {
        key: "stage5",
        runningVerb: "rendering",
        idleLabel: "render refs",
        idleTitle: "render unlit reference images from the camera plan (Stage 5)",
        doneLabel: "re-render",
        doneTitle: "re-render reference images",
        errLabel: "refs failed — retry",
        doneText: (s) => {
            const n = s.summary && s.summary.views;
            if (n == null) return null;
            const r = s.summary && s.summary.img_per_s;
            return r ? `${n} views · ${r}/s` : `${n} views`;
        },
        gate: (c) =>
            c.stage4 && c.stage4.status === "done"
                ? null
                : "plan cameras first (Stage 4)",
        start: (c) => startStage(c, "stage5", api.splatStage5Start),
    });
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
            text: `${cell.total} objects${cell.source ? ` · ${cell.source}` : ""} · click to open`,
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
        freeSpaceControl(cell),
        surfelControl(cell),
        stage4Control(cell),
        stage5Control(cell),
    );
    row.addEventListener("click", () => openCellPage(cell));
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
