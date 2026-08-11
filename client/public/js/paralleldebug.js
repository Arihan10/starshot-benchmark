// Parallel-flow debugger. Answers two questions about one cell:
//
//   WHEN did each call run — drawn as a schedule, so overlap is visible as bars
//   sharing a column and a stall is visible as a red band where nothing landed.
//   WHY is a zone's interior still waiting — its gate's frontier, annotated with
//   which regions actually touch it and which the ray cast merely reached across
//   open space.
//
// All the analysis is server-side (`/parallel-debug`), computed by the same
// helpers the scheduler itself uses, so this can't drift from the real decision.

import { SERVER_URL, api } from "./api.js";

const $ = (id) => document.getElementById(id);
const runSel = $("run");
const cellSel = $("cell");
const chartwrap = $("chartwrap");
const gatelist = $("gatelist");
const statsEl = $("stats");
const tip = $("tip");

// One colour per pipeline step. The structural steps get distinct hues because
// telling them apart in the chart IS the point; the secondary traffic collapses
// to one muted grey, since it is off the critical path and only ever shown as
// background texture.
const STEP_COLOR = {
    zone_plan: "#7fb3ff",
    zone_decompose: "#9d8bff",
    child_bbox_batch: "#5fd0d8",
    encapsulating_decompose: "#e8c07d",
    anchor_decompose: "#8bd17c",
    object_bbox_batch: "#4a9d5f",
    next_object: "#c8e07d",
    negative_space_decompose: "#d88bd1",
};
const SECONDARY_COLOR = "#3d434e";
const STRUCTURAL = new Set(Object.keys(STEP_COLOR));

let data = null;
let mode = "step"; // "step" | "zone"
let structuralOnly = true;
let autoTimer = null;

// --- data ---------------------------------------------------------------------

async function loadRuns() {
    const { runs } = await api.runs();
    runSel.innerHTML = "";
    for (const r of runs) {
        const name = typeof r === "string" ? r : r.name;
        runSel.append(new Option(name, name));
    }
    const saved = localStorage.getItem("pdbg.run");
    if (saved && [...runSel.options].some((o) => o.value === saved)) runSel.value = saved;
    await loadCells();
}

async function loadCells() {
    const run = runSel.value;
    localStorage.setItem("pdbg.run", run);
    cellSel.innerHTML = "";
    const { slots } = await api.slots(run);
    for (const s of slots) {
        for (const [model, v] of Object.entries(s.runs ?? {})) {
            if (!v.events_count) continue; // never-started cells have nothing to show
            cellSel.append(
                new Option(`${s.id} · ${model}  (${v.events_count} ev, ${v.status})`, `${s.id}|${model}`),
            );
        }
    }
    if (!cellSel.options.length) {
        cellSel.append(new Option("no started cells in this run", ""));
        render(null);
        return;
    }
    const saved = localStorage.getItem("pdbg.cell");
    if (saved && [...cellSel.options].some((o) => o.value === saved)) cellSel.value = saved;
    await loadCell();
}

async function loadCell() {
    const v = cellSel.value;
    if (!v) return;
    localStorage.setItem("pdbg.cell", v);
    const [slot, model] = v.split("|");
    chartwrap.innerHTML = `<div id="msg">loading ${slot} · ${model}…</div>`;
    try {
        const url = new URL(
            `/slots/${encodeURIComponent(slot)}/${encodeURIComponent(model)}/parallel-debug`,
            SERVER_URL,
        );
        url.searchParams.set("run", runSel.value);
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
        data = await res.json();
    } catch (e) {
        chartwrap.innerHTML = `<div id="msg" class="err">failed: ${e.message}</div>`;
        return;
    }
    render(data);
}

// --- summary ------------------------------------------------------------------

function fmt(s) {
    if (s < 90) return `${s.toFixed(0)}s`;
    if (s < 5400) return `${(s / 60).toFixed(1)}m`;
    return `${(s / 3600).toFixed(2)}h`;
}

function renderStats(d) {
    const idleS = d.idle.reduce((a, g) => a + (g.end - g.start), 0);
    const anchors = d.zones.filter((z) => z.gate !== null);
    const released = anchors.filter((z) => z.released);
    const forced = released.filter((z) => z.released.forced);
    const queued = anchors.filter((z) => !z.released);
    // Work packed into the time something was actually running. >1 means the
    // schedule overlapped; ~1 means it ran serially.
    const packing = d.busy_s > 0 ? d.flight_s / d.busy_s : 0;
    const cells = [
        ["wall", fmt(d.wall_s), ""],
        ["busy", fmt(d.busy_s), "something in flight"],
        ["idle", fmt(idleS), `${d.idle.length} gaps`, idleS > d.wall_s * 0.2 ? "bad" : ""],
        ["packing", `${packing.toFixed(2)}x`, "work ÷ busy", packing >= 2 ? "good" : packing < 1.3 ? "warn" : ""],
        ["calls", String(d.flights.length), ""],
        ["anchors", `${released.length}/${anchors.length}`, "released"],
        ["mid-walk", String(released.length - forced.length), "not forced", "good"],
        ["at drain", String(forced.length), "forced", forced.length ? "warn" : ""],
        ["queued", String(queued.length), "gate shut", queued.length ? "warn" : ""],
        ["status", d.status, ""],
    ];
    statsEl.innerHTML = "";
    for (const [label, value, note, tone] of cells) {
        const el = document.createElement("div");
        el.className = `s ${tone || ""}`;
        el.innerHTML = `<span>${label}</span><b></b>${note ? `<span>${note}</span>` : ""}`;
        el.querySelector("b").textContent = value;
        statsEl.append(el);
    }
}

// --- schedule chart -----------------------------------------------------------

function renderChart(d) {
    const flights = structuralOnly ? d.flights.filter((f) => STRUCTURAL.has(f.step)) : d.flights;
    if (!flights.length) {
        chartwrap.innerHTML = `<div id="msg">no calls recorded${structuralOnly ? " (try turning off “structural only”)" : ""}</div>`;
        return;
    }
    const laneOf = (f) => (mode === "step" ? f.step : f.node || "—");
    const lanes = [...new Set(flights.map(laneOf))].sort();
    // In zone mode the lane order is the walk's own order, which reads far more
    // naturally than alphabetical: you can see the walk descend.
    if (mode === "zone") {
        const firstSeen = new Map();
        for (const f of flights) if (!firstSeen.has(laneOf(f))) firstSeen.set(laneOf(f), f.start);
        lanes.sort((a, b) => firstSeen.get(a) - firstSeen.get(b));
    }

    const padL = mode === "zone" ? 300 : 190;
    const padR = 18;
    const padT = 26;
    const rowH = mode === "zone" ? 15 : 22;
    const width = Math.max(chartwrap.clientWidth - 30, 620);
    const plotW = width - padL - padR;
    const height = padT + lanes.length * rowH + 26;
    const span = Math.max(d.wall_s, 1);
    const x = (s) => padL + (s / span) * plotW;

    const parts = [];
    parts.push(`<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);

    // Idle bands first, so bars draw over them.
    for (const g of d.idle) {
        const w = Math.max(x(g.end) - x(g.start), 1);
        parts.push(`<rect class="idle" x="${x(g.start)}" y="${padT - 6}" width="${w}" height="${lanes.length * rowH + 6}"/>`);
        if (w > 34) {
            parts.push(
                `<text class="idlelab" x="${x(g.start) + w / 2}" y="${padT - 10}" text-anchor="middle">idle ${fmt(g.end - g.start)}</text>`,
            );
        }
    }

    // Time axis.
    const ticks = 8;
    for (let i = 0; i <= ticks; i++) {
        const t = (span * i) / ticks;
        parts.push(`<line class="grid" x1="${x(t)}" y1="${padT - 6}" x2="${x(t)}" y2="${padT + lanes.length * rowH}"/>`);
        parts.push(`<text x="${x(t)}" y="${padT + lanes.length * rowH + 14}" text-anchor="middle">${fmt(t)}</text>`);
    }

    lanes.forEach((lane, i) => {
        const y = padT + i * rowH;
        const label = lane.length > (mode === "zone" ? 42 : 26) ? `…${lane.slice(-(mode === "zone" ? 41 : 25))}` : lane;
        parts.push(`<text class="lane" x="${padL - 8}" y="${y + rowH / 2 + 3}" text-anchor="end">${label}</text>`);
    });

    const laneIdx = new Map(lanes.map((l, i) => [l, i]));
    for (const f of flights) {
        const y = padT + laneIdx.get(laneOf(f)) * rowH;
        const bw = Math.max(x(f.end) - x(f.start), 1.5);
        const color = f.ok === false ? "#ff8080" : (STEP_COLOR[f.step] ?? SECONDARY_COLOR);
        const meta = [
            f.step,
            f.node ?? "",
            `${fmt(f.end - f.start)}`,
            `start ${fmt(f.start)}`,
            f.tokens_in ? `in ${f.tokens_in.toLocaleString()}` : "",
            f.tokens_out ? `out ${f.tokens_out.toLocaleString()}` : "",
        ].filter(Boolean).join("\n");
        parts.push(
            `<rect class="bar" x="${x(f.start)}" y="${y + 2}" width="${bw}" height="${rowH - 5}" rx="1.5" fill="${color}" data-tip="${meta.replace(/"/g, "&quot;")}"/>`,
        );
    }
    parts.push("</svg>");
    chartwrap.innerHTML = parts.join("");

    const svg = chartwrap.querySelector("svg");
    svg.addEventListener("mousemove", (e) => {
        const t = e.target.getAttribute?.("data-tip");
        if (!t) return void (tip.style.display = "none");
        tip.textContent = t;
        tip.style.display = "block";
        tip.style.left = `${Math.min(e.clientX + 12, window.innerWidth - 260)}px`;
        tip.style.top = `${e.clientY + 12}px`;
    });
    svg.addEventListener("mouseleave", () => (tip.style.display = "none"));
}

// --- gates --------------------------------------------------------------------

function zoneState(z) {
    if (z.started) return ["built", "anchor ran"];
    if (z.released) return ["dead", "released, never ran"];
    if (z.gate) return ["running", "gate open"];
    return ["queued", "waiting"];
}

function renderGates(d) {
    const anchors = d.zones.filter((z) => z.gate !== null);
    // Blocked first — that is what you opened this page to read.
    anchors.sort((a, b) => (a.started ? 1 : 0) - (b.started ? 1 : 0) || (a.gate ? 1 : 0) - (b.gate ? 1 : 0));
    gatelist.innerHTML = "";
    for (const z of anchors) {
        const [cls, label] = zoneState(z);
        const el = document.createElement("div");
        el.className = "zone";
        const blockers = z.frontier.filter((f) => f.counts && !f.settled).length;
        const ignored = z.frontier.filter((f) => !f.counts).length;
        el.innerHTML =
            `<div class="head">` +
            `<span class="zid" title="${z.id}">${z.id}</span>` +
            (blockers ? `<span class="chip queued">${blockers} blocking</span>` : "") +
            (ignored ? `<span class="chip">${ignored} far</span>` : "") +
            `<span class="chip ${cls}">${label}</span>` +
            `</div><div class="body"></div>`;
        const body = el.querySelector(".body");
        if (!z.frontier.length) {
            body.innerHTML = `<div class="fr"><span class="why">nothing borders this zone</span></div>`;
        }
        for (const f of z.frontier) {
            const row = document.createElement("div");
            row.className = `fr ${!f.counts ? "ignored" : f.settled ? "ok" : "block"}`;
            row.innerHTML =
                `<div class="l1">` +
                `<span class="gap">${f.gap_m.toFixed(2)} m</span>` +
                `<span class="rid" title="${f.id}">${f.id}</span>` +
                `</div><div class="why">${f.why}</div>`;
            body.append(row);
        }
        if (z.released) {
            const r = document.createElement("div");
            r.className = "fr";
            r.innerHTML = `<span class="why">released at event #${z.released.index} · ${z.released.forced ? "FORCED at the drain" : "mid-walk"} · scene was ${z.released.scene_nodes} nodes · ${z.released.still_waiting} still queued behind it</span>`;
            body.append(r);
        }
        el.querySelector(".head").addEventListener("click", () => el.classList.toggle("open"));
        if (!z.started && !z.gate) el.classList.add("open"); // blocked zones open by default
        gatelist.append(el);
    }
    if (!anchors.length) gatelist.innerHTML = `<p class="hint">no zone has reached an anchor pass yet.</p>`;
}

function render(d) {
    if (!d) {
        statsEl.innerHTML = "";
        gatelist.innerHTML = "";
        chartwrap.innerHTML = `<div id="msg">nothing to show</div>`;
        return;
    }
    renderStats(d);
    renderChart(d);
    renderGates(d);
}

// --- wiring -------------------------------------------------------------------

runSel.addEventListener("change", loadCells);
cellSel.addEventListener("change", loadCell);
$("reload").addEventListener("click", loadCell);
$("mode-step").addEventListener("click", () => {
    mode = "step";
    $("mode-step").classList.add("on");
    $("mode-zone").classList.remove("on");
    if (data) renderChart(data);
});
$("mode-zone").addEventListener("click", () => {
    mode = "zone";
    $("mode-zone").classList.add("on");
    $("mode-step").classList.remove("on");
    if (data) renderChart(data);
});
$("structural").addEventListener("click", () => {
    structuralOnly = !structuralOnly;
    $("structural").classList.toggle("on", structuralOnly);
    if (data) renderChart(data);
});
$("auto").addEventListener("click", () => {
    if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
    } else {
        autoTimer = setInterval(loadCell, 5000);
    }
    $("auto").classList.toggle("on", !!autoTimer);
});
window.addEventListener("resize", () => {
    if (data) renderChart(data);
});

loadRuns().catch((e) => {
    chartwrap.innerHTML = `<div id="msg" class="err">${e.message}</div>`;
});
