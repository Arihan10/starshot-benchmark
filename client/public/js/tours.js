// The tours page: every cell that has a PLANNED or CAPTURED matterport
// walkthrough, with a per-scene publish button.
//
// Capture itself lives on the splat page (the cell's "capture tour" control) —
// this page is the other half: it indexes what capture produced across every run
// and lets you publish each one on demand. Publishing bakes the vertex-colored
// dollhouse and, unless the server runs with STARSHOT_LOCAL_PUBLISH, uploads the
// preview / tour / proxy / panos to R2 and upserts the D1 catalog row. That's a
// multi-minute job server-side, so we start it and poll rather than holding a
// request open.

import { api } from "./api.js";
import { el } from "./ui.js";

const POLL_MS = 1500;
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const keyOf = (t) => `${t.run}/${t.slot}/${t.model}`;

// Cells with a publish in flight, so a re-render keeps their button disabled and
// their note visible even though /tours is re-fetched underneath.
const inFlight = new Map(); // key -> {note, cls}

function setMessage(text, isErr) {
    const msg = $("msg");
    msg.textContent = text || "";
    msg.className = `msg${isErr ? " err" : ""}`;
    msg.hidden = !text;
}

function stateTags(t) {
    const tags = [];
    if (t.captured) {
        tags.push(el("span", { class: "tag captured", text: "captured" }));
    } else if (t.planned) {
        // The plan survives a capture, so "planned" is only worth showing on its
        // own — a saved plan still waiting for its render.
        tags.push(
            el("span", {
                class: "tag planned",
                text: `planned ${t.planned}`,
                title: "anchors planned + saved; the capture (render) hasn't run yet",
            }),
        );
    }
    if (!tags.length) tags.push(el("span", { class: "tag no", text: "—" }));
    return tags;
}

const yesNo = (ok) =>
    el("span", { class: `tag ${ok ? "yes" : "no"}`, text: ok ? "yes" : "no" });

function row(t) {
    const key = keyOf(t);
    const live = inFlight.get(key);
    const running = !!live || !!(t.publish && t.publish.running);
    // Only a captured tour is worth publishing — a bare plan has no panos yet.
    const canPublish = t.captured && !running;

    const note = el("span", {
        class: `row-note${live && live.cls ? ` ${live.cls}` : ""}`,
        text: live ? live.note : t.publish && t.publish.error ? t.publish.error : "",
    });

    const btn = el("button", {
        class: canPublish ? "primary" : "",
        text: running ? "publishing…" : t.published ? "re-publish" : "publish",
        title: t.captured
            ? "bake the dollhouse and publish this scene"
            : "capture the tour first (splat page → matterport → capture tour)",
        disabled: !canPublish,
        onclick: () => void publish(t, btn, note),
    });

    const actions = el("div", { class: "row-actions" }, note);
    if (t.tour_url) {
        actions.append(
            el("a", {
                href: api.absUrl(t.tour_url),
                target: "_blank",
                rel: "noreferrer",
                text: "tour.json",
                style: "color:var(--accent);text-decoration:none;font-size:10px",
            }),
        );
    }
    actions.append(btn);

    return el(
        "tr",
        {},
        el(
            "td",
            {},
            el("div", { class: "cell-run", text: `${t.slot} · ${t.model}` }),
            el("div", { class: "cell-sub", text: t.run }),
        ),
        el("td", {}, ...stateTags(t)),
        el("td", { class: "num", text: String(t.panos ?? 0) }),
        el("td", { class: "num", text: String(t.minimaps ?? 0) }),
        el("td", {}, yesNo(!!t.proxy)),
        el(
            "td",
            {},
            yesNo(!!t.published),
            t.published_at
                ? el("div", { class: "cell-sub", text: t.published_at.replace("T", " ") })
                : null,
        ),
        el("td", { class: "cell-sub", text: (t.updated_at || "").replace("T", " ") }),
        el("td", {}, actions),
    );
}

function render(payload) {
    const tours = payload.tours || [];
    const target = $("target");
    target.textContent = payload.local ? "local disk" : "cloudflare r2 + d1";
    target.className = `pill ${payload.local ? "local" : "r2"}`;

    const body = $("rows");
    body.replaceChildren(...tours.map(row));
    $("tbl").hidden = tours.length === 0;
    setMessage(
        tours.length
            ? ""
            : "no planned or captured tours yet — run “capture tour” on a cell from the splat page",
    );
}

let loading = false;

async function load() {
    if (loading) return;
    loading = true;
    try {
        render(await api.tours());
    } catch (e) {
        setMessage(`failed to load tours: ${e.message}`, true);
    } finally {
        loading = false;
    }
}

// Start the publish job and poll it to completion, then refresh the table so the
// published column + timestamp reflect what actually landed on disk / in R2.
async function publish(t, btn, note) {
    const key = keyOf(t);
    if (inFlight.has(key)) return;
    inFlight.set(key, { note: "starting…", cls: "" });
    btn.disabled = true;
    btn.textContent = "publishing…";
    note.textContent = "starting…";
    note.className = "row-note";
    try {
        let job = await api.tourPublish(t.run, t.slot, t.model);
        while (job && job.running) {
            await sleep(POLL_MS);
            try {
                job = await api.tourPublishStatus(t.run, t.slot, t.model);
            } catch {
                continue; // transient — keep polling
            }
            inFlight.set(key, { note: job.status || "publishing…", cls: "" });
            note.textContent = job.status || "publishing…";
        }
        if (!job || job.status === "error") {
            throw new Error((job && job.error) || "publish failed");
        }
        const panos = job.result && job.result.pano_count;
        inFlight.set(key, {
            note: `published${panos ? ` · ${panos} panos` : ""}`,
            cls: "ok",
        });
    } catch (e) {
        inFlight.set(key, { note: e.message, cls: "err" });
    } finally {
        // Keep the outcome note for one refresh cycle, then let the table own it.
        const outcome = inFlight.get(key);
        inFlight.delete(key);
        await load();
        if (outcome) {
            const cell = [...$("rows").querySelectorAll("tr")].find((tr) =>
                tr.textContent.includes(`${t.slot} · ${t.model}`),
            );
            const n = cell && cell.querySelector(".row-note");
            if (n) {
                n.textContent = outcome.note;
                n.className = `row-note ${outcome.cls}`;
            }
        }
    }
}

$("refresh").addEventListener("click", () => void load());
void load();
