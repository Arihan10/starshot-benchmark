// Side-by-side 3D compare: the cell's live run (PREVIOUS) against its
// simulation branch (CURRENT — the prompt-lab edit run downstream), with the
// two cameras optionally locked together and a text panel showing the input
// diff + both outputs. Launched from a review card's "compare 3D" button.

import { api } from "./api.js";
import { state, on } from "./state.js";
import { el, toast, diffPre, fmtJson } from "./ui.js";
import { createViewer } from "./scene3d.js";
import { applySceneProjection } from "./events.js";

const root = document.getElementById("compare");
const titleEl = document.getElementById("compare-title");
const subEl = document.getElementById("compare-sub");
const textEl = document.getElementById("compare-text");

let prevViewer = null;
let curViewer = null;
let linked = true;
let ready = false; // suppress camera sync until both scenes have settled
let syncing = false; // re-entrancy guard for the A<->B copy
let openSeq = 0;

function ensureViewers() {
  if (prevViewer) return;
  prevViewer = createViewer(document.getElementById("cmp-prev-host"));
  curViewer = createViewer(document.getElementById("cmp-cur-host"));
  const link = (a, b) =>
    a.onCameraChange(() => {
      if (!ready || !linked || syncing) return;
      syncing = true;
      b.setView(a.getView());
      syncing = false;
    });
  link(prevViewer, curViewer);
  link(curViewer, prevViewer);
}

async function paint(viewer, slot, model, opts, seq) {
  try {
    const proj = await api.scene(state.run, slot, model, opts);
    if (seq !== openSeq) return;
    applySceneProjection(viewer, proj);
    viewer.prefetchBundle(api.meshesUrl(state.run, slot, model, opts));
  } catch (e) {
    if (seq === openSeq) toast(`compare scene load failed: ${e.message}`, "err");
  }
}

async function loadText(slot, model, step, node, index, seq) {
  textEl.replaceChildren(el("div", { class: "muted", text: "loading…" }));
  const [prevR, curR] = await Promise.allSettled([
    index == null
      ? Promise.reject(new Error("no index"))
      : api.stepEvent(state.run, slot, model, index, step),
    api.branchStepEvent(state.run, slot, model, step, node),
  ]);
  if (seq !== openSeq) return;
  const prev = prevR.status === "fulfilled" ? prevR.value : null;
  const cur = curR.status === "fulfilled" ? curR.value : null;
  if (!cur) {
    textEl.replaceChildren(el("div", { class: "muted", text: "no simulated-edit branch call for this cell yet." }));
    return;
  }
  const frag = document.createDocumentFragment();
  frag.appendChild(el("div", { class: "ct-sec" }, [
    el("div", { class: "ct-h", text: "input · diff (previous → current)" }),
    diffPre((prev && prev.user) || "", cur.user || ""),
  ]));
  frag.appendChild(el("div", { class: "ct-grid" }, [
    el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: "previous output" }),
      el("pre", { class: "fit-full", text: prev ? fmtJson(prev.output) : "(unavailable)" }),
    ]),
    el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: "current output" }),
      el("pre", { class: "fit-full", text: fmtJson(cur.output) }),
    ]),
  ]));
  textEl.replaceChildren(frag);
}

async function openCompare({ slot, model, step, node, index }) {
  ensureViewers();
  const seq = ++openSeq;
  ready = false;
  titleEl.textContent = `${slot} · ${model}`;
  subEl.textContent = step ? `step: ${step}` : "";
  root.classList.add("open");
  textEl.classList.add("open"); // surface the input diff alongside the 3D
  prevViewer.setActive(true);
  curViewer.setActive(true);
  prevViewer.clear();
  curViewer.clear();
  await Promise.all([
    paint(prevViewer, slot, model, {}, seq),
    paint(curViewer, slot, model, { branch: true }, seq),
  ]);
  if (seq !== openSeq) return;
  // Settle the framing once, then align previous to current and start syncing.
  setTimeout(() => {
    if (seq !== openSeq) return;
    curViewer.fit();
    prevViewer.setView(curViewer.getView());
    ready = true;
  }, 700);
  loadText(slot, model, step, node, index, seq);
}

function closeCompare() {
  openSeq += 1;
  ready = false;
  root.classList.remove("open");
  prevViewer?.setActive(false);
  curViewer?.setActive(false);
  prevViewer?.clear();
  curViewer?.clear();
}

export function initCompare() {
  document.getElementById("compare-close").addEventListener("click", closeCompare);
  document.getElementById("compare-sync").addEventListener("click", () => {
    if (prevViewer && curViewer) prevViewer.setView(curViewer.getView());
  });
  const linkCb = document.getElementById("compare-link");
  linked = linkCb.checked;
  linkCb.addEventListener("change", () => { linked = linkCb.checked; });
  document.getElementById("compare-text-toggle").addEventListener("click", () => {
    textEl.classList.toggle("open");
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && root.classList.contains("open") && !document.getElementById("modal-root").firstChild) {
      closeCompare();
    }
  });
  on("open-compare", openCompare);
}
