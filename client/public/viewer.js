import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";

const SERVER_URL = document
  .querySelector('meta[name="server-url"]')
  .getAttribute("content");

const SLOT_STORAGE_KEY = "starshot.selectedSlot";
const BBOX_VISIBLE_STORAGE_KEY = "starshot.bboxesVisible";
const FRAMES_VISIBLE_STORAGE_KEY = "starshot.framesVisible";
const PICK_INNER_STORAGE_KEY = "starshot.pickInner";
const SOLID_FILL_STORAGE_KEY = "starshot.solidFill";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const slotsEl = document.getElementById("slots");
const resetEl = document.getElementById("slot-reset");
const resumeEl = document.getElementById("slot-resume");
const bboxToggleEl = document.getElementById("bbox-toggle");
const framesToggleEl = document.getElementById("frames-toggle");
const pickInnerToggleEl = document.getElementById("pick-inner-toggle");
const solidFillToggleEl = document.getElementById("solid-fill-toggle");
const exportGlbEl = document.getElementById("export-glb");
const replayGifEl = document.getElementById("replay-gif");
const replayModalEl = document.getElementById("replay-modal");
const replayCloseEl = document.getElementById("replay-close");
const replayEventCountEl = document.getElementById("replay-event-count");
const replayDurationEstEl = document.getElementById("replay-duration-est");
const replaySpeedEl = document.getElementById("replay-speed");
const replaySpeedValEl = document.getElementById("replay-speed-val");
const replayFpsEl = document.getElementById("replay-fps");
const replayFpsValEl = document.getElementById("replay-fps-val");
const replayProgressBarEl = document.getElementById("replay-progress-bar");
const replayStatusEl = document.getElementById("replay-status");
const replayPreviewEl = document.getElementById("replay-preview");
const replayRenderEl = document.getElementById("replay-render");
const replayDownloadEl = document.getElementById("replay-download");
const replayStageEl = document.getElementById("replay-stage");
const replayPreviewCanvasEl = document.getElementById("replay-preview-canvas");
const replayResultImgEl = document.getElementById("replay-result-img");
const replayPreviewCtx = replayPreviewCanvasEl.getContext("2d");

// URL of the last-encoded gif blob, kept so we can revoke when the modal
// closes or a fresh render starts.
let lastGifUrl = null;
let lastGifBlob = null;

function showReplayPlaceholder() {
  replayStageEl.classList.remove("show-result");
  replayStageEl.classList.add("empty");
  replayPreviewCtx.clearRect(0, 0, replayPreviewCanvasEl.width, replayPreviewCanvasEl.height);
}

function drawReplayFrame(srcCanvas) {
  // Match the preview canvas's backing buffer to the source so frames keep
  // their native aspect; CSS object-fit handles the visual letterboxing.
  if (
    replayPreviewCanvasEl.width !== srcCanvas.width ||
    replayPreviewCanvasEl.height !== srcCanvas.height
  ) {
    replayPreviewCanvasEl.width = srcCanvas.width;
    replayPreviewCanvasEl.height = srcCanvas.height;
  }
  replayPreviewCtx.drawImage(srcCanvas, 0, 0);
  replayStageEl.classList.remove("empty");
  replayStageEl.classList.remove("show-result");
}

function showReplayGifResult(blob) {
  if (lastGifUrl) URL.revokeObjectURL(lastGifUrl);
  lastGifBlob = blob;
  lastGifUrl = URL.createObjectURL(blob);
  replayResultImgEl.src = lastGifUrl;
  replayStageEl.classList.add("show-result");
  replayStageEl.classList.remove("empty");
  replayDownloadEl.disabled = false;
}
const assetsEl = document.getElementById("assets");
const assetsBodyEl = document.getElementById("assets-body");
const assetsCountEl = document.getElementById("assets-count");
const assetsHeaderEl = document.getElementById("assets-header");
const assetsToggleEl = document.getElementById("assets-toggle");
const trellisQueueEl = document.getElementById("trellis-queue");
const trellisQueueHeaderEl = document.getElementById("trellis-queue-header");
const trellisQueueToggleEl = document.getElementById("trellis-queue-toggle");
const trellisQueueCountsEl = document.getElementById("trellis-queue-counts");
const tqProcessingEl = document.getElementById("tq-processing");
const tqProcessingCapEl = document.getElementById("tq-processing-cap");
const tqWaitingEl = document.getElementById("tq-waiting");
const tqWaitingCapEl = document.getElementById("tq-waiting-cap");
// Server-side `GENERATE_CONCURRENCY` (threed.py). Hard-coded mirror so the
// "X/10" cap reads correctly; bump alongside the server constant if it
// changes.
const TRELLIS_CONCURRENCY_CAP = 10;
const treeEl = document.getElementById("tree");
const treeBodyEl = document.getElementById("tree-body");
const treeDetailEl = document.getElementById("tree-detail");
const treeHeaderEl = document.getElementById("tree-header");
const treeToggleEl = document.getElementById("tree-toggle");
const treeActiveEl = document.getElementById("tree-active");
const treeExpandEl = document.getElementById("tree-expand");
const treeSearchEl = document.getElementById("tree-search");
const treeSearchClearEl = document.getElementById("tree-search-clear");
const treeSearchCountEl = document.getElementById("tree-search-count");
const treeModalEl = document.getElementById("tree-modal");
const treeModalBodyEl = document.getElementById("tree-modal-body");
const treeModalCloseEl = document.getElementById("tree-modal-close");
const treeModalSearchEl = document.getElementById("tree-modal-search");
const treeModalImageToggleEl = document.getElementById("tree-modal-image-toggle");

// --- log panel --------------------------------------------------------------

const KIND_COLOR = {
  "run.start": "#9ad4ff",
  "run.done": "#8bd17c",
  "run.error": "#ff8080",
  "run.paused": "#e09050",
  "bbox": "#e0c271",
  "image": "#f6a96a",
  "model": "#c586d1",
  "step": "#4a8fd8",
  "divider.decompose": "#e0c271",
  "generation.decompose": "#c586d1",
  "mesh.error": "#ff8080",
};

function setStatus(text, cls = "hdr") {
  statusEl.innerHTML = "";
  const p = document.createElement("p");
  p.className = `line ${cls}`;
  p.textContent = text;
  statusEl.appendChild(p);
}

function fmtValue(v) {
  if (Array.isArray(v)) return "[" + v.map(fmtValue).join(", ") + "]";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (v && typeof v === "object")
    return "{" + Object.entries(v).map(([k, x]) => `${k}=${fmtValue(x)}`).join(", ") + "}";
  if (typeof v === "string") return v;
  return String(v);
}

// Fields whose values are LLM thinking / chain-of-thought traces. Stripped at
// the display layer so the log panel stays focused on pipeline signal; the
// underlying events.jsonl still contains them for debugging.
const HIDDEN_LOG_FIELDS = new Set(["reasoning", "thinking"]);

function appendEvent(event) {
  const { kind, index, ...rest } = event;
  const fields = Object.fromEntries(
    Object.entries(rest).filter(([k]) => !HIDDEN_LOG_FIELDS.has(k)),
  );
  const p = document.createElement("p");
  p.className = "line";
  if (typeof index === "number") p.dataset.eventIndex = String(index);

  if (typeof index === "number") {
    const btn = document.createElement("button");
    btn.className = "rewind";
    btn.type = "button";
    btn.textContent = "↶ rewind";
    btn.title = `Rewind to event ${index} (discards this event and everything after)`;
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      rewindTo(index);
    });
    p.appendChild(btn);
  }

  if (typeof index === "number") {
    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = `#${index}`;
    p.appendChild(idx);
  }

  const tag = document.createElement("span");
  tag.className = "step";
  tag.textContent = `[${kind}]`;
  tag.style.color = KIND_COLOR[kind] ?? "#8bd17c";
  p.appendChild(tag);

  const entries = Object.entries(fields);
  if (entries.length === 0) {
    p.appendChild(document.createTextNode(""));
  } else {
    for (const [k, v] of entries) {
      const kv = document.createElement("span");
      kv.className = "kv";
      const label = document.createElement("span");
      label.className = "k";
      label.textContent = ` ${k}=`;
      kv.appendChild(label);
      kv.appendChild(document.createTextNode(fmtValue(v)));
      p.appendChild(kv);
    }
  }
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  logEl.innerHTML = "";
}

// id -> error message for every mesh that errored during the current run.
// Drives the per-node "error" phase in the tree and the aggregated count
// shown on run.done so silent partial failures don't slip past as success.
const meshErrors = new Map();

function clearMeshErrors() {
  meshErrors.clear();
  retryingIds.clear();
  runFinished = false;
}

// ids that the user clicked "retry" on whose follow-up image/model/mesh.error
// hasn't landed yet. Drives button disabled state + label so the user can't
// double-fire a retry mid-flight. Cleared on slot switch / reset / rewind
// (same lifecycle as meshErrors). The set is also pruned when a retry-targeted
// event arrives (mesh.error → retry available again; image/model → success).
const retryingIds = new Set();

// Flips to true on the live `run.done` for this slot. Lets post-run mesh
// updates (retry success / retry failure) refresh the top status line so a
// stale "run complete — N meshes failed" doesn't outlive the N it counted.
// Reset on slot switch / reset / rewind / resume alongside the rest of the
// per-slot state.
let runFinished = false;

function refreshPostRunStatus() {
  if (!runFinished) return;
  const inFlight = retryingIds.size;
  if (inFlight > 0) {
    const ids = [...retryingIds];
    const head = ids.slice(0, 3).join(", ");
    const suffix = ids.length > 3 ? `, +${ids.length - 3}` : "";
    setStatus(`retrying ${inFlight} mesh${inFlight === 1 ? "" : "es"}: ${head}${suffix}`);
    return;
  }
  if (meshErrors.size > 0) showRunCompleteWithErrors();
  else setStatus("run complete");
}

async function retryMesh(id) {
  if (currentSlotId === null) return;
  if (retryingIds.has(id)) return;
  retryingIds.add(id);
  // Local optimistic state: clear the error so the asset/tree both reflect
  // an in-flight retry. The server-side mesh.retry event will arrive too,
  // but doing it locally first removes the visible-flicker between click
  // and the SSE round-trip.
  meshErrors.delete(id);
  upsertAsset(id, { status: "pending", errorMessage: null });
  if (treeNodes.has(id)) treeSetPhase(id, "generating_mesh");

  // The slot's SSE may have closed itself on the prior run's run.done.
  // Re-subscribe (without resetting highestEventIndex) so the snapshot
  // is deduped and the retry's new events flow through the live queue.
  if (currentSource === null) {
    subscribe(slotEventsUrl(currentSlotId));
  }

  try {
    const res = await fetch(
      new URL(
        `/slots/${encodeURIComponent(currentSlotId)}/retry-mesh/${encodeURIComponent(id)}`,
        SERVER_URL,
      ),
      { method: "POST" },
    );
    if (!res.ok) {
      const detail = await res.text();
      const msg = `retry failed: HTTP ${res.status} ${detail}`;
      retryingIds.delete(id);
      meshErrors.set(id, msg);
      treeSetPhase(id, "error");
      upsertAsset(id, { status: "error", errorMessage: msg });
    }
  } catch (e) {
    retryingIds.delete(id);
    const msg = `retry failed: ${e.message}`;
    meshErrors.set(id, msg);
    treeSetPhase(id, "error");
    upsertAsset(id, { status: "error", errorMessage: msg });
  }
}

// Render-or-update a retry button inside `container`. `status` is the asset
// status; the button is shown only for `error` (retry) or while a prior retry
// is in flight (greyed out so the user can't double-click). `cls` lets the
// caller scope the CSS (`asset-retry` vs `detail-retry`).
function syncRetryButton(container, id, status, cls) {
  let btn = container.querySelector(`.${cls}`);
  const retrying = retryingIds.has(id);
  const visible = status === "error" || retrying;
  if (!visible) {
    if (btn) btn.remove();
    return;
  }
  if (!btn) {
    btn = document.createElement("button");
    btn.type = "button";
    btn.className = cls;
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      retryMesh(id);
    });
    container.appendChild(btn);
  }
  btn.classList.toggle("retrying", retrying);
  btn.disabled = retrying;
  btn.textContent = retrying ? "retrying…" : "retry mesh";
  btn.title = retrying
    ? "Re-running banana + Trellis for this mesh"
    : "Re-run banana + Trellis for this mesh (fresh API calls)";
}

function showRunCompleteWithErrors() {
  statusEl.innerHTML = "";
  const head = document.createElement("p");
  head.className = "line warn";
  const n = meshErrors.size;
  head.textContent = `run complete — ${n} mesh${n === 1 ? "" : "es"} failed`;
  statusEl.appendChild(head);
  const ids = [...meshErrors.keys()];
  const shown = ids.slice(0, 6);
  const detail = document.createElement("p");
  detail.className = "line warn";
  const suffix = ids.length > shown.length ? `, +${ids.length - shown.length} more` : "";
  detail.textContent = shown.join(", ") + suffix;
  detail.title = ids.map((id) => `${id}: ${meshErrors.get(id)}`).join("\n");
  statusEl.appendChild(detail);
}

// --- asset browser ----------------------------------------------------------

// id -> { imageUrl, prompt, modelUrl, status: "pending" | "loaded" | "error", errorMessage }
const assets = new Map();

function assetStatus(a) {
  return a.status ?? "pending";
}

function upsertAsset(id, patch) {
  const cur = assets.get(id) ?? { imageUrl: null, prompt: null, modelUrl: null, status: "pending" };
  assets.set(id, { ...cur, ...patch });
  renderAsset(id);
  assetsCountEl.textContent = `(${assets.size})`;
  if (id === selectedBboxId) renderTreeDetail();
}

function renderAsset(id) {
  const a = assets.get(id);
  if (!a) return;
  let card = assetsBodyEl.querySelector(`[data-id="${CSS.escape(id)}"]`);
  if (!card) {
    card = document.createElement("div");
    card.className = "asset-card";
    card.dataset.id = id;
    card.innerHTML = `
      <a class="asset-thumb-link" target="_blank" rel="noopener">
        <div class="asset-thumb placeholder">no image</div>
      </a>
      <div class="asset-body">
        <div class="asset-id"></div>
        <div class="asset-status pending">pending</div>
        <div class="asset-prompt"></div>
      </div>
    `;
    assetsBodyEl.appendChild(card);
    const promptEl = card.querySelector(".asset-prompt");
    promptEl.addEventListener("click", () => {
      promptEl.classList.toggle("expanded");
    });
  }
  card.querySelector(".asset-id").textContent = id;

  const status = assetStatus(a);
  card.className = `asset-card status-${status}`;
  const statusTag = card.querySelector(".asset-status");
  statusTag.className = `asset-status ${status}`;
  statusTag.textContent = status === "error" && a.errorMessage
    ? `error: ${a.errorMessage}`
    : status;
  syncRetryButton(card.querySelector(".asset-body"), id, status, "asset-retry");

  const link = card.querySelector(".asset-thumb-link");
  const thumb = card.querySelector(".asset-thumb");
  if (a.imageUrl) {
    const absImg = new URL(a.imageUrl, SERVER_URL).toString();
    link.href = absImg;
    if (thumb.tagName !== "IMG") {
      const img = document.createElement("img");
      img.className = "asset-thumb";
      img.loading = "lazy";
      img.alt = id;
      img.src = absImg;
      thumb.replaceWith(img);
    } else if (thumb.src !== absImg) {
      thumb.src = absImg;
    }
  }

  const promptEl = card.querySelector(".asset-prompt");
  promptEl.textContent = a.prompt ?? "";
}

function clearAssets() {
  assets.clear();
  assetsBodyEl.innerHTML = "";
  assetsCountEl.textContent = "(0)";
}

assetsHeaderEl.addEventListener("click", () => {
  const collapsed = assetsEl.classList.toggle("collapsed");
  assetsToggleEl.textContent = collapsed ? "▸" : "▾";
});

// --- trellis queue panel ----------------------------------------------------
//
// Authoritative state comes from the server via GET /trellis/queue. We do
// NOT derive queue state from streamed events: SSE replays the full event
// log on every subscribe, and any historical `trellis.submit` whose run
// was killed before logging `trellis.done` would leak as a stale
// "processing" row forever. Polling the live snapshot sidesteps that
// entirely — a server restart resets the queue to empty (correct
// behaviour), and a single poll trumps any amount of historical noise.
//
// The semaphore is process-global across all 7 benchmark slots, so the
// "cap" we display is the global concurrency. Rows from slots other than
// the one currently being viewed are shown with a slot tag and aren't
// clickable (their node ids aren't in this slot's tree).

// Latest snapshot from /trellis/queue: { cap, entries: [{slot_id, job_id, state, since, task_id?}, ...] }
// `since` is the server's epoch-seconds timestamp from when the job entered
// the queue (set on the server side, persists across client reconnects).
// Render uses it directly so the elapsed timer reflects true wall-clock
// age, not "time since this browser first noticed the row".
let trellisQueueSnapshot = { cap: 10, entries: [] };

function fmtElapsed(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

async function pollTrellisQueue() {
  try {
    const resp = await fetch(`${SERVER_URL}/trellis/queue`, { cache: "no-store" });
    if (!resp.ok) return;
    trellisQueueSnapshot = await resp.json();
    renderTrellisQueue();
  } catch {
    // Server transiently down / network blip — keep the last good snapshot
    // on screen. Next tick will refresh.
  }
}

function renderTrellisQueue() {
  const now = Date.now();
  const cap = trellisQueueSnapshot.cap ?? TRELLIS_CONCURRENCY_CAP;
  const entries = trellisQueueSnapshot.entries ?? [];
  const processing = [];
  const waiting = [];
  for (const e of entries) {
    // Server `since` is epoch seconds (time.time() at the moment the job
    // was admitted to the queue). Convert to ms once so each row carries
    // a value comparable to Date.now().
    const sinceMs = (e.since ?? 0) * 1000;
    const row = { ...e, sinceMs };
    if (e.state === "processing") processing.push(row);
    else waiting.push(row);
  }
  processing.sort((a, b) => a.sinceMs - b.sinceMs);
  waiting.sort((a, b) => a.sinceMs - b.sinceMs);

  trellisQueueCountsEl.textContent =
    `${processing.length} processing · ${waiting.length} waiting`;
  tqProcessingCapEl.textContent = `(${processing.length}/${cap})`;
  tqWaitingCapEl.textContent = `(${waiting.length})`;

  function renderList(parent, rows, kind) {
    parent.textContent = "";
    if (rows.length === 0) {
      const empty = document.createElement("div");
      empty.className = "tq-empty";
      empty.textContent = kind === "processing" ? "(idle)" : "(none)";
      parent.appendChild(empty);
      return;
    }
    for (const row of rows) {
      const el = document.createElement("div");
      el.className = `tq-row ${kind}`;
      const inThisSlot = row.slot_id === currentSlotId;
      if (inThisSlot) {
        el.addEventListener("click", () => selectTreeNode(row.job_id));
      } else {
        el.classList.add("other-slot");
        el.title = `from slot ${row.slot_id} — switch slots to inspect`;
      }
      const dot = document.createElement("span");
      dot.className = "tq-dot";
      el.appendChild(dot);
      const idEl = document.createElement("span");
      idEl.className = "tq-id";
      idEl.textContent = row.job_id;
      el.appendChild(idEl);
      if (!inThisSlot) {
        const slotTag = document.createElement("span");
        slotTag.className = "tq-slot-tag";
        slotTag.textContent = row.slot_id;
        el.appendChild(slotTag);
      }
      const elapsed = document.createElement("span");
      elapsed.className = "tq-elapsed";
      elapsed.textContent = fmtElapsed(Math.max(0, now - row.sinceMs));
      el.appendChild(elapsed);
      parent.appendChild(el);
    }
  }
  renderList(tqProcessingEl, processing, "processing");
  renderList(tqWaitingEl, waiting, "waiting");
}

// Poll every 1.5s. Cheap (an in-memory dict snapshot on the server) and
// keeps the UI within a heartbeat of reality without flooding.
setInterval(pollTrellisQueue, 1500);
// Re-render once a second between polls so elapsed timers tick smoothly.
setInterval(() => {
  if ((trellisQueueSnapshot.entries ?? []).length > 0) renderTrellisQueue();
}, 1000);
// Kick an initial fetch so the panel populates without waiting a full tick.
pollTrellisQueue();

trellisQueueHeaderEl.addEventListener("click", () => {
  const collapsed = trellisQueueEl.classList.toggle("collapsed");
  trellisQueueToggleEl.textContent = collapsed ? "▸" : "▾";
});

renderTrellisQueue();

// --- tree view --------------------------------------------------------------

// Mirror of the server-side recursion. Nodes are upserted by `bbox` (when
// placed) or by `divider.decompose` (announces children before their bboxes
// are resolved, so the tree shows pending placeholders). The `step` event
// drives the per-node phase badge and the global "active" highlight.
//
// Extra fields used by the hover tooltip:
//   plan         — zone plan text from `divider.zone_plan` (zones only)
//   imagePrompt  — noun phrase actually sent to Banana+Trellis, from the
//                  `image` event (objects/frames once they've been generated)
const treeNodes = new Map(); // id -> { id, parentId, prompt, kind, phase, order, plan?, imagePrompt? }
const treeChildren = new Map(); // parentId -> [childIds] in insertion order
let treeRootId = null;
let treeActiveId = null;
let treeOrderCounter = 0;

// Per-node visibility overrides driven from the tree. A node in this set
// hides its own bbox/proxy/mesh; hiding a parent transitively hides every
// descendant in the scene (computed lazily via ancestor walk). State is
// per-run — cleared on slot switch / rewind / reset alongside the rest of
// the tree, and not persisted.
const hiddenIds = new Set();

function treeUpsert(id, patch) {
  const cur = treeNodes.get(id);
  if (!cur) {
    treeNodes.set(id, {
      id, parentId: null, prompt: null, kind: "zone",
      phase: "pending", order: treeOrderCounter++,
      ...patch,
    });
    const parentId = patch.parentId ?? null;
    if (parentId !== null) {
      const arr = treeChildren.get(parentId) ?? [];
      if (!arr.includes(id)) arr.push(id);
      treeChildren.set(parentId, arr);
    } else if (treeRootId === null) {
      treeRootId = id;
    }
    return;
  }
  // Existing node: merge patch, but if parentId changes from null to a real
  // one, wire it into the children index lazily.
  const prevParent = cur.parentId;
  Object.assign(cur, patch);
  if (prevParent === null && cur.parentId) {
    const arr = treeChildren.get(cur.parentId) ?? [];
    if (!arr.includes(id)) arr.push(id);
    treeChildren.set(cur.parentId, arr);
    if (treeRootId === id) treeRootId = null; // was mis-rooted
  }
}

function treeSetPhase(id, phase) {
  const cur = treeNodes.get(id);
  if (!cur) {
    // Step fired before any bbox / decompose — stash the phase so it renders
    // as soon as we have the node.
    treeUpsert(id, { phase });
  } else {
    cur.phase = phase;
  }
  if (phase !== "done") {
    treeActiveId = id;
  } else if (treeActiveId === id) {
    // A node finishing doesn't move the focus elsewhere on its own; leave
    // the highlight on it until the next step event moves it.
  }
  renderTree();
  if (id === selectedBboxId) renderTreeDetail();
  if (treeModalOpen) renderTreeModal();
}

function treeClear() {
  treeNodes.clear();
  treeChildren.clear();
  treeRootId = null;
  treeActiveId = null;
  treeOrderCounter = 0;
  hiddenIds.clear();
  treeBodyEl.innerHTML = "";
  treeActiveEl.textContent = "";
  destroyDetailPreview();
  treeDetailEl.innerHTML = "";
  treeEl.classList.remove("detail-open");
}

// True if `id` itself or any ancestor is in `hiddenIds`. The descendant
// case is what makes hiding a zone hide everything underneath it without
// having to mark each child individually.
function effectivelyHidden(id) {
  let cur = treeNodes.get(id);
  while (cur) {
    if (hiddenIds.has(cur.id)) return true;
    cur = cur.parentId ? treeNodes.get(cur.parentId) : null;
  }
  return false;
}

function toggleNodeHidden(id) {
  if (hiddenIds.has(id)) hiddenIds.delete(id);
  else hiddenIds.add(id);
  // Effective-hidden status changed for this node and (potentially) every
  // descendant, so re-evaluate visibility for the whole subtree.
  refreshSubtreeVisibility(id);
  renderTree();
}

function refreshSubtreeVisibility(rootId) {
  const stack = [rootId];
  while (stack.length) {
    const cur = stack.pop();
    applyModelVisibility(cur);
    applyBboxVisibility(cur);
    applySolidFillVisibility(cur);
    const kids = treeChildren.get(cur) ?? [];
    for (const k of kids) stack.push(k);
  }
}

function truncate(s, n = 60) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function renderTreeNode(id, ctx) {
  const node = treeNodes.get(id);
  if (!node) return null;
  if (ctx && !ctx.visible.has(id)) return null;
  const wrap = document.createElement("div");
  const classes = ["tree-node"];
  if (id === treeActiveId) classes.push("active");
  if (id === selectedBboxId) classes.push("selected");
  if (ctx && ctx.matches.has(id)) classes.push("matched", "match-highlight");
  if (effectivelyHidden(id)) classes.push("vis-hidden");
  wrap.className = classes.join(" ");
  wrap.dataset.id = id;

  const row = document.createElement("div");
  row.className = "tree-row";
  // Click the row (not a nested child-tree row) to select this node.
  row.addEventListener("click", (ev) => {
    ev.stopPropagation();
    selectTreeNode(id);
  });

  // Per-node visibility toggle. Reflects only the self-hidden state — a
  // descendant of a hidden ancestor still shows ● because its own bit is
  // off; the row's `vis-hidden` class communicates effective state.
  const visBtn = document.createElement("button");
  visBtn.type = "button";
  const selfHidden = hiddenIds.has(id);
  visBtn.className = `tree-vis-toggle${selfHidden ? " hidden" : ""}`;
  visBtn.textContent = selfHidden ? "○" : "●";
  visBtn.title = selfHidden ? "Show node" : "Hide node";
  visBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    toggleNodeHidden(id);
  });
  row.appendChild(visBtn);

  const idEl = document.createElement("span");
  idEl.className = `tree-id ${node.kind}`;
  idEl.textContent = node.id;
  row.appendChild(idEl);

  const promptEl = document.createElement("span");
  promptEl.className = "tree-prompt";
  promptEl.textContent = truncate(node.prompt, 80);
  promptEl.title = node.prompt ?? "";
  row.appendChild(promptEl);

  const phaseEl = document.createElement("span");
  phaseEl.className = `tree-phase phase-${node.phase}`;
  phaseEl.textContent = node.phase;
  row.appendChild(phaseEl);

  wrap.appendChild(row);

  const childIds = treeChildren.get(id) ?? [];
  if (childIds.length > 0) {
    const kidsEl = document.createElement("div");
    kidsEl.className = "tree-children";
    for (const cid of childIds) {
      const cEl = renderTreeNode(cid, ctx);
      if (cEl) kidsEl.appendChild(cEl);
    }
    if (kidsEl.childNodes.length > 0) wrap.appendChild(kidsEl);
  }
  return wrap;
}

let treeSearchQuery = "";
// Index of the currently-focused match within `orderedMatches`. Repeated
// Enter advances; Shift+Enter goes back. -1 = nothing selected yet for
// this query (next Enter selects [0]).
let treeMatchIndex = -1;

// A node is a `match` if its id or prompt contains the query.
// `visible` = matches ∪ ancestors-of-matches ∪ descendants-of-matches.
// Returns null when no query (everything visible, no match highlighting).
function computeTreeFilter() {
  const q = treeSearchQuery.trim().toLowerCase();
  if (!q) return null;
  const matches = new Set();
  for (const [id, node] of treeNodes) {
    const idHit = id.toLowerCase().includes(q);
    const promptHit = (node.prompt ?? "").toLowerCase().includes(q);
    if (idHit || promptHit) matches.add(id);
  }
  const visible = new Set(matches);
  // Walk ancestors up.
  for (const id of matches) {
    let cur = treeNodes.get(id);
    while (cur && cur.parentId) {
      if (visible.has(cur.parentId)) break;
      visible.add(cur.parentId);
      cur = treeNodes.get(cur.parentId);
    }
  }
  // Walk descendants down so users can see the matched subtree expanded.
  const stack = [...matches];
  while (stack.length) {
    const id = stack.pop();
    const kids = treeChildren.get(id) ?? [];
    for (const cid of kids) {
      if (!visible.has(cid)) {
        visible.add(cid);
        stack.push(cid);
      }
    }
  }
  return { matches, visible };
}

// Match order for "Enter selects first match" — use insertion order so
// the first match in the rendered tree is what gets selected.
function orderedMatches(filter) {
  if (!filter) return [];
  const sorted = [...treeNodes.values()]
    .filter((n) => filter.matches.has(n.id))
    .sort((a, b) => a.order - b.order);
  return sorted.map((n) => n.id);
}

function renderTree() {
  treeBodyEl.innerHTML = "";
  const filter = computeTreeFilter();
  if (treeRootId !== null) {
    const el = renderTreeNode(treeRootId, filter);
    if (el) treeBodyEl.appendChild(el);
  }
  if (treeActiveId !== null) {
    const n = treeNodes.get(treeActiveId);
    if (n) treeActiveEl.textContent = `${n.phase} · ${n.id}`;
  }
  if (filter) {
    const n = filter.matches.size;
    if (n === 0) {
      treeSearchCountEl.textContent = "no matches";
    } else if (treeMatchIndex >= 0) {
      treeSearchCountEl.textContent = `${(treeMatchIndex % n) + 1}/${n}`;
    } else {
      treeSearchCountEl.textContent = `${n} match${n === 1 ? "" : "es"}`;
    }
  } else {
    treeSearchCountEl.textContent = "";
  }
}

treeHeaderEl.addEventListener("click", () => {
  const collapsed = treeEl.classList.toggle("collapsed");
  treeToggleEl.textContent = collapsed ? "▸" : "▾";
});

treeSearchEl.addEventListener("input", () => {
  treeSearchQuery = treeSearchEl.value;
  treeSearchClearEl.classList.toggle("visible", treeSearchQuery.length > 0);
  treeMatchIndex = -1;
  renderTree();
});

treeSearchEl.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    ev.preventDefault();
    const ordered = orderedMatches(computeTreeFilter());
    if (ordered.length === 0) return;
    if (treeMatchIndex < 0) {
      treeMatchIndex = ev.shiftKey ? ordered.length - 1 : 0;
    } else {
      const delta = ev.shiftKey ? -1 : 1;
      treeMatchIndex = (treeMatchIndex + delta + ordered.length) % ordered.length;
    }
    const target = ordered[treeMatchIndex];
    // Bypass selectTreeNode's toggle behaviour — re-pressing Enter on the
    // same node should cycle, not deselect.
    if (selectedBboxId !== target) selectTreeNode(target);
    const row = treeBodyEl.querySelector(`.tree-node[data-id="${CSS.escape(target)}"]`);
    if (row) row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    renderTree();
  } else if (ev.key === "Escape") {
    treeSearchEl.value = "";
    treeSearchEl.dispatchEvent(new Event("input"));
  }
});

treeSearchClearEl.addEventListener("click", () => {
  treeSearchEl.value = "";
  treeSearchEl.dispatchEvent(new Event("input"));
  treeSearchEl.focus();
});

// --- three.js scene ---------------------------------------------------------

const host = document.getElementById("canvas-host");
// preserveDrawingBuffer keeps the WebGL framebuffer readable after present,
// which is required for the gif export path (gif.js calls getImageData on
// the canvas). Small perf cost on every frame; acceptable for a debug tool.
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101114);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const sceneRoot = new THREE.Group();
scene.add(sceneRoot);

// Bboxes live in a sibling group so they don't participate in fit-to-scene,
// and so clearScene can nuke them independently.
const bboxRoot = new THREE.Group();
scene.add(bboxRoot);
let bboxesShown = localStorage.getItem(BBOX_VISIBLE_STORAGE_KEY) !== "0";
function applyBboxToggleLabel() {
  bboxToggleEl.textContent = `bboxes: ${bboxesShown ? "on" : "off"}`;
  bboxToggleEl.classList.toggle("off", !bboxesShown);
}
applyBboxToggleLabel();
bboxToggleEl.addEventListener("click", () => {
  bboxesShown = !bboxesShown;
  localStorage.setItem(BBOX_VISIBLE_STORAGE_KEY, bboxesShown ? "1" : "0");
  applyBboxToggleLabel();
  refreshAllBboxVisibility();
  // Picking rule depends on bboxesShown — re-pick at the current cursor
  // position so the hover updates immediately instead of waiting for a
  // mouse move.
  pointerDirty = true;
});

let framesShown = localStorage.getItem(FRAMES_VISIBLE_STORAGE_KEY) !== "0";
function applyFramesToggleLabel() {
  framesToggleEl.textContent = `frames: ${framesShown ? "on" : "off"}`;
  framesToggleEl.classList.toggle("off", !framesShown);
}
applyFramesToggleLabel();
framesToggleEl.addEventListener("click", () => {
  framesShown = !framesShown;
  localStorage.setItem(FRAMES_VISIBLE_STORAGE_KEY, framesShown ? "1" : "0");
  applyFramesToggleLabel();
  refreshAllFrameModelVisibility();
  refreshAllSolidFillVisibility();
});

// "pick: inner" — only matters in bbox-only mode (bboxes: on). When ON,
// picking prefers the smallest/innermost bbox under the cursor so a nested
// object inside another object's bbox can be hovered & selected. When OFF,
// picking uses closest ray-entry (the visible wireframe face you're pointing
// at wins). Default OFF so newly-loaded sessions keep the "what you see is
// what you pick" behaviour.
let pickInner = localStorage.getItem(PICK_INNER_STORAGE_KEY) === "1";
function applyPickInnerToggleLabel() {
  pickInnerToggleEl.textContent = `pick: ${pickInner ? "inner" : "outer"}`;
  pickInnerToggleEl.classList.toggle("off", !pickInner);
}
applyPickInnerToggleLabel();
pickInnerToggleEl.addEventListener("click", () => {
  pickInner = !pickInner;
  localStorage.setItem(PICK_INNER_STORAGE_KEY, pickInner ? "1" : "0");
  applyPickInnerToggleLabel();
  pointerDirty = true;
});

// Solid-fill mode — drops a solid mesh into every object/frame bbox using its
// proxy shape (or the AABB itself when no proxy_shape is set). Zones stay
// wireframe. Intended for bbox-only mode so the scene reads as solids without
// running Trellis. Independent of the bbox wireframe toggle.
let solidFillShown = localStorage.getItem(SOLID_FILL_STORAGE_KEY) === "1";
function applySolidFillToggleLabel() {
  solidFillToggleEl.textContent = `fill: ${solidFillShown ? "on" : "off"}`;
  solidFillToggleEl.classList.toggle("off", !solidFillShown);
}
applySolidFillToggleLabel();
solidFillToggleEl.addEventListener("click", () => {
  solidFillShown = !solidFillShown;
  localStorage.setItem(SOLID_FILL_STORAGE_KEY, solidFillShown ? "1" : "0");
  applySolidFillToggleLabel();
  if (solidFillShown) rebuildAllSolidFills();
  else clearSolidFills();
});

const bboxes = new Map(); // id -> THREE.Box3Helper
const proxies = new Map(); // id -> THREE.Mesh (wireframe proxy silhouette)
const solidFills = new Map(); // id -> THREE.Mesh (solid proxy/AABB fill)
const modelsById = new Map(); // id -> THREE.Object3D (the loaded gltf.scene)
let hoveredBboxId = null;

const BBOX_COLOR_DEFAULT = 0xff3b3b;
const BBOX_COLOR_OBJECT = 0x6bd96e;
const BBOX_COLOR_FRAME = 0x7fb3d5;
const BBOX_COLOR_PROXY = 0xb46aff;
const BBOX_COLOR_HOVER = 0xffe14a;
const BBOX_COLOR_SELECTED = 0x4af0e0;
let selectedBboxId = null;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const bboxPickPoint = new THREE.Vector3();
const bboxPickSize = new THREE.Vector3();
let pointerDirty = false;
let lastPointerClientX = 0;
let lastPointerClientY = 0;
let controlsInteracting = false;

const tooltip = document.createElement("div");
tooltip.id = "bbox-tooltip";
tooltip.style.cssText = [
  "position: fixed",
  "padding: 5px 9px",
  "background: rgba(22, 24, 29, 0.94)",
  "color: #e6e6e6",
  "border: 1px solid #2a2d35",
  "border-radius: 4px",
  "font: 12px ui-monospace, SFMono-Regular, Menlo, monospace",
  "pointer-events: none",
  "display: none",
  "z-index: 10",
  "max-width: 360px",
  "white-space: pre-wrap",
  "line-height: 1.35",
].join("; ");
document.body.appendChild(tooltip);

const TOOLTIP_KIND_COLOR = {
  zone: "#9ad4ff",
  object: "#8bd17c",
  frame: "#7fb3d5",
};

const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.05,
  5000,
);
camera.position.set(8, 6, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 1, 0);
controls.update();

// Once the user drags the camera, stop auto-fitting so subsequent runs
// preserve their chosen angle. The flag deliberately survives clearScene.
let cameraUserMoved = false;
controls.addEventListener("start", () => {
  cameraUserMoved = true;
  controlsInteracting = true;
  setHoveredBbox(null);
  tooltip.style.display = "none";
});
controls.addEventListener("end", () => {
  controlsInteracting = false;
  pointerDirty = true;
});

// --- WASD fly controls (complementary to OrbitControls) --------------------
// WASD strafes on the horizontal plane relative to the camera direction;
// Q/E moves world-down/up; Shift multiplies speed. Translates camera and
// target together so OrbitControls' pivot follows the camera.
const pressedKeys = new Set();
let _lastMoveT = performance.now();
const _MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "r", "f"]);

function _isTypingTarget(t) {
  return t instanceof HTMLElement &&
    (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}

window.addEventListener("keydown", (ev) => {
  if (_isTypingTarget(ev.target)) return;
  const k = ev.key.toLowerCase();
  if (_MOVE_KEYS.has(k)) {
    pressedKeys.add(k);
    ev.preventDefault();
  } else if (k === "shift") {
    pressedKeys.add("shift");
  }
});

window.addEventListener("keyup", (ev) => {
  pressedKeys.delete(ev.key.toLowerCase());
});

// Alt-tab / focus-loss: drop held keys so they don't stick on.
window.addEventListener("blur", () => pressedKeys.clear());

const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);
const _move = new THREE.Vector3();

function _applyKeyboardMove(dt) {
  if (pressedKeys.size === 0) return;
  const shifted = pressedKeys.has("shift");
  const speed = 2 * (shifted ? 3 : 1) * dt;

  _fwd.subVectors(controls.target, camera.position);
  _fwd.y = 0;
  if (_fwd.lengthSq() === 0) return;
  _fwd.normalize();
  _right.crossVectors(_fwd, _worldUp).normalize();

  _move.set(0, 0, 0);
  if (pressedKeys.has("w")) _move.addScaledVector(_fwd, speed);
  if (pressedKeys.has("s")) _move.addScaledVector(_fwd, -speed);
  if (pressedKeys.has("d")) _move.addScaledVector(_right, speed);
  if (pressedKeys.has("a")) _move.addScaledVector(_right, -speed);
  if (pressedKeys.has("e")) _move.addScaledVector(_worldUp, speed);
  if (pressedKeys.has("q")) _move.addScaledVector(_worldUp, -speed);

  if (_move.lengthSq() !== 0) {
    camera.position.add(_move);
    controls.target.add(_move);
    cameraUserMoved = true;
  }

  // Dolly toward / away from the orbit target. Held key = continuous zoom;
  // ~1.5x per second baseline, 4x with shift.
  if (pressedKeys.has("r") || pressedKeys.has("f")) {
    const rate = shifted ? 4 : 1.5;
    let factor = 1;
    if (pressedKeys.has("r")) factor *= Math.pow(1 / rate, dt);
    if (pressedKeys.has("f")) factor *= Math.pow(rate, dt);
    _dolly(factor);
  }
}

function _dolly(factor) {
  // factor < 1 zooms in, factor > 1 zooms out. Implemented as scaling the
  // camera->target distance so OrbitControls' pivot semantics stay intact.
  const offset = camera.position.clone().sub(controls.target);
  const dist = offset.length();
  if (dist === 0) return;
  const minDist = 0.05;
  const maxDist = 4000;
  const newDist = Math.max(minDist, Math.min(maxDist, dist * factor));
  offset.multiplyScalar(newDist / dist);
  camera.position.copy(controls.target).add(offset);
  cameraUserMoved = true;
}

scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(8, 12, 6);
scene.add(dir);
scene.add(new THREE.AxesHelper(1));

// Infinite ground grid: a huge plane with a procedural grid shader. Lines
// antialias via screen-space derivatives and fade with distance so the plane
// never looks like it has an edge. Fade distance is driven from camera
// distance each frame so detail scales naturally as the user zooms.
const gridGeom = new THREE.PlaneGeometry(100000, 100000);
gridGeom.rotateX(-Math.PI / 2);
const gridMat = new THREE.ShaderMaterial({
  uniforms: {
    uCameraPos: { value: new THREE.Vector3() },
    uMinorColor: { value: new THREE.Color(0x202020) },
    uMajorColor: { value: new THREE.Color(0x505050) },
    uMinorSpacing: { value: 1.0 },
    uMajorSpacing: { value: 10.0 },
    uFadeStart: { value: 20.0 },
    uFadeEnd: { value: 200.0 },
  },
  vertexShader: `
    varying vec3 vWorldPos;
    void main() {
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }
  `,
  fragmentShader: `
    uniform vec3 uCameraPos;
    uniform vec3 uMinorColor;
    uniform vec3 uMajorColor;
    uniform float uMinorSpacing;
    uniform float uMajorSpacing;
    uniform float uFadeStart;
    uniform float uFadeEnd;
    varying vec3 vWorldPos;

    float gridLine(vec2 p, float spacing) {
      vec2 q = p / spacing;
      vec2 g = abs(fract(q - 0.5) - 0.5) / fwidth(q);
      return 1.0 - min(min(g.x, g.y), 1.0);
    }

    void main() {
      float minor = gridLine(vWorldPos.xz, uMinorSpacing);
      float major = gridLine(vWorldPos.xz, uMajorSpacing);
      float d = distance(vWorldPos.xz, uCameraPos.xz);
      float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, d);
      float alpha = max(minor * 0.5, major) * fade;
      if (alpha < 0.002) discard;
      vec3 col = mix(uMinorColor, uMajorColor, major);
      gl_FragColor = vec4(col, alpha);
    }
  `,
  transparent: true,
  depthWrite: false,
  side: THREE.DoubleSide,
});
const groundGrid = new THREE.Mesh(gridGeom, gridMat);
groundGrid.renderOrder = -1;
scene.add(groundGrid);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min(0.1, (now - _lastMoveT) / 1000);
  _lastMoveT = now;
  _applyKeyboardMove(dt);
  controls.update();

  gridMat.uniforms.uCameraPos.value.copy(camera.position);
  const camDist = Math.max(1, camera.position.distanceTo(controls.target));
  gridMat.uniforms.uFadeStart.value = camDist * 0.5;
  gridMat.uniforms.uFadeEnd.value = camDist * 6.0;

  if (pointerDirty && !controlsInteracting) {
    pointerDirty = false;
    raycaster.setFromCamera(pointer, camera);
    const hoveredId = pickHoveredBboxId();
    setHoveredBbox(hoveredId);
    if (hoveredId !== null) {
      positionTooltip(lastPointerClientX, lastPointerClientY, hoveredId);
    } else {
      tooltip.style.display = "none";
    }
  }

  renderer.render(scene, camera);
}
animate();

function clearScene() {
  resetModelQueue();
  while (sceneRoot.children.length > 0) {
    const child = sceneRoot.children[0];
    sceneRoot.remove(child);
    child.traverse?.((n) => {
      if (n.isMesh) {
        n.geometry?.dispose?.();
        const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
        for (const m of mats) m.dispose?.();
      }
    });
  }
  for (const helper of bboxes.values()) {
    bboxRoot.remove(helper);
    helper.geometry?.dispose?.();
    helper.material?.dispose?.();
  }
  bboxes.clear();
  for (const mesh of proxies.values()) {
    bboxRoot.remove(mesh);
    mesh.geometry?.dispose?.();
    mesh.material?.dispose?.();
  }
  proxies.clear();
  clearSolidFills();
  modelsById.clear();
  hoveredBboxId = null;
  selectedBboxId = null;
  tooltip.style.display = "none";
}

// Fit controls to the union of all loaded models after each addition.
// Skipped once the user has manually adjusted the camera.
function fitToScene() {
  if (cameraUserMoved) return;
  const box = new THREE.Box3().setFromObject(sceneRoot);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = 0.5 * Math.max(size.x, size.y, size.z);
  if (!isFinite(radius) || radius === 0) return;
  controls.target.copy(center);
  const dist = radius / Math.tan((camera.fov * Math.PI) / 360);
  const dirVec = new THREE.Vector3(1, 0.7, 1).normalize();
  camera.position.copy(center).addScaledVector(dirVec, dist * 1.6);
  camera.near = Math.max(0.01, radius / 100);
  camera.far = Math.max(100, radius * 100);
  camera.updateProjectionMatrix();
  controls.update();
}

// --- model loading ----------------------------------------------------------

const loader = new GLTFLoader();

// Concurrent loads — each GLB fetches/parses/uploads independently. The
// browser's per-origin connection limit (~6) naturally throttles the network
// side. `sceneGen` invalidates loads still in flight when the scene is reset
// (rewind / fresh snapshot on reconnect).
let sceneGen = 0;

function resetModelQueue() {
  sceneGen += 1;
}

function loadModel(event) {
  _loadModelNow(event, sceneGen);
}

async function _loadModelNow(event, gen) {
  if (gen !== sceneGen) return;
  const absUrl = new URL(event.url, SERVER_URL).toString();
  // Skip a re-load when this id already errored on the *same URL* during
  // this scene generation. The server occasionally emits `model` more
  // than once for one id (anchor completion loop, cached replay), and
  // re-running GLTFLoader on a known-bad GLB just spams the same parse
  // error. A new URL or a new sceneGen still goes through.
  const prior = assets.get(event.id);
  if (prior?.status === "error" && prior.modelUrl === event.url) return;
  upsertAsset(event.id, { modelUrl: event.url });
  try {
    const gltf = await loader.loadAsync(absUrl);
    if (gen !== sceneGen) return;
    gltf.scene.traverse((child) => {
      if (child.isMesh && child.material) {
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        for (const m of mats) m.side = THREE.DoubleSide;
      }
    });
    gltf.scene.name = `${event.artifact_kind}:${event.id}`;
    const prevModel = modelsById.get(event.id);
    if (prevModel) {
      sceneRoot.remove(prevModel);
      prevModel.traverse?.((n) => {
        if (n.isMesh) {
          n.geometry?.dispose?.();
          const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
          for (const m of mats) m.dispose?.();
        }
      });
    }
    sceneRoot.add(gltf.scene);
    modelsById.set(event.id, gltf.scene);
    applyModelVisibility(event.id);
    fitToScene();
    upsertAsset(event.id, { status: "loaded" });
  } catch (e) {
    appendEvent({ kind: "model.error", id: event.id, message: e.message });
    upsertAsset(event.id, { status: "error", errorMessage: e.message });
  }
}

// --- bbox overlays ----------------------------------------------------------

// `{ id, origin: [x,y,z], dimensions: [dx,dy,dz], proxy_shape?: ... }` —
// matches the Python BoundingBox+Node serialization. Signed and
// zero-valued dimensions are allowed (walls/floors are flat). If a proxy
// shape is set, we draw its wireframe silhouette in addition to the AABB
// wireframe so the user can see what the LLM and surface-snap are
// actually reasoning about.
function loadBbox(event) {
  const { id, origin, dimensions } = event;
  if (bboxes.has(id)) {
    const prev = bboxes.get(id);
    bboxRoot.remove(prev);
    prev.geometry?.dispose?.();
    prev.material?.dispose?.();
    if (hoveredBboxId === id) hoveredBboxId = null;
  }
  if (proxies.has(id)) {
    const prev = proxies.get(id);
    bboxRoot.remove(prev);
    prev.geometry?.dispose?.();
    prev.material?.dispose?.();
    proxies.delete(id);
  }
  disposeSolidFill(id);
  const ox = origin[0], oy = origin[1], oz = origin[2];
  const fx = ox + dimensions[0], fy = oy + dimensions[1], fz = oz + dimensions[2];
  const box3 = new THREE.Box3(
    new THREE.Vector3(Math.min(ox, fx), Math.min(oy, fy), Math.min(oz, fz)),
    new THREE.Vector3(Math.max(ox, fx), Math.max(oy, fy), Math.max(oz, fz)),
  );
  const helper = new THREE.Box3Helper(box3, BBOX_COLOR_DEFAULT);
  helper.userData.bboxId = id;
  const nodeKind = event.node_kind ?? "zone";
  helper.userData.nodeKind = nodeKind;
  helper.userData.proxyShape = event.proxy_shape ?? null;
  helper.userData.origin = origin;
  helper.userData.dimensions = dimensions;
  bboxRoot.add(helper);
  bboxes.set(id, helper);

  const proxyMesh = buildProxyWireframe(event.proxy_shape, origin, dimensions);
  if (proxyMesh !== null) {
    bboxRoot.add(proxyMesh);
    proxies.set(id, proxyMesh);
  }

  if (solidFillShown && nodeKind !== "zone") {
    const fill = buildSolidFill(event.proxy_shape, origin, dimensions, nodeKind);
    if (fill !== null) {
      bboxRoot.add(fill);
      solidFills.set(id, fill);
      applySolidFillVisibility(id);
    }
  }

  // If this id is already selected (user clicked before bbox arrived, or a
  // bbox is being replaced), reapply the selection color.
  applyBboxColor(id);
  applyBboxVisibility(id);
}

function buildProxyWireframe(proxyShape, origin, dimensions) {
  if (!proxyShape) return null;
  const sx = Math.abs(dimensions[0]);
  const sy = Math.abs(dimensions[1]);
  const sz = Math.abs(dimensions[2]);
  if (sx === 0 || sy === 0 || sz === 0) return null;
  const cx = origin[0] + dimensions[0] / 2;
  const cy = origin[1] + dimensions[1] / 2;
  const cz = origin[2] + dimensions[2] / 2;
  const yMin = Math.min(origin[1], origin[1] + dimensions[1]);

  let geom;
  let anchorY;
  if (proxyShape === "SPHERE") {
    // Ellipsoid inscribed in the AABB: unit sphere (diameter 1) scaled
    // to each AABB extent.
    geom = new THREE.SphereGeometry(0.5, 24, 16);
    geom.scale(sx, sy, sz);
    anchorY = cy;
  } else if (proxyShape === "HEMISPHERE") {
    // Top hemisphere with equatorial disk on the AABB's bottom face.
    // thetaLength = PI/2 starting at the north pole gives the upper half.
    geom = new THREE.SphereGeometry(
      0.5, 24, 16,
      0, Math.PI * 2,
      0, Math.PI / 2,
    );
    // The unit hemisphere spans y in [0, 0.5]; scale y by (sy / 0.5) so
    // the apex reaches +sy above the equator.
    geom.scale(sx, sy * 2, sz);
    anchorY = yMin;
  } else if (proxyShape === "CAPSULE") {
    const r = Math.min(sx, sz) / 2;
    const cylHeight = Math.max(0, sy - 2 * r);
    geom = new THREE.CapsuleGeometry(r, cylHeight, 8, 24);
    anchorY = cy;
  } else {
    return null;
  }

  const mat = new THREE.MeshBasicMaterial({
    color: BBOX_COLOR_DEFAULT, wireframe: true, transparent: true, opacity: 0.55,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(cx, anchorY, cz);
  mesh.renderOrder = 1;
  return mesh;
}

// Solid-fill counterpart to buildProxyWireframe. Same geometry math, but
// returns a lit, opaque mesh. When proxyShape is null/unset we fall back to
// a solid AABB box — types.py treats a missing proxy_shape as "the AABB is
// the proxy". `nodeKind` only steers the tint so frames read as walls vs.
// objects in the scene.
const SOLID_FILL_COLOR = {
  object: 0x4f7a45,
  frame: 0x4a6a82,
};

function buildSolidFill(proxyShape, origin, dimensions, nodeKind) {
  const sx = Math.abs(dimensions[0]);
  const sy = Math.abs(dimensions[1]);
  const sz = Math.abs(dimensions[2]);
  if (sx === 0 || sy === 0 || sz === 0) return null;
  const cx = origin[0] + dimensions[0] / 2;
  const cy = origin[1] + dimensions[1] / 2;
  const cz = origin[2] + dimensions[2] / 2;
  const yMin = Math.min(origin[1], origin[1] + dimensions[1]);

  let geom;
  let anchorY = cy;
  if (proxyShape === "SPHERE") {
    geom = new THREE.SphereGeometry(0.5, 32, 20);
    geom.scale(sx, sy, sz);
  } else if (proxyShape === "HEMISPHERE") {
    geom = new THREE.SphereGeometry(
      0.5, 32, 20,
      0, Math.PI * 2,
      0, Math.PI / 2,
    );
    geom.scale(sx, sy * 2, sz);
    anchorY = yMin;
  } else if (proxyShape === "CAPSULE") {
    const r = Math.min(sx, sz) / 2;
    const cylHeight = Math.max(0, sy - 2 * r);
    geom = new THREE.CapsuleGeometry(r, cylHeight, 12, 32);
  } else {
    // No proxy: fill the AABB itself as a solid box.
    geom = new THREE.BoxGeometry(sx, sy, sz);
  }

  const mat = new THREE.MeshLambertMaterial({
    color: SOLID_FILL_COLOR[nodeKind] ?? SOLID_FILL_COLOR.object,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(cx, anchorY, cz);
  return mesh;
}

function disposeSolidFill(id) {
  const prev = solidFills.get(id);
  if (!prev) return;
  bboxRoot.remove(prev);
  prev.geometry?.dispose?.();
  prev.material?.dispose?.();
  solidFills.delete(id);
}

function clearSolidFills() {
  for (const id of Array.from(solidFills.keys())) disposeSolidFill(id);
}

function rebuildAllSolidFills() {
  clearSolidFills();
  for (const [id, helper] of bboxes) {
    const nodeKind = helper.userData.nodeKind ?? "zone";
    if (nodeKind === "zone") continue;
    const origin = helper.userData.origin;
    const dimensions = helper.userData.dimensions;
    if (!origin || !dimensions) continue;
    const fill = buildSolidFill(helper.userData.proxyShape ?? null, origin, dimensions, nodeKind);
    if (fill === null) continue;
    bboxRoot.add(fill);
    solidFills.set(id, fill);
    applySolidFillVisibility(id);
  }
}

function applySolidFillVisibility(id) {
  const mesh = solidFills.get(id);
  if (!mesh) return;
  const isFrame = treeNodes.get(id)?.kind === "frame";
  const frameOk = isFrame ? framesShown : true;
  mesh.visible = frameOk && !effectivelyHidden(id);
}

function refreshAllSolidFillVisibility() {
  for (const id of solidFills.keys()) applySolidFillVisibility(id);
}

function applyBboxColor(id) {
  const helper = id !== null ? bboxes.get(id) : null;
  if (!helper) return;
  const base =
    helper.userData.proxyShape ? BBOX_COLOR_PROXY
    : helper.userData.nodeKind === "object" ? BBOX_COLOR_OBJECT
    : helper.userData.nodeKind === "frame" ? BBOX_COLOR_FRAME
    : BBOX_COLOR_DEFAULT;
  const color =
    id === selectedBboxId ? BBOX_COLOR_SELECTED
    : id === hoveredBboxId ? BBOX_COLOR_HOVER
    : base;
  helper.material.color.setHex(color);
  const proxy = proxies.get(id);
  if (proxy) proxy.material.color.setHex(color);
}

function applyBboxVisibility(id) {
  // Tree-driven hide always wins — hovering or selecting a hidden node
  // doesn't sneak its bbox back into view.
  const userVisible = !effectivelyHidden(id);
  const visible =
    userVisible &&
    (bboxesShown || id === hoveredBboxId || id === selectedBboxId);
  const helper = bboxes.get(id);
  if (helper) helper.visible = visible;
  const proxy = proxies.get(id);
  if (proxy) proxy.visible = visible;
}

function refreshAllBboxVisibility() {
  for (const id of bboxes.keys()) applyBboxVisibility(id);
}

function applyModelVisibility(id) {
  const model = modelsById.get(id);
  if (!model) return;
  const isFrame = treeNodes.get(id)?.kind === "frame";
  const frameOk = isFrame ? framesShown : true;
  model.visible = frameOk && !effectivelyHidden(id);
}

function refreshAllFrameModelVisibility() {
  for (const id of modelsById.keys()) applyModelVisibility(id);
}

function setHoveredBbox(id) {
  if (id === hoveredBboxId) return;
  const prev = hoveredBboxId;
  hoveredBboxId = id;
  applyBboxColor(prev);
  applyBboxColor(id);
  if (prev !== null) applyBboxVisibility(prev);
  if (id !== null) applyBboxVisibility(id);
}

// Fit the camera to a single Box3 — parameterised variant of fitToScene.
// Used by tree-click selection.
function frameBbox(box) {
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(0.5 * Math.max(size.x, size.y, size.z), 0.5);
  controls.target.copy(center);
  const dist = radius / Math.tan((camera.fov * Math.PI) / 360);
  const dirVec = new THREE.Vector3(1, 0.7, 1).normalize();
  camera.position.copy(center).addScaledVector(dirVec, dist * 1.8);
  camera.near = Math.max(0.01, radius / 100);
  camera.far = Math.max(100, radius * 100);
  camera.updateProjectionMatrix();
  controls.update();
}

// Frame `box` so that *every* corner is guaranteed inside the frustum, with
// a small padding margin. Differs from frameBbox in two ways:
//   1) uses the bounding-sphere radius (box diagonal) so any viewing angle
//      that doesn't align with a face still fits;
//   2) picks the limiting FOV between vertical and horizontal (aspect-aware)
//      so portrait windows still capture the whole box.
// Used by the gif replay to lock the camera at a framing that captures the
// entire scene volume for the duration of the export.
function frameBoxFully(box, { padding = 1.1, dirVec = null } = {}) {
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(0.5 * size.length(), 0.5);
  controls.target.copy(center);
  const fovV = (camera.fov * Math.PI) / 180;
  const fovH = 2 * Math.atan(Math.tan(fovV / 2) * camera.aspect);
  const limitingHalfFov = Math.min(fovV, fovH) / 2;
  const dist = (radius * padding) / Math.sin(limitingHalfFov);
  const dv = dirVec ?? new THREE.Vector3(1, 0.7, 1).normalize();
  camera.position.copy(center).addScaledVector(dv, dist);
  camera.near = Math.max(0.01, dist / 1000);
  camera.far = Math.max(100, dist * 10);
  camera.updateProjectionMatrix();
  controls.update();
}

// Find the root zone's bbox in a snapshot of recorded events. Returns a
// THREE.Box3 if found, or null. The root is identified by parent_id === null
// (which is how the server emits it) so we don't have to hard-code the id.
function findRootBboxFromEvents(events) {
  for (const ev of events) {
    if (ev.kind !== "bbox") continue;
    if (ev.parent_id != null) continue;
    if (!Array.isArray(ev.origin) || !Array.isArray(ev.dimensions)) continue;
    const ox = ev.origin[0], oy = ev.origin[1], oz = ev.origin[2];
    const fx = ox + ev.dimensions[0], fy = oy + ev.dimensions[1], fz = oz + ev.dimensions[2];
    return new THREE.Box3(
      new THREE.Vector3(Math.min(ox, fx), Math.min(oy, fy), Math.min(oz, fz)),
      new THREE.Vector3(Math.max(ox, fx), Math.max(oy, fy), Math.max(oz, fz)),
    );
  }
  return null;
}

function selectTreeNode(id) {
  const prev = selectedBboxId;
  // Toggle off if re-clicking the same node.
  selectedBboxId = prev === id ? null : id;
  if (prev !== null) {
    applyBboxColor(prev);
    applyBboxVisibility(prev);
  }
  if (selectedBboxId !== null) {
    applyBboxColor(selectedBboxId);
    applyBboxVisibility(selectedBboxId);
  }
  renderTree();
  renderTreeDetail();
  if (selectedBboxId !== null) {
    const helper = bboxes.get(selectedBboxId);
    if (helper) {
      // User took explicit camera control — stop auto-fit from later snapping
      // the view back to the full scene when new meshes land.
      cameraUserMoved = true;
      frameBbox(helper.box);
    }
  }
}

// --- detail panel -----------------------------------------------------------
// When a node is selected, the tree-body is hidden and this panel takes over
// the same slot to show the full prompts/plans/image-prompts for that node.
// The panel re-renders on every event that mutates the selected node (bbox,
// zone_plan, image, step) so it stays in sync with streamed updates.

function fmtMeters(arr) {
  if (!Array.isArray(arr) || arr.length !== 3) return "—";
  return arr.map((n) => (Number.isInteger(n) ? String(n) : n.toFixed(2))).join(", ");
}

function ancestorChain(id) {
  const chain = [];
  let cur = treeNodes.get(id)?.parentId ?? null;
  while (cur !== null) {
    const node = treeNodes.get(cur);
    if (!node) break;
    chain.unshift(node);
    cur = node.parentId ?? null;
  }
  return chain;
}

// Persistent preview state for the detail panel. We hold onto the container
// and live WebGL viewer across `renderTreeDetail` calls so frequent re-renders
// (phase ticks, sibling bbox updates) don't reset the user's camera or thrash
// the GLB load. Rebuilt only when the selected id or its underlying urls
// change; destroyed on deselect / slot switch.
let detailPreviewState = null; // { id, modelUrl, imageUrl, container, viewer }

function destroyDetailPreview() {
  if (detailPreviewState?.viewer) {
    try { detailPreviewState.viewer.dispose(); } catch {}
  }
  if (detailPreviewState?.container?.parentNode) {
    detailPreviewState.container.parentNode.removeChild(detailPreviewState.container);
  }
  detailPreviewState = null;
}

// Stand-alone GLB viewer inside the detail panel. Lives in its own scene/
// camera/renderer (separate from the main sandbox) so spinning the preview
// doesn't move the main camera. Returns a dispose() handle.
function mountMiniViewer(container, modelUrl) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1014);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  camera.position.set(2, 2, 3);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.className = "detail-mini-canvas";
  container.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.enablePan = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(5, 8, 6);
  scene.add(dir);

  function resize() {
    const w = container.clientWidth || 280;
    const h = Math.max(180, Math.round(w * 0.7));
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = w + "px";
    renderer.domElement.style.height = h + "px";
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(container);

  let disposed = false;
  let model = null;
  const localLoader = new GLTFLoader();
  localLoader.loadAsync(new URL(modelUrl, SERVER_URL).toString())
    .then((gltf) => {
      if (disposed) return;
      model = gltf.scene;
      model.traverse((c) => {
        if (c.isMesh && c.material) {
          const mats = Array.isArray(c.material) ? c.material : [c.material];
          for (const m of mats) m.side = THREE.DoubleSide;
        }
      });
      scene.add(model);
      const box = new THREE.Box3().setFromObject(model);
      if (!box.isEmpty()) {
        const center = new THREE.Vector3();
        box.getCenter(center);
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const dist = maxDim * 2.2;
        camera.position.copy(center).add(new THREE.Vector3(dist, dist * 0.7, dist));
        camera.near = Math.max(maxDim / 1000, 0.001);
        camera.far = Math.max(maxDim * 100, 100);
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();
      }
    })
    .catch(() => { /* keep the empty viewer; user already sees an error in the asset row */ });

  let rafId = 0;
  function tick() {
    if (disposed) return;
    controls.update();
    renderer.render(scene, camera);
    rafId = requestAnimationFrame(tick);
  }
  tick();

  return {
    dispose() {
      disposed = true;
      cancelAnimationFrame(rafId);
      try { ro.disconnect(); } catch {}
      try { controls.dispose(); } catch {}
      if (model) {
        model.traverse((n) => {
          if (n.isMesh) {
            n.geometry?.dispose?.();
            const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
            for (const m of mats) m.dispose?.();
          }
        });
      }
      try { renderer.dispose(); } catch {}
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    },
  };
}

// Builds (or returns the cached) preview container for `node`. Returns null
// when nothing has been generated yet so the caller can skip the section.
function ensureDetailPreview(node) {
  const a = assets.get(node.id);
  const imageUrl = a?.imageUrl ?? null;
  const modelUrl = a?.modelUrl ?? null;
  const modelLoaded = !!modelUrl && a?.status === "loaded";
  if (!imageUrl && !modelLoaded) {
    destroyDetailPreview();
    return null;
  }
  if (
    detailPreviewState
    && detailPreviewState.id === node.id
    && detailPreviewState.imageUrl === imageUrl
    && detailPreviewState.modelUrl === (modelLoaded ? modelUrl : null)
  ) {
    return detailPreviewState.container;
  }
  destroyDetailPreview();

  const wrap = document.createElement("div");
  wrap.className = "detail-section detail-preview";
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = "preview";
  wrap.appendChild(label);

  if (imageUrl) {
    const absImg = new URL(imageUrl, SERVER_URL).toString();
    const link = document.createElement("a");
    link.className = "detail-preview-image-link";
    link.href = absImg;
    link.target = "_blank";
    link.rel = "noopener";
    const img = document.createElement("img");
    img.className = "detail-preview-image";
    img.loading = "lazy";
    img.alt = node.id;
    img.src = absImg;
    link.appendChild(img);
    wrap.appendChild(link);
  }

  let viewer = null;
  if (modelLoaded) {
    const viewerWrap = document.createElement("div");
    viewerWrap.className = "detail-preview-viewer";
    wrap.appendChild(viewerWrap);
    viewer = mountMiniViewer(viewerWrap, modelUrl);
  }

  detailPreviewState = { id: node.id, imageUrl, modelUrl: modelLoaded ? modelUrl : null, container: wrap, viewer };
  return wrap;
}

function renderTreeDetail() {
  const id = selectedBboxId;
  if (id === null) {
    destroyDetailPreview();
    treeEl.classList.remove("detail-open");
    treeDetailEl.innerHTML = "";
    return;
  }
  const node = treeNodes.get(id);
  if (!node) {
    destroyDetailPreview();
    treeEl.classList.remove("detail-open");
    treeDetailEl.innerHTML = "";
    return;
  }
  treeEl.classList.add("detail-open");
  // Detach the live preview before wiping the panel so its WebGL canvas and
  // OrbitControls listeners survive the rebuild and we don't lose the user's
  // camera position on every event tick.
  if (detailPreviewState?.container?.parentNode === treeDetailEl) {
    treeDetailEl.removeChild(detailPreviewState.container);
  }
  treeDetailEl.textContent = "";

  // Back-to-tree button — also clears the selection so the bbox highlight
  // drops and the tree resumes its normal listing.
  const back = document.createElement("button");
  back.type = "button";
  back.className = "detail-back";
  back.textContent = "← back to tree";
  back.addEventListener("click", () => selectTreeNode(id)); // toggle off
  treeDetailEl.appendChild(back);
  // Quick jump into the full-tree modal — keeps the user's current selection
  // so they land on the same card in the larger view.
  const openFull = document.createElement("button");
  openFull.type = "button";
  openFull.className = "detail-open-full";
  openFull.textContent = "open full tree ⛶";
  openFull.title = "Open every node's prompts, plans, and bboxes in a modal";
  openFull.addEventListener("click", openTreeModal);
  treeDetailEl.appendChild(openFull);

  // Header row: kind tag + id.
  const idRow = document.createElement("div");
  idRow.className = "detail-id-row";
  const kindTag = document.createElement("span");
  kindTag.className = "detail-kind";
  kindTag.textContent = `[${node.kind ?? "zone"}]`;
  idRow.appendChild(kindTag);
  const idEl = document.createElement("span");
  idEl.className = `detail-id ${node.kind ?? "zone"}`;
  idEl.textContent = node.id;
  idRow.appendChild(idEl);
  treeDetailEl.appendChild(idRow);

  // Hierarchy — vertical, indented list of ancestors with the current node
  // at the bottom. Each ancestor row is clickable to jump selection. Shows
  // the [kind] tag and a short prompt preview so the user can see *what*
  // each parent is, not just its id.
  const chain = ancestorChain(id);
  if (chain.length > 0) {
    const hier = document.createElement("div");
    hier.className = "detail-hierarchy";
    function hierRow(n, depth, isCurrent) {
      const row = document.createElement(isCurrent ? "div" : "a");
      row.className = `detail-hier-row${isCurrent ? " current" : ""}`;
      row.style.paddingLeft = `${depth * 12}px`;
      if (!isCurrent) {
        row.addEventListener("click", (ev) => {
          ev.stopPropagation();
          selectTreeNode(n.id);
        });
      }
      if (depth > 0) {
        const branch = document.createElement("span");
        branch.className = "detail-hier-branch";
        branch.textContent = "└";
        row.appendChild(branch);
      }
      const kindEl = document.createElement("span");
      kindEl.className = `detail-hier-kind ${n.kind ?? "zone"}`;
      kindEl.textContent = `[${n.kind ?? "zone"}]`;
      row.appendChild(kindEl);
      const idEl2 = document.createElement("span");
      idEl2.className = `detail-hier-id ${n.kind ?? "zone"}`;
      idEl2.textContent = n.id;
      row.appendChild(idEl2);
      if (n.prompt) {
        const promptEl2 = document.createElement("span");
        promptEl2.className = "detail-hier-prompt";
        promptEl2.textContent = truncate(n.prompt, 50);
        promptEl2.title = n.prompt;
        row.appendChild(promptEl2);
      }
      return row;
    }
    for (let i = 0; i < chain.length; i++) {
      hier.appendChild(hierRow(chain[i], i, false));
    }
    hier.appendChild(hierRow(node, chain.length, true));
    treeDetailEl.appendChild(hier);
  }

  // Meta row: phase / bbox / proxy shape.
  const meta = document.createElement("div");
  meta.className = "detail-meta-row";
  function metaEntry(label, value) {
    const span = document.createElement("span");
    const lbl = document.createElement("span");
    lbl.textContent = `${label} `;
    span.appendChild(lbl);
    const b = document.createElement("b");
    b.textContent = value;
    span.appendChild(b);
    return span;
  }
  meta.appendChild(metaEntry("phase:", node.phase ?? "pending"));
  if (Array.isArray(node.origin) && Array.isArray(node.dimensions)) {
    meta.appendChild(metaEntry("origin:", `[${fmtMeters(node.origin)}]`));
    meta.appendChild(metaEntry("size:", `[${fmtMeters(node.dimensions)}] m`));
  }
  if (node.proxyShape) {
    meta.appendChild(metaEntry("proxy:", node.proxyShape));
  }
  treeDetailEl.appendChild(meta);

  // Prompt / plan / image-prompt sections — labelled so a reader can tell
  // which pipeline step authored each piece of text.
  function section(label, text) {
    const wrap = document.createElement("div");
    wrap.className = "detail-section";
    const lab = document.createElement("div");
    lab.className = "label";
    lab.textContent = label;
    wrap.appendChild(lab);
    const body = document.createElement("div");
    body.className = "body";
    if (text) {
      body.textContent = text;
    } else {
      body.classList.add("detail-empty");
      body.textContent = "(not yet authored)";
    }
    wrap.appendChild(body);
    treeDetailEl.appendChild(wrap);
  }

  section("seed prompt", node.prompt);
  if (node.kind === "zone" || node.plan) {
    section("zone plan", node.plan);
  }
  if (node.kind !== "zone") {
    section("image prompt", node.imagePrompt);
  }

  // Retry control — only meaningful for non-zone meshes that errored (or are
  // mid-retry from this client). Zones never produce meshes, so a retry would
  // have nothing to regenerate.
  if (node.kind !== "zone") {
    const a = assets.get(id);
    const status = a ? assetStatus(a) : "pending";
    if (status === "error" || retryingIds.has(id)) {
      const wrap = document.createElement("div");
      wrap.className = "detail-section";
      syncRetryButton(wrap, id, status, "detail-retry");
      treeDetailEl.appendChild(wrap);
    }
  }

  // Generated image + interactive mini 3D viewer when this node has assets.
  // Built (or reused) by ensureDetailPreview so the WebGL context isn't torn
  // down on incidental re-renders.
  const preview = ensureDetailPreview(node);
  if (preview) treeDetailEl.appendChild(preview);

  // Children list — for zones with sub-zones, lets the user drill into a
  // child without leaving the detail view.
  const childIds = treeChildren.get(id) ?? [];
  if (childIds.length > 0) {
    const wrap = document.createElement("div");
    wrap.className = "detail-section";
    const lab = document.createElement("div");
    lab.className = "label";
    lab.textContent = `children (${childIds.length})`;
    wrap.appendChild(lab);
    const list = document.createElement("div");
    list.className = "detail-children";
    for (const cid of childIds) {
      const a = document.createElement("a");
      const cn = treeNodes.get(cid);
      a.textContent = cn ? `${cn.id} — ${truncate(cn.prompt ?? "", 60)}` : cid;
      a.addEventListener("click", () => selectTreeNode(cid));
      list.appendChild(a);
    }
    wrap.appendChild(list);
    treeDetailEl.appendChild(wrap);
  }

  treeDetailEl.scrollTop = 0;
}

// --- full-tree modal -------------------------------------------------------
// Renders every node as a "detail card" in a single scrollable column,
// indented by depth, so the user can read all prompts/plans/bboxes
// side-by-side without clicking through nodes one at a time. Auto-refreshes
// while open so streaming runs update in place. Keep this rendering cheap
// (no live WebGL viewers per node — just static thumbnails) so big trees
// stay responsive.

let treeModalOpen = false;
let treeModalQuery = "";
let treeModalImagesOn = true;

function openTreeModal() {
  treeModalOpen = true;
  treeModalEl.classList.add("open");
  renderTreeModal();
  // Defer focus so the modal is in layout first.
  setTimeout(() => treeModalSearchEl?.focus(), 0);
}

function closeTreeModal() {
  treeModalOpen = false;
  treeModalEl.classList.remove("open");
}

function treeOrderedDepthFirst() {
  // DFS from the root, yielding [node, depth] in render order. Falls back to
  // insertion-ordered nodes when there is no root yet (early in a run).
  const out = [];
  if (treeRootId === null) {
    const sorted = [...treeNodes.values()].sort((a, b) => a.order - b.order);
    for (const n of sorted) out.push([n, 0]);
    return out;
  }
  const stack = [[treeRootId, 0]];
  while (stack.length) {
    const [id, depth] = stack.pop();
    const node = treeNodes.get(id);
    if (!node) continue;
    out.push([node, depth]);
    const kids = treeChildren.get(id) ?? [];
    for (let i = kids.length - 1; i >= 0; i--) stack.push([kids[i], depth + 1]);
  }
  return out;
}

function renderTreeModalCard(node, depth) {
  const card = document.createElement("div");
  const kind = node.kind ?? "zone";
  card.className = `tm-card kind-${kind}`;
  card.dataset.id = node.id;
  card.style.marginLeft = `${Math.min(depth, 8) * 18}px`;
  if (node.id === selectedBboxId) card.classList.add("selected");

  // Head row: depth, kind, id (clicks select), parent crumb, phase
  const head = document.createElement("div");
  head.className = "tm-card-head";
  if (depth > 0) {
    const d = document.createElement("span");
    d.className = "tm-depth";
    d.textContent = `d${depth}`;
    head.appendChild(d);
  }
  const kindEl = document.createElement("span");
  kindEl.className = "tm-kind";
  kindEl.textContent = `[${kind}]`;
  head.appendChild(kindEl);
  const idEl = document.createElement("span");
  idEl.className = `tm-id ${kind}`;
  idEl.textContent = node.id;
  idEl.title = "Click to select in scene";
  idEl.addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (selectedBboxId !== node.id) selectTreeNode(node.id);
    // Keep the modal open — user is doing observability, not navigating away.
    renderTreeModal();
  });
  head.appendChild(idEl);
  if (node.parentId) {
    const parent = treeNodes.get(node.parentId);
    if (parent) {
      const p = document.createElement("span");
      p.className = "tm-parent";
      p.textContent = `↑ ${parent.id}`;
      p.title = `Jump to parent: ${parent.id}`;
      p.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const target = treeModalBodyEl.querySelector(`.tm-card[data-id="${CSS.escape(parent.id)}"]`);
        if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
        if (selectedBboxId !== parent.id) selectTreeNode(parent.id);
        renderTreeModal();
      });
      head.appendChild(p);
    }
  }
  const phaseEl = document.createElement("span");
  phaseEl.className = `tm-phase phase-${node.phase ?? "pending"}`;
  phaseEl.textContent = node.phase ?? "pending";
  head.appendChild(phaseEl);
  card.appendChild(head);

  // Meta line: bbox + proxy shape, all on one row so the eye can scan a
  // column of cards quickly.
  if (Array.isArray(node.origin) || Array.isArray(node.dimensions) || node.proxyShape) {
    const meta = document.createElement("div");
    meta.className = "tm-meta";
    function metaCell(label, value) {
      const s = document.createElement("span");
      const l = document.createElement("span"); l.textContent = `${label} `; s.appendChild(l);
      const b = document.createElement("b"); b.textContent = value; s.appendChild(b);
      return s;
    }
    if (Array.isArray(node.origin)) meta.appendChild(metaCell("origin:", `[${fmtMeters(node.origin)}]`));
    if (Array.isArray(node.dimensions)) meta.appendChild(metaCell("size:", `[${fmtMeters(node.dimensions)}] m`));
    if (node.proxyShape) meta.appendChild(metaCell("proxy:", node.proxyShape));
    card.appendChild(meta);
  }

  // Body: sections on the left, optional thumbnail on the right. Wrapping
  // it in a grid keeps the thumbnail aligned to the top regardless of
  // section height.
  const body = document.createElement("div");
  body.className = "tm-card-body";
  const sections = document.createElement("div");
  sections.className = "tm-sections";
  body.appendChild(sections);

  function addSection(label, text) {
    const wrap = document.createElement("div");
    wrap.className = "tm-section";
    const lab = document.createElement("div");
    lab.className = "tm-section-label";
    lab.textContent = label;
    wrap.appendChild(lab);
    const b = document.createElement("div");
    b.className = "tm-section-body";
    if (text) {
      b.textContent = text;
    } else {
      b.classList.add("empty");
      b.textContent = "(not yet authored)";
    }
    wrap.appendChild(b);
    sections.appendChild(wrap);
  }

  addSection("seed prompt", node.prompt);
  if (kind === "zone" || node.plan) addSection("zone plan", node.plan);
  if (kind !== "zone") addSection("image prompt", node.imagePrompt);

  // Thumbnail: image only, never a live WebGL viewer (rendering N viewers
  // would torpedo the page for big trees). Click jumps to the asset.
  const a = assets.get(node.id);
  if (treeModalImagesOn && a?.imageUrl) {
    card.classList.add("has-thumb");
    const img = document.createElement("img");
    img.className = "tm-thumb";
    img.loading = "lazy";
    img.alt = node.id;
    img.src = new URL(a.imageUrl, SERVER_URL).toString();
    img.title = "Click to select in scene";
    img.addEventListener("click", () => {
      if (selectedBboxId !== node.id) selectTreeNode(node.id);
      renderTreeModal();
    });
    body.appendChild(img);
  }
  card.appendChild(body);

  // Children quick-nav: id chips so the user can hop down without finding
  // the row visually. Useful for fan-out zones with 20+ kids.
  const childIds = treeChildren.get(node.id) ?? [];
  if (childIds.length > 0) {
    const cwrap = document.createElement("div");
    cwrap.className = "tm-children-block";
    const lab = document.createElement("span");
    lab.textContent = `children (${childIds.length}):`;
    cwrap.appendChild(lab);
    for (const cid of childIds) {
      const link = document.createElement("a");
      link.textContent = cid;
      link.addEventListener("click", () => {
        const target = treeModalBodyEl.querySelector(`.tm-card[data-id="${CSS.escape(cid)}"]`);
        if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
        if (selectedBboxId !== cid) selectTreeNode(cid);
        renderTreeModal();
      });
      cwrap.appendChild(link);
    }
    card.appendChild(cwrap);
  }

  return card;
}

function renderTreeModal() {
  if (!treeModalOpen) return;
  treeModalBodyEl.innerHTML = "";
  const ordered = treeOrderedDepthFirst();
  if (ordered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "tm-empty";
    empty.textContent = "no nodes yet — start a run to populate the tree";
    treeModalBodyEl.appendChild(empty);
    return;
  }
  const q = treeModalQuery.trim().toLowerCase();
  const hits = q
    ? new Set(ordered
        .filter(([n]) => n.id.toLowerCase().includes(q)
          || (n.prompt ?? "").toLowerCase().includes(q)
          || (n.plan ?? "").toLowerCase().includes(q)
          || (n.imagePrompt ?? "").toLowerCase().includes(q))
        .map(([n]) => n.id))
    : null;

  for (const [node, depth] of ordered) {
    const card = renderTreeModalCard(node, depth);
    if (hits) {
      if (hits.has(node.id)) card.classList.add("hit");
      else card.classList.add("dimmed");
    }
    treeModalBodyEl.appendChild(card);
  }

  // Scroll the selected node into view on first open so the user lands at
  // their context, not at the root.
  if (selectedBboxId) {
    const sel = treeModalBodyEl.querySelector(`.tm-card[data-id="${CSS.escape(selectedBboxId)}"]`);
    if (sel) sel.scrollIntoView({ block: "center" });
  }
}

treeExpandEl?.addEventListener("click", (ev) => {
  ev.stopPropagation(); // don't toggle the tree-header collapse
  openTreeModal();
});
treeModalCloseEl?.addEventListener("click", closeTreeModal);
treeModalEl?.addEventListener("click", (ev) => {
  // Click on the backdrop (not the panel) closes — matches the replay modal.
  if (ev.target === treeModalEl) closeTreeModal();
});
treeModalSearchEl?.addEventListener("input", () => {
  treeModalQuery = treeModalSearchEl.value;
  renderTreeModal();
});
treeModalImageToggleEl?.addEventListener("change", () => {
  treeModalImagesOn = treeModalImageToggleEl.checked;
  renderTreeModal();
});
window.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && treeModalOpen) {
    // Don't steal Escape from the search input — let it clear there first.
    if (document.activeElement === treeModalSearchEl && treeModalSearchEl.value) {
      treeModalSearchEl.value = "";
      treeModalQuery = "";
      renderTreeModal();
      return;
    }
    closeTreeModal();
  }
});

function positionTooltip(clientX, clientY, id) {
  const node = treeNodes.get(id);
  const kind = node?.kind ?? "zone";
  const kindColor = TOOLTIP_KIND_COLOR[kind] ?? "#e6e6e6";

  // Build with DOM nodes rather than innerHTML so the prompt (LLM-authored
  // text) can't smuggle markup into the tooltip.
  tooltip.textContent = "";
  const head = document.createElement("div");
  const kindEl = document.createElement("span");
  kindEl.textContent = `[${kind}]`;
  kindEl.style.color = kindColor;
  head.appendChild(kindEl);
  head.appendChild(document.createTextNode(` ${id}`));
  tooltip.appendChild(head);

  // Sections: each carries a small label so the user can tell which prompt
  // came from which pipeline step. Order is "earliest in the pipeline first"
  // so reading top-to-bottom matches the order of LLM rewrites.
  //   seed  — zone_decompose / object_decomp output (what the LLM was given
  //           as the brief for this node)
  //   plan  — zone_plan output (zones only)
  //   image — image-prompt noun phrase actually sent to Banana+Trellis
  //           (objects/frames only, once generated)
  const sections = [];
  if (node?.prompt) sections.push(["seed", node.prompt]);
  if (kind === "zone" && node?.plan) sections.push(["plan", node.plan]);
  if (kind !== "zone" && node?.imagePrompt && node.imagePrompt !== node.prompt) {
    sections.push(["image", node.imagePrompt]);
  }
  for (const [label, text] of sections) {
    const row = document.createElement("div");
    row.style.marginTop = "4px";
    row.style.color = "#bdbdbd";
    const lbl = document.createElement("span");
    lbl.textContent = `${label}: `;
    lbl.style.color = "#7a8190";
    row.appendChild(lbl);
    row.appendChild(document.createTextNode(text));
    tooltip.appendChild(row);
  }

  // Flip left/up when the tooltip would overflow the viewport so the cursor
  // can keep approaching the hovered bbox from any direction.
  tooltip.style.display = "block";
  tooltip.style.left = "0px";
  tooltip.style.top = "0px";
  const w = tooltip.offsetWidth;
  const h = tooltip.offsetHeight;
  const pad = 12;
  let x = clientX + pad;
  let y = clientY + pad;
  if (x + w > window.innerWidth) x = clientX - pad - w;
  if (y + h > window.innerHeight) y = clientY - pad - h;
  tooltip.style.left = `${Math.max(0, x)}px`;
  tooltip.style.top = `${Math.max(0, y)}px`;
}

function pickHoveredBboxId() {
  let bestId = null;
  let bestKindRank = Infinity;
  let bestDistance = Infinity;
  let bestSize = Infinity;
  // Three regimes:
  //  - bboxes: off → user is visually pointing at meshes. Prefer the inner
  //    object/frame so a tiny prop inside a big zone is still pickable.
  //  - bboxes: on, pick: outer → wireframes are visible; pick the one whose
  //    face the ray hits first (closest entry point), so hovering matches
  //    what's drawn.
  //  - bboxes: on, pick: inner → still in bbox-only mode, but the user is
  //    drilling into nested objects: prefer the smallest/innermost bbox so
  //    an object enclosed by another object's bbox becomes pickable.
  const innerPreferred = !bboxesShown || pickInner;
  for (const [id, helper] of bboxes) {
    if (effectivelyHidden(id)) continue;
    const kind = treeNodes.get(id)?.kind;
    if (!raycaster.ray.intersectBox(helper.box, bboxPickPoint)) continue;

    const kindRank = kind === "object" ? 0 : kind === "frame" ? 1 : 2;
    const distance = bboxPickPoint.distanceToSquared(camera.position);
    helper.box.getSize(bboxPickSize);
    const sizeTieBreaker = bboxPickSize.lengthSq();

    const better = innerPreferred
      ? (
          kindRank < bestKindRank ||
          (kindRank === bestKindRank && sizeTieBreaker < bestSize) ||
          (kindRank === bestKindRank && sizeTieBreaker === bestSize && distance < bestDistance)
        )
      : (
          distance < bestDistance ||
          (distance === bestDistance && kindRank < bestKindRank) ||
          (distance === bestDistance && kindRank === bestKindRank && sizeTieBreaker < bestSize)
        );
    if (better) {
      bestId = id;
      bestKindRank = kindRank;
      bestDistance = distance;
      bestSize = sizeTieBreaker;
    }
  }
  return bestId;
}

renderer.domElement.addEventListener("pointermove", (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  lastPointerClientX = ev.clientX;
  lastPointerClientY = ev.clientY;
  pointerDirty = true;
});

renderer.domElement.addEventListener("pointerleave", () => {
  setHoveredBbox(null);
  tooltip.style.display = "none";
});

// Click-to-select on the canvas. OrbitControls also listens for pointerdown
// to start orbiting, so we have to distinguish a click from the end of an
// orbit drag. Approach: snapshot the down position + time, and only treat
// pointerup as a selection click when the cursor barely moved and the gesture
// was short. Anything more is a camera drag — pass through to OrbitControls.
const CLICK_MAX_MOVE_PX = 4;
const CLICK_MAX_DURATION_MS = 400;
let _downX = 0;
let _downY = 0;
let _downT = 0;
let _downButton = -1;

renderer.domElement.addEventListener("pointerdown", (ev) => {
  _downX = ev.clientX;
  _downY = ev.clientY;
  _downT = performance.now();
  _downButton = ev.button;
});

renderer.domElement.addEventListener("pointerup", (ev) => {
  if (_downButton !== 0 || ev.button !== 0) return; // left-click only
  const dx = ev.clientX - _downX;
  const dy = ev.clientY - _downY;
  const dt = performance.now() - _downT;
  if (Math.hypot(dx, dy) > CLICK_MAX_MOVE_PX || dt > CLICK_MAX_DURATION_MS) return;

  // Reuse the hover picker — same kind-rank / inner-vs-outer rules apply
  // so selecting matches whatever the hover tooltip is showing.
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const id = pickHoveredBboxId();
  if (id !== null) {
    selectTreeNode(id);
    // Scroll the corresponding tree row into view so the link between the
    // 3D click and the (now-open) detail panel is obvious if the user closes
    // the detail view.
    const row = treeBodyEl.querySelector(`.tree-node[data-id="${CSS.escape(id)}"]`);
    if (row) row.scrollIntoView({ block: "nearest" });
  }
});

// --- event dispatch ---------------------------------------------------------

// Event-index high-water mark. The server re-replays the entire snapshot
// from index 0 on every SSE (re)connect. EventSource auto-reconnects on its
// own (server idle, network blip), so without this guard we'd wipe and
// reload every model on every reconnect — which is exactly the "models keep
// reloading" behaviour the user reports when inspecting a finished run. We
// dedupe by index instead and let already-processed events fall through.
// Reset to -1 only on explicit user-driven state wipes (slot switch, reset,
// rewind), where a fresh replay genuinely needs to be re-applied.
let highestEventIndex = -1;

// Buffered event log for the active slot — captured as events stream in so
// the replay-to-gif feature can re-dispatch the build from scratch. Reset by
// the same hooks that reset `highestEventIndex` (slot switch / rewind /
// reset). Stored in dispatch order, deduped by index.
const recordedEvents = [];

function dispatch(event) {
  if (typeof event.index === "number") {
    if (event.index <= highestEventIndex) return;
    highestEventIndex = event.index;
  }
  recordedEvents.push(event);
  updateReplayButton();
  appendEvent(event);
  switch (event.kind) {
    case "run.start":
      setStatus(`run :: ${event.model}`);
      break;
    case "run.done":
      runFinished = true;
      if (meshErrors.size > 0) showRunCompleteWithErrors();
      else setStatus("run complete");
      if (currentSource) { currentSource.close(); currentSource = null; }
      refreshSlots();
      break;
    case "run.error":
      runFinished = true;
      setStatus(`error: ${event.message}`, "err");
      if (currentSource) { currentSource.close(); currentSource = null; }
      refreshSlots();
      break;
    case "run.paused":
      runFinished = true;
      setStatus("paused");
      if (currentSource) { currentSource.close(); currentSource = null; }
      refreshSlots();
      break;
    case "mesh.error":
      // Surface the failure: track for the run.done summary, paint the
      // tree node + asset card as errored so users see it without grepping
      // the log panel.
      retryingIds.delete(event.id);
      meshErrors.set(event.id, event.message ?? "unknown error");
      treeSetPhase(event.id, "error");
      upsertAsset(event.id, { status: "error", errorMessage: event.message });
      refreshPostRunStatus();
      break;
    case "mesh.retry":
      // Server-side retry kickoff. The user may have triggered it from
      // *this* client (retryingIds already set) or another tab; either
      // way, clear the prior error state so the asset/tree both flip back
      // to in-flight and the retry button greys out.
      retryingIds.add(event.id);
      meshErrors.delete(event.id);
      upsertAsset(event.id, { status: "pending", errorMessage: null });
      if (treeNodes.has(event.id)) treeSetPhase(event.id, "generating_mesh");
      refreshPostRunStatus();
      break;
    case "bbox":
      loadBbox(event);
      treeUpsert(event.id, {
        parentId: event.parent_id ?? null,
        prompt: event.prompt ?? null,
        kind: event.node_kind ?? "zone",
        origin: event.origin,
        dimensions: event.dimensions,
        proxyShape: event.proxy_shape ?? null,
      });
      renderTree();
      if (event.id === selectedBboxId) renderTreeDetail();
      if (treeModalOpen) renderTreeModal();
      break;
    case "divider.decompose":
      // Pre-declare children so the tree shows them (in pending state) before
      // their bboxes resolve. `children` ships as [{id, prompt}, ...].
      for (const c of event.children ?? []) {
        treeUpsert(c.id, { parentId: event.node, prompt: c.prompt, kind: "zone" });
      }
      renderTree();
      if (treeModalOpen) renderTreeModal();
      break;
    case "divider.zone_plan":
      // Stash the authored zone plan on the node so the tooltip can surface
      // it. The plan often arrives before the bbox is resolved, so upsert
      // (which tolerates a half-formed node) rather than gating on existence.
      if (event.node && typeof event.plan === "string") {
        treeUpsert(event.node, { plan: event.plan });
        if (event.node === selectedBboxId) renderTreeDetail();
        if (treeModalOpen) renderTreeModal();
      }
      break;
    case "step":
      treeSetPhase(event.node, event.phase);
      break;
    case "mesh.submit":
      // Object mesh generation kicked off — show it on the tree.
      treeSetPhase(event.id, "generating_mesh");
      break;
    case "image":
      upsertAsset(event.id, { imageUrl: event.url, prompt: event.prompt });
      // Image-prompt noun phrase (post-rewrite) — distinct from the seed
      // prompt stored on the bbox event. Tooltip + detail panel show both.
      if (typeof event.prompt === "string") {
        treeUpsert(event.id, { imagePrompt: event.prompt });
        if (event.id === selectedBboxId) renderTreeDetail();
      }
      if (treeModalOpen) renderTreeModal();
      break;
    case "model":
      loadModel(event);
      treeSetPhase(event.id, "done");
      retryingIds.delete(event.id);
      // A `model` event for this id is proof the mesh exists now, so any
      // prior `mesh.error` for the same id is stale. Without this clear,
      // snapshot replay (where a past error then a later success both
      // appear in the recorded log) leaves `meshErrors` permanently
      // overcounted — the tree shows `done` but the run-complete summary
      // still reads "N meshes failed". Equally applies to in-pipeline
      // recoveries that ship a `model` without a preceding `mesh.retry`.
      meshErrors.delete(event.id);
      if (event.id === selectedBboxId) renderTreeDetail();
      if (treeModalOpen) renderTreeModal();
      refreshPostRunStatus();
      break;
    // Everything else is already shown as a log line above.
  }
}

// --- slot picker + run lifecycle --------------------------------------------

// All seven pipelines run in the background on the server. The client
// chooses which one to view — switching closes the active SSE, clears the
// scene, and reconnects to the selected slot's stream.

let currentSource = null;
let currentSlotId = null;
let slotSummaries = [];  // latest /slots payload, for tab rendering
let slotNeedsResume = false;

function renderSlotTabs() {
  // Wipe any existing .slot-tab children; keep the #slot-reset button.
  for (const child of Array.from(slotsEl.querySelectorAll(".slot-tab"))) {
    child.remove();
  }
  for (const s of slotSummaries) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "slot-tab" + (s.id === currentSlotId ? " active" : "");
    tab.dataset.slotId = s.id;
    tab.title = s.prompt ?? "";

    const dot = document.createElement("span");
    dot.className = `slot-dot status-${s.status ?? "idle"}`;
    tab.appendChild(dot);

    const label = document.createElement("span");
    label.textContent = s.id;
    tab.appendChild(label);

    tab.addEventListener("click", () => switchSlot(s.id));
    slotsEl.insertBefore(tab, resetEl);
  }
}

function updateResumeButton() {
  const slot = slotSummaries.find((s) => s.id === currentSlotId);
  const status = slot?.status;
  resumeEl.className = "";
  resumeEl.style.display = "none";
  if (status === "idle") {
    resumeEl.style.display = "";
    resumeEl.className = "paused";
    resumeEl.textContent = "start";
    resumeEl.title = "Start this run";
  } else if (status === "paused") {
    resumeEl.style.display = "";
    resumeEl.className = "paused";
    resumeEl.textContent = "resume";
    resumeEl.title = "Resume the interrupted run";
  } else if (status === "error") {
    resumeEl.style.display = "";
    resumeEl.className = "error";
    resumeEl.textContent = "retry";
    resumeEl.title = "Retry the failed run";
  } else if (status === "running") {
    resumeEl.style.display = "";
    resumeEl.className = "running";
    resumeEl.textContent = "pause";
    resumeEl.title = "Pause this run";
  } else if (slotNeedsResume && currentSlotId !== null) {
    slotNeedsResume = false;
    subscribe(slotEventsUrl(currentSlotId));
  }
}

async function refreshSlots() {
  try {
    const res = await fetch(new URL("/slots", SERVER_URL));
    if (!res.ok) return;
    slotSummaries = await res.json();
    renderSlotTabs();
    updateResumeButton();
  } catch {
    // Transient; next tick will retry.
  }
}

function switchSlot(id) {
  if (id === currentSlotId) return;
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
  clearScene();
  clearLog();
  clearAssets();
  treeClear();
  clearMeshErrors();
  highestEventIndex = -1;
  recordedEvents.length = 0;
  updateReplayButton();
  currentSlotId = id;
  try { localStorage.setItem(SLOT_STORAGE_KEY, id); } catch {}

  const slot = slotSummaries.find((s) => s.id === id);
  const status = slot?.status;
  slotNeedsResume = status === "idle" || status === "paused" || status === "error";
  renderSlotTabs();
  updateResumeButton();
  if (slotNeedsResume) {
    setStatus(`slot :: ${id} — ${status}`);
  } else {
    setStatus(`slot :: ${id}`);
    subscribe(slotEventsUrl(id));
  }
}

function slotEventsUrl(id) {
  return new URL(`/slots/${encodeURIComponent(id)}/events`, SERVER_URL).toString();
}

async function resetSlot(id) {
  const ok = window.confirm(
    `Wipe runs/${id}/ and restart the pipeline for this slot?`,
  );
  if (!ok) return;
  resetEl.disabled = true;
  try {
    const res = await fetch(
      new URL(`/slots/${encodeURIComponent(id)}/reset`, SERVER_URL),
      { method: "POST" },
    );
    if (!res.ok) {
      setStatus(`reset failed: HTTP ${res.status}`, "err");
      return;
    }
    if (currentSource) {
      currentSource.close();
      currentSource = null;
    }
    clearScene();
    clearLog();
    clearAssets();
    treeClear();
    clearMeshErrors();
    highestEventIndex = -1;
    recordedEvents.length = 0;
    updateReplayButton();
    slotNeedsResume = false;
    setStatus(`slot ${id} reset — streaming events…`);
    subscribe(slotEventsUrl(id));
    refreshSlots();
  } catch (e) {
    setStatus(`reset failed: ${e.message}`, "err");
  } finally {
    resetEl.disabled = false;
  }
}

function subscribe(url) {
  const es = new EventSource(url);
  currentSource = es;
  es.onmessage = (ev) => {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch {
      return;
    }
    dispatch(data);
  };
  es.onerror = () => {
    // EventSource auto-reconnects on transient failures; only surface a hard close.
    if (es.readyState === EventSource.CLOSED && currentSource === es) {
      setStatus("stream closed", "err");
      currentSource = null;
    }
  };
}

async function rewindTo(index) {
  if (currentSlotId === null) return;
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
  clearScene();
  clearLog();
  clearAssets();
  treeClear();
  clearMeshErrors();
  highestEventIndex = -1;
  recordedEvents.length = 0;
  updateReplayButton();
  setStatus(`POST /slots/${currentSlotId}/rewind to ${index} …`);

  let res;
  try {
    res = await fetch(
      new URL(`/slots/${encodeURIComponent(currentSlotId)}/rewind`, SERVER_URL),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ to_event_index: index }),
      },
    );
  } catch (e) {
    setStatus(`rewind failed: ${e.message}`, "err");
    return;
  }
  if (!res.ok) {
    setStatus(`HTTP ${res.status}: ${await res.text()}`, "err");
    return;
  }
  slotNeedsResume = false;
  setStatus(`rewound to ${index} — streaming events…`);
  subscribe(slotEventsUrl(currentSlotId));
  refreshSlots();
}

resetEl.addEventListener("click", () => {
  if (currentSlotId !== null) resetSlot(currentSlotId);
});

async function resumeSlot(id) {
  resumeEl.disabled = true;
  try {
    const res = await fetch(
      new URL(`/slots/${encodeURIComponent(id)}/resume`, SERVER_URL),
      { method: "POST" },
    );
    if (!res.ok) {
      setStatus(`resume failed: HTTP ${res.status}`, "err");
      return;
    }
    slotNeedsResume = false;
    if (currentSource) {
      currentSource.close();
      currentSource = null;
    }
    clearScene();
    clearLog();
    clearAssets();
    treeClear();
    clearMeshErrors();
    highestEventIndex = -1;
    recordedEvents.length = 0;
    updateReplayButton();
    setStatus(`resumed — streaming events…`);
    subscribe(slotEventsUrl(id));
    refreshSlots();
  } catch (e) {
    setStatus(`resume failed: ${e.message}`, "err");
  } finally {
    resumeEl.disabled = false;
  }
}

async function pauseSlot(id) {
  resumeEl.disabled = true;
  try {
    const res = await fetch(
      new URL(`/slots/${encodeURIComponent(id)}/pause`, SERVER_URL),
      { method: "POST" },
    );
    if (!res.ok) {
      setStatus(`pause failed: HTTP ${res.status}`, "err");
      return;
    }
    setStatus("paused");
    refreshSlots();
  } catch (e) {
    setStatus(`pause failed: ${e.message}`, "err");
  } finally {
    resumeEl.disabled = false;
  }
}

resumeEl.addEventListener("click", () => {
  if (currentSlotId === null) return;
  const slot = slotSummaries.find((s) => s.id === currentSlotId);
  if (slot?.status === "running") {
    pauseSlot(currentSlotId);
  } else {
    resumeSlot(currentSlotId);
  }
});

document.getElementById("zoom-in").addEventListener("click", () => _dolly(0.8));
document.getElementById("zoom-out").addEventListener("click", () => _dolly(1.25));

// --- GLB export --------------------------------------------------------------

exportGlbEl.addEventListener("click", async () => {
  if (modelsById.size === 0) return;
  exportGlbEl.disabled = true;
  exportGlbEl.textContent = "exporting…";
  try {
    // Temporarily force all models visible so hidden frames are included.
    const wasHidden = [];
    sceneRoot.traverse((obj) => {
      if (!obj.visible) {
        wasHidden.push(obj);
        obj.visible = true;
      }
    });

    const exporter = new GLTFExporter();
    const glb = await exporter.parseAsync(sceneRoot, { binary: true });

    for (const obj of wasHidden) obj.visible = false;

    const blob = new Blob([glb], { type: "model/gltf-binary" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentSlotId ?? "scene"}.glb`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    appendEvent({ kind: "run.error", message: `GLB export failed: ${e.message}` });
  } finally {
    exportGlbEl.disabled = false;
    exportGlbEl.textContent = "export .glb";
  }
});

// Boot: load slot list, pick the remembered slot (or the first), subscribe.
(async () => {
  await refreshSlots();
  if (slotSummaries.length === 0) {
    setStatus("no slots reported by server", "err");
    return;
  }
  let saved = null;
  try { saved = localStorage.getItem(SLOT_STORAGE_KEY); } catch {}
  const pick = slotSummaries.find((s) => s.id === saved)?.id ?? slotSummaries[0].id;
  switchSlot(pick);
})();

// Keep tab status dots fresh for slots the user isn't viewing.
setInterval(refreshSlots, 2000);

// --- replay → gif ----------------------------------------------------------
//
// User clicks "replay → gif" → a modal opens. The user can preview the
// build (re-dispatches the entire `recordedEvents` log against the live
// scene at controlled speed) and then export an animated GIF.
//
// Mechanism: we wipe the current scene, then walk the recorded log,
// chunking events into per-frame batches. Each batch is dispatched, the
// renderer draws once, and (in record mode) the canvas pixels are pushed
// into a gif.js encoder. Mesh GLB loads remain async — they'll pop into
// the gif whenever the loader resolves, which for cached artifacts is
// usually within a frame or two.

function updateReplayButton() {
  if (!replayGifEl) return;
  replayGifEl.disabled = recordedEvents.length === 0;
}
updateReplayButton();

let replayActive = false;       // true while preview or record is running
let replayCancelRequested = false;
let replayInProgress = false;   // gates concurrent invocations of run()

function setReplayStatus(text, cls = "") {
  replayStatusEl.textContent = text;
  replayStatusEl.className = cls;
}

function setReplayProgress(frac) {
  replayProgressBarEl.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
}

function readReplaySpeed() {
  return parseFloat(replaySpeedEl.value) || 1;
}

function readReplayFps() {
  return parseInt(replayFpsEl.value, 10) || 15;
}

// Walk the buffered events and group them into per-frame batches so the
// preview/recording can advance at a fixed framerate. `eventsPerFrame =
// max(1, round(totalEvents / targetTotalFrames))`, where targetTotalFrames
// derives from the chosen fps and a base "1 event = 1/speed seconds" pace.
function planFrames(events, fps, speed) {
  // Target a build that takes roughly events / speed seconds at 1× — so
  // total frames at our fps is (events / speed) * fps. Cap at 600 frames
  // so very long runs don't try to encode minute-long gifs.
  const targetFrames = Math.min(
    600,
    Math.max(8, Math.round((events.length / Math.max(0.25, speed)) * (fps / 30))),
  );
  const perFrame = Math.max(1, Math.ceil(events.length / targetFrames));
  const batches = [];
  for (let i = 0; i < events.length; i += perFrame) {
    batches.push(events.slice(i, i + perFrame));
  }
  return batches;
}

function updateDurationEstimate() {
  const fps = readReplayFps();
  const speed = readReplaySpeed();
  const batches = planFrames(recordedEvents, fps, speed);
  const seconds = batches.length / fps;
  replayEventCountEl.textContent = String(recordedEvents.length);
  replayDurationEstEl.textContent = batches.length === 0
    ? "—"
    : `${seconds.toFixed(1)}s · ${batches.length} frames`;
}

replaySpeedEl.addEventListener("input", () => {
  replaySpeedValEl.textContent = `${parseFloat(replaySpeedEl.value).toFixed(2)}×`;
  updateDurationEstimate();
});
replayFpsEl.addEventListener("input", () => {
  replayFpsValEl.textContent = `${replayFpsEl.value} fps`;
  updateDurationEstimate();
});

function openReplayModal() {
  if (recordedEvents.length === 0) return;
  replayModalEl.classList.add("open");
  replaySpeedValEl.textContent = `${parseFloat(replaySpeedEl.value).toFixed(2)}×`;
  replayFpsValEl.textContent = `${replayFpsEl.value} fps`;
  updateDurationEstimate();
  setReplayStatus("idle");
  setReplayProgress(0);
  showReplayPlaceholder();
}

function closeReplayModal() {
  // If a replay is mid-run, request cancellation; the run loop checks the
  // flag between frames and bails out, then we close. If nothing is
  // running, close immediately.
  if (replayActive) {
    replayCancelRequested = true;
    return;
  }
  replayModalEl.classList.remove("open");
  // Free the gif blob URL when the modal closes — keeping it alive across
  // sessions would leak memory if the user repeatedly opens/saves.
  if (lastGifUrl) {
    URL.revokeObjectURL(lastGifUrl);
    lastGifUrl = null;
    lastGifBlob = null;
    replayResultImgEl.removeAttribute("src");
    replayDownloadEl.disabled = true;
  }
}

replayGifEl.addEventListener("click", openReplayModal);
replayCloseEl.addEventListener("click", closeReplayModal);
replayModalEl.addEventListener("click", (ev) => {
  if (ev.target === replayModalEl) closeReplayModal();
});

// Snapshot the live scene + ephemeral UI state, wipe everything, replay the
// recorded log, optionally encode frames into a gif, then leave the scene
// holding the final-state of the replay (which matches the live state).
// Returns the encoded blob in record mode, or null in preview mode.
async function runReplay({ record }) {
  if (replayInProgress) return null;
  replayInProgress = true;
  replayActive = true;
  replayCancelRequested = false;
  setReplayProgress(0);

  // Disconnect from the live SSE stream so server-pushed events don't race
  // the replay. The user can reconnect via the resume / slot-switch paths
  // afterwards; for a finished run nothing's incoming anyway.
  const reconnectAfter = currentSource !== null;
  const slotForReconnect = currentSlotId;
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }

  // Snapshot the recorded log + the current camera state so we restore them
  // after the replay finishes (or the user cancels). The replay reuses the
  // same dispatch path which resets `highestEventIndex` and clears state, so
  // we have to take care to leave the user where they started.
  const events = recordedEvents.slice();
  const camSnapshot = {
    pos: camera.position.clone(),
    target: controls.target.clone(),
    userMoved: cameraUserMoved,
  };

  // Wipe everything that the replay will rebuild.
  clearScene();
  clearLog();
  clearAssets();
  treeClear();
  clearMeshErrors();
  highestEventIndex = -1;
  recordedEvents.length = 0;
  updateReplayButton();

  const fps = readReplayFps();
  const speed = readReplaySpeed();
  const batches = planFrames(events, fps, speed);
  const frameIntervalMs = 1000 / fps;

  let gif = null;
  if (record) {
    if (typeof window.GIF === "undefined") {
      setReplayStatus("gif.js library failed to load", "err");
      replayActive = false;
      replayInProgress = false;
      return null;
    }
    gif = new window.GIF({
      workers: 2,
      quality: 10,
      workerScript: "/vendor/gifjs/gif.worker.js",
      width: renderer.domElement.width,
      height: renderer.domElement.height,
      background: "#101114",
    });
    setReplayStatus("recording…", "recording");
  } else {
    setReplayStatus("previewing…");
  }

  // Lock the camera for the duration of the replay — otherwise fitToScene
  // would refit on every mesh load and the gif would jitter. The user's
  // original camera framing is restored from camSnapshot at the end.
  cameraUserMoved = true;

  // Frame the camera on the root zone bbox so the entire scene volume sits
  // inside the gif frustum. We look it up in the event snapshot rather than
  // the live scene because clearScene() just dropped every bbox helper.
  const rootBox = findRootBboxFromEvents(events);
  if (rootBox && !rootBox.isEmpty()) {
    frameBoxFully(rootBox, { padding: 1.12 });
  }

  // Start the preview stage with a blank frame so the user sees the modal
  // switch out of the placeholder state immediately.
  replayStageEl.classList.remove("show-result");
  replayStageEl.classList.remove("empty");

  for (let i = 0; i < batches.length; i++) {
    if (replayCancelRequested) break;
    const batch = batches[i];
    for (const ev of batch) dispatchForReplay(ev);
    // One render after each batch (renderer.render is also driven by the
    // animate() loop, but doing it explicitly here guarantees a fresh frame
    // before capture).
    controls.update();
    renderer.render(scene, camera);
    if (gif) {
      gif.addFrame(renderer.domElement, { copy: true, delay: frameIntervalMs });
    }
    // Mirror the freshly-rendered frame into the in-modal preview canvas
    // so the user can watch the build play out without needing to look
    // past the modal at the main viewport.
    drawReplayFrame(renderer.domElement);
    setReplayProgress(i / batches.length);
    // Yield to the browser so it can paint the preview frame and the user
    // can see progress in real time.
    await sleep(frameIntervalMs);
  }
  setReplayProgress(1);

  let blob = null;
  if (gif && !replayCancelRequested) {
    setReplayStatus("encoding gif…", "recording");
    blob = await new Promise((resolve) => {
      gif.on("finished", resolve);
      gif.render();
    });
    setReplayStatus(`gif ready · ${(blob.size / 1024).toFixed(1)} KB`, "done");
    // Swap the stage to an <img> of the encoded blob — animated gifs loop
    // by default in the browser, so the user sees the final result play.
    showReplayGifResult(blob);
  } else if (replayCancelRequested) {
    setReplayStatus("cancelled");
  } else {
    setReplayStatus("preview done", "done");
  }

  // Restore camera state.
  camera.position.copy(camSnapshot.pos);
  controls.target.copy(camSnapshot.target);
  cameraUserMoved = camSnapshot.userMoved;
  controls.update();

  replayActive = false;
  replayInProgress = false;
  // recordedEvents has been refilled by dispatchForReplay during the loop;
  // re-enable the button to reflect that.
  updateReplayButton();

  // Reconnect to the live stream so newly-arriving events flow back into
  // the buffer. The server replays the snapshot from index 0 on reconnect,
  // which re-populates `recordedEvents` for the next gif export.
  if (reconnectAfter && slotForReconnect !== null) {
    subscribe(slotEventsUrl(slotForReconnect));
  }

  // If the user clicked the close button while a replay was running, honor
  // that now that we've cleaned up.
  if (replayCancelRequested) {
    replayModalEl.classList.remove("open");
  }
  return blob;
}

// Sibling of dispatch() used during replay. The buffer was wiped before
// replay started; we refill it as we re-dispatch so the user can immediately
// run another preview/export without waiting for an SSE reconnect.
function dispatchForReplay(event) {
  if (typeof event.index === "number") {
    if (event.index <= highestEventIndex) return;
    highestEventIndex = event.index;
  }
  recordedEvents.push(event);
  appendEvent(event);
  switch (event.kind) {
    case "run.start": setStatus(`run :: ${event.model}`); break;
    case "run.done":
      if (meshErrors.size > 0) showRunCompleteWithErrors();
      else setStatus("run complete");
      break;
    case "run.error": setStatus(`error: ${event.message}`, "err"); break;
    case "run.paused": setStatus("paused"); break;
    case "mesh.error":
      meshErrors.set(event.id, event.message ?? "unknown error");
      treeSetPhase(event.id, "error");
      upsertAsset(event.id, { status: "error", errorMessage: event.message });
      break;
    case "mesh.retry":
      meshErrors.delete(event.id);
      upsertAsset(event.id, { status: "pending", errorMessage: null });
      if (treeNodes.has(event.id)) treeSetPhase(event.id, "generating_mesh");
      break;
    case "bbox":
      loadBbox(event);
      treeUpsert(event.id, {
        parentId: event.parent_id ?? null,
        prompt: event.prompt ?? null,
        kind: event.node_kind ?? "zone",
        origin: event.origin,
        dimensions: event.dimensions,
        proxyShape: event.proxy_shape ?? null,
      });
      renderTree();
      break;
    case "divider.decompose":
      for (const c of event.children ?? []) {
        treeUpsert(c.id, { parentId: event.node, prompt: c.prompt, kind: "zone" });
      }
      renderTree();
      break;
    case "divider.zone_plan":
      if (event.node && typeof event.plan === "string") {
        treeUpsert(event.node, { plan: event.plan });
      }
      break;
    case "step": treeSetPhase(event.node, event.phase); break;
    case "mesh.submit": treeSetPhase(event.id, "generating_mesh"); break;
    case "image":
      upsertAsset(event.id, { imageUrl: event.url, prompt: event.prompt });
      if (typeof event.prompt === "string") {
        treeUpsert(event.id, { imagePrompt: event.prompt });
      }
      break;
    case "model":
      loadModel(event);
      treeSetPhase(event.id, "done");
      meshErrors.delete(event.id);
      break;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function lockReplayButtons(locked) {
  replayPreviewEl.disabled = locked;
  replayRenderEl.disabled = locked;
  // Download stays disabled until a successful render completes — flip it
  // off here so an in-flight render can't be re-downloaded mid-encode, but
  // don't blindly re-enable on unlock; showReplayGifResult will.
  if (locked) replayDownloadEl.disabled = true;
}

replayPreviewEl.addEventListener("click", async () => {
  if (replayInProgress) return;
  lockReplayButtons(true);
  try {
    await runReplay({ record: false });
  } finally {
    lockReplayButtons(false);
    // Re-enable download only if we still have a previously-rendered gif.
    replayDownloadEl.disabled = lastGifBlob === null;
  }
});

replayRenderEl.addEventListener("click", async () => {
  if (replayInProgress) return;
  lockReplayButtons(true);
  try {
    await runReplay({ record: true });
  } finally {
    lockReplayButtons(false);
    replayDownloadEl.disabled = lastGifBlob === null;
  }
});

replayDownloadEl.addEventListener("click", () => {
  if (!lastGifBlob) return;
  const a = document.createElement("a");
  a.href = lastGifUrl;
  a.download = `${currentSlotId ?? "replay"}.gif`;
  a.click();
});
