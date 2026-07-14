// Event plumbing: the single SSE stream, the scene dispatch (events → 3D),
// and the observability model (events → node tree + LLM calls).

import { api } from "./api.js";

// --- SSE ----------------------------------------------------------------------

// One live stream at a time, owned by whichever panel is open. close() is
// idempotent; a terminal event closes the socket (status polls take over).
export function openStream(url, { onEvent, onTerminal } = {}) {
    const es = new EventSource(url);
    let closed = false;
    const close = () => {
        if (closed) return;
        closed = true;
        es.close();
    };
    es.onmessage = (ev) => {
        let event;
        try {
            event = JSON.parse(ev.data);
        } catch {
            return;
        }
        onEvent?.(event);
        if (
            ["run.done", "run.error", "run.paused", "run.cap_reached"].includes(
                event.kind,
            )
        ) {
            close();
            onTerminal?.(event);
        }
    };
    es.onerror = () => {
        // Browser auto-reconnects with a full snapshot replay; index dedup in the
        // consumers makes that harmless. A dead server just keeps erroring until
        // the panel is closed.
    };
    return { close };
}

// --- scene dispatch -------------------------------------------------------------

// The subset of event kinds that change the 3D scene. Mirrors the server's
// /scene projection cases.
export function dispatchSceneEvent(viewer, event) {
    switch (event.kind) {
        case "bbox":
            viewer.loadBbox(event);
            return;
        case "divider.zone_decompose":
        case "divider.decompose":
            for (const c of event.children ?? []) {
                if (c && typeof c === "object" && c.id)
                    viewer.setKind(c.id, "zone");
            }
            return;
        case "model":
            if (event.url && !viewer.hasModel(event.id)) {
                viewer.loadModel(event, api.absUrl(event.url));
            }
            return;
        default:
    }
}

// Paint the server's CQRS /scene projection (fast first paint; meshes follow
// via the bundle stream + live model events).
export function applySceneProjection(viewer, projection) {
    const nodes = projection.nodes ?? [];
    // A projection is a COMPLETE snapshot of the scene at its cut, so first prune
    // anything the viewer holds that isn't in it (handles the cut moving backward
    // — scrub/revert — which the old additive paint left stale). `mesh_url` marks
    // the nodes whose mesh is committed by this cut.
    const bboxIds = new Set();
    const meshIds = new Set();
    for (const n of nodes) {
        if (Array.isArray(n.origin) && Array.isArray(n.dimensions))
            bboxIds.add(n.id);
        if (n.mesh_url) meshIds.add(n.id);
    }
    viewer.pruneTo({ bboxIds, meshIds });
    for (const n of nodes) {
        if (n.node_kind) viewer.setKind(n.id, n.node_kind);
        if (Array.isArray(n.origin) && Array.isArray(n.dimensions)) {
            viewer.loadBbox({
                id: n.id,
                origin: n.origin,
                dimensions: n.dimensions,
                node_kind: n.node_kind ?? "zone",
                proxy_shape: n.proxy_shape,
            });
        }
    }
}

// --- observability model ----------------------------------------------------------
//
// Folds the FULL event log (history backfill + live tail) into the node tree
// the obs dock renders: nodes (id/parent/kind/prompt/phase) each carrying its
// LLM calls in execution order. Calls keep the raw event object — the logged
// system/user/output/reasoning ARE the post-injection ground truth bytes.

// --- log strip classification -------------------------------------------------
//
// The non-verbose event log: lifecycle + every error/warning, with the
// firehose kinds (cache.llm payloads, bbox/step/image/model traffic,
// provider-job bookkeeping) left out. Errors are what you open an errored
// cell to read; retries/accepted-invalid rows explain degraded output.

const LOG_BUILDERS = {
    "run.start": (e) => ["info", `run started · ${e.model ?? ""}`],
    "run.done": () => ["info", "run done"],
    "run.paused": () => ["info", "run paused"],
    "run.error": (e) => ["error", e.message ?? "run error"],
    "run.cap_reached": (e) => [
        "warn",
        `spend cap reached — $${(e.spend ?? 0).toFixed(2)} of $${(e.cap ?? 0).toFixed(2)}`,
    ],
    "run.cap_override": (e) => [
        "info",
        (e.cap ?? 0) > 0 ? `spend cap set to $${e.cap.toFixed(2)}` : "spend cap removed",
    ],
    "mesh.error": (e) => ["error", `${e.id}: ${e.message ?? "mesh error"}`],
    "mesh.retry": (e) => ["warn", `${e.id}: mesh retried`],
    "mesh.submit": null,
    "llm.json_decode_error": (e) => [
        e.final ? "error" : "warn",
        `LLM JSON decode${e.final ? " (final attempt)" : ""}${e.code != null ? ` [HTTP ${e.code}]` : ""}: ${e.reason ?? ""}`,
    ],
    "llm.retry": (e) => ["warn", `LLM parse retry: ${e.reason ?? ""}`],
    "llm.transport_retry": (e) => [
        "warn",
        `provider flap (attempt ${e.attempt ?? "?"}${e.code != null ? `, HTTP ${e.code}` : ""}, backoff ${e.backoff_s ?? "?"}s): ${e.reason ?? ""}`,
    ],
    "llm.validation_retry": (e) => [
        "warn",
        `${e.step ?? "step"}: output ids mismatched (attempt ${e.attempt ?? "?"}): ${e.reason ?? ""}`,
    ],
    "generation.decompose.retry": (e) => [
        "warn",
        `${e.zone}: decompose rejected (attempt ${e.attempt ?? "?"}): ${e.reason ?? ""}`,
    ],
    "generation.next.retry": (e) => [
        "warn",
        `${e.zone}: next-object rejected (attempt ${e.attempt ?? "?"}): ${e.reason ?? ""}`,
    ],
    "generation.decompose.accept_invalid": (e) => [
        "warn",
        `${e.zone}: accepted decompose with dangling refs: ${e.reason ?? ""}`,
    ],
    "generation.next.accept_invalid": (e) => [
        "warn",
        `${e.zone}: accepted next-object with dangling refs: ${e.reason ?? ""}`,
    ],
    "generation.next.stuck": (e) => [
        "warn",
        `${e.zone}: completion loop stuck re-proposing ${e.id ?? "?"} — stopped`,
    ],
    "generation.dedup_drop": (e) => [
        "warn",
        `${e.zone}: dropped duplicate ${e.id} (${e.reason ?? ""})`,
    ],
    "generation.decompose.no_objects": (e) => [
        "info",
        `${e.zone}: no objects needed`,
    ],
    "divider.validate.referenced_ids.accept_invalid": (e) => [
        "warn",
        `${e.node}: accepted children with dangling refs: ${e.reason ?? ""}`,
    ],
    "library.asset_missing": (e) => [
        "warn",
        `${e.id}: library asset missing (${e.library_id})`,
    ],
    "library.bounds_missing": (e) => [
        "warn",
        `${e.id}: library bounds missing (${e.library_id})`,
    ],
};

function classifyLogEvent(event) {
    const builder = LOG_BUILDERS[event.kind];
    if (builder === null) return null;
    if (builder) {
        const [severity, text] = builder(event);
        return { index: event.index, kind: event.kind, severity, text };
    }
    // Unmapped kinds: surface anything that smells like a failure, skip the rest.
    if (/\.(error|failed)$/.test(event.kind ?? "")) {
        return {
            index: event.index,
            kind: event.kind,
            severity: "error",
            text: String(event.message ?? event.reason ?? ""),
        };
    }
    return null;
}

export function createObsModel() {
    const model = {
        nodes: new Map(), // id -> node record
        order: [], // node ids in first-seen order
        calls: [], // every cache.llm event, log order
        log: [], // filtered lifecycle/error/warning entries, log order
        // id -> [{relation, call}] — the calls that brought this node into
        // existence: "emitted_by" (a decompose step that NAMED it) and "placed_by"
        // (a bbox-batch step that gave it its box). Those calls run on the node's
        // REGION (event.node), so they're bucketed on a different id in `nodes` —
        // this is the only place an object can learn the step + region that made
        // it, which matters most for negative-space objects whose region isn't
        // their structural parent.
        provenance: new Map(),
        // id -> the committed authored spec (ObjectSpec / SubregionSpec dump) from
        // `generation.decompose` / `generation.next` / `divider.zone_decompose` —
        // final ids with rebound parent / relationships, so the info panel's
        // authored fields match the scene (vs the raw call output, pre-rename).
        specs: new Map(),
        errorCount: 0,
        maxIndex: -1,
    };

    function node(id) {
        let n = model.nodes.get(id);
        if (!n) {
            n = {
                id,
                parentId: null,
                prompt: null,
                plan: null,
                imagePrompt: null,
                kind: "zone",
                phase: null,
                calls: [],
                meshUrl: null,
                error: null,
                // World-space placement, carried on the `bbox` event (origin/dimensions
                // are global; orientation is the resolved yaw in degrees, 0 for zones).
                origin: null,
                dimensions: null,
                proxyShape: null,
                orientation: null,
            };
            model.nodes.set(id, n);
            model.order.push(id);
        }
        return n;
    }

    // Scan a finished LLM call's structured output for the ids it brought into
    // existence and record provenance for each. `event.node` is the call site —
    // for a decompose / bbox-batch step that's the REGION the step ran on, which
    // is what we surface as "<step> on <region>". Only the id-bearing output
    // shapes matter: `subregions`/`objects`/`children` + single `object`
    // (decompose, named → emitted_by) and `assignments` (bbox batch, positioned →
    // placed_by); every other step (zone_plan, image_prompt, overall_bbox) names
    // no ids. `subregions` is what zone_decompose emits — tagging it is what gives
    // zones (not just objects) an emitted_by, so the emittance lineage can climb
    // region→region all the way to the root.
    function recordProvenance(event) {
        const out = event.output;
        if (!out || typeof out !== "object") return;
        const relation =
            event.step === "child_bbox_batch" ||
            event.step === "object_bbox_batch"
                ? "placed_by"
                : "emitted_by";
        const tag = (cid) => {
            if (typeof cid !== "string" || cid === event.node) return;
            const arr = model.provenance.get(cid) ?? [];
            // One entry per (relation, event index): an SSE reconnect replays the
            // same call, and a malformed output could name the same id twice.
            if (
                arr.some(
                    (e) =>
                        e.relation === relation && e.call.index === event.index,
                )
            )
                return;
            arr.push({ relation, call: event });
            model.provenance.set(cid, arr);
        };
        const tagList = (list) => {
            if (Array.isArray(list)) for (const x of list) tag(x?.id);
        };
        tagList(out.subregions);
        tagList(out.children);
        tagList(out.objects);
        tagList(out.assignments);
        if (out.object && typeof out.object === "object") tag(out.object.id);
    }

    // A committed `*.id_collision` re-keys an `emitted_by` entry from the raw id
    // the model proposed to the final id `uniquify_ids` assigned, so provenance
    // keys on the same ids as the scene (bbox / committed emissions). The
    // colliding call is the most-recent `emitted_by` on `old` before the event;
    // the pre-existing node that legitimately holds `old` keeps its own entry.
    function relocateEmitted(event) {
        const { old: from, new: to } = event;
        if (typeof from !== "string" || typeof to !== "string") return;
        const arr = model.provenance.get(from);
        if (!arr) return;
        let pick = null;
        for (const e of arr) {
            if (e.relation !== "emitted_by") continue;
            if (typeof event.index === "number" && e.call.index >= event.index)
                continue;
            if (!pick || (e.call.index ?? -1) > (pick.call.index ?? -1)) pick = e;
        }
        if (!pick) return;
        const rest = arr.filter((e) => e !== pick);
        if (rest.length) model.provenance.set(from, rest);
        else model.provenance.delete(from);
        const dst = model.provenance.get(to) ?? [];
        dst.push({ relation: "emitted_by", call: pick.call });
        model.provenance.set(to, dst);
    }

    function feed(event) {
        const idx = typeof event.index === "number" ? event.index : null;
        if (idx !== null) {
            if (idx <= model.maxIndex) return false; // SSE reconnect replay — dedupe
            model.maxIndex = idx;
        }
        const entry = classifyLogEvent(event);
        if (entry) {
            model.log.push(entry);
            if (entry.severity === "error") model.errorCount += 1;
        }
        switch (event.kind) {
            case "cache.llm": {
                model.calls.push(event);
                node(event.node ?? "(unattributed)").calls.push(event);
                recordProvenance(event);
                return true;
            }
            case "bbox": {
                const n = node(event.id);
                n.parentId = event.parent_id ?? n.parentId;
                n.prompt = event.prompt ?? n.prompt;
                n.kind = event.node_kind ?? n.kind;
                if (Array.isArray(event.origin)) n.origin = event.origin;
                if (Array.isArray(event.dimensions))
                    n.dimensions = event.dimensions;
                if (event.proxy_shape !== undefined)
                    n.proxyShape = event.proxy_shape;
                if (typeof event.orientation === "number")
                    n.orientation = event.orientation;
                return true;
            }
            case "divider.zone_decompose":
            case "divider.decompose": {
                for (const c of event.children ?? []) {
                    if (!c || typeof c !== "object" || !c.id) continue;
                    const n = node(c.id);
                    n.parentId = n.parentId ?? (c.parent || event.node);
                    n.prompt = n.prompt ?? c.prompt;
                    n.kind = "zone";
                    model.specs.set(c.id, c);
                }
                return true;
            }
            case "divider.zone_plan": {
                if (
                    typeof event.node === "string" &&
                    typeof event.plan === "string"
                ) {
                    node(event.node).plan = event.plan;
                }
                return true;
            }
            case "image": {
                if (
                    typeof event.id === "string" &&
                    typeof event.prompt === "string"
                ) {
                    node(event.id).imagePrompt = event.prompt;
                }
                return true;
            }
            case "step": {
                if (typeof event.node === "string")
                    node(event.node).phase = event.phase ?? null;
                return true;
            }
            case "model": {
                const n = node(event.id);
                n.meshUrl = event.url ?? n.meshUrl;
                n.error = null;
                return true;
            }
            case "mesh.error": {
                node(event.id).error = event.message ?? "mesh error";
                return true;
            }
            case "generation.decompose": {
                for (const o of event.objects ?? [])
                    if (o && typeof o === "object" && o.id)
                        model.specs.set(o.id, o);
                return true;
            }
            case "generation.next": {
                if (event.object && typeof event.object === "object" && event.id)
                    model.specs.set(event.id, event.object);
                return true;
            }
            case "generation.id_collision":
            case "divider.id_collision": {
                relocateEmitted(event);
                return true;
            }
            default:
                return entry !== null || idx !== null;
        }
    }

    function reset() {
        model.nodes.clear();
        model.order.length = 0;
        model.calls.length = 0;
        model.log.length = 0;
        model.provenance.clear();
        model.specs.clear();
        model.errorCount = 0;
        model.maxIndex = -1;
    }

    // Resume/retry truncates the cell's trailing run.paused / run.error sentinel
    // server-side and reuses its log index for the first new event. We already
    // folded that sentinel, so undo it before re-tailing the stream — otherwise
    // the dedup floor (`idx <= maxIndex`) swallows the first resumed event. The
    // sentinel is always the last event folded (the stream closed on it), so
    // dropping its log row and stepping maxIndex back one is exact.
    function rewindTerminal() {
        const last = model.log[model.log.length - 1];
        if (last && (last.kind === "run.paused" || last.kind === "run.error")) {
            model.log.pop();
            if (last.severity === "error")
                model.errorCount = Math.max(0, model.errorCount - 1);
            model.maxIndex -= 1;
        }
    }

    function lastError() {
        for (let i = model.log.length - 1; i >= 0; i--) {
            if (model.log[i].severity === "error") return model.log[i];
        }
        return null;
    }

    // The latest call of `templateStep` — feeds the pinned panel during
    // downstream simulation.
    function lastCallOf(templateStep) {
        for (let i = model.calls.length - 1; i >= 0; i--) {
            const c = model.calls[i];
            if (
                c.template === templateStep ||
                (!c.template && c.step === templateStep)
            )
                return c;
        }
        return null;
    }

    return { model, feed, reset, rewindTerminal, lastCallOf, lastError };
}

// The decomposition step recorded as a node's `emitted_by` provenance — the
// pass that NAMED it (anchor_decompose / next_object / negative_space_decompose
// / encapsulating_decompose). The 3D viewer keys object colors off this; the
// obs tree shows the same value as the node row's "via {step}" pill. Null when
// nothing has named the node yet (provenance not folded, or a plain zone).
export function emittedStep(model, id) {
    const e = (model.provenance?.get(id) ?? []).find(
        (p) => p.relation === "emitted_by",
    );
    return e ? (e.call.template ?? e.call.step ?? null) : null;
}

// The REGION a node was actually generated in — the call site of the decompose
// step that named it (`emitted_by`), NOT its structural parent. A couch emitted
// by next_object on `living_room` but anchored ON the floor has emitting region
// `living_room` and structural parent `floor`; the emittance trace follows the
// former. Falls back to the structural parent only when no emitted_by was
// recorded (legacy logs, or a node whose decompose output didn't name it).
export function emittingRegion(model, id) {
    const e = (model.provenance?.get(id) ?? []).find(
        (p) => p.relation === "emitted_by",
    );
    if (e && typeof e.call.node === "string") return e.call.node;
    return model.nodes.get(id)?.parentId ?? null;
}

// The emittance lineage of a node: root → … → node, climbing emitting region to
// emitting region (see `emittingRegion`). Stops at the root (no emitting region)
// or a missing/cyclic link. This is the chain the trace panel walks — every hop
// is "the region that generated the one below it", which is what we care about,
// not the structural parent chain.
export function emittanceLineage(model, focusId) {
    const chain = [focusId];
    const seen = new Set([focusId]);
    let cur = focusId;
    while (cur) {
        const region = emittingRegion(model, cur);
        if (!region || seen.has(region) || !model.nodes.has(region)) break;
        chain.unshift(region);
        seen.add(region);
        cur = region;
    }
    return chain;
}

// Slice a finished LLM call's structured output down to just the part that
// concerns `nodeId`: the emitted item (objects / subregions / children) or the
// placed item (assignments) whose id matches. Returns `{ value, truncated }` —
// `truncated` is true when a strict subset was found, false when the whole
// output is already node-specific (zone_plan, image_prompt, overall_bbox) or
// `nodeId` isn't an addressable item in it (a region's own decompose output,
// where the node is the call site, not an emitted id).
export function extractRelevantOutput(output, nodeId) {
    if (!output || typeof output !== "object" || !nodeId) {
        return { value: output, truncated: false };
    }
    for (const key of ["objects", "subregions", "children", "assignments"]) {
        const list = output[key];
        if (Array.isArray(list)) {
            const item = list.find((x) => x && x.id === nodeId);
            if (item) return { value: item, truncated: true };
        }
    }
    if (
        output.object &&
        typeof output.object === "object" &&
        output.object.id === nodeId
    ) {
        return { value: output.object, truncated: true };
    }
    return { value: output, truncated: false };
}
