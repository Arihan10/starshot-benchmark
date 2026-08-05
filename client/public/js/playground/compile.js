// Compile a streamed-SOG bundle from the playground.
//
// A cell leaves the pipeline as a PLY. Streaming needs the other shape — an
// octree manifest plus per-LOD chunks — and that is a minutes-long encode, so it
// runs as a detached child on the API host (client/tools/ply-to-lod-sog.mjs, via
// POST /runs/{run}/splat/lodsog/{slot}/{model}). This panel picks the source,
// sets the bundling knobs, kicks the job off and polls it, so a splat can go from
// "trained" to "streaming in this viewer" without leaving the page.
//
// The job outlives the page: it is a detached process, so a reload re-adopts a
// build already in flight rather than starting a second one.

import { control, button, note } from "./ui.js";

const POLL_MS = 1500;

const PARAMS = [
    {
        key: "levels",
        kind: "range",
        label: "LOD levels",
        min: 2, max: 8, step: 1, int: true, def: 4,
        fmt: (v) => `${v} levels`,
        ladderOnly: true,
        hint:
            "How many levels to build when the bundler has to DECIMATE its own " +
            "ladder. Ignored when the trainer already exported one beside the PLY.",
    },
    {
        key: "ratio",
        kind: "range",
        label: "decimation ratio",
        min: 0.05, max: 0.9, step: 0.05, def: 0.35,
        fmt: (v) => `${Math.round(v * 100)}% kept`,
        ladderOnly: true,
        hint:
            "Splats each generated level keeps from the one above it, so 35% is " +
            "roughly a third the Gaussians per step. Also ignored when a trainer " +
            "ladder exists — those levels are moment-matched, which holds up far " +
            "better at distance than dropping splats does.",
    },
    {
        key: "chunk_count",
        kind: "range",
        label: "splats per chunk",
        min: 64, max: 4096, step: 64, int: true, def: 256,
        fmt: (v) => `${v}`,
        hint:
            "Target Gaussians per octree chunk, which sets the granularity of the " +
            "whole thing: small chunks refine in finer steps and follow the camera " +
            "more closely, but multiply the request count (every chunk is 6 files).",
    },
    {
        key: "chunk_extent",
        kind: "range",
        label: "chunk extent",
        min: 1, max: 64, step: 1, int: true, def: 16,
        fmt: (v) => `${v} m`,
        hint:
            "World size of a chunk. Should sit near the scale at which the scene's " +
            "detail actually varies — room-sized for an interior, larger for terrain.",
    },
    {
        key: "harmonics",
        kind: "select",
        label: "spherical harmonics",
        def: "keep",
        options: [
            { value: "keep", label: "keep every band" },
            { value: "0", label: "strip to flat colour (0)" },
            { value: "1", label: "keep band 1" },
            { value: "2", label: "keep bands 1-2" },
            { value: "3", label: "keep bands 1-3" },
        ],
        hint:
            "Drops view-dependent colour above the chosen band. Free on our splats: " +
            "stage 6 trains degree 0, so there are no higher bands to lose.",
    },
    {
        key: "node_heap",
        kind: "range",
        label: "encoder heap",
        min: 0, max: 16384, step: 512, int: true, def: 0,
        fmt: (v) => (v ? `${(v / 1024).toFixed(1)} GB` : "node default"),
        hint:
            "Memory ceiling for the encoder process. Raise it when a " +
            "many-million-Gaussian PLY dies during the merge — that failure looks " +
            "like a heap out-of-memory in the log, not a bad bundle.",
    },
];

const bytesFmt = (b) =>
    b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${(b / (1 << 10)).toFixed(0)} KB`;
const nfmt = (n) => Math.round(n).toLocaleString();

export class LodCompiler {
    // `host` is the group body to build into; `onBuilt(source, manifestUrl)` fires
    // once a bundle finishes so the caller can rescan and stream it.
    constructor({ host, apiOrigin, onBuilt }) {
        this.apiOrigin = apiOrigin;
        this.onBuilt = onBuilt;
        this.sources = [];
        this.controls = {};
        this.timer = null;

        this.picker = document.createElement("select");
        this.picker.addEventListener("change", () => this._onSelect());
        host.appendChild(this.picker);

        this.summary = note("", "note");
        host.appendChild(this.summary);

        for (const spec of PARAMS) {
            const handle = control(spec);
            this.controls[spec.key] = handle;
            host.appendChild(handle.ctl);
        }

        this.action = button("compile bundle", {
            className: "primary",
            onClick: () => this._start(),
        });
        const row = document.createElement("div");
        row.className = "field";
        this.action.style.flex = "1";
        row.appendChild(this.action);
        host.appendChild(row);

        host.appendChild(
            note("Runs as a detached process on the API host, so it survives a reload of this page."),
        );

        // Job state only. Kept separate from the static notes above because a
        // finished build triggers a catalogue rescan, and the rescan re-selects this
        // same source — so anything `_onSelect` writes here would erase the result
        // the moment it arrived.
        this.status = note("", "note");
        host.appendChild(this.status);

        this.log = document.createElement("pre");
        this.log.className = "log";
        this.log.hidden = true;
        host.appendChild(this.log);

        this.lastSelected = null;
    }

    get selected() {
        return this.sources.find((s) => s.id === this.picker.value) ?? null;
    }

    setSources(sources) {
        this.sources = sources ?? [];
        const previous = this.picker.value;
        this.picker.innerHTML = "";
        if (this.sources.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "- no trained/healed PLY found -";
            this.picker.appendChild(option);
            this.action.disabled = true;
            this.status.textContent =
                "Nothing to compile: a cell needs splat/trained.ply or splat/healed.ply on the API host.";
            return;
        }
        // Buildable first: a source outside the server's runs root is listed (so it
        // is obvious why it can't be rebuilt) but never the default choice.
        const ordered = [...this.sources].sort(
            (a, b) => Number(!a.build_path) - Number(!b.build_path),
        );
        for (const source of ordered) {
            const option = document.createElement("option");
            option.value = source.id;
            const bits = [bytesFmt(source.bytes)];
            if (source.ladder_levels > 0) bits.push(`ladder x${source.ladder_levels}`);
            if (source.bundle_url) bits.push("bundle built");
            if (!source.build_path) bits.push("read-only root");
            option.textContent = `${source.name} — ${bits.join(", ")}`;
            this.picker.appendChild(option);
        }
        if (this.sources.some((s) => s.id === previous)) this.picker.value = previous;
        this._onSelect();
    }

    _onSelect() {
        const source = this.selected;
        if (!source) return;
        const buildable = !!source.build_path;
        const hasLadder = source.ladder_levels > 0;
        // The bundler prefers a trainer-built ladder, so the decimation knobs are
        // dead in that case — say so rather than let them look effective.
        for (const spec of PARAMS) {
            this.controls[spec.key].setDisabled(!buildable || (spec.ladderOnly && hasLadder));
        }
        this.summary.textContent = !buildable
            ? source.build_note ?? "This PLY can't be compiled by this server."
            : hasLadder
                ? `This PLY has ${source.ladder_levels} trainer-built LOD level(s) beside it; ` +
                  "the bundler will use those and skip decimation."
                : "No trainer ladder beside this PLY, so the levels below will be decimated at compile time.";
        this.action.textContent = !buildable
            ? "not compilable here"
            : source.bundle_url ? "rebuild bundle" : "compile bundle";
        this.action.disabled = !buildable;
        if (this.lastSelected !== source.id) {
            this.lastSelected = source.id;
            this.log.hidden = true;
            this.status.textContent = "";
            this.status.classList.remove("bad");
        }
        // Re-adopt a build this or a previous page already started.
        if (buildable) this._poll(source, { adopt: true });
    }

    _body() {
        const harmonics = this.controls.harmonics.read();
        const heap = this.controls.node_heap.read();
        return {
            levels: this.controls.levels.read(),
            ratio: this.controls.ratio.read(),
            chunk_count: this.controls.chunk_count.read(),
            chunk_extent: this.controls.chunk_extent.read(),
            harmonics: harmonics === "keep" ? null : parseInt(harmonics, 10),
            node_heap: heap > 0 ? heap : null,
        };
    }

    _url(source) {
        return new URL(
            `${source.build_path}?which=${encodeURIComponent(source.which)}`,
            this.apiOrigin,
        );
    }

    async _start() {
        const source = this.selected;
        if (!source) return;
        this._stopPolling();
        this.action.disabled = true;
        this.status.textContent = "starting encoder…";
        try {
            const res = await fetch(this._url(source), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(this._body()),
            });
            const payload = await res.json();
            if (!res.ok) throw new Error(payload.detail ?? `${res.status} ${res.statusText}`);
            this._render(source, payload.builders?.[source.which]);
            this._schedule(source);
        } catch (e) {
            this.action.disabled = false;
            this.status.textContent = `could not start: ${e.message ?? e}`;
            this.status.classList.add("bad");
        }
    }

    _schedule(source) {
        this._stopPolling();
        this.timer = setTimeout(() => this._poll(source), POLL_MS);
    }

    async _poll(source, { adopt = false } = {}) {
        try {
            const res = await fetch(new URL(source.build_path, this.apiOrigin), {
                cache: "no-store",
            });
            if (!res.ok) return;
            const payload = await res.json();
            const state = payload.builders?.[source.which];
            // Adoption only reports a build that is genuinely in flight; a finished
            // one is already visible in the source list as "bundle built".
            if (adopt && state?.status !== "running") return;
            this._render(source, state);
            if (state?.status === "running") this._schedule(source);
        } catch {
            /* transient — the next user action re-polls */
        }
    }

    _render(source, state) {
        if (!state) return;
        this.status.classList.remove("bad");
        const running = state.status === "running";
        this.action.disabled = running;

        if (state.log_tail) {
            this.log.hidden = false;
            const lines = state.log_tail.trimEnd().split("\n");
            this.log.textContent = lines.slice(-14).join("\n");
            this.log.scrollTop = this.log.scrollHeight;
        }

        if (running) {
            this.status.textContent = `encoding on the API host (pid ${state.pid})…`;
            return;
        }
        if (state.status === "error") {
            this.status.textContent = state.error ?? "encode failed — see the log above";
            this.status.classList.add("bad");
            return;
        }
        if (state.status === "done" && state.url) {
            const s = state.summary ?? {};
            const bits = [];
            if (s.lod_levels) bits.push(`${s.lod_levels} levels`);
            if (s.splats) bits.push(`${nfmt(s.splats)} splats at full detail`);
            if (s.chunk_files) bits.push(`${s.chunk_files} chunks`);
            this.status.textContent = `bundle ready${bits.length ? ` — ${bits.join(", ")}` : ""}.`;
            this.onBuilt?.(source, new URL(state.url, this.apiOrigin).toString());
            return;
        }
        this.status.textContent = state.source_ready
            ? "not built yet."
            : "source PLY is missing on the API host.";
    }

    _stopPolling() {
        if (this.timer !== null) clearTimeout(this.timer);
        this.timer = null;
    }
}
