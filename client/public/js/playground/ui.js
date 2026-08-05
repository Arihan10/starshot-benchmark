// Shared control widgets for the playground panel.
//
// Every knob here is something whose effect is invisible until you know what it
// does, so a control is never just a slider: it renders its label, its live
// value, a written description, and the engine default it started from. The
// description is part of the control rather than a tooltip because a tooltip you
// have to discover is a description nobody reads.
//
// One `control()` factory covers ranges, toggles and selects and returns a
// uniform handle, so callers (the lever panel, the compile panel) never branch on
// widget kind.

const el = (tag, className) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    return node;
};

function describe(spec) {
    const parts = [];
    if (spec.hint) parts.push(spec.hint);
    const shown = defaultLabel(spec);
    if (shown !== null) parts.push(`Default ${shown}.`);
    if (parts.length === 0) return null;
    const p = el("p", "desc");
    p.textContent = parts.join(" ");
    return p;
}

function defaultLabel(spec) {
    if (spec.def === undefined || spec.def === null) return null;
    if (spec.kind === "toggle") return spec.def ? "on" : "off";
    if (spec.kind === "select") {
        return spec.options.find((o) => o.value === spec.def)?.label ?? String(spec.def);
    }
    return spec.fmt ? spec.fmt(spec.def) : String(spec.def);
}

function buildRange(spec, ctl, onChange) {
    const value = el("span", "val");
    const row = el("div", "row");
    row.appendChild(Object.assign(el("label"), { textContent: spec.label }));
    row.appendChild(value);
    ctl.appendChild(row);

    const input = el("input");
    input.type = "range";
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.step;
    input.value = spec.def;
    ctl.appendChild(input);

    const read = () => (spec.int ? parseInt(input.value, 10) : parseFloat(input.value));
    const sync = () => {
        const v = read();
        value.textContent = spec.fmt ? spec.fmt(v) : String(v);
        ctl.classList.toggle("changed", v !== spec.def);
    };
    input.addEventListener("input", () => {
        sync();
        onChange(read());
    });
    return { input, read, sync };
}

function buildToggle(spec, ctl, onChange) {
    const wrap = el("label", "check");
    const input = el("input");
    input.type = "checkbox";
    input.checked = !!spec.def;
    wrap.appendChild(input);
    wrap.appendChild(document.createTextNode(spec.label));
    ctl.appendChild(wrap);

    const read = () => input.checked;
    const sync = () => ctl.classList.toggle("changed", input.checked !== !!spec.def);
    input.addEventListener("change", () => {
        sync();
        onChange(read());
    });
    return { input, read, sync };
}

function buildSelect(spec, ctl, onChange) {
    const row = el("div", "row");
    row.appendChild(Object.assign(el("label"), { textContent: spec.label }));
    ctl.appendChild(row);

    const input = el("select");
    for (const opt of spec.options) {
        const option = el("option");
        option.value = opt.value;
        option.textContent = opt.label;
        input.appendChild(option);
    }
    input.value = spec.def;
    ctl.appendChild(input);

    const read = () => input.value;
    const sync = () => ctl.classList.toggle("changed", input.value !== spec.def);
    input.addEventListener("change", () => {
        sync();
        onChange(read());
    });
    return { input, read, sync };
}

// A control bound to `spec`, calling `onChange(value)` on every user edit.
// The handle also exposes programmatic moves that DON'T re-fire onChange, which
// is what lets the adaptive controller drive the panel without feedback loops.
export function control(spec, onChange = () => {}) {
    const ctl = el("div", "ctl");
    const built =
        spec.kind === "toggle" ? buildToggle(spec, ctl, onChange)
        : spec.kind === "select" ? buildSelect(spec, ctl, onChange)
        : buildRange(spec, ctl, onChange);

    const description = describe(spec);
    if (description) ctl.appendChild(description);

    const handle = {
        spec,
        ctl,
        input: built.input,
        read: built.read,
        sync: built.sync,
        set(value) {
            if (spec.kind === "toggle") built.input.checked = !!value;
            else built.input.value = value;
            built.sync();
        },
        reset() {
            handle.set(spec.def);
            onChange(built.read());
        },
        setDisabled(disabled) {
            built.input.disabled = !!disabled;
            ctl.classList.toggle("disabled", !!disabled);
        },
        // The LOD-range knobs are bounded by the loaded asset's level count, which
        // is only known once it has streamed its manifest.
        setMax(max) {
            built.input.max = String(max);
        },
    };
    built.sync();
    return handle;
}

// A titled, collapsible section. `blurb` sits on the right of the header and says
// in a few words what the whole group is for.
export function group({ id, title, blurb, open = true }) {
    const box = el("details", "group");
    box.open = open;
    const summary = el("summary");
    const name = el("span");
    name.textContent = title;
    summary.appendChild(name);
    if (blurb) {
        const em = el("em");
        em.textContent = blurb;
        summary.appendChild(em);
    }
    box.appendChild(summary);
    const body = el("div", "body");
    body.id = `grp-${id}`;
    box.appendChild(body);
    return { box, body, summary };
}

export function button(label, { title, className, onClick } = {}) {
    const b = el("button", className);
    b.textContent = label;
    if (title) b.title = title;
    if (onClick) b.addEventListener("click", onClick);
    return b;
}

export function note(text, className = "note") {
    const p = el("p", className);
    if (text) p.textContent = text;
    return p;
}
