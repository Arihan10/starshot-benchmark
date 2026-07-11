"""Shared scene-context renderer + span-tracking role classifier.

This is the SINGLE SOURCE for the scene-context schema experiment (soft-JSON vs
XML vs prose). It is deliberately **stdlib-only** (no other `app` imports, no
pydantic / Node types — it works on the rendered soft-JSON string and a parsed
entity tree of plain dicts), so it can be imported by:

  * the pipeline (`app.core.scene_context` — render the treated step's scene
    context in the chosen format when a schema ablation is bound),
  * the attention attribution (`app.services.teacher_forcing` — consume the
    logged span-map instead of scraping soft-JSON), and
  * the standalone verification tool (`scripts/scene_schema.py --roles`).

Two capabilities:

  1. **Emitters** — `emit_xml` / `emit_prose` render the parsed entity tree; the
     `*_roles` / `build_scene_map` variants ALSO return, in the SAME pass, the
     per-attribute role spans (context / frame / content). Because rendering and
     span extraction come from one pass, the string the model sees and the spans
     attention is scored on can never drift (this is the "log the span-map via
     the shared emitter" design — see ablation-experiments.md §9.5, Option 1).

  2. **Role classification** — per ATTRIBUTE, per FORMAT:
       * context — the field/key NAME (JSON key, XML tag; prose has none)
       * frame   — the scaffolding around the value (`:`, brackets, quotes,
                   commas, unit labels `m`/`deg`, XML markup + structural attr
                   names, prose connective grammar)
       * content — the actual value token(s)
     The split is driven by each attribute's serialized form, NOT one pattern,
     and is CHAR-SPAN based (tokenizer-agnostic): a token inherits the role of
     the span it overlaps most.

`build_scene_map` returns the entity-grouped structure the attribution pipeline
expects (mirrors `teacher_forcing`'s soft-JSON `scene_map`: entities → component
spans → role sub-spans), with component names COARSENED to match the server's
`_ENTRY_LINE_COMPONENTS` labels so the /tf attribute axis lines up across formats.
"""

from __future__ import annotations

import re
from typing import Any

# Natural-language mappings for the prose emitter.
REL_PHRASE = {
    "ON": "resting on", "BESIDE": "beside", "ABOVE": "above",
    "BELOW": "below", "ATTACHED": "attached to", "IN": "inside",
}
PARENT_KIND_PHRASE = {"ON": "on", "ATTACHED": "attached to", "IN": "inside"}


# --------------------------------------------------------------------------- #
# Brace-format parsing (soft-JSON -> entity tree)
# --------------------------------------------------------------------------- #
def find_matching(s: str, i: int) -> int:
    """Index of the `}` matching the `{` at s[i], respecting quoted strings."""
    assert s[i] == "{", "find_matching must start on '{'"
    depth = 0
    in_str = False
    while i < len(s):
        c = s[i]
        if in_str:
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces in scene context")


def inner_of(s: str) -> str:
    """Content between the first `{` and its matching `}`."""
    start = s.find("{")
    if start < 0:
        return ""
    end = find_matching(s, start)
    return s[start + 1:end]


def split_blocks(inner: str) -> list[str]:
    """The inner text of every top-level `{...}` block within `inner`."""
    blocks: list[str] = []
    i = 0
    while i < len(inner):
        if inner[i] == "{":
            j = find_matching(inner, i)
            blocks.append(inner[i + 1:j])
            i = j + 1
        else:
            i += 1
    return blocks


_FIELD_RE = re.compile(r"^([^:]+):\s*(.*)$")


def parse_fields(text: str) -> dict[str, tuple[str, str]]:
    """Parse `key: value` lines into {key: (kind, raw)} where kind is
    'scalar' or 'block' ('block' raw is the nested {...} inner text). Nested
    blocks (parent / parent_region) are consumed across lines."""
    fields: dict[str, tuple[str, str]] = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        i += 1
        s = raw_line.strip()
        if not s or s in ("{", "}"):
            continue
        m = _FIELD_RE.match(s)
        if not m:
            continue  # prose / marker line handled by the caller
        key = m.group(1).strip()
        val = m.group(2).strip()
        if val == "{":
            depth = 1
            buf: list[str] = []
            while i < len(lines) and depth > 0:
                l2 = lines[i]
                i += 1
                depth += l2.count("{") - l2.count("}")
                if depth > 0:
                    buf.append(l2)
            fields[key] = ("block", "\n".join(buf))
        else:
            fields[key] = ("scalar", val)
    return fields


def _floats(raw: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", raw)]


def scalar_str(raw: str) -> str | None:
    """A quoted or bare scalar → its string value, or None for '(none)'/empty."""
    raw = raw.strip()
    if raw in ("", "(none)", "(none - no other subregions have been planned yet)"):
        return None
    if raw.startswith('"'):
        last = raw.rfind('"')
        return raw[1:last] if last > 0 else raw[1:]
    return raw


def parse_relationships(raw: str) -> list[dict[str, str]]:
    raw = raw.strip()
    if not raw.startswith("["):
        return []
    body = raw[1:raw.rfind("]")] if "]" in raw else raw[1:]
    body = body.strip()
    if not body:
        return []
    out: list[dict[str, str]] = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            target, kind = part.split(":", 1)
            out.append({"target": target.strip(), "kind": kind.strip()})
        else:
            out.append({"target": part, "kind": ""})
    return out


def _coords(fields: dict, key: str) -> list[float] | None:
    v = fields.get(key)
    if not v or v[0] != "scalar":
        return None
    nums = _floats(v[1])
    return nums[:3] if len(nums) >= 3 else None


def _parent_block(raw: str) -> dict:
    """Parse a `parent {...}` / `parent_region {...}` sub-block."""
    sub = parse_fields(raw)
    out: dict = {}
    for k, (kind, val) in sub.items():
        if k in ("parent_id", "parent_name"):
            out["id"] = scalar_str(val)
        elif k == "parent_relationship_kind":
            out["kind"] = scalar_str(val)
        elif k in ("parent_dimensions", "parent_region_dimensions"):
            out["dimensions"] = _floats(val)[:3] or None
        elif k in ("parent_global_origin_corner",):
            out["global_origin"] = _floats(val)[:3] or None
        elif k == "parent_placement":
            out["placement"] = scalar_str(val)
    return out


def _local_origin(fields: dict) -> dict | None:
    for k, (kind, val) in fields.items():
        if k.startswith("Local origin corner"):
            m = re.search(r"relative to (\S+?)[,)]", k)
            return {
                "relative_to": m.group(1) if m else None,
                "corner": _floats(val)[:3] or None,
            }
    return None


def parse_object(inner: str) -> dict:
    fields = parse_fields(inner)
    name_raw = fields.get("Name", ("scalar", ""))[1] or fields.get("id", ("scalar", ""))[1]
    ent: dict = {
        "kind": "object",
        "id": name_raw.split()[0] if name_raw else None,
    }
    for k, (fk, val) in fields.items():
        if k == "prompt":
            ent["prompt"] = scalar_str(val)
        elif k.startswith("noun_phrase"):
            ent["noun_phrase"] = scalar_str(val)
        elif k == "placement":
            ent["placement"] = scalar_str(val)
        elif k == "relationships":
            ent["relationships"] = parse_relationships(val)
        elif k == "proxy_shape":
            ent["proxy_shape"] = scalar_str(val)
        elif k == "orientation":
            ent.setdefault("orientation", {})["description"] = scalar_str(val)
        elif k == "global yaw":
            nums = re.findall(r"-?\d+", val)
            ent.setdefault("orientation", {})["yaw"] = int(nums[0]) if nums else None
        elif k == "Dimensions":
            ent["dimensions"] = _floats(val)[:3] or None
        elif k == "parent" and fk == "block":
            ent["parent"] = _parent_block(val)
        elif k == "parent" and fk == "scalar":  # to_place object spec: "parent: id"
            ent.setdefault("parent", {})["id"] = scalar_str(val)
        elif k == "parent_relationship_kind":
            ent.setdefault("parent", {})["kind"] = scalar_str(val)
        elif k == "parent_dimensions":
            ent.setdefault("parent", {})["dimensions"] = _floats(val)[:3] or None
        elif k == "parent_global_origin_corner":
            ent.setdefault("parent", {})["global_origin"] = _floats(val)[:3] or None
        elif k == "parent_region":
            ent["parent_region"] = scalar_str(val)
        elif k == "parent_region_dimensions":
            ent["parent_region_dimensions"] = _floats(val)[:3] or None
    ent["global_origin"] = _coords(fields, "Global origin corner")
    lo = _local_origin(fields)
    if lo:
        ent["local_origin"] = lo
    return ent


def parse_region(inner: str) -> dict:
    obj_m = re.search(r"Objects placed (?:directly )?within", inner)
    scalar_end = obj_m.start() if obj_m else len(inner)
    sub_search_from = 0

    objects: list[dict] = []
    if obj_m:
        grp_start = inner.find("{", obj_m.end())
        grp_end = find_matching(inner, grp_start)
        for b in split_blocks(inner[grp_start + 1:grp_end]):
            objects.append(dispatch_entry(b))
        sub_search_from = grp_end

    sub_m = re.search(r"subregions that are present within", inner[sub_search_from:])
    subs: list[dict] = []
    if sub_m:
        base = sub_search_from + sub_m.end()
        if not obj_m:
            scalar_end = min(scalar_end, sub_search_from + sub_m.start())
        grp_start = inner.find("{", base)
        grp_end = find_matching(inner, grp_start)
        for b in split_blocks(inner[grp_start + 1:grp_end]):
            subs.append(dispatch_entry(b))

    fields = parse_fields(inner[:scalar_end])
    name_raw = fields.get("Subregion name", ("scalar", ""))[1] or fields.get("id", ("scalar", ""))[1]
    ent: dict = {
        "kind": "subregion",
        "id": name_raw.split()[0] if name_raw else None,
        "is_target": "<-- TARGET" in name_raw,
    }
    for k, (fk, val) in fields.items():
        if k == "prompt":
            ent["prompt"] = scalar_str(val)
        elif k == "description":
            ent["description"] = scalar_str(val)
        elif k == "placement":
            ent["placement"] = scalar_str(val)
        elif k == "relationships":
            ent["relationships"] = parse_relationships(val)
        elif k == "proxy_shape":
            ent["proxy_shape"] = scalar_str(val)
        elif k == "Dimensions":
            ent["dimensions"] = _floats(val)[:3] or None
        elif k == "parent_region" and fk == "block":
            ent["parent"] = _parent_block(val)
    ent["global_origin"] = _coords(fields, "Global origin corner")
    lo = _local_origin(fields)
    if lo:
        ent["local_origin"] = lo
    if objects:
        ent["objects"] = objects
    if subs:
        ent["subregions"] = subs
    return ent


def dispatch_entry(inner: str) -> dict:
    head = "\n".join(inner.split("\n")[:3])
    if "Subregion name" in head:
        return parse_region(inner)
    return parse_object(inner)


def parse_scene(text: str) -> list[dict]:
    """Parse a scene-bearing variable's value into a list of entities."""
    if not text or text.strip().startswith("{(none"):
        return []
    if "{" not in text:
        return []
    return [dispatch_entry(b) for b in split_blocks(inner_of(text))]


# --------------------------------------------------------------------------- #
# Attribute token-role classification (context / frame / content)
# --------------------------------------------------------------------------- #
# key-line prefix -> attribute name. Mirrors app/services/teacher_forcing.py
# `_ENTRY_LINE_COMPONENTS` (most specific prefix first).
ENTRY_LINE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Subregion name:", "name"),
    ("Name:", "name"),
    ("id:", "name"),
    ("noun_phrase", "noun_phrase"),
    ("prompt:", "prompt"),
    ("description:", "description"),
    ("placement:", "placement"),
    ("relationships:", "relationships"),
    ("proxy_shape:", "proxy_shape"),
    ("orientation:", "orientation"),
    ("global yaw:", "yaw"),
    ("Dimensions:", "dimensions"),
    ("Global origin corner:", "global_origin"),
    ("Local origin corner", "local_origin"),
    ("parent_region_dimensions:", "parent_region_dimensions"),
    ("parent_region:", "parent_region"),
    ("parent_relationship_kind:", "parent_relationship_kind"),
    ("parent_id:", "parent_id"),
    ("parent_dimensions:", "parent_dimensions"),
    ("parent_global_origin_corner:", "parent_global_origin_corner"),
    ("parent_name:", "parent_name"),
    ("parent_placement:", "parent_placement"),
    ("parent:", "parent"),
)

# attribute -> value serialization kind (how its CONTENT tokens are found).
COORD_COMPONENTS = frozenset({
    "dimensions", "global_origin", "local_origin",
    "parent_dimensions", "parent_region_dimensions", "parent_global_origin_corner",
})
QUOTED_COMPONENTS = frozenset({
    "prompt", "description", "noun_phrase", "placement", "orientation", "parent_placement",
})
# Attributes whose context/frame/content split is well-defined enough to chart.
WELL_DEFINED = COORD_COMPONENTS | QUOTED_COMPONENTS | frozenset({
    "relationships", "yaw", "name", "proxy_shape", "parent_region",
    "parent_relationship_kind", "parent_id", "parent_name",
})

# Server-facing coarsening of the fine component labels above onto the coarser
# `teacher_forcing._ENTRY_LINE_COMPONENTS` labels, so the /tf attribute axis (and
# the attr_role names) match between the soft-JSON scraper and the XML/prose
# span-map. (For XML/prose the emitters already use the coarse names, so this is a
# safety net + collapses the JSON-fine parent_* labels.)
_SERVER_COMPONENT = {
    "parent_id": "parent", "parent_name": "parent", "parent_dimensions": "parent",
    "parent_global_origin_corner": "parent", "parent_relationship_kind": "parent",
    "parent_placement": "parent", "parent_region_dimensions": "parent_region",
}

# Coordinate-frame ablation levels (mirror app/ablation/coord). input_repr in
# {both, local, global} decides which coordinate ORIGIN lines the model SEES.
COORD_MODE_INPUT = {
    "baseline": "both", "lg2g": "both", "l2l": "local", "g2g": "global", "g2l": "global",
}

# A refined mock tokenizer (digit runs vs letter runs vs single punctuation) — a
# closer stand-in for the real BPE than a single \w+ grab. The server uses the
# REAL HF tokenizer; the char-span roles are tokenizer-agnostic, so only printed
# token boundaries differ, never the classification.
_ROLE_TOKEN_RE = re.compile(r"\s*(?:\d+(?:\.\d+)?|[A-Za-z]+|[^\w\s])")
_NUM_SPAN_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WORD_SPAN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
_SCALAR_SPAN_RE = re.compile(r"[A-Za-z0-9_\-./]+")


def component_of(key_stripped: str) -> str | None:
    for prefix, comp in ENTRY_LINE_COMPONENTS:
        if key_stripped.startswith(prefix):
            return comp
    return None


def _content_spans(value: str, comp: str) -> list[tuple[int, int]]:
    """CONTENT char spans within `value` (offsets relative to `value`), per the
    attribute's serialized format. Whatever isn't content (and isn't the key) is
    FRAME. Unit labels (m/deg) are frame by construction — the number regex never
    matches them."""
    if comp in COORD_COMPONENTS or comp == "yaw":
        return [m.span() for m in _NUM_SPAN_RE.finditer(value)]
    if comp in QUOTED_COMPONENTS:
        a = value.find('"')
        b = value.rfind('"')
        if a >= 0 and b > a:
            return [(a + 1, b)]                       # inside the quotes
        s = value.strip()
        if s:
            i = value.find(s)
            return [(i, i + len(s))]                  # unquoted fallback
        return []
    if comp == "relationships":
        a = value.find("[")
        b = value.rfind("]")
        lo = a + 1 if a >= 0 else 0
        hi = b if b > lo else len(value)
        return [(lo + m.start(), lo + m.end()) for m in _WORD_SPAN_RE.finditer(value[lo:hi])]
    # scalar: name / proxy_shape / parent_region / parent_id / parent_relationship_kind
    return [m.span() for m in _SCALAR_SPAN_RE.finditer(value)]


def classify_line_roles(line: str) -> dict | None:
    """Split one `key: value` line into role char-spans (offsets relative to
    `line`): {component, context:[(s,e)], frame:[(s,e)], content:[(s,e)]}. None
    for a structural / non-`key: value` line (braces, prose markers)."""
    comp = component_of(line.strip())
    if comp is None:
        return None
    ci = line.find(":")
    if ci < 0:
        return None
    key_part = line[:ci]
    value = line[ci + 1:]
    vbase = ci + 1
    key_lo = len(key_part) - len(key_part.lstrip())
    paren = key_part.find("(")                        # local-origin qualifier → frame
    ctx_hi = paren if paren >= 0 else len(key_part.rstrip())
    context = [(key_lo, ctx_hi)] if ctx_hi > key_lo else []
    content = [(vbase + s, vbase + e) for (s, e) in _content_spans(value, comp)]
    claimed = bytearray(len(line))
    for s, e in context + content:
        for i in range(s, e):
            claimed[i] = 1
    frame: list[tuple[int, int]] = []
    i = key_lo
    n = len(line)
    while i < n:
        if not claimed[i] and not line[i].isspace():
            j = i
            while j < n and not claimed[j] and not line[j].isspace():
                j += 1
            frame.append((i, j))
            i = j
        else:
            i += 1
    return {"component": comp, "context": context, "frame": frame, "content": content}


def classify_text_roles(text: str, coord_mode: str = "baseline") -> list[dict]:
    """Every attribute line in a soft-JSON SCENE_CONTEXT blob, classified into
    role spans with offsets relative to `text`. `coord_mode` drops the origin
    line(s) the model would NOT see (local-only hides Global-origin; global-only
    hides Local-origin)."""
    inp = COORD_MODE_INPUT.get(coord_mode, "both")
    out: list[dict] = []
    pos = 0
    for raw in text.splitlines(keepends=True):
        base = pos
        pos += len(raw)
        line = raw.rstrip("\n")
        roles = classify_line_roles(line)
        if roles is None:
            continue
        comp = roles["component"]
        if comp == "global_origin" and inp == "local":
            continue
        if comp == "local_origin" and inp == "global":
            continue
        out.append({
            "component": comp,
            "context": [(base + s, base + e) for s, e in roles["context"]],
            "frame": [(base + s, base + e) for s, e in roles["frame"]],
            "content": [(base + s, base + e) for s, e in roles["content"]],
        })
    return out


def _role_at(spans_by_role: dict[str, list[tuple[int, int]]], ts: int, te: int) -> str | None:
    """The role a token [ts,te) belongs to = the role it overlaps the most chars
    of (deterministic tie-break by role order). None if it overlaps no role."""
    best_role, best_ov = None, 0
    for role in ("content", "context", "frame"):
        ov = 0
        for s, e in spans_by_role.get(role, ()):
            ov += max(0, min(te, e) - max(ts, s))
        if ov > best_ov:
            best_ov, best_role = ov, role
    return best_role


def assign_token_roles(text: str, line_roles: list[dict]) -> list[tuple[str, int, int, str]]:
    """Tokenize `text` (mock BPE) and tag each token (tok, start, end, role|"-").
    A token inherits the role of the char-span it overlaps most."""
    merged: dict[str, list[tuple[int, int]]] = {"context": [], "frame": [], "content": []}
    for lr in line_roles:
        for role in merged:
            merged[role].extend(lr[role])
    toks: list[tuple[str, int, int, str]] = []
    for m in _ROLE_TOKEN_RE.finditer(text):
        s = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
        e = m.end()
        if e <= s:
            continue
        toks.append((text[s:e], s, e, _role_at(merged, s, e) or "-"))
    return toks


# --------------------------------------------------------------------------- #
# Span-tracking text builder (renders + records role spans in one pass)
# --------------------------------------------------------------------------- #
class _RoleBuf:
    """Accumulates rendered text while recording, per rendered ATTRIBUTE, the
    char-spans of `context` / `content` (FRAME is derived as the non-whitespace
    complement inside the attribute's span) and, per rendered ENTITY, which
    attributes belong to it (so `scene_map()` can group by entity)."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self.n = 0
        self._attrs: list[dict] = []
        self._cur: dict | None = None
        self._entities: list[dict] = []
        self._cur_entity: dict | None = None

    def raw(self, s: str) -> None:
        if s:
            self._parts.append(s)
            self.n += len(s)

    def _emit(self, s: str) -> tuple[int, int]:
        a = self.n
        self.raw(s)
        return (a, self.n)

    def begin(self, component: str) -> None:
        self._cur = {"component": component, "start": self.n, "context": [], "content": []}

    def ctx(self, s: str) -> None:
        self._cur["context"].append(self._emit(s))

    def content(self, s: str) -> None:
        self._cur["content"].append(self._emit(s))

    def end(self) -> None:
        self._cur["end"] = self.n
        idx = len(self._attrs)
        self._attrs.append(self._cur)
        if self._cur_entity is not None:
            self._cur_entity["attr_idx"].append(idx)
        self._cur = None

    def begin_entity(self, eid: str | None, kind: str) -> None:
        self._cur_entity = {"id": eid, "kind": kind, "start": self.n, "attr_idx": []}

    def end_entity(self) -> None:
        if self._cur_entity is not None:
            self._cur_entity["end"] = self.n
            self._entities.append(self._cur_entity)
            self._cur_entity = None

    def text(self) -> str:
        return "".join(self._parts)

    def _finalize(self, text: str) -> list[dict]:
        """Per-attribute {component, start, end, context, frame, content} with
        FRAME derived (non-ws runs in [start,end] not claimed by ctx/content)."""
        out: list[dict] = []
        for c in self._attrs:
            claimed = bytearray(c["end"])
            for s, e in c["context"] + c["content"]:
                for i in range(s, e):
                    claimed[i] = 1
            frame: list[tuple[int, int]] = []
            i = c["start"]
            while i < c["end"]:
                if not claimed[i] and not text[i].isspace():
                    j = i
                    while j < c["end"] and not claimed[j] and not text[j].isspace():
                        j += 1
                    frame.append((i, j))
                    i = j
                else:
                    i += 1
            out.append({"component": c["component"], "start": c["start"], "end": c["end"],
                        "context": c["context"], "frame": frame, "content": c["content"]})
        return out

    def finish(self) -> tuple[str, list[dict]]:
        """`(text, attrs)` — the flat per-attribute view (same shape as
        classify_text_roles), for the --roles verification harness."""
        text = self.text()
        return text, [{"component": a["component"], "context": a["context"],
                       "frame": a["frame"], "content": a["content"]}
                      for a in self._finalize(text)]

    def scene_map(self) -> tuple[str, list[dict]]:
        """`(text, scene_map)` — entity-grouped, matching teacher_forcing's
        soft-JSON `scene_map`: [{id, kind, start, end, components:[{component,
        start, end, roles:[{role, start, end}]}]}]. Component names are coarsened
        to the server labels."""
        text = self.text()
        fin = self._finalize(text)
        smap: list[dict] = []
        for ent in self._entities:
            comps: list[dict] = []
            for idx in ent["attr_idx"]:
                a = fin[idx]
                comp = _SERVER_COMPONENT.get(a["component"], a["component"])
                roles = [{"role": role, "start": s, "end": e}
                         for role in ("context", "frame", "content")
                         for s, e in a[role]]
                comps.append({"component": comp, "start": a["start"], "end": a["end"], "roles": roles})
            smap.append({"id": ent["id"], "kind": ent["kind"],
                         "start": ent["start"], "end": ent["end"], "components": comps})
        return text, smap


# --------------------------------------------------------------------------- #
# Soft-schema guidance (reintroduced into XML/prose so only STRUCTURE varies)
# --------------------------------------------------------------------------- #
def _objects_note(region_id: str | None, anchor: str) -> str:
    return (f'Objects placed directly within "{region_id}" (a flat list \u2014 an object '
            f'anchored to a peer object names that peer in its {anchor} rather than '
            f'nesting beneath it)')


def _subregions_note(region_id: str | None) -> str:
    return f'Here\'s the list of subregions that are present within "{region_id}"'


def _fmt_vec(v: list[float] | None) -> str:
    if not v:
        return "?"
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def _fmt_dims(v: list[float] | None) -> str:
    if not v:
        return "? by ? by ?"
    return " by ".join(f"{x:.2f}m" for x in v)


def _article(word: str | None) -> str:
    if not word:
        return "a"
    return "an" if word.strip()[:1].lower() in "aeiou" else "a"


# --------------------------------------------------------------------------- #
# Prose emitter (span-aware). No key/scope → 2-way content vs frame, NO context.
# --------------------------------------------------------------------------- #
_PROSE_INTRO = ("Here's the list of other subregions that have been planned for this scene so "
                "far, with the objects placed inside each subregion listed inline beneath it. "
                "Treat the bounding boxes for other subregions as space that has been reserved "
                "already for that subregion — something you should not intrude on unless "
                "there's a good spatial or narrative reason.")


def _rb_dims(buf: _RoleBuf, vals: list[float] | None) -> None:
    if not vals:
        buf.raw("? by ? by ?")
        return
    for i, x in enumerate(vals):
        if i:
            buf.raw(" by ")
        buf.content(f"{x:.2f}")
        buf.raw("m")


def _rb_vec(buf: _RoleBuf, vals: list[float] | None) -> None:
    if not vals:
        buf.raw("?")
        return
    buf.raw("(")
    for i, x in enumerate(vals):
        if i:
            buf.raw(", ")
        buf.content(f"{x:.2f}")
    buf.raw(")")


def _rb_rel(buf: _RoleBuf, rels: list[dict] | None) -> None:
    if not rels:
        return
    buf.begin("relationships")
    buf.raw(" Loosely, you can say it is ")
    for i, r in enumerate(rels):
        if i:
            buf.raw(", ")
        kind = r.get("kind") or ""
        phrase = REL_PHRASE.get(kind, kind.lower())
        if phrase:
            buf.content(phrase)
            buf.raw(" ")
        buf.content(str(r.get("target") or ""))
    buf.raw(".")
    buf.end()


def _rb_parent(buf: _RoleBuf, parent: dict | None, is_object: bool) -> None:
    if not parent or not parent.get("id"):
        return
    buf.begin("parent")
    if is_object:
        anchor = PARENT_KIND_PHRASE.get(parent.get("kind") or "", "to")
        buf.raw(f" It is anchored {anchor} ")
    else:
        buf.raw(" It is parented to the region ")
    buf.content(str(parent["id"]))
    buf.raw(", which measures ")
    _rb_dims(buf, parent.get("dimensions"))
    buf.raw(" and whose minimum corner sits at ")
    _rb_vec(buf, parent.get("global_origin"))
    buf.raw(" m in world space.")
    buf.end()
    if not is_object and parent.get("placement"):
        buf.raw(f" Its parent sits {parent['placement']}.")


def _rb_prose_object(buf: _RoleBuf, o: dict, indent: str) -> None:
    buf.begin_entity(o.get("id"), "object")
    prompt = o.get("prompt") or ""
    buf.begin("name")
    buf.raw(f"{indent}This object is ")
    buf.content(str(o.get("id")))
    buf.end()
    buf.raw(" which is ")
    buf.begin("prompt")
    buf.raw(f"{_article(prompt)} ")
    buf.content(prompt)
    buf.end()
    if o.get("noun_phrase"):
        np = o["noun_phrase"]
        buf.raw("; ")
        buf.begin("noun_phrase")
        buf.raw(f"visually it is {_article(np)} ")
        buf.content(np)
        buf.end()
    buf.raw(".")
    if o.get("parent_region"):
        buf.begin("parent_region")
        buf.raw(" It sits within the subregion ")
        buf.content(str(o["parent_region"]))
        buf.end()
        if o.get("parent_region_dimensions"):
            buf.begin("parent_region")
            buf.raw(", which measures ")
            _rb_dims(buf, o["parent_region_dimensions"])
            buf.end()
        buf.raw(".")
    buf.begin("proxy_shape")
    buf.raw(" It approximately occupies the shape of a ")
    buf.content(str(o.get("proxy_shape", "BOX")))
    buf.end()
    buf.begin("dimensions")
    buf.raw(" and measures ")
    _rb_dims(buf, o.get("dimensions"))
    buf.end()
    buf.raw(".")
    if o.get("global_origin"):
        buf.begin("global_origin")
        buf.raw(" In world space its minimum corner sits at ")
        _rb_vec(buf, o["global_origin"])
        buf.raw(" m")
        buf.end()
        lo = o.get("local_origin")
        if lo and lo.get("corner"):
            buf.begin("local_origin")
            buf.raw("; measured from ")
            buf.raw(str(lo.get("relative_to")))  # ref id → frame
            buf.raw("'s minimum corner, it sits at ")
            _rb_vec(buf, lo["corner"])
            buf.raw(" m")
            buf.end()
        buf.raw(".")
    if o.get("placement"):
        buf.begin("placement")
        buf.raw(" Within the scene it sits ")
        buf.content(str(o["placement"]))
        buf.end()
        ori = o.get("orientation") or {}
        if ori.get("description"):
            buf.begin("orientation")
            buf.raw(" and is ")
            buf.content(str(ori["description"]))
            buf.end()
        if ori.get("yaw") is not None:
            buf.begin("yaw")
            buf.raw(", rotated ")
            buf.content(str(ori["yaw"]))
            buf.raw("°")
            buf.end()
        buf.raw(".")
    _rb_rel(buf, o.get("relationships"))
    _rb_parent(buf, o.get("parent"), is_object=True)
    buf.end_entity()


def _rb_prose_region(buf: _RoleBuf, r: dict, indent: str) -> None:
    buf.begin_entity(r.get("id"), "zone")  # "zone" matches teacher_forcing's kind for a Subregion
    prompt = r.get("prompt") or ""
    buf.begin("name")
    buf.raw(f"{indent}This subregion is ")
    buf.content(str(r.get("id")))
    buf.end()
    buf.raw(" which is ")
    buf.begin("prompt")
    buf.raw(f"{_article(prompt)} ")
    buf.content(prompt)
    buf.end()
    buf.raw(".")
    if r.get("description"):
        buf.begin("description")
        buf.raw(" Here is an overview of the region: ")
        buf.content(str(r["description"]))
        buf.end()
        buf.raw(".")
    buf.begin("proxy_shape")
    buf.raw(" The subregion approximately occupies the shape of a ")
    buf.content(str(r.get("proxy_shape", "BOX")))
    buf.end()
    buf.begin("dimensions")
    buf.raw(" and measures ")
    _rb_dims(buf, r.get("dimensions"))
    buf.end()
    buf.raw(".")
    if r.get("global_origin"):
        buf.begin("global_origin")
        buf.raw(" In world space its minimum corner sits at ")
        _rb_vec(buf, r["global_origin"])
        buf.raw(" m")
        buf.end()
        lo = r.get("local_origin")
        if lo and lo.get("corner"):
            buf.begin("local_origin")
            buf.raw("; measured from ")
            buf.raw(str(lo.get("relative_to")))
            buf.raw("'s minimum corner, it sits at ")
            _rb_vec(buf, lo["corner"])
            buf.raw(" m")
            buf.end()
        buf.raw(".")
    if r.get("placement"):
        buf.begin("placement")
        buf.raw(" Within the scene it sits ")
        buf.content(str(r["placement"]))
        buf.end()
        buf.raw(".")
    _rb_rel(buf, r.get("relationships"))
    _rb_parent(buf, r.get("parent"), is_object=False)
    buf.end_entity()  # region's OWN span ends before its inline children
    rid = r.get("id")
    if r.get("objects"):
        buf.raw(f"\n{indent}Below are the objects placed directly within {rid} (a flat list "
                f"\u2014 an object anchored to a peer object names that peer as its parent "
                f"rather than nesting beneath it):")
        for o in r["objects"]:
            buf.raw("\n")
            _rb_prose_object(buf, o, indent + "  ")
    if r.get("subregions"):
        buf.raw(f"\n{indent}Here's the list of subregions that are present within {rid}:")
        for sub in r["subregions"]:
            buf.raw("\n")
            _rb_prose_region(buf, sub, indent + "  ")
    buf.raw(f"\n{indent}That was all the contents within {rid}.")


def _build_prose(entities: list[dict]) -> _RoleBuf:
    buf = _RoleBuf()
    buf.raw(_PROSE_INTRO)
    for e in entities:
        buf.raw("\n\n")
        if e.get("kind") == "subregion":
            _rb_prose_region(buf, e, "")
        else:
            _rb_prose_object(buf, e, "")
    return buf


def emit_prose_roles(entities: list[dict]) -> tuple[str, list[dict]]:
    return _build_prose(entities).finish()


def emit_prose(entities: list[dict]) -> str:
    return _build_prose(entities).text()


# --------------------------------------------------------------------------- #
# XML emitter (span-aware). tag name = context; markup + structural attr-names =
# frame; text nodes + data attribute values = content.
# --------------------------------------------------------------------------- #
_XML_INDENT = "  "


def _xml_scalar(buf: _RoleBuf, indent: str, tag: str, comp: str, value: str) -> None:
    buf.raw(indent)
    buf.begin(comp)
    buf.raw("<")
    buf.ctx(tag)
    buf.raw(">")
    buf.content(str(value))
    buf.raw("</")
    buf.ctx(tag)
    buf.raw(">")
    buf.end()
    buf.raw("\n")


def _xml_vec(buf: _RoleBuf, indent: str, tag: str, comp: str,
             vals: list[float] | None, ref: str | None = None) -> None:
    if not vals:
        return
    buf.raw(indent)
    buf.begin(comp)
    buf.raw("<")
    buf.ctx(tag)
    if ref is not None:
        buf.raw(f' relative_to="{ref}"')  # attr-name + ref id → frame
    for axis, v in zip("xyz", vals):
        buf.raw(f' {axis}="')
        buf.content(f"{v:.2f}")
        buf.raw('"')
    buf.raw("/>")
    buf.end()
    buf.raw("\n")


def _xml_entity(buf: _RoleBuf, indent: str, e: dict) -> None:
    tag = "subregion" if e.get("kind") == "subregion" else "object"
    ci = indent + _XML_INDENT
    buf.begin_entity(e.get("id"), "zone" if tag == "subregion" else "object")
    buf.raw(indent)
    buf.begin("name")
    buf.raw("<")
    buf.ctx(tag)
    if e.get("id"):
        buf.raw(' id="')
        buf.content(str(e["id"]))
        buf.raw('"')
    if e.get("is_target"):
        buf.raw(' target="true"')
    buf.raw(">")
    buf.end()
    buf.raw("\n")
    for field, comp in (("prompt", "prompt"), ("description", "description"),
                        ("noun_phrase", "noun_phrase"), ("placement", "placement"),
                        ("proxy_shape", "proxy_shape"), ("parent_region", "parent_region")):
        if e.get(field):
            _xml_scalar(buf, ci, field, comp, e[field])
    _xml_vec(buf, ci, "dimensions", "dimensions", e.get("dimensions"))
    _xml_vec(buf, ci, "parent_region_dimensions", "parent_region", e.get("parent_region_dimensions"))
    _xml_vec(buf, ci, "global_origin", "global_origin", e.get("global_origin"))
    lo = e.get("local_origin")
    if lo and lo.get("corner"):
        _xml_vec(buf, ci, "local_origin", "local_origin", lo["corner"], ref=lo.get("relative_to"))
    ori = e.get("orientation") or {}
    if ori:
        buf.raw(ci)
        buf.begin("orientation")
        buf.raw("<")
        buf.ctx("orientation")
        if ori.get("description"):
            buf.raw(' description="')
            buf.content(str(ori["description"]))
            buf.raw('"')
        if ori.get("yaw") is not None:
            buf.raw(' yaw="')
            buf.content(str(ori["yaw"]))
            buf.raw('"')
        buf.raw("/>")
        buf.end()
        buf.raw("\n")
    if e.get("relationships"):
        buf.raw(ci)
        buf.begin("relationships")
        buf.raw("<")
        buf.ctx("relationships")
        buf.raw(">")
        for r in e["relationships"]:
            buf.raw("<")
            buf.ctx("relationship")
            buf.raw(' target="')
            buf.content(str(r.get("target") or ""))
            buf.raw('" kind="')
            buf.content(str(r.get("kind") or ""))
            buf.raw('"/>')
        buf.raw("</")
        buf.ctx("relationships")
        buf.raw(">")
        buf.end()
        buf.raw("\n")
    if e.get("parent"):
        p = e["parent"]
        buf.raw(ci)
        buf.begin("parent")
        buf.raw("<")
        buf.ctx("parent")
        for k in ("id", "kind", "placement"):
            if p.get(k):
                buf.raw(f' {k}="')
                buf.content(str(p[k]))
                buf.raw('"')
        if p.get("dimensions") or p.get("global_origin"):
            buf.raw(">")
            for sub_tag, sub_v in (("dimensions", p.get("dimensions")),
                                   ("global_origin", p.get("global_origin"))):
                if not sub_v:
                    continue
                buf.raw("<")
                buf.ctx(sub_tag)
                for axis, v in zip("xyz", sub_v):
                    buf.raw(f' {axis}="')
                    buf.content(f"{v:.2f}")
                    buf.raw('"')
                buf.raw("/>")
            buf.raw("</")
            buf.ctx("parent")
            buf.raw(">")
        else:
            buf.raw("/>")
        buf.end()
        buf.raw("\n")
    buf.end_entity()  # entity's OWN span ends before its inline children
    if e.get("objects"):
        buf.raw(f"{ci}<objects>\n")
        buf.raw(f"{ci}{_XML_INDENT}<!-- {_objects_note(e.get('id'), anchor='`parent` element')} -->\n")
        for o in e["objects"]:
            _xml_entity(buf, ci + _XML_INDENT, o)
        buf.raw(f"{ci}</objects>\n")
    if e.get("subregions"):
        buf.raw(f"{ci}<subregions>\n")
        buf.raw(f"{ci}{_XML_INDENT}<!-- {_subregions_note(e.get('id'))} -->\n")
        for sub in e["subregions"]:
            _xml_entity(buf, ci + _XML_INDENT, sub)
        buf.raw(f"{ci}</subregions>\n")
    buf.raw(f"{indent}</{tag}>\n")


def _build_xml(entities: list[dict]) -> _RoleBuf:
    buf = _RoleBuf()
    buf.raw("<scene_context>\n")
    for e in entities:
        _xml_entity(buf, _XML_INDENT, e)
    buf.raw("</scene_context>\n")
    return buf


def emit_xml_roles(entities: list[dict]) -> tuple[str, list[dict]]:
    return _build_xml(entities).finish()


def emit_xml(entities: list[dict]) -> str:
    return _build_xml(entities).text()


# --------------------------------------------------------------------------- #
# Public dispatch
# --------------------------------------------------------------------------- #
def render(fmt: str, raw_text: str, entities: list[dict]) -> str:
    """Render one schema variant. json = the pipeline's soft-JSON bytes VERBATIM;
    xml / prose are built from the parsed entity tree."""
    if fmt == "json":
        return raw_text
    if fmt == "xml":
        return emit_xml(entities)
    if fmt == "prose":
        return emit_prose(entities)
    raise ValueError(f"unknown format {fmt!r}")


def classify_roles_for_format(fmt: str, raw_text: str, entities: list[dict],
                              coord_mode: str = "baseline") -> tuple[str, list[dict]]:
    """(rendered_text, flat per-attribute role spans) for `fmt` — json via the raw
    line classifier, xml/prose via the span-tracking emitters. The --roles harness
    view."""
    if fmt == "json":
        return raw_text, classify_text_roles(raw_text, coord_mode)
    text, attrs = emit_xml_roles(entities) if fmt == "xml" else emit_prose_roles(entities)
    inp = COORD_MODE_INPUT.get(coord_mode, "both")
    if inp == "local":
        attrs = [a for a in attrs if a["component"] != "global_origin"]
    elif inp == "global":
        attrs = [a for a in attrs if a["component"] != "local_origin"]
    return text, attrs


def build_scene_map(fmt: str, raw_text: str, entities: list[dict],
                    coord_mode: str = "baseline") -> tuple[str, list[dict]]:
    """(rendered_text, entity-grouped scene_map) for xml/prose — the structure the
    attention attribution pipeline consumes (mirrors teacher_forcing's soft-JSON
    `scene_map`). json returns `(raw_text, [])` since the soft-JSON scraper already
    handles it. `coord_mode` drops the origin component the mode hides."""
    if fmt == "json" or not entities:
        return raw_text, []
    text, smap = (_build_xml(entities) if fmt == "xml" else _build_prose(entities)).scene_map()
    inp = COORD_MODE_INPUT.get(coord_mode, "both")
    drop = "global_origin" if inp == "local" else "local_origin" if inp == "global" else None
    if drop:
        for ent in smap:
            ent["components"] = [c for c in ent["components"] if c["component"] != drop]
    return text, smap
