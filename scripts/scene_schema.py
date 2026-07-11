#!/usr/bin/env python3
"""Scene-context schema previewer for the input-format ablation.

Given a run / slot / model / step firing, this pulls the EXACT scene context
that was fed to the model at that step (the `cache.llm` event's
`variables["SCENE_CONTEXT"]`, verbatim) and produces three schema variants:

    * json   — the pipeline's own "soft JSON" bytes, passed through VERBATIM
               (byte-identical to what the model saw / what scene_context.py's
               constructors produced). It is NOT valid JSON and isn't meant to
               be — the only consumers are token-mapping and prompt insertion.
    * xml    — an XML rendering, built from the parsed entity tree
    * prose  — a natural-language rendering, built from the parsed entity tree

XML and prose are rebuilt from the same tree parsed out of the soft JSON, so
all three carry identical information — the ablation varies only the
*structure*, not the content. Everything reads what the model actually saw
rather than re-deriving it from node state.

This is a standalone, stdlib-only script (no server imports).

Two conveniences on top of the raw addressing:

    * INTERACTIVE MODE — run the script with no (or partial) selection and it
      walks you through picking the run, slot, model and step from menus built
      by scanning the runs tree and the cell's events.jsonl. Only runs / slots
      / models that actually have completions (a `cache.llm` firing) beneath
      them are offered, and runs whose name contains "__abl-" are hidden. The
      firing menu itself lists only firings that carry scene context (the
      image_prompt / library_match / etc. calls are dropped). Anything you pass
      on the command line is used as-is and skips the corresponding prompt (and
      an explicit --step can still target any firing, scene-bearing or not).
    * FILE OUTPUT — by default the rendered schemas are written to files in a
      temp directory (one per format) and only the paths are printed, instead
      of dumping everything to the terminal. Use --stdout for the old behavior.

Addressing (see --list to discover firings):
    cell   = <runs>/<run>/<slot>/<model>/events.jsonl
    slot   = the scene (e.g. "modern-house")
    step   = the pipeline step / template (e.g. "object_bbox_batch")
    region = the step's target `node` id (optional filter)

Examples:
    # fully interactive — pick everything from menus, results saved to files
    scene_schema.py

    # partially interactive — run is fixed, pick the rest from menus
    scene_schema.py --run MINGLET

    # classic, fully addressed (still writes files unless --stdout)
    scene_schema.py --run MINGLET --slot modern-house --model gemma \
        --step object_bbox_batch --node root --format all

    scene_schema.py --run MINGLET --slot modern-house --model gemma \
        --step child_bbox_batch --which to_place --format prose --stdout

    scene_schema.py --run MINGLET --slot modern-house --model gemma --list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = REPO_ROOT / "runs"

# Scene-bearing variables this tool can reformat (all share the same entry
# grammar produced by app/core/scene_context.py).
WHICH_VARS = {
    "scene_context": "SCENE_CONTEXT",
    "root_objects": "ROOT_OBJECTS",
    "zone_objects": "ZONE_OBJECTS",
    "adjacent_zones": "ADJACENT_ZONES",
    "to_place": "TO_PLACE",
}

# Natural-language mappings for the prose emitter.
REL_PHRASE = {
    "ON": "resting on", "BESIDE": "beside", "ABOVE": "above",
    "BELOW": "below", "ATTACHED": "attached to", "IN": "inside",
}
PARENT_KIND_PHRASE = {"ON": "on", "ATTACHED": "attached to", "IN": "inside"}

# All three are plain text blobs (token-mapping + prompt insertion only). The
# "json" one is the soft-JSON that is NOT valid JSON, so it is saved as .txt too.
FORMAT_EXT = {"json": "json.txt", "xml": "xml", "prose": "prose.txt"}

# Runs whose name contains this marker are hidden from interactive selection
# (they can still be addressed explicitly with --run).
RUN_EXCLUDE = "__abl-"


# --------------------------------------------------------------------------- #
# Event loading
# --------------------------------------------------------------------------- #
def cell_dir(runs: Path, run: str, slot: str, model: str) -> Path:
    return runs / run / slot / model


def load_events(events_path: Path) -> list[dict]:
    if not events_path.is_file():
        raise SystemExit(f"no events.jsonl at {events_path}")
    out: list[dict] = []
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def llm_firings(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("kind") == "cache.llm"]


def find_firing(events: list[dict], step: str, node: str | None, occurrence: int) -> dict:
    """Pick a cache.llm firing by step (template, falling back to step) and
    optional target node. `occurrence` indexes the matches (default -1 = last)."""
    matches = [
        e for e in llm_firings(events)
        if (e.get("template") == step or e.get("step") == step)
        and (node is None or e.get("node") == node)
    ]
    if not matches:
        raise SystemExit(
            f"no cache.llm firing for step={step!r}"
            + (f" node={node!r}" if node else "")
            + " (try --list)"
        )
    try:
        return matches[occurrence]
    except IndexError:
        raise SystemExit(
            f"occurrence {occurrence} out of range: {len(matches)} firing(s) match"
        )


# --------------------------------------------------------------------------- #
# Brace-format parsing
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
# For the "structure-vs-content attention" spider we split every scene-context
# ATTRIBUTE (one rendered `key: value` line) into three token roles:
#
#   * context — the key/field NAME itself ("placement", "Dimensions",
#     "Global origin corner"). What the value is labelled as.
#   * frame   — the structural scaffolding around the value: the colon,
#     brackets / braces / parens, commas, quotes, and the fixed unit labels
#     (m, deg), plus any descriptive parenthetical on the key (local-origin's
#     "(relative to X, measured from its min corner)").
#   * content — the actual value token(s): the numbers, the prose, the
#     relationship targets/kinds, the enum value, the id.
#
# The split is PER ATTRIBUTE, driven by that attribute's serialized format (see
# app/core/scene_context.py) — NOT one hard-coded pattern. Every rule here is
# CHAR-SPAN based and therefore tokenizer-agnostic: a token inherits the role of
# the span it overlaps most (see `assign_token_roles`). This module is the
# CANONICAL design + verification harness; app/services/teacher_forcing.py (the
# server) mirrors these exact rules to emit the per-role token spans the worker
# reduces attention onto, so both agree. Run `--roles` to print the split for a
# real firing and confirm it against the schema by eye.
#
# NB: the local-vs-global coordinate ablation renders every field as `key: value`
# (never XML-tagged), so only the `key: value` classification applies here.

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
# Nested `parent` / `parent_region` BLOCK lines (the `{`-opening ones) are not a
# single value line, so they're excluded; their scalar/coord sub-fields above
# (parent_id, parent_dimensions, …) are classified on their own.
WELL_DEFINED = COORD_COMPONENTS | QUOTED_COMPONENTS | frozenset({
    "relationships", "yaw", "name", "proxy_shape", "parent_region",
    "parent_relationship_kind", "parent_id", "parent_name",
})

# Coordinate-frame ablation levels (mirror app/ablation/coord). input_repr in
# {both, local, global} decides which coordinate ORIGIN lines the model SEES.
COORD_MODE_INPUT = {
    "baseline": "both", "lg2g": "both", "l2l": "local", "g2g": "global", "g2l": "global",
}

# A refined mock tokenizer that separates digit runs from letter runs (so
# "90deg" -> "90","deg" and "(0.50," -> "(","0.50",",") — a closer stand-in for
# the real Gemma/Qwen BPE than a single \w+ grab. The server uses the REAL HF
# tokenizer; the char-span roles are tokenizer-agnostic, so only the printed
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
    # Component is keyed on the FULL line (the prefix table includes the colon,
    # e.g. "placement:"), mirroring teacher_forcing._line_component.
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
    # frame = every remaining NON-whitespace char from the key start onward,
    # grouped into contiguous runs (the colon, parens, commas, quotes, units).
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
    """Every attribute line in a SCENE_CONTEXT blob, classified into role spans
    with offsets relative to `text`. `coord_mode` drops the origin line(s) the
    model would NOT see (local-only hides Global-origin; global-only hides
    Local-origin) so the printed split matches what that condition renders."""
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
    # Merge all line role-spans into one per-role list for overlap tests.
    merged: dict[str, list[tuple[int, int]]] = {"context": [], "frame": [], "content": []}
    for lr in line_roles:
        for role in merged:
            merged[role].extend(lr[role])
    toks: list[tuple[str, int, int, str]] = []
    for m in _ROLE_TOKEN_RE.finditer(text):
        s = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))  # drop leading ws
        e = m.end()
        if e <= s:
            continue
        toks.append((text[s:e], s, e, _role_at(merged, s, e) or "-"))
    return toks


# --------------------------------------------------------------------------- #
# Span-tracking emitters (the Option-1 attribution prototype)
# --------------------------------------------------------------------------- #
# For XML and prose there is no line-prefix `key: value` grammar to scrape, so we
# record the role char-spans AS WE RENDER: the same pass produces the string the
# model sees AND the spans attention is scored on (no fragile re-parse — prose in
# particular cannot be parsed back). `_RoleBuf` is that recorder.
#
# Per rendered ATTRIBUTE occurrence you `begin(component)`, then emit its pieces:
#   * `ctx(s)`     — the field NAME (xml tag name; prose has none)
#   * `content(s)` — the actual value token(s)
#   * `raw(s)`     — scaffolding/grammar/units (becomes FRAME) or, outside a
#                    begin/end pair, pure connective glue that belongs to no field
# then `end()`. FRAME is DERIVED (never hand-tagged): every non-whitespace char in
# the attribute's [begin,end] span not claimed by context/content — mirroring the
# JSON `classify_line_roles` frame rule, so the three formats agree by construction.
class _RoleBuf:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self.n = 0
        self._attrs: list[dict] = []
        self._cur: dict | None = None

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
        self._attrs.append(self._cur)
        self._cur = None

    def finish(self) -> tuple[str, list[dict]]:
        """`(text, attrs)` where each attr = {component, context, frame, content}
        with absolute char offsets into `text` — same shape as classify_text_roles."""
        text = "".join(self._parts)
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
            out.append({"component": c["component"], "context": c["context"],
                        "frame": frame, "content": c["content"]})
        return text, out


# --------------------------------------------------------------------------- #
# Emitters (all consume the parsed entity list)
# --------------------------------------------------------------------------- #
# Soft-schema guidance. The pipeline's "soft JSON" scene context carries
# natural-language guidance lines (see app/core/scene_context.py). The JSON
# variant keeps them verbatim (it IS the pipeline bytes — see render()); we
# reintroduce the SAME guidance into XML (as comments) and prose (as sentences)
# so all three carry identical information. The anchor noun is adapted per
# format ("element" for XML, plain "parent" in the prose sentence).
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


_PROSE_INTRO = ("Here's the list of other subregions that have been planned for this scene so "
                "far, with the objects placed inside each subregion listed inline beneath it. "
                "Treat the bounding boxes for other subregions as space that has been reserved "
                "already for that subregion — something you should not intrude on unless "
                "there's a good spatial or narrative reason.")


# In PROSE there is no key/scope, so (per the design) each attribute is a 2-way
# content-vs-frame split with NO context: `content()` = the slotted value(s);
# every other non-whitespace char inside the attribute's clause = FRAME (the
# introducing phrase, unit words 'm'/'°'/'by', punctuation). Connective glue
# between clauses is `raw()` OUTSIDE a begin/end pair (belongs to no attribute).
def _rb_dims(buf: _RoleBuf, vals: list[float] | None) -> None:
    """`_fmt_dims` body — numbers CONTENT, the 'm'/'by' unit words FRAME."""
    if not vals:
        buf.raw("? by ? by ?")
        return
    for i, x in enumerate(vals):
        if i:
            buf.raw(" by ")
        buf.content(f"{x:.2f}")
        buf.raw("m")


def _rb_vec(buf: _RoleBuf, vals: list[float] | None) -> None:
    """`_fmt_vec` body — numbers CONTENT, the parens/commas FRAME."""
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
    """`_rel_clause` — the relationship phrase + target are BOTH content (parity
    with JSON's `[…]` word scrape); the connective sentence is frame."""
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
    """`_parent_clause` — the parent id + its dims/origin numbers are content; the
    anchor phrase ('on'/'inside'/…) and connective grammar are frame."""
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
        buf.raw(f" Its parent sits {parent['placement']}.")  # parent's own placement → glue


def _rb_prose_object(buf: _RoleBuf, o: dict, indent: str) -> None:
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


def _rb_prose_region(buf: _RoleBuf, r: dict, indent: str) -> None:
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
            buf.raw(str(lo.get("relative_to")))  # ref id → frame
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


def emit_prose_roles(entities: list[dict]) -> tuple[str, list[dict]]:
    """Prose rendering + per-attribute role spans (the Option-1 span-tracking
    emitter). `emit_prose` returns just the text."""
    buf = _RoleBuf()
    buf.raw(_PROSE_INTRO)
    for e in entities:
        buf.raw("\n\n")
        if e.get("kind") == "subregion":
            _rb_prose_region(buf, e, "")
        else:
            _rb_prose_object(buf, e, "")
    return buf.finish()


def emit_prose(entities: list[dict]) -> str:
    return emit_prose_roles(entities)[0]


_XML_INDENT = "  "


# In XML: the element TAG NAME (open + close occurrences) = context; the markup
# (`< > / = "`) plus every structural attribute-NAME (`x`/`y`/`z`, `target`,
# `kind`, `description`, `yaw`, `relative_to`, `id`) = frame; text nodes and DATA
# attribute values = content. A referencing id (`relative_to`) is FRAME (parity
# with the JSON local-origin qualifier), so it is emitted with `raw()`.
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
        buf.raw(f' {axis}="')             # axis name + `="` → frame
        buf.content(f"{v:.2f}")
        buf.raw('"')
    buf.raw("/>")
    buf.end()
    buf.raw("\n")


def _xml_entity(buf: _RoleBuf, indent: str, e: dict) -> None:
    tag = "subregion" if e.get("kind") == "subregion" else "object"
    ci = indent + _XML_INDENT
    # entity open tag carries the id — the NAME attribute (tag=context, id=content)
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
    buf.raw(f"{indent}</{tag}>\n")  # entity close tag → structural glue


def emit_xml_roles(entities: list[dict]) -> tuple[str, list[dict]]:
    """XML rendering + per-attribute role spans (the Option-1 span-tracking
    emitter). `emit_xml` returns just the text."""
    buf = _RoleBuf()
    buf.raw("<scene_context>\n")
    for e in entities:
        _xml_entity(buf, _XML_INDENT, e)
    buf.raw("</scene_context>\n")
    return buf.finish()


def emit_xml(entities: list[dict]) -> str:
    return emit_xml_roles(entities)[0]


# key-line prefix classifier (JSON) vs span-tracking emitters (XML/prose): one
# entry point returning (rendered_text, per-attribute role spans) for any format.
def classify_roles_for_format(fmt: str, raw_text: str, entities: list[dict],
                              coord_mode: str = "baseline") -> tuple[str, list[dict]]:
    if fmt == "json":
        return raw_text, classify_text_roles(raw_text, coord_mode)
    text, attrs = emit_xml_roles(entities) if fmt == "xml" else emit_prose_roles(entities)
    # Mirror the coord-ablation origin drop so a schema×coord cell classifies only
    # what that condition renders (baseline keeps both).
    inp = COORD_MODE_INPUT.get(coord_mode, "both")
    if inp == "local":
        attrs = [a for a in attrs if a["component"] != "global_origin"]
    elif inp == "global":
        attrs = [a for a in attrs if a["component"] != "local_origin"]
    return text, attrs


def render(fmt: str, raw_text: str, entities: list[dict]) -> str:
    """Render one schema variant. JSON is the pipeline's own soft-JSON bytes
    passed through VERBATIM (byte-identical to what scene_context.py's
    constructors produced and what the model saw); XML and prose are built from
    the parsed entity tree."""
    if fmt == "json":
        return raw_text
    if fmt == "xml":
        return emit_xml(entities)
    if fmt == "prose":
        return emit_prose(entities)
    raise ValueError(f"unknown format {fmt!r}")


def formats_for(fmt: str) -> list[str]:
    return ["json", "xml", "prose"] if fmt == "all" else [fmt]


# --------------------------------------------------------------------------- #
# Discovery (for interactive mode)
# --------------------------------------------------------------------------- #
def _subdirs(p: Path) -> list[Path]:
    return sorted((c for c in p.iterdir() if c.is_dir()), key=lambda c: c.name) if p.is_dir() else []


def list_runs(runs_dir: Path) -> list[Path]:
    return _subdirs(runs_dir)


def list_slots(runs_dir: Path, run: str) -> list[Path]:
    return _subdirs(runs_dir / run)


def list_models(runs_dir: Path, run: str, slot: str) -> list[Path]:
    return _subdirs(runs_dir / run / slot)


def cell_has_completions(events_path: Path) -> bool:
    """True if this cell's events.jsonl records at least one LLM completion
    (a `cache.llm` firing). Streams the file and early-exits on the first hit."""
    if not events_path.is_file():
        return False
    try:
        with events_path.open(encoding="utf-8") as f:
            for line in f:
                if '"cache.llm"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("kind") == "cache.llm":
                    return True
    except OSError:
        return False
    return False


def models_with_completions(runs_dir: Path, run: str, slot: str) -> list[Path]:
    return [m for m in list_models(runs_dir, run, slot)
            if cell_has_completions(m / "events.jsonl")]


def slots_with_completions(runs_dir: Path, run: str) -> list[Path]:
    return [s for s in list_slots(runs_dir, run)
            if models_with_completions(runs_dir, run, s.name)]


def runs_with_completions(runs_dir: Path) -> list[Path]:
    """Runs that (a) don't match RUN_EXCLUDE and (b) have at least one cell
    with completions somewhere beneath them."""
    out: list[Path] = []
    for r in list_runs(runs_dir):
        if RUN_EXCLUDE in r.name:
            continue
        if slots_with_completions(runs_dir, r.name):
            out.append(r)
    return out


def _latest_idx(paths: list[Path]) -> int | None:
    """Index of the most-recently-modified path (handy default on Enter)."""
    if not paths:
        return None
    try:
        mtimes = [p.stat().st_mtime for p in paths]
    except OSError:
        return None
    return max(range(len(paths)), key=lambda i: mtimes[i])


# --------------------------------------------------------------------------- #
# Interactive prompting
# --------------------------------------------------------------------------- #
def _choose(title: str, labels: list[str], default_idx: int | None = None) -> int:
    """Render a numbered menu and return the selected 0-based index.

    A single option is auto-selected. Enter accepts the default (marked)."""
    if len(labels) == 1:
        print(f"\n{title}\n     → {labels[0]}   (only option)")
        return 0
    print(f"\n{title}")
    for i, lab in enumerate(labels, 1):
        tag = "   ← default" if default_idx is not None and i - 1 == default_idx else ""
        print(f"  {i:>3}) {lab}{tag}")
    hint = f" [{default_idx + 1}]" if default_idx is not None else ""
    while True:
        try:
            raw = input(f"  choose 1-{len(labels)}{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\naborted.")
        if not raw and default_idx is not None:
            return default_idx
        if raw.isdigit() and 1 <= int(raw) <= len(labels):
            return int(raw) - 1
        print(f"    please enter a number between 1 and {len(labels)}")


def resolve_dir_choice(kind: str, provided: str | None,
                       all_options: list[Path], menu_options_fn, interactive: bool) -> str:
    """Resolve a run / slot / model.

    An explicitly provided value is validated against everything on disk
    (`all_options`) so ablation / empty cells stay addressable. When nothing is
    provided, the interactive menu is built from `menu_options_fn()` — the
    filtered set (has completions, not RUN_EXCLUDE). The filtered scan is lazy
    so fully-addressed CLI runs never pay for it."""
    all_names = [p.name for p in all_options]
    if provided is not None:
        if provided in all_names:
            return provided
        raise SystemExit(
            f"{kind} {provided!r} not found; available: {', '.join(all_names) or '(none)'}"
        )
    menu_options = menu_options_fn()
    if not menu_options:
        extra = f" (after excluding '{RUN_EXCLUDE}')" if kind == "run" else ""
        raise SystemExit(f"no {kind}s with completions found{extra}")
    if not interactive:
        raise SystemExit(
            f"--{kind} is required (or run in a terminal for interactive select). "
            f"available: {', '.join(p.name for p in menu_options)}"
        )
    idx = _choose(f"Select {kind}:", [p.name for p in menu_options],
                  default_idx=_latest_idx(menu_options))
    return menu_options[idx].name


def _var_present(val: str | None) -> bool:
    return bool(val) and not val.strip().startswith("{(none")


def _has_scene_context(e: dict) -> bool:
    return _var_present((e.get("variables") or {}).get("SCENE_CONTEXT"))


def _firing_scene_vars(e: dict) -> list[str]:
    """WHICH_VARS keys that this firing actually carries a value for."""
    vs = e.get("variables") or {}
    return [key for key, var in WHICH_VARS.items() if _var_present(vs.get(var))]


def _firing_label(e: dict) -> str:
    tmpl = e.get("template") or e.get("step") or "?"
    node = e.get("node") or "?"
    idx = e.get("index", "?")
    mark = "scene" if _has_scene_context(e) else "  -  "
    return f"[{mark}] #{idx:<4} {tmpl:<26} node={node}"


def resolve_firing_interactive(events: list[dict]) -> dict:
    firings = llm_firings(events)
    if not firings:
        raise SystemExit("no cache.llm firings in this cell (nothing to preview)")

    # Only offer firings that actually fed the model scene context; drop the
    # image_prompt / library_match / etc. calls that carry none.
    pool = [e for e in firings if _has_scene_context(e)]
    note = ""
    if not pool:  # fall back so the user is never left with an empty menu
        pool = [e for e in firings if _firing_scene_vars(e)]
        note = "  (no SCENE_CONTEXT firings — showing any with scene data)"
    if not pool:
        pool = firings
        note = "  (no scene-bearing firings — showing all)"

    labels = [_firing_label(e) for e in pool]
    idx = _choose("Select step / firing:" + note, labels, default_idx=len(pool) - 1)
    return pool[idx]


def resolve_which(provided: str | None, interactive: bool) -> str:
    opts = sorted(WHICH_VARS)
    if provided:
        return provided
    if not interactive:
        return "scene_context"
    idx = _choose("Which scene variable to reformat:", opts,
                  default_idx=opts.index("scene_context"))
    return opts[idx]


def resolve_format(provided: str | None, interactive: bool) -> str:
    opts = ["all", "json", "xml", "prose"]
    if provided:
        return provided
    if not interactive:
        return "all"
    idx = _choose("Output format:", opts, default_idx=0)
    return opts[idx]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _slug(x: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_") or "x"


def save_outputs(out_dir: Path, base: str, fmt: str, raw_text: str,
                 entities: list[dict]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for f in formats_for(fmt):
        # Written VERBATIM (no trailing newline added) so the json variant stays
        # byte-identical to the pipeline's SCENE_CONTEXT — matters for both
        # token-mapping and prompt insertion.
        body = render(f, raw_text, entities)
        path = out_dir / f"{base}.{FORMAT_EXT[f]}"
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_list(events: list[dict]) -> None:
    print(f"{'idx':>5}  {'step/template':<28} {'node':<24} scene?")
    print("-" * 70)
    for e in llm_firings(events):
        idx = e.get("index", "?")
        tmpl = e.get("template") or e.get("step") or "?"
        node = e.get("node") or "?"
        sc = (e.get("variables") or {}).get("SCENE_CONTEXT") or ""
        has = "yes" if sc and not sc.strip().startswith("{(none") else "no"
        print(f"{idx:>5}  {tmpl:<28} {node:<24} {has}")


def _spans_text(text: str, spans: list[tuple[int, int]]) -> str:
    return "  ".join(repr(text[s:e]) for s, e in spans) or "-"


def cmd_roles(fmt: str, raw_text: str, entities: list[dict], coord_mode: str) -> None:
    """Print the per-attribute context/frame/content token-role split for one
    firing's scene context in `fmt` (json / xml / prose), for manual confirmation
    against that format's actual serialization. This is the classification the
    server mirrors to measure structure-vs-content attention. JSON is classified
    from the raw pipeline bytes (`classify_text_roles`); XML/prose from the
    span-tracking emitters (`emit_*_roles`) — the Option-1 attribution prototype."""
    if fmt == "json" and (not raw_text or "{" not in raw_text or raw_text.strip().startswith("{(none")):
        print(f"\n##### FORMAT = {fmt} #####\n(no scene context in this firing)")
        return
    if fmt != "json" and not entities:
        print(f"\n##### FORMAT = {fmt} #####\n(no entities parsed from this firing)")
        return
    text, line_roles = classify_roles_for_format(fmt, raw_text, entities, coord_mode)
    inp = COORD_MODE_INPUT.get(coord_mode, "both")
    print(f"\n##### FORMAT = {fmt}   coord_mode={coord_mode}  (input representation = {inp}; "
          f"classified {len(line_roles)} attribute occurrence(s)) #####\n")

    # One representative split per unique attribute (first occurrence), so the
    # rules can be eyeballed without dumping every line.
    print("=" * 80)
    print("PER-ATTRIBUTE SPLIT (one example each)   context = key/tag name | frame = scaffolding+units | content = value")
    if fmt == "prose":
        print("(prose has no key/scope → context is empty by design; content vs surrounding grammar)")
    print("=" * 80)
    seen: set[str] = set()
    for lr in line_roles:
        comp = lr["component"]
        if comp in seen:
            continue
        seen.add(comp)
        note = "" if comp in WELL_DEFINED else "   [not charted — split ill-defined]"
        print(f"\n• {comp}{note}")
        print(f"    context: {_spans_text(text, lr['context'])}")
        print(f"    frame  : {_spans_text(text, lr['frame'])}")
        print(f"    content: {_spans_text(text, lr['content'])}")

    # Per-attribute token counts (mock BPE) — the granularity attention is scored
    # on. Build a per-char (component, role) label ONCE, then attribute each token
    # to the (component, role) it covers the most chars of. O(chars + tokens).
    from collections import defaultdict
    n = len(text)
    comp_lab: list[str | None] = [None] * n
    role_lab: list[str | None] = [None] * n
    for lr in line_roles:
        comp = lr["component"]
        for role in ("context", "frame", "content"):
            for s, e in lr[role]:
                for i in range(s, min(e, n)):
                    comp_lab[i] = comp
                    role_lab[i] = role
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"context": 0, "frame": 0, "content": 0})
    for m in _ROLE_TOKEN_RE.finditer(text):
        s = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
        e = m.end()
        if e <= s:
            continue
        tally: dict[tuple[str, str], int] = {}
        best, best_n = None, 0
        for i in range(s, e):
            c, r = comp_lab[i], role_lab[i]
            if c is None or r is None:
                continue
            k = (c, r)
            tally[k] = tally.get(k, 0) + 1
            if tally[k] > best_n:
                best_n, best = tally[k], k
        if best is not None:
            counts[best[0]][best[1]] += 1

    print("\n" + "=" * 80)
    print("AGGREGATE per attribute   (mock-BPE token counts; server uses the real HF tokenizer)")
    print("=" * 80)
    hdr = f"{'attribute':<28}{'context':>8}{'frame':>8}{'content':>9}{'frame%':>9}{'chart?':>8}"
    print(hdr)
    print("-" * len(hdr))
    for comp in sorted(counts):
        c = counts[comp]
        denom = c["frame"] + c["content"]
        fs = (100.0 * c["frame"] / denom) if denom else 0.0
        print(f"{comp:<28}{c['context']:>8}{c['frame']:>8}{c['content']:>9}{fs:>8.1f}%"
              f"{('yes' if comp in WELL_DEFINED else 'no'):>8}")
    print("\nThe context / frame / content columns are each role's TOKEN COUNT — the LENGTH the")
    print("server normalizes by. The measured quantity per role is:")
    print("    density = (attention mass on that role's tokens) / (that role's token count)")
    print("i.e. length-normalized mean per-token attention, computed for context, frame AND content")
    print("(so a role with many tokens isn't credited just for being long). frame% = frame/(frame+")
    print("content) is shown only as a quick structural-share sanity check.")


# --------------------------------------------------------------------------- #
# Classification self-test (--roles-test)
# --------------------------------------------------------------------------- #
# A synthetic soft-JSON object exercising every charted attribute, parsed once,
# then classified in all three formats. Asserts the TRICKY per-attribute rules
# (the ones easy to get wrong) plus a role-disjointness invariant. No run needed.
_SELFTEST_OBJECT_LINES = (
    "Name: obj_id_x",
    'prompt: "unique prompt phrase"',
    'noun_phrase/description: "unique noun phrase"',
    "parent: {",
    "parent_id: parent_id_x",
    "parent_relationship_kind: ON",
    "parent_dimensions: (1.00, 2.00, 3.00) m",
    "parent_global_origin_corner: (0.10, 0.20, 0.30) m",
    "}",
    "parent_region: region_id_x",
    "parent_region_dimensions: (10.00, 11.00, 12.00) m",
    'placement: "unique placement text"',
    "relationships: [rel_target_id: BESIDE]",
    "proxy_shape: BOX",
    'orientation: "facing north"',
    "global yaw: 45deg",
    "Dimensions: (1.11, 2.22, 3.33) m",
    "Global origin corner: (4.44, 5.55, 6.66) m",
    "Local origin corner (relative to rel_ref_id, measured from its min corner): (7.77, 8.88, 9.99) m",
)


def _roles_selftest() -> int:
    obj_block = "{\n" + "\n".join(_SELFTEST_OBJECT_LINES) + "\n}"
    syn_json = "{\n" + obj_block + "\n}"
    entities = parse_scene(syn_json)
    problems: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            problems.append(msg)

    # parse round-trip (guards the hand-crafted fixture)
    print("# parse_scene round-trip")
    ok_parse = (len(entities) == 1 and entities[0].get("id") == "obj_id_x")
    check(ok_parse, "one object parsed with id=obj_id_x")
    if ok_parse:
        o = entities[0]
        check(o.get("dimensions") == [1.11, 2.22, 3.33], "dimensions parsed = [1.11, 2.22, 3.33]")
        check((o.get("local_origin") or {}).get("relative_to") == "rel_ref_id", "local_origin.relative_to = rel_ref_id")
        rels = o.get("relationships") or [{}]
        check(rels[0].get("target") == "rel_target_id" and rels[0].get("kind") == "BESIDE", "relationship = rel_target_id:BESIDE")
    if not ok_parse:
        print("\nSELF-TEST ABORTED — fixture did not parse")
        return 1

    def texts(text: str, spans: list[tuple[int, int]]) -> list[str]:
        return [text[s:e] for s, e in spans]

    def first_attr(attrs: list[dict], comp: str, must_contain: str | None = None):
        for a in attrs:
            if a["component"] != comp:
                continue
            if must_contain is None:
                return a
            if any(must_contain in t for t in texts(cur_text, a["content"])):
                return a
        return None

    cur_text = ""
    for fmt in ("json", "xml", "prose"):
        cur_text, attrs = classify_roles_for_format(fmt, syn_json, entities, "baseline")
        print(f"\n# FORMAT = {fmt}   ({len(attrs)} attribute occurrence(s))")

        # role disjointness invariant (content/context never overlap; frame is
        # the derived complement so it can't overlap either) — holds per format.
        overlap_ok = True
        for a in attrs:
            cset: set[int] = set()
            for s, e in a["content"] + a["context"]:
                span = set(range(s, e))
                if cset & span:
                    overlap_ok = False
                cset |= span
        check(overlap_ok, "content/context spans are disjoint within every attribute")

        # dimensions: numbers are content, NO digit is frame; the unit label is
        # frame ('m' for json/prose, the x/y/z axis names for xml).
        dim = first_attr(attrs, "dimensions", must_contain="1.11")
        if dim is None:
            check(False, "dimensions attribute present")
        else:
            dc = set(texts(cur_text, dim["content"]))
            df = texts(cur_text, dim["frame"])
            check(dc == {"1.11", "2.22", "3.33"}, f"dimensions content = the 3 numbers (got {sorted(dc)})")
            check(not any(any(ch.isdigit() for ch in t) for t in df), "dimensions frame has no digits (units/markup only)")
            unit = "m" if fmt != "xml" else "x"
            check(any(unit in t for t in df), f"dimensions frame carries the {'unit m' if fmt != 'xml' else 'axis names'}")

        # local_origin: the reference id is FRAME (a label), never content.
        lo = first_attr(attrs, "local_origin")
        if lo is None:
            check(False, "local_origin attribute present")
        else:
            check(any("rel_ref_id" in t for t in texts(cur_text, lo["frame"])), "local_origin ref id (rel_ref_id) is FRAME")
            check(not any("rel_ref_id" in t for t in texts(cur_text, lo["content"])), "local_origin ref id is NOT content")
            check(set(texts(cur_text, lo["content"])) == {"7.77", "8.88", "9.99"}, "local_origin content = its 3 numbers")

        # relationships: target AND kind are BOTH content (parity across formats).
        rel = first_attr(attrs, "relationships")
        if rel is None:
            check(False, "relationships attribute present")
        else:
            rc = " ".join(texts(cur_text, rel["content"]))
            check("rel_target_id" in rc, "relationships content includes the target id")
            kind_tok = "beside" if fmt == "prose" else "BESIDE"
            check(kind_tok in rc, f"relationships content includes the kind ({kind_tok})")

        # name: the entity id is content.
        nm = first_attr(attrs, "name")
        check(nm is not None and any("obj_id_x" in t for t in texts(cur_text, nm["content"])), "name content = the entity id")

        # prompt: value is content; key/tag is context for json/xml, ABSENT for prose.
        pr = first_attr(attrs, "prompt")
        if pr is None:
            check(False, "prompt attribute present")
        else:
            check(any("unique prompt phrase" in t for t in texts(cur_text, [(s, e) for s, e in pr["content"]]))
                  or "unique prompt phrase" in "".join(texts(cur_text, pr["content"])),
                  "prompt content = the phrase")
            if fmt == "prose":
                check(pr["context"] == [], "prompt has NO context in prose")
            else:
                check(any("prompt" in t for t in texts(cur_text, pr["context"])), "prompt context = the key/tag name")

        # prose has no key/scope anywhere.
        if fmt == "prose":
            check(all(not a["context"] for a in attrs), "prose: NO attribute has a context span")

    print()
    if problems:
        print(f"SELF-TEST FAILED — {len(problems)} check(s) did not pass:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("SELF-TEST PASSED — all per-format role rules hold.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run id (omit to pick interactively)")
    ap.add_argument("--slot", help="the scene, e.g. modern-house (omit to pick interactively)")
    ap.add_argument("--model", help="e.g. gemma, qwen-122b (omit to pick interactively)")
    ap.add_argument("--step", help="pipeline step / template (omit to pick interactively)")
    ap.add_argument("--node", default=None, help="target region id (optional filter, CLI only)")
    ap.add_argument("--occurrence", type=int, default=-1,
                    help="which matching firing when addressing by --step (default -1 = last)")
    ap.add_argument("--which", choices=sorted(WHICH_VARS), default=None,
                    help="which scene-bearing variable to reformat (default: scene_context)")
    ap.add_argument("--format", choices=["json", "xml", "prose", "all"], default=None,
                    help="output schema (default: all)")
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    ap.add_argument("--out-dir", default=None,
                    help="directory for output files (default: a fresh temp dir)")
    ap.add_argument("--stdout", action="store_true",
                    help="print the rendered schemas to the terminal instead of writing files")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="force interactive selection even if arguments are given")
    ap.add_argument("--list", action="store_true", help="list firings in the cell and exit")
    ap.add_argument("--roles", action="store_true",
                    help="print the per-attribute context/frame/content token-role split for the "
                         "firing's scene variable, per --format (verification for the "
                         "structure-vs-content graph) and exit")
    ap.add_argument("--roles-test", action="store_true",
                    help="run the built-in classification self-test on a synthetic scene (asserts the "
                         "tricky per-attribute × per-format role rules) and exit — needs no run")
    ap.add_argument("--coord-mode", choices=sorted(COORD_MODE_INPUT), default="baseline",
                    help="coordinate-ablation level for --roles; drops the origin line that mode hides "
                         "(l2l hides Global origin, g2g/g2l hide Local origin)")
    args = ap.parse_args()

    if args.roles_test:
        return _roles_selftest()

    runs_dir = Path(args.runs_dir)

    # Decide whether to prompt: explicitly forced, or something needed is missing
    # and we have a real terminal to ask on.
    need_step = not args.list
    missing = (args.run is None or args.slot is None or args.model is None
               or (need_step and args.step is None))
    interactive = args.interactive or (missing and sys.stdin.isatty())

    if interactive:
        print("scene_schema — interactive select  (Enter accepts the [default])")

    run = resolve_dir_choice(
        "run", args.run, list_runs(runs_dir),
        lambda: runs_with_completions(runs_dir), interactive)
    slot = resolve_dir_choice(
        "slot", args.slot, list_slots(runs_dir, run),
        lambda: slots_with_completions(runs_dir, run), interactive)
    model = resolve_dir_choice(
        "model", args.model, list_models(runs_dir, run, slot),
        lambda: models_with_completions(runs_dir, run, slot), interactive)

    events_path = cell_dir(runs_dir, run, slot, model) / "events.jsonl"
    events = load_events(events_path)

    if args.list:
        cmd_list(events)
        return 0

    # Pick the firing: exact address via --step, else an interactive menu.
    if args.step is not None:
        event = find_firing(events, args.step, args.node, args.occurrence)
    elif interactive:
        event = resolve_firing_interactive(events)
    else:
        raise SystemExit("--step is required (or use --list, or run interactively)")

    which = resolve_which(args.which, interactive)
    fmt = resolve_format(args.format, interactive)

    var_name = WHICH_VARS[which]
    text = (event.get("variables") or {}).get(var_name) or ""

    # --roles: print the per-attribute token-role classification (per --format)
    # and exit (the verification harness for the structure-vs-content graph).
    if args.roles:
        step_name = event.get("template") or event.get("step") or "step"
        entities = parse_scene(text)
        print(f"# {run}/{slot}/{model}  step={step_name} node={event.get('node') or 'none'} "
              f"index={event.get('index')}  which={which}  ({len(entities)} top-level entities)")
        for f in formats_for(fmt):
            cmd_roles(f, text, entities, args.coord_mode)
        return 0

    entities = parse_scene(text)

    step_name = event.get("template") or event.get("step") or "step"
    node_name = event.get("node") or "none"
    hdr = (f"# {run}/{slot}/{model}  step={step_name} node={node_name} "
           f"index={event.get('index')}  which={which} "
           f"({len(entities)} top-level entities)")

    # --stdout: old behavior, dump everything to the terminal.
    if args.stdout:
        def block(title: str, body: str) -> str:
            return f"\n===== {title} =====\n{body}"

        print(hdr)
        for f in formats_for(fmt):
            print(block(f.upper(), render(f, text, entities)))
        return 0

    # Default: write one file per format to a temp (or --out-dir) directory.
    base = "_".join(_slug(x) for x in (run, slot, model, step_name, node_name,
                                       which, event.get("index", "")))
    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="scene_schema_"))
    written = save_outputs(out_dir, base, fmt, text, entities)

    print(hdr)
    print(f"\nwrote {len(written)} file(s) to {out_dir}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())