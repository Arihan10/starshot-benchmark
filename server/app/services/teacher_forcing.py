"""Teacher-forcing export: reconstruct the *exact* token-scale sequence a step
was (or would be) fed to Gemma, and map every scene-context region/object (and
every rendered variable) to a character span inside it.

Downstream goal
---------------
Re-run a teacher-forcing diagnosis on a local Gemma: feed it the same input +
reasoning + output the pipeline saw, score per-token logprobs, and attribute
those tokens back to individual scene entities (zones / objects / to-place
items / output assignments).

What's faithful vs. what's MOCK
-------------------------------
* FAITHFUL — `system` / `user` are the exact bytes the pipeline sent to
  OpenRouter (`cache.llm.system` / `.user`); the scene-context + to-place
  strings are the exact `variables` values substituted verbatim into `user`;
  the id -> char-span maps are therefore exact.
* MOCK — the Gemma chat template and the reasoning ("thinking") wrapper are a
  documented-plausible stand-in, NOT the real `chat_template.jinja`. OpenRouter
  normalizes reasoning into a unified field and strips the provider's native
  thinking delimiters, so the true wrapper can only come from the real Gemma
  tokenizer. Everything is assembled from labeled pieces (see `segments`) so a
  consumer can swap the real template in and keep the same span math. The
  boolean `meta.mock` and `meta.notes` flag every assumption.

The whole sequence is assembled from labeled `pieces`, so `segments` +
`boundaries` describe it exactly and every span below indexes into `text.full`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.services import llm

# --- MOCK Gemma-4 chat template constants ------------------------------------
# Gemma uses <start_of_turn>/<end_of_turn> turn delimiters with a <bos> prefix
# (added by the tokenizer). Gemma 4 adds native system-role support, so we emit
# a dedicated system turn — matching how the pipeline sends [{system},{user}].
# These are the ONE place to change when wiring the real tokenizer.
_BOS = "<bos>"
_TURN_OPEN = "<start_of_turn>{role}\n"
_TURN_CLOSE = "<end_of_turn>\n"
# Stand-in for Gemma's native thinking tokens (OpenRouter strips the real ones).
_THINK_OPEN = "<thinking>\n"
_THINK_CLOSE = "\n</thinking>\n"

# Scene-context / to-place entry markers (see core/scene_context.py). Node ids
# are slug-like (no whitespace), so `\S+` captures exactly the id and stops
# before a target-zone marker suffix on the same line.
_SCENE_MARKER_RE = re.compile(r"(?m)^[ \t]*(Subregion name|Name): (\S+)")
_TOPLACE_MARKER_RE = re.compile(r"(?m)^[ \t]*id: (\S+)")

# Every variable that can carry named zone/object entries. Different steps
# inject different context blocks — the full embedded tree (SCENE_CONTEXT), the
# neighbour tree (ADJACENT_ZONES), the per-zone/root object lists, and the
# image step's graduated blocks (SIBLING_OBJECTS / *_BRIEF). We scan whichever
# ones actually landed in `user` so EVERY step gets a scene map, not just the
# bbox solvers. An entity named in two blocks yields two occurrences (each a
# distinct char span), which is what token attribution wants.
_SCENE_BEARING_VARS = (
    "SCENE_CONTEXT",
    "SCENE_CONTEXT_COMPACT",
    "ADJACENT_ZONES",
    "ROOT_OBJECTS",
    "ZONE_OBJECTS",
    "SIBLING_OBJECTS",
    "OTHER_SUBREGIONS_BRIEF",
    "ROOT_OBJECTS_BRIEF",
)


# Per-line component labels within a scene-context / to-place entry (see
# core/scene_context.py's `_object_entry` / `_region_embedded_entry` /
# `render_to_place_block`). Order matters — the most specific prefix wins. A
# line inside a `parent {…}` / `parent_region {…}` brace sub-block is attributed
# to that block regardless of its own label, so a parent's `parent_dimensions`
# reads as `parent`, not `dimensions`.
_ENTRY_LINE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Subregion name:", "name"),
    ("Name:", "name"),
    ("id:", "name"),
    ("noun_phrase", "noun_phrase"),  # noun_phrase: and noun_phrase/description:
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
    ("parent_region_dimensions:", "parent_region"),
    ("parent_region:", "parent_region"),
    ("parent_relationship_kind:", "parent"),
    ("parent_id:", "parent"),
    ("parent_dimensions:", "parent"),
    ("parent_global_origin_corner:", "parent"),
    ("parent_name:", "parent"),
    ("parent_placement:", "parent"),
    ("parent:", "parent"),
)


def _line_component(stripped: str) -> str | None:
    for prefix, comp in _ENTRY_LINE_COMPONENTS:
        if stripped.startswith(prefix):
            return comp
    return None


# --- per-attribute token ROLES (context / frame / content) -------------------
# Within each component's char span, tokens are further split into three roles
# for the structure-vs-content attention readout:
#   * context — the key/field NAME ("placement", "Dimensions", "Global origin
#     corner"); the label the value is filed under.
#   * frame   — the structural scaffolding around the value: colon, brackets,
#     braces, parens, commas, quotes, and fixed unit labels (m, deg), plus any
#     descriptive parenthetical on the key (local-origin's "(relative to X …)").
#   * content — the actual value token(s): numbers, prose, relationship
#     targets/kinds, enum, id.
# The rules are per value-KIND (coord / quoted / relationships / yaw / scalar),
# char-span based (tokenizer-agnostic — a token inherits the role it overlaps
# most, resolved downstream in semantic.build_scene_entities), and MIRROR the
# canonical, human-verifiable classifier in scripts/scene_schema.py
# (classify_line_roles). The coord ablation renders `key: value` (never XML), so
# only that form is handled here.
_ROLE_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ROLE_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
_ROLE_SCALAR_RE = re.compile(r"[A-Za-z0-9_\-./]+")


def _line_value_kind(stripped: str) -> str:
    if stripped.startswith((
        "Dimensions:", "Global origin corner:", "Local origin corner",
        "parent_dimensions:", "parent_region_dimensions:", "parent_global_origin_corner:",
    )):
        return "coord"
    if stripped.startswith((
        "prompt:", "description:", "placement:", "orientation:", "noun_phrase", "parent_placement:",
    )):
        return "quoted"
    if stripped.startswith("relationships:"):
        return "relationships"
    if stripped.startswith("global yaw:"):
        return "yaw"
    return "scalar"


def _value_content_spans(value: str, kind: str) -> list[tuple[int, int]]:
    """CONTENT char spans within a field's value (offsets relative to `value`),
    per its serialized kind. Everything else in the value is FRAME."""
    if kind in ("coord", "yaw"):
        return [m.span() for m in _ROLE_NUM_RE.finditer(value)]
    if kind == "quoted":
        a = value.find('"')
        b = value.rfind('"')
        if a >= 0 and b > a:
            return [(a + 1, b)]
        s = value.strip()
        if s:
            i = value.find(s)
            return [(i, i + len(s))]
        return []
    if kind == "relationships":
        a = value.find("[")
        b = value.rfind("]")
        lo = a + 1 if a >= 0 else 0
        hi = b if b > lo else len(value)
        return [(lo + m.start(), lo + m.end()) for m in _ROLE_WORD_RE.finditer(value[lo:hi])]
    return [m.span() for m in _ROLE_SCALAR_RE.finditer(value)]


def _line_roles(line: str) -> list[tuple[str, int, int]]:
    """Role sub-spans `(role, start, end)` for one `key: value` line (offsets
    relative to `line`). Empty for a non-field line."""
    ci = line.find(":")
    if ci < 0:
        return []
    stripped = line.strip()
    if _line_component(stripped) is None:
        return []
    key_part = line[:ci]
    value = line[ci + 1:]
    vbase = ci + 1
    key_lo = len(key_part) - len(key_part.lstrip())
    paren = key_part.find("(")                       # local-origin qualifier → frame
    ctx_hi = paren if paren >= 0 else len(key_part.rstrip())
    spans: list[tuple[str, int, int]] = []
    if ctx_hi > key_lo:
        spans.append(("context", key_lo, ctx_hi))
    for s, e in _value_content_spans(value, _line_value_kind(stripped)):
        spans.append(("content", vbase + s, vbase + e))
    claimed = bytearray(len(line))
    for _r, s, e in spans:
        for i in range(s, min(e, len(line))):
            claimed[i] = 1
    i, n = key_lo, len(line)
    while i < n:
        if not claimed[i] and not line[i].isspace():
            j = i
            while j < n and not claimed[j] and not line[j].isspace():
                j += 1
            spans.append(("frame", i, j))
            i = j
        else:
            i += 1
    return spans


def _entry_components(text: str) -> list[dict[str, Any]]:
    """Split one entry's own-field text into per-component spans (offsets
    relative to `text`), merging consecutive lines of the same component so a
    multi-line `parent {…}` block is one span. Structural lines (braces, blanks,
    the "Objects placed directly within…" preamble) are skipped."""
    comps: list[dict[str, Any]] = []
    pos = 0
    ctx: str | None = None  # inside a parent/parent_region brace sub-block
    ctx_depth = 0
    cur: dict[str, Any] | None = None
    for line in text.splitlines(keepends=True):
        start = pos
        pos += len(line)
        stripped = line.strip()
        if ctx is not None:
            comp: str | None = ctx
            ctx_depth += line.count("{") - line.count("}")
            if ctx_depth <= 0:
                ctx = None
        else:
            comp = _line_component(stripped)
            if comp in ("parent", "parent_region") and stripped.endswith("{"):
                ctx = comp
                ctx_depth = 1
        if comp is None:
            cur = None
            continue
        if cur is not None and cur["component"] == comp:
            cur["end"] = pos
        else:
            cur = {"component": comp, "start": start, "end": pos, "roles": []}
            comps.append(cur)
        # Fold this line's context/frame/content role sub-spans (absolute to the
        # entry text) onto the component. Parent-block sub-lines contribute too.
        for role, rs, rre in _line_roles(line.rstrip("\n")):
            cur["roles"].append({"role": role, "start": start + rs, "end": start + rre})
    return comps


def _components_and_parent(seg: str, abs_base: int) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Per-component spans (shifted to absolute offsets by `abs_base`) plus the
    entry's structural `parent` id and owning `region` id, parsed from its text.
    Works for both scene-context entries (parent inside a `parent {…}` block or
    `parent_name`) and to-place entries (a flat `parent: <id>` line)."""
    comps = _entry_components(seg)
    for c in comps:
        c["start"] += abs_base
        c["end"] += abs_base
        for r in c.get("roles", []):
            r["start"] += abs_base
            r["end"] += abs_base
    m_pid = re.search(r"(?m)^[ \t]*parent_id: (\S+)", seg)
    m_pval = re.search(r"(?m)^[ \t]*parent: (\S+)", seg)
    m_pname = re.search(r"(?m)^[ \t]*parent_name: (\S+)", seg)
    m_region = re.search(r"(?m)^[ \t]*parent_region: (\S+)", seg)
    parent: str | None = None
    for mm in (m_pid, m_pval, m_pname):
        if mm and mm.group(1) != "{":
            parent = mm.group(1)
            break
    region = m_region.group(1) if (m_region and m_region.group(1) != "{") else None
    return comps, parent, region


def _enclosing_json_object(text: str, id_value: str) -> tuple[int, int] | None:
    """Span of the innermost `{...}` object that contains the first `"id_value"`
    occurrence in `text` — used to bound one output assignment/spec in the raw
    output JSON. None when the id isn't present or braces don't balance."""
    anchor = text.find(f'"{id_value}"')
    if anchor < 0:
        return None
    depth = 0
    start = -1
    i = anchor
    while i >= 0:
        c = text[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                start = i
                break
            depth -= 1
        i -= 1
    if start < 0:
        return None
    depth = 0
    for j in range(start, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (start, j + 1)
    return None


def _output_ids(output: Any) -> list[str]:
    """Every id an output structure names, in declaration order — covering the
    id-bearing shapes the pipeline emits (assignments / objects / subregions /
    children, and a single `object`). Mirrors the client obs model's provenance
    scan so the two agree on what an output 'placed'."""
    ids: list[str] = []
    if not isinstance(output, dict):
        return ids
    for key in ("assignments", "objects", "subregions", "children"):
        seq = output.get(key)
        if isinstance(seq, list):
            for item in seq:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    ids.append(item["id"])
    obj = output.get("object")
    if isinstance(obj, dict) and isinstance(obj.get("id"), str):
        ids.append(obj["id"])
    # Preserve order, drop dupes.
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def _canonical_output_text(event: dict[str, Any], logprobs: dict[str, Any] | None) -> tuple[str, str]:
    """The raw output string the model emitted, plus how we sourced it.

    Prefer the logprobs sidecar's `text` (the verbatim token concatenation — the
    ONLY lossless copy of the emitted string). Fall back to a compact re-dump of
    the parsed `output` dict, which loses the model's original whitespace/key
    order (flagged so a consumer knows the output map is approximate)."""
    if logprobs and isinstance(logprobs.get("text"), str) and logprobs["text"]:
        return logprobs["text"], "logprobs_sidecar_verbatim"
    import json

    return json.dumps(event.get("output"), ensure_ascii=False), "reserialized_parsed_output"


def has_scene_context(event: dict[str, Any]) -> bool:
    """Whether this step's prompt actually contains scene entities (so an
    attention analysis would have something to attend to). Cheap: scans the
    scene-bearing variables for a scene marker that's present verbatim in the
    user turn — the same condition `build_export` uses to populate `scene_map`,
    without doing the full reconstruction. Early steps (root plans, the overall
    bbox) have no scene yet and return False."""
    variables: dict[str, Any] = event.get("variables") or {}
    user = str(event.get("user") or "")
    # A SCHEMA ablation renders the scene as XML/prose — which carries NO soft-JSON
    # `Name:` / `Subregion name:` markers — but logs the per-attribute role span-map
    # under `__SCENE_ROLES__`. Treat a non-empty logged map whose rendered var is
    # present verbatim in `user` as scene-bearing, MIRRORING how `build_export`
    # populates `scene_map` from it — else XML/prose variants read as "no scene".
    try:
        logged: dict[str, Any] = json.loads(str(variables.get("__SCENE_ROLES__") or "{}"))
    except Exception:
        logged = {}
    if isinstance(logged, dict):
        for var_name, smap in logged.items():
            block = str(variables.get(var_name) or "")
            if smap and block and block in user:
                return True
    for var_name in _SCENE_BEARING_VARS:
        block = str(variables.get(var_name) or "")
        if block and _SCENE_MARKER_RE.search(block) and block in user:
            return True
    return False


def build_export(
    event: dict[str, Any],
    *,
    run: str,
    slot: str,
    model_alias: str,
    model_id: str,
    logprobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct the teacher-forcing sequence for one `cache.llm` event and
    map every scene entity into it. All char offsets index `text.full`."""
    system = str(event.get("system") or "")
    user = str(event.get("user") or "")
    reasoning = str(event.get("reasoning") or "")
    output_text, output_source = _canonical_output_text(event, logprobs)
    variables: dict[str, Any] = event.get("variables") or {}

    # Assemble the sequence from labeled pieces so segment/boundary offsets are
    # exact and a real template can drop in without disturbing the span math.
    pieces: list[tuple[str, str, str]] = []  # (label, kind, text)
    pieces.append(("bos", "control", _BOS))
    if system.strip():
        pieces.append(("system_turn_open", "control", _TURN_OPEN.format(role="system")))
        pieces.append(("system", "input", system))
        pieces.append(("system_turn_close", "control", _TURN_CLOSE))
    pieces.append(("user_turn_open", "control", _TURN_OPEN.format(role="user")))
    pieces.append(("user", "input", user))
    pieces.append(("user_turn_close", "control", _TURN_CLOSE))
    pieces.append(("model_turn_open", "control", _TURN_OPEN.format(role="model")))
    # --- everything below is the model's generated completion (teacher-forced) ---
    if reasoning.strip():
        pieces.append(("thinking_open", "control", _THINK_OPEN))
        pieces.append(("reasoning", "reasoning", reasoning))
        pieces.append(("thinking_close", "control", _THINK_CLOSE))
    pieces.append(("output", "output", output_text))
    pieces.append(("model_turn_close", "control", "<end_of_turn>"))

    # Cumulative offsets.
    full_parts: list[str] = []
    segments: list[dict[str, Any]] = []
    offset_of: dict[str, int] = {}
    pos = 0
    for label, kind, text in pieces:
        offset_of[label] = pos
        segments.append({"label": label, "kind": kind, "start": pos, "end": pos + len(text)})
        full_parts.append(text)
        pos += len(text)
    full = "".join(full_parts)

    completion_start = offset_of["model_turn_open"] + len(_TURN_OPEN.format(role="model"))
    user_start = offset_of["user"]

    def _span_in_user(value: str) -> tuple[int, int] | None:
        """Absolute span of a verbatim `user` substring within `full`."""
        if not value:
            return None
        rel = user.find(value)
        if rel < 0:
            return None
        return (user_start + rel, user_start + rel + len(value))

    # --- scene zones/objects -> spans (contiguous own-field segments) ----------
    # Scan every scene-bearing variable that landed in `user`, so image_prompt
    # (SIBLING_OBJECTS / *_BRIEF) and the bbox solvers (SCENE_CONTEXT) alike get
    # a populated map. Within one block, an entry owns the text from its marker
    # up to the next marker; `source` records which variable it came from.
    # A SCHEMA ablation renders the scene context as XML/prose, which has no
    # soft-JSON `key:` grammar to scrape, so the pipeline logs the per-attribute
    # role span-map (from the shared emitter) under `__SCENE_ROLES__` (offsets
    # RELATIVE to each variable's string). When present for a variable we consume
    # it (rebased into `full`) instead of the marker scan; otherwise the soft-JSON
    # scrape runs — so base runs / legacy events are unaffected.
    try:
        _logged_roles: dict[str, Any] = json.loads(str(variables.get("__SCENE_ROLES__") or "{}"))
    except Exception:
        _logged_roles = {}

    def _from_logged(entries: list[dict[str, Any]], base: int, source: str | None) -> list[dict[str, Any]]:
        out_entries: list[dict[str, Any]] = []
        for ent in entries or []:
            comps = [{
                "component": c.get("component"),
                "start": base + c["start"], "end": base + c["end"],
                "roles": [{"role": r["role"], "start": base + r["start"], "end": base + r["end"]}
                          for r in c.get("roles", [])],
            } for c in ent.get("components", [])]
            e = {
                "id": ent.get("id"),
                "start": base + ent["start"], "end": base + ent["end"],
                "parent": None, "region": None, "components": comps,
            }
            if source is not None:
                e["kind"] = ent.get("kind") or "object"
                e["source"] = source
            out_entries.append(e)
        return out_entries

    scene_map: list[dict[str, Any]] = []
    for var_name in _SCENE_BEARING_VARS:
        block_text = str(variables.get(var_name) or "")
        block_span = _span_in_user(block_text)
        if not block_text or block_span is None:
            continue
        base = block_span[0]
        logged = _logged_roles.get(var_name)
        if logged:
            scene_map.extend(_from_logged(logged, base, var_name))
            continue
        marks = list(_SCENE_MARKER_RE.finditer(block_text))
        for i, m in enumerate(marks):
            nxt = marks[i + 1].start() if i + 1 < len(marks) else len(block_text)
            seg = block_text[m.start():nxt]
            abs_base = base + m.start()
            comps, parent, region = _components_and_parent(seg, abs_base)
            scene_map.append({
                "id": m.group(2),
                "kind": "zone" if m.group(1) == "Subregion name" else "object",
                "start": abs_base,
                "end": base + nxt,
                "source": var_name,
                "parent": parent,
                "region": region,
                "components": comps,
            })
    scene_map.sort(key=lambda e: e["start"])

    # --- to-place batch ids -> spans --------------------------------------------
    to_place_map: list[dict[str, Any]] = []
    tp = str(variables.get("TO_PLACE") or "")
    tp_span = _span_in_user(tp)
    if tp and tp_span is not None:
        tp_start = tp_span[0]
        logged_tp = _logged_roles.get("TO_PLACE")
        if logged_tp:
            to_place_map.extend(_from_logged(logged_tp, tp_start, source=None))
            marks = []  # consumed from the logged span-map (XML/prose TO_PLACE)
        else:
            marks = list(_TOPLACE_MARKER_RE.finditer(tp))
        for i, m in enumerate(marks):
            nxt = marks[i + 1].start() if i + 1 < len(marks) else len(tp)
            seg = tp[m.start():nxt]
            abs_base = tp_start + m.start()
            comps, parent, region = _components_and_parent(seg, abs_base)
            to_place_map.append({
                "id": m.group(1),
                "start": abs_base,
                "end": tp_start + nxt,
                "parent": parent,
                "region": region,
                "components": comps,
            })

    # --- XML-gravity instruction-block SENTENCES + tag spans --------------------
    # A gravity variant logs `__GRAVITY__` (offsets RELATIVE to `user`): the
    # instruction block's per-sentence spans (+ snippets) + the moved <prompt>
    # open/close tag spans. Rebase into `full` (+ keep `user_rel` for the native
    # reconstruct). Empty for every non-gravity step, so base runs are unaffected.
    gravity_out: dict[str, Any] = {}
    try:
        _logged_gravity: dict[str, Any] = json.loads(str(variables.get("__GRAVITY__") or "{}"))
    except Exception:
        _logged_gravity = {}
    if _logged_gravity.get("sentences") is not None:
        def _grav_span(pair: Any) -> dict[str, Any] | None:
            if not pair:
                return None
            return {"start": user_start + int(pair[0]), "end": user_start + int(pair[1]),
                    "user_rel": [int(pair[0]), int(pair[1])]}
        gravity_out = {
            "mode": _logged_gravity.get("mode"),
            "block": _grav_span(_logged_gravity.get("block")),
            "open_tag": _grav_span(_logged_gravity.get("open_tag")),
            "close_tag": _grav_span(_logged_gravity.get("close_tag")),
            "sentences": [
                {"i": s.get("i"), "snippet": s.get("snippet"),
                 **(_grav_span([s.get("start"), s.get("end")]) or {})}
                for s in _logged_gravity.get("sentences", [])
            ],
        }

    # --- output assignments/specs -> spans in the emitted output ----------------
    output_map: list[dict[str, Any]] = []
    out_start = offset_of["output"]
    for oid in _output_ids(event.get("output")):
        span = _enclosing_json_object(output_text, oid)
        if span is not None:
            output_map.append({"id": oid, "start": out_start + span[0], "end": out_start + span[1]})

    # --- every rendered variable ("different kinds of param") -> span -----------
    # First occurrence within `user` of each non-empty resolved variable value,
    # so a consumer can locate ROOT_HEADER, ZONE_PLAN, ADJACENT_ZONES, TO_PLACE,
    # SCENE_CONTEXT, etc. inside the reconstructed input.
    variables_map: list[dict[str, Any]] = []
    for name, value in variables.items():
        if not isinstance(value, str) or not value.strip():
            continue
        span = _span_in_user(value)
        if span is not None:
            variables_map.append({"name": name, "start": span[0], "end": span[1], "len": len(value)})
    variables_map.sort(key=lambda v: v["start"])

    # --- express every map in the model's RAW input / output frames -----------
    # Downstream teacher-forcing tokenizes the model's OWN input (the `user`
    # message it was fed) and output (the string it emitted) — not our
    # reconstructed `full` sequence. So alongside the `full` offsets (which drive
    # the sequence viewer), tag each entry with its frame and give raw-frame-
    # relative offsets: input entities index `text.user`; output assignments
    # index `text.output` — the SAME frame the logprobs sidecar token offsets
    # live in, so `output_rel` maps an assignment straight onto its scored tokens.
    def _tag_input(entry: dict[str, Any]) -> None:
        entry["frame"] = "input"
        entry["user_rel"] = [entry["start"] - user_start, entry["end"] - user_start]
        for c in entry.get("components", []):
            c["user_rel"] = [c["start"] - user_start, c["end"] - user_start]
            for r in c.get("roles", []):
                r["user_rel"] = [r["start"] - user_start, r["end"] - user_start]

    for entry in scene_map:
        _tag_input(entry)
    for entry in to_place_map:
        _tag_input(entry)
    for entry in variables_map:
        entry["frame"] = "input"
        entry["user_rel"] = [entry["start"] - user_start, entry["end"] - user_start]
    for entry in output_map:
        entry["frame"] = "output"
        entry["output_rel"] = [entry["start"] - out_start, entry["end"] - out_start]

    # The model's raw input/output frames within `full`, so a consumer knows
    # which region is context (not scored) vs the scored completion, and which
    # raw field each map's frame-relative offsets index.
    frames = {
        "input": {
            "start": 0, "end": completion_start, "len": completion_start,
            "raw_field": "user",
            "note": "prompt / context — NOT scored under teacher forcing; `user_rel` offsets index text.user",
        },
        "completion": {
            "start": completion_start, "end": len(full), "len": len(full) - completion_start,
            "note": "the model's generated tokens — teacher-forced / scored",
        },
        "reasoning": (
            {"start": offset_of["reasoning"], "end": offset_of["thinking_close"]}
            if "reasoning" in offset_of else None
        ),
        "output": {
            "start": out_start, "end": out_start + len(output_text),
            "raw_field": "output",
            "note": "`output_rel` offsets index text.output and align 1:1 with the logprobs sidecar token offsets",
        },
    }

    lp_summary = {
        "available": bool(logprobs),
        "top_n": (logprobs or {}).get("top_n"),
        "token_count": len((logprobs or {}).get("tokens") or []),
        "covers": "output tokens only (OpenRouter content logprobs; reasoning tokens are a separate channel)",
    }

    return {
        "meta": {
            "run": run,
            "slot": slot,
            "model": model_alias,
            "model_id": model_id,
            "event_index": event.get("index"),
            "step": event.get("step"),
            "template": event.get("template"),
            "node": event.get("node"),
            "schema": event.get("schema"),
            "key": event.get("key"),
            "mock": True,
            "output_source": output_source,
            "notes": [
                "chat template + thinking wrapper are a MOCK stand-in for Gemma's real "
                "chat_template.jinja / native thinking tokens (OpenRouter normalizes reasoning).",
                "system/user are the exact bytes sent; scene-context + variable spans are exact.",
                "completion_start marks where the model's generated (teacher-forced) tokens begin.",
            ],
        },
        # Generation params ("different kinds of param") the call rode on.
        "params": {
            "response_format": event.get("schema"),
            "reasoning_effort": llm._reasoning_effort(model_id),
            "require_parameters": True,
            "provider_sort": "latency",
            "temperature": None,
            "top_logprobs": (logprobs or {}).get("top_n"),
            "logprobs_requested": bool(logprobs),
            "tokens_in": event.get("tokens_in"),
            "tokens_out": event.get("tokens_out"),
            "generation_id": event.get("generation_id"),
        },
        "text": {
            "system": system,
            "user": user,
            "reasoning": reasoning,
            "output": output_text,
            "full": full,
        },
        "boundaries": {
            "input_end": completion_start,
            "completion_start": completion_start,
            "reasoning": {"start": offset_of.get("reasoning"), "end": offset_of.get("thinking_close")}
            if "reasoning" in offset_of
            else None,
            "output": {"start": out_start, "end": out_start + len(output_text)},
            "total_len": len(full),
        },
        "frames": frames,
        "segments": segments,
        "scene_map": scene_map,
        "to_place_map": to_place_map,
        "output_map": output_map,
        "variables_map": variables_map,
        "gravity": gravity_out,
        "logprobs": lp_summary,
    }
