"""Token <-> char <-> semantic-entity remapping.

The attention math works in TOKEN space; the frontend works in ENTITY space
(regions / objects / attributes). This module bridges them, carefully, using
the tf-export char-span maps:

  * A `Tokenizer` produces an OFFSET MAPPING — for each token, its [start, end)
    char span in the reconstructed `full` text. On the real (GPU) path the Modal
    worker's `HFAttentionProvider` IS the tokenizer (Gemma HF `return_offsets_
    mapping`, kept 1:1 with the forward-pass `input_ids`); a mock regex tokenizer
    (contiguous, subword-ish) lets the whole pipeline + all the span math run
    locally with no model. Both expose identical offset semantics, so the real
    provider changes token boundaries but not any downstream logic.

  * `build_scene_entities` turns each scene_map entry (char spans + components +
    parent, from tf-export) into its TOKEN indices — merged across every
    occurrence of an id (an entity named in two context blocks owns both). This
    yields the scene token column set S, and per-entity / per-component token
    sets that the aggregation reduces attention onto.

Char/token alignment rule: a token belongs to a char span [cs, ce) iff it
overlaps it (`tok.start < ce and tok.end > cs`).
"""

from __future__ import annotations

import bisect
import re
from typing import Any, Protocol


class Tokenizer(Protocol):
    name: str

    def encode_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """Return [(token_text, char_start, char_end), …] covering `text`."""
        ...


# Contiguous, subword-ish mock: each token is optional leading whitespace plus
# either a word run or a single punctuation char — so offsets tile the whole
# string and JSON/coordinate output fragments into many tokens (a decent stand-
# in for a real BPE without any model).
_MOCK_TOKEN_RE = re.compile(r"\s*(?:\w+|[^\w\s])")


class MockTokenizer:
    name = "mock-regex"

    def encode_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in _MOCK_TOKEN_RE.finditer(text)]


def get_tokenizer(model_id: str) -> Tokenizer:
    """The local, model-free MOCK regex tokenizer.

    There is deliberately NO real-HF-tokenizer variant here. The real (GPU)
    attention path never tokenizes separately: the Modal worker's
    `HFAttentionProvider` (modal_app.py) IS its own tokenizer — it keeps its offset
    list 1:1 with the forward pass `input_ids` and is passed straight into
    `worker.analyze(tokenizer=provider, provider=provider)`. A separately-loaded HF
    tokenizer here would be a TRAP: to stay aligned it would have to reproduce that
    EXACT 1:1 token set, and any divergence (e.g. dropping zero-width/special
    tokens) silently shifts every char-span → token mapping. So a real char→token
    mapping must ONLY ever come from the same provider that produced the attention."""
    return MockTokenizer()


def _overlaps(tok_s: int, tok_e: int, cs: int, ce: int) -> bool:
    return tok_s < ce and tok_e > cs


def tokens_in_span(offsets: list[tuple[str, int, int]], cs: int, ce: int) -> list[int]:
    """Token indices whose char span overlaps [cs, ce). O(n) linear scan — kept
    for one-off callers; use `OffsetIndex` for the many-span hot paths."""
    return [i for i, (_, s, e) in enumerate(offsets) if _overlaps(s, e, cs, ce)]


class OffsetIndex:
    """Precomputed sorted-offset index for O(log n) repeated `tokens_in_span`.

    HF tokenizers (and the mock) emit char spans that are SORTED and
    NON-OVERLAPPING, so the tokens overlapping [cs, ce) form a CONTIGUOUS index
    range recoverable with two binary searches, instead of the O(n_tokens) scan
    `tokens_in_span` does per span. `build_scene_entities` / `region_segments`
    issue thousands of span lookups against the SAME offsets, so building this
    once turns their cost from O(n_spans · n_tokens) into O(n_spans · log n).

    The result is IDENTICAL to `tokens_in_span` (same overlap predicate). A
    defensive monotonicity check falls back to the linear scan if a tokenizer
    ever emits non-monotonic offsets (not expected for the tokenizers here)."""

    __slots__ = ("_offsets", "_starts", "_ends", "_monotonic")

    def __init__(self, offsets: list[tuple[str, int, int]]) -> None:
        self._offsets = offsets
        self._starts = [s for _, s, _ in offsets]
        self._ends = [e for _, _, e in offsets]
        st, en = self._starts, self._ends
        self._monotonic = (
            all(st[i] <= st[i + 1] for i in range(len(st) - 1))
            and all(en[i] <= en[i + 1] for i in range(len(en) - 1))
        )

    def tokens_in_span(self, cs: int, ce: int) -> list[int]:
        """Token indices whose char span overlaps [cs, ce) — identical to the
        free `tokens_in_span`, in O(log n) on monotonic offsets. With starts and
        ends both ascending, {s < ce} is a prefix and {e > cs} is a suffix, so
        their intersection is the contiguous range [first end>cs, first start>=ce)."""
        if self._monotonic and ce > cs:
            lo = bisect.bisect_right(self._ends, cs)     # first token with end > cs
            hi = bisect.bisect_left(self._starts, ce)    # first token with start >= ce
            return list(range(lo, hi)) if hi > lo else []
        return [i for i, (_, s, e) in enumerate(self._offsets) if s < ce and e > cs]


def completion_token_start(offsets: list[tuple[str, int, int]], completion_start: int) -> int:
    """Index of the first token at/after the completion boundary — the first
    query token (everything before it is context / not scored)."""
    for i, (_, _s, e) in enumerate(offsets):
        if e > completion_start:  # token straddling or after the boundary
            return i
    return len(offsets)


class SceneTokenIndex:
    """Token-space view of the scene, built once per analysis.

    `entities[id]` = {kind, parent, region, tokens: sorted[int],
    components: {component: sorted[int]}}. `scene_tokens` is the column set S
    (union of all entity tokens)."""

    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.scene_tokens: set[int] = set()

    def entity_ids(self) -> list[str]:
        return list(self.entities.keys())


def build_scene_entities(
    scene_map: list[dict[str, Any]],
    offsets: list[tuple[str, int, int]],
) -> SceneTokenIndex:
    """Fold every scene_map occurrence into per-id token sets (+ per-component),
    merging duplicate occurrences of the same entity id."""
    idx = SceneTokenIndex()
    oi = OffsetIndex(offsets)  # built once; every span lookup below is O(log n)
    for entry in scene_map:
        eid = entry.get("id")
        if not isinstance(eid, str):
            continue
        toks = oi.tokens_in_span(entry["start"], entry["end"])
        if not toks:
            continue
        rec = idx.entities.setdefault(eid, {
            "kind": entry.get("kind", "object"),
            "parent": entry.get("parent"),
            "region": entry.get("region"),
            "tokens": set(),
            "components": {},
            "roles": {},
        })
        rec["tokens"].update(toks)
        idx.scene_tokens.update(toks)
        for comp in entry.get("components", []) or []:
            cname = comp["component"]
            ctoks = oi.tokens_in_span(comp["start"], comp["end"])
            if ctoks:
                rec["components"].setdefault(cname, set()).update(ctoks)
            # Per-attribute token ROLES (context / frame / content), for the
            # structure-vs-content attention readout. Char sub-spans → token sets.
            for r in comp.get("roles", []) or []:
                rtoks = oi.tokens_in_span(r["start"], r["end"])
                if rtoks:
                    rec["roles"].setdefault(cname, {}).setdefault(r["role"], set()).update(rtoks)
    # Freeze to sorted lists for stable, JSON-friendly output. Roles are made
    # DISJOINT by priority (content > context > frame) so a token that straddles
    # a `:` / quote boundary is counted in exactly one role (no double-counted mass).
    for rec in idx.entities.values():
        rec["tokens"] = sorted(rec["tokens"])
        rec["components"] = {k: sorted(v) for k, v in rec["components"].items()}
        roles_out: dict[str, dict[str, list[int]]] = {}
        for cname, rr in rec["roles"].items():
            content = set(rr.get("content", ()))
            context = set(rr.get("context", ())) - content
            frame = set(rr.get("frame", ())) - content - context
            out = {}
            if context:
                out["context"] = sorted(context)
            if frame:
                out["frame"] = sorted(frame)
            if content:
                out["content"] = sorted(content)
            if out:
                roles_out[cname] = out
        rec["roles"] = roles_out
    return idx


def gravity_segments(
    gravity: dict[str, Any] | None,
    offsets: list[tuple[str, int, int]],
) -> tuple[list[str], list[list[int]], dict[str, Any]]:
    """Map a rebased XML-gravity block onto per-SENTENCE token segments. Returns
    `(names, segs, meta)`: `names` = ['s1', 's2', …] (one per sentence present);
    `segs` = each sentence's overlapping token indices (fed to the SAME whole-row
    reduction as region/type/attr_role); `meta` = {mode, sentences:[{i, tok_start,
    tok_end, n_tokens, snippet}], open_tok, close_tok, block:[t0,t1]} — the token
    positions + text the per-sentence tag-distance readout needs. Empty when the
    step carries no gravity block."""
    sentences = (gravity or {}).get("sentences") or []
    if not sentences:
        return [], [], {}
    oi = OffsetIndex(offsets)

    def _span_toks(obj: dict[str, Any] | None) -> list[int]:
        if not obj or obj.get("start") is None or obj.get("end") is None:
            return []
        return oi.tokens_in_span(int(obj["start"]), int(obj["end"]))

    names: list[str] = []
    segs: list[list[int]] = []
    smeta: list[dict[str, Any]] = []
    for s in sentences:
        toks = _span_toks(s)
        names.append(f"s{s.get('i')}")
        segs.append(toks)
        smeta.append({"i": s.get("i"), "snippet": s.get("snippet"),
                      "tok_start": (toks[0] if toks else None),
                      "tok_end": (toks[-1] if toks else None), "n_tokens": len(toks)})
    blk_toks = _span_toks((gravity or {}).get("block"))
    open_toks = _span_toks((gravity or {}).get("open_tag"))
    close_toks = _span_toks((gravity or {}).get("close_tag"))
    meta = {
        "mode": (gravity or {}).get("mode"),
        "sentences": smeta,
        "open_tok": (open_toks[0] if open_toks else None),
        "close_tok": (close_toks[0] if close_toks else None),
        "block": ([blk_toks[0], blk_toks[-1]] if blk_toks else None),
    }
    return names, segs, meta


# --- aggregation expansion: context regions + word/token types --------------
# Two orthogonal, whole-sequence decompositions of where/what a token attends,
# built on the SAME offsets the attention columns use (so they line up exactly).
# Both are plain column-index lists here; the reduction (on-device on the GPU
# path, per-row on the fallback) sums the renormalized attention row onto them.

# Completion regions, always present with stable names. The input/prompt frame is
# NOT a single region — it is decomposed into its XML-tag sections (below), because
# the prompts are partly organized into tags (<intro>, <judging_criteria>, <output>,
# …) and partly free text. That lets the readout show which tagged section the model
# attends to, and whether organized structure draws more attention than the rest.
COMPLETION_REGIONS = ("reasoning", "output")

# One markup tag delimiter: <name ...> or </name>. Chat-template control tokens
# (<start_of_turn>, <end_of_turn>, <bos>) are their own un-paired names, so the
# stack matcher below never pairs them into a section.
_TAG_TOKEN_RE = re.compile(r"<(/?)([A-Za-z][\w\-]*)(?:\s[^>]*)?>", re.DOTALL)


def xml_tag_spans(text: str, lo: int, hi: int) -> list[tuple[str, int, int]]:
    """Every paired XML/markup section within [lo, hi) of `text`, as
    (tag_name, char_start, char_end) INCLUDING the delimiters. A simple name-matched
    stack so nested sections are all captured (outer + inner); unclosed opens are
    ignored. The caller resolves overlaps (innermost wins)."""
    stack: list[tuple[str, int]] = []
    spans: list[tuple[str, int, int]] = []
    for m in _TAG_TOKEN_RE.finditer(text, lo, hi):
        closing, name = m.group(1), m.group(2)
        if not closing:
            stack.append((name, m.start()))
            continue
        for k in range(len(stack) - 1, -1, -1):     # match nearest open of this name
            if stack[k][0] == name:
                _, os = stack.pop(k)
                spans.append((name, os, m.end()))
                break
    return spans


# Prompt sections (by XML tag name) that are split into per-SENTENCE sub-regions
# (prompt.<TAG>#NN) instead of a single whole-section region, so the attention
# readout resolves onto individual instructions rather than the whole block. The
# team iterates heavily on VERY_IMPORTANT_INSTRUCTIONS, so it's split by default.
SENTENCE_SPLIT_TAGS = frozenset({"VERY_IMPORTANT_INSTRUCTIONS"})

# Sentence terminators. VII is split into whole SENTENCES — NOT clauses: commas,
# semicolons and colons do NOT split, so a full instruction stays one segment.
_SENT_END = frozenset(".!?")


def _sentence_spans(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Split [lo, hi) of `text` into SENTENCE sub-spans. A boundary is a sentence-
    ending mark (. ! ?) FOLLOWED BY whitespace or the span end — so decimals
    ("0.5"), inline dots and mid-token abbreviations don't split — or a newline
    (line / bullet items). The terminator stays with its sentence; whitespace-only
    fragments are dropped (never an empty segment)."""
    spans: list[tuple[int, int]] = []
    start = i = lo
    while i < hi:
        ch = text[i]
        if ch == "\n" or (ch in _SENT_END and (i + 1 >= hi or text[i + 1].isspace())):
            j = i + 1
            if text[start:j].strip():
                spans.append((start, j))
            start = i = j
        else:
            i += 1
    if start < hi and text[start:hi].strip():
        spans.append((start, hi))
    return spans


# Large rendered context variables → the "Variables" region category. Names mirror
# teacher_forcing._SCENE_BEARING_VARS (kept here to avoid a service import); every
# other rendered variable at least LARGE_VAR_MIN_CHARS long falls into `var.other`.
SCENE_CONTEXT_VARS = frozenset({
    "SCENE_CONTEXT", "SCENE_CONTEXT_COMPACT", "ADJACENT_ZONES", "ROOT_OBJECTS",
    "ZONE_OBJECTS", "SIBLING_OBJECTS", "OTHER_SUBREGIONS_BRIEF", "ROOT_OBJECTS_BRIEF",
})
TOPLACE_VAR = "TO_PLACE"
LARGE_VAR_MIN_CHARS = 200


def _variable_owner(
    variables_map: list[dict[str, Any]], oi: "OffsetIndex", in_set: set[int],
) -> dict[int, str]:
    """Map each input token inside a large rendered context variable to its Variables
    SUBCATEGORY leaf (`var.scene_content` / `var.to_place` / `var.other`). Assigned
    lowest-priority first so scene_content wins any (rare) span overlap."""
    owner: dict[int, str] = {}

    def _assign(pred, leaf: str) -> None:
        for v in variables_map:
            if not pred(v):
                continue
            for ti in oi.tokens_in_span(v["start"], v["end"]):
                if ti in in_set:
                    owner[ti] = leaf

    _assign(lambda v: v.get("name") not in SCENE_CONTEXT_VARS and v.get("name") != TOPLACE_VAR
            and int(v.get("len", 0)) >= LARGE_VAR_MIN_CHARS, "var.other")
    _assign(lambda v: v.get("name") == TOPLACE_VAR, "var.to_place")
    _assign(lambda v: v.get("name") in SCENE_CONTEXT_VARS, "var.scene_content")
    return owner


def region_segments(
    frames: dict[str, Any] | None, offsets: list[tuple[str, int, int]], full_text: str = "",
    variables_map: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[list[int]], list[dict[str, Any]]]:
    """DISJOINT, CATEGORIZED region partition of the sequence (masses sum to ~1).
    Each leaf carries `{category, sub[, tag]}` so the frontend rolls subcategory
    areas up by category. Categories:
      * completion — `reasoning`, `output` (the model's own generated text);
      * variables  — `var.scene_content` / `var.to_place` / `var.other` (large
        rendered context blocks in the prompt);
      *         text       — `prompt.<tag>` (organized XML sections; tag kept for the
        which-section view) and `prompt.free` (free prompt text). Tags in
        SENTENCE_SPLIT_TAGS are further split into per-sentence leaves
        `prompt.<tag>#NN` (each meta carries a `snippet` of the sentence text).
    Priority for an input token: VARIABLE → XML tag (organized) → free. Leaf names
    + meta travel with the grid (they vary per step), so aggregation keys by name."""
    frames = frames or {}
    variables_map = variables_map or []
    oi = OffsetIndex(offsets)  # built once; shared across every span lookup below
    names: list[str] = []
    segs: list[list[int]] = []
    meta: list[dict[str, Any]] = []

    def _add(name: str, toks: list[int], m: dict[str, Any]) -> None:
        names.append(name)
        segs.append(toks)
        meta.append(m)

    for nm in COMPLETION_REGIONS:
        fr = frames.get(nm)
        toks = (oi.tokens_in_span(fr["start"], fr["end"])
                if isinstance(fr, dict) and isinstance(fr.get("start"), int)
                and isinstance(fr.get("end"), int) and fr["end"] > fr["start"] else [])
        _add(nm, toks, {"category": "completion", "sub": nm})

    inp = frames.get("input") or {}
    ilo, ihi = inp.get("start"), inp.get("end")
    if isinstance(ilo, int) and isinstance(ihi, int) and ihi > ilo:
        input_tokens = oi.tokens_in_span(ilo, ihi)
        in_set = set(input_tokens)
        owner = _variable_owner(variables_map, oi, in_set)      # variables win
        sentence_snippets: dict[str, str] = {}                      # split-sentence leaf -> its text
        for name, s, e in sorted(xml_tag_spans(full_text, ilo, ihi), key=lambda t: -(t[2] - t[1])):
            if name in SENTENCE_SPLIT_TAGS:
                # Split this section's INNER text (drop the <tag>…</tag> delimiters)
                # into per-sentence leaves prompt.<TAG>#NN, so attention resolves onto
                # individual instructions rather than the whole block. #NN is zero-
                # padded so leaves sort in reading order.
                gt = full_text.find(">", s, e)
                lt = full_text.rfind("<", s, e)
                c_lo = gt + 1 if gt != -1 else s
                c_hi = lt if lt > c_lo else e
                for k, (cs, ce) in enumerate(_sentence_spans(full_text, c_lo, c_hi), start=1):
                    leaf = f"prompt.{name}#{k:02d}"
                    for ti in oi.tokens_in_span(cs, ce):
                        if ti in in_set and ti not in owner:        # don't override a variable
                            owner[ti] = leaf
                    sentence_snippets.setdefault(leaf, " ".join(full_text[cs:ce].split())[:400])
            else:
                leaf = f"prompt.{name}"
                for ti in oi.tokens_in_span(s, e):
                    if ti in in_set and ti not in owner:            # don't override a variable
                        owner[ti] = leaf
        for ti in input_tokens:
            owner.setdefault(ti, "prompt.free")                     # remainder = free text
        groups: dict[str, list[int]] = {}
        for ti in input_tokens:
            groups.setdefault(owner[ti], []).append(ti)
        for leaf in sorted(groups):
            if leaf.startswith("var."):
                m = {"category": "variables", "sub": leaf.split(".", 1)[1]}
            elif leaf == "prompt.free":
                m = {"category": "text", "sub": "free"}
            else:
                m = {"category": "text", "sub": "organized", "tag": leaf.split(".", 1)[1]}
                if leaf in sentence_snippets:                        # per-instruction snippet
                    m["snippet"] = sentence_snippets[leaf]
            _add(leaf, groups[leaf], m)
    # Catch-all: every token in NO named region above (chat-template control tokens
    # like <bos>/turn markers/<end_of_turn>, and any completion tokens a step's frames
    # don't cover) — so the whole-row masses PARTITION the full sequence and sum to
    # ~1. Without it, attention on unframed tokens (notably the model's reasoning when
    # a step carries no reasoning frame) is silently dropped and shares undercount.
    covered: set[int] = set()
    for s in segs:
        covered.update(s)
    other = [ti for ti in range(len(offsets)) if ti not in covered]
    if other:
        _add("other", other, {"category": "other", "sub": "template"})
    return names, segs, meta


# Stable word/token class order. Orthographic kinds (number / whitespace) + the
# STRUCTURAL punctuation family — split into its tag kinds `bracket` ({}[]()),
# `separator` (:,), `quote` ("'`) and `operator` (=) so the JSON scaffolding the
# model emits is resolved, not lumped — plus closed-class function words (incl. a
# spatial-relation list); open-class content words collapse into `content` (a real
# POS tagger can refine them behind this same interface). `entity_name` is the free
# overlap with the scene layer (a token inside an entity's NAME span).
TYPE_NAMES = (
    "number", "bracket", "separator", "quote", "operator",
    "whitespace", "spatial", "function", "entity_name", "content", "other",
)

_NUM_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
# The structural punctuation family, partitioned into tag kinds. Their union is the
# set that classifies a token as structural (unchanged from the single-bucket days);
# `_structural_kind` then picks the leaf by the token's leading char.
_BRACKET_CHARS = frozenset("{}[]()")
_SEPARATOR_CHARS = frozenset(":,")
_QUOTE_CHARS = frozenset("\"'`")
_OPERATOR_CHARS = frozenset("=")
_STRUCTURAL_CHARS = _BRACKET_CHARS | _SEPARATOR_CHARS | _QUOTE_CHARS | _OPERATOR_CHARS


def _structural_kind(stripped: str) -> str:
    """Which structural tag a punctuation token belongs to, keyed on its first char
    (tokenizers emit these mostly one char at a time; the lead char breaks the rare
    mixed run like `),`). Falls back to `operator` for the residual symbol set."""
    ch = stripped[0]
    if ch in _BRACKET_CHARS:
        return "bracket"
    if ch in _SEPARATOR_CHARS:
        return "separator"
    if ch in _QUOTE_CHARS:
        return "quote"
    return "operator"
_SPATIAL_WORDS = frozenset({
    "above", "below", "beside", "under", "over", "on", "in", "inside", "into",
    "onto", "beneath", "behind", "front", "atop", "upon", "adjacent", "near",
    "against", "top", "bottom", "left", "right", "attached", "between", "around",
    "within", "outside", "up", "down", "aligned",
})
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "of", "to", "for", "with",
    "from", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "will", "would", "should", "shall", "can", "could", "must", "may", "might",
    "it", "its", "they", "them", "their", "which", "as", "at", "by", "if", "then",
    "so", "not", "no", "than", "each", "per", "we", "you", "i", "he", "she",
    "there", "here", "also", "have", "has", "had", "do", "does", "did",
})


def classify_tokens(
    offsets: list[tuple[str, int, int]], name_tokens: set[int] | None = None,
) -> tuple[list[str], list[int]]:
    """Per-token WORD/TOKEN class id (index into TYPE_NAMES) — dependency-free and
    model-agnostic (works for any tokenizer's `offsets`). Precedence: whitespace →
    entity-name (proper-noun/id, from `name_tokens`) → number → structural punct
    (resolved to its bracket/separator/quote/operator tag) → spatial relation word →
    other function word → open-class `content` → `other` (sentential punctuation,
    symbols). Returns (TYPE_NAMES, [class_id per token])."""
    names = list(TYPE_NAMES)
    cid = {n: i for i, n in enumerate(names)}
    name_tokens = name_tokens or set()
    ids: list[int] = []
    for ti, (tx, _s, _e) in enumerate(offsets):
        stripped = tx.strip()
        low = stripped.lower()
        if stripped == "":
            c = "whitespace"
        elif ti in name_tokens:
            c = "entity_name"
        elif _NUM_RE.match(stripped):
            c = "number"
        elif all(ch in _STRUCTURAL_CHARS for ch in stripped):
            c = _structural_kind(stripped)
        elif low in _SPATIAL_WORDS:
            c = "spatial"
        elif low in _FUNCTION_WORDS:
            c = "function"
        elif any(ch.isalpha() for ch in stripped):
            c = "content"
        else:
            c = "other"
        ids.append(cid[c])
    return names, ids


def name_token_set(*indexes: "SceneTokenIndex | None") -> set[int]:
    """Union of every entity's NAME-component token indices across the given scene
    indexes (scene + to-place) — the tokens `classify_tokens` marks `entity_name`."""
    out: set[int] = set()
    for idx in indexes:
        if idx is None:
            continue
        for rec in idx.entities.values():
            out.update(rec.get("components", {}).get("name", []) or [])
    return out


def output_entity_at(output_map: list[dict[str, Any]], tok_s: int, tok_e: int, offsets: list[tuple[str, int, int]]) -> str | None:
    """Which output assignment id (if any) this query token belongs to — so a
    generated token can be labelled 'the token that emitted assignment X'."""
    cs, ce = offsets[tok_s][1], offsets[tok_e][2] if tok_e < len(offsets) else offsets[tok_s][2]
    for entry in output_map:
        if _overlaps(entry["start"], entry["end"], cs, ce):
            return entry.get("id")
    return None
