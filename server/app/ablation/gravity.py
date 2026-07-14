"""XML-gravity ablation axis (`gravity_mode`).

Probe whether an XML tag acts as an attention "gravity well": does the POSITION
of a tag pull the model's attention toward the text near it, and does that pull
decay with distance? We hold the instruction CONTENT fixed and vary only where a
neutral closing tag sits.

The treated step's `<VERY_IMPORTANT_INSTRUCTIONS> ... </VERY_IMPORTANT_INSTRUCTIONS>`
block is segmented into SENTENCES (the measurement unit — attention is read per
sentence, like the data tab's VII instruction view). The closing tag moves across
a few positions spanning the block so each sentence's distance to it varies.
Levels:

    id        what the treated block becomes
    baseline  untouched (real VERY_IMPORTANT_INSTRUCTIONS tags) — the base cell
    none      tags STRIPPED (bare paragraphs) — the experiment's comparison anchor
    q1..q4    a NEUTRAL <prompt> opened at the block start + </prompt> closed after
              the sentence nearest the 1/4, 2/4, 3/4, 4/4 word-count mark (q4 =
              full wrap). The opening tag is fixed; only the closing tag moves.

The neutral `<prompt>` name is ABLATION-ONLY (production prompts untouched) so the
measurement is about tag STRUCTURE, not the words "very important".

Runtime seam
------------
Unlike coord/schema (bind-time template rewrite / var re-render), gravity rewrites
the treated step's RENDERED user prompt inside `llm.call_llm`, AFTER the cache/gate
checks — so only the variant's re-inferred treated firing is rewritten (replayed
prefix firings cache-hit on their original text and are untouched), and the block
can always be located by the original VII tags still present in the rendered text
(needed for the `none` level, which has no tag of its own to anchor on).

Both the rewritten string AND the per-sentence/tag char-span map come from the SAME
pass (`rewrite_and_stash`), so the string the model sees and the spans the
attention worker scores on can never drift. The map rides in the step's
`variables` dict under `ROLES_VAR` (logged verbatim with the `cache.llm` event;
never a template variable) and `app.services.teacher_forcing` reads it back.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.ablation import context as _ctx

DEFAULT_MODE = "baseline"
MODES: tuple[str, ...] = ("baseline", "none", "q1", "q2", "q3", "q4")
# Number of closing-tag POSITIONS (q1..q4) — a few spots spanning the block, at
# the 1/4..4/4 word-count marks. NOT the measurement unit (that is per SENTENCE).
N_CLOSE = 4

# The neutral, ablation-only tag that replaces the semantically-loaded production
# tag. Opening is fixed at the block start; closing moves across the positions.
OPEN_TAG = "<prompt>"
CLOSE_TAG = "</prompt>"

# The production instruction-block tag whose open/close we relocate. Matched with
# its own-line whitespace/newline so stripping leaves the paragraphs clean.
_OPEN_RE = re.compile(r"<VERY_IMPORTANT_INSTRUCTIONS>[ \t]*\n?")
_CLOSE_RE = re.compile(r"\n?[ \t]*</VERY_IMPORTANT_INSTRUCTIONS>[ \t]*")

# Synthetic `variables` key the span-map rides in — logged with the cache.llm
# event and read back by teacher_forcing. NOT a template variable (leading
# underscores can't match prompt_store._VAR_RE), so it never affects a render.
ROLES_VAR = "__GRAVITY__"


def _norm(mode: str | None) -> str:
    return mode if mode in MODES else DEFAULT_MODE


def current_mode() -> str:
    """The bound variant's gravity_mode, or `baseline` on a normal run."""
    rt = _ctx.current()
    if rt is None:
        return DEFAULT_MODE
    return _norm(getattr(rt.treatment, "gravity_mode", DEFAULT_MODE))


def is_active() -> bool:
    """True when a gravity treatment asks for a rewrite (none / q1..q4)."""
    return current_mode() != DEFAULT_MODE


def _target_kind() -> str | None:
    rt = _ctx.current()
    return rt.target_step_kind if rt is not None else None


# --------------------------------------------------------------------------- #
# Block location + sentence segmentation
# --------------------------------------------------------------------------- #
def _locate_block(text: str) -> tuple[int, int, int, int] | None:
    """`(open_start, inner_start, inner_end, close_end)` of the
    VERY_IMPORTANT_INSTRUCTIONS block in `text`, or None. `inner` is the
    paragraph text between the tags (tag-adjacent newlines consumed)."""
    om = _OPEN_RE.search(text)
    if not om:
        return None
    cm = _CLOSE_RE.search(text, om.end())  # match positions are ABSOLUTE into `text`
    if not cm:
        return None
    return (om.start(), om.end(), cm.start(), cm.end())


def _sentences(text: str) -> list[tuple[int, int]]:
    """Sentence / line spans (non-whitespace) within `text`. A boundary is a
    sentence-ending mark (. ! ?) FOLLOWED BY whitespace/end (so decimals and
    mid-token dots don't split) or a newline. Mirrors semantic._sentence_spans."""
    spans: list[tuple[int, int]] = []
    start = i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n" or (ch in ".!?" and (i + 1 >= n or text[i + 1].isspace())):
            j = i + 1
            if text[start:j].strip():
                spans.append((start, j))
            start = i = j
        else:
            i += 1
    if start < n and text[start:n].strip():
        spans.append((start, n))
    return spans


def _close_sentence_index(inner: str, sents: list[tuple[int, int]], j: int) -> int:
    """Index (into `sents`) of the sentence the closing tag goes AFTER for level
    q`j`: the sentence whose cumulative word count is nearest j/`N_CLOSE` of the
    block total (j == N_CLOSE -> the last sentence == full wrap). This is just the
    tag POSITION — the measurement is per sentence — so a handful of positions
    spanning the block gives each sentence a range of distances to the tag."""
    if not sents:
        return 0
    words = [max(1, len(inner[s:e].split())) for s, e in sents]
    cum: list[int] = []
    acc = 0
    for w in words:
        acc += w
        cum.append(acc)
    total = acc or 1
    target = j * total / N_CLOSE
    best_k, best_d = 0, None
    for k in range(len(sents)):
        d = abs(cum[k] - target)
        if best_d is None or d < best_d:
            best_d, best_k = d, k
    return best_k


def _trim_span(text: str, s: int, e: int) -> tuple[int, int]:
    """Shrink [s, e) to its non-whitespace extent (empty -> unchanged)."""
    seg = text[s:e]
    lead = len(seg) - len(seg.lstrip())
    trail = len(seg) - len(seg.rstrip())
    if lead + trail >= len(seg):
        return (s, e)
    return (s + lead, e - trail)


# --------------------------------------------------------------------------- #
# Rewrite + span-map (single pass)
# --------------------------------------------------------------------------- #
def _build(prefix: str, inner: str, suffix: str, mode: str) -> tuple[str, dict[str, Any]]:
    """Reassemble the user prompt for `mode`, recording (same pass) EACH SENTENCE's
    content span (+ a snippet) and the open/close tag spans. The block content is
    emitted VERBATIM (only the tags are inserted) — the closing tag splits `inner`
    into head/tail after the level's close sentence, and each sentence span is
    rebased onto whichever side it lands in (the close is a sentence boundary, so
    no sentence straddles it)."""
    sents = _sentences(inner)
    close_char: int | None = None
    if mode != "none":
        idx = _close_sentence_index(inner, sents, int(mode[1:])) if sents else 0
        close_char = sents[idx][1] if sents else len(inner)

    parts: list[str] = []
    pos = 0

    def emit(s: str) -> int:
        nonlocal pos
        a = pos
        parts.append(s)
        pos += len(s)
        return a

    emit(prefix)
    block_start = pos
    open_span: list[int] | None = None
    close_span: list[int] | None = None
    if mode != "none":
        os = emit(OPEN_TAG)
        open_span = [os, os + len(OPEN_TAG)]
        emit("\n")
    head = inner if close_char is None else inner[:close_char]
    head_base = emit(head)
    tail_base = pos
    if mode != "none":
        emit("\n")
        cs = emit(CLOSE_TAG)
        close_span = [cs, cs + len(CLOSE_TAG)]
        tail_base = emit(inner[close_char:])
    block_end = pos
    emit(suffix)
    text = "".join(parts)

    sentences: list[dict[str, Any]] = []
    for k, (ss, se) in enumerate(sents):
        if close_char is None or se <= close_char:
            ns, ne = head_base + ss, head_base + se
        else:
            ns, ne = tail_base + (ss - close_char), tail_base + (se - close_char)
        ns, ne = _trim_span(text, ns, ne)
        sentences.append({"i": k + 1, "start": ns, "end": ne, "snippet": text[ns:ne][:200]})

    smap = {
        "mode": mode,
        "block": [block_start, block_end],
        "open_tag": open_span,
        "close_tag": close_span,
        "sentences": sentences,
    }
    return text, smap


def rederive_sentences(user: str, logged: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-sentence spans (offsets into `user`, + snippets) DERIVED at attribution
    time from the committed rewritten `user` and a logged `__GRAVITY__` map's
    `block` + tag positions — independent of the map's OWN segmentation.

    This is what makes per-sentence attribution work on ALREADY-committed variants
    of ANY logged shape (e.g. an older `quarters` map) WITHOUT re-inference: the
    block/tag char spans are stable across shapes, so we carve the tags out of the
    block and re-segment the instruction text here, at compute time. Mirrors the
    split `_build` records for a fresh variant, so fresh + old variants agree."""
    block = logged.get("block")
    if not block:
        return []
    bs, be = int(block[0]), int(block[1])
    tags = sorted((t for t in (logged.get("open_tag"), logged.get("close_tag")) if t),
                  key=lambda t: int(t[0]))
    ranges: list[tuple[int, int]] = []          # block minus the inserted tag spans
    cur = bs
    for t in tags:
        ts, te = int(t[0]), int(t[1])
        if ts > cur:
            ranges.append((cur, ts))
        cur = max(cur, te)
    if be > cur:
        ranges.append((cur, be))
    if not ranges:
        ranges = [(bs, be)]
    out: list[tuple[int, int]] = []
    for rs, re_ in ranges:
        for ss, se in _sentences(user[rs:re_]):
            a, b = _trim_span(user, rs + ss, rs + se)
            if user[a:b].strip():
                out.append((a, b))
    out.sort()
    return [{"i": k + 1, "start": a, "end": b, "snippet": user[a:b][:200]}
            for k, (a, b) in enumerate(out)]


def rewrite_and_stash(user: str, variables: dict[str, Any] | None, template: str | None) -> str:
    """On a variant's fresh treated call, rewrite the instruction block per the
    bound gravity_mode and stash the per-sentence/tag span-map under `ROLES_VAR`.
    Returns the (possibly rewritten) user. A NO-OP — returns `user` unchanged and
    stashes nothing — for a normal run, the baseline level, a non-target step, or
    a prompt with no VII block, so non-gravity calls stay byte-identical."""
    mode = current_mode()
    if mode == DEFAULT_MODE:
        return user
    if not template or template != _target_kind():
        return user
    loc = _locate_block(user)
    if loc is None:
        return user
    o_start, inner_start, inner_end, c_end = loc
    prefix = user[:o_start]
    inner = user[inner_start:inner_end]
    suffix = user[c_end:]
    new_user, smap = _build(prefix, inner, suffix, mode)
    if variables is not None:
        try:
            variables[ROLES_VAR] = json.dumps(smap)
        except Exception:
            pass
    return new_user
