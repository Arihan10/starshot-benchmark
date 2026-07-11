"""Derive small, view-specific projections from a full attention result.

A full `<ev>.json` result carries, for EVERY generated token, the per-head
(up to `max_heads`) top-k attended entities/attributes (+ a parallel to-place
readout). With no query-token subsampling that balloons to hundreds of MB per
step -- far too large to ship to (and parse in) the browser, which is what made
the /tf page crash.

The frontend, though, only ever needs three shapes:

  * ``compact``  -- scalars for the heatmap (per-token, per-head scene mass) plus
    the head-grid, entity maps, and PRECOMPUTED aggregates (entity/attribute/
    kind totals, reasoning-vs-output, head x entity, entity x token, and per
    output-object rollups). Powers the attention heatmap, the summary tab, the
    overview tab, and the placement tab -- WITHOUT any per-token entity lists.
  * ``token``    -- one token's full per-head detail (the top entities/attributes
    for the token detail table), fetched lazily as the user scrubs.
  * ``present``  -- per-token, head-SUMMED entities/attributes/to-place (what the
    present-mode animation reduces the heads to), fetched only when presenting.

These are derived from the canonical full result ONCE, cached on disk next to it
(``attention/.derived/``), and rebuilt when the source changes. Serving them
lets the client lazy/partial-load instead of pulling the whole blob.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import zlib
from pathlib import Path
from typing import Any

# Bump when the derived schema changes so stale sidecars are rebuilt.
#   1: initial (uncompressed compact/present + plain-text tokens.ndjson).
#   2: heavy artifacts stored gzip-compressed at rest — the per-token store is
#      `tokens.ndjson.gz` (each record zlib-deflated, seeked via a [offset, len]
#      index). Attention JSON is ~10x compressible, so this shrinks the on-disk
#      footprint massively without dropping any detail.
#   3: aggregation expansion — the compact carries the unified `buckets` view
#      (region x word-type mass over generation progression).
#   4: buckets extended — categorized region partition (variables/text/completion)
#      with per-leaf meta + token counts, plus word-type organized/free splits.
#   5: word-type `structural` split into bracket/separator/quote/operator tag kinds
#      (wider type grid) — rebuild compact so the frontend gets the new leaves.
#   6: progression bucket resolution raised (48 → 128) — rebuild compact.
# NOTE: buckets now also carry the per-attribute `attr_role` grid; it's passed
# through verbatim on fresh computes and its absence on old sidecars is tolerated,
# so it deliberately does NOT bump this gate (no forced sidecar rebuild).
DERIVED_VERSION = 6

# How many entities/attributes to keep in the precomputed matrices/rollups.
# Generous vs. what the UI slices (heatmaps 12, head x entity 10, output 12) so
# the client can still slice without losing rows.
_HE_TOP = 12      # head x entity matrix rows
_ET_TOP = 12      # entity x token matrix rows
_OUT_TOP = 12     # per output-object top entities
_OUT_COMP = 10    # per output-object top attributes
# Present mode only graphs the top ~5 objects per token (GRAPH_TOP_OBJS) and
# glows a handful, so keep the per-token entity list bounded rather than shipping
# every attended object -- keeps the (opt-in) present payload small.
_PRESENT_TOP = 16

# Deriving parses the (possibly huge) source file. Serialize it process-wide so
# concurrent requests (e.g. the overview tab fetching many steps in parallel)
# can't trigger several hundred-MB JSON parses at once and blow up memory.
_BUILD_LOCK = threading.Lock()


def _derived_dir(cell_dir: Path) -> Path:
    return cell_dir / "attention" / ".derived"


def _paths(cell_dir: Path, event_index: int) -> dict[str, Path]:
    d = _derived_dir(cell_dir)
    stem = f"{event_index}.v{DERIVED_VERSION}"
    return {
        "compact": d / f"{stem}.compact.json",
        "present": d / f"{stem}.present.json",
        "ndjson": d / f"{stem}.tokens.ndjson.gz",  # v2: zlib-per-record, seeked via index
        "index": d / f"{stem}.tokens.idx.json",
    }


def source_paths(cell_dir: Path, event_index: int) -> tuple[Path, Path]:
    """The heavy per-token/per-head result's (gzipped, plain) candidate paths.
    Stored gzip-compressed at rest (`{ev}.full.json.gz`, ~10x smaller); the plain
    `{ev}.full.json` is still accepted for back-compat / mid-migration reads."""
    base = cell_dir / "attention"
    return base / f"{event_index}.full.json.gz", base / f"{event_index}.full.json"


def source_path(cell_dir: Path, event_index: int) -> Path | None:
    """The heavy source that actually exists (gz preferred), or None."""
    gz, plain = source_paths(cell_dir, event_index)
    if gz.is_file():
        return gz
    if plain.is_file():
        return plain
    return None


def read_full(cell_dir: Path, event_index: int) -> dict[str, Any] | None:
    """Parse the heavy full result, transparently handling gzip or plain."""
    src = source_path(cell_dir, event_index)
    return _read_json_maybe_gz(src) if src is not None else None


def _read_json_maybe_gz(path: Path) -> dict[str, Any] | None:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# --- aggregation helpers -----------------------------------------------------


def _sorted_entities(
    acc: dict[str, list[Any]],
    comps: dict[str, dict[str, float]] | None = None,
    *,
    comp_top: int = 12,
) -> list[dict[str, Any]]:
    """`{id: [kind, score]}` -> `[{id, kind, score}]` sorted by score desc.

    When `comps` (a per-entity `{component: score}` map) is supplied, each entity
    also carries its own `components` breakdown (top `comp_top`, sorted). This is
    what lets the ablation "scene ordering" view plot attention to a SPECIFIC
    object's SPECIFIC attribute — a per-object × per-attribute split the marginal
    `componentTotals` can't provide."""
    out: list[dict[str, Any]] = []
    for k, v in acc.items():
        e: dict[str, Any] = {"id": k, "kind": v[0], "score": v[1]}
        if comps is not None:
            cm = comps.get(k)
            if cm:
                top = sorted(cm.items(), key=lambda kv: kv[1], reverse=True)[:comp_top]
                e["components"] = [{"component": ck, "score": cv} for ck, cv in top]
        out.append(e)
    out.sort(key=lambda e: e["score"], reverse=True)
    return out


def _sorted_components(acc: dict[str, float]) -> list[dict[str, Any]]:
    out = [{"component": k, "score": v} for k, v in acc.items()]
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def _add_entity(acc: dict[str, list[Any]], eid: str, kind: str, score: float) -> None:
    slot = acc.get(eid)
    if slot is None:
        acc[eid] = [kind, score]
    else:
        slot[1] += score


def build_compact(full: dict[str, Any]) -> dict[str, Any]:
    """The small summary + precomputed aggregates the client loads per step."""
    meta = full.get("meta", {})
    heads = full.get("selected_heads", []) or []
    tokens = full.get("tokens", []) or []
    n_heads = len(heads)
    n_tokens = len(tokens)

    out_start = n_tokens
    for i, t in enumerate(tokens):
        if t.get("output_entity") is not None:
            out_start = i
            break

    # Single pass over the (heavy) per-token/per-head detail, building every
    # rollup the summary/overview/placement tabs need so they never touch the
    # per-token entity lists themselves.
    ent: dict[str, list[Any]] = {}            # scene entity totals
    comp: dict[str, float] = {}               # scene attribute totals (marginal)
    ent_comp: dict[str, dict[str, float]] = {}  # scene entity -> per-attribute totals
    r_ent: dict[str, list[Any]] = {}          # reasoning-phase entity totals
    o_ent: dict[str, list[Any]] = {}          # output-phase entity totals
    he: dict[str, list[float]] = {}           # entity -> per-head score
    et: dict[str, list[float]] = {}           # entity -> per-token score
    tp_ent: dict[str, list[Any]] = {}         # to-place entity totals
    tp_comp: dict[str, float] = {}            # to-place attribute totals
    outs: dict[str, dict[str, Any]] = {}      # output_entity -> rollup accumulators

    mass = [0.0] * n_tokens
    entropy = [0.0] * n_tokens
    tp_mass = [0.0] * n_tokens
    tp_present = False
    ctokens: list[dict[str, Any]] = []

    for i, t in enumerate(tokens):
        ths = t.get("heads", []) or []
        hscale = [0.0] * n_heads
        sm = se = 0.0
        n = 0
        tsm = 0.0
        tn = 0
        oe = t.get("output_entity")
        og = None
        if oe is not None:
            og = outs.get(oe)
            if og is None:
                og = outs[oe] = {"n": 0, "scene": {}, "scene_comp": {}, "sc_mass": 0.0,
                                 "tp": {}, "tp_comp": {}, "tp_mass": 0.0, "hn": 0}
            og["n"] += 1

        for r in range(n_heads):
            h = ths[r] if r < len(ths) else None
            if h is None:
                continue
            hs = h.get("scale") or 0.0
            hscale[r] = hs
            sm += hs
            se += h.get("entropy_ratio") or 0.0
            n += 1
            for e in h.get("top_entities", []) or []:
                eid = e.get("id")
                ek = e.get("kind")
                es = e.get("score") or 0.0
                _add_entity(ent, eid, ek, es)
                _add_entity(o_ent if i >= out_start else r_ent, eid, ek, es)
                arr = he.get(eid)
                if arr is None:
                    arr = he[eid] = [0.0] * n_heads
                arr[r] += es
                row = et.get(eid)
                if row is None:
                    row = et[eid] = [0.0] * n_tokens
                row[i] += es
                ec = ent_comp.get(eid)
                if ec is None:
                    ec = ent_comp[eid] = {}
                for ck, cv in (e.get("components") or {}).items():
                    comp[ck] = comp.get(ck, 0.0) + cv
                    ec[ck] = ec.get(ck, 0.0) + cv
                if og is not None:
                    _add_entity(og["scene"], eid, ek, es)
                    for ck, cv in (e.get("components") or {}).items():
                        og["scene_comp"][ck] = og["scene_comp"].get(ck, 0.0) + cv
            tp = h.get("to_place")
            if tp:
                tp_present = True
                tsm += tp.get("scale") or 0.0
                tn += 1
                # to-place entity + attribute totals come from the to-place
                # top_entities' per-attribute breakdown, mirroring the client's
                # aggregateTokens(tokens, "to_place").
                for e in tp.get("top_entities", []) or []:
                    eid = e.get("id")
                    ek = e.get("kind")
                    es = e.get("score") or 0.0
                    _add_entity(tp_ent, eid, ek, es)
                    if og is not None:
                        _add_entity(og["tp"], eid, ek, es)
                    for ck, cv in (e.get("components") or {}).items():
                        tp_comp[ck] = tp_comp.get(ck, 0.0) + cv
                        if og is not None:
                            og["tp_comp"][ck] = og["tp_comp"].get(ck, 0.0) + cv

        mass[i] = (sm / n) if n else 0.0
        entropy[i] = (se / n) if n else 0.0
        tp_mass[i] = (tsm / tn) if tn else 0.0
        if og is not None:
            og["sc_mass"] += mass[i]
            og["tp_mass"] += tp_mass[i]
        ctokens.append({
            "index": t.get("index"),
            "char": t.get("char"),
            "text": t.get("text"),
            "output_entity": oe,
            "logprob": t.get("logprob"),
            "remote_logprob": t.get("remote_logprob"),
            "hscale": hscale,
        })

    scene_totals = _sorted_entities(ent, ent_comp)
    he_entities = [{"id": e["id"], "kind": e["kind"]} for e in scene_totals[:_HE_TOP]]
    et_entities = [{"id": e["id"], "kind": e["kind"]} for e in scene_totals[:_ET_TOP]]

    outputs = []
    for eid, g in outs.items():
        n = max(g["n"], 1)
        scene_tot = _sorted_entities(g["scene"])[:_OUT_TOP]
        scene_comp = _sorted_components(g["scene_comp"])[:_OUT_COMP]
        tp_tot = _sorted_entities(g["tp"])[:_OUT_TOP]
        tp_comp_l = _sorted_components(g["tp_comp"])[:_OUT_COMP]
        outputs.append({
            "entity": eid,
            "n": g["n"],
            "scene": {"mass": g["sc_mass"] / n, "entityTotals": scene_tot, "componentTotals": scene_comp},
            "to_place": ({"mass": g["tp_mass"] / n, "entityTotals": tp_tot, "componentTotals": tp_comp_l}
                         if (tp_tot or tp_comp_l) else None),
        })

    agg: dict[str, Any] = {
        "mass": mass,
        "entropy": entropy,
        "scene": {"entityTotals": scene_totals, "componentTotals": _sorted_components(comp)},
        "reasoning": {"entityTotals": _sorted_entities(r_ent)},
        "output": {"entityTotals": _sorted_entities(o_ent)},
        "head_entity": {"entities": he_entities, "M": [he.get(e["id"], [0.0] * n_heads) for e in he_entities]},
        "entity_token": {"entities": et_entities, "M": [et.get(e["id"], [0.0] * n_tokens) for e in et_entities]},
        "outputs": outputs,
        "to_place": ({"mass": tp_mass, "entityTotals": _sorted_entities(tp_ent),
                      "componentTotals": _sorted_components(tp_comp)} if tp_present else None),
    }

    return {
        "compact": True,
        "meta": meta,
        "selected_heads": heads,
        "head_grid": full.get("head_grid", []),
        "scene_entities": full.get("scene_entities", []),
        "to_place_entities": full.get("to_place_entities", []),
        "logprob_check": full.get("logprob_check", {}),
        "n_tokens": n_tokens,
        "out_start": out_start,
        "tokens": ctokens,
        "agg": agg,
        # Aggregation expansion: the unified region/type/progression view, passed
        # through verbatim from the full result (already small + head-averaged).
        "buckets": full.get("buckets") or {},
    }


def build_present(full: dict[str, Any]) -> dict[str, Any]:
    """Per-token, head-SUMMED entities/attributes/to-place -- the reduction the
    present-mode animation performs. Each token gets a single synthetic head so
    present.js can consume it unchanged (it sums over `t.heads`)."""
    meta = full.get("meta", {})
    heads = full.get("selected_heads", []) or []
    tokens = full.get("tokens", []) or []
    n_heads = max(len(heads), 1)
    ptokens: list[dict[str, Any]] = []
    for t in tokens:
        ths = t.get("heads", []) or []
        sm = 0.0
        em: dict[str, dict[str, Any]] = {}     # entity -> {id, kind, score, components}
        tsm = 0.0
        tn = 0
        tpm: dict[str, dict[str, Any]] = {}
        for h in ths:
            sm += h.get("scale") or 0.0
            for e in h.get("top_entities", []) or []:
                eid = e.get("id")
                ce = em.get(eid)
                if ce is None:
                    ce = em[eid] = {"id": eid, "kind": e.get("kind"), "score": 0.0, "components": {}}
                ce["score"] += e.get("score") or 0.0
                for ck, cv in (e.get("components") or {}).items():
                    ce["components"][ck] = ce["components"].get(ck, 0.0) + cv
            tp = h.get("to_place")
            if tp:
                tsm += tp.get("scale") or 0.0
                tn += 1
                for e in tp.get("top_entities", []) or []:
                    eid = e.get("id")
                    ce = tpm.get(eid)
                    if ce is None:
                        ce = tpm[eid] = {"id": eid, "score": 0.0}
                    ce["score"] += e.get("score") or 0.0
        ents = sorted(em.values(), key=lambda e: e["score"], reverse=True)[:_PRESENT_TOP]
        tps = sorted(tpm.values(), key=lambda e: e["score"], reverse=True)[:_PRESENT_TOP]
        head = {"scale": sm / n_heads, "top_entities": ents}
        if tn:
            head["to_place"] = {"scale": tsm / tn, "top_entities": tps}
        ptokens.append({"index": t.get("index"), "output_entity": t.get("output_entity"), "heads": [head]})
    return {
        "present": True,
        "meta": meta,
        "selected_heads": heads,
        "scene_entities": full.get("scene_entities", []),
        "to_place_entities": full.get("to_place_entities", []),
        "tokens": ptokens,
    }


# --- build + serve -----------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_ndjson(full: dict[str, Any], ndjson: Path, index: Path) -> None:
    """One zlib-deflated record per token = its full per-head detail, concatenated;
    plus a `[byte_offset, compressed_len]` index so a single token can be seeked +
    inflated without touching the rest. Per-record (not whole-file) compression is
    what keeps random access while still shrinking this (largest) sidecar ~5x."""
    tokens = full.get("tokens", []) or []
    offsets: list[list[int]] = []
    tmp = ndjson.with_name(f".{ndjson.name}.{os.getpid()}.tmp")
    try:
        pos = 0
        with tmp.open("wb") as f:
            for t in tokens:
                rec = {"index": t.get("index"), "text": t.get("text"),
                       "output_entity": t.get("output_entity"), "heads": t.get("heads", [])}
                blob = zlib.compress(json.dumps(rec).encode("utf-8"), 6)
                f.write(blob)
                offsets.append([pos, len(blob)])
                pos += len(blob)
        os.replace(tmp, ndjson)
    finally:
        tmp.unlink(missing_ok=True)
    _atomic_write(index, json.dumps(offsets))


# Which derived file(s) back each view. A view is (re)built only when ITS files
# are stale — so a present request never triggers the heavy per-token `ndjson`
# build (present.js only consumes `present.json`), and the cheap compact/present
# caches can be pre-warmed without also materializing the big token sidecar.
_VIEW_FILES: dict[str, tuple[str, ...]] = {
    "compact": ("compact",),
    "present": ("present",),
    "tokens": ("ndjson", "index"),
}


def _fresh_files(paths: dict[str, Path], keys: tuple[str, ...], src: Path) -> bool:
    """True when every derived file for `keys` exists and is >= the source mtime."""
    try:
        src_m = src.stat().st_mtime
    except OSError:
        return False
    for k in keys:
        try:
            if paths[k].stat().st_mtime < src_m:
                return False
        except OSError:
            return False
    return True


def _ensure_views(cell_dir: Path, event_index: int, views: tuple[str, ...]) -> dict[str, Path] | None:
    """Build only the requested `views` ({compact, present, tokens}) from the full
    result, parsing the (possibly huge) source ONCE and only when something is
    actually stale. Returns the derived paths, or None if the source is missing."""
    src = source_path(cell_dir, event_index)
    if src is None:
        return None
    paths = _paths(cell_dir, event_index)
    stale = [v for v in views if not _fresh_files(paths, _VIEW_FILES[v], src)]
    if not stale:
        return paths
    with _BUILD_LOCK:
        stale = [v for v in views if not _fresh_files(paths, _VIEW_FILES[v], src)]
        if not stale:  # another request built them while we waited
            return paths
        full = _read_json_maybe_gz(src)
        if full is None:
            return None
        _derived_dir(cell_dir).mkdir(parents=True, exist_ok=True)
        try:
            if "compact" in stale:
                _atomic_write(paths["compact"], json.dumps(build_compact(full)))
            if "present" in stale:
                _atomic_write(paths["present"], json.dumps(build_present(full)))
            if "tokens" in stale:
                _write_ndjson(full, paths["ndjson"], paths["index"])
        finally:
            del full
    return paths


def ensure(cell_dir: Path, event_index: int) -> dict[str, Path] | None:
    """Build/refresh ALL derived views for a step (compact + present + tokens)."""
    return _ensure_views(cell_dir, event_index, ("compact", "present", "tokens"))


def warm(cell_dir: Path, event_index: int, *, tokens: bool = False) -> dict[str, Path] | None:
    """Pre-build the CHEAP serving views (compact + present) so the first present /
    overview access is a fast cache hit instead of a heavy source parse. The big
    per-token `ndjson` is only built when `tokens=True` (it's rebuilt lazily on the
    first token-detail open otherwise, keeping the warmed footprint small)."""
    views = ("compact", "present", "tokens") if tokens else ("compact", "present")
    return _ensure_views(cell_dir, event_index, views)


def get_compact(cell_dir: Path, event_index: int) -> dict[str, Any] | None:
    paths = _ensure_views(cell_dir, event_index, ("compact",))
    if paths is None:
        return None
    try:
        return json.loads(paths["compact"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_present(cell_dir: Path, event_index: int) -> dict[str, Any] | None:
    paths = _ensure_views(cell_dir, event_index, ("present",))
    if paths is None:
        return None
    try:
        return json.loads(paths["present"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_token(cell_dir: Path, event_index: int, i: int) -> dict[str, Any] | None:
    """One token's full per-head detail, seeked out of the NDJSON sidecar."""
    paths = _ensure_views(cell_dir, event_index, ("tokens",))
    if paths is None:
        return None
    try:
        offsets = json.loads(paths["index"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(offsets, list) or i < 0 or i >= len(offsets):
        return None
    off, length = offsets[i]
    try:
        with paths["ndjson"].open("rb") as f:
            f.seek(off)
            raw = f.read(length)
        return json.loads(zlib.decompress(raw).decode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, zlib.error):
        return None
