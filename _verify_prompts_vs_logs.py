"""Verify the pre-reclass-frame prompt templates against what was ACTUALLY
sent to the models, as recorded verbatim in each run's event log.

For every `cache.llm` event under a run:
  * its SYSTEM bytes must byte-match a candidate template's `<step>.system.txt`
    (root vs nested variants disambiguated by which system matches);
  * its USER bytes must fit that same candidate's `<step>.user.txt` SKELETON
    (the template's static segments, in order, anchored at both ends, with the
    `{VARIABLE}` slots as arbitrary per-scene fills);
  * the `ZONE_ID` fill, if present, must equal the event's node id.

These events predate the `template`/`variables` logging fields, so we rely on
the skeleton match (same technique as server/scripts/test_prompt_templates.py
section 7) rather than a re-render.

Streams the (multi-GB) logs in fixed byte chunks so no whole line is forced
into memory beyond its own size, and only JSON-parses `cache.llm` records.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from starshot_paths import runs_root

VERSION_DIR = Path("versions/pre-reclass-frame")
RUNS_ROOT = runs_root() / "against-the-gods"

# event-log step id -> candidate template names (root/nested share a step id)
EVENT_STEP_TEMPLATES = {
    "zone_plan": ("zone_plan_root", "zone_plan"),
    "overall_bbox": ("overall_bbox",),
    "zone_decompose": ("zone_decompose_root", "zone_decompose"),
    "child_bbox_batch": ("child_bbox_batch",),
    "anchor_decompose": ("anchor_decompose",),
    "encapsulating_decompose": ("encapsulating_decompose",),
    "negative_space_decompose": ("negative_space_decompose",),
    "object_bbox_batch": ("object_bbox_batch",),
    "next_object": ("next_object",),
    "image_prompt": ("image_prompt",),
}

# Steps that make cache.llm calls but are NOT part of the versioned prompt set
# (their prompts are hardcoded in code, not in versions/<name>/). Out of scope
# for this comparison — counted and reported, never a mismatch.
KNOWN_UNVERSIONED_STEPS = {"library_match"}

# runtime-only injection appended to the user message for DeepSeek (never in a
# template). Verbatim from server/app/services/llm.py.
DEEPSEEK_INJECTION = (
    "\n<IMPORTANT_THINKING>\n"
    "Be very opinionated in your thinking - do not go around in circles. Once a "
    "conclusion is reached, stick to it. Aim to end reasoning as soon as a "
    "concrete plan has been formed.\n"
    "</IMPORTANT_THINKING>\n"
)

VAR_TOKEN_RE = re.compile(r"`\{([A-Z][A-Z0-9_]*)\}`")
CACHE = b'"cache.llm"'
CHUNK = 8 * 1024 * 1024


def template_skeleton(text: str):
    parts = VAR_TOKEN_RE.split(text)
    return parts[0::2], parts[1::2]


def skeleton_match(template: str, rendered: str):
    """Match `rendered` against `template`'s static skeleton. Returns
    {var_name: [fill, ...]} on success, else None. (Ported verbatim in spirit
    from test_prompt_templates.skeleton_match.)"""
    statics, names = template_skeleton(template)
    statics = list(statics)
    rendered = rendered.rstrip()
    j = len(statics) - 1
    while j >= 0 and statics[j].strip() == "":
        statics[j] = ""
        j -= 1
    if j >= 0:
        statics[j] = statics[j].rstrip()
    if not rendered.startswith(statics[0]):
        return None
    pos = len(statics[0])
    fills: dict[str, list[str]] = {}
    for name, static in zip(names, statics[1:]):
        if static == "":
            fills.setdefault(name, []).append(rendered[pos:].strip())
            pos = len(rendered)
            continue
        idx = rendered.find(static, pos)
        if idx == -1:
            return None
        fills.setdefault(name, []).append(rendered[pos:idx])
        pos = idx + len(static)
    if rendered[pos:].strip():
        return None
    return fills


def first_missing_static(template: str, rendered: str) -> str:
    """Diagnostic: the first static template segment that fails to locate in
    `rendered`, to pinpoint where a user template drifted."""
    statics, _ = template_skeleton(template)
    statics = [s for s in statics if s.strip()]
    r = rendered.rstrip()
    pos = 0
    for s in statics:
        idx = r.find(s.rstrip() if s is statics[-1] else s, pos)
        if idx == -1:
            snippet = s.strip().splitlines()[0][:80] if s.strip() else s[:80]
            return snippet
        pos = idx + len(s)
    return "(all statics found; likely trailing/extra content)"


def strip_deepseek(user: str, model: str) -> str:
    if model and "deepseek" in model.lower() and user.endswith(DEEPSEEK_INJECTION):
        return user[: -len(DEEPSEEK_INJECTION)]
    return user


def iter_lines(path: Path):
    buf = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl == -1:
                    break
                yield buf[:nl]
                buf = buf[nl + 1 :]
        if buf:
            yield buf


def load_templates():
    sys_tpl, user_tpl = {}, {}
    for cands in EVENT_STEP_TEMPLATES.values():
        for cand in cands:
            sys_tpl[cand] = (VERSION_DIR / f"{cand}.system.txt").read_text(
                encoding="utf-8"
            ).rstrip("\n")
            user_tpl[cand] = (VERSION_DIR / f"{cand}.user.txt").read_text(
                encoding="utf-8"
            )
    return sys_tpl, user_tpl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--max-samples", type=int, default=12)
    args = ap.parse_args()

    sys_tpl, user_tpl = load_templates()

    files = sorted(args.runs_root.rglob("events.jsonl"))
    files = [p for p in files if "_branches" not in p.parts]
    print(f"pre-reclass-frame: {VERSION_DIR}")
    print(f"scanning {len(files)} event logs under {args.runs_root}\n")

    per_cell = {}
    matched_by_template = Counter()
    unknown_steps = Counter()
    unversioned_steps = Counter()
    zone_id_violations = []
    samples = []
    total = matched = sys_mm = user_mm = 0
    n_tpl_field = n_vars_field = 0  # log-format documentation

    for path in files:
        rel = path.relative_to(args.runs_root)
        parts = rel.parts
        scene = parts[0] if len(parts) > 1 else "?"
        model_alias = parts[1] if len(parts) > 2 else "?"
        cell = f"{scene}/{model_alias}"
        c = per_cell.setdefault(
            cell, {"total": 0, "matched": 0, "sys_mm": 0, "user_mm": 0,
                   "unknown": 0, "unversioned": 0, "steps": Counter()}
        )
        for line in iter_lines(path):
            if CACHE not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") != "cache.llm":
                continue
            step = e.get("step")
            cands = EVENT_STEP_TEMPLATES.get(step)
            total += 1
            c["total"] += 1
            if "template" in e:
                n_tpl_field += 1
            if "variables" in e:
                n_vars_field += 1
            if cands is None:
                if step in KNOWN_UNVERSIONED_STEPS:
                    unversioned_steps[step] += 1
                    c["unversioned"] += 1
                else:
                    unknown_steps[step] += 1
                    c["unknown"] += 1
                continue
            system = (e.get("system") or "").rstrip("\n")
            model = e.get("model") or ""
            user = strip_deepseek(e.get("user") or "", model)
            node = e.get("node")

            sys_hit = None
            hit = None
            for cand in cands:
                if system == sys_tpl[cand]:
                    sys_hit = cand
                    if skeleton_match(user_tpl[cand], user) is not None:
                        hit = cand
                        break
            if hit is not None:
                matched += 1
                c["matched"] += 1
                c["steps"][hit] += 1
                matched_by_template[hit] += 1
                fills = skeleton_match(user_tpl[hit], user)
                zids = set(fills.get("ZONE_ID", [])) if fills else set()
                if zids and isinstance(node, str) and zids != {node}:
                    zone_id_violations.append((cell, hit, sorted(zids), node))
            elif sys_hit is not None:
                user_mm += 1
                c["user_mm"] += 1
                if len(samples) < args.max_samples:
                    samples.append(
                        f"[USER-SKELETON] {cell} step={step} node={node} "
                        f"sys matched {sys_hit!r} but user drifted; first "
                        f"unmatched static: {first_missing_static(user_tpl[sys_hit], user)!r}"
                    )
            else:
                sys_mm += 1
                c["sys_mm"] += 1
                if len(samples) < args.max_samples:
                    tpl = sys_tpl[cands[0]]
                    diff = "\n".join(
                        difflib.unified_diff(
                            tpl.splitlines(), system.splitlines(),
                            "pre-reclass-frame", "logged", lineterm="",
                        )
                    )
                    samples.append(
                        f"[SYSTEM] {cell} step={step} node={node} — no candidate "
                        f"system matched ({', '.join(cands)}); diff vs {cands[0]!r}:\n"
                        + "\n".join(diff.splitlines()[:24])
                    )

    print("=" * 78)
    print("PER-CELL RESULTS (scene/model)")
    print("=" * 78)
    for cell in sorted(per_cell):
        c = per_cell[cell]
        flag = "OK " if (c["sys_mm"] == 0 and c["user_mm"] == 0 and c["unknown"] == 0) else "!! "
        print(f"{flag}{cell}")
        print(f"     cache.llm={c['total']}  matched={c['matched']}  "
              f"sys_mismatch={c['sys_mm']}  user_skeleton_mismatch={c['user_mm']}  "
              f"unversioned(library_match)={c['unversioned']}  unknown_step={c['unknown']}")
        print(f"     matched templates: {dict(c['steps'])}")

    print("\n" + "=" * 78)
    print("TEMPLATES EXERCISED (matched event counts)")
    print("=" * 78)
    for cand in sorted(matched_by_template):
        print(f"   {cand}: {matched_by_template[cand]}")
    never = [cand for cands in EVENT_STEP_TEMPLATES.values() for cand in cands
             if cand not in matched_by_template]
    if never:
        print(f"   (no logged calls for: {', '.join(sorted(set(never)))})")

    if unversioned_steps:
        print("\nUNVERSIONED STEPS (excluded — prompts live in code, not versions/):",
              dict(unversioned_steps))
    if unknown_steps:
        print("\nUNKNOWN STEPS (not in EVENT_STEP_TEMPLATES):", dict(unknown_steps))
    if zone_id_violations:
        print(f"\nZONE_ID fill != node id ({len(zone_id_violations)}):")
        for v in zone_id_violations[:12]:
            print("   ", v)

    if samples:
        print("\n" + "=" * 78)
        print(f"MISMATCH SAMPLES (showing {len(samples)})")
        print("=" * 78)
        for s in samples:
            print("\n" + s)

    unversioned_total = sum(unversioned_steps.values())
    unknown_total = sum(unknown_steps.values())
    versioned_total = total - unversioned_total - unknown_total
    print("\n" + "=" * 78)
    print("TOTALS")
    print("=" * 78)
    print(f"   cache.llm events         : {total}")
    print(f"   versioned-step events    : {versioned_total}")
    print(f"     matched                : {matched}")
    print(f"     system mismatch        : {sys_mm}")
    print(f"     user skeleton mismatch : {user_mm}")
    print(f"   unversioned (library_match): {unversioned_total}")
    print(f"   unknown step             : {unknown_total}")
    print(f"   events carrying a 'template' field : {n_tpl_field}")
    print(f"   events carrying a 'variables' field: {n_vars_field}")
    ok = (sys_mm == 0 and user_mm == 0 and unknown_total == 0
          and not zone_id_violations and matched == versioned_total)
    print("\nVERDICT:",
          "ALL logged versioned-step prompts match pre-reclass-frame templates"
          if ok else "MISMATCHES FOUND (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
