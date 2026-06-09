"""Build runs/v3-iter5/prompts_snapshot.py.

Scaffold = git blob of prompts.py at commit 7f7534b (validated to reproduce
v3-iter5's render output), with every SYSTEM_* constant the run exercised
overridden VERBATIM from the event log (safe append via repr()).

Writes to temp_analysis/_v3iter5_snapshot.py (NOT into runs/ yet — validate first).

  server/.venv/bin/python temp_analysis/build_snapshot.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUN = "v3-iter5"
SCAFFOLD_COMMIT = "7f7534b"
PROMPTS_REL = "server/app/core/prompts.py"
OUT = Path(__file__).resolve().parent / "_v3iter5_snapshot.py"
sys.path.insert(0, str(ROOT / "server"))

# step -> SYSTEM constant name. zone_plan is split by node (root vs non-root).
STEP_CONST = {
    "overall_bbox": "SYSTEM_OVERALL_BBOX",
    "zone_decompose": "SYSTEM_ZONE_DECOMPOSE",
    "child_bbox_batch": "SYSTEM_ZONE_BBOX_BATCH",
    "object_bbox_batch": "SYSTEM_OBJECT_BBOX_BATCH",
    "anchor_decompose": "SYSTEM_ANCHOR_DECOMP",
    "encapsulating_decompose": "SYSTEM_ENCAPSULATING_DECOMP",
    "negative_space_decompose": "SYSTEM_NEGATIVE_SPACE_DECOMP",
    "next_object": "SYSTEM_NEXT_OBJECT",
}


def collect_systems() -> dict[str, str]:
    """SYSTEM constant name -> verbatim logged system string (asserts uniqueness)."""
    found: dict[str, set[str]] = {}
    for ev in (RUNS / RUN).glob("*/*/events.jsonl"):
        for line in ev.open():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") != "cache.llm" or not isinstance(e.get("system"), str):
                continue
            step = e.get("step")
            if step == "zone_plan":
                name = "SYSTEM_ROOT_ZONE_PLAN" if e.get("node") == "root" else "SYSTEM_ZONE_PLAN"
            else:
                name = STEP_CONST.get(step)
            if not name:
                continue
            found.setdefault(name, set()).add(e["system"])
    out = {}
    for name, vals in found.items():
        if len(vals) != 1:
            raise SystemExit(f"!! {name}: expected 1 distinct system, got {len(vals)}")
        out[name] = next(iter(vals))
    return out


def main() -> None:
    scaffold = subprocess.run(
        ["git", "show", f"{SCAFFOLD_COMMIT}:{PROMPTS_REL}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if scaffold.returncode != 0:
        raise SystemExit(scaffold.stderr)
    src = scaffold.stdout

    systems = collect_systems()
    print(f"recovered {len(systems)} SYSTEM constants from log:")
    for name in sorted(systems):
        print(f"  {name:30} {len(systems[name])} chars")

    # sanity: scaffold must already define these names (so overrides line up
    # with what the divider/generation read off the module)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        spath = f.name
    spec = importlib.util.spec_from_file_location("scaffold_7f7534b", spath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [n for n in systems if not hasattr(mod, n)]
    if missing:
        raise SystemExit(f"!! scaffold missing SYSTEM names: {missing}")
    matched = sum(1 for n, v in systems.items() if getattr(mod, n) == v)
    print(f"\nscaffold already matches {matched}/{len(systems)} verbatim; "
          f"overriding all {len(systems)} to be safe.")

    lines = [
        "",
        "",
        "# " + "=" * 72,
        f"# v3-iter5 SYSTEM prompts recovered VERBATIM from the event log.",
        f"# Scaffold (render functions, schemas, helpers) = prompts.py @ {SCAFFOLD_COMMIT}.",
        f"# These overrides pin the exact system prompts this run used; the run's",
        f"# working tree had uncommitted edits to {len(systems)-matched} of them.",
        "# " + "=" * 72,
    ]
    for name in sorted(systems):
        lines.append(f"{name} = {systems[name]!r}")
    out_src = src + "\n".join(lines) + "\n"
    OUT.write_text(out_src)
    print(f"\nwrote {OUT}  ({len(out_src)} bytes)")


if __name__ == "__main__":
    main()
