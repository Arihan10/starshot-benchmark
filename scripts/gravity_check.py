#!/usr/bin/env python3
"""gravity_check.py — validate the XML-gravity tag moving / removal + span-map
logic against REAL prompts sampled from runs.

It loads a cell's `cache.llm` firings, finds the ones whose prompt carries a
`<VERY_IMPORTANT_INSTRUCTIONS>` block, and for each gravity level (none, q1..q4)
runs the ACTUAL `app.ablation.gravity.rewrite_and_stash` on the exact user text
the model saw — then checks the invariants the whole experiment relies on:

  * REMOVAL — the production `<VERY_IMPORTANT_INSTRUCTIONS>` tags are gone.
  * INSERTION — qN adds exactly one neutral `<prompt>` (block start) + one
    `</prompt>` (after a sentence near the j/4 word mark); `none` adds no tags.
  * SPAN-MAP round-trip — every SENTENCE / tag span in the logged `__GRAVITY__`
    map decodes back to the right text (no offset drift), sentence spans are
    ordered, and no sentence straddles the closing tag.
  * PURITY — only the block changes; the text before/after it is byte-identical.
  * STABLE PARTITION — the per-sentence split is identical across every level, so
    the baseline (`none`) subtraction aligns sentence-for-sentence.
  * REVERSIBILITY — stripping the inserted neutral tags recovers the `none` block.

This exercises the SAME functions the pipeline calls (`app.ablation.gravity`), so
a pass means the refactor + tag logic hold on the prompts really in the runs.

Usage:
    # interactive: pick run / slot / model, check every VII firing in the cell
    gravity_check.py

    # address one cell / firing (verbose; --show also prints the rewrites)
    gravity_check.py --run MINGLET --slot modern-house --model gemma --step object_bbox_batch --show

    # list the VII-bearing firings in a cell
    gravity_check.py --run MINGLET --slot modern-house --model gemma --list

    # SWEEP many runs (cross-run coverage) and aggregate pass/fail — the main check
    gravity_check.py --sample 120
    gravity_check.py --all            # every run/cell/firing (reads full logs; slow)

    # synthetic self-test, needs no run
    gravity_check.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))     # app.ablation.* (stdlib-only, no server deps)
sys.path.insert(0, str(SCRIPTS_DIR))    # reuse scene_schema's run/event discovery

import scene_schema as ss  # noqa: E402
from app.ablation import config  # noqa: E402
from app.ablation import context as abl_ctx  # noqa: E402
from app.ablation import gravity  # noqa: E402

MODES = ["none", "q1", "q2", "q3", "q4"]
_OPEN = "<VERY_IMPORTANT_INSTRUCTIONS>"
_CLOSE = "</VERY_IMPORTANT_INSTRUCTIONS>"


# --------------------------------------------------------------------------- #
# Firing / prompt selection
# --------------------------------------------------------------------------- #
def firing_prompt(e: dict) -> tuple[str | None, str]:
    """`(field, text)` for whichever prompt field carries a VII block (`user`
    first — that's where production puts it), or `(None, "")`."""
    for field in ("user", "system"):
        t = str(e.get(field) or "")
        if _OPEN in t and _CLOSE in t:
            return field, t
    return None, ""


def vii_firings(events: list[dict]) -> list[tuple[dict, str, str]]:
    """Every `cache.llm` firing that carries a VII block: (event, field, text)."""
    out = []
    for e in ss.llm_firings(events):
        field, t = firing_prompt(e)
        if field:
            out.append((e, field, t))
    return out


def firing_template(e: dict) -> str:
    return str(e.get("template") or e.get("step") or "step")


# --------------------------------------------------------------------------- #
# The rewrite (through the REAL bound-treatment entry point)
# --------------------------------------------------------------------------- #
def rewrite(user: str, template: str, mode: str) -> tuple[str, dict]:
    """Bind a gravity treatment for `template` and run the real
    `gravity.rewrite_and_stash`, returning `(new_user, span_map)`."""
    abl_ctx.set_runtime(config.AblationRuntime(
        target_step_kind=template, treatment=config.Treatment(gravity_mode=mode)))
    try:
        variables: dict = {}
        new_user = gravity.rewrite_and_stash(user, variables, template)
        smap = json.loads(variables.get(gravity.ROLES_VAR) or "{}")
        return new_user, smap
    finally:
        abl_ctx.clear()


def _strip_neutral_tags(text: str) -> str:
    """Undo the tag insertion: gravity emits `<prompt>\\n` at the block start and
    `\\n</prompt>` after quarter j, so removing exactly those recovers the block."""
    return text.replace(gravity.OPEN_TAG + "\n", "", 1).replace("\n" + gravity.CLOSE_TAG, "", 1)


# --------------------------------------------------------------------------- #
# The core verification for one firing
# --------------------------------------------------------------------------- #
def check_firing(user: str, template: str) -> tuple[list[str], dict]:
    """Run every invariant across all levels for one firing. Returns
    `(problems, info)` — `problems` empty == all checks passed."""
    problems: list[str] = []

    def chk(cond: bool, msg: str) -> bool:
        if not cond:
            problems.append(msg)
        return bool(cond)

    loc = gravity._locate_block(user)
    if loc is None:
        return ["could not locate the VII block (tags malformed?)"], {}
    o_start, inner_start, inner_end, c_end = loc
    prefix, inner, suffix = user[:o_start], user[inner_start:inner_end], user[c_end:]

    per_mode: dict[str, dict] = {}
    sent_texts: dict[str, list[str]] = {}
    close_after: dict[str, int] = {}   # mode -> #sentences before the closing tag
    none_user = None
    for mode in MODES:
        new_user, smap = rewrite(user, template, mode)
        ss = smap.get("sentences", [])
        per_mode[mode] = {"n_sent": len(ss), "new_user": new_user, "smap": smap}

        chk(_OPEN not in new_user and _CLOSE not in new_user, f"{mode}: production VII tags removed")
        chk(len(ss) >= 1, f"{mode}: >=1 sentence recorded (got {len(ss)})")
        chk(new_user.startswith(prefix), f"{mode}: text BEFORE the block is byte-identical")
        chk(new_user.endswith(suffix), f"{mode}: text AFTER the block is byte-identical")

        texts, starts, ok_spans = [], [], True
        for s in ss:
            seg = new_user[s["start"]:s["end"]]
            if not seg.strip():
                ok_spans = False
            texts.append(seg.strip())
            starts.append(s["start"])
        chk(ok_spans, f"{mode}: every sentence span decodes to non-empty text")
        chk(starts == sorted(starts), f"{mode}: sentence spans are in reading order")
        sent_texts[mode] = texts

        if mode == "none":
            none_user = new_user
            chk(gravity.OPEN_TAG not in new_user and gravity.CLOSE_TAG not in new_user, "none: no neutral tags inserted")
        else:
            ot, ct = smap.get("open_tag"), smap.get("close_tag")
            chk(new_user.count(gravity.OPEN_TAG) == 1 and new_user.count(gravity.CLOSE_TAG) == 1, f"{mode}: exactly one <prompt> pair")
            chk(bool(ot) and new_user[ot[0]:ot[1]] == gravity.OPEN_TAG, f"{mode}: open_tag span == <prompt>")
            chk(bool(ct) and new_user[ct[0]:ct[1]] == gravity.CLOSE_TAG, f"{mode}: close_tag span == </prompt>")
            if ot:
                chk(ot[0] == len(prefix), f"{mode}: <prompt> opens at the block start")
            if ct:
                before = [s for s in ss if s["end"] <= ct[0]]
                after = [s for s in ss if s["start"] >= ct[1]]
                close_after[mode] = len(before)
                chk(len(before) >= 1, f"{mode}: </prompt> sits after >=1 sentence")
                chk(len(before) + len(after) == len(ss), f"{mode}: no sentence straddles </prompt>")

    # cross-mode: the SENTENCE partition is identical everywhere (stable anchor —
    # so the per-sentence baseline subtraction aligns sentence-for-sentence).
    ref = sent_texts.get("none")
    for mode in MODES:
        if mode != "none":
            chk(sent_texts[mode] == ref, f"{mode}: sentence partition matches `none` (stable across levels)")
    # the closing tag advances (or holds) q1 <= q2 <= q3 <= q4.
    order = [close_after[m] for m in ("q1", "q2", "q3", "q4") if m in close_after]
    chk(order == sorted(order), "closing tag advances across positions (q1<=q2<=q3<=q4)")
    # reversibility: stripping the neutral tags from each qN recovers `none`.
    if none_user is not None:
        for mode in MODES:
            if mode != "none":
                chk(_strip_neutral_tags(per_mode[mode]["new_user"]) == none_user, f"{mode}: stripping <prompt> tags recovers the `none` block")

    sents = gravity._sentences(inner)
    words = [len(inner[s:e].split()) for s, e in sents]
    info = {
        "field_len": len(user), "block_words": sum(words) or 0, "n_sentences": len(sents),
        "sent_words": words, "close_after": close_after, "per_mode": per_mode,
    }
    return problems, info


# --------------------------------------------------------------------------- #
# Displays
# --------------------------------------------------------------------------- #
def _preview(s: str, n: int = 60) -> str:
    s = " ".join(s.split())
    return (s[:n] + "…") if len(s) > n else s


def cmd_check_one(run: str, slot: str, model: str, event: dict, field: str, text: str, show: bool) -> int:
    tmpl = firing_template(event)
    print(f"# {run}/{slot}/{model}  step={tmpl}  node={event.get('node') or 'none'}  "
          f"index={event.get('index')}  (VII in `{field}`)")
    problems, info = check_firing(text, tmpl)
    if info:
        ca = info["close_after"]
        print(f"\nblock: {info['block_words']} words \u2192 {info['n_sentences']} sentences "
              f"(measurement unit) · closing tag after sentence "
              f"{{{', '.join(f'{m}:{ca[m]}' for m in ('q1', 'q2', 'q3', 'q4') if m in ca)}}}")
        for mode in MODES:
            pm = info["per_mode"].get(mode, {})
            sm = pm.get("smap", {})
            tags = ""
            if mode != "none" and sm.get("open_tag") and sm.get("close_tag"):
                tags = f"  <prompt>@{sm['open_tag'][0]} </prompt>@{sm['close_tag'][0]}"
            print(f"  {mode:<5} {pm.get('n_sent', 0)} sentences{tags}")
    print()
    if problems:
        print(f"FAIL — {len(problems)} check(s) did not pass:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("PASS — all tag-move / removal / span-map invariants hold for this firing.")
    if show and info:
        # Print ONLY the instruction (VII) block region — the logged `block` span —
        # not the whole (scene-context-heavy) prompt.
        for mode in MODES:
            pm = info["per_mode"][mode]
            blk = pm["smap"].get("block")
            section = pm["new_user"][blk[0]:blk[1]] if blk else "(block not located)"
            print(f"\n===== gravity_mode = {mode} \u2014 VII section =====")
            print(section)
    return 1 if problems else 0


def cmd_list(events: list[dict]) -> None:
    firings = vii_firings(events)
    print(f"{'idx':>5}  {'step/template':<28} {'node':<22} {'field':<7} block-preview")
    print("-" * 96)
    for e, field, t in firings:
        loc = gravity._locate_block(t)
        blk = t[loc[1]:loc[2]] if loc else ""
        print(f"{e.get('index', '?'):>5}  {firing_template(e):<28} {str(e.get('node') or '?'):<22} {field:<7} {_preview(blk, 48)}")
    print(f"\n{len(firings)} VII-bearing firing(s).")


# --------------------------------------------------------------------------- #
# Sweep (cross-run sampling)
# --------------------------------------------------------------------------- #
def cmd_sample(runs_dir: Path, total_cap: int, per_cell: int, run_cap: int, all_firings: bool) -> int:
    """Sweep runs (newest first), pull VII firings, and run the checks on each —
    aggregating pass/fail across the corpus. One cell per run (first with
    completions) and up to `per_cell` DISTINCT-template firings, for cross-run
    breadth without reading every giant events.jsonl."""
    runs = ss.runs_with_completions(runs_dir)
    try:
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    if run_cap and not all_firings:
        runs = runs[:run_cap]

    # Cells to visit: for --all, every (slot, model) with completions; for a
    # sample, we stop at the FIRST cell per run that actually has VII firings.
    def cells_of(run_name: str) -> list[tuple[str, str]]:
        out = []
        for s in ss.slots_with_completions(runs_dir, run_name):
            for m in ss.models_with_completions(runs_dir, run_name, s.name):
                out.append((s.name, m.name))
        return out

    checked = passed = runs_touched = 0
    fail_rows: list[tuple[str, list[str]]] = []
    kinds: dict[str, int] = {}
    for r in runs:
        if not all_firings and checked >= total_cap:
            break
        used_a_cell = False
        for sname, mname in cells_of(r.name):
            if not all_firings and (used_a_cell or checked >= total_cap):
                break
            events = ss.load_events(ss.cell_dir(runs_dir, r.name, sname, mname) / "events.jsonl")
            firings = vii_firings(events)
            if not firings:
                continue
            runs_touched += 1
            used_a_cell = True
            # For a sample, prefer distinct step templates (the block differs per
            # step) up to `per_cell`; for --all, take everything.
            picked, seen = [], set()
            for fr in firings:
                if all_firings:
                    picked.append(fr)
                    continue
                tmpl = firing_template(fr[0])
                if len(picked) >= per_cell or tmpl in seen:
                    continue
                seen.add(tmpl)
                picked.append(fr)
            for e, field, t in picked:
                if not all_firings and checked >= total_cap:
                    break
                tmpl = firing_template(e)
                problems, _ = check_firing(t, tmpl)
                checked += 1
                kinds[tmpl] = kinds.get(tmpl, 0) + 1
                label = f"{r.name}/{sname}/{mname} step={tmpl} idx={e.get('index')}"
                if problems:
                    fail_rows.append((label, problems))
                    print(f"FAIL  {label}")
                    for p in problems[:6]:
                        print(f"        - {p}")
                else:
                    passed += 1
                    print(f"ok    {label}  ({field})")

    print("\n" + "=" * 70)
    print(f"checked {checked} VII firing(s) across {runs_touched} cell(s) · step kinds: "
          + ", ".join(f"{k}\u00d7{v}" for k, v in sorted(kinds.items())))
    if fail_rows:
        print(f"\n{len(fail_rows)} FIRING(S) FAILED:")
        for label, probs in fail_rows:
            print(f"  - {label}: {probs[0]}{(' (+%d more)' % (len(probs) - 1)) if len(probs) > 1 else ''}")
        return 1
    print(f"ALL {passed} FIRING(S) PASSED — tag-move / removal / span-map hold on real prompts.")
    return 0


# --------------------------------------------------------------------------- #
# Synthetic self-test (no run needed)
# --------------------------------------------------------------------------- #
_SELFTEST_BLOCK = (
    "using the placement description, determine the concrete coordinates of each object.\n\n"
    "think very hard about the dimensions of the other regions already placed. reason spatially.\n\n"
    "treat predefined subregions as solid reserved spaces. slot your objects into the gaps.\n\n"
    "align the scale of the objects with the description. use the size of the region well.\n\n"
    "the bounding boxes should be defined via a minimum corner and dimensions.\n\n"
    "output valid JSON only. do not add commentary. be careful and detailed."
)
_SELFTEST_USER = (
    "<objects_to_place>\nplace the lamp\n</objects_to_place>\n\n"
    f"<VERY_IMPORTANT_INSTRUCTIONS>\n{_SELFTEST_BLOCK}\n</VERY_IMPORTANT_INSTRUCTIONS>\n"
)


def _selftest() -> int:
    print("gravity_check --selftest (synthetic prompt, real gravity module)\n")
    problems, info = check_firing(_SELFTEST_USER, "object_bbox_batch")
    print(f"block: {info.get('block_words')} words \u2192 {info.get('n_sentences')} sentences · "
          f"close after {info.get('close_after')}")
    for mode in MODES:
        print(f"  {mode:<5} ok")
    print()
    if problems:
        print(f"SELF-TEST FAILED — {len(problems)} check(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("SELF-TEST PASSED — all tag-move / removal / span-map invariants hold.")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run id (omit to pick interactively)")
    ap.add_argument("--slot", help="scene, e.g. modern-house")
    ap.add_argument("--model", help="e.g. gemma, qwen-122b")
    ap.add_argument("--step", help="pipeline step / template to check (omit = check every VII firing in the cell)")
    ap.add_argument("--node", default=None, help="target region id (optional --step filter)")
    ap.add_argument("--occurrence", type=int, default=-1, help="which matching firing for --step (default -1 = last)")
    ap.add_argument("--runs-dir", default=str(ss.DEFAULT_RUNS))
    ap.add_argument("-i", "--interactive", action="store_true", help="force interactive run/slot/model selection")
    ap.add_argument("--list", action="store_true", help="list the cell's VII-bearing firings and exit")
    ap.add_argument("--show", action="store_true", help="also print the rewritten prompt for each level")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="SWEEP runs and check up to N VII firings across cells (cross-run coverage)")
    ap.add_argument("--per-cell", type=int, default=3, help="max firings sampled per cell in --sample (distinct steps preferred)")
    ap.add_argument("--runs", type=int, default=25, help="max runs visited in --sample")
    ap.add_argument("--all", action="store_true", help="check EVERY VII firing in EVERY cell (reads full logs; slow)")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic self-test (no run needed) and exit")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    runs_dir = Path(args.runs_dir)

    if args.sample or args.all:
        return cmd_sample(runs_dir, args.sample or 10 ** 9, args.per_cell, args.runs, args.all)

    need_cell = args.run is None or args.slot is None or args.model is None
    interactive = args.interactive or (need_cell and sys.stdin.isatty())
    if interactive:
        print("gravity_check — interactive select  (Enter accepts the [default])")

    run = ss.resolve_dir_choice("run", args.run, ss.list_runs(runs_dir),
                                lambda: ss.runs_with_completions(runs_dir), interactive)
    slot = ss.resolve_dir_choice("slot", args.slot, ss.list_slots(runs_dir, run),
                                 lambda: ss.slots_with_completions(runs_dir, run), interactive)
    model = ss.resolve_dir_choice("model", args.model, ss.list_models(runs_dir, run, slot),
                                  lambda: ss.models_with_completions(runs_dir, run, slot), interactive)

    events = ss.load_events(ss.cell_dir(runs_dir, run, slot, model) / "events.jsonl")

    if args.list:
        cmd_list(events)
        return 0

    # A specific step, else every VII firing in the cell.
    if args.step is not None:
        event = ss.find_firing(events, args.step, args.node, args.occurrence)
        field, text = firing_prompt(event)
        if not field:
            raise SystemExit(f"the {args.step!r} firing has no <VERY_IMPORTANT_INSTRUCTIONS> block")
        return cmd_check_one(run, slot, model, event, field, text, args.show)

    firings = vii_firings(events)
    if not firings:
        raise SystemExit("no VII-bearing firings in this cell (nothing to check) — try --list on another cell")
    rc = 0
    for e, field, text in firings:
        rc |= cmd_check_one(run, slot, model, e, field, text, args.show)
        print("-" * 70)
    print("ALL FIRINGS PASSED" if rc == 0 else "SOME FIRINGS FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
