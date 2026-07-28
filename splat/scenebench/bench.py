"""Planner A/B benchmark driver (Stage 3/4 redesign verification).

Builds shadow bench cells under runs/_bench4/<slot>--<variant>/<model> that
share the source cell's freespace grid + mesh tier but carry variant clouds:

  * old — legacy Stage 3 spacing (texel refinement disabled) + legacy Stage 4
          flags (texel_feat_px=0, fair_patches=False, exterior_shells=False)
  * new — texel-anchored Stage 3 (+ stex sidecar) + new Stage 4 defaults

Phases (run in the right venv):
  gen   (splat/.venv)  — stage 3 per variant into the bench cells
  run   (server/.venv) — push + spawn stages on Modal per variant
  watch (server/.venv) — poll job status; print stage-4 summaries when done

Usage:
  splat/.venv/bin/python -m splat.scenebench.bench gen  <src_cell> [--tier DIR]
  server/.venv/bin/python -m splat.scenebench.bench run <src_cell> --stages 4 [--variants old,new]
  server/.venv/bin/python -m splat.scenebench.bench watch <src_cell>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

BENCH_RUN = "_bench4"
VARIANTS = ("old", "new")
PLAN_FLAGS = {
    "old": {"texel_feat_px": 0.0, "fair_patches": False, "exterior_shells": False},
    "new": {},
}


def _bench_cell(src_cell: Path, variant: str) -> Path:
    run_dir = src_cell.parents[2]
    slot, model = src_cell.parts[-2], src_cell.parts[-1]
    return run_dir.parent / BENCH_RUN / f"{slot}--{variant}" / model


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _resolve_tier(src_cell: Path, override: str | None) -> Path:
    if override:
        return src_cell / override
    for name in ("objects-generated-lite", "objects-generated-optimized",
                 "objects-generated", "objects", "objects-optimized"):
        d = src_cell / name
        if d.is_dir() and list(d.glob("*.glb")):
            return d
    raise FileNotFoundError(f"no mesh tier under {src_cell}")


def cmd_gen(src_cell: Path, tier: str | None) -> None:
    import numpy as np  # noqa: F401  (env check)

    import splat.stage3 as stage3
    from splat.stage3 import SampleParams, sample_cell

    tier_dir = _resolve_tier(src_cell, tier)
    fs = src_cell / "splat" / "freespace.npz"
    if not fs.is_file():
        raise FileNotFoundError(f"{fs} missing — run stage 2 first")

    for variant in VARIANTS:
        cell = _bench_cell(src_cell, variant)
        splat_dir = cell / "splat"
        splat_dir.mkdir(parents=True, exist_ok=True)
        _link(fs, splat_dir / "freespace.npz")
        _link(src_cell / "splat" / "freespace.npz.skin.npy",
              splat_dir / "freespace.npz.skin.npy")
        # Mesh tier: hardlink the placed GLBs (content-addressed upload dedupes
        # anyway; the link keeps local disk flat).
        tier_dst = cell / tier_dir.name
        for glb in tier_dir.glob("*.glb"):
            if not glb.name.endswith(".raw.glb"):
                _link(glb, tier_dst / glb.name)

        out = splat_dir / "cloud.ply"
        mult_prev = stage3._TEXEL_SPACING_MULT
        stage3._TEXEL_SPACING_MULT = float("inf") if variant == "old" else mult_prev
        t0 = time.perf_counter()
        try:
            # workers=1: the variant switch is a module-global monkey-patch,
            # which spawn workers would not inherit.
            summary = sample_cell(
                run=BENCH_RUN, slot=cell.parts[-2], model=cell.parts[-1],
                raw_dir=tier_dst, freespace_path=splat_dir / "freespace.npz",
                out_path=out, params=SampleParams(),
                progress=lambda d, t, m: None, workers=1,
            )
        finally:
            stage3._TEXEL_SPACING_MULT = mult_prev
        if variant == "old":
            # Legacy cells must plan through the radius proxy even if a sidecar
            # got written — remove it so the old variant is faithful end-to-end.
            (splat_dir / "cloud.ply.stex.npy").unlink(missing_ok=True)
        (splat_dir / "stage3.json").write_text(json.dumps(summary, indent=1))
        print(f"[{variant}] {cell}: splats={summary['splats']} "
              f"({time.perf_counter() - t0:.1f}s, {summary['bytes'] / 1e6:.1f} MB)",
              flush=True)


def cmd_run(src_cell: Path, stages: list[int], variants: list[str],
            train_json: str, force: bool) -> None:
    from splat.modal_sync import push_cell, spawn_cell

    for variant in variants:
        cell = _bench_cell(src_cell, variant)
        tier_dirs = [d for d in cell.iterdir() if d.name.startswith("objects")]
        pushed = push_cell(cell, tier_dirs[0] if tier_dirs else None)
        call_id = spawn_cell(
            cell, pushed, stages,
            plan=PLAN_FLAGS[variant],
            train=json.loads(train_json),
            force=force,
        )
        print(f"[{variant}] spawned {call_id} stages={stages}", flush=True)


def cmd_watch(src_cell: Path, variants: list[str]) -> None:
    from splat.modal_sync import job_status, pull_cell

    pending = {v: _bench_cell(src_cell, v) for v in variants}
    while pending:
        done = []
        for v, cell in pending.items():
            st = job_status(cell)
            if st["state"] == "running":
                hb = st.get("heartbeat") or {}
                print(f"[{v}] running: [{hb.get('stage')}] "
                      f"{hb.get('done')}/{hb.get('total')} {str(hb.get('msg', ''))[:90]}",
                      flush=True)
                continue
            done.append(v)
            if st["state"] == "failed":
                print(f"[{v}] FAILED: {st.get('error')}", flush=True)
                continue
            pull_cell(cell, include_ply=False, quiet=True)
            status = json.loads((cell / "splat" / "status.json").read_text())
            for k, rec in status["stages"].items():
                s = rec.get("summary", {})
                if k == "4":
                    print(f"[{v}] stage4: patches={s.get('patches')} views={s.get('views')} "
                          f"cameras={s.get('cameras')} ext_cands={s.get('exterior_candidates')} "
                          f"ext_cams={s.get('cameras_exterior')} "
                          f"satisfied={s.get('coverage', {}).get('satisfied_pct')}%", flush=True)
                elif k == "5":
                    print(f"[{v}] stage5: rendered={s.get('rendered')} in {s.get('seconds')}s",
                          flush=True)
                elif k == "6":
                    print(f"[{v}] stage6: splats {s.get('splats_init')}->{s.get('splats_final')} "
                          f"metrics={s.get('metrics')}", flush=True)
        for v in done:
            pending.pop(v)
        if pending:
            time.sleep(20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("gen", "run", "watch"))
    ap.add_argument("src_cell", type=Path)
    ap.add_argument("--tier", default=None)
    ap.add_argument("--stages", default="4")
    ap.add_argument("--variants", default="old,new")
    ap.add_argument("--train-json", default="{}")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    stages = [int(s) for s in str(args.stages).replace("-", ",").split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    if args.cmd == "gen":
        cmd_gen(args.src_cell, args.tier)
    elif args.cmd == "run":
        cmd_run(args.src_cell, stages, variants, args.train_json, args.force)
    else:
        cmd_watch(args.src_cell, variants)


if __name__ == "__main__":
    main()
