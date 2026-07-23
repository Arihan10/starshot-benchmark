"""Client side of the Modal splat pipeline (`splat/modal_app.py`).

Everything the Mac does to run stages 4-7 remotely, importable by the server
and usable standalone:

    python -m splat.modal_sync probe                          # WebGL renderer on the A100
    python -m splat.modal_sync run    --cell runs/R/S/M       # push + spawn + record job
    python -m splat.modal_sync watch  --cell runs/R/S/M       # follow heartbeat, pull when done
    python -m splat.modal_sync status --cell runs/R/S/M
    python -m splat.modal_sync pull   --cell runs/R/S/M [--no-ply]

TRANSMISSION (the whole point):
  * The mesh tier uploads into a CONTENT-ADDRESSED store (`objects/{sha}.glb`
    on the Volume) — only blobs the Volume has never seen are sent, so the
    ~0.2-1 GB tier is paid once per unique object, and re-runs / hardlinked
    branch cells / shared library objects upload nothing. A local hash cache
    (keyed path+size+mtime) makes the no-op push take seconds.
  * `cloud.ply` (float32 tables, ~1.6-2x) and `freespace.npz.skin.npy`
    (bitmasks, ~3-5x) are zstd-compressed for transport; `freespace.npz` is
    already deflate and GLBs are already KTX2/Meshopt-coded, so they go as-is.
  * Inputs are hashed UNCOMPRESSED; `inputs/manifest.json` on the Volume lets
    an unchanged input skip its upload entirely.
  * Stage artifacts come back selectively: plan + status + `.sqz` always,
    `trained.ply` (+ LODs) optional — refs never leave the datacenter.

The tier defaults to the same resolution order the server's splat source uses
(generated lite → generated optimized → generated raw → library objects); pass
`--tier` (or `tier_dir=`) to pin one explicitly, e.g. when the cell's
`splat/source.json` pref says library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from splat.modal_app import APP_NAME, STATUS_DICT_NAME, VOLUME_NAME

_HASH_CACHE = Path.home() / ".cache" / "starshot-splat-hashes.json"
_JOB_NAME = "modal-job.json"
# Mirrors the server's auto source resolution (routes `_splat_source`).
_TIER_CANDIDATES = (
    "objects-generated-lite",
    "objects-generated-optimized",
    "objects-generated",
    "objects",
    "objects-optimized",
)
# Artifacts pulled back after a run; refs stay remote.
_PULL_ALWAYS = ("cameras.json", "patches.bin", "patch_views.json", "status.json")


def _zstd():
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "modal_sync needs the `zstandard` package (the server env has it: "
            "run via `uv run` in server/, or pip install zstandard)"
        ) from exc
    return zstandard


def _volume() -> modal.Volume:
    # Must match modal_app's version (v2) — the same named Volume.
    return modal.Volume.from_name(VOLUME_NAME, version=2, create_if_missing=True)


# --- hashing (cached) --------------------------------------------------------------


def _load_hash_cache() -> dict[str, str]:
    try:
        return json.loads(_HASH_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_hash_cache(cache: dict[str, str]) -> None:
    _HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _HASH_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(_HASH_CACHE)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _hash_files(paths: list[Path]) -> dict[Path, str]:
    """sha256 per file, through the mtime+size cache (a 1 GB tier re-hashes
    only what changed since the last push)."""
    cache = _load_hash_cache()
    out: dict[Path, str] = {}
    dirty = False
    for p in paths:
        st = p.stat()
        key = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        sha = cache.get(key)
        if sha is None:
            sha = _sha256_file(p)
            cache[key] = sha
            dirty = True
        out[p] = sha
    if dirty:
        _save_hash_cache(cache)
    return out


# --- cell / tier resolution ---------------------------------------------------------


def _cell_parts(cell_dir: Path) -> tuple[str, str, str]:
    parts = cell_dir.resolve().parts
    if len(parts) < 3:
        raise ValueError(f"{cell_dir} is not a runs/<run>/<slot>/<model> cell dir")
    return parts[-3], parts[-2], parts[-1]


def _placed_glbs(tier_dir: Path) -> list[Path]:
    return sorted(
        p for p in tier_dir.glob("*.glb") if not p.name.endswith(".raw.glb")
    )


def resolve_tier_dir(cell_dir: Path, override: Path | None = None) -> Path:
    if override is not None:
        if not _placed_glbs(override):
            raise FileNotFoundError(f"no placed .glb meshes in {override}")
        return override
    for name in _TIER_CANDIDATES:
        d = cell_dir / name
        if d.is_dir() and _placed_glbs(d):
            return d
    raise FileNotFoundError(
        f"no mesh tier found under {cell_dir} (tried {', '.join(_TIER_CANDIDATES)})"
    )


# --- push ---------------------------------------------------------------------------


def _compress_to_tmp(zstandard, src: Path) -> Path:
    # Close the fd mkstemp opened (via os.fdopen) — a leaked handle blocks the
    # later unlink on Windows (WinError 32); POSIX tolerates unlinking it.
    fd, name = tempfile.mkstemp(suffix=".zst")
    with os.fdopen(fd, "wb") as fo, src.open("rb") as fi:
        zstandard.ZstdCompressor(level=3).copy_stream(fi, fo)
    return Path(name)


def _existing_cas_shas(vol: modal.Volume) -> set[str]:
    try:
        entries = vol.listdir("/objects", recursive=True)
    except Exception:
        return set()  # volume is brand new — nothing stored yet
    return {
        Path(e.path).name[: -len(".glb")]
        for e in entries
        if e.path.endswith(".glb")
    }


def _remote_inputs_manifest(vol: modal.Volume, cell_key: str) -> dict[str, str]:
    try:
        data = b"".join(vol.read_file(f"cells/{cell_key}/inputs/manifest.json"))
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}


def push_cell(
    cell_dir: Path, tier_dir: Path | None = None, quiet: bool = False
) -> dict[str, Any]:
    """Upload one cell's stage-4-onward inputs (deduped, compressed where it
    pays). Returns {run, slot, model, input_sha:{freespace, skin, cloud, tier}}
    — exactly the identity block `run_cell` needs."""
    zstandard = _zstd()
    run, slot, model = _cell_parts(cell_dir)
    cell_key = f"{run}/{slot}/{model}"
    tier = resolve_tier_dir(cell_dir, tier_dir)
    splat_dir = cell_dir / "splat"

    freespace = splat_dir / "freespace.npz"
    skin = splat_dir / "freespace.npz.skin.npy"
    cloud = splat_dir / "cloud.ply"
    for p in (freespace, skin, cloud):
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing — run stages 2-3 locally first")

    def log(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    glbs = _placed_glbs(tier)
    t0 = time.perf_counter()
    hashes = _hash_files([*glbs, freespace, skin, cloud])
    tier_manifest = {p.name[: -len(".glb")]: hashes[p] for p in glbs}
    tier_sha = hashlib.sha256(
        json.dumps(tier_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    input_sha = {
        "freespace": hashes[freespace],
        "skin": hashes[skin],
        "cloud": hashes[cloud],
        "tier": tier_sha,
    }
    log(f"hashed {len(glbs)} meshes + 3 inputs in {time.perf_counter() - t0:.1f}s")

    vol = _volume()
    have = _existing_cas_shas(vol)
    missing = [p for p in glbs if hashes[p] not in have]
    remote_manifest = _remote_inputs_manifest(vol, cell_key)

    inputs_prefix = f"/cells/{cell_key}/inputs"
    tmp_files: list[Path] = []
    uploaded_bytes = 0
    try:
        with vol.batch_upload(force=True) as batch:
            for p in missing:
                sha = hashes[p]
                batch.put_file(p, f"/objects/{sha[:2]}/{sha}.glb")
                uploaded_bytes += p.stat().st_size

            def put_json(payload: Any, remote: str) -> None:
                # Close mkstemp's fd (see _compress_to_tmp) so the finally-unlink
                # doesn't hit WinError 32 on Windows.
                fd, name = tempfile.mkstemp(suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=1)
                tmp = Path(name)
                tmp_files.append(tmp)
                batch.put_file(tmp, remote)

            sent: list[str] = []
            # cloud + skin travel zstd-compressed; freespace.npz is already
            # deflate. Unchanged inputs (per the remote manifest) are skipped.
            for src, name, compress in (
                (freespace, "freespace.npz", False),
                (skin, "freespace.npz.skin.npy", True),
                (cloud, "cloud.ply", True),
            ):
                if remote_manifest.get(name) == hashes[src]:
                    continue
                if compress:
                    tmp = _compress_to_tmp(zstandard, src)
                    tmp_files.append(tmp)
                    batch.put_file(tmp, f"{inputs_prefix}/{name}.zst")
                    uploaded_bytes += tmp.stat().st_size
                else:
                    batch.put_file(src, f"{inputs_prefix}/{name}")
                    uploaded_bytes += src.stat().st_size
                sent.append(name)

            put_json(tier_manifest, f"{inputs_prefix}/tier.json")
            put_json(
                {
                    "freespace.npz": hashes[freespace],
                    "freespace.npz.skin.npy": hashes[skin],
                    "cloud.ply": hashes[cloud],
                    "tier": tier_sha,
                    "tier_dir": tier.name,
                    "pushed_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                },
                f"{inputs_prefix}/manifest.json",
            )
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)

    log(
        f"pushed {len(missing)}/{len(glbs)} meshes "
        f"({len(glbs) - len(missing)} deduped) + {len(sent)} input file(s), "
        f"{uploaded_bytes / 1e6:.1f} MB in {time.perf_counter() - t0:.1f}s"
    )
    return {"run": run, "slot": slot, "model": model, "input_sha": input_sha,
            "tier_dir": tier.name}


# --- spawn / status / pull -----------------------------------------------------------


def _job_path(cell_dir: Path) -> Path:
    return cell_dir / "splat" / _JOB_NAME


def spawn_cell(
    cell_dir: Path,
    pushed: dict[str, Any],
    stages: list[int],
    plan: dict[str, Any] | None = None,
    train: dict[str, Any] | None = None,
    quant: dict[str, Any] | None = None,
    force: bool = False,
) -> str:
    """Spawn `run_cell` for a pushed cell; record the call id beside the local
    splat artifacts. Returns the Modal function-call id."""
    spec = {
        "run": pushed["run"], "slot": pushed["slot"], "model": pushed["model"],
        "stages": stages,
        "input_sha": pushed["input_sha"],
        "plan": plan or {},
        "train": train or {},
        "quant": quant or {},
        "force": force,
    }
    fn = modal.Function.from_name(APP_NAME, "run_cell")
    call = fn.spawn(spec)
    record = {
        "call_id": call.object_id,
        "spec": spec,
        "spawned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = _job_path(cell_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return call.object_id


def job_status(cell_dir: Path) -> dict[str, Any]:
    """{state: running|done|failed, heartbeat, result|error} for the cell's
    recorded job. `heartbeat` is the live stage/done/total the container last
    published (survives across retries; `t` is its wall-clock timestamp)."""
    record = json.loads(_job_path(cell_dir).read_text(encoding="utf-8"))
    run, slot, model = _cell_parts(cell_dir)
    out: dict[str, Any] = {"call_id": record["call_id"]}
    try:
        out["heartbeat"] = modal.Dict.from_name(STATUS_DICT_NAME).get(
            f"{run}/{slot}/{model}"
        )
    except Exception:
        out["heartbeat"] = None
    call = modal.FunctionCall.from_id(record["call_id"])
    try:
        out["result"] = call.get(timeout=0)
        out["state"] = "done"
    except TimeoutError:
        out["state"] = "running"
    except Exception as exc:
        out["state"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def cancel_cell(cell_dir: Path) -> str | None:
    """Cancel the cell's recorded Modal function call, TERMINATING its container
    so the A100 stops billing immediately (not just draining its input queue).
    Returns the cancelled call id, or None when there's no job record / call id.
    The local `modal-job.json` is left in place for the caller to remove — the
    detach flow drops it so the cell reads idle and re-launches clean."""
    path = _job_path(cell_dir)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    call_id = record.get("call_id")
    if not call_id:
        return None
    call = modal.FunctionCall.from_id(call_id)
    try:
        call.cancel(terminate_containers=True)
    except TypeError:  # older Modal SDK without the kwarg
        call.cancel()
    return call_id


def pull_samples(cell_dir: Path) -> list[str]:
    """Download the mid-run debug sample PNGs (`refs/samples/*.png`) into the
    local `splat/refs/samples/` dir. Cheap + idempotent (skips same-size files);
    returns the names present locally after the pull. Empty when none exist yet."""
    run, slot, model = _cell_parts(cell_dir)
    vol = _volume()
    remote = f"cells/{run}/{slot}/{model}/splat/refs/samples"
    local = cell_dir / "splat" / "refs" / "samples"
    try:
        entries = {Path(e.path).name: e for e in vol.listdir(f"/{remote}")}
    except Exception:
        return []  # samples dir not created yet
    local.mkdir(parents=True, exist_ok=True)
    for name in sorted(entries):
        if not name.endswith(".png"):
            continue
        dst = local / name
        size = getattr(entries[name], "size", None)
        if size is not None and dst.is_file() and dst.stat().st_size == size:
            continue
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in vol.read_file(f"{remote}/{name}"):
                f.write(chunk)
        tmp.replace(dst)
    return sorted(p.name for p in local.glob("*.png"))


def pull_cell(cell_dir: Path, include_ply: bool = True, quiet: bool = False) -> list[str]:
    """Download the run's artifacts into the local `splat/` dir: plan + status
    + every `.sqz`, plus the plys unless `include_ply` is off — the RAW Stage-6
    `trained.ply` AND the delivered Stage-7 `healed.ply` + its LOD ladder, so the
    viewer can show trained vs healed side by side. Skips same-size files."""
    run, slot, model = _cell_parts(cell_dir)
    vol = _volume()
    remote = f"cells/{run}/{slot}/{model}/splat"
    local = cell_dir / "splat"
    local.mkdir(parents=True, exist_ok=True)
    entries = {Path(e.path).name: e for e in vol.listdir(f"/{remote}")}

    wanted = [n for n in _PULL_ALWAYS if n in entries]
    wanted += [n for n in entries if n.endswith(".sqz")]
    if include_ply:
        # trained.ply (raw, stage 6) + healed.ply and both LOD ladders (stage 7).
        wanted += [
            n for n in entries
            if n in ("trained.ply", "healed.ply")
            or (n.endswith(".ply") and (".lod" in n) and n.startswith(("trained", "healed")))
        ]

    got: list[str] = []
    for name in wanted:
        dst = local / name
        size = getattr(entries[name], "size", None)
        if size is not None and dst.is_file() and dst.stat().st_size == size:
            continue
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in vol.read_file(f"{remote}/{name}"):
                f.write(chunk)
        tmp.replace(dst)
        got.append(name)
        if not quiet:
            print(f"pulled {name} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
    return got


# --- CLI -----------------------------------------------------------------------------


def _parse_stages(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-", 1)
        stages = list(range(int(a), int(b) + 1))
    else:
        stages = [int(s) for s in text.split(",") if s]
    bad = [s for s in stages if s not in (4, 5, 6, 7)]
    if bad or not stages:
        raise argparse.ArgumentTypeError(f"stages must be within 4-7, got {text!r}")
    return stages


def _fmt_heartbeat(hb: dict[str, Any] | None) -> str:
    if not hb:
        return "no heartbeat yet"
    age = time.time() - float(hb.get("t", 0))
    pct = 100.0 * hb["done"] / max(hb["total"], 1)
    return (
        f"[{hb['stage']}] {hb['done']}/{hb['total']} ({pct:4.1f}%) "
        f"{hb.get('msg', '')} ({age:.0f}s ago)"
    )


def _main() -> None:
    ap = argparse.ArgumentParser(description="Run splat stages 4-7 on Modal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_cell(p: argparse.ArgumentParser, tier: bool = False) -> None:
        p.add_argument("--cell", required=True, type=Path,
                       help="runs/<run>/<slot>/<model> directory")
        if tier:
            p.add_argument("--tier", type=Path, default=None,
                           help="mesh tier dir (default: auto, lite first)")

    p = sub.add_parser("push", help="upload inputs (deduped) without running")
    add_cell(p, tier=True)

    p = sub.add_parser("run", help="push + spawn stages on the A100")
    add_cell(p, tier=True)
    p.add_argument("--stages", type=_parse_stages, default=[4, 5, 6, 7],
                   help="e.g. 4-7, 4, or 6,7 (default 4-7)")
    p.add_argument("--plan-json", default="{}", help="PlanParams overrides")
    p.add_argument("--train-json", default="{}", help="TrainParams overrides")
    p.add_argument("--quant-json", default="{}", help="QuantConfig overrides")
    p.add_argument("--force", action="store_true",
                   help="re-run the stage window even when signatures match")

    p = sub.add_parser("status", help="one-shot job state + heartbeat")
    add_cell(p)

    p = sub.add_parser("watch", help="follow the heartbeat; pull when done")
    add_cell(p)
    p.add_argument("--no-pull", action="store_true")
    p.add_argument("--no-ply", action="store_true",
                   help="skip trained.ply/LOD plys when pulling")

    p = sub.add_parser("pull", help="download artifacts (plan, sqz, ply)")
    add_cell(p)
    p.add_argument("--no-ply", action="store_true")

    p = sub.add_parser(
        "cancel", help="cancel the cell's running Modal job (terminates the A100 container)"
    )
    add_cell(p)

    sub.add_parser("probe", help="report the A100 container's WebGL renderer")

    args = ap.parse_args()

    if args.cmd == "probe":
        out = modal.Function.from_name(APP_NAME, "probe_webgl").remote()
        print(json.dumps(out, indent=1))
        return

    if args.cmd == "push":
        push_cell(args.cell, args.tier)
        return

    if args.cmd == "run":
        pushed = push_cell(args.cell, args.tier)
        call_id = spawn_cell(
            args.cell, pushed, args.stages,
            plan=json.loads(args.plan_json),
            train=json.loads(args.train_json),
            quant=json.loads(args.quant_json),
            force=args.force,
        )
        print(f"spawned {call_id} (stages {args.stages}) — "
              f"`watch --cell {args.cell}` to follow")
        return

    if args.cmd == "status":
        print(json.dumps(job_status(args.cell), indent=1, default=str))
        return

    if args.cmd == "watch":
        while True:
            st = job_status(args.cell)
            if st["state"] == "running":
                print(_fmt_heartbeat(st.get("heartbeat")), flush=True)
                time.sleep(10)
                continue
            if st["state"] == "failed":
                print(f"job failed: {st.get('error')}", file=sys.stderr)
                raise SystemExit(1)
            print(json.dumps(st["result"], indent=1))
            if not args.no_pull:
                pull_cell(args.cell, include_ply=not args.no_ply)
            return

    if args.cmd == "pull":
        pull_cell(args.cell, include_ply=not args.no_ply)
        return

    if args.cmd == "cancel":
        cid = cancel_cell(args.cell)
        print(f"cancelled {cid}" if cid else "no recorded Modal job to cancel")


if __name__ == "__main__":
    _main()
