"""Client side of the Modal splat pipeline (`splat/modal_app.py`).

Everything the Mac does to run stages 4-7 remotely, importable by the server
and usable standalone:

    python -m splat.modal_sync probe                          # WebGL renderer on the GPU box
    python -m splat.modal_sync run    --cell runs/R/S/M       # push + spawn + record job
    python -m splat.modal_sync run    --cell runs/R/S/M --reuse captures   # retrain only
    python -m splat.modal_sync watch  --cell runs/R/S/M       # follow heartbeat, pull when done
    python -m splat.modal_sync status --cell runs/R/S/M
    python -m splat.modal_sync state  --cell runs/R/S/M       # what the Volume already holds
    python -m splat.modal_sync pull   --cell runs/R/S/M [--no-ply]

REUSE (retraining without re-doing the expensive stages):
  * Every stage's output already persists on the Volume, so a re-run can START
    from any of them. `remote_cell_state` reports what is there (inputs, camera
    plan, captures, trained model) and `remote_pushed` hands `spawn_cell` the
    identity block it would otherwise get from a push — so a run can skip the
    upload, skip stage 4, or skip stages 4+5 and train alone. Nothing local is
    read on that path, so a cell whose local artifacts are gone still retrains.
  * `push_cameras` is the one-file variant for the other direction: keep the
    Volume's inputs but render from the LOCAL camera plan.

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
import uuid
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
_PULL_ALWAYS = ("cameras.json", "status.json")
_CAMERAS_NAME = "cameras.json"
# The cell inputs `run_cell` materializes, mapping its `input_sha` key -> the
# file name (which is also the key it is hashed under in `inputs/manifest.json`).
# Transport may add a `.zst` suffix (see `push_cell`), so presence checks accept
# either spelling.
_INPUT_FILES = {
    "freespace": "freespace.npz",
    "skin": "freespace.npz.skin.npy",
    "cloud": "cloud.ply",
    "scene": "scene.json",
}
# Failures that mean "we could not ASK Modal", never "the run failed" — Modal's
# client transport raises these (a socket error or an RPC deadline inside
# `FunctionCall.get` is re-wrapped as its own ConnectionError), whereas a
# container that genuinely died delivers ITS exception class through the call.
# Reading one of these as a failure kills a healthy multi-hour train over a
# single dropped packet, so `job_status` reports them as `unreachable` instead.
_UNREACHABLE = (
    modal.exception.ConnectionError,
    modal.exception.ServiceError,
    modal.exception.InternalError,
    modal.exception.ClientClosed,
)


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


def _remote_entries(vol: modal.Volume, remote_dir: str) -> dict[str, Any]:
    """`{basename: entry}` for one Volume directory, NON-recursively; empty when
    the directory doesn't exist. Non-recursive matters: `splat/refs` holds tens of
    thousands of frames, and every caller here only needs the names beside them."""
    try:
        return {Path(e.path).name: e for e in vol.listdir(remote_dir)}
    except Exception:
        return {}


def _remote_json(vol: modal.Volume, remote: str) -> dict[str, Any]:
    try:
        return json.loads(b"".join(vol.read_file(remote)).decode("utf-8"))
    except Exception:
        return {}


def push_cell(
    cell_dir: Path, tier_dir: Path | None = None, quiet: bool = False,
    include_cameras: bool = False,
) -> dict[str, Any]:
    """Upload one cell's stage-4-onward inputs (deduped, compressed where it
    pays). Returns {run, slot, model, input_sha:{freespace, skin, cloud,
    scene, tier}} — exactly the identity block `run_cell` needs.

    `include_cameras` ALSO uploads the LOCAL Stage-4 plan (`splat/cameras.json`)
    to the cell's `splat/` dir on the Volume — where `run_cell`'s stage 5 reads
    it — so a locally-planned camera set drives the remote render and stage 4 is
    skipped ('continue on modal'). Without it, stage 4 (re)plans on the GPU box."""
    zstandard = _zstd()
    run, slot, model = _cell_parts(cell_dir)
    cell_key = f"{run}/{slot}/{model}"
    tier = resolve_tier_dir(cell_dir, tier_dir)
    splat_dir = cell_dir / "splat"

    freespace = splat_dir / "freespace.npz"
    skin = splat_dir / "freespace.npz.skin.npy"
    cloud = splat_dir / "cloud.ply"
    scene = splat_dir / "scene.json"
    cameras = splat_dir / "cameras.json"
    required = [freespace, skin, cloud, scene] + ([cameras] if include_cameras else [])
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} missing — run stages 1-{4 if include_cameras else 3} locally first"
            )

    def log(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    glbs = _placed_glbs(tier)
    t0 = time.perf_counter()
    hashes = _hash_files([*glbs, *required])
    tier_manifest = {p.name[: -len(".glb")]: hashes[p] for p in glbs}
    tier_sha = hashlib.sha256(
        json.dumps(tier_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    input_sha = {
        "freespace": hashes[freespace],
        "skin": hashes[skin],
        "cloud": hashes[cloud],
        "scene": hashes[scene],
        "tier": tier_sha,
    }
    if include_cameras:
        input_sha["cameras"] = hashes[cameras]
    log(f"hashed {len(glbs)} meshes + {len(required)} inputs in {time.perf_counter() - t0:.1f}s")

    vol = _volume()
    have = _existing_cas_shas(vol)
    # One put per unique BLOB, not per file. The destination is content-addressed,
    # so a scene that repeats geometry (a tiled facade here places ONE mesh 405
    # times) would enqueue the same remote path hundreds of times in a single
    # batch — Modal keys its per-file upload progress by path, so the copies race
    # on one shared record, and they PUT one identical block concurrently until
    # the remote resets the connection.
    missing: dict[str, Path] = {}
    for p in glbs:
        sha = hashes[p]
        if sha not in have:
            missing.setdefault(sha, p)
    remote_manifest = _remote_inputs_manifest(vol, cell_key)

    inputs_prefix = f"/cells/{cell_key}/inputs"
    tmp_files: list[Path] = []
    uploaded_bytes = 0
    try:
        with vol.batch_upload(force=True) as batch:
            for sha, p in missing.items():
                batch.put_file(p, f"/objects/{sha[:2]}/{sha}.glb")
                uploaded_bytes += p.stat().st_size

            # The local Stage-4 plan is a stage OUTPUT, not an input: it lands in
            # the cell's splat/ dir on the Volume (persistent stage state), where
            # run_cell's stage 5 reads `out/cameras.json`. Always re-sent (tiny),
            # so 'continue' always renders THIS local plan.
            if include_cameras:
                batch.put_file(cameras, f"/cells/{cell_key}/splat/cameras.json")
                uploaded_bytes += cameras.stat().st_size

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
                (scene, "scene.json", False),
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
                    "scene.json": hashes[scene],
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


def push_cameras(cell_dir: Path) -> str:
    """Upload JUST the local Stage-4 plan to the cell's `splat/` dir on the Volume
    — where `run_cell`'s stage 5 reads it. This is the one file the 'render from my
    LOCAL camera plan' path needs, so it can be sent without the full `push_cell`
    when the rest of the inputs are being reused. Returns the plan's sha256."""
    run, slot, model = _cell_parts(cell_dir)
    cameras = cell_dir / "splat" / _CAMERAS_NAME
    if not cameras.is_file():
        raise FileNotFoundError(f"{cameras} missing — run stage 4 locally first")
    vol = _volume()
    with vol.batch_upload(force=True) as batch:
        batch.put_file(cameras, f"/cells/{run}/{slot}/{model}/splat/{_CAMERAS_NAME}")
    return _hash_files([cameras])[cameras]


# --- reuse: what the Volume already holds -------------------------------------------


def remote_pushed(cell_dir: Path) -> dict[str, Any]:
    """The identity block `spawn_cell` normally gets from `push_cell`, read off the
    inputs ALREADY on the Volume — the seam that lets a re-run skip the upload
    entirely (no tier hashing, no CAS listing, no batch upload) and keeps
    `run_cell`'s `input_sha` describing what is genuinely there.

    Raises FileNotFoundError when the cell was never pushed (or the manifest is
    incomplete), so the caller can say so instead of failing inside the container
    when `_ensure_input` finds nothing."""
    run, slot, model = _cell_parts(cell_dir)
    cell_key = f"{run}/{slot}/{model}"
    vol = _volume()
    manifest = _remote_inputs_manifest(vol, cell_key)
    names = set(_remote_entries(vol, f"/cells/{cell_key}/inputs"))
    absent = [
        name
        for name in (*_INPUT_FILES.values(), "tier.json")
        if name not in names and f"{name}.zst" not in names
    ]
    unhashed = [name for name in _INPUT_FILES.values() if not manifest.get(name)]
    if absent or unhashed or not manifest.get("tier"):
        raise FileNotFoundError(
            f"cells/{cell_key}/inputs is not a complete push"
            + (f" — missing {', '.join(absent)}" if absent else "")
            + (f" — unhashed {', '.join(unhashed)}" if unhashed else "")
        )
    input_sha = {key: manifest[name] for key, name in _INPUT_FILES.items()}
    input_sha["tier"] = manifest["tier"]
    return {"run": run, "slot": slot, "model": model, "input_sha": input_sha,
            "tier_dir": manifest.get("tier_dir")}


def remote_cell_state(cell_dir: Path) -> dict[str, Any]:
    """What of this cell's pipeline is ALREADY on the Volume, per reusable stage:
    the pushed inputs, the Stage-4 camera plan, the Stage-5 captures and the
    Stage-6/7 models — each with the recorded summary + completion time. This is
    what the client's reuse toggles are gated and labelled on.

    Cheap by construction: three non-recursive listdirs + two small JSON reads. It
    NEVER reads `refs/transforms.json` (tens of MB at 60k frames) — capture counts
    come from the stage summaries `run_cell` recorded instead."""
    run, slot, model = _cell_parts(cell_dir)
    cell_key = f"{run}/{slot}/{model}"
    vol = _volume()
    inputs = _remote_entries(vol, f"/cells/{cell_key}/inputs")
    splat = _remote_entries(vol, f"/cells/{cell_key}/splat")
    refs = _remote_entries(vol, f"/cells/{cell_key}/splat/refs")
    manifest = _remote_inputs_manifest(vol, cell_key)
    tier = _remote_json(vol, f"cells/{cell_key}/inputs/tier.json")
    stages = _remote_json(vol, f"cells/{cell_key}/splat/status.json").get("stages") or {}

    def rec(n: int) -> dict[str, Any]:
        return (stages.get(str(n)) or {}).get("summary") or {}

    # Project each stage record down to the few facts a reuse decision needs. The
    # records themselves carry every dataclass param the stage ran with (kilobytes
    # each), and their field names differ by trainer and by vintage — brush counts
    # `splats`, the gsplat loop `splats_final`; captures gained `img_per_s` after
    # some of these cells were rendered — so read both spellings here rather than
    # in the UI, and don't ship what nobody reads.
    def at(n: int) -> str | None:
        return (stages.get(str(n)) or {}).get("done_at")

    cams, caps, tr = rec(4), rec(5), rec(6)
    heal = rec(7).get("heal") or {}
    missing = [
        name
        for name in (*_INPUT_FILES.values(), "tier.json")
        if name not in inputs and f"{name}.zst" not in inputs
    ]
    return {
        "cell": cell_key,
        "inputs": {
            "available": not missing and bool(manifest.get("tier")),
            "missing": missing,
            "pushed_at": manifest.get("pushed_at"),
            "tier_dir": manifest.get("tier_dir"),
            "meshes": len(tier) or None,
        },
        # The plan stage 5 renders from, and the captures stage 6 trains on.
        "cameras": {
            "available": _CAMERAS_NAME in splat,
            "at": at(4),
            "views": cams.get("cameras") or cams.get("views"),
        },
        "captures": {
            "available": "transforms.json" in refs,
            "at": at(5),
            "views": caps.get("views"),
            "seconds": caps.get("seconds"),
            "rate": (
                caps.get("img_per_s")
                or (caps.get("capture_stats") or {}).get("views_per_s")
            ),
        },
        "trained": {
            "available": "trained.ply" in splat,
            "at": at(6),
            "splats": tr.get("splats") or tr.get("splats_final"),
            "trainer": tr.get("trainer") or ("gsplat" if tr else None),
            "iterations": tr.get("iterations"),
        },
        "healed": {
            "available": "healed.ply" in splat,
            "at": at(7),
            "splats": heal.get("splats_final"),
        },
    }


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
    trainer: str = "brush",
    brush: dict[str, Any] | None = None,
) -> str:
    """Spawn `run_cell` for a pushed cell; record the call id beside the local
    splat artifacts. Returns the Modal function-call id.

    `trainer` picks the stage-6 back-end: "brush" (the upstream binary, tuned by
    `brush`) or "gsplat" (the in-house loop, tuned by `train`). The unused
    back-end's params ride along harmlessly — `run_cell` keys its cache on the
    selected one only.

    A `force` carries a per-spawn token, which is what makes it INVALIDATE-ONCE in
    the container: Modal retries a preempted run with this exact spec, and without
    the token the retry would drop the frames / checkpoint the first attempt
    produced and start over (see `run_cell`)."""
    spec = {
        "run": pushed["run"], "slot": pushed["slot"], "model": pushed["model"],
        "stages": stages,
        "input_sha": pushed["input_sha"],
        "trainer": trainer,
        "plan": plan or {},
        "train": train or {},
        "brush": brush or {},
        "quant": quant or {},
        "force": force,
    }
    if force:
        spec["force_token"] = uuid.uuid4().hex
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
    """{state: running|done|failed|unreachable, heartbeat, result|error} for the
    cell's recorded job. `heartbeat` is the live stage/done/total the container
    last published (survives across retries; `t` is its wall-clock timestamp).

    `unreachable` means the POLL failed, not the run: the container is untouched
    by a dropped RPC and keeps going, so the caller must retry rather than treat
    the cell as failed. The two can't be confused — a container that really died
    reports its own exception class through the call, while everything in
    `_UNREACHABLE` is raised by Modal's client transport."""
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
    except _UNREACHABLE as exc:
        out["state"] = "unreachable"
        out["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        out["state"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def cancel_cell(cell_dir: Path) -> str | None:
    """Cancel the cell's recorded Modal function call, TERMINATING its container
    so the GPU box stops billing immediately (not just draining its input queue).
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
    entries = _remote_entries(vol, f"/{remote}")
    if not entries:
        return []  # samples dir not created yet
    _download(
        vol, remote, local, entries,
        sorted(n for n in entries if n.endswith(".png")), quiet=True,
    )
    return sorted(p.name for p in local.glob("*.png"))


def read_cell_artifact(run: str, slot: str, model: str, relpath: str) -> bytes:
    """Read ONE file from a cell's `splat/` dir on the Volume → bytes (e.g.
    `refs/transforms.json` or `refs/frames/cam00001_ball.szf`). Lets the server
    stream remote reference frames to the debug viewer ON DEMAND — one SZF per
    request — WITHOUT syncing the thousands of frames locally (refs otherwise
    never leave the datacenter). Raises FileNotFoundError when the file is absent
    on the Volume (the caller maps that to a 404)."""
    rel = str(relpath).strip("/")
    if not rel or ".." in rel.split("/"):
        raise ValueError(f"illegal artifact relpath: {relpath!r}")
    remote = f"cells/{run}/{slot}/{model}/splat/{rel}"
    vol = _volume()
    try:
        return b"".join(vol.read_file(remote))
    except Exception as exc:  # missing file / volume error → 404 upstream
        raise FileNotFoundError(remote) from exc


def _download(
    vol: modal.Volume, remote: str, local: Path, entries: dict[str, Any],
    names: list[str], quiet: bool, force: bool = False,
) -> list[str]:
    """Fetch `names` from one Volume dir into `local`, skipping same-size files
    unless `force`. Returns the names actually transferred.

    VERIFIED. A `read_file` stream that ends early yields a SHORT file, and a
    truncated splat is not a failed download — it is a plausible-looking one. The
    `.ply` still declares its full vertex count, so a loader reads the missing tail
    as zeros: 1-metre mid-grey Gaussians stacked at the origin (log-scale 0 → 1 m,
    SH DC 0 → 0.5 grey) plus a degenerate quaternion that NaNs the sort. It renders
    as a grey sphere in the middle of an otherwise wrecked scene, which reads
    exactly like a bad TRAINING run — the one failure the size is enough to catch.
    So every transfer is checked against the listed size, and a short one is
    retried once before failing loudly rather than replacing a good artifact."""
    local.mkdir(parents=True, exist_ok=True)
    got: list[str] = []
    for name in names:
        dst = local / name
        size = getattr(entries[name], "size", None)
        if not force and size is not None and dst.is_file() and dst.stat().st_size == size:
            continue
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        for attempt in (1, 2):
            with tmp.open("wb") as f:
                for chunk in vol.read_file(f"{remote}/{name}"):
                    f.write(chunk)
            wrote = tmp.stat().st_size
            if size is None or wrote == size:
                break
            if attempt == 2:
                tmp.unlink(missing_ok=True)
                raise OSError(
                    f"{remote}/{name}: truncated download — got {wrote:,} of "
                    f"{size:,} bytes twice; the local copy was left untouched"
                )
            if not quiet:
                print(
                    f"short read on {name} ({wrote:,} of {size:,} bytes) — retrying",
                    flush=True,
                )
        tmp.replace(dst)
        got.append(name)
        if not quiet:
            print(f"pulled {name} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
    return got


def pull_plan(cell_dir: Path, quiet: bool = True) -> bool:
    """Download JUST the Volume's Stage-4 plan (+ the stage record beside it), so a
    run that REUSES the remote plan shows the camera overlay for the plan actually
    being rendered. Returns whether a plan is local afterwards.

    Unconditional (not same-size-skipped): a locally planned `cameras.json` that
    happened to match the remote one in byte length would otherwise shadow it and
    the overlay would show a plan that never rendered."""
    run, slot, model = _cell_parts(cell_dir)
    vol = _volume()
    remote = f"cells/{run}/{slot}/{model}/splat"
    entries = _remote_entries(vol, f"/{remote}")
    names = [n for n in _PULL_ALWAYS if n in entries]
    _download(vol, remote, cell_dir / "splat", entries, names, quiet, force=True)
    return (cell_dir / "splat" / _CAMERAS_NAME).is_file()


def pull_cell(cell_dir: Path, include_ply: bool = True, quiet: bool = False) -> list[str]:
    """Download the run's artifacts into the local `splat/` dir: plan + status
    + every `.sqz`, plus the plys unless `include_ply` is off — the RAW Stage-6
    `trained.ply` AND the delivered Stage-7 `healed.ply` + its LOD ladder, so the
    viewer can show trained vs healed side by side. Skips same-size files."""
    run, slot, model = _cell_parts(cell_dir)
    vol = _volume()
    remote = f"cells/{run}/{slot}/{model}/splat"
    local = cell_dir / "splat"
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
    return _download(vol, remote, local, entries, wanted, quiet)


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

    p = sub.add_parser("run", help="push + spawn stages on the GPU box")
    add_cell(p, tier=True)
    p.add_argument("--stages", type=_parse_stages, default=None,
                   help="e.g. 4-7, 4, or 6,7 (default: derived from --reuse + --trainer)")
    p.add_argument("--reuse", choices=("none", "assets", "cameras", "captures"),
                   default="none",
                   help="reuse what the Volume already holds: 'assets' skips the "
                        "upload, 'cameras' also skips stage 4, 'captures' also skips "
                        "stage 5 (train only)")
    p.add_argument("--plan-source", choices=("volume", "local"), default="volume",
                   help="whose stage-4 plan stage 5 renders: the Volume's, or the "
                        "LOCAL splat/cameras.json (uploaded, and stage 4 skipped)")
    p.add_argument("--trainer", choices=("brush", "gsplat"), default="brush",
                   help="stage-6 back-end (default brush)")
    p.add_argument("--plan-json", default="{}", help="PlanParams overrides")
    p.add_argument("--train-json", default="{}", help="TrainParams overrides (gsplat)")
    p.add_argument("--brush-json", default="{}", help="BrushParams overrides (brush)")
    p.add_argument("--quant-json", default="{}", help="QuantConfig overrides")
    p.add_argument("--force", action="store_true",
                   help="re-run the stage window even when signatures match")

    p = sub.add_parser("status", help="one-shot job state + heartbeat")
    add_cell(p)

    p = sub.add_parser(
        "state", help="what the Volume already holds for this cell (reuse targets)"
    )
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
        "cancel", help="cancel the cell's running Modal job (terminates the GPU container)"
    )
    add_cell(p)

    sub.add_parser("probe", help="report the GPU container's WebGL renderer")

    args = ap.parse_args()

    if args.cmd == "probe":
        out = modal.Function.from_name(APP_NAME, "probe_webgl").remote()
        print(json.dumps(out, indent=1))
        return

    if args.cmd == "push":
        push_cell(args.cell, args.tier)
        return

    if args.cmd == "run":
        # `--plan-source local` means "render MY plan", which skips stage 4 by
        # definition — the same window `--reuse cameras` asks for, differing only
        # in whose plan gets rendered.
        local_plan = args.plan_source == "local"
        reuse_assets = args.reuse != "none"
        if reuse_assets:
            pushed = remote_pushed(args.cell)
            print(f"reusing the Volume's inputs (tier {pushed.get('tier_dir')})")
            if local_plan:
                push_cameras(args.cell)
        else:
            pushed = push_cell(args.cell, args.tier, include_cameras=local_plan)
        stages = args.stages or list(
            range(
                6 if args.reuse == "captures" else
                5 if args.reuse == "cameras" or local_plan else 4,
                (6 if args.trainer == "brush" else 7) + 1,
            )
        )
        call_id = spawn_cell(
            args.cell, pushed, stages,
            plan=json.loads(args.plan_json),
            train=json.loads(args.train_json),
            quant=json.loads(args.quant_json),
            force=args.force,
            trainer=args.trainer,
            brush=json.loads(args.brush_json),
        )
        print(f"spawned {call_id} (stages {stages}) — "
              f"`watch --cell {args.cell}` to follow")
        return

    if args.cmd == "status":
        print(json.dumps(job_status(args.cell), indent=1, default=str))
        return

    if args.cmd == "state":
        print(json.dumps(remote_cell_state(args.cell), indent=1, default=str))
        return

    if args.cmd == "watch":
        while True:
            st = job_status(args.cell)
            if st["state"] == "running":
                print(_fmt_heartbeat(st.get("heartbeat")), flush=True)
                time.sleep(10)
                continue
            if st["state"] == "unreachable":
                print(f"lost contact with Modal ({st.get('error')}) — "
                      f"the run continues; retrying", flush=True)
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
