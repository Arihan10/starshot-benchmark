"""Validation harness for the file-based prompt versioning.

Run:  cd server && uv run python scripts/test_prompt_templates.py

Covers, for every prompt version under versions/ and every pipeline step:

  1. resolver semantics — substitution, backtick consumption, literal-brace
     preservation, unknown-variable failure, injection/escape safety
  2. static template integrity — full file set, no orphan files, no
     malformed variable tokens (bare braces, casing typos), per-step
     variable availability
  3. pipeline cross-checks — STEPS/template names and target-marker texts
     match what divider.py / generation.py actually use
  4. end-to-end renders — every step rendered from realistic scene state
     (plus edge states: empty scene, root-only, unplanned zone, peer-anchored
     objects, hostile node text) and scanned for malformation: unresolved
     tokens, leaked `None`, wrong/missing target markers
  5. information routing — parent dims, local frames, orientation
     visibility, retry feedback, alias field names land in the right prompts
  6. snapshot lifecycle — copy/load isolation from source edits, loud
     failure on incomplete sets
  7. production ground truth — the newest run's `cache.llm` events carry the
     exact bytes the pre-cutover pipeline sent. For every step exercised by
     that run: at least one logged SYSTEM must byte-match the baseline
     template, at least one logged USER must match the template's skeleton
     (static segments in order, variables as fills), and the `ZONE_ID` fill
     must equal the event's node id. Mixed-era cells (resumed after prompt
     edits) are expected and reported, not failed. Override the reference run
     with STARSHOT_PARITY_RUN; the section skips when runs/ has no event
     logs.

No network, no event-log writes, no run dirs touched (read-only).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
REPO = SERVER.parent
sys.path.insert(0, str(SERVER))

from app.core import prompt_store, scene_context  # noqa: E402
from app.core.schemas import ChildNodeSpec, ObjectSpec  # noqa: E402
from app.core.types import (  # noqa: E402
    BoundingBox,
    Node,
    ParentRelationshipKind,
    ProxyShape,
    Relationship,
    RelationshipKind,
)
from app.core import util  # noqa: E402
from app.services.llm import DEEPSEEK_INJECTION, _current_model, apply_model_quirks  # noqa: E402

BASELINE = REPO / "versions" / "baseline"

NON_DEEPSEEK_MODEL = "google/gemini-3.5-flash"
DEEPSEEK_MODEL = "deepseek/deepseek-v4-pro"

# The exact marker texts the pipeline passes for `{SCENE_CONTEXT}`'s target.
MARKERS = {
    "zone_plan": "This is the region you are to plan and flesh out from.",
    "zone_decompose": "This is the region you are to break down and decompose.",
    "child_bbox_batch": "This is the region whose subregions you are to place.",
    "anchor_decompose": "This is the subregion you are to generate a list of anchor objects for.",
    "encapsulating_decompose": "This is the region you are to decide whether a boundary is needed for, and if so, what objects form that boundary",
    "negative_space_decompose": "This is the region whose interstitial negative space you are filling.",
    "object_bbox_batch": "This is the subregion whose objects you are to place.",
    "next_object": "This is the subregion you are deciding whether to add more objects to.",
}

VAR_TOKEN_RE = re.compile(r"`\{([A-Z][A-Z0-9_]*)\}`")
BARE_UPPER_RE = re.compile(r"(?<!`)\{[A-Z][A-Z0-9_]*\}(?!`)")

# Runtime contract: every render receives the FULL vocabulary (variables a
# step can't back resolve to empty/placeholder), so any variable is legal in
# any template. prompt_store.STEP_VARIABLES keeps the "natively populated"
# subsets for docs + the editor's chip styling.
PROVIDED = {step: set(prompt_store.ALL_VARIABLES) for step in prompt_store.STEPS}

_passed: list[str] = []
_failed: list[tuple[str, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        detail = str(e) if str(e) else traceback.format_exc()
        _failed.append((name, detail))
        print(f"FAIL  {name}\n      {detail.splitlines()[0] if detail else e}")
    else:
        _passed.append(name)
        print(f"ok    {name}")


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def diff_head(a: str, b: str, n: int = 30) -> str:
    lines = list(
        difflib.unified_diff(a.splitlines(), b.splitlines(), "old", "new", lineterm="")
    )
    return "\n".join(lines[:n])


# --- fixtures -------------------------------------------------------------------


def bb(origin, dims) -> BoundingBox:
    return BoundingBox(origin=origin, dimensions=dims)


def build_scene() -> list[Node]:
    """Realistic mid-run scene: root + planned/unplanned zones, root-anchored
    shell geometry, region objects (incl. a peer-anchored one), relationships,
    orientations, a non-BOX proxy, quotes in placement text."""
    root = Node(
        id="root", prompt="A modern house", is_zone=True,
        bbox=bb((-15, 0, -10), (30, 10, 20)),
        plan="A coastal modern house on a level lot: a two-story glass-and-concrete volume to the north, an open garden sloping south, and a clear approach path binding them.",
    )
    ground = Node(
        id="ground_slab", prompt="a broad concrete ground slab",
        bbox=bb((-15, 0, -10), (30, 0.3, 20)),
        parent_id="root", parent_kind=ParentRelationshipKind.ON, parent_region="root",
        placement="spanning the entire scene footprint",
        mesh_url="/artifacts/r/ground_slab.glb",
    )
    house = Node(
        id="house_zone", prompt="A two-story modern house volume", is_zone=True,
        bbox=bb((-12, 0.3, -9), (20, 8, 9)),
        parent_id="root", parent_kind=ParentRelationshipKind.IN,
        placement="occupying the northern half of the lot",
        plan="Ground floor opens onto the garden through a glass wall; the upper floor cantilevers west over the entry.",
    )
    living = Node(
        id="living_room", prompt="An open living room", is_zone=True,
        bbox=bb((-11, 0.3, -8), (6, 3, 5)),
        parent_id="house_zone", parent_kind=ParentRelationshipKind.IN,
        placement="the south-west corner of the ground floor",
        plan="A sunken conversation pit faces the garden glass wall; built-in shelving lines the north side.",
    )
    kitchen = Node(  # placed but not yet planned — exercises the no-plan path
        id="kitchen", prompt="A galley kitchen", is_zone=True,
        bbox=bb((-5, 0.3, -8), (5, 3, 4)),
        parent_id="house_zone", parent_kind=ParentRelationshipKind.IN,
        placement="east of the living room along the north wall",
    )
    sofa = Node(
        id="sofa", prompt="a low-slung charcoal fabric sofa",
        bbox=bb((-10.5, 0.3, -7.5), (2.4, 0.8, 1.0)),
        parent_id="living_room", parent_kind=ParentRelationshipKind.ON,
        parent_region="living_room", orientation=90,
        placement='against the "garden" glass wall, facing the conversation pit',
        referenced_ids=[Relationship(target="rug", kind=RelationshipKind.BESIDE)],
        mesh_url="/artifacts/r/sofa.glb",
    )
    rug = Node(
        id="rug", prompt="a flat-woven wool area rug",
        bbox=bb((-10.8, 0.3, -7.8), (3.2, 0.02, 2.4)),
        parent_id="living_room", parent_kind=ParentRelationshipKind.ON,
        parent_region="living_room",
        placement="centered under the conversation pit seating",
        mesh_url="/artifacts/r/rug.glb",
    )
    side_table = Node(
        id="side_table", prompt="a round walnut side table",
        bbox=bb((-8.2, 0.3, -7.4), (0.5, 0.55, 0.5)),
        parent_id="living_room", parent_kind=ParentRelationshipKind.ON,
        parent_region="living_room",
        placement="at the sofa's right arm",
        mesh_url="/artifacts/r/side_table.glb",
    )
    lamp = Node(  # peer-anchored: parent is an object, region differs
        id="lamp", prompt="a brass stem table lamp with linen shade",
        bbox=bb((-8.1, 0.85, -7.3), (0.3, 0.6, 0.3)),
        parent_id="side_table", parent_kind=ParentRelationshipKind.ON,
        parent_region="living_room",
        placement="centered on the side table",
        mesh_url="/artifacts/r/lamp.glb",
    )
    garden = Node(
        id="garden_zone", prompt="An open terraced garden", is_zone=True,
        bbox=bb((-12, 0.3, 1), (24, 4, 8)),
        parent_id="root", parent_kind=ParentRelationshipKind.IN,
        proxy_shape=ProxyShape.HEMISPHERE,
        placement="the southern half of the lot, sloping away from the house",
        plan="Low planted terraces step down from the house's glass wall to a gravel sitting circle at the south edge.",
    )
    return [root, ground, house, living, kitchen, sofa, rug, side_table, lamp, garden]


def root_only_scene() -> list[Node]:
    return [
        Node(
            id="root", prompt="A hotel room", is_zone=True,
            bbox=bb((0, 0, 0), (8, 4, 6)), plan="A compact hotel room.",
        )
    ]


def child_specs() -> list[ChildNodeSpec]:
    return [
        ChildNodeSpec(
            id="garage", prompt="An attached two-car garage",
            parent="house_zone", parent_kind=ParentRelationshipKind.IN,
            placement="abutting the house volume's east face",
            referenced_ids=[Relationship(target="kitchen", kind=RelationshipKind.BESIDE)],
        ),
        ChildNodeSpec(  # intra-batch parent
            id="garage_loft", prompt="A storage loft above the garage bay",
            parent="garage", parent_kind=ParentRelationshipKind.IN,
            placement="the upper third of the garage volume",
        ),
        ChildNodeSpec(  # unresolvable parent
            id="mystery_annex", prompt="An annex with an unknown anchor",
            parent="not_a_real_id", parent_kind=ParentRelationshipKind.IN,
            placement="somewhere unclear",
        ),
    ]


def object_specs() -> list[ObjectSpec]:
    return [
        ObjectSpec(
            id="armchair", prompt="a worn tan leather armchair",
            parent="living_room", parent_kind=ParentRelationshipKind.ON,
            orientation="angled to face the sofa",
            placement="angled toward the sofa across the rug",
            referenced_ids=[Relationship(target="sofa", kind=RelationshipKind.BESIDE)],
        ),
        ObjectSpec(
            id="book_stack", prompt="a short stack of hardcover books",
            parent="side_table", parent_kind=ParentRelationshipKind.ON,
            placement="on the side table beside the lamp",
        ),
    ]


def zv(nodes, zone: Node, step: str, plan_override=...) -> dict[str, str]:
    plan = zone.plan if plan_override is ... else plan_override
    return scene_context.zone_vars(
        zone_id=zone.id, zone_prompt=zone.prompt, zone_plan=plan,
        nodes=nodes, target_text=MARKERS[step],
    )


# --- 1. resolver semantics ------------------------------------------------------


def test_resolver() -> None:
    r = prompt_store.resolve

    expect(r("a `{X}` b", {"X": "1"}, where="t") == "a 1 b", "basic substitution")
    expect(r("`{X}` and `{X}`", {"X": "v"}, where="t") == "v and v", "repeat substitution")
    expect(r("`{A}``{B}`", {"A": "1", "B": "2"}, where="t") == "12", "adjacent tokens")
    expect(r("q'`{X}`'q", {"X": "v"}, where="t") == "q'v'q", "backticks consumed, quotes kept")
    expect(r("e: `{X}`.", {"X": ""}, where="t") == "e: .", "empty value")
    expect(r("u `{X}`", {"X": "café 🌿"}, where="t") == "u café 🌿", "unicode value")

    # literals that must never substitute
    for lit in ("{target, kind}", "`{target, kind}`", "{}", "{(none - x)}",
                "{X}", "`{Mixed_Case}`", "`{lower}`", "((x-cx)/hx)"):
        expect(r(lit, {"X": "BOOM"}, where="t") == lit, f"literal preserved: {lit}")

    # values are never re-scanned (injection safety)
    out = r("ctx: `{X}`", {"X": "say `{ROOT_PROMPT}` aloud", "ROOT_PROMPT": "p"}, where="t")
    expect(out == "ctx: say `{ROOT_PROMPT}` aloud", "value containing a token is not re-expanded")

    # regex-escape safety in values (function replacement, no template escapes)
    out = r("x `{V}` y", {"V": r"a\path \g<0> $1 \n"}, where="t")
    expect(out == r"x a\path \g<0> $1 \n y", "backslashes/escapes pass through verbatim")

    # unknown variable fails loudly with location + availability
    try:
        r("`{NOT_A_VAR}`", {"X": "1"}, where="step.user.txt")
        raise AssertionError("unknown variable did not raise")
    except prompt_store.PromptTemplateError as e:
        expect("NOT_A_VAR" in str(e) and "step.user.txt" in str(e), f"error detail: {e}")

    ps = prompt_store._load_dir(BASELINE, "baseline")
    try:
        ps.system("no_such_step", {})
        raise AssertionError("missing step did not raise")
    except prompt_store.PromptTemplateError:
        pass


# --- 2. static template integrity -----------------------------------------------


def test_static_integrity() -> None:
    versions = sorted(p for p in (REPO / "versions").iterdir() if p.is_dir())
    expect(len(versions) >= 1, "no version folders found")
    expected_files = {f"{s}.{r}.txt" for s in prompt_store.STEPS for r in ("system", "user")}
    for vdir in versions:
        prompt_store._load_dir(vdir, vdir.name)  # raises on missing files
        files = {p.name for p in vdir.glob("*.txt")}
        expect(files == expected_files,
               f"{vdir.name}: unexpected/missing txt files: {sorted(files ^ expected_files)}")
        for f in sorted(vdir.glob("*.txt")):
            text = f.read_text(encoding="utf-8")
            expect(text.strip() != "", f"{vdir.name}/{f.name} is empty")
            bare = BARE_UPPER_RE.findall(text)
            expect(not bare, f"{vdir.name}/{f.name}: bare uppercase brace tokens "
                             f"(would leak literally, missing backticks?): {bare}")
            # backticked brace text that is identifier-like but not UPPER_SNAKE
            # is almost certainly a typo'd variable
            for m in re.finditer(r"`\{([^}`\n]*)\}`", text):
                body = m.group(1)
                if re.fullmatch(r"[A-Za-z0-9_]+", body) and not re.fullmatch(r"[A-Z][A-Z0-9_]*", body):
                    raise AssertionError(f"{vdir.name}/{f.name}: suspicious token `{{{body}}}`")
            step = f.name.split(".")[0]
            used = set(VAR_TOKEN_RE.findall(text))
            extra = used - PROVIDED[step]
            expect(not extra, f"{vdir.name}/{f.name}: uses unavailable variables {sorted(extra)}")


# --- 3. pipeline cross-checks ---------------------------------------------------


def test_pipeline_crosschecks() -> None:
    div_src = (SERVER / "app/pipeline/divider.py").read_text()
    gen_src = (SERVER / "app/pipeline/generation.py").read_text()
    src = div_src + gen_src
    for step in prompt_store.STEPS:
        expect(f'"{step}"' in src, f"step {step!r} never referenced by the pipeline")
    called = set(re.findall(r'ps\.(?:system|user)\(\s*"([a-z_]+)"', src))
    unknown = called - set(prompt_store.STEPS)
    expect(not unknown, f"pipeline renders steps missing from STEPS: {sorted(unknown)}")
    for step, marker in MARKERS.items():
        expect(marker in src, f"marker text for {step} not found in pipeline source")
    expect(prompt_store.list_versions() != [] if prompt_store.VERSIONS_DIR.is_dir() else True,
           "list_versions empty with versions dir present")


# --- 4 + 5. end-to-end renders + content routing --------------------------------


def scan_render(name: str, text: str, *, expect_markers: int) -> None:
    expect(text.strip(), f"{name}: empty render")
    leftover = VAR_TOKEN_RE.findall(text)
    expect(not leftover, f"{name}: unresolved variables {leftover}")
    bare = BARE_UPPER_RE.findall(text)
    expect(not bare, f"{name}: bare uppercase brace tokens {bare}")
    expect(re.search(r"\bNone\b", text) is None, f"{name}: leaked Python None")
    got = text.count("<-- TARGET:")
    expect(got == expect_markers, f"{name}: {got} target markers, expected {expect_markers}")


def render_all_steps() -> dict[str, str]:
    """Render every step exactly as its call site does, on the realistic scene."""
    _current_model.set(NON_DEEPSEEK_MODEL)
    ps = prompt_store._load_dir(BASELINE, "baseline")
    nodes = build_scene()
    by_id = {n.id: n for n in nodes}
    root, house, living, garden = by_id["root"], by_id["house_zone"], by_id["living_room"], by_id["garden_zone"]
    out: dict[str, str] = {}

    v = {"ZONE_ID": "root", "ZONE_PROMPT": root.prompt}
    out["zone_plan_root"] = ps.system("zone_plan_root", v) + "\n###\n" + ps.user("zone_plan_root", v)

    v = {"ROOT_PROMPT": root.prompt, "ROOT_PLAN": root.plan}
    out["overall_bbox"] = ps.system("overall_bbox", v) + "\n###\n" + ps.user("overall_bbox", v)

    v = zv(nodes, living, "zone_plan", plan_override=None)
    out["zone_plan"] = ps.system("zone_plan", v) + "\n###\n" + ps.user("zone_plan", v)

    v = zv(nodes, root, "zone_decompose")
    out["zone_decompose_root"] = ps.system("zone_decompose_root", v) + "\n###\n" + ps.user("zone_decompose_root", v)

    v = zv(nodes, house, "zone_decompose")
    out["zone_decompose"] = ps.system("zone_decompose", v) + "\n###\n" + ps.user("zone_decompose", v)

    v = zv(nodes, house, "child_bbox_batch")
    v["TO_PLACE"] = scene_context.render_to_place_block(
        child_specs(), by_id, parent_zone=house.id)
    out["child_bbox_batch"] = ps.system("child_bbox_batch", v) + "\n###\n" + ps.user("child_bbox_batch", v)

    for step, zone in (("anchor_decompose", living),
                       ("encapsulating_decompose", garden),
                       ("negative_space_decompose", house)):
        v = zv(nodes, zone, step)
        v["RETRY_BLOCK"] = scene_context.render_retry_block(None)
        out[step] = ps.system(step, v) + "\n###\n" + ps.user(step, v)

    v = zv(nodes, living, "object_bbox_batch")
    v["TO_PLACE"] = scene_context.render_to_place_block(object_specs(), by_id, parent_zone=living.id)
    out["object_bbox_batch"] = ps.system("object_bbox_batch", v) + "\n###\n" + ps.user("object_bbox_batch", v)

    v = zv(nodes, living, "next_object")
    v["RETRY_BLOCK"] = scene_context.render_next_object_retry_block(None)
    out["next_object"] = ps.system("next_object", v) + "\n###\n" + ps.user("next_object", v)

    iv = scene_context.image_prompt_vars(
        prompt="a worn tan leather armchair", bbox=by_id["sofa"].bbox,
        proxy_shape=None, prior_prompts=["a broad concrete ground slab", "a flat-woven wool area rug"])
    out["image_prompt"] = ps.system("image_prompt", iv) + "\n###\n" + ps.user("image_prompt", iv)
    return out


def test_render_all_steps() -> None:
    renders = render_all_steps()
    no_marker = {"zone_plan_root", "overall_bbox", "image_prompt",
                 "zone_decompose_root"}  # root is not part of the embedded tree
    for step, text in renders.items():
        scan_render(step, text, expect_markers=0 if step in no_marker else 1)


def test_information_routing() -> None:
    _current_model.set(NON_DEEPSEEK_MODEL)
    ps = prompt_store._load_dir(BASELINE, "baseline")
    nodes = build_scene()
    by_id = {n.id: n for n in nodes}
    root, house, living = by_id["root"], by_id["house_zone"], by_id["living_room"]

    # root header numbers + root objects vs embedded tree split
    v = zv(nodes, living, "anchor_decompose")
    v["RETRY_BLOCK"] = ""
    u = ps.user("anchor_decompose", v)
    expect("30.00m by 10.00m by 20.00m, with its origin corner at (-15.00, 0.00, -10.00) m" in u,
           "root header numbers wrong")
    expect("Name: ground_slab" in u, "root-anchored shell missing from ROOT_OBJECTS")
    ctx = v["SCENE_CONTEXT"]
    expect("ground_slab" not in ctx, "root-anchored object leaked into the embedded tree")
    expect(f"Subregion name: living_room   <-- TARGET: {MARKERS['anchor_decompose']}" in ctx,
           "target marker not on the targeted zone line")
    expect("Subregion name: kitchen" in ctx and "Plan for this region" not in
           ctx.split("Subregion name: kitchen")[1].split("Subregion name:")[0],
           "unplanned zone should render without a plan line")
    # peer-anchored object carries its region; direct-anchored object doesn't
    lamp_entry = ctx.split("Name: lamp")[1].split("Name: ")[0]
    expect("parent_id: side_table" in lamp_entry and "parent_region: living_room" in lamp_entry
           and "parent_region_dimensions:" in lamp_entry, "peer-anchored object entry wrong")
    sofa_entry = ctx.split("Name: sofa")[1].split("Name: ")[0]
    expect("parent_region:" not in sofa_entry, "direct-anchored object should omit parent_region")
    expect("relationships: [rug: BESIDE]" in sofa_entry, "relationship list wrong")
    expect("orientation: 90deg" in sofa_entry, "object orientation missing")
    expect('placement: "against the "garden" glass wall, facing the conversation pit"' in sofa_entry,
           "quoted placement text mangled")
    # local frame math: living_room relative to house_zone
    expected_local = util.format_local_origin(living.bbox, house.bbox)
    expect(f"Local origin corner (relative to house_zone, measured from its min corner): {expected_local}" in ctx,
           "local-frame coordinates wrong")
    expect(expected_local == "(1.00, 0.00, 1.00) m", f"local origin math wrong: {expected_local}")

    # child_bbox_batch: zone fields + TO_PLACE block specifics
    v = zv(nodes, house, "child_bbox_batch")
    v["TO_PLACE"] = scene_context.render_to_place_block(
        child_specs(), by_id, parent_zone=house.id)
    u = ps.user("child_bbox_batch", v)
    expect("Parent region name: 'house_zone'" in u, "parent zone id not routed")
    to_place = u.split("Here is the list of subregions you must place:")[1]
    expect(f"parent_dimensions: {util.format_dimensions(house.bbox)}" in to_place,
           "child spec parent dims wrong")
    expect("(parent is also being placed in this batch — use your emitted dimensions for it)" in to_place,
           "intra-batch parent placeholder missing")
    expect("(parent id not recognised in current scene)" in to_place,
           "unknown-parent placeholder missing")
    expect("orientation:" not in to_place, "zone placement block must not show orientation")
    expect("parent_region: house_zone" in to_place, "peer-parented spec lost its region")

    # object_bbox_batch: orientation shown
    v = zv(nodes, living, "object_bbox_batch")
    v["TO_PLACE"] = scene_context.render_to_place_block(object_specs(), by_id, parent_zone=living.id)
    to_place = ps.user("object_bbox_batch", v).split("Here is the list of objects you must place:")[1]
    expect('orientation: "angled to face the sofa"' in to_place, "object orientation text missing from TO_PLACE")
    expect(f"parent_dimensions: {util.format_dimensions(by_id['side_table'].bbox)}" in to_place,
           "object spec peer-parent dims wrong")

    # retry feedback: alias wire names + flavor wording
    rejected = object_specs()
    retry = scene_context.render_retry_block([(rejected, "relationships entry 'ghost' does not resolve")])
    expect("PRIOR ATTEMPTS" in retry and "relationships entry 'ghost' does not resolve" in retry,
           "retry block missing attempt/reason")
    expect('"parent_relationship_kind"' in retry and '"relationships"' in retry,
           "retry dump must use wire alias names")
    expect('"parent_kind"' not in retry and '"referenced_ids"' not in retry,
           "retry dump leaked python attribute names")
    nretry = scene_context.render_next_object_retry_block([(rejected, "reason-x")])
    expect("or set done=true" in nretry, "next_object retry flavor wrong")
    v = zv(nodes, living, "anchor_decompose")
    v["RETRY_BLOCK"] = retry
    expect("PRIOR ATTEMPTS" in ps.user("anchor_decompose", v), "retry block not injected")

    # image_prompt: dims/proxy/prior routing + all proxy shapes render
    iv = scene_context.image_prompt_vars(
        prompt="a worn tan leather armchair", bbox=bb((0, 0, 0), (2.4, 0.8, 1.0)),
        proxy_shape=None, prior_prompts=[])
    u = ps.user("image_prompt", iv)
    expect("width=2.40m, height=0.80m, depth=1.00m" in u, "image dims wrong")
    expect("Proxy shape: BOX" in u, "proxy shape wrong")
    expect("<<<SUBJECT>>>" in u, "subject slot missing")
    expect("(none — this is the first object; you are setting the aesthetic baseline)" in u,
           "empty prior-subjects placeholder missing")
    for shape, term in ((ProxyShape.SPHERE, "ellipsoid"), (ProxyShape.CAPSULE, "vertical capsule"),
                        (ProxyShape.HEMISPHERE, "dome"), (None, "rectangular prism")):
        iv = scene_context.image_prompt_vars(
            prompt="x", bbox=bb((0, 0, 0), (1, 1, 1)), proxy_shape=shape, prior_prompts=["a rug"])
        u = ps.user("image_prompt", iv)
        expect(term in u, f"hitbox term for {shape} missing")
        expect("1. a rug" in u, "prior subjects list missing")

    # deepseek pin is a CALL-BOUNDARY quirk now, never template content:
    # no template/variable mentions it, and apply_model_quirks owns it.
    expect("DEEPSEEK_SUFFIX" not in ps.user("zone_plan_root", {"ZONE_ID": "r", "ZONE_PROMPT": "x"}),
           "deepseek token still rendered from baseline")
    expect(apply_model_quirks("base", DEEPSEEK_MODEL) == "base" + DEEPSEEK_INJECTION,
           "deepseek injection not applied at call boundary")
    expect(apply_model_quirks("base", NON_DEEPSEEK_MODEL) == "base",
           "injection leaked to non-deepseek model")
    once = apply_model_quirks("base", DEEPSEEK_MODEL)
    expect(apply_model_quirks(once, DEEPSEEK_MODEL) == once,
           "injection not idempotent (pre-cutover prompts would double up)")
    # pre-cutover snapshots still resolve the legacy token — to nothing
    legacy = prompt_store.resolve("a `{DEEPSEEK_SUFFIX}` b", scene_context.base_vars(), where="t")
    expect(legacy == "a  b", "legacy DEEPSEEK_SUFFIX token no longer resolves to empty")


def test_edge_scenes() -> None:
    _current_model.set(NON_DEEPSEEK_MODEL)
    ps = prompt_store._load_dir(BASELINE, "baseline")

    # root-only scene: placeholders, no markers, nothing unresolved
    nodes = root_only_scene()
    v = zv(nodes, nodes[0], "negative_space_decompose")
    v["RETRY_BLOCK"] = ""
    u = ps.user("negative_space_decompose", v)
    scan_render("negative_space root-only", u, expect_markers=0)
    expect("{(none - no other subregions have been planned yet)}" in u,
           "empty-tree placeholder missing")
    expect("No objects are parented directly to the root yet." in u,
           "empty root-objects placeholder missing")

    # zone id not yet in the node list: falls back to root bbox, never crashes
    v = scene_context.zone_vars(zone_id="ghost_zone", zone_prompt="g", zone_plan="p",
                                nodes=nodes, target_text="t")
    expect(v["ZONE_DIMENSIONS"] == util.format_dimensions(nodes[0].bbox),
           "missing-zone bbox fallback broken")

    # unplanned zone rendered through a plan-bearing template: explicit ""
    v = zv(nodes, nodes[0], "anchor_decompose", plan_override=None)
    v["RETRY_BLOCK"] = ""
    expect('Subregion plan: ""' in ps.user("anchor_decompose", v),
           "None plan must render as empty string, not 'None'")

    # hostile node text: template-like tokens inside scene state stay verbatim
    nodes = build_scene()
    by_id = {n.id: n for n in nodes}
    sofa = by_id["sofa"]
    hostile = sofa.model_copy(update={
        "prompt": "a sofa whose throw pillow reads `{ROOT_PROMPT}` in stitching",
        "placement": "near the {LOOT} crate and the literal `{target, kind}` sign",
    })
    nodes[nodes.index(sofa)] = hostile
    v = zv(nodes, by_id["living_room"], "anchor_decompose")
    v["RETRY_BLOCK"] = ""
    u = ps.user("anchor_decompose", v)
    expect(u.count("`{ROOT_PROMPT}`") == 1, "planted token was substituted or duplicated")
    expect("{LOOT}" in u and "`{target, kind}`" in u, "hostile literals mangled")
    expect("A modern house" in u, "real root prompt missing alongside planted token")


# --- 6. snapshot lifecycle -------------------------------------------------------


def test_snapshot_lifecycle() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="prompt-suite-"))
    saved = prompt_store.VERSIONS_DIR
    try:
        vdir = tmp / "versions"
        shutil.copytree(BASELINE, vdir / "vtest")
        prompt_store.VERSIONS_DIR = vdir
        expect(prompt_store.list_versions() == ["vtest"], "tmp version not listed")
        expect(prompt_store.version_exists("vtest") and not prompt_store.version_exists("nope"),
               "version_exists wrong")
        prompt_store.validate_version("vtest")

        run_a = tmp / "runA"
        run_a.mkdir()
        prompt_store.snapshot_into_run("vtest", run_a)
        expect(prompt_store.has_run_prompts(run_a), "snapshot dir missing")
        var = {"ROOT_PROMPT": "p", "ROOT_PLAN": "q"}
        before = prompt_store.load_run_prompts(run_a).user("overall_bbox", var)

        # mutate the SOURCE after snapshotting — the run must not move
        (vdir / "vtest" / "overall_bbox.user.txt").write_text("CHANGED `{ROOT_PROMPT}`")
        expect(prompt_store.load_run_prompts(run_a).user("overall_bbox", var) == before,
               "cached snapshot drifted after source edit")
        fresh = prompt_store._load_dir(run_a / "prompts", "fresh")
        expect(fresh.user("overall_bbox", var) == before,
               "snapshot file bytes drifted after source edit")

        # incomplete snapshot fails loudly, naming the missing file
        run_b = tmp / "runB"
        run_b.mkdir()
        prompt_store.snapshot_into_run("vtest", run_b)
        (run_b / "prompts" / "next_object.user.txt").unlink()
        try:
            prompt_store._load_dir(run_b / "prompts", "broken")
            raise AssertionError("incomplete snapshot loaded silently")
        except prompt_store.PromptTemplateError as e:
            expect("next_object.user.txt" in str(e), f"missing-file detail absent: {e}")
    finally:
        prompt_store.VERSIONS_DIR = saved
        shutil.rmtree(tmp, ignore_errors=True)


# --- 7. production ground truth (run event logs) ---------------------------------
#
# Every `cache.llm` event carries the exact system/user bytes the pipeline
# sent. For each step we hunt recent runs for an event whose SYSTEM
# byte-matches a baseline template; its USER from the same event must then
# match the user template's SKELETON — the template's static text segments, in
# order, anchored at both ends, with variables as arbitrary fills — and the
# `ZONE_ID` fills must equal the event's node id. Cells resumed across prompt
# edits hold a mix of eras, so non-matching events are expected; a step only
# fails when NO recent event matches at all.

# event-log step id -> candidate template steps (root/nested variants share one id)
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

_STEP_PEEK = re.compile(r'"step": "([a-z_]+)"')
_MAX_EVENTS_PER_STEP_PER_CELL = 6
_MAX_RUNS = 8


def template_skeleton(text: str) -> tuple[list[str], list[str]]:
    """Split a template into (static_segments, variable_names). Variables sit
    between consecutive static segments."""
    parts = VAR_TOKEN_RE.split(text)
    return parts[0::2], parts[1::2]


def skeleton_match(template: str, rendered: str) -> dict[str, list[str]] | None:
    """Match `rendered` against the template's static skeleton: every static
    segment must appear in order, anchored at the start, with nothing but
    variable fills between and nothing but whitespace after. Returns
    {var_name: [fill, ...]} on success, else None.

    Trailing whitespace is normalized on both sides: when a template ends in
    variables (e.g. a first-attempt-empty `` `{RETRY_BLOCK}` ``, or the
    removed-but-logged `` `{DEEPSEEK_SUFFIX}` `` in pre-cutover events), the
    newline gluing the last prose segment to those variables disappears from
    the rendered text whenever the fills are empty — so the last non-empty
    static is compared right-stripped too."""
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


def iter_logged_calls(runs_root: Path, wanted: set[str]):
    """Yield (run, step, event) for cache.llm events of `wanted` steps across
    recent runs, newest first, lightly capped per cell to bound the scan."""
    if not runs_root.is_dir():
        return
    run_dirs = sorted(
        (p for p in runs_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    forced = os.environ.get("STARSHOT_PARITY_RUN")
    if forced:
        run_dirs = [runs_root / forced]
    for run_dir in run_dirs[:_MAX_RUNS]:
        for ev_path in sorted(run_dir.glob("*/*/events.jsonl")):
            seen: dict[str, int] = {}
            try:
                with ev_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if '"kind": "cache.llm"' not in line:
                            continue
                        peek = _STEP_PEEK.search(line, 0, 400)
                        if peek is None or peek.group(1) not in wanted:
                            continue
                        if seen.get(peek.group(1), 0) >= _MAX_EVENTS_PER_STEP_PER_CELL:
                            continue
                        try:
                            e = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        step = e.get("step")
                        if not isinstance(step, str) or step not in wanted:
                            continue
                        seen[step] = seen.get(step, 0) + 1
                        yield run_dir.name, step, e
            except OSError:
                continue


def add_ground_truth_checks() -> None:
    ps = prompt_store._load_dir(BASELINE, "baseline")
    sys_templates = {
        step: {cand: (BASELINE / f"{cand}.system.txt").read_text(encoding="utf-8").rstrip("\n")
               for cand in cands}
        for step, cands in EVENT_STEP_TEMPLATES.items()
    }
    user_templates = {
        step: {cand: (BASELINE / f"{cand}.user.txt").read_text(encoding="utf-8")
               for cand in cands}
        for step, cands in EVENT_STEP_TEMPLATES.items()
    }

    matched: dict[tuple[str, str], str] = {}     # (step, candidate) -> run
    sys_only: dict[str, tuple[str, str]] = {}    # system matched, user skeleton didn't
    closest_sys: dict[str, tuple[str, str]] = {} # step -> (run, last non-matching system)
    seen_steps: set[str] = set()

    def scan() -> None:
        # Hunt until every (step, variant) pair has a logged match or the
        # recent-run budget is exhausted; root and nested variants are
        # verified independently.
        pending = {(s, c) for s, cands in EVENT_STEP_TEMPLATES.items() for c in cands}
        for run, step, e in iter_logged_calls(REPO / "runs", set(EVENT_STEP_TEMPLATES)):
            if not any(p[0] == step for p in pending):
                continue
            seen_steps.add(step)
            system = (e.get("system") or "").rstrip("\n")
            user = e.get("user") or ""
            node = e.get("node")
            if not any(system == text for text in sys_templates[step].values()):
                closest_sys.setdefault(step, (run, system))
                continue
            # Same-era event: the user bytes must fit a paired user template's
            # skeleton. zone_plan/zone_decompose systems are shared between
            # the root and nested variants, so try both user skeletons.
            for cand in EVENT_STEP_TEMPLATES[step]:
                fills = skeleton_match(user_templates[step][cand], user)
                if fills is None:
                    continue
                zone_fills = set(fills.get("ZONE_ID", []))
                expect(not zone_fills or not isinstance(node, str) or zone_fills == {node},
                       f"{step}/{cand}: ZONE_ID fills {sorted(zone_fills)} != logged "
                       f"node {node!r} (run {run})")
                for tpl_fill in fills.get("IMAGE_TEMPLATE_FRONT", []):
                    expect("<<<SUBJECT>>>" in tpl_fill,
                           f"{step}: image wrapper fill lost the subject slot (run {run})")
                matched.setdefault((step, cand), run)
                pending.discard((step, cand))
                break
            else:
                sys_only.setdefault(step, (run, user))
            if not pending:
                break

    check("ground truth: event scan + fill consistency", scan)
    if not seen_steps:
        print("skip  ground truth: no cache.llm events found under runs/")
        return

    for step in sorted(EVENT_STEP_TEMPLATES):
        if step not in seen_steps:
            # Absence of evidence, not a defect: e.g. nomesh runs never fire
            # image_prompt. Sections 1-5 still cover the step synthetically.
            print(f"skip  ground truth: {step} — no logged calls in the "
                  f"{_MAX_RUNS} most recent runs")
            continue

        def fn(step=step):
            if any(s == step for s, _c in matched):
                return  # at least one variant fully byte/skeleton/node matched
            if step in sys_only:
                run, _user = sys_only[step]
                raise AssertionError(
                    f"system bytes match but no logged user fits the template skeleton "
                    f"(run {run}) — static text drifted in {step}.user.txt")
            run, system = closest_sys.get(step, ("?", ""))
            tpl = next(iter(sys_templates[step].values()))
            raise AssertionError(
                "no recent logged call matches the template (all scanned events are "
                f"older prompt eras; closest from run {run}):\n{diff_head(system, tpl)}")
        check(f"ground truth: {step} matches latest logged bytes", fn)

    for step, cands in sorted(EVENT_STEP_TEMPLATES.items()):
        for cand in cands:
            run = matched.get((step, cand))
            note = f"verified against run {run!r}" if run else "no logged evidence in recent runs"
            print(f"      · {cand}: {note}")


# --- main -----------------------------------------------------------------------


def main() -> int:
    _current_model.set(NON_DEEPSEEK_MODEL)
    check("resolver semantics", test_resolver)
    check("static template integrity (all versions)", test_static_integrity)
    check("pipeline step/marker cross-checks", test_pipeline_crosschecks)
    check("end-to-end render of every step (malformation scan)", test_render_all_steps)
    check("information routing (dims, frames, aliases, retries, proxies)", test_information_routing)
    check("edge scenes (empty tree, unplanned zone, hostile text, fallbacks)", test_edge_scenes)
    check("snapshot lifecycle (isolation + loud failures)", test_snapshot_lifecycle)
    add_ground_truth_checks()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("\nfailures:")
        for name, detail in _failed:
            print(f"\n--- {name}\n{detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
