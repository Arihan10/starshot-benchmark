"""Scratch check for `generation.neighbours_framed`.

Negative space counts as a wall: a ray that meets a decomposed container's
unclaimed volume stops there, so nothing behind that gap is our boundary.
"""

from types import SimpleNamespace as NS

from app.core import util
from app.core.types import BoundingBox, Node
from app.pipeline import committed, generation


def Z(i, p, o, d):
    return Node(id=i, prompt=i, bbox=BoundingBox(origin=o, dimensions=d),
                parent_id=p, plan="x", is_zone=True)


def wire(plan, decomp, encap, bbox):
    committed.zone_plan = lambda z: plan.get(z)
    committed.zone_decompose = lambda z: decomp.get(z)
    committed.object_specs = lambda z, s: encap.get(z)
    committed.bbox = lambda i: bbox.get(i)


results = []


def check(label, got, want):
    results.append(got == want)
    print(f"{'PASS' if got == want else 'FAIL'}  {label:46} got={got!s:5} want={want}")


# --- part-solid, part-gap container, with C hidden behind it -----------------
# root 20x3x10 : A | B(container) | C(behind B)
root = Z("root", None, (0, 0, 0), (20, 3, 10))
A = Z("A", "root", (0, 0, 0), (5, 3, 10))
B = Z("B", "root", (5, 0, 0), (5, 3, 10))
B1 = Z("B1", "B", (5, 0, 0), (5, 3, 4))       # covers z 0-4; z 4-10 is a GAP
C = Z("C", "root", (10, 0, 0), (10, 3, 10))   # behind B, never decomposed
nodes = [root, A, B, B1, C]

PLAN = {"B": NS(is_atomic=False), "B1": NS(is_atomic=True), "C": NS(is_atomic=False)}
DECOMP = {"B": NS(subregions=[NS(id="B1")])}
ENCAP = {"B": [], "B1": []}
BBOX = {"A": 1, "B": 1, "B1": 1, "C": 1}
wire(PLAN, DECOMP, ENCAP, BBOX)

print("frontier of A:", [n.id for n in util.adjacent_zones("A", nodes)])
check("gap + flush atomic child, both settled", generation.neighbours_framed("A", nodes), True)

# --- three levels: B1 becomes a container holding a flush B1a ----------------
B1a = Z("B1a", "B1", (5, 0, 0), (5, 3, 2))
nodes2 = [root, A, B, B1, B1a, C]
PLAN["B1"] = NS(is_atomic=False)
check("flush container not yet decomposed", generation.neighbours_framed("A", nodes2), False)
DECOMP["B1"] = NS(subregions=[NS(id="B1a")])
BBOX["B1a"] = 1
check("flush grandchild still unplanned", generation.neighbours_framed("A", nodes2), False)
PLAN["B1a"] = NS(is_atomic=True)
ENCAP["B1a"] = []
check("flush grandchild atomic + framed", generation.neighbours_framed("A", nodes2), True)

# --- negative space as a wall: an INSET child behind a gap is not ours -------
# A | gap(x 5-7 unclaimed in D) | D1 — D1 is unframed and must NOT block A.
root2 = Z("root", None, (0, 0, 0), (20, 3, 10))
A2 = Z("A", "root", (0, 0, 0), (5, 3, 10))
D = Z("D", "root", (5, 0, 0), (15, 3, 10))
D1 = Z("D1", "D", (7, 0, 0), (8, 3, 10))      # inset: x 5-7 inside D is a GAP
nodes3 = [root2, A2, D, D1]
wire(
    {"D": NS(is_atomic=False), "D1": NS(is_atomic=True)},
    {"D": NS(subregions=[NS(id="D1")])},
    {"D": []},                                 # D framed; D1 deliberately NOT
    {"A": 1, "D": 1, "D1": 1},
)
print("frontier of A:", [n.id for n in util.adjacent_zones("A", nodes3)])
check("inset child behind a gap, unframed", generation.neighbours_framed("A", nodes3), True)


# --- a standing frame seals off whatever is behind it -------------------------
# The front-yard case: the yard borders the house volume, whose rooms are not
# planned yet — but the house's outer shell is already up between them.
def house_scene(*, decomposed, wall_box, ground_planned=False):
    root2 = Z("root", None, (0, 0, 0), (20, 5, 10))
    yard = Z("front_yard", "root", (0, 0, 0), (5, 5, 10))
    house = Z("house", "root", (5, 0, 0), (15, 5, 10))
    ground = Z("ground_floor", "house", (5, 0, 0), (15, 5, 10))
    wall = Node(
        id="house_west_wall", prompt="wall",
        bbox=BoundingBox(origin=wall_box[0], dimensions=wall_box[1]),
        parent_id="house", is_zone=False,
    )
    nodes = [root2, yard, house] + ([ground] if decomposed else []) + [wall]
    plan = {"front_yard": NS(is_atomic=True), "house": NS(is_atomic=False)}
    if ground_planned:
        plan["ground_floor"] = NS(is_atomic=False)
    wire(
        plan,
        {"house": NS(subregions=[NS(id="ground_floor")])} if decomposed else {},
        {"house": [NS(id="house_west_wall")], "front_yard": []},
        {"root": 1, "front_yard": 1, "house": 1, "ground_floor": 1, "house_west_wall": 1},
    )
    return nodes


# Wall on the house's west face: touches the yard AND the volume behind it.
SEALS = ((5.0, 0, 0), (0.2, 5, 10))
# Wall on the far east face: standing, but not between the yard and anything.
FAR = ((19.8, 0, 0), (0.2, 5, 10))

n = house_scene(decomposed=False, wall_box=SEALS)
print("frontier of front_yard:", [x.id for x in util.adjacent_zones("front_yard", n)])
check("undecomposed volume, outer shell up", generation.neighbours_framed("front_yard", n), True)

n = house_scene(decomposed=True, wall_box=SEALS)
print("frontier of front_yard:", [x.id for x in util.adjacent_zones("front_yard", n)])
check("unplanned room behind that shell", generation.neighbours_framed("front_yard", n), True)

n = house_scene(decomposed=True, wall_box=FAR)
check("shell exists but not between us", generation.neighbours_framed("front_yard", n), False)

n = house_scene(decomposed=False, wall_box=SEALS)
check("the volume itself is still unsettled", generation._region_settled(
    next(x for x in n if x.id == "house")), False)

print()
print("ALL PASS" if all(results) else "FAILURES PRESENT")
