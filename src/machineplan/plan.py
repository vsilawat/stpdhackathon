from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .features import Chamfer, Features, Hole, Pocket

# o2 by base op name; o1 is a pure function of o2.
O2_BY_NAME = {"AREA_MILL": "AREA_MILL", "SPOT_DRILL": "SPOT_DRILLING", "BORE_BLIND_HOLE": "BORING_REAMING"}
O1_BY_O2 = {"AREA_MILL": "mill_contour", "FLOOR_WALL": "mill_planar", "SPOT_DRILLING": "hole_making",
            "DRILLING": "hole_making", "DEEP_HOLE_DRILLING": "hole_making",
            "HOLE_MILLING": "hole_making", "BORING_REAMING": "hole_making"}
POCKET_OP = {"corner": "MILL_CORNER_NOTCH_RECTANGULAR", "center": "MILL_RECTANGULAR_POCKET",
             "slot": "MILL_RECTANGULAR_SLOT", "edge": "MILL_SLOT"}
MILL_ORDER = ["AREA_MILL", "MILL_OPEN_POCKET", "MILL_CORNER_NOTCH_RECTANGULAR", "MILL_RECTANGULAR_POCKET",
              "MILL_RECTANGULAR_SLOT", "MILL_SLOT", "MILL_THROUGH_HOLE_FROM_SOLID_MATERIAL",
              "MILL_BLIND_HOLE_FROM_SOLID_MATERIAL", "MILL_ROUGH_BLIND_HOLE_CONTOUR",
              "MILL_FINISH_BLIND_HOLE_FLAT_BOTTOM"]

def o2_of(name: str) -> str:
    if name in O2_BY_NAME: return O2_BY_NAME[name]
    if "GUN_DRILL" in name: return "DEEP_HOLE_DRILLING"
    if name.startswith("MILL_") and "HOLE" in name: return "HOLE_MILLING"
    if name.startswith("MILL_"): return "FLOOR_WALL"
    return "DRILLING"

@dataclass(slots=True)
class Op:
    name: str
    feature: Any = None
    tool_type: str | None = None
    tool_diameter: float | None = None
    extra: dict = field(default_factory=dict)
    @property
    def o2(self) -> str: return o2_of(self.name)
    @property
    def o1(self) -> str: return O1_BY_O2[self.o2]

# one AREA_MILL per chamfer face (exact on 10k)
def chamfer_ops(found: Features) -> list[Op]:
    return [Op("AREA_MILL", feature=c, tool_type="chamfer_mill", tool_diameter=20.0, extra={"face": f})
            for c in found.chamfers for f in (c.faces or [None])]

def pocket_ops(found: Features) -> list[Op]:
    return [Op(POCKET_OP[p.kind], feature=p, tool_type="end_mill",
               tool_diameter=2 * (p.fillet_radius or 5.0)) for p in found.pockets]

def hole_chain(h: Hole) -> list[Op]:
    """Mined chain rules (derived/chains.csv)."""
    d, ld = h.diameter, (h.depth / h.diameter if h.diameter else 0.0)
    def op(name, dia=None): return Op(name, feature=h, tool_diameter=dia)
    spot = op("SPOT_DRILL", 12.0)
    if h.through:
        if abs(d - 25.0) < 0.25 or abs(d - 32.0) < 0.25:
            return [op("MILL_THROUGH_HOLE_FROM_SOLID_MATERIAL", d)]
        if d < 12.0: return [spot, op("DRILL_THROUGH_HOLE_INTO_CENTER", d)]
        if ld >= 5.0 and d < 32.5:
            ops = [spot, op("DRILL_BLIND_HOLE_INTO_CENTER"), op("GUN_DRILL_THROUGH_HOLE")]
            if d >= 19.3: ops.append(op("SPADE_DRILL_TO_ENLARGE_THROUGH_HOLE"))
            ops.append(op("DRILL_TO_ENLARGE_THROUGH_HOLE", d))
            return ops
        if d < 15.0:
            return [spot, op("DRILL_THROUGH_HOLE_INTO_CENTER"), op("DRILL_TO_ENLARGE_THROUGH_HOLE", d)]
        return [op("INDEXABLE_INSERT_DRILL_THROUGH_HOLE_FROM_SOLID", d)]
    if d < 10.0: return [spot, op("DRILL_BLIND_HOLE_INTO_CENTER", d)]
    if d < 12.0:
        return [spot, op("DRILL_BLIND_HOLE_INTO_CENTER", d), op("MILL_ROUGH_BLIND_HOLE_CONTOUR", d)]
    if d < 15.0: return [op("MILL_BLIND_HOLE_FROM_SOLID_MATERIAL", d)]
    ops = [op("INDEXABLE_INSERT_DRILL_BLIND_HOLE_FROM_SOLID", d)]
    if d >= 32.0: ops.append(op("DRILL_TO_ENLARGE_BLIND_HOLE", d))
    return ops

_MODEL: object = None
def chain_model():
    global _MODEL
    if _MODEL is None:
        import pickle
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/chain_tree.pkl"
        _MODEL = pickle.load(p.open("rb")) if p.exists() else False
    return _MODEL

def hole_chains(found: Features) -> list[list[Op]]:
    holes, model = found.holes, chain_model()
    if not model or not holes:
        out = [hole_chain(h) for h in holes]
        for ops in out: assign_tools(ops, found)
        return out
    import numpy as np
    sx, sy, sz = found.stock
    X = np.array([[h.diameter, h.depth, h.depth / h.diameter, float(h.through),
                   sz, found.top_z - h.mouth_z, h.bottom_z,
                   h.diameter % 1.0, float(abs(h.diameter - round(h.diameter)) < 0.01),
                   min(h.x, sx - h.x, h.y, sy - h.y)] for h in holes])
    out = []
    for h, chain in zip(holes, model.predict(X), strict=True):
        names = chain.split(">")
        ops = [Op(n, feature=h) for n in names]
        out.append(ops)
    for ops in out: assign_tools(ops, found)
    return out

def assign_tools(ops: list[Op], found: Features) -> None:
    from . import tooling
    for i, o in enumerate(ops):
        o.tool_diameter = tooling.hole_tool_dia(o.name, o.feature, found, i, len(ops))
        o.tool_type = tooling.tool_type(o.name)
    for i, o in enumerate(ops[:-1]):
        if o.name != "SPOT_DRILL": o.extra["pilot"] = True
    pilots = [o for o in ops if o.name == "DRILL_BLIND_HOLE_INTO_CENTER"]
    for o in ops:
        if "GUN_DRILL" in o.name and pilots: o.extra["pilot_dia"] = pilots[0].tool_diameter

def is_mill_op(o: Op) -> bool: return o.name in MILL_ORDER or o.name == "AREA_MILL"

_ORDER: object = None
def order_model():
    global _ORDER
    if _ORDER is None:
        import pickle
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/order_model.pkl"
        _ORDER = pickle.load(p.open("rb")) if p.exists() else False
    return _ORDER

def order_features(found: Features) -> list[float]:
    hs, ps, cs = found.holes, found.pockets, found.chamfers
    tz = found.top_z
    sub = [h for h in hs if h.mouth_z < tz - 0.01]
    return [len(hs), len(ps), len(cs), sum(h.through for h in hs), sum(not h.through for h in hs),
            max((h.diameter for h in hs), default=0), min((h.diameter for h in hs), default=0),
            max((h.depth / h.diameter for h in hs), default=0),
            max((p.depth for p in ps), default=0), sum(p.area for p in ps),
            found.stock[2], found.stock[0] * found.stock[1],
            len(sub), len(sub) / max(len(hs), 1),
            sum(1 for h in hs if h.diameter >= 15), sum(1 for h in hs if h.depth / h.diameter >= 5)]

# fallback rule: any blind hole -> drilling first (91.9% holdout); model is 92.2%
def predict_drilling_first(found: Features) -> bool:
    model = order_model()
    if model: return not int(model.predict([order_features(found)])[0])
    return any(not h.through for h in found.holes) or (bool(found.holes) and len(found.pockets) > 6)

def plan(found: Features, drilling_first: bool | None = None) -> list[Op]:
    if drilling_first is None: drilling_first = predict_drilling_first(found)
    chains = hole_chains(found)
    spots = [c[0] for c in chains if c and c[0].name == "SPOT_DRILL"]
    per_hole = [op for c in chains for op in (c[1:] if c and c[0].name == "SPOT_DRILL" else c)]
    drill = spots + [o for o in per_hole if not is_mill_op(o)]
    mill = chamfer_ops(found) + pocket_ops(found) + [o for o in per_hole if is_mill_op(o)]
    mill.sort(key=lambda o: MILL_ORDER.index(o.name) if o.name in MILL_ORDER else 0)
    return drill + mill if drilling_first and drill else mill + drill

def features_from_row(row: dict) -> Features:
    from .brep import Size
    found = Features(stock=Size(*row["stock"]), top_z=row["top_z"], bottom_z=row["bottom_z"])
    found.holes = [Hole(x=h["x"], y=h["y"], diameter=h["d"], depth=h["depth"], through=h["through"],
                        bottom_z=h["bottom_z"], mouth_z=h.get("mouth_z", row["top_z"])) for h in row["holes"]]
    found.pockets = [Pocket(floor_z=p["floor_z"], depth=p["depth"], area=p["area"], kind=p["kind"],
                            open_sides=p["open_sides"], fillet_radius=p["fillet_radius"],
                            corners=p["corners"]) for p in row["pockets"]]
    found.chamfers = [Chamfer(width=c["width"], angle_deg=c["angle_deg"],
                              faces=list(range(c.get("n_faces", 1)))) for c in row["chamfers"]]
    return found

def plan_from_row(row: dict, drilling_first: bool | None = None) -> list[Op]:
    return plan(features_from_row(row), drilling_first)
