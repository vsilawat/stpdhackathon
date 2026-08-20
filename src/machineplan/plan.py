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
MILL_ORDER = ["AREA_MILL", "MILL_CORNER_NOTCH_RECTANGULAR", "MILL_OPEN_POCKET", "MILL_RECTANGULAR_POCKET",
              "MILL_SLOT", "MILL_RECTANGULAR_SLOT", "MILL_THROUGH_HOLE_FROM_SOLID_MATERIAL",
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
    ops = [Op("AREA_MILL", feature=c, tool_type="chamfer_mill", tool_diameter=20.0, extra={"face": f})
           for c in found.chamfers for f in (c.faces or [None])]
    # GT orders chamfer ops by face id globally, not grouped by chamfer
    return sorted(ops, key=lambda o: (o.extra["face"] is None, o.extra["face"]))

_PKIND: object = None
def pocket_kind_model():
    global _PKIND
    if _PKIND is None:
        import pickle
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/pocket_kind_model.pkl"
        _PKIND = pickle.load(p.open("rb")) if p.exists() else False
    return _PKIND

def pocket_feats(p: Pocket) -> list[float]:
    from .tooling import pocket_dia_feats
    return pocket_dia_feats(p)

# 31% of GT parts repeat a base kind -> per-pocket argmax, never force-distinct
def pocket_names(ps: list[Pocket]) -> list[str]:
    if not ps: return []
    m = pocket_kind_model()
    if not m or any(p.w <= 0 for p in ps): return [POCKET_OP[p.kind] for p in ps]
    import numpy as np
    P = m.predict_proba([pocket_feats(p) for p in ps])
    cls = list(m.classes_)
    return [str(cls[int(np.argmax(row))]) for row in P]

def pocket_ops(found: Features) -> list[Op]:
    from . import tooling
    return [Op(nm, feature=p, tool_type="end_mill", tool_diameter=d)
            for nm, p, d in zip(pocket_names(found.pockets), found.pockets,
                                tooling.pocket_dias(found.pockets), strict=True)]

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
    lut = tooling.chain_dias([o.name for o in ops], ops[0].feature.diameter) if ops else None
    if lut:
        for o, d in zip(ops, lut, strict=True):
            o.tool_diameter = d
            o.tool_type = tooling.tool_type(o.name)
        if "MILL" not in ops[-1].name:  # drill/bore finals cut the exact hole dia; mill finals use lib tools
            ops[-1].tool_diameter = ops[-1].feature.diameter
    else:
        for i, o in enumerate(ops):
            o.tool_diameter = tooling.hole_tool_dia(o.name, o.feature, found, i, len(ops))
            o.tool_type = tooling.tool_type(o.name)
        # hole mills cut at the hole dia, except a pre-bore rough mill which undersizes
        has_bore = any(o.name == "BORE_BLIND_HOLE" for o in ops)
        for o in ops:
            if o.name.startswith("MILL_ROUGH_BLIND_HOLE"): o.tool_diameter = o.feature.diameter
            elif o.name.startswith("MILL_BLIND_HOLE_FROM_SOLID"):
                o.tool_diameter = tooling.prebore_mill_dia(o.feature.diameter) if has_bore else o.feature.diameter
    for i, o in enumerate(ops[:-1]):
        if o.name != "SPOT_DRILL": o.extra["pilot"] = True
    pilots = [o for o in ops if o.name == "DRILL_BLIND_HOLE_INTO_CENTER"]
    gun = next((o for o in ops if "GUN_DRILL" in o.name), None)
    if gun and pilots and not lut and gun.tool_diameter and ops[-1].tool_diameter:
        d = tooling.gun_pilot(gun.tool_diameter, ops[-1].tool_diameter)
        if d: pilots[0].tool_diameter = d
    for o in ops:
        if "GUN_DRILL" in o.name and pilots: o.extra["pilot_dia"] = pilots[0].tool_diameter
    if any("GUN_DRILL" in o.name for o in ops):
        for o in ops:
            if o.name == "DRILL_TO_ENLARGE_THROUGH_HOLE": o.extra["gun_chain"] = True

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

_HM: object = None
def hm_model():
    global _HM
    if _HM is None:
        import pickle
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/hm_model.pkl"
        _HM = pickle.load(p.open("rb")) if p.exists() else False
    return _HM

def chain_names(found: Features) -> list[list[str]]:
    holes, model = found.holes, chain_model()
    if not holes: return []
    if not model: return [[o.name for o in hole_chain(h)] for h in holes]
    import numpy as np
    sx, sy, sz = found.stock
    X = np.array([[h.diameter, h.depth, h.depth / h.diameter, float(h.through),
                   sz, found.top_z - h.mouth_z, h.bottom_z,
                   h.diameter % 1.0, float(abs(h.diameter - round(h.diameter)) < 0.01),
                   min(h.x, sx - h.x, h.y, sy - h.y)] for h in holes])
    return [c.split(">") for c in model.predict(X)]

def order_features(found: Features) -> list[float]:
    hs, ps, cs = found.holes, found.pockets, found.chamfers
    tz = found.top_z
    sub = [h for h in hs if h.mouth_z < tz - 0.01]
    chains = chain_names(found)
    names = [n for c in chains for n in c]
    finals = [c[-1] for c in chains if c]
    deps = [h.depth for h in hs]
    bl = [h.depth for h in hs if not h.through]
    th = [h.depth for h in hs if h.through]
    lds = [h.depth / h.diameter for h in hs if h.diameter]
    return [len(hs), len(ps), len(cs), sum(h.through for h in hs), sum(not h.through for h in hs),
            max((h.diameter for h in hs), default=0), min((h.diameter for h in hs), default=0),
            max((h.depth / h.diameter for h in hs), default=0),
            max((p.depth for p in ps), default=0), sum(p.area for p in ps),
            found.stock[2], found.stock[0] * found.stock[1],
            len(sub), len(sub) / max(len(hs), 1),
            sum(1 for h in hs if h.diameter >= 15), sum(1 for h in hs if h.depth / h.diameter >= 5),
            sum(f in INSB_FINAL for f in finals), sum(n.startswith("MILL_") for n in names),
            sum("GUN" in n for n in names), sum(n == "INDEXABLE_INSERT_DRILL_THROUGH_HOLE_FROM_SOLID" for n in names),
            sum(n == "SPOT_DRILL" for n in names), sum(n == "BORE_BLIND_HOLE" for n in names),
            sum(n == "DRILL_BLIND_HOLE_INTO_CENTER" for n in finals),
            min(deps, default=0), max(deps, default=0), sum(deps) / max(len(deps), 1),
            min(bl, default=0), max(bl, default=0), min(th, default=0),
            min(lds, default=0), sum(lds) / max(len(lds), 1),
            min((tz - h.mouth_z for h in hs), default=0), max((tz - h.mouth_z for h in hs), default=0),
            len(chains), sum(len(c) for c in chains) / max(len(chains), 1),
            sum("HOLE" in n and n.startswith("MILL_") for n in names),
            found.stock[0], found.stock[1], tz,
            min((p.depth for p in ps), default=0), sum(p.kind == "edge" for p in ps),
            min((h.bottom_z for h in hs), default=0),
            sum(h.diameter * h.diameter * h.depth for h in hs) / 1000.0,
            *_floor_match_feats(hs, ps, tz)]

def _floor_match_feats(hs, ps, tz: float) -> list[float]:
    matched = []
    for h in hs:
        if h.mouth_z >= tz - 0.01: continue
        m = [p for p in ps if abs(p.floor_z - h.mouth_z) < 0.05]
        if m: matched.append((h, max(m, key=lambda p: p.area)))
    return [len(matched), len(matched) / max(len(hs), 1),
            sum(h.through for h, _ in matched), sum(not h.through for h, _ in matched),
            max((h.diameter for h, _ in matched), default=0),
            min((h.diameter for h, _ in matched), default=0),
            sum(1 for h, _ in matched if h.diameter < 15),
            max((p.depth for _, p in matched), default=0),
            sum(p.area for _, p in matched),
            sum(1 for h in hs if h.mouth_z < tz - 0.01) - len(matched),
            len({round(p.floor_z, 1) for p in ps})]

# model predicts the spot/twist block's placement relative to the mill block
def predict_drilling_first(found: Features) -> bool:
    model = order_model()
    if model: return bool(model.predict([order_features(found)])[0])
    return any(not h.through for h in found.holes) or (bool(found.holes) and len(found.pockets) > 6)

# spot block sits at the first spotted chain's position (R4, 90.5% of GT drill runs)
def drill_phase(chains: list[list[Op]], keep_mill: bool) -> tuple[list[Op], list[Op]]:
    spots = [c[0] for c in chains if c and c[0].name == "SPOT_DRILL"]
    out, milled, placed = [], [], False
    for c in chains:
        rem = c[1:] if c and c[0].name == "SPOT_DRILL" else c
        if c and c[0].name == "SPOT_DRILL" and not placed: out += spots; placed = True
        prev_milled = False
        for o in rem:
            # a bore stays glued to its hole's mill op (GT: bore follows the HM)
            if not keep_mill and (is_mill_op(o) or (o.name == "BORE_BLIND_HOLE" and prev_milled)):
                milled.append(o); prev_milled = True
            else: out.append(o); prev_milled = False
    return out, milled

KEEP_MILL_IN_CHAIN = False

# insert-blind chains lead the whole plan (93.7% of GT); when the twist block
# is first, hole-mill ops glue to its end (100%), else to the mill block's end
INSB_FINAL = ("INDEXABLE_INSERT_DRILL_BLIND_HOLE_FROM_SOLID", "DRILL_TO_ENLARGE_BLIND_HOLE")

def plan(found: Features, drilling_first: bool | None = None) -> list[Op]:
    if drilling_first is None: drilling_first = predict_drilling_first(found)
    chains = sorted(hole_chains(found), key=lambda c: not (c and c[0].feature and c[0].feature.through))
    # insert-blind chains lead a mill-first plan but stay at their chain position
    # among the drill chains when the twist block leads (GT run patterns)
    front = [] if drilling_first else [c for c in chains if c and c[-1].name in INSB_FINAL]
    rest = [c for c in chains if c not in front]
    drill, hole_mills = drill_phase(rest, KEEP_MILL_IN_CHAIN)
    lead = [o for c in front for o in c]
    units: list[list[Op]] = []
    for o in hole_mills:
        if o.name == "BORE_BLIND_HOLE" and units: units[-1].append(o)
        else: units.append([o])
    units.sort(key=lambda u: MILL_ORDER.index(u[0].name) if u[0].name in MILL_ORDER else 0)
    hole_mills = [o for u in units for o in u]
    mill = chamfer_ops(found) + pocket_ops(found)
    mill.sort(key=lambda o: MILL_ORDER.index(o.name) if o.name in MILL_ORDER else 0)
    if drilling_first and drill:
        m = hm_model()
        # when the twist block leads, insert-blind chains follow it (TW>I>M, 583 vs 214 GT);
        # hole mills glue to the twist block's end unless the model says they trail the mills
        if hole_mills and m and not bool(m.predict([order_features(found)])[0]):
            return drill + lead + mill + hole_mills
        return drill + lead + hole_mills + mill
    return lead + mill + hole_mills + drill

def features_from_row(row: dict) -> Features:
    from .brep import Size
    found = Features(stock=Size(*row["stock"]), top_z=row["top_z"], bottom_z=row["bottom_z"])
    found.holes = [Hole(x=h["x"], y=h["y"], diameter=h["d"], depth=h["depth"], through=h["through"],
                        bottom_z=h["bottom_z"], mouth_z=h.get("mouth_z", row["top_z"]),
                        flat=h.get("flat", False)) for h in row["holes"]]
    found.pockets = [Pocket(floor_z=p["floor_z"], depth=p["depth"], area=p["area"], kind=p["kind"],
                            open_sides=p["open_sides"], fillet_radius=p["fillet_radius"], corners=p["corners"],
                            w=p.get("w", 0.0), l=p.get("l", 0.0), mi=p.get("mi", 0.0), hull=p.get("hull", 1.0))
                     for p in row["pockets"]]
    found.chamfers = [Chamfer(width=c["width"], angle_deg=c["angle_deg"],
                              faces=list(range(c.get("n_faces", 1)))) for c in row["chamfers"]]
    return found

def plan_from_row(row: dict, drilling_first: bool | None = None) -> list[Op]:
    return plan(features_from_row(row), drilling_first)
