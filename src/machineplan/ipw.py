from __future__ import annotations

import manifold3d as m3d

from . import solids
from .brep import Part
from .features import Features
from .mesh import stock_box
from .plan import Op

TIP_ANGLE = 118.0
PILOT_RATIO = 0.627
GUN_RATIO = 0.387
SPADE_RATIO = 0.644

def op_diameter(op: Op) -> float:
    if op.tool_diameter is not None: return op.tool_diameter
    d = op.feature.diameter
    match op.name:
        case "GUN_DRILL_THROUGH_HOLE": return GUN_RATIO * d
        case "SPADE_DRILL_TO_ENLARGE_THROUGH_HOLE": return SPADE_RATIO * d
        case "DRILL_THROUGH_HOLE_INTO_CENTER": return PILOT_RATIO * d
        case "DRILL_BLIND_HOLE_INTO_CENTER": return GUN_RATIO * d
        case _: return d

def op_solid(part: Part, op: Op, found: Features) -> m3d.Manifold | None:
    top, bot = found.top_z, found.bottom_z
    n = op.name
    if n == "AREA_MILL":
        fid = op.extra.get("face")
        return (solids.chamfer_face_solid(part, fid, top) if isinstance(fid, int)
                else solids.chamfer_solid(part, op.feature, top))
    if op.feature.__class__.__name__ == "Pocket": return solids.pocket_solid(part, op.feature, top)
    h = op.feature
    if n == "SPOT_DRILL":
        return solids.spot_solid(h.x, h.y, 0.0, h.mouth_z, 3.0 if h.diameter > 17.45 else 0.043 * h.diameter)
    r = op_diameter(op) / 2
    flat = n.startswith(("MILL_", "INDEXABLE", "BORE"))
    if h.through and n == "DRILL_BLIND_HOLE_INTO_CENTER":
        return solids.hole_solid(h.x, h.y, r, top, h.mouth_z - 3.0 * r, tip_angle_deg=TIP_ANGLE)
    if h.through: return solids.hole_solid(h.x, h.y, r, top, bot - solids.EPS)
    return solids.hole_solid(h.x, h.y, r, top, h.bottom_z, tip_angle_deg=None if flat else TIP_ANGLE)

def ipws(part: Part, found: Features, ops: list[Op]) -> list[m3d.Manifold]:
    cur = stock_box(part.bbox)
    out = []
    for op in ops:
        sol = op_solid(part, op, found)
        if sol is not None: cur = cur - sol
        out.append(cur)
    return out
