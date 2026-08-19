from __future__ import annotations

import math

import numpy as np

from . import brep
from .brep import Part
from .features import Features, Hole
from .ipw import op_diameter
from .plan import Op

# mined from GT ptp corpus (scripts/mine_ptp.py): R plane, Z targets, gun sequence
PROT = {"DRILL_THROUGH_HOLE_INTO_CENTER": (0.3004, 1.5, "G73"),
        "DRILL_TO_ENLARGE_THROUGH_HOLE": (0.3004, 1.5, "G73"),
        "SPADE_DRILL_TO_ENLARGE_THROUGH_HOLE": (0.2226, 1.501, "G73"),
        "INDEXABLE_INSERT_DRILL_THROUGH_HOLE_FROM_SOLID": (0.0175, 1.495, "G81"),
        "DRILL_THROUGH_HOLE_FROM_SOLID_MATERIAL": (0.3004, 1.5, "G81")}

def num(v: float) -> str:
    s = f"{v:.3f}".rstrip("0")
    return s if s.endswith(".") or "." in s else s + "."

class Prog:
    def __init__(self, partname: str, opname: str, tool: str):
        self.n = 10
        self.out = [f"(DATE            : 14.07.2026 , 17:20                      )",
                    f"(PARTNAME        : {partname.upper() + '.PRT':<41})"]
        self.line("G17 G21 G94 G90")
        self.sep()
        self.out.append(f"({opname} , TOOL : {tool})")
        self.sep()
        self.line("T00 M6")
        self.line("G54")
    def line(self, s: str): self.out.append(f"N{self.n} {s}"); self.n += 2
    def sep(self): self.out.append(" ")
    def text(self) -> str: return "\n".join(self.out) + "\n"

def spot_depth(h: Hole) -> float: return 3.0 if h.diameter > 17.45 else 0.043 * h.diameter

def cycle_params(op: Op, h: Hole, dt: float):
    mouth = h.mouth_z
    n = op.name
    if n == "SPOT_DRILL": return "G81", mouth - spot_depth(h), mouth + 3, None
    if n == "BORE_BLIND_HOLE": return "G85", h.bottom_z, mouth + 2, None
    if not h.through: return "G81", h.bottom_z, mouth + 2, None
    if n == "DRILL_BLIND_HOLE_INTO_CENTER": return "G81", mouth - 1.5 * dt, mouth + 2, None
    if n == "DRILL_THROUGH_HOLE_INTO_CENTER" and op.extra.get("pilot"):
        return "G73", h.bottom_z - (0.182 * dt + 1.5), mouth + 2, 0.94 * dt
    a, b, cyc = PROT.get(n) or PROT.get(n.replace("BLIND", "THROUGH")) or (0.33, 0.8, "G81")
    return cyc, h.bottom_z - (a * dt + b), mouth + 2, (0.94 * dt if cyc == "G73" else None)

def emit_cycle(p: Prog, op: Op, h: Hole, zc: float, dt: float):
    cyc, z, r, q = cycle_params(op, h, dt)
    p.line(f"G17 G0 G90 X{num(h.x)} Y{num(h.y)} S0 M3")
    p.line(f"G43 Z{num(zc)} H0")
    qs = f" Q{num(q)}" if q else ""
    p.line(f"G94 {cyc} G98 Z{num(z)} F250.{qs} R{num(r)}")
    p.line("G80")
    p.line("M5")
    p.line("M2")

def emit_gun(p: Prog, op: Op, h: Hole, zc: float, dt: float):
    mouth = h.mouth_z
    pilot = op.extra.get("pilot_dia") or op_diameter(Op("DRILL_BLIND_HOLE_INTO_CENTER", feature=h))
    rapid = mouth - 2.6
    zfin = h.bottom_z - (0.2319 * dt + 1.5165) if h.through else h.bottom_z
    p.line(f"G17 G0 G90 X{num(h.x)} Y{num(h.y)} M8")
    p.line(f"G43 Z{num(zc)} H0")
    p.line(f"Z{num(rapid)}")
    p.line("S716 M3")
    p.line(f"G94 G1 Z{num(mouth - 1.5 * pilot)} F125. M26")
    p.line("G4 X.084")
    p.line(f"Z{num(h.bottom_z + 0.12 * h.diameter)} F250.")
    p.line("G4 X.084")
    p.line(f"Z{num(zfin)} F125.")
    p.line("G4 X.084")
    p.line(f"G0 Z{num(rapid)}")
    p.line("M5")
    p.line("M9")
    p.line(f"Z{num(zc)}")
    p.line("M2")

def emit_mill_hole(p: Prog, op: Op, h: Hole, zc: float, dt: float, top: float):
    z_bot = h.bottom_z
    rp = h.diameter / 2 - dt / 2
    p.line(f"G17 G0 G90 X{num(h.x + rp)} Y{num(h.y)} S0 M3")
    p.line(f"G43 Z{num(zc)} H0")
    if rp < 0.05:
        p.line(f"Z{num(z_bot)}")
        p.line("G80")
    else:
        step = max(0.5 * dt, 1.0)
        levels = np.arange(h.mouth_z - step, z_bot, -step).tolist() + [z_bot]
        first = True
        for z in levels:
            p.line(f"{'G94 G1 ' if first else 'G1 '}Z{num(z)} F250.")
            first = False
            p.line(f"G3 X{num(h.x - rp)} Y{num(h.y)} R{num(rp)}")
            p.line(f"G3 X{num(h.x + rp)} Y{num(h.y)} R{num(rp)}")
    p.line(f"G0 Z{num(zc)}")
    p.line("M5")
    p.line("M2")

def _rings(poly_xy: list[tuple[float, float]], dt: float):
    from shapely.geometry import Polygon
    poly = Polygon(poly_xy)
    off, rings = 0.5 * dt, []
    while True:
        inner = poly.buffer(-off, join_style=2)
        if inner.is_empty: break
        geoms = inner.geoms if inner.geom_type == "MultiPolygon" else [inner]
        rings.extend(list(g.exterior.coords) for g in geoms)
        off += 0.65 * dt
    return rings[::-1]  # inside-out

def emit_pocket(p: Prog, op: Op, part: Part, zc: float, dt: float, top: float):
    pk = op.feature
    poly = [(x, y) for x, y, _ in brep.outline(part, pk.faces[0])]
    rings = _rings(poly, dt)
    if not rings: rings = [[(sum(x for x, _ in poly) / len(poly), sum(y for _, y in poly) / len(poly))] * 2]
    step = min(0.5 * dt, pk.depth) or pk.depth
    levels = np.arange(pk.floor_z + pk.depth - step, pk.floor_z, -step).tolist() + [pk.floor_z]
    x0, y0 = rings[0][0]
    p.line(f"G17 G0 G90 X{num(x0)} Y{num(y0)} S0 M3")
    p.line(f"G43 Z{num(zc)} H0")
    for z in levels:
        p.line(f"G0 X{num(x0)} Y{num(y0)}")
        p.line(f"G94 G1 Z{num(z)} F250.")
        for ring in rings:
            for x, y in ring: p.line(f"G1 X{num(x)} Y{num(y)} F500.")
    p.line(f"G0 Z{num(zc)}")
    p.line("M5")
    p.line("M2")

def emit_area_mill(p: Prog, op: Op, part: Part, zc: float):
    fid = op.extra.get("face")
    fids = [fid] if isinstance(fid, int) else op.feature.faces
    pts = [pt for f in fids for pt in brep.outline(part, f)]
    x0, y0, z0 = pts[0]
    p.line(f"G17 G0 G90 X{num(x0)} Y{num(y0)} S1061 M3")
    p.line(f"G43 Z{num(zc)} H0")
    p.line(f"Z{num(z0 + 1)}")
    first = True
    for x, y, z in pts:
        p.line(f"{'G94 G1 ' if first else ''}X{num(x)} Y{num(y)} Z{num(z)}{' F250.' if first else ''}")
        first = False
    p.line(f"G0 Z{num(zc)}")
    p.line("M5")
    p.line("M2")

def emit(partname: str, op: Op, found: Features, part: Part) -> str:
    zc = found.top_z + 10.0
    tool = op.tool_type or "TOOL"
    p = Prog(partname, op.name, tool)
    n, feat = op.name, op.feature
    if n == "AREA_MILL": emit_area_mill(p, op, part, zc)
    elif feat.__class__.__name__ == "Pocket":
        emit_pocket(p, op, part, zc, op.tool_diameter or 2 * (feat.fillet_radius or 5.0), found.top_z)
    elif "GUN_DRILL" in n: emit_gun(p, op, feat, zc, op_diameter(op))
    elif n.startswith("MILL_"): emit_mill_hole(p, op, feat, zc, op_diameter(op), found.top_z)
    else: emit_cycle(p, op, feat, zc, op_diameter(op))
    return p.text()
