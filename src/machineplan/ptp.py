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
        self.out = ["(DATE            : 14.07.2026 , 17:20                      )",
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

_PA: dict | None = None
def point_angle(name: str, dt: float) -> float | None:
    global _PA
    if _PA is None:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "derived/point_angles.json"
        _PA = json.loads(p.read_text()) if p.exists() else {}
    return _PA.get(f"{name}|{round(dt, 2)}")

# tip protrusion from tool point angle (details.txt PA), fallback to mined constants
def prot_z(name: str, h: Hole, dt: float, default: float) -> float:
    pa = point_angle(name, dt)
    if pa and pa < 179.0: return h.bottom_z - (dt / 2 / math.tan(math.radians(pa) / 2) + 1.5)
    return default

def cycle_params(op: Op, h: Hole, dt: float):
    mouth = h.mouth_z
    n = op.name
    if n == "SPOT_DRILL": return "G81", mouth - spot_depth(h), mouth + 3, None
    if n == "BORE_BLIND_HOLE": return "G85", h.bottom_z, mouth + 2, None
    if not h.through: return "G81", h.bottom_z, mouth + 2, None
    if n == "DRILL_BLIND_HOLE_INTO_CENTER":
        z = mouth - (1.5019 * dt + 0.02) if h.through else h.bottom_z
        return "G81", z, mouth + 2, None
    if n == "DRILL_THROUGH_HOLE_INTO_CENTER" and op.extra.get("pilot"):
        return "G73", prot_z(n, h, dt, h.bottom_z - (0.182 * dt + 1.5)), mouth + 2, 0.94 * dt
    if n == "DRILL_TO_ENLARGE_THROUGH_HOLE" and op.extra.get("gun_chain") and dt < 15:
        return "G73", prot_z(n, h, dt, h.bottom_z - (0.182 * dt + 1.5)), mouth + 2, 0.94 * dt
    a, b, cyc = PROT.get(n) or PROT.get(n.replace("BLIND", "THROUGH")) or (0.33, 0.8, "G81")
    return cyc, prot_z(n, h, dt, h.bottom_z - (a * dt + b)), mouth + 2, (0.94 * dt if cyc == "G73" else None)

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

# open sides: tool center overruns the stock boundary by ~r (GT slots run r past both ends)
def _extend_open(poly, dt: float, sx: float, sy: float):
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    ext, tol = [], 0.5
    cs = list(poly.exterior.coords)
    for (x1, y1), (x2, y2) in zip(cs, cs[1:]):
        if (abs(x1) < tol and abs(x2) < tol) or (abs(x1 - sx) < tol and abs(x2 - sx) < tol) \
           or (abs(y1) < tol and abs(y2) < tol) or (abs(y1 - sy) < tol and abs(y2 - sy) < tol):
            ext.append(LineString([(x1, y1), (x2, y2)]).buffer(dt + 0.5, cap_style=2))
    return unary_union([poly] + ext) if ext else poly

def _rings(poly_xy: list[tuple[float, float]], dt: float, stock=None):
    from shapely.geometry import Polygon
    poly = Polygon(poly_xy)
    if not poly.is_valid: poly = poly.buffer(0)
    if stock: poly = _extend_open(poly, dt, stock[0], stock[1])
    off, rings = 0.5 * dt, []
    while True:
        inner = poly.buffer(-off, join_style=2)
        if inner.is_empty: break
        geoms = inner.geoms if inner.geom_type == "MultiPolygon" else [inner]
        rings.extend(list(g.exterior.coords) for g in geoms)
        off += 0.65 * dt
    return rings[::-1]  # inside-out

# NX open-slot template: rails at wall+r, corner pivot arcs, cross pass at far end + r
def emit_slot(p: Prog, op: Op, part: Part, zc: float, dt: float):
    from shapely.geometry import Polygon
    pk = op.feature
    pts = [(x, y) for x, y, _ in brep.outline(part, pk.faces[0])]
    C = np.array(Polygon(pts).minimum_rotated_rectangle.exterior.coords[:4])
    e0, e1 = C[1] - C[0], C[2] - C[1]
    l0, l1 = np.linalg.norm(e0), np.linalg.norm(e1)
    uhat, L, W = (e0 / l0, l0, l1) if l0 >= l1 else (e1 / l1, l1, l0)
    ctr = C.mean(axis=0)
    endA, endB = ctr - uhat * L / 2, ctr + uhat * L / 2
    if (round(endB[1], 3), round(endB[0], 3)) < (round(endA[1], 3), round(endA[0], 3)): uhat = -uhat
    vhat = np.array([-uhat[1], uhat[0]])
    r = dt / 2
    vr, vl = max(W / 2 - r, 0.0), min(-W / 2 + r, 0.0)
    un, uf = -L / 2, L / 2

    def P(u, v): return ctr + u * uhat + v * vhat

    def arc(cu, cv, a0, a1):
        return [P(cu + r * np.cos(np.radians(a)), cv + r * np.sin(np.radians(a)))
                for a in np.linspace(a0, a1, 7)]
    loop = (arc(un, W / 2, 180, 270) + [P(uf, vr)] + arc(uf, W / 2, 270, 360)
            + [P(uf + r, -W / 2)] + arc(uf, -W / 2, 0, 90) + [P(un, vl)] + arc(un, -W / 2, 90, 180))
    step = min(0.5 * dt, pk.depth) or pk.depth
    levels = np.arange(pk.floor_z + pk.depth - step, pk.floor_z, -step).tolist() + [pk.floor_z]
    x0, y0 = loop[0]
    p.line(f"G17 G0 G90 X{num(x0)} Y{num(y0)} S0 M3")
    p.line(f"G43 Z{num(zc)} H0")
    for z in levels:
        p.line(f"G0 X{num(x0)} Y{num(y0)}")
        p.line(f"G94 G1 Z{num(z)} F250.")
        for x, y in loop: p.line(f"G1 X{num(x)} Y{num(y)} F500.")
    p.line(f"G0 Z{num(zc)}")
    p.line("M5")
    p.line("M2")

def emit_pocket(p: Prog, op: Op, part: Part, zc: float, dt: float, top: float, stock=None):
    pk = op.feature
    poly = [(x, y) for x, y, _ in brep.outline(part, pk.faces[0])]
    rings = _rings(poly, dt, stock if (pk.open_sides or 0) > 0 else None)
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

# GT strategy (mined from corpus): perimeter loops inset in-plane by 2.0mm/pass,
# whole path shifted 1.5mm horizontally downslope, z riding the face
def _area_loops(pts):
    from shapely.geometry import Polygon
    P = np.array(pts, dtype=float)
    n = np.zeros(3)
    for a, b in zip(P, np.roll(P, -1, axis=0)): n += np.cross(a, b)
    if np.linalg.norm(n) < 1e-9 or abs(n[2]) / np.linalg.norm(n) < 0.05: return [pts]
    n /= np.linalg.norm(n)
    if n[2] < 0: n = -n
    h = np.array([n[0], n[1], 0.0])
    hn = np.linalg.norm(h)
    if hn < 1e-6: return [pts]
    d = h / hn
    e2 = np.array([-d[1], d[0], 0.0])
    e1 = np.cross(e2, n)
    p0 = P[0]
    uv = [((q - p0) @ e1, (q - p0) @ e2) for q in P]
    poly = Polygon(uv)
    if not poly.is_valid: poly = poly.buffer(0)
    shift = 1.5 * d
    loops, off = [], 0.0
    while True:
        inner = poly.buffer(-off, join_style=2) if off else poly
        if inner.is_empty: break
        geoms = inner.geoms if inner.geom_type == "MultiPolygon" else [inner]
        for g in geoms:
            loops.append([tuple(p0 + u * e1 + v * e2 + shift) for u, v in g.exterior.coords])
        off += 2.0
    return loops or [pts]

def emit_area_mill(p: Prog, op: Op, part: Part, zc: float):
    fid = op.extra.get("face")
    fids = [fid] if isinstance(fid, int) else op.feature.faces
    loops = [lp for f in fids for lp in _area_loops(brep.outline(part, f))]
    x0, y0, z0 = loops[0][0]
    p.line(f"G17 G0 G90 X{num(x0)} Y{num(y0)} S1061 M3")
    p.line(f"G43 Z{num(zc)} H0")
    p.line(f"Z{num(z0 + 1)}")
    first = True
    for lp in loops:
        for x, y, z in lp:
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
    elif feat.__class__.__name__ == "Pocket" and n == "MILL_RECTANGULAR_SLOT" and (feat.open_sides or 0) >= 2:
        emit_slot(p, op, part, zc, op.tool_diameter or 2 * (feat.fillet_radius or 5.0))
    elif feat.__class__.__name__ == "Pocket":
        emit_pocket(p, op, part, zc, op.tool_diameter or 2 * (feat.fillet_radius or 5.0), found.top_z, found.stock)
    elif "GUN_DRILL" in n: emit_gun(p, op, feat, zc, op_diameter(op))
    elif n.startswith("MILL_"): emit_mill_hole(p, op, feat, zc, op_diameter(op), found.top_z)
    else: emit_cycle(p, op, feat, zc, op_diameter(op))
    return p.text()
