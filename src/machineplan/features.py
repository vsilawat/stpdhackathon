from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .brep import CONCAVE, FULL_TURN, TANGENT, Box, Cylinder, FaceId, Part, Plane, Size

PocketKind = Literal["center", "edge", "corner", "slot"]
Side = Literal["x0", "x1", "y0", "y1"]

CENTRE: PocketKind = "center"; EDGE: PocketKind = "edge"
CORNER: PocketKind = "corner"; SLOT: PocketKind = "slot"

TOL = 1e-4
GRID = 4
OPPOSITE: tuple[set[Side], set[Side]] = ({"x0", "x1"}, {"y0", "y1"})
BY_OPEN: dict[int, PocketKind] = {0: CENTRE, 1: EDGE, 2: CORNER}
BY_CORNERS: dict[int, PocketKind] = {4: CENTRE, 2: EDGE, 1: CORNER, 0: SLOT}

@dataclass(slots=True)
class Hole:
    x: float
    y: float
    diameter: float
    depth: float
    through: bool
    bottom_z: float
    mouth_z: float = 0.0
    faces: list[FaceId] = field(default_factory=list)

@dataclass(slots=True)
class Pocket:
    floor_z: float
    depth: float
    area: float
    kind: PocketKind
    open_sides: int
    fillet_radius: float | None
    corners: int
    faces: list[FaceId] = field(default_factory=list)

@dataclass(slots=True)
class Chamfer:
    width: float
    angle_deg: float
    faces: list[FaceId] = field(default_factory=list)

@dataclass(slots=True)
class Features:
    stock: Size
    top_z: float
    bottom_z: float
    holes: list[Hole] = field(default_factory=list)
    pockets: list[Pocket] = field(default_factory=list)
    chamfers: list[Chamfer] = field(default_factory=list)
    @property
    def counts(self): return {"chamfers": len(self.chamfers), "pockets": len(self.pockets), "holes": len(self.holes)}
    @property
    def total(self): return len(self.chamfers) + len(self.pockets) + len(self.holes)

def _is_blend(part: Part, face: Cylinder):
    # A pocket corner runs smoothly into its walls; a hole sliced open by a pocket does not.
    if face.is_full_circle or not face.is_upright: return False
    return sum(1 for e in part.edges if face.index in e.faces and e.kind == TANGENT) >= 2

def _hole_bottom(part: Part, floor: Plane):
    # circular-ish floor bounded by an upright cylinder centred on it
    b = floor.bbox
    for i in part.neighbours(floor.index):
        f = part.faces[i]
        if not isinstance(f, Cylinder) or not f.is_upright: continue
        r = f.radius
        if (b.xmax - b.xmin <= 2 * r + 1e-3 and b.ymax - b.ymin <= 2 * r + 1e-3
                and floor.area <= math.pi * r * r + 1e-3
                and abs((b.xmax + b.xmin) / 2 - f.origin[0]) < 1e-3
                and abs((b.ymax + b.ymin) / 2 - f.origin[1]) < 1e-3): return True
    return False

def _open_sides(floor: Plane, stock: Box) -> set[Side]:
    reach: tuple[tuple[Side, float, float], ...] = (
        ("x0", floor.bbox.xmin, stock.xmin), ("x1", floor.bbox.xmax, stock.xmax),
        ("y0", floor.bbox.ymin, stock.ymin), ("y1", floor.bbox.ymax, stock.ymax))
    return {name for name, got, want in reach if abs(got - want) < TOL}

def _pocket_kind(open_at: set[Side], corners: int) -> PocketKind:
    if open_at in OPPOSITE: return SLOT
    return BY_OPEN.get(len(open_at)) or BY_CORNERS.get(corners, CENTRE)

def find_chamfers(part: Part) -> list[Chamfer]:
    groups: dict[tuple[float, ...], list[Plane]] = {}
    for f in part.planes():
        if not f.is_tilted: continue
        offset = sum(n * o for n, o in zip(f.normal, f.origin, strict=True))
        groups.setdefault((*(round(c, GRID) for c in f.normal), round(offset, GRID)), []).append(f)
    out = []
    for faces in groups.values():
        drop = max(f.bbox.zmax for f in faces) - min(f.bbox.zmin for f in faces)
        angle = math.degrees(math.acos(min(1.0, abs(faces[0].normal[2]))))
        width = drop / math.tan(math.radians(angle)) if angle not in (0.0, 90.0) else drop
        out.append(Chamfer(width=width, angle_deg=angle, faces=[f.index for f in faces]))
    return out

def find_holes(part: Part, top_z: float, bottom_z: float) -> list[Hole]:
    # partial arcs count as a hole when one axis+radius covers most of a turn
    groups: dict[tuple[float, float, float], list[Cylinder]] = {}
    for f in part.cylinders():
        if not f.is_upright or (not f.is_full_circle and _is_blend(part, f)): continue
        groups.setdefault((round(f.origin[0], GRID), round(f.origin[1], GRID), round(f.radius, GRID)), []).append(f)
    out = []
    for (x, y, radius), faces in groups.items():
        by_span: dict[tuple[float, float], float] = {}
        for f in faces: by_span[(round(f.bbox.zmin, 3), round(f.bbox.zmax, 3))] = by_span.get((round(f.bbox.zmin, 3), round(f.bbox.zmax, 3)), 0.0) + f.sweep
        if not any(f.is_full_circle for f in faces) and max(by_span.values()) < 0.55 * FULL_TURN: continue
        low = min(f.bbox.zmin for f in faces)
        out.append(Hole(x=x, y=y, diameter=2 * radius, depth=top_z - low,
                        through=abs(low - bottom_z) < TOL, bottom_z=low,
                        mouth_z=max(f.bbox.zmax for f in faces), faces=[f.index for f in faces]))
    return out

def find_pockets(part: Part, top_z: float, hole_faces: set[FaceId] = frozenset()) -> list[Pocket]:
    out = []
    for floor in part.planes():
        if not floor.faces_up or floor.bbox.zmin >= top_z - 1e-2: continue
        if _hole_bottom(part, floor): continue
        hole_nbrs = [part.faces[i] for i in part.neighbours(floor.index) if i in hole_faces]
        if hole_nbrs and (floor.area < 100
                          or any(floor.area <= math.pi * f.radius ** 2 + 1e-3 for f in hole_nbrs)): continue
        sides = [part.faces[i] for i, kind in part.neighbours(floor.index).items() if kind == CONCAVE]
        blends = [f for f in sides if isinstance(f, Cylinder) and _is_blend(part, f)]
        radii = {round(b.radius, GRID) for b in blends}
        open_at = _open_sides(floor, part.bbox)
        out.append(Pocket(floor_z=floor.bbox.zmin, depth=top_z - floor.bbox.zmin, area=floor.area,
                          kind=_pocket_kind(open_at, len(blends)), open_sides=len(open_at),
                          fillet_radius=(radii.pop() if len(radii) == 1 else None),
                          corners=len(blends), faces=[floor.index, *(f.index for f in sides)]))
    return out

def extract(part: Part) -> Features:
    found = Features(stock=part.size, top_z=part.bbox.zmax, bottom_z=part.bbox.zmin)
    found.chamfers = find_chamfers(part)
    found.holes = find_holes(part, top_z=found.top_z, bottom_z=found.bottom_z)
    hole_faces = {i for h in found.holes for i in h.faces}
    found.pockets = find_pockets(part, top_z=found.top_z, hole_faces=hole_faces)
    return found
