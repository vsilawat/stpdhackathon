from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GCPnts import GCPnts_TangentialDeflection
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Dir, gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

Vec3 = tuple[float, float, float]
FaceId = int
Corner = Literal["convex", "concave", "tangent"]
CONVEX: Corner = "convex"; CONCAVE: Corner = "concave"; TANGENT: Corner = "tangent"

ANGLE_TOL = 1e-6
SMOOTH_TOL = 1e-7
FULL_TURN = 2 * math.pi

class Box(NamedTuple):
    xmin: float; ymin: float; zmin: float
    xmax: float; ymax: float; zmax: float

class Size(NamedTuple):
    length: float; width: float; height: float

@dataclass(slots=True)
class Plane:
    index: FaceId
    area: float
    bbox: Box
    normal: Vec3
    origin: Vec3
    @property
    def is_horizontal(self): return abs(abs(self.normal[2]) - 1.0) < ANGLE_TOL
    @property
    def faces_up(self): return self.normal[2] > 1.0 - ANGLE_TOL
    @property
    def is_vertical(self): return abs(self.normal[2]) < ANGLE_TOL
    @property
    def is_tilted(self): return not self.is_horizontal and not self.is_vertical

@dataclass(slots=True)
class Cylinder:
    index: FaceId
    area: float
    bbox: Box
    origin: Vec3
    axis: Vec3
    radius: float
    sweep: float
    @property
    def is_full_circle(self): return abs(self.sweep - FULL_TURN) < 1e-4
    @property
    def is_upright(self): return abs(abs(self.axis[2]) - 1.0) < ANGLE_TOL

@dataclass(slots=True)
class Other:
    index: FaceId
    area: float
    bbox: Box

Face = Plane | Cylinder | Other

@dataclass(slots=True)
class Edge:
    index: int
    faces: tuple[FaceId, FaceId]
    kind: Corner
    length: float

@dataclass(slots=True)
class Part:
    faces: list[Face]
    edges: list[Edge]
    bbox: Box
    path: Path | None = None
    topo: list[TopoDS_Face] | None = None
    @property
    def size(self) -> Size:
        b = self.bbox; return Size(b.xmax - b.xmin, b.ymax - b.ymin, b.zmax - b.zmin)
    def planes(self) -> Iterator[Plane]: return (f for f in self.faces if isinstance(f, Plane))
    def cylinders(self) -> Iterator[Cylinder]: return (f for f in self.faces if isinstance(f, Cylinder))
    def neighbours(self, face_index: FaceId) -> dict[FaceId, Corner]:
        out: dict[FaceId, Corner] = {}
        for e in self.edges:
            a, b = e.faces
            if a == face_index: out[b] = e.kind
            elif b == face_index: out[a] = e.kind
        return out
    def concave_faces(self) -> set[FaceId]: return {f for e in self.edges if e.kind == CONCAVE for f in e.faces}

def read_step(path: str | Path) -> TopoDS_Shape:
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != 1: raise OSError(f"could not read STEP file: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull(): raise ValueError(f"STEP file contained no shape: {path}")
    return shape

def _box(shape: TopoDS_Shape) -> Box:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    lo, hi = box.CornerMin(), box.CornerMax()
    return Box(lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z())

def _xyz(d: gp_Dir | gp_Pnt) -> Vec3: return (d.X(), d.Y(), d.Z())

def _outward_normal(face: TopoDS_Face, point: gp_Pnt) -> gp_Vec | None:
    proj = GeomAPI_ProjectPointOnSurf(point, BRep_Tool.Surface_s(face))
    if proj.NbPoints() == 0: return None
    u, v = proj.LowerDistanceParameters()
    props = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, 1e-9)
    if not props.IsNormalDefined(): return None
    n = gp_Vec(props.Normal())
    if face.Orientation() == TopAbs_REVERSED: n.Reverse()
    return n

def _same_edge_in_face(face: TopoDS_Face, edge: TopoDS_Edge) -> TopoDS_Edge | None:
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        candidate = TopoDS.Edge_s(exp.Current())
        if candidate.IsSame(edge): return candidate
        exp.Next()
    return None

def _describe_face(face: TopoDS_Face, index: FaceId) -> Face:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    area, bbox = props.Mass(), _box(face)
    surf = BRepAdaptor_Surface(face)
    kind = surf.GetType()
    if kind == GeomAbs_Plane:
        plane = surf.Plane()
        n = gp_Vec(plane.Axis().Direction())
        if face.Orientation() == TopAbs_REVERSED: n.Reverse()
        return Plane(index, area, bbox, normal=_xyz(n), origin=_xyz(plane.Location()))
    if kind == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        umin, umax, _, _ = BRepTools.UVBounds_s(face)
        return Cylinder(index, area, bbox, origin=_xyz(cyl.Location()),
                        axis=_xyz(cyl.Axis().Direction()), radius=cyl.Radius(),
                        sweep=abs(umax - umin))
    return Other(index, area, bbox)

def _classify_edge(edge: TopoDS_Edge, face_a: TopoDS_Face, face_b: TopoDS_Face) -> tuple[Corner, float]:
    curve = BRepAdaptor_Curve(edge)
    point, tangent = gp_Pnt(), gp_Vec()
    curve.D1(0.5 * (curve.FirstParameter() + curve.LastParameter()), point, tangent)
    length = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, length)
    n_a, n_b = _outward_normal(face_a, point), _outward_normal(face_b, point)
    if n_a is None or n_b is None: return TANGENT, length.Mass()
    cross = n_a.Crossed(n_b)
    if cross.Magnitude() < SMOOTH_TOL: return TANGENT, length.Mass()
    oriented = _same_edge_in_face(face_a, edge)
    if oriented is not None and oriented.Orientation() == TopAbs_REVERSED: tangent.Reverse()
    return (CONVEX if cross.Dot(tangent) > 0 else CONCAVE), length.Mass()

def outline(part: Part, face_id: FaceId, deflection: float = 0.05) -> list[Vec3]:
    """Ordered 3D polygon of a face's outer wire (last point != first)."""
    face = part.topo[face_id]
    wire = BRepTools.OuterWire_s(face)
    points: list[Vec3] = []
    exp = BRepTools_WireExplorer(wire, face)
    while exp.More():
        edge = exp.Current()
        curve = BRepAdaptor_Curve(edge)
        sampler = GCPnts_TangentialDeflection(curve, deflection, 0.05)
        pts = [_xyz(sampler.Value(i)) for i in range(1, sampler.NbPoints() + 1)]
        if edge.Orientation() == TopAbs_REVERSED: pts.reverse()
        points.extend(pts[:-1])
        exp.Next()
    return points

def describe(shape: TopoDS_Shape, path: str | Path | None = None) -> Part:
    faces: list[Face] = []
    topo: list[TopoDS_Face] = []
    face_ids: dict[int, FaceId] = {}
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        key = face.TShape()
        if key not in face_ids:
            face_ids[key] = len(faces)
            faces.append(_describe_face(face, len(faces)))
            topo.append(face)
        exp.Next()
    shared = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, shared)
    edges: list[Edge] = []
    for i in range(1, shared.Extent() + 1):
        edge = TopoDS.Edge_s(shared.FindKey(i))
        touching = [TopoDS.Face_s(f) for f in shared.FindFromIndex(i)]
        if len(touching) != 2: continue
        ids = (face_ids[touching[0].TShape()], face_ids[touching[1].TShape()])
        if ids[0] == ids[1]: continue
        kind, length = _classify_edge(edge, touching[0], touching[1])
        edges.append(Edge(index=len(edges), faces=ids, kind=kind, length=length))
    return Part(faces=faces, edges=edges, bbox=_box(shape), path=Path(path) if path else None, topo=topo)

def load(path: str | Path) -> Part: return describe(read_step(path), path)
