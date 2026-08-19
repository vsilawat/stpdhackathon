from __future__ import annotations

from pathlib import Path

from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.GeomAbs import GeomAbs_Line
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Shape

from .features import PocketKind

# Negative bounds hang outside the block so that side of the pocket opens out.
POCKET_FOOTPRINTS: dict[PocketKind, tuple[float, float, float, float]] = {
    "center": (60.0, 60.0, 140.0, 140.0),
    "edge": (-10.0, 60.0, 60.0, 140.0),
    "corner": (-10.0, -10.0, 60.0, 60.0),
    "slot": (80.0, -10.0, 120.0, 210.0),
}

def _edges(shape: TopoDS_Shape, vertical: bool) -> list[TopoDS_Edge]:
    out, seen = [], set()
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        key = edge.TShape()
        if key not in seen:
            seen.add(key)
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() == GeomAbs_Line and (abs(abs(curve.Line().Direction().Z()) - 1.0) < 1e-9) == vertical:
                out.append(edge)
        exp.Next()
    return out

def _top_edges(shape: TopoDS_Shape, height: float) -> list[TopoDS_Edge]:
    out = []
    for edge in _edges(shape, vertical=False):
        curve = BRepAdaptor_Curve(edge)
        p0, p1 = curve.Value(curve.FirstParameter()), curve.Value(curve.LastParameter())
        if abs(p0.Z() - height) < 1e-9 and abs(p1.Z() - height) < 1e-9: out.append(edge)
    return out

def _fillet_verticals(tool: TopoDS_Shape, radius: float) -> TopoDS_Shape:
    rounder = BRepFilletAPI_MakeFillet(tool)
    for edge in _edges(tool, vertical=True): rounder.Add(radius, edge)
    return rounder.Shape()

def block_with_features(length=200.0, width=200.0, height=60.0, pocket=(60.0, 50.0, 40.0, 20.0),
                        fillet=8.0, hole=(150.0, 150.0, 10.0), chamfer=6.0,
                        hole_depth=None) -> TopoDS_Shape:
    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), length, width, height).Shape()
    if chamfer:
        maker = BRepFilletAPI_MakeChamfer(shape)
        maker.Add(chamfer, _top_edges(shape, height)[0])
        shape = maker.Shape()
    if pocket:
        px, py, size, depth = pocket
        tool = BRepPrimAPI_MakeBox(gp_Pnt(px, py, height - depth), size, size, depth + 1.0).Shape()
        if fillet: tool = _fillet_verticals(tool, fillet)
        shape = BRepAlgoAPI_Cut(shape, tool).Shape()
    if hole:
        hx, hy, radius = hole
        bottom = -1.0 if hole_depth is None else height - hole_depth
        span = (height + 2.0) if hole_depth is None else (hole_depth + 1.0)
        drill = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(hx, hy, bottom), gp_Dir(0, 0, 1)), radius, span).Shape()
        shape = BRepAlgoAPI_Cut(shape, drill).Shape()
    return shape

def block_with_pocket(kind: PocketKind, length=200.0, width=200.0, height=60.0, depth=20.0,
                      fillet=8.0) -> TopoDS_Shape:
    x0, y0, x1, y1 = POCKET_FOOTPRINTS[kind]
    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), length, width, height).Shape()
    tool = BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, height - depth), x1 - x0, y1 - y0, depth + 1.0).Shape()
    if fillet and kind != "slot": tool = _fillet_verticals(tool, fillet)
    return BRepAlgoAPI_Cut(shape, tool).Shape()

def write_step(shape: TopoDS_Shape, path: str | Path) -> Path:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != 1: raise OSError(f"could not write STEP file: {path}")
    return Path(path)
