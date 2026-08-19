from __future__ import annotations

from pathlib import Path

import manifold3d as m3d
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape

from .brep import Box

LINEAR_DEFLECTION = 0.05
ANGULAR_DEFLECTION = 0.2

def tessellate(shape: TopoDS_Shape) -> m3d.Manifold:
    """Weld OCCT face triangulations into one watertight manifold."""
    BRepMesh_IncrementalMesh(shape, LINEAR_DEFLECTION, False, ANGULAR_DEFLECTION, True)
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current()); loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            t = loc.Transformation()
            pts = np.empty((tri.NbNodes(), 3))
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(t); pts[i - 1] = (p.X(), p.Y(), p.Z())
            idx = np.array([[tri.Triangle(i).Value(1), tri.Triangle(i).Value(2), tri.Triangle(i).Value(3)]
                            for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
            if face.Orientation() == TopAbs_REVERSED: idx = idx[:, ::-1]
            chunks.append((pts, idx))
        exp.Next()
    verts = np.vstack([p for p, _ in chunks]).astype(np.float32)
    offset, faces = 0, []
    for pts, idx in chunks: faces.append(idx + offset); offset += len(pts)
    raw = m3d.Mesh(verts, np.vstack(faces).astype(np.uint32))
    raw.merge()
    solid = m3d.Manifold(raw)
    if solid.status() != m3d.Error.NoError: raise ValueError(f"tessellation not manifold: {solid.status()}")
    return solid

def stock_box(bbox: Box) -> m3d.Manifold:
    return (m3d.Manifold.cube([bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin, bbox.zmax - bbox.zmin])
            .translate([bbox.xmin, bbox.ymin, bbox.zmin]))

def iou(a: m3d.Manifold, b: m3d.Manifold) -> float:
    inter = (a ^ b).volume()
    union = a.volume() + b.volume() - inter
    return inter / union if union > 0 else 1.0

def to_trimesh(solid: m3d.Manifold):
    import trimesh
    mesh = solid.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
                           faces=np.asarray(mesh.tri_verts, dtype=np.int64), process=False)

def load_stl(path: str | Path) -> m3d.Manifold:
    import trimesh
    tm = trimesh.load(path, process=False)
    raw = m3d.Mesh(np.asarray(tm.vertices, dtype=np.float32), np.asarray(tm.faces, dtype=np.uint32))
    raw.merge()
    solid = m3d.Manifold(raw)
    if solid.status() != m3d.Error.NoError: raise ValueError(f"stl not manifold: {path}")
    return solid

def save_stl(solid: m3d.Manifold, path: str | Path) -> None: to_trimesh(solid).export(str(path))
