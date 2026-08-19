from __future__ import annotations

import manifold3d as m3d
import numpy as np

from . import brep
from .brep import Part
from .features import Chamfer, Pocket

SEGMENTS = 64
EPS = 0.05

def _area2(poly: list[tuple[float, float]]) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True))

def prism(poly_xy: list[tuple[float, float]], z0: float, z1: float) -> m3d.Manifold:
    pts = poly_xy if _area2(poly_xy) > 0 else poly_xy[::-1]
    section = m3d.CrossSection([pts])
    return m3d.Manifold.extrude(section, z1 - z0).translate([0.0, 0.0, z0])

def hull_points(points: np.ndarray) -> m3d.Manifold: return m3d.Manifold.hull_points(points.tolist())

def pocket_solid(part: Part, pocket: Pocket, top_z: float) -> m3d.Manifold:
    floor = pocket.faces[0]
    poly = [(x, y) for x, y, _ in brep.outline(part, floor)]
    return prism(poly, pocket.floor_z, top_z + EPS)

def chamfer_solid(part: Part, chamfer: Chamfer, top_z: float) -> m3d.Manifold:
    wedges = []
    for fid in chamfer.faces:
        pts = np.array(brep.outline(part, fid))
        lifted = pts.copy(); lifted[:, 2] = top_z + EPS
        wedges.append(hull_points(np.vstack([pts, lifted])))
    return m3d.Manifold.batch_boolean(wedges, m3d.OpType.Add)

def hole_solid(hole_x: float, hole_y: float, radius: float, z_top: float, z_bottom: float,
               tip_angle_deg: float | None = None) -> m3d.Manifold:
    body = (m3d.Manifold.cylinder(z_top + EPS - z_bottom, radius, radius, SEGMENTS)
            .translate([hole_x, hole_y, z_bottom]))
    if tip_angle_deg is not None:
        tip_depth = radius / np.tan(np.radians(tip_angle_deg / 2))
        tip = (m3d.Manifold.cylinder(tip_depth, radius, 0.0, SEGMENTS)
               .translate([hole_x, hole_y, z_bottom - tip_depth]))
        body += tip
    return body

def spot_solid(x: float, y: float, tip_radius: float, z_top: float, depth: float,
               shank_radius: float | None = None) -> m3d.Manifold:
    """Spot drill plunge: cone from surface radius r=depth*tan(half) — calibrated against GT."""
    r_at_top = depth * np.tan(np.radians(45.0)) if shank_radius is None else shank_radius
    return (m3d.Manifold.cylinder(depth + EPS, r_at_top + EPS * np.tan(np.radians(45.0)), 0.0, SEGMENTS)
            .translate([x, y, z_top - depth]))
