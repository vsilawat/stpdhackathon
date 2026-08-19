from __future__ import annotations

import math

import pytest

from machineplan import brep, testparts


@pytest.fixture(scope="module")
def part(): return brep.describe(testparts.block_with_features())

@pytest.fixture(scope="module")
def blind_part(): return brep.describe(testparts.block_with_features(pocket=None, chamfer=0.0, hole_depth=25.0))

def _up_faces(part): return sorted((f for f in part.planes() if f.faces_up), key=lambda f: f.bbox.zmin)

def test_block_size_is_recovered(part):
    assert part.size == pytest.approx((200.0, 200.0, 60.0), abs=1e-6)

def test_size_fields_are_named(part):
    assert part.size.length == pytest.approx(200.0)
    assert part.size.height == pytest.approx(60.0)

def test_through_hole_is_one_full_cylinder(part):
    full = [f for f in part.cylinders() if f.is_full_circle]
    assert len(full) == 1
    assert full[0].radius == pytest.approx(10.0)
    assert (full[0].bbox.zmin, full[0].bbox.zmax) == pytest.approx((0.0, 60.0), abs=1e-6)

def test_pocket_corners_are_quarter_cylinders(part):
    blends = [f for f in part.cylinders() if not f.is_full_circle]
    assert len(blends) == 4
    assert all(f.radius == pytest.approx(8.0) for f in blends)
    assert all(math.degrees(f.sweep) == pytest.approx(90.0, abs=1e-6) for f in blends)
    assert all((f.bbox.zmin, f.bbox.zmax) == pytest.approx((40.0, 60.0), abs=1e-6) for f in blends)

def test_chamfer_is_the_only_tilted_face(part):
    tilted = [f for f in part.planes() if f.is_tilted]
    assert len(tilted) == 1
    assert abs(tilted[0].normal[2]) == pytest.approx(math.sqrt(0.5), abs=1e-6)

def test_pocket_floor_sits_below_the_top_face(part):
    floor, top = _up_faces(part)
    assert floor.bbox.zmin == pytest.approx(40.0)
    assert top.bbox.zmin == pytest.approx(60.0)

def test_pocket_floor_meets_its_walls_concavely(part):
    floor = _up_faces(part)[0]
    assert set(part.neighbours(floor.index).values()) == {brep.CONCAVE}

def test_outer_block_corners_are_convex(part):
    outer = {f.index for f in part.planes() if f.is_vertical and f.area > 5000}
    between = [e for e in part.edges if e.faces[0] in outer and e.faces[1] in outer]
    assert len(between) == 4
    assert all(e.kind == brep.CONVEX for e in between)

def test_through_hole_has_no_concave_edges(part):
    hole = next(f for f in part.cylinders() if f.is_full_circle)
    assert set(part.neighbours(hole.index).values()) == {brep.CONVEX}

def test_blind_hole_has_a_concave_floor(blind_part):
    hole = next(f for f in blind_part.cylinders() if f.is_full_circle)
    assert hole.bbox.zmin == pytest.approx(35.0)
    assert brep.CONCAVE in set(blind_part.neighbours(hole.index).values())

def test_wall_to_blend_joins_are_smooth(part):
    assert len([e for e in part.edges if e.kind == brep.TANGENT]) == 8

def test_face_bbox_covers_all_three_axes():
    plain = brep.describe(testparts.block_with_pocket("center"))
    top = _up_faces(plain)[-1]
    assert (top.bbox.xmin, top.bbox.ymin) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert (top.bbox.xmax, top.bbox.ymax) == pytest.approx((200.0, 200.0), abs=1e-6)

def test_chamfer_trims_the_top_face(part):
    assert _up_faces(part)[-1].bbox.xmin == pytest.approx(6.0, abs=1e-6)

def test_faces_split_into_planes_and_cylinders(part):
    assert len(list(part.planes())) + len(list(part.cylinders())) == len(part.faces)
    assert all(isinstance(f, brep.Plane) for f in part.planes())
    assert all(isinstance(f, brep.Cylinder) for f in part.cylinders())
