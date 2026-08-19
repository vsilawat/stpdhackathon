from __future__ import annotations

import pytest

from machineplan import brep, features, testparts


@pytest.fixture(scope="module")
def found(): return features.extract(brep.describe(testparts.block_with_features()))

def test_finds_one_of_each(found):
    assert found.counts == {"chamfers": 1, "pockets": 1, "holes": 1}

def test_stock_block_is_recovered(found):
    assert found.stock == pytest.approx((200.0, 200.0, 60.0), abs=1e-6)
    assert (found.bottom_z, found.top_z) == pytest.approx((0.0, 60.0), abs=1e-6)

def test_chamfer_width_and_angle(found):
    assert found.chamfers[0].width == pytest.approx(6.0, abs=1e-6)
    assert found.chamfers[0].angle_deg == pytest.approx(45.0, abs=1e-6)

def test_through_hole_measurements(found):
    hole = found.holes[0]
    assert hole.diameter == pytest.approx(20.0)
    assert (hole.x, hole.y) == pytest.approx((150.0, 150.0))
    assert hole.through and hole.depth == pytest.approx(60.0)

def test_pocket_measurements(found):
    pocket = found.pockets[0]
    assert pocket.depth == pytest.approx(20.0)
    assert pocket.fillet_radius == pytest.approx(8.0)
    assert pocket.corners == 4

def test_blind_hole_is_not_mistaken_for_a_pocket():
    part = brep.describe(testparts.block_with_features(pocket=None, chamfer=0.0, hole_depth=25.0))
    got = features.extract(part)
    assert got.counts == {"chamfers": 0, "pockets": 0, "holes": 1}
    assert not got.holes[0].through
    assert got.holes[0].depth == pytest.approx(25.0)

@pytest.mark.parametrize(("kind", "corners"), [("center", 4), ("edge", 2), ("corner", 1), ("slot", 0)])
def test_pocket_type_from_open_sides(kind, corners):
    got = features.extract(brep.describe(testparts.block_with_pocket(kind)))
    assert len(got.pockets) == 1
    assert got.pockets[0].kind == kind
    assert got.pockets[0].corners == corners
