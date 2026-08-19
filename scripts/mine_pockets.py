from __future__ import annotations

import csv
import multiprocessing as mp
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA = Path("data/MachinePlan-10K")
GT = Path("data/MachinePlan-10K-gt")
POCKET_NAMES = {"MILL_RECTANGULAR_POCKET", "MILL_SLOT", "MILL_RECTANGULAR_SLOT",
                "MILL_CORNER_NOTCH_RECTANGULAR", "MILL_OPEN_POCKET"}
XY = re.compile(r"X(-?[\d.]+) Y(-?[\d.]+)")
_GTOPS: dict | None = None

def gt_ops() -> dict:
    global _GTOPS
    if _GTOPS is None:
        _GTOPS = {}
        for r in csv.DictReader(open("derived/opdetails.csv")):
            base = re.sub(r"_\d+$", "", r["op_name"])
            if base in POCKET_NAMES:
                _GTOPS.setdefault(r["part"], []).append(
                    (int(r["op_index"]), base, float(r["tool_diameter_mm"])))
    return _GTOPS

def max_inscribed(poly) -> float:
    lo, hi = 0.0, 0.5 * max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])
    for _ in range(24):
        mid = (lo + hi) / 2
        if poly.buffer(-mid).is_empty: hi = mid
        else: lo = mid
    return 2 * lo

def do_part(part_id: str) -> list | str:
    from shapely.geometry import Point, Polygon
    from machineplan import brep, features
    ops = gt_ops().get(part_id) or []
    if not ops: return []
    try:
        part = brep.load(str(DATA / part_id / f"{part_id}.stp"))
        found = features.extract(part)
        polys = []
        for k, p in enumerate(found.pockets):
            try: polys.append((k, p, Polygon([(x, y) for x, y, _ in brep.outline(part, p.faces[0])])))
            except Exception: polys.append((k, p, None))
        rows = []
        for op_index, op_name, gd in sorted(ops):
            fs = list((GT / part_id).glob(f"{op_index:03d}_*.ptp"))
            if not fs: continue
            pts = XY.findall(fs[0].read_text(errors="ignore"))
            if not pts: continue
            mx = sum(float(x) for x, _ in pts) / len(pts)
            my = sum(float(y) for _, y in pts) / len(pts)
            pt = Point(mx, my)
            hit = [(k, p) for k, p, poly in polys if poly is not None and poly.buffer(1.0).contains(pt)]
            if len(hit) != 1:
                cand = [(poly.exterior.distance(pt) if poly else 1e9, k, p) for k, p, poly in polys]
                if not cand: continue
                cand.sort(key=lambda t: t[0])
                k, p = cand[0][1], cand[0][2]
                by = "near"
            else:
                k, p = hit[0]
                by = "poly"
            poly = next(pl for kk, _, pl in polys if kk == k)
            mi = round(max_inscribed(poly), 3) if poly is not None else 0.0
            hull = round(p.area / poly.convex_hull.area, 4) if poly is not None and poly.convex_hull.area > 0 else 1.0
            rows.append([part_id, op_index, op_name, gd, p.kind, p.open_sides, p.fillet_radius or 0.0,
                         round(p.depth, 3), round(p.area, 2), round(p.w, 3), round(p.l, 3), k, by, mi, hull])
        return rows
    except Exception:
        return f"ERR {part_id} {traceback.format_exc(limit=1)}"

def main() -> int:
    parts = sorted(gt_ops())
    nerr = 0
    with open("derived/pocket_train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "op_index", "op_name", "gt_dia", "kind", "open_sides", "fillet_radius",
                    "depth", "area", "w", "l", "det_idx", "matched_by", "max_inscribed", "hull_ratio"])
        with mp.Pool(processes=max(mp.cpu_count() - 2, 2)) as pool:
            for i, res in enumerate(pool.imap_unordered(do_part, parts, chunksize=8)):
                if isinstance(res, str): nerr += 1; print(res, flush=True)
                else: w.writerows(res)
                if (i + 1) % 1000 == 0: print(f"[{i + 1}/{len(parts)}] errs={nerr}", flush=True)
    print(f"done, {nerr} errors", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
