from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from machineplan import brep, dataset, features

GT_OPS: dict[str, Counter] = defaultdict(Counter)

def load_gt_ops() -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    with open("derived/opdetails.csv") as fh:
        for r in csv.DictReader(fh):
            base = r["op_name"].rstrip("0123456789").rstrip("_")
            out[r["part"]][base] += 1
    return out

def scan(part_dir: Path) -> tuple | None:
    try:
        found = features.extract(brep.load(dataset.step_file(part_dir)))
    except Exception:
        return None
    blinds = [h for h in found.holes if not h.through]
    return part_dir.name, sum(h.flat for h in blinds), sum(not h.flat for h in blinds)

def main() -> int:
    gt = load_gt_ops()
    parts = [p for p in dataset.part_dirs() if p.name in gt][:: max(1, len(gt) // 800)]
    print(f"scanning {len(parts)} parts...", flush=True)
    ct = Counter()
    mism = []
    with Pool(8) as pool:
        for r in pool.imap_unordered(scan, parts, chunksize=10):
            if r is None: continue
            part, n_flat, n_cone = r
            ops = gt[part]
            n_mb = ops.get("MILL_BLIND_HOLE_FROM_SOLID_MATERIAL", 0)
            n_rc = ops.get("MILL_ROUGH_BLIND_HOLE_CONTOUR", 0)
            n_db = ops.get("DRILL_BLIND_HOLE_INTO_CENTER", 0) + ops.get("INDEXABLE_INSERT_DRILL_BLIND_HOLE_FROM_SOLID", 0)
            ct[("flat==mb+rc", n_flat == n_mb + n_rc)] += 1
            ct[("flat==mb", n_flat == n_mb)] += 1
            if n_flat != n_mb + n_rc and len(mism) < 15:
                mism.append((part, n_flat, n_cone, dict(ops)))
    for k, v in sorted(ct.items()): print(k, v)
    for m in mism: print(m)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
