from __future__ import annotations

import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_easy import load_features, load_gt

from machineplan import plan as planmod

MILLS = {"AREA_MILL", "FLOOR_WALL", "HOLE_MILLING"}
GT, FEATS = None, None

def init():
    global GT, FEATS
    GT, FEATS = load_gt(), load_features()

def scan(part: str) -> tuple | None:
    true, row = GT[part], FEATS[part]
    if not row.get("holes") or not (row.get("pockets") or row.get("chamfers")): return None
    y = next(o2 not in MILLS for _, o2 in true)
    ops = planmod.plan_from_row(row, drilling_first=True)
    pred_hm = any(o.o2 == "HOLE_MILLING" for o in ops)
    gt_hm = any(o2 == "HOLE_MILLING" for _, o2 in true)
    pred_blind = any(not h["through"] for h in row["holes"])
    return part, y, pred_hm, gt_hm, pred_blind

def main() -> int:
    gt = load_gt()
    parts = sorted(gt)
    agree = tot = 0
    law = Counter()
    with Pool(8, initializer=init) as pool:
        for r in pool.imap_unordered(scan, parts, chunksize=50):
            if r is None: continue
            part, y, pred_hm, gt_hm, pred_blind = r
            tot += 1
            agree += pred_hm == gt_hm
            if not pred_blind: law[("law1_millfirst", y is False)] += 1
            elif not pred_hm: law[("law2_drillfirst", y is True)] += 1
            else: law[("subset", y)] += 1
    print(f"HOLE_MILLING presence agreement: {agree/tot:.4f} (n={tot})")
    for k, v in sorted(law.items()): print(k, v)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
