from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from machineplan import plan as planmod

GT, FEATS = None, None

def load_gt():
    gt = defaultdict(list)
    for r in csv.DictReader(open("derived/opdetails.csv")):
        gt[r["part"]].append((int(r["op_index"]), r["op_name"], float(r["tool_diameter_mm"])))
    return gt

def init():
    global GT, FEATS
    GT = load_gt()
    FEATS = {r["part"]: r for r in map(json.loads, open("derived/features.jsonl")) if "error" not in r}

def scan(part: str):
    row = FEATS.get(part)
    if not row: return None
    ops = planmod.plan_from_row(row)
    g = sorted(GT[part])
    if len(ops) != len(g): return None
    out = []
    for k in range(len(g)):
        _, gname, gdia = g[k]
        base = gname.rstrip("0123456789").rstrip("_")
        mdia = round(ops[k].tool_diameter or 10.0, 2)
        ok = abs(mdia - gdia) / gdia <= 0.02
        out.append((base, ok, gdia, mdia))
    return out

def main() -> int:
    gt = load_gt()
    parts = sorted(gt)
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4]
    tot, bad = Counter(), Counter()
    ex = defaultdict(Counter)
    with Pool(8, initializer=init) as pool:
        for res in pool.imap_unordered(scan, holdout, chunksize=25):
            if not res: continue
            for base, ok, gdia, mdia in res:
                tot[base] += 1
                if not ok:
                    bad[base] += 1
                    ex[base][(gdia, mdia)] += 1
    for base, n in tot.most_common():
        b = bad[base]
        print(f"{base:50s} n={n:5d} bad={b:4d} acc={1 - b / n:.3f}")
        if b: print("   top errs (gt,mine):", ex[base].most_common(4))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
