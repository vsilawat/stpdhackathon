from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_tools import vocab_type

from machineplan import plan as planmod

GT, FEATS = None, None

def load_gt():
    gt = defaultdict(list)
    for r in csv.DictReader(open("derived/opdetails.csv")):
        gt[r["part"]].append((int(r["op_index"]), vocab_type(r["tool_template_subtype"], r["tool_type"]),
                              float(r["tool_diameter_mm"])))
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
    res = [0, 0, 0, 0]  # n, type ok, dia ok, both
    for k in range(min(len(ops), len(g))):
        _, gtyp, gdia = g[k]
        mtyp, mdia = ops[k].tool_type or "end_mill", round(ops[k].tool_diameter or 10.0, 2)
        tm, dm = mtyp == gtyp, abs(mdia - gdia) / gdia <= 0.02
        res[0] += 1; res[1] += tm; res[2] += dm; res[3] += tm and dm
    return res

def main() -> int:
    gt = load_gt()
    parts = sorted(gt)
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4]
    tot = [0, 0, 0, 0]
    with Pool(8, initializer=init) as pool:
        for r in pool.imap_unordered(scan, holdout, chunksize=25):
            if r: tot = [a + b for a, b in zip(tot, r)]
    n = tot[0]
    print(f"ops={n} type={tot[1]/n:.4f} dia={tot[2]/n:.4f} BOTH={tot[3]/n:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
