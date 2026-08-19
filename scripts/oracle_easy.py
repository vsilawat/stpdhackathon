from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_easy import levenshtein, load_features, load_gt

from machineplan import plan as planmod

GT, FEATS = None, None

def init():
    global GT, FEATS
    GT, FEATS = load_gt(), load_features()

def scan(part: str) -> tuple[float, float, bool]:
    true, row = GT[part], FEATS[part]
    levs = {}
    for df in (True, False):
        pred = [(o.o1, o.o2) for o in planmod.plan_from_row(row, drilling_first=df)]
        levs[df] = levenshtein(pred, true) / max(len(pred), len(true), 1)
    cur = planmod.predict_drilling_first(planmod.features_from_row(row))
    return levs[cur], min(levs.values()), levs[not cur] < levs[cur] - 1e-9

def main() -> int:
    gt = load_gt()
    parts = sorted(gt)
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4]
    cur = ora = 0.0
    helps = 0
    with Pool(8, initializer=init) as pool:
        for c, o, h in pool.imap_unordered(scan, holdout, chunksize=25):
            cur += c; ora += o; helps += h
    n = len(holdout)
    print(f"current={cur/n:.4f}  oracle-binary={ora/n:.4f}  flip-helps={helps}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
