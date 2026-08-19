from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

INSERT_GRID = [x / 2 for x in range(30, 61)] + [40.0, 42.0, 50.0]
def grid_dist(d: float) -> float: return min(abs(d - g) for g in INSERT_GRID)

def load():
    groups = defaultdict(list)
    for r in csv.DictReader(open("derived/chains.csv")):
        if r["group_prefix"].startswith(("STEP1HOLE", "STEP1POCKET")):
            groups[r["part"]].append((float(r["final_diameter"]), r["chain"]))
    feats = {r["part"]: r for r in map(json.loads, open("derived/features.jsonl"))}
    X, y, parts = [], [], []
    for p, rows in groups.items():
        row = feats[p]
        holes = list(row["holes"])
        sx, sy, sz = row["stock"]
        for fd, chain in rows:
            if not holes: break
            i = min(range(len(holes)), key=lambda k: abs(holes[k]["d"] - fd))
            h = holes.pop(i)
            if abs(h["d"] - fd) > 0.6: continue
            X.append([h["d"], h["depth"], h["depth"] / h["d"], float(h["through"]),
                      sz, row["top_z"] - h["mouth_z"], h["bottom_z"],
                      h["d"] % 1.0, float(abs(h["d"] - round(h["d"])) < 0.01),
                      min(h["x"], sx - h["x"], h["y"], sy - h["y"]),
                      grid_dist(h["d"]), float(grid_dist(h["d"]) < 0.05)])
            y.append(chain); parts.append(p)
    return np.array(X), np.array(y), parts

def main() -> int:
    X, y, parts = load()
    idx = {p: i for i, p in enumerate(sorted(set(parts)))}
    test = np.array([idx[p] % 5 == 4 for p in parts])
    best = None
    models = [DecisionTreeClassifier(max_depth=d, min_samples_leaf=3, random_state=0) for d in (12, 16)]
    models.append(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1))
    for m in models:
        m.fit(X[~test], y[~test])
        acc = m.score(X[test], y[test])
        print(f"{type(m).__name__}: holdout chain acc {acc:.4f}")
        if best is None or acc > best[1]: best = (m, acc)
    pickle.dump(best[0], open("derived/chain_tree12.pkl", "wb"))
    print(f"saved derived/chain_tree12.pkl (n={len(y)}, acc={best[1]:.4f})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
