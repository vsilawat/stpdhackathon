from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from machineplan import plan as planmod

MILL = ("CH", "PK")
TW = ("SPOT", "TW", "DEEP")

def cat(name: str) -> str:
    b = name.rstrip("0123456789").rstrip("_")
    if b == "AREA_MILL": return "CH"
    if b == "SPOT_DRILL": return "SPOT"
    if b == "INDEXABLE_INSERT_DRILL_BLIND_HOLE_FROM_SOLID": return "INSB"
    if b == "INDEXABLE_INSERT_DRILL_THROUGH_HOLE_FROM_SOLID": return "INST"
    if b == "BORE_BLIND_HOLE": return "BORE"
    if b in ("MILL_BLIND_HOLE_FROM_SOLID_MATERIAL", "MILL_THROUGH_HOLE_FROM_SOLID_MATERIAL",
             "MILL_ROUGH_BLIND_HOLE_CONTOUR", "MILL_FINISH_BLIND_HOLE_FLAT_BOTTOM"): return "HM"
    if b.startswith("MILL_"): return "PK"
    if "GUN" in b or "SPADE" in b: return "DEEP"
    if b == "DRILL_TO_ENLARGE_BLIND_HOLE": return "ENLB"
    return "TW"

# labels: TW block before mill block; and for TW-before parts, HM before mills
def labels() -> tuple[dict[str, bool], dict[str, bool]]:
    seqs = defaultdict(list)
    for r in csv.DictReader(open("derived/opdetails.csv")):
        seqs[r["part"]].append((int(r["op_index"]), cat(r["op_name"])))
    tw, hm = {}, {}
    for part, ops in seqs.items():
        ops.sort()
        cats = [c for _, c in ops]
        mi = [i for i, c in enumerate(cats) if c in MILL]
        ti = [i for i, c in enumerate(cats) if c in TW]
        if not mi or not ti: continue
        if max(ti) < min(mi): tw[part] = True
        elif min(ti) > max(mi): tw[part] = False
        hi = [i for i, c in enumerate(cats) if c == "HM"]
        if tw.get(part) and hi:
            if max(hi) < min(mi): hm[part] = True
            elif min(hi) > max(mi): hm[part] = False
    return tw, hm

FEATS: dict | None = None
def init():
    global FEATS
    FEATS = {r["part"]: r for r in map(json.loads, open("derived/features.jsonl"))}

def feat_row(part: str) -> tuple[str, list[float]] | None:
    row = FEATS.get(part)
    if not row or "error" in row: return None
    return part, planmod.order_features(planmod.features_from_row(row))

def train(name: str, y_by_part: dict[str, bool], rows: dict[str, list[float]], all_parts: list[str]):
    keep = [p for p in sorted(y_by_part) if p in rows]
    X = np.array([rows[p] for p in keep])
    y = np.array([y_by_part[p] for p in keep])
    idx = {p: i for i, p in enumerate(all_parts)}
    test = np.array([idx[p] % 5 == 4 for p in keep])
    m = VotingClassifier([
        ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)),
        ("hgb", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
                                               l2_regularization=1.0, random_state=0))], voting="soft")
    m.fit(X[~test], y[~test])
    print(f"{name}: n={len(keep)} rate={y.mean():.3f} holdout acc: {m.score(X[test], y[test]):.4f} (n={test.sum()})")
    m.fit(X, y)
    pickle.dump(m, open(f"derived/{name}.pkl", "wb"))

def main() -> int:
    tw, hm = labels()
    parts = sorted(set(tw) | set(hm))
    rows = {}
    with Pool(8, initializer=init) as pool:
        for r in pool.imap(feat_row, parts, chunksize=50):
            if r: rows[r[0]] = r[1]
    all_parts = sorted({r["part"] for r in map(json.loads, open("derived/features.jsonl"))})
    train("order_model", tw, rows, all_parts)
    train("hm_model", hm, rows, all_parts)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
