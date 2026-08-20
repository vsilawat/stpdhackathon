from __future__ import annotations

import csv
import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

KINDS = ["corner", "center", "slot", "edge"]

def row_x(p: dict, stock) -> list[float]:
    return [KINDS.index(p["kind"]) if p["kind"] in KINDS else -1, p["fillet_radius"] or 0.0,
            p["depth"], p["area"], p["area"] / max(p["depth"], 0.1), float(p["open_sides"] or 0),
            float(p.get("corners") or 0), stock[0], stock[1], stock[2]]

def main() -> int:
    feats = {r["part"]: r for r in map(json.loads, open("derived/features.jsonl"))}
    X, y, parts = [], [], []
    for r in csv.DictReader(open("derived/chains.csv")):
        if not r["group_prefix"].startswith("STEP1POCKET"): continue
        row = feats.get(r["part"])
        if not row or len(row["pockets"]) != 1: continue
        dias = [float(v) for v in r["tool_diams"].split(">") if v]
        if not dias: continue
        X.append(row_x(row["pockets"][0], row["stock"]))
        y.append(f"{dias[-1]:.2f}"); parts.append(r["part"])
    X, y = np.array(X), np.array(y)
    idx = {p: i for i, p in enumerate(sorted(set(parts)))}
    test = np.array([idx[p] % 5 == 4 for p in parts])
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
    m.fit(X[~test], y[~test])
    pred = m.predict(X[test]).astype(float)
    yt = y[test].astype(float)
    rel = np.abs(pred - yt) / yt
    base = np.abs(2 * X[test, 1] - yt) / yt
    print(f"n={len(y)} exact={np.mean(pred == yt):.3f} within2%={np.mean(rel <= 0.02):.3f} "
          f"(2xfillet baseline {np.mean(base <= 0.02):.3f})")
    pickle.dump(m, open("derived/pocket_tool_model.pkl", "wb"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
