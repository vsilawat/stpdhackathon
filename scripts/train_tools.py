from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def base(n: str) -> str: return n.rsplit("_", 1)[0] if n.split("_")[-1].isdigit() else n

def load():
    groups = defaultdict(list)
    for r in csv.DictReader(open("derived/chains.csv")):
        if r["group_prefix"].startswith(("STEP1HOLE", "STEP1POCKET")):
            groups[r["part"]].append((float(r["final_diameter"]), r["chain"], r["tool_diams"]))
    feats = {r["part"]: r for r in map(json.loads, open("derived/features.jsonl"))}
    rows = []
    for p, items in groups.items():
        row = feats[p]
        holes = list(row["holes"])
        sx, sy, sz = row["stock"]
        for fd, chain, diams in items:
            if not holes: break
            i = min(range(len(holes)), key=lambda k: abs(holes[k]["d"] - fd))
            h = holes.pop(i)
            if abs(h["d"] - fd) > 0.6: continue
            names = chain.split(">")
            dias = [float(v) for v in diams.split(">")]
            ctx = (sz, row["top_z"] - h["mouth_z"], h["bottom_z"], h["d"] % 1.0,
                   float(abs(h["d"] - round(h["d"])) < 0.01), min(h["x"], sx - h["x"], h["y"], sy - h["y"]))
            for j, (n, dv) in enumerate(zip(names, dias, strict=True)):
                rows.append((p, base(n), j, len(names), h["d"], h["depth"], float(h["through"]), dv, ctx))
    return rows

def main() -> int:
    rows = load()
    idx = {p: i for i, p in enumerate(sorted({r[0] for r in rows}))}
    by_op = defaultdict(list)
    for r in rows: by_op[r[1]].append(r)
    models = {}
    for op, rs in sorted(by_op.items(), key=lambda kv: -len(kv[1])):
        X = np.array([[r[4], r[5], r[5] / r[4], r[6], r[2], r[3], *r[8]] for r in rs])
        y = np.array([f"{r[7]:.2f}" for r in rs])
        test = np.array([idx[r[0]] % 5 == 4 for r in rs])
        yf = np.array([float(v) for v in y]); n_final = np.mean(np.abs(yf - X[:, 0]) < 0.011)
        if len(set(y[~test])) == 1:
            models[op] = ("const", y[0])
            print(f"{op:48s} n={len(rs):5d} CONST {y[0]}")
            continue
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=-1)
        m.fit(X[~test], y[~test])
        pred = m.predict(X[test]).astype(float)
        yt = y[test].astype(float)
        acc = float(np.mean(pred == yt))
        rel = np.abs(pred - yt) / yt
        within2 = float(np.mean(rel <= 0.02))
        final_acc = float(np.mean(np.abs(X[test, 0] - yt) / yt <= 0.02))
        models[op] = ("rf", m)
        print(f"{op:48s} n={len(rs):5d} exact={acc:.4f} within2%={within2:.4f} (final-d baseline {final_acc:.4f}, final%={n_final:.2f})")
    pickle.dump(models, open("derived/tool_dia_models.pkl", "wb"))
    print("saved derived/tool_dia_models.pkl")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
