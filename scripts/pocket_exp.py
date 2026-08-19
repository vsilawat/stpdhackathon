from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from machineplan.tooling import pocket_dia_X
from train_pocket_models import pocket_of

sys.path.insert(0, str(Path(__file__).resolve().parent))

def within2(pred, gt):
    return (np.abs(pred - gt) / gt <= 0.02).mean()

def main() -> int:
    allparts = sorted({r["part"] for r in csv.DictReader(open("derived/opdetails.csv"))})
    hold = {p for i, p in enumerate(allparts) if i % 5 == 4}
    rows = list(csv.DictReader(open("derived/pocket_train.csv")))
    byp = defaultdict(list)
    for r in rows: byp[r["part"]].append(r)
    X, y, parts = [], [], []
    for p, prs in byp.items():
        pks = [pocket_of(r) for r in prs]
        for x, r in zip(pocket_dia_X(pks), prs, strict=True):
            X.append(x); y.append(float(r["gt_dia"])); parts.append(p)
    X, y = np.array(X), np.array(y)
    mask = np.array([p in hold for p in parts])
    Xtr, Xte, ytr, yte = X[~mask], X[mask], y[~mask], y[mask]

    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    rf = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xtr, ytr.astype(str))
    print(f"rf baseline: {within2(rf.predict(Xte).astype(float), yte):.4f}", flush=True)

    hg = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, max_leaf_nodes=63,
                                        l2_regularization=1.0, random_state=0, early_stopping=False).fit(Xtr, ytr.astype(str))
    print(f"hgb: {within2(hg.predict(Xte).astype(float), yte):.4f}", flush=True)

    # merge classes whose values pass each other's 2% test; predict the group's modal dia
    vals = sorted(set(ytr) | set(yte))
    group = {}
    for v in vals:
        for g in group.values():
            if abs(v - g) / min(v, g) <= 0.02: group[v] = g; break
        else: group[v] = v
    mode = {}
    cnt = Counter(ytr)
    for v in vals:
        g = group[v]
        if g not in mode or cnt[v] > cnt[mode[g]]: mode[g] = v
    ytr_g = np.array([group[v] for v in ytr]).astype(str)
    rf2 = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xtr, ytr_g)
    hg2 = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, max_leaf_nodes=63,
                                         l2_regularization=1.0, random_state=0, early_stopping=False).fit(Xtr, ytr_g)
    pr = np.array([mode[float(v)] for v in rf2.predict(Xte)])
    print(f"rf merged: {within2(pr, yte):.4f}", flush=True)
    ph = np.array([mode[float(v)] for v in hg2.predict(Xte)])
    print(f"hgb merged: {within2(ph, yte):.4f}", flush=True)

    cls = rf2.classes_.astype(float)
    proba = rf2.predict_proba(Xte) + hg2.predict_proba(Xte)
    pe = np.array([mode[cls[i]] for i in proba.argmax(1)])
    print(f"ens merged: {within2(pe, yte):.4f}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
