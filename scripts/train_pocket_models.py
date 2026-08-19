from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from machineplan.features import Pocket
from machineplan.tooling import pocket_dia_feats, pocket_dia_X

def pocket_of(r: dict) -> Pocket:
    return Pocket(floor_z=0.0, depth=float(r["depth"]), area=float(r["area"]), kind=r["kind"],
                  open_sides=int(float(r["open_sides"])), fillet_radius=float(r["fillet_radius"]) or None,
                  corners=0, w=float(r["w"]), l=float(r["l"]),
                  mi=float(r["max_inscribed"]), hull=float(r["hull_ratio"]))

def main() -> int:
    allparts = sorted({r["part"] for r in csv.DictReader(open("derived/opdetails.csv"))})
    hold = {p for i, p in enumerate(allparts) if i % 5 == 4}
    rows = list(csv.DictReader(open("derived/pocket_train.csv")))
    byp = defaultdict(list)
    for r in rows: byp[r["part"]].append(r)
    Xd, Xk, yd, yk, parts = [], [], [], [], []
    for p, prs in byp.items():
        pks = [pocket_of(r) for r in prs]
        for x, xk, r in zip(pocket_dia_X(pks), [pocket_dia_feats(pk) for pk in pks], prs, strict=True):
            Xd.append(x); Xk.append(xk)
            yd.append(f'{float(r["gt_dia"]):g}'); yk.append(r["op_name"]); parts.append(p)
    mask = np.array([p in hold for p in parts])
    Xd, Xk, yd, yk = np.array(Xd), np.array(Xk), np.array(yd), np.array(yk)
    print(f"train={int((~mask).sum())} test={int(mask.sum())}")

    md = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xd[~mask], yd[~mask])
    pred = md.predict(Xd[mask]).astype(float)
    gt = yd[mask].astype(float)
    print(f"dia within2%: {(np.abs(pred - gt) / gt <= 0.02).mean():.4f}")

    mk = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xk[~mask], yk[~mask])
    print(f"kind indep acc: {(mk.predict(Xk[mask]) == yk[mask]).mean():.4f}")

    cls = list(mk.classes_)
    ok = n = 0
    for p, prs in byp.items():
        if p not in hold: continue
        P = np.log(mk.predict_proba([pocket_dia_feats(pocket_of(r)) for r in prs]) + 1e-9)
        if len(prs) <= len(cls):
            best = max(permutations(range(len(cls)), len(prs)),
                       key=lambda pm: sum(P[i][j] for i, j in enumerate(pm)))
            names = [cls[j] for j in best]
        else:
            names = [cls[int(np.argmax(row))] for row in P]
        ok += sum(nm == r["op_name"] for nm, r in zip(names, prs))
        n += len(prs)
    print(f"kind assigned acc: {ok / n:.4f}")

    # dia: RF+HGB ensemble over within-2%-merged classes; predict each group's modal dia
    from sklearn.ensemble import HistGradientBoostingClassifier
    vals = sorted({float(v) for v in yd})
    group: dict[float, float] = {}
    for v in vals:
        for g in group.values():
            if abs(v - g) / min(v, g) <= 0.02: group[v] = g; break
        else: group[v] = v
    cnt = Counter(float(v) for v in yd)
    mode: dict[float, float] = {}
    for v in vals:
        g = group[v]
        if g not in mode or cnt[v] > cnt[mode[g]]: mode[g] = v
    yg = np.array([group[float(v)] for v in yd]).astype(str)
    rf_all = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xd, yg)
    hg_all = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, max_leaf_nodes=63,
                                            l2_regularization=1.0, random_state=0,
                                            early_stopping=False).fit(Xd, yg)
    mk_all = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xk, yk)
    pickle.dump({"rf": rf_all, "hgb": hg_all, "mode": mode}, open("derived/pocket_tool_model.pkl", "wb"))
    pickle.dump(mk_all, open("derived/pocket_kind_model.pkl", "wb"))
    print("saved production models")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
