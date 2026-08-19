from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from machineplan import plan as planmod
from train_order import labels

FEATS: dict | None = None

def init():
    global FEATS
    FEATS = {r["part"]: r for r in map(json.loads, open("derived/features.jsonl"))}

def ext_features(found) -> list[float]:
    hs, ps = found.holes, found.pockets
    tz = found.top_z
    deps = [h.depth for h in hs]
    bl = [h.depth for h in hs if not h.through]
    th = [h.depth for h in hs if h.through]
    lds = [h.depth / h.diameter for h in hs if h.diameter]
    chains = planmod.chain_names(found)
    return [min(deps, default=0), max(deps, default=0), float(np.mean(deps)) if deps else 0,
            min(bl, default=0), max(bl, default=0), min(th, default=0),
            min(lds, default=0), float(np.mean(lds)) if lds else 0,
            min((tz - h.mouth_z for h in hs), default=0), max((tz - h.mouth_z for h in hs), default=0),
            len(chains), float(np.mean([len(c) for c in chains])) if chains else 0,
            sum("HOLE" in n and n.startswith("MILL_") for c in chains for n in c),
            found.stock[0], found.stock[1], tz,
            min((p.depth for p in ps), default=0), sum(p.kind == "edge" for p in ps),
            min((h.bottom_z for h in hs), default=0),
            sum(h.diameter * h.diameter * h.depth for h in hs) / 1000.0]

def feat_row(part: str):
    row = FEATS.get(part)
    if not row or "error" in row: return None
    f = planmod.features_from_row(row)
    return part, planmod.order_features(f), ext_features(f)

def evals(name, X, y, test):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)
    rf.fit(X[~test], y[~test])
    hg = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
                                        l2_regularization=1.0, random_state=0)
    hg.fit(X[~test], y[~test])
    pr = rf.predict_proba(X[test])[:, 1] + hg.predict_proba(X[test])[:, 1]
    ens = (pr >= 1.0) == y[test]
    print(f"{name}: rf={rf.score(X[test], y[test]):.4f} hgb={hg.score(X[test], y[test]):.4f} "
          f"ens={ens.mean():.4f} (n={test.sum()})", flush=True)

def main() -> int:
    tw, hm = labels()
    parts = sorted(set(tw) | set(hm))
    rows = {}
    with Pool(8, initializer=init) as pool:
        for r in pool.imap(feat_row, parts, chunksize=50):
            if r: rows[r[0]] = (r[1], r[2])
    all_parts = sorted({r["part"] for r in map(json.loads, open("derived/features.jsonl"))})
    idx = {p: i for i, p in enumerate(all_parts)}
    for name, yb in (("order", tw), ("hm", hm)):
        keep = [p for p in sorted(yb) if p in rows]
        y = np.array([yb[p] for p in keep])
        test = np.array([idx[p] % 5 == 4 for p in keep])
        X23 = np.array([rows[p][0] for p in keep])
        X43 = np.array([rows[p][0] + rows[p][1] for p in keep])
        evals(f"{name}/23f", X23, y, test)
        evals(f"{name}/43f", X43, y, test)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
