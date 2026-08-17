"""Predict whether a part's plan starts with drilling or with milling.

On parts containing both kinds of work, NX puts each kind in a contiguous block
(93.3% of parts) but the order between the two blocks is close to a coin flip
(59/41). Getting it right is worth up to 4 rubric points on the Easy track,
because it roughly halves the normalized Levenshtein distance.

Fits a small logistic regression on geometric features. Pure Python -- no
third-party dependencies, consistent with the rest of the pipeline.
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_parse import parse
from baseline import (ROOT, DER, load_ops, load_model, milled_features,
                      fillet_clusters)

FEATURES = [
    "n_holes", "n_sunk", "n_flush", "n_feats", "n_fillet", "n_flooronly",
    "n_cham", "max_hole_dia", "min_hole_dia", "max_hole_depth",
    "max_feat_dia", "max_feat_depth", "hole_frac", "log_vol",
]


def extract(p):
    """Geometric features for one parsed part."""
    up = [f["origin"][2] for f in p.faces
          if f["kind"] == "plane" and f.get("axis") and f.get("origin")
          and abs(abs(f["axis"][2]) - 1) < 1e-6 and f["axis"][2] > 0]
    top = max(up) if up else (p.bbox[2][1] if p.bbox else 0.0)

    sunk = flush = 0
    dias, depths = [], []
    for c in p.hole_cylinders():
        ax = c.get("axis") or (0, 0, 1)
        dias.append(2 * c["radius"])
        if c.get("depth") is not None:
            depths.append(c["depth"])
        ex = c.get("extent")
        if ex is None or abs(abs(ax[2]) - 1) > 1e-6:
            continue
        if ex[1] < top - 1e-3:
            sunk += 1
        else:
            flush += 1

    feats = milled_features(p)
    fdia = [2 * f["radius"] for f in feats if f["radius"] is not None]
    fdep = [f["depth"] for f in feats if f["depth"] is not None]
    n_cham = sum(1 for f in p.faces
                 if f["kind"] == "plane" and f.get("axis_aligned") is False)
    n_holes = len(p.hole_cylinders())
    vol = 1.0
    if p.bbox:
        for lo, hi in p.bbox:
            vol *= max(1e-6, hi - lo)

    return {
        "n_holes": n_holes, "n_sunk": sunk, "n_flush": flush,
        "n_feats": len(feats), "n_fillet": len(fillet_clusters(p)),
        "n_flooronly": sum(1 for f in feats if f["radius"] is None),
        "n_cham": n_cham,
        "max_hole_dia": max(dias) if dias else 0.0,
        "min_hole_dia": min(dias) if dias else 0.0,
        "max_hole_depth": max(depths) if depths else 0.0,
        "max_feat_dia": max(fdia) if fdia else 0.0,
        "max_feat_depth": max(fdep) if fdep else 0.0,
        "hole_frac": n_holes / max(1, n_holes + len(feats)),
        "log_vol": math.log10(vol),
    }


def fit_logistic(X, y, epochs=400, lr=0.2, l2=1e-3):
    """Plain batch gradient descent on standardized features."""
    d = len(X[0])
    mu = [sum(r[j] for r in X) / len(X) for j in range(d)]
    sd = [math.sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / len(X)) or 1.0
          for j in range(d)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(d)] for r in X]
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for z, t in zip(Z, y):
            s = b + sum(w[j] * z[j] for j in range(d))
            pr = 1 / (1 + math.exp(-max(-30, min(30, s))))
            e = pr - t
            for j in range(d):
                gw[j] += e * z[j]
            gb += e
        n = len(Z)
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return {"w": w, "b": b, "mu": mu, "sd": sd, "features": FEATURES}


def predict_drill_first(clf, feats):
    z = [(feats[k] - m) / s for k, m, s in
         zip(clf["features"], clf["mu"], clf["sd"])]
    s = clf["b"] + sum(a * b for a, b in zip(clf["w"], z))
    return 1 / (1 + math.exp(-max(-30, min(30, s))))


def main():
    by_part, _ = load_ops()
    model = load_model()
    train, val = model["train_ids"], model["val_ids"]

    def gather(ids):
        X, y, pids = [], [], []
        for pid in ids:
            f = os.path.join(ROOT, pid, pid + ".stp")
            if not os.path.exists(f):
                continue
            kinds = [("d" if o["op_type"] == "Drilling" else "m")
                     for o in by_part[pid]]
            if "d" not in kinds or "m" not in kinds:
                continue          # only mixed parts have an order to choose
            try:
                p = parse(f)
            except Exception:
                continue
            fe = extract(p)
            X.append([fe[k] for k in FEATURES])
            y.append(1 if kinds[0] == "d" else 0)
            pids.append(pid)
        return X, y, pids

    print("gathering training features (mixed parts only)...")
    Xtr, ytr, _ = gather(train)
    print(f"  train mixed parts: {len(Xtr):,}  drill-first "
          f"{100*sum(ytr)/len(ytr):.1f}%")
    clf = fit_logistic(Xtr, ytr)

    Xv, yv, _ = gather(val)
    base = max(sum(yv), len(yv) - sum(yv)) / len(yv)
    acc = sum(1 for x, t in zip(Xv, yv)
              if (predict_drill_first(clf, dict(zip(FEATURES, x))) >= 0.5) == t) \
        / len(yv)
    print(f"  val mixed parts  : {len(Xv):,}")
    print(f"  majority baseline: {100*base:.1f}%")
    print(f"  logistic accuracy: {100*acc:.1f}%")

    order = sorted(zip(FEATURES, clf["w"]), key=lambda kv: -abs(kv[1]))
    print("\n  strongest weights (positive = drill first):")
    for k, wv in order[:8]:
        print(f"    {wv:+7.3f}  {k}")

    out = os.path.join(DER, "block_order.json")
    json.dump(clf, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    if acc <= base + 0.01:
        print("WARNING: no better than always predicting the majority class.")


if __name__ == "__main__":
    main()
