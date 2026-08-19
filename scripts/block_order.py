import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_parse import parse
from baseline import (ROOT, DER, load_ops, load_model, milled_features,
                      fillet_clusters, build_plan, plan_composition)

FEATURES = [
    "min_hole_depth", "min_hole_dia", "hole_minus_feat_depth",
    "max_hole_depth_over_h", "max_hole_depth", "aspect", "n_distinct_hole_dia",
    "deepest_is_hole", "max_feat_depth_over_h", "n_sunk", "n_holes",
    "max_feat_depth", "n_feats", "n_fillet", "n_flooronly", "n_cham",
    "max_hole_dia", "max_feat_dia", "hole_frac", "log_vol", "part_height",
    "n_drill_ops", "n_mill_ops", "drill_op_frac", "n_ops",
]


def extract(p):
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

    h = (p.bbox[2][1] - p.bbox[2][0]) if p.bbox else 1.0
    h = h or 1.0
    mx_hz = max(depths) if depths else 0.0
    mx_fz = max(fdep) if fdep else 0.0
    mx_hd = max(dias) if dias else 0.0
    return {
        "n_holes": n_holes, "n_sunk": sunk, "n_flush": flush,
        "n_feats": len(feats), "n_fillet": len(fillet_clusters(p)),
        "n_flooronly": sum(1 for f in feats if f["radius"] is None),
        "n_cham": n_cham,
        "max_hole_dia": mx_hd,
        "min_hole_dia": min(dias) if dias else 0.0,
        "max_hole_depth": mx_hz,
        "min_hole_depth": min(depths) if depths else 0.0,
        "max_feat_dia": max(fdia) if fdia else 0.0,
        "max_feat_depth": mx_fz,
        "hole_frac": n_holes / max(1, n_holes + len(feats)),
        "log_vol": math.log10(vol),
        "part_height": h,
        "max_hole_depth_over_h": mx_hz / h,
        "max_feat_depth_over_h": mx_fz / h,
        "hole_minus_feat_depth": mx_hz - mx_fz,
        "deepest_is_hole": 1.0 if mx_hz > mx_fz else 0.0,
        "aspect": (mx_hz / mx_hd) if mx_hd else 0.0,
        "n_distinct_hole_dia": float(len({round(x, 2) for x in dias})),
    }


def fit_logistic(X, y, epochs=400, lr=0.2, l2=1e-3):
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


def _bin_edges(col, nbins=32):
    vals = sorted(set(col))
    if len(vals) <= nbins:
        return vals[:-1] if len(vals) > 1 else []
    s = sorted(col)
    return [s[int(len(s) * (i + 1) / nbins)] for i in range(nbins - 1)]


def _best_split(rows, grad, feats_idx, B, edges):
    n = len(rows)
    if n < 20:
        return None
    tot = sum(grad[i] for i in rows)
    best = None
    for j in feats_idx:
        nb = len(edges[j]) + 1
        sums = [0.0] * nb
        cnts = [0] * nb
        for i in rows:
            b = B[i][j]
            sums[b] += grad[i]
            cnts[b] += 1
        ls = lc = 0.0
        for b in range(nb - 1):
            ls += sums[b]
            lc += cnts[b]
            rc = n - lc
            if lc < 10 or rc < 10:
                continue
            gain = ls * ls / lc + (tot - ls) ** 2 / rc
            if best is None or gain > best[0]:
                best = (gain, j, b)
    return best


def _fit_tree(rows, grad, feats_idx, B, edges, depth):
    if depth == 0:
        v = sum(grad[i] for i in rows) / max(1, len(rows))
        return {"leaf": v}
    sp = _best_split(rows, grad, feats_idx, B, edges)
    if sp is None:
        return {"leaf": sum(grad[i] for i in rows) / max(1, len(rows))}
    _, j, b = sp
    L = [i for i in rows if B[i][j] <= b]
    R = [i for i in rows if B[i][j] > b]
    if not L or not R:
        return {"leaf": sum(grad[i] for i in rows) / max(1, len(rows))}
    return {"f": j, "b": b,
            "l": _fit_tree(L, grad, feats_idx, B, edges, depth - 1),
            "r": _fit_tree(R, grad, feats_idx, B, edges, depth - 1)}


def _apply(tree, bins):
    while "leaf" not in tree:
        tree = tree["l"] if bins[tree["f"]] <= tree["b"] else tree["r"]
    return tree["leaf"]


def _to_bins(x, edges):
    out = []
    for j, v in enumerate(x):
        e = edges[j]
        lo, hi = 0, len(e)
        while lo < hi:
            mid = (lo + hi) // 2
            if v <= e[mid]:
                hi = mid
            else:
                lo = mid + 1
        out.append(lo)
    return out


def fit_gbm(X, y, rounds=180, lr=0.12, depth=3):
    d = len(X[0])
    edges = [_bin_edges([r[j] for r in X]) for j in range(d)]
    B = [_to_bins(r, edges) for r in X]
    F = [0.0] * len(X)
    trees = []
    idx = list(range(d))
    for _ in range(rounds):
        grad = [y[i] - 1 / (1 + math.exp(-max(-30, min(30, F[i]))))
                for i in range(len(X))]
        t = _fit_tree(list(range(len(X))), grad, idx, B, edges, depth)
        trees.append(t)
        for i in range(len(X)):
            F[i] += lr * _apply(t, B[i])
    return {"kind": "gbm", "trees": trees, "edges": edges, "lr": lr,
            "features": FEATURES}


def predict_drill_first(clf, feats):
    if clf.get("kind") == "ensemble":
        ps = [predict_drill_first(m, feats) for m in clf["members"]]
        return sum(ps) / len(ps)
    if clf.get("kind") == "gbm":
        x = [feats[k] for k in clf["features"]]
        bins = _to_bins(x, clf["edges"])
        F = sum(clf["lr"] * _apply(t, bins) for t in clf["trees"])
        return 1 / (1 + math.exp(-max(-30, min(30, F))))
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
                continue
            try:
                p = parse(f)
            except Exception:
                continue
            fe = extract(p)
            fe.update(plan_composition(build_plan(p, model), model))
            X.append([fe[k] for k in FEATURES])
            y.append(1 if kinds[0] == "d" else 0)
            pids.append(pid)
        return X, y, pids

    print("gathering training features (mixed parts only)...")
    Xtr, ytr, _ = gather(train)
    print(f"  train mixed parts: {len(Xtr):,}  drill-first "
          f"{100*sum(ytr)/len(ytr):.1f}%")
    Xv, yv, _ = gather(val)
    base = max(sum(yv), len(yv) - sum(yv)) / len(yv)

    def acc_of(c):
        return sum(1 for x, t in zip(Xv, yv)
                   if (predict_drill_first(c, dict(zip(FEATURES, x))) >= 0.5) == t) \
            / len(yv)

    cut = int(0.8 * len(Xtr))
    Xa, ya, Xb, yb = Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:]

    def dev_acc(c):
        return sum(1 for x, t in zip(Xb, yb)
                   if (predict_drill_first(c, dict(zip(FEATURES, x))) >= 0.5) == t) \
            / len(yb)

    grid = [(180, 0.12, 3), (400, 0.06, 3), (400, 0.06, 4),
            (700, 0.04, 4), (400, 0.10, 5), (900, 0.03, 5)]
    print("  tuning on inner split:")
    best_cfg, best_dev = None, -1
    dev_scores = []
    for rounds, lr, depth in grid:
        c = fit_gbm(Xa, ya, rounds=rounds, lr=lr, depth=depth)
        a = dev_acc(c)
        dev_scores.append(((rounds, lr, depth), a))
        print(f"    rounds={rounds:4d} lr={lr:.2f} depth={depth}  dev {100*a:.1f}%")
        if a > best_dev:
            best_cfg, best_dev = (rounds, lr, depth), a
    print(f"  -> best config {best_cfg} (dev {100*best_dev:.1f}%)")

    ranked = sorted(dev_scores, key=lambda kv: -kv[1])[:3]
    members = [fit_gbm(Xtr, ytr, rounds=r, lr=l, depth=d)
               for (r, l, d), _ in ranked]
    lin = fit_logistic(Xtr, ytr)
    gbm = members[0]
    ens = {"kind": "ensemble", "members": members + [lin],
           "features": FEATURES}
    ens_gbm_only = {"kind": "ensemble", "members": members,
                    "features": FEATURES}

    a_lin, a_gbm = acc_of(lin), acc_of(gbm)
    a_ens, a_eg = acc_of(ens), acc_of(ens_gbm_only)
    print(f"\n  val mixed parts  : {len(Xv):,}")
    print(f"  majority baseline: {100*base:.1f}%")
    print(f"  logistic         : {100*a_lin:.1f}%")
    print(f"  best single gbm  : {100*a_gbm:.1f}%")
    print(f"  ensemble (gbm x3): {100*a_eg:.1f}%")
    print(f"  ensemble + linear: {100*a_ens:.1f}%")
    cands = [(a_gbm, gbm, "single gbm"), (a_eg, ens_gbm_only, "gbm ensemble"),
             (a_ens, ens, "gbm+linear ensemble"), (a_lin, lin, "logistic")]
    acc, clf, label = max(cands, key=lambda x: x[0])
    print(f"  -> keeping {label}")

    order = sorted(zip(FEATURES, lin["w"]), key=lambda kv: -abs(kv[1]))
    print("\n  strongest linear weights (positive = drill first):")
    for k, wv in order[:6]:
        print(f"    {wv:+7.3f}  {k}")

    out = os.path.join(DER, "block_order.json")
    json.dump(clf, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    if acc <= base + 0.01:
        print("WARNING: no better than always predicting the majority class.")


if __name__ == "__main__":
    main()
