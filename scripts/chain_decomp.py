#!/usr/bin/env python3
"""Decompose hole chains, optimal assignment, explicit rules + tree fallback."""
from __future__ import annotations

import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.tree import DecisionTreeClassifier, _tree

FEAT_NAMES = [
    "d",
    "depth",
    "ld",
    "through",
    "stock_h",
    "mouth_off",
    "bottom_z",
    "d_frac",
    "d_is_int",
    "edge_dist",
]


def load_groups():
    groups = defaultdict(list)
    for r in csv.DictReader(open("derived/chains.csv")):
        if r["group_prefix"].startswith(("STEP1HOLE", "STEP1POCKET")):
            groups[r["part"]].append(
                {
                    "fd": float(r["final_diameter"]),
                    "chain": r["chain"],
                    "prefix": r["group_prefix"],
                    "diams": r["tool_diams"],
                }
            )
    feats = {r["part"]: r for r in map(json.loads, open("derived/features.jsonl"))}
    return groups, feats


def hole_feat(h, row):
    sx, sy, sz = row["stock"]
    d = h["d"]
    return {
        "d": d,
        "depth": h["depth"],
        "ld": h["depth"] / d if d else 0,
        "through": float(h["through"]),
        "stock_h": sz,
        "mouth_off": row["top_z"] - h["mouth_z"],
        "bottom_z": h["bottom_z"],
        "d_frac": d % 1.0,
        "d_is_int": float(abs(d - round(d)) < 0.01),
        "edge_dist": min(h["x"], sx - h["x"], h["y"], sy - h["y"]),
        "x": h["x"],
        "y": h["y"],
    }


def greedy_match(holes, rows, thresh=0.6):
    holes = list(holes)
    used = []
    for row in rows:
        if not holes:
            break
        i = min(range(len(holes)), key=lambda k: abs(holes[k]["d"] - row["fd"]))
        h = holes.pop(i)
        if abs(h["d"] - row["fd"]) > thresh:
            continue
        used.append((h, row))
    return used


def hungarian_match(holes, rows, thresh=0.6):
    if not holes or not rows:
        return []
    C = np.array([[abs(h["d"] - r["fd"]) for r in rows] for h in holes])
    ri, ci = linear_sum_assignment(C)
    out = []
    for i, j in zip(ri, ci):
        if C[i, j] <= thresh:
            out.append((holes[i], rows[j]))
    return out


def vec(f):
    return [f[k] for k in FEAT_NAMES]


def decompose(chain: str):
    ops = chain.split(">")
    has_spot = any(o.startswith("SPOT") for o in ops)
    has_spade = any("SPADE" in o for o in ops)
    has_bore = any(o.startswith("BORE") for o in ops)
    has_finish_mill = any("MILL_FINISH" in o for o in ops)
    n_enlarge = sum("ENLARGE" in o for o in ops)
    body = "other"
    if any("INDEXABLE" in o for o in ops):
        body = "indexable"
    elif any("GUN" in o for o in ops):
        body = "gun"
    elif any("MILL_ROUGH" in o for o in ops):
        body = "mill_rough_contour"
    elif any("MILL_THROUGH" in o for o in ops):
        body = "mill_through"
    elif any("MILL_BLIND" in o for o in ops):
        body = "mill_blind"
    elif any(o.startswith("DRILL") for o in ops):
        body = "plain_drill"
    n_bore = sum(o.startswith("BORE") for o in ops)
    return {
        "has_spot": int(has_spot),
        "body": body,
        "n_enlarge": n_enlarge,
        "has_spade": int(has_spade),
        "has_bore": int(has_bore),
        "has_finish_mill": int(has_finish_mill),
        "n_bore": n_bore,
    }


def export_tree_text(clf, feat_names, max_depth=4):
    t = clf.tree_
    lines = []

    def rec(node, depth, path):
        if depth > max_depth:
            return
        if t.feature[node] == _tree.TREE_UNDEFINED:
            vals = t.value[node][0]
            cls = clf.classes_[int(np.argmax(vals))]
            n = int(t.n_node_samples[node])
            lines.append(f"{'  '*depth}leaf -> {cls} n={n} path={path}")
            return
        fn = feat_names[t.feature[node]]
        thr = t.threshold[node]
        rec(t.children_left[node], depth + 1, path + [f"{fn}<={thr:.4g}"])
        rec(t.children_right[node], depth + 1, path + [f"{fn}>{thr:.4g}"])

    rec(0, 0, [])
    return "\n".join(lines[:80])


def top_splits(clf, feat_names, k=20):
    t = clf.tree_
    out = []

    def rec(node, depth):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            return
        fn = feat_names[t.feature[node]]
        thr = t.threshold[node]
        n = int(t.n_node_samples[node])
        out.append((n, depth, fn, thr))
        rec(t.children_left[node], depth + 1)
        rec(t.children_right[node], depth + 1)

    rec(0, 0)
    out.sort(reverse=True)
    return out[:k]


def compose_chain(pred):
    """Compose sub-decisions into a chain list of op names."""
    body = pred["body"]
    thru = pred.get("through", 0) > 0.5
    spot = pred["has_spot"]
    n_en = int(pred["n_enlarge"])
    spade = pred["has_spade"]
    bore = pred["has_bore"]
    fin = pred["has_finish_mill"]
    ops = []
    if body == "indexable":
        ops.append(
            "INDEXABLE_INSERT_DRILL_THROUGH_HOLE_FROM_SOLID"
            if thru
            else "INDEXABLE_INSERT_DRILL_BLIND_HOLE_FROM_SOLID"
        )
        if fin:
            ops.append("MILL_FINISH_BLIND_HOLE_FLAT_BOTTOM")
            if pred.get("ld", 0) >= 5:
                ops.append("GUN_DRILL_THROUGH_HOLE")
        for _ in range(n_en):
            ops.append(
                "DRILL_TO_ENLARGE_THROUGH_HOLE" if thru else "DRILL_TO_ENLARGE_BLIND_HOLE"
            )
        if bore:
            ops.append("BORE_BLIND_HOLE")
        return ops
    if body == "mill_through":
        return ["MILL_THROUGH_HOLE_FROM_SOLID_MATERIAL"]
    if body == "mill_blind":
        ops = ["MILL_BLIND_HOLE_FROM_SOLID_MATERIAL"]
        nb = int(pred.get("n_bore", 1 if bore else 0))
        for _ in range(max(nb, 1 if bore else 0)):
            ops.append("BORE_BLIND_HOLE")
        return ops
    if body == "mill_rough_contour":
        ops = []
        if spot:
            ops.append("SPOT_DRILL")
        ops.append("DRILL_BLIND_HOLE_INTO_CENTER")
        ops.append("MILL_ROUGH_BLIND_HOLE_CONTOUR")
        return ops
    # gun / plain_drill
    if spot:
        ops.append("SPOT_DRILL")
    if body == "gun":
        # typical: spot > drill_blind > gun_through > [spade] > enlarge*
        ops.append("DRILL_BLIND_HOLE_INTO_CENTER")
        if n_en and not thru:
            # enlarge then gun (rare)
            for _ in range(n_en):
                ops.append("DRILL_TO_ENLARGE_BLIND_HOLE")
            ops.append("GUN_DRILL_BLIND_HOLE" if not thru else "GUN_DRILL_THROUGH_HOLE")
            if bore:
                ops.append("BORE_BLIND_HOLE")
            return ops
        ops.append("GUN_DRILL_THROUGH_HOLE" if thru or True else "GUN_DRILL_BLIND_HOLE")
        if pred.get("gun_blind"):
            ops[-1] = "GUN_DRILL_BLIND_HOLE"
        if spade:
            ops.append("SPADE_DRILL_TO_ENLARGE_THROUGH_HOLE")
        for _ in range(n_en):
            ops.append("DRILL_TO_ENLARGE_THROUGH_HOLE")
        return ops
    # plain drill
    if thru:
        if spot:
            ops.append("DRILL_THROUGH_HOLE_INTO_CENTER")
        else:
            if not ops:
                ops.append("DRILL_THROUGH_HOLE_FROM_SOLID_MATERIAL")
            else:
                ops.append("DRILL_THROUGH_HOLE_INTO_CENTER")
        for _ in range(n_en):
            ops.append("DRILL_TO_ENLARGE_THROUGH_HOLE")
    else:
        if spot or ops:
            ops.append("DRILL_BLIND_HOLE_INTO_CENTER")
        else:
            ops.append("DRILL_THROUGH_HOLE_FROM_SOLID_MATERIAL")
        for _ in range(n_en):
            ops.append("DRILL_TO_ENLARGE_BLIND_HOLE")
        if bore:
            ops.append("BORE_BLIND_HOLE")
    return ops


def rule_body(f):
    """Explicit body rules from tree split mining. Return (body or None if residual)."""
    d, ld, thru, depth = f["d"], f["ld"], f["through"] > 0.5, f["depth"]
    # mill through: large through holes
    if thru and d >= 39.9:
        return "mill_through"
    # mill blind / mill contour / bore family: large blind
    if (not thru) and d >= 39.9:
        # mill_rough_contour vs mill_blind: typically contour after pilot when not huge?
        # residual handled later
        if d < 50.1 and f.get("d_is_int", 0) < 0.5:
            return "mill_rough_contour"
        return "mill_blind"
    # indexable: medium-large, modest L/D
    if d >= 15.9 and ld < 4.95 and d < 39.9:
        return "indexable"
    # gun: deep
    if ld >= 4.95 and d < 39.9:
        return "gun"
    # small: plain drill
    if d < 15.9:
        return "plain_drill"
    return None


def rule_spot(f, body):
    if body in ("indexable", "mill_through", "mill_blind"):
        return 0
    if body in ("plain_drill", "gun", "mill_rough_contour"):
        # FROM_SOLID no spot when no center: rare through from solid at tiny d
        if f["d"] < 3.05 and f["through"] > 0.5:
            return 0
        return 1
    return 1


def rule_enlarge(f, body):
    d, ld, thru = f["d"], f["ld"], f["through"] > 0.5
    if body == "mill_through" or body == "mill_blind" or body == "mill_rough_contour":
        return 0
    if body == "indexable":
        # enlarge when not landing on insert size
        if d >= 20.1 and abs(d - round(d * 2) / 2) > 0.05 and d < 32:
            return 1
        if 16.05 < d < 19.9:
            return 1
        return 0
    if body == "gun":
        # after gun, enlarge to final if d not equal to gun tool
        # typical: 1 enlarge; 2 if larger; 3 rare
        if d >= 32.4:
            return 2 if d < 36 else 3
        if d >= 20.05:
            return 1
        if d >= 16.05:
            return 1
        return 0
    # plain
    if thru and d >= 8.05:
        return 1
    if (not thru) and d >= 10.05:
        return 1
    return 0


def rule_spade(f, body):
    if body != "gun":
        return 0
    d, ld = f["d"], f["ld"]
    # spade after gun for mid-large
    if d >= 20.05 and d < 32.6 and ld >= 4.95:
        return 1
    return 0


def rule_bore(f, body):
    if body == "mill_blind" and f["d"] >= 50:
        return 1
    if body == "plain_drill" and f["through"] < 0.5 and f["d"] >= 12 and f["d_is_int"] < 0.5:
        return 1
    return 0


def rule_finish(f, body):
    return 0


def predict_rules(f):
    body = rule_body(f)
    residual = body is None
    if body is None:
        body = "plain_drill"
    pred = {
        "body": body,
        "has_spot": rule_spot(f, body),
        "n_enlarge": rule_enlarge(f, body),
        "has_spade": rule_spade(f, body),
        "has_bore": rule_bore(f, body),
        "has_finish_mill": rule_finish(f, body),
        "through": f["through"],
        "ld": f["ld"],
        "n_bore": 1 if rule_bore(f, body) else 0,
        "residual": residual,
    }
    return pred, ">".join(compose_chain(pred))


def main():
    groups, feats = load_groups()
    n_gt = sum(len(v) for v in groups.values())
    n_holes = sum(len(feats[p]["holes"]) for p in groups if p in feats)
    g_n = h_n = 0
    matched = []
    greedy_only = hung_only = 0
    for p, rows in groups.items():
        row = feats[p]
        holes = list(row["holes"])
        g = greedy_match(holes, rows)
        h = hungarian_match(list(row["holes"]), rows)
        g_n += len(g)
        h_n += len(h)
        for hh, rr in h:
            f = hole_feat(hh, row)
            f["part"] = p
            f["chain"] = rr["chain"]
            f["fd"] = rr["fd"]
            f["prefix"] = rr["prefix"]
            matched.append(f)
    print(f"GT hole/pocket rows: {n_gt}  detected holes: {n_holes}")
    print(f"greedy matched: {g_n}  hungarian: {h_n}  recovered extra: {h_n - g_n}")
    print(f"match rate hung {h_n/n_gt:.4f} greedy {g_n/n_gt:.4f}")

    X = np.array([vec(f) for f in matched])
    parts = [f["part"] for f in matched]
    idx = {p: i for i, p in enumerate(sorted(set(parts)))}
    test = np.array([idx[p] % 5 == 4 for p in parts])
    print(f"matched n={len(matched)} train={ (~test).sum()} holdout={test.sum()} parts_holdout={sum(1 for p,i in idx.items() if i%5==4)}")

    decs = [decompose(f["chain"]) for f in matched]
    trees = {}
    for name, yfn in [
        ("has_spot", lambda d: d["has_spot"]),
        ("body", lambda d: d["body"]),
        ("n_enlarge", lambda d: d["n_enlarge"]),
        ("has_spade", lambda d: d["has_spade"]),
        ("has_bore", lambda d: d["has_bore"]),
        ("has_finish_mill", lambda d: d["has_finish_mill"]),
    ]:
        y = np.array([yfn(d) for d in decs])
        clf = DecisionTreeClassifier(max_depth=16, min_samples_leaf=8, random_state=0)
        clf.fit(X[~test], y[~test])
        acc = clf.score(X[test], y[test])
        trees[name] = clf
        print(f"\n===== {name} holdout acc={acc:.4f} classes={list(clf.classes_)} =====")
        print("top splits (n, depth, feat, thr):")
        for item in top_splits(clf, FEAT_NAMES, 15):
            print("  ", item)
        print(export_tree_text(clf, FEAT_NAMES, 5))

    # refine rules by looking at empirical tables
    print("\n===== empirical body by d bins / ld / through =====")
    bodies = [d["body"] for d in decs]
    for thru in (0, 1):
        for lo, hi in [(0, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 32.5), (32.5, 40), (40, 50), (50, 80), (80, 200)]:
            for ldlo, ldhi in [(0, 5), (5, 20)]:
                mask = [
                    (f["through"] == thru) and lo <= f["d"] < hi and ldlo <= f["ld"] < ldhi
                    for f in matched
                ]
                n = sum(mask)
                if n < 8:
                    continue
                c = Counter(b for b, m in zip(bodies, mask) if m)
                print(f" thru={thru} d=[{lo},{hi}) L/D=[{ldlo},{ldhi}) n={n} {c.most_common(3)}")

    print("\n===== n_enlarge by body,d =====")
    for body in ("plain_drill", "gun", "indexable"):
        for lo, hi in [(0, 8), (8, 10), (10, 12), (12, 16), (16, 20), (20, 25), (25, 32.5), (32.5, 40)]:
            sel = [(d["n_enlarge"], f["d"]) for d, f in zip(decs, matched) if d["body"] == body and lo <= f["d"] < hi]
            if len(sel) < 10:
                continue
            c = Counter(x[0] for x in sel)
            print(f"  {body} d=[{lo},{hi}) n={len(sel)} {c}")

    print("\n===== has_spade by d, ld (gun only) =====")
    for lo, hi in [(0, 12), (12, 16), (16, 20), (20, 25), (25, 32.5), (32.5, 40)]:
        sel = [d["has_spade"] for d, f in zip(decs, matched) if d["body"] == "gun" and lo <= f["d"] < hi]
        if len(sel) < 8:
            continue
        print(f"  d=[{lo},{hi}) n={len(sel)} mean_spade={np.mean(sel):.3f}")

    print("\n===== has_bore by body,d =====")
    for body in set(bodies):
        sel = [(d["has_bore"], f["d"], f["through"]) for d, f in zip(decs, matched) if d["body"] == body]
        if sum(x[0] for x in sel) < 5:
            continue
        print(body, "n", len(sel), "bore", sum(x[0] for x in sel), "d_when_bore", np.median([x[1] for x in sel if x[0]]))

    # evaluate rules + hybrid
    y_true = [f["chain"] for f in matched]
    y_rule = []
    residuals = []
    for f in matched:
        pred, ch = predict_rules(f)
        y_rule.append(ch)
        residuals.append(pred["residual"])

    def acc(mask):
        yt = [a for a, m in zip(y_true, mask) if m]
        yp = [a for a, m in zip(y_rule, mask) if m]
        return np.mean([a == b for a, b in zip(yt, yp)]) if yt else 0

    print("\n===== RULE exact-chain =====")
    print("all", acc([True] * len(matched)), "holdout", acc(test), "train", acc(~test))
    print("residual fraction holdout", np.mean([r for r, m in zip(residuals, test) if m]))

    # hybrid: use trees for residual or always compose from tree subpreds
    y_hyb = []
    y_treecomp = []
    for i, f in enumerate(matched):
        xf = np.array(vec(f)).reshape(1, -1)
        tp = {
            "has_spot": int(trees["has_spot"].predict(xf)[0]),
            "body": trees["body"].predict(xf)[0],
            "n_enlarge": int(trees["n_enlarge"].predict(xf)[0]),
            "has_spade": int(trees["has_spade"].predict(xf)[0]),
            "has_bore": int(trees["has_bore"].predict(xf)[0]),
            "has_finish_mill": int(trees["has_finish_mill"].predict(xf)[0]),
            "through": f["through"],
            "ld": f["ld"],
            "n_bore": 1 if trees["has_bore"].predict(xf)[0] else 0,
        }
        y_treecomp.append(">".join(compose_chain(tp)))
        if residuals[i]:
            y_hyb.append(y_treecomp[-1])
        else:
            y_hyb.append(y_rule[i])

    def acc2(yp, mask):
        return np.mean([a == b for a, b, m in zip(y_true, yp, mask) if m])

    print("tree-compose holdout", acc2(y_treecomp, test))
    print("hybrid rules+tree-residual holdout", acc2(y_hyb, test))

    # confusions on holdout
    print("\n===== top confusions (rules holdout) =====")
    c = Counter()
    for yt, yp, m in zip(y_true, y_rule, test):
        if m and yt != yp:
            c[(yt, yp)] += 1
    for (yt, yp), n in c.most_common(15):
        print(n, "TRUE", yt)
        print("   PRED", yp)

    print("\n===== top confusions (tree-compose holdout) =====")
    c = Counter()
    for yt, yp, m in zip(y_true, y_treecomp, test):
        if m and yt != yp:
            c[(yt, yp)] += 1
    for (yt, yp), n in c.most_common(12):
        print(n, "TRUE", yt)
        print("   PRED", yp)

    Path("derived").mkdir(exist_ok=True)
    pickle.dump(trees, open("derived/chain_subtrees.pkl", "wb"))
    print("saved derived/chain_subtrees.pkl")

    # dump arrays for findings via pickle of summary
    summary = {
        "g_n": g_n,
        "h_n": h_n,
        "n_gt": n_gt,
        "n_matched": len(matched),
        "rule_holdout": float(acc2(y_rule, test)),
        "treecomp_holdout": float(acc2(y_treecomp, test)),
        "hybrid_holdout": float(acc2(y_hyb, test)),
        "sub_acc": {k: float(trees[k].score(X[test], np.array([decompose(f["chain"])[k if k!='body' else 'body'] for f in [matched[i] for i in np.where(test)[0]]]))) for k in trees},
    }
    # fix sub_acc simpler
    hold_idx = np.where(test)[0]
    summary["sub_acc"] = {}
    for name in trees:
        y = np.array([decs[i][name] for i in hold_idx])
        summary["sub_acc"][name] = float(trees[name].score(X[hold_idx], y))
    pickle.dump(summary, open("/tmp/chaindecomp_summary.pkl", "wb"))
    print(summary)


if __name__ == "__main__":
    main()
