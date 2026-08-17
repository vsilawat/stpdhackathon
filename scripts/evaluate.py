"""Score the baseline against the true NX process plans on held-out parts.

Metrics, in plain terms:

  step F1       did we list the right machining steps? (order ignored)
  tool F1       did we pick the right cutting tools?   (order ignored)
  count error   did we get the number of steps right?
  sequence      how close is our ordering to the real one
                (1.0 = identical, 0.0 = nothing in common)
  exact match   fraction of parts where the whole plan is perfect
"""
import collections, csv, json, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import load_model, predict, load_ops, strip

DER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derived")
ROOT = "/Users/vasusilawat/Desktop/stpd/data/MachinePlan-10K"


def multiset_f1(pred, true):
    p, t = collections.Counter(pred), collections.Counter(true)
    overlap = sum((p & t).values())
    if not overlap:
        return 0.0
    prec, rec = overlap / max(1, sum(p.values())), overlap / max(1, sum(t.values()))
    return 2 * prec * rec / (prec + rec)


def seq_similarity(a, b):
    """1 - normalised edit distance between two step sequences."""
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


def main():
    model = load_model()
    by_part, _ = load_ops()
    val = model["val_ids"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(val)
    val = val[:limit]
    print(f"evaluating on {len(val):,} held-out parts\n")

    m = collections.defaultdict(list)
    fails = 0
    for pid in val:
        f = os.path.join(ROOT, pid, pid + ".stp")
        if not os.path.exists(f):
            continue
        try:
            pred = predict(pid, model)
        except Exception:
            fails += 1
            continue
        true_ops = by_part[pid]
        tn = [strip(o["op_name"]) for o in true_ops]
        tt = [o["tool"] for o in true_ops]
        pn = [o["name"] for o in pred]
        pt = [o["tool"] for o in pred]

        m["step_f1"].append(multiset_f1(pn, tn))
        m["tool_f1"].append(multiset_f1(pt, tt))
        m["seq"].append(seq_similarity(pn, tn))
        m["count_err"].append(len(pn) - len(tn))
        m["abs_count_err"].append(abs(len(pn) - len(tn)))
        m["exact"].append(1.0 if pn == tn else 0.0)
        # per-target breakdowns
        m["chamfer_ok"].append(
            1.0 if pn.count("AREA_MILL") == tn.count("AREA_MILL") else 0.0)
        drill_p = [x for x in pn if model["is_drill"].get(x)]
        drill_t = [x for x in tn if model["is_drill"].get(x)]
        m["drill_f1"].append(multiset_f1(drill_p, drill_t))
        for label, jt in [("mill25d_f1", "VolumeBased25DMillingOperation"),
                          ("cylmill_f1", "CylinderMilling"),
                          ("contour_f1", "SurfaceContour")]:
            a = [x for x in pn if model["op_type"].get(x) == jt]
            b = [x for x in tn if model["op_type"].get(x) == jt]
            if a or b:
                m[label].append(multiset_f1(a, b))

    def row(k, pct=True):
        v = m[k]
        mu = sum(v) / len(v)
        return f"{100*mu:6.1f}%" if pct else f"{mu:+7.2f}"

    print("=== OVERALL (mean over parts) ===")
    print(f"  step F1  (right machining steps) : {row('step_f1')}")
    print(f"  tool F1  (right cutting tools)   : {row('tool_f1')}")
    print(f"  sequence similarity              : {row('seq')}")
    print(f"  whole plan exactly right         : {row('exact')}")
    print()
    print("=== BREAKDOWN ===")
    print(f"  chamfer step count exactly right : {row('chamfer_ok')}")
    print(f"  drilling steps F1                : {row('drill_f1')}")
    print(f"  pocket/slot/notch steps F1       : {row('mill25d_f1')}")
    print(f"  milled-hole steps F1             : {row('cylmill_f1')}")
    print(f"  chamfer contour steps F1         : {row('contour_f1')}")
    print(f"  mean signed step-count error     : {row('count_err', False)}")
    print(f"  mean absolute step-count error   : {row('abs_count_err', False)}")
    if fails:
        print(f"\n  parts that failed to parse: {fails}")

    ce = sorted(m["count_err"])
    print(f"  step-count error p10/median/p90  : "
          f"{ce[int(.1*len(ce))]:+d} / {ce[len(ce)//2]:+d} / {ce[int(.9*len(ce))]:+d}")


if __name__ == "__main__":
    main()
