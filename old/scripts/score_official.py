"""Validate every submission file and score it against the organizers' rubric.

Runs the organizers' own validate_submission.py (imported, not shelled out) over
the whole submission, then computes the Easy-track metrics exactly as the
rubric defines them:

    normalized Levenshtein = edit_distance(P, G) / max(|P|, |G|)   (lower better)
    F1                     = harmonic mean of precision/recall over the
                             multiset of predicted operations

and converts both to rubric points (10 each, 20 total).

Scored on held-out parts only -- the training parts would flatter us.
"""
import collections, csv, importlib.util, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import DER, load_model

REF = Path(DER).parent / "reference"
SUB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DER).parent / "submission" / "easy"

_spec = importlib.util.spec_from_file_location(
    "validate_submission", REF / "validate_submission.py")
_vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vs)

# rubric tables (lower band edge -> points)
LEV_BANDS = [(0.1, 10), (0.2, 8), (0.3, 6), (0.4, 4), (0.5, 2), (1.01, 0)]
F1_BANDS = [(0.95, 10), (0.85, 8), (0.75, 6), (0.65, 4), (0.55, 2), (-1, 0)]


def lev_points(d):
    for hi, p in LEV_BANDS:
        if d <= hi:
            return p
    return 0


def f1_points(f):
    for lo, p in F1_BANDS:
        if f >= lo:
            return p
    return 0


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def f1(pred, true):
    p, t = collections.Counter(pred), collections.Counter(true)
    ov = sum((p & t).values())
    if not ov:
        return 0.0
    pr, rc = ov / sum(p.values()), ov / sum(t.values())
    return 2 * pr * rc / (pr + rc)


def main():
    vocab = _vs.load_vocabularies(REF / "vocabularies.json")

    print("=== FORMAT VALIDATION (organizers' script, all files) ===")
    files = sorted(SUB.glob("*_sequence.json"))
    bad = []
    for f in files:
        errs = _vs.validate_submission(f, "easy", vocab)
        if errs:
            bad.append((f.name, errs))
    print(f"  files checked : {len(files):,}")
    print(f"  VALID         : {len(files)-len(bad):,}")
    print(f"  INVALID       : {len(bad):,}")
    for name, errs in bad[:5]:
        print(f"    {name}: {errs[0]}")
    if bad:
        print("\n  -> fix format before scoring")
        return 1

    # ---- ground truth in (o1,o2) terms ------------------------------------
    o1o2 = {k: tuple(v) for k, v in
            json.load(open(os.path.join(DER, "o1o2_map.json"))).items()}
    truth = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DER, "operations.csv"))):
        truth[r["part_id"]].append((int(r["seq"]), r["base_name"]))
    for v in truth.values():
        v.sort()

    val = set(load_model()["val_ids"])

    print("\n=== EASY TRACK SCORE (held-out parts only) ===")
    levs, f1s, pts = [], [], []
    n_missing = 0
    for f in files:
        pid = f.name[: -len("_sequence.json")]
        if pid not in val:
            continue
        g = [o1o2.get(n, ("OTHER", "OTHER")) for _, n in truth.get(pid, [])]
        if not g:
            n_missing += 1
            continue
        d = json.load(open(f))
        p = [(o["o1"], o["o2"]) for o in d["operations"]]
        lv = edit_distance(p, g) / max(len(p), len(g)) if max(len(p), len(g)) else 0.0
        ff = f1(p, g)
        levs.append(lv); f1s.append(ff)
        pts.append(lev_points(lv) + f1_points(ff))

    n = len(levs)
    mlev, mf1 = sum(levs) / n, sum(f1s) / n
    print(f"  parts scored                : {n:,}")
    print(f"  mean normalized Levenshtein : {mlev:.4f}   (lower is better)")
    print(f"  mean F1                     : {mf1:.4f}")
    print()
    print(f"  points from Levenshtein     : {lev_points(mlev)}/10")
    print(f"  points from F1              : {f1_points(mf1)}/10")
    print(f"  EASY TRACK TOTAL            : "
          f"{lev_points(mlev)+f1_points(mf1)}/20")
    print()
    print(f"  (mean of per-part points    : {sum(pts)/n:.2f}/20 -- shown for"
          f" reference; the rubric's aggregation method is not specified)")

    # distance to the next band up
    for hi, p_ in LEV_BANDS:
        if mlev <= hi:
            print(f"\n  Levenshtein is {mlev:.4f}; next band up needs "
                  f"<= {[b for b in [0.1,0.2,0.3,0.4,0.5] if b < hi][-1] if hi>0.1 else 0.1:.2f}")
            break
    for lo, p_ in F1_BANDS:
        if mf1 >= lo:
            nxt = {10: None, 8: 0.95, 6: 0.85, 4: 0.75, 2: 0.65, 0: 0.55}[p_]
            if nxt:
                print(f"  F1 is {mf1:.4f}; next band up needs >= {nxt:.2f} "
                      f"(+{ (0.0 if nxt is None else 2) } points)")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
