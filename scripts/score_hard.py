import collections, csv, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import DER, load_model
from export_hard import CODE, RE_TOOL, tool_tables

SUB = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(DER, "..", "submission", "hard")


def band(rel):
    if rel <= 0.02:
        return 10
    if rel <= 0.05:
        return 8
    if rel <= 0.10:
        return 6
    if rel <= 0.20:
        return 3
    return 1


def main():
    dia, cat = tool_tables()

    truth = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DER, "operations.csv"))):
        truth[r["part_id"]].append((int(r["seq"]), r["tool"]))
    for v in truth.values():
        v.sort()

    val = set(load_model()["val_ids"])

    tot_pts = tot_ops = 0
    type_ok = 0
    dia_rel = []
    miss_len = 0
    by_type = collections.defaultdict(lambda: [0, 0])

    for f in sorted(glob.glob(os.path.join(SUB, "*_tools.json"))):
        pid = os.path.basename(f)[: -len("_tools.json")]
        if pid not in val:
            continue
        g = truth.get(pid, [])
        if not g:
            continue
        pred = json.load(open(f))["operations"]
        n = max(len(pred), len(g))
        miss_len += abs(len(pred) - len(g))
        for i in range(n):
            tot_ops += 1
            if i >= len(pred) or i >= len(g):
                continue
            gt_tool = g[i][1]
            gt_type = cat.get(gt_tool)
            gt_dia = dia.get(gt_tool)
            p = pred[i]
            by_type[gt_type][1] += 1
            if gt_type is None or gt_dia is None:
                continue
            if p["tool_type"] != gt_type:
                continue
            type_ok += 1
            by_type[gt_type][0] += 1
            rel = abs(p["tool_diameter_mm"] - gt_dia) / gt_dia
            dia_rel.append(rel)
            tot_pts += band(rel)

    print(f"=== HARD TRACK: TOOL SELECTION (held-out parts) ===")
    print(f"  operation slots compared : {tot_ops:,}")
    print(f"  tool TYPE correct        : {type_ok:,} "
          f"({100*type_ok/tot_ops:.1f}%)")
    if dia_rel:
        dia_rel.sort()
        within = lambda t: 100 * sum(1 for r in dia_rel if r <= t) / len(dia_rel)
        print(f"  of those, diameter within 2%  : {within(.02):.1f}%")
        print(f"                          5%  : {within(.05):.1f}%")
        print(f"                         10%  : {within(.10):.1f}%")
        print(f"  median relative diameter error: "
              f"{dia_rel[len(dia_rel)//2]*100:.2f}%")
    print()
    mean = tot_pts / tot_ops if tot_ops else 0
    print(f"  MEAN POINTS PER OPERATION: {mean:.2f}/10")
    print(f"  -> scaled to the 20-point tool-selection budget: "
          f"{2*mean:.1f}/20")
    print(f"     (the rubric's table maxes at 10 per operation but the section")
    print(f"      is worth 20; the exact aggregation is unspecified)")

    print("\n  accuracy by true tool type:")
    for t, (a, b) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        if b:
            print(f"    {str(t):14s} {a:6,}/{b:6,}  ({100*a/b:5.1f}%)")


if __name__ == "__main__":
    main()
