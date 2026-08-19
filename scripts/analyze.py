import collections, csv, glob, json, math, os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/vasusilawat/Desktop/stpd/data/MachinePlan-10K"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derived")

SUFFIX = re.compile(r"_\d+$")


def base_name(n):
    return SUFFIX.sub("", n)


def entropy(counter):
    tot = sum(counter.values())
    return -sum((c / tot) * math.log2(c / tot) for c in counter.values() if c)


def cond_entropy(pairs):
    by_x = collections.defaultdict(collections.Counter)
    for x, y in pairs:
        by_x[x][y] += 1
    n = len(pairs)
    return sum(sum(c.values()) / n * entropy(c) for c in by_x.values())


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*_operations.json")))
    print(f"found {len(files):,} process plans")
    if not files:
        sys.exit("no *_operations.json found — check the extraction path")

    rows = []
    per_part = []
    for f in files:
        pid = os.path.basename(os.path.dirname(f))
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  !! unreadable {pid}: {e}")
            continue
        s = d.get("machining_summary", {})
        ops = d.get("operations", [])
        per_part.append({
            "part_id": pid, "num_operations": s.get("num_operations", len(ops)),
            "tool_changes": s.get("tool_changes"),
            "total_toolpath_time_min": s.get("total_toolpath_time_min"),
            "total_cutting_time_min": s.get("total_cutting_time_min"),
        })
        for o in ops:
            rows.append({
                "part_id": pid,
                "seq": o.get("sequence_number"),
                "name": o.get("name"),
                "base_name": base_name(o.get("name", "")),
                "type": o.get("type"),
                "tool": o.get("tool_name"),
                "time_min": o.get("toolpath_time_min"),
                "cut_time_min": o.get("toolpath_cutting_time_min"),
                "path_mm": o.get("toolpath_length_mm"),
                "cut_path_mm": o.get("toolpath_cutting_length_mm"),
                "vol_after_mm3": o.get("volume_mm3"),
                "vol_removed_mm3": o.get("volume_removed_mm3"),
            })

    print(f"total operations: {len(rows):,}")

    with open(os.path.join(OUT, "operations.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "parts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_part[0].keys()))
        w.writeheader(); w.writerows(per_part)

    def hist(key, label, top=None):
        c = collections.Counter(r[key] for r in rows)
        print(f"\n=== {label}: {len(c)} distinct ===")
        for k, v in c.most_common(top):
            print(f"  {v:8,}  {100*v/len(rows):5.2f}%  {k}")
        return c

    hist("type", "OPERATION TYPE")
    hist("base_name", "OPERATION (base name, suffix stripped)", 40)
    tools = hist("tool", "TOOL", 30)

    print(f"\n  raw 'name' values (unstripped): "
          f"{len(set(r['name'] for r in rows)):,}")

    print("\n=== PREDICTABILITY (bits; 0 = fully determined) ===")
    H_tool = entropy(tools)
    print(f"  H(tool)                    = {H_tool:.3f}")
    for cond in ("type", "base_name"):
        h = cond_entropy([(r[cond], r["tool"]) for r in rows])
        print(f"  H(tool | {cond:<10s})      = {h:.3f}   "
              f"(reduces {100*(1-h/H_tool):.1f}%)")
    h = cond_entropy([((r["base_name"], r["type"]), r["tool"]) for r in rows])
    print(f"  H(tool | name,type)        = {h:.3f}")

    by_op = collections.defaultdict(collections.Counter)
    for r in rows:
        by_op[r["base_name"]][r["tool"]] += 1
    det = [k for k, c in by_op.items() if len(c) == 1]
    print(f"\n  operations with exactly ONE tool ever: {len(det)}/{len(by_op)}")

    print("\n=== ORDERING STRUCTURE ===")
    by_part = collections.defaultdict(list)
    for r in rows:
        by_part[r["part_id"]].append(r)
    for v in by_part.values():
        v.sort(key=lambda r: r["seq"])

    type_first = collections.Counter()
    pair = collections.Counter()
    for v in by_part.values():
        ts = [r["type"] for r in v]
        type_first[ts[0]] += 1
        for a, b in zip(ts, ts[1:]):
            if a != b:
                pair[(a, b)] += 1
    print("  first operation type:")
    for k, c in type_first.most_common():
        print(f"    {c:6,}  {100*c/len(by_part):5.1f}%  {k}")
    print("  observed type transitions (a -> b, a!=b):")
    for (a, b), c in pair.most_common(20):
        print(f"    {c:6,}  {a} -> {b}")

    grouped = sum(
        1 for v in by_part.values()
        if len({r["type"] for r in v}) ==
           len([1 for i, r in enumerate(v)
                if i == 0 or r["type"] != v[i-1]["type"]]))
    print(f"\n  parts where each type forms ONE contiguous block: "
          f"{grouped:,}/{len(by_part):,} ({100*grouped/len(by_part):.1f}%)")

    bad = sum(1 for v in by_part.values()
              if any(a["vol_after_mm3"] is not None and b["vol_after_mm3"] is not None
                     and b["vol_after_mm3"] > a["vol_after_mm3"] + 1e-6
                     for a, b in zip(v, v[1:])))
    print(f"  parts with non-monotone IPW volume: {bad:,} (expect 0)")

    ncnt = collections.Counter(p["num_operations"] for p in per_part)
    print("\n=== OPERATIONS PER PART ===")
    ks = sorted(ncnt)
    print(f"  min {ks[0]}  max {ks[-1]}  mean "
          f"{sum(p['num_operations'] for p in per_part)/len(per_part):.2f}")
    for k in ks:
        print(f"    {k:3d} ops: {'#'*max(1,ncnt[k]//20)} {ncnt[k]}")

    print(f"\nwrote {OUT}/operations.csv and parts.csv")


if __name__ == "__main__":
    main()
