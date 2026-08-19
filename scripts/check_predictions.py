import collections, csv, glob, json, math, os, re, sys

ROOT = "/Users/vasusilawat/Desktop/stpd/data/MachinePlan-10K"
DER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derived")
PRED = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DER, "predictions")

strip = lambda s: re.sub(r"_\d+$", "", s or "")
fails = []


def check(name, bad, total, examples=()):
    ok = not bad
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {name}"
    if total is not None:
        line += f"  ({total - bad:,}/{total:,} ok)"
    print(line)
    for e in list(examples)[:3]:
        print(f"           e.g. {e}")
    if not ok:
        fails.append(name)


def main():
    print(f"checking {PRED}\n")

    true_names, true_types, true_tools = set(), set(), set()
    for r in csv.DictReader(open(os.path.join(DER, "operations.csv"))):
        true_names.add(r["base_name"]); true_types.add(r["type"])
        true_tools.add(r["tool"])
    name_type = {}
    for r in csv.DictReader(open(os.path.join(DER, "operations.csv"))):
        name_type.setdefault(r["base_name"], r["type"])

    expected = {d for d in os.listdir(ROOT) if d.startswith("featured_part_")}
    files = sorted(glob.glob(os.path.join(PRED, "*_operations.json")))
    print(f"prediction files: {len(files):,}   dataset parts: {len(expected):,}\n")

    print("=== FILE COVERAGE ===")
    got = {os.path.basename(f)[: -len("_operations.json")] for f in files}
    missing, extra = expected - got, got - expected
    check("every dataset part has a prediction", len(missing), len(expected),
          sorted(missing))
    check("no unexpected prediction files", len(extra), len(got), sorted(extra))

    print("\n=== STRUCTURE ===")
    docs, unreadable = {}, []
    for f in files:
        try:
            docs[os.path.basename(f)[: -len("_operations.json")]] = json.load(open(f))
        except Exception as e:
            unreadable.append(f"{os.path.basename(f)}: {e}")
    check("valid JSON", len(unreadable), len(files), unreadable)

    bad_keys, bad_seq, bad_count, empty = [], [], [], []
    bad_name, bad_type, bad_tool, bad_time = [], [], [], []
    bad_typemap, bad_suffix, bad_total, bad_changes = [], [], [], []
    nops = []

    for pid, d in docs.items():
        if not {"machining_summary", "operations"} <= set(d):
            bad_keys.append(pid); continue
        ops, s = d["operations"], d["machining_summary"]
        nops.append(len(ops))
        if not ops:
            empty.append(pid); continue
        if [o.get("sequence_number") for o in ops] != list(range(len(ops))):
            bad_seq.append(pid)
        if s.get("num_operations") != len(ops):
            bad_count.append(f"{pid}: says {s.get('num_operations')}, has {len(ops)}")

        seen = collections.Counter()
        tot = 0.0
        tools_seq = []
        for o in ops:
            n, t, tl = o.get("name"), o.get("type"), o.get("tool_name")
            base = strip(n)
            if base not in true_names:
                bad_name.append(f"{pid}: {n!r}")
            if t not in true_types:
                bad_type.append(f"{pid}: {t!r}")
            elif base in name_type and name_type[base] != t:
                bad_typemap.append(f"{pid}: {base} typed {t}, dataset says "
                                   f"{name_type[base]}")
            if tl not in true_tools:
                bad_tool.append(f"{pid}: {tl!r}")
            tm = o.get("toolpath_time_min")
            if tm is None or not isinstance(tm, (int, float)) or \
               math.isnan(tm) or tm < 0:
                bad_time.append(f"{pid}: {n} time={tm!r}")
            else:
                tot += tm
            want = base if seen[base] == 0 else f"{base}_{seen[base]}"
            if n != want:
                bad_suffix.append(f"{pid}: got {n!r}, expected {want!r}")
            seen[base] += 1
            tools_seq.append(tl)

        if abs((s.get("total_toolpath_time_min") or 0) - tot) > 0.01:
            bad_total.append(f"{pid}: says {s.get('total_toolpath_time_min')}, "
                             f"sums to {round(tot, 4)}")
        ch = sum(1 for a, b in zip(tools_seq, tools_seq[1:]) if a != b)
        if s.get("tool_changes") != ch:
            bad_changes.append(f"{pid}: says {s.get('tool_changes')}, computed {ch}")

    n = len(docs)
    check("required top-level keys present", len(bad_keys), n, bad_keys)
    check("no empty plans", len(empty), n, empty)
    check("sequence_number is 0..n-1 contiguous", len(bad_seq), n, bad_seq)
    check("num_operations matches operations list", len(bad_count), n, bad_count)

    print("\n=== VOCABULARY (must exist in the real dataset) ===")
    check("operation names are known", len(bad_name), None, bad_name)
    check("operation types are known", len(bad_type), None, bad_type)
    check("name -> type mapping matches dataset", len(bad_typemap), None,
          bad_typemap)
    check("tool names exist in the 431-tool library", len(bad_tool), None,
          bad_tool)

    print("\n=== INTERNAL ARITHMETIC ===")
    check("per-step times present and non-negative", len(bad_time), None, bad_time)
    check("total_toolpath_time_min equals sum of steps", len(bad_total), n,
          bad_total)
    check("tool_changes matches the tool sequence", len(bad_changes), n,
          bad_changes)
    check("name suffixing follows NX convention", len(bad_suffix), None,
          bad_suffix)

    print("\n=== PLAUSIBILITY vs REAL DATA ===")
    real = [int(r["num_operations"]) for r in
            csv.DictReader(open(os.path.join(DER, "parts.csv")))]
    nops.sort(); real.sort()
    def pct(a, q):
        return a[min(len(a) - 1, int(q * len(a)))]
    print(f"  steps per part  ours: min {nops[0]}  median {pct(nops,.5)}  "
          f"p90 {pct(nops,.9)}  max {nops[-1]}   mean {sum(nops)/len(nops):.2f}")
    print(f"                  real: min {real[0]}  median {pct(real,.5)}  "
          f"p90 {pct(real,.9)}  max {real[-1]}   mean {sum(real)/len(real):.2f}")
    print(f"  total steps     ours: {sum(nops):,}   real: {sum(real):,}  "
          f"({100*sum(nops)/sum(real):.1f}%)")

    print()
    if fails:
        print(f"RESULT: {len(fails)} check(s) FAILED")
        for f_ in fails:
            print(f"  - {f_}")
        sys.exit(1)
    print("RESULT: all internal consistency checks PASSED")
    print("NOTE: this does not verify the organizers' submission format,")
    print("      which we do not have.")


if __name__ == "__main__":
    main()
