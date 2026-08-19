import collections, csv, math, os, re, sys

DER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derived")

RE_TYPE = re.compile(r"DB\(Type\)\s*==\s*(\d+)")
RE_SUBTYPE = re.compile(r"DB\(SubType\)\s*==\s*(\d+)")
RE_DIA = re.compile(r"DB\(Diameter\)\s*(>=|<=|>|<|==)\s*([\d.]+)")
RE_OTHER = re.compile(r"DB\((\w+)\)")


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
    ops = list(csv.DictReader(open(os.path.join(DER, "op_details.csv"))))
    tools = {t["tool"]: t for t in
             csv.DictReader(open(os.path.join(DER, "tools.csv")))}
    ops = [o for o in ops if o["tool_query"]]
    print(f"operations with a recorded query: {len(ops):,}")

    H_tool = entropy(collections.Counter(o["tool"] for o in ops))
    h_q = cond_entropy([(o["tool_query"], o["tool"]) for o in ops])
    print(f"\n=== IS THE TOOL DETERMINED BY THE QUERY? ===")
    print(f"  H(tool)          = {H_tool:.4f} bits")
    print(f"  H(tool | query)  = {h_q:.4f} bits")
    byq = collections.defaultdict(set)
    for o in ops:
        byq[o["tool_query"]].add(o["tool"])
    amb = {q: t for q, t in byq.items() if len(t) > 1}
    print(f"  distinct queries : {len(byq):,}")
    print(f"  queries mapping to >1 tool: {len(amb):,} "
          f"({100*len(amb)/len(byq):.2f}%)")
    if amb:
        q, t = next(iter(amb.items()))
        print(f"    e.g. {len(t)} tools: {sorted(t)[:6]}")

    preds = collections.Counter()
    for o in ops:
        preds.update(set(RE_OTHER.findall(o["tool_query"])))
    print(f"\n=== PREDICATES USED IN RULES ===")
    for k, c in preds.most_common():
        print(f"  {c:8,}  {100*c/len(ops):5.1f}%  DB({k})")

    print(f"\n=== (Type, SubType) CLASS CODE -> tool_type ===")
    cls = collections.defaultdict(collections.Counter)
    for o in ops:
        t, s = RE_TYPE.search(o["tool_query"]), RE_SUBTYPE.search(o["tool_query"])
        if t and s:
            tt = tools.get(o["tool"], {}).get("tool_type", "?")
            cls[(t.group(1), s.group(1))][tt] += 1
    for k in sorted(cls):
        tot = sum(cls[k].values())
        kinds = ", ".join(f"{n} ({c:,})" for n, c in cls[k].most_common(3))
        print(f"  Type={k[0]} SubType={k[1]}  n={tot:7,}  -> {kinds}")

    print(f"\n=== DIAMETER CONSTRAINT vs CHOSEN TOOL DIAMETER ===")
    exact = near = tot = 0
    gaps = []
    for o in ops:
        bounds = RE_DIA.findall(o["tool_query"])
        td = tools.get(o["tool"], {}).get("diameter")
        if not bounds or not td:
            continue
        td = float(td)
        ups = [float(v) for op_, v in bounds if op_ in ("<=", "<")]
        if not ups:
            continue
        tot += 1
        hi = min(ups)
        if abs(hi - td) < 1e-6:
            exact += 1
        elif abs(hi - td) < 0.5:
            near += 1
        gaps.append(hi - td)
    if tot:
        gaps.sort()
        print(f"  ops with an upper diameter bound: {tot:,}")
        print(f"    bound == tool diameter exactly : {exact:,} "
              f"({100*exact/tot:.1f}%)")
        print(f"    within 0.5 mm                  : {near:,} "
              f"({100*near/tot:.1f}%)")
        print(f"    gap (bound - tool_dia) median  : {gaps[len(gaps)//2]:.4f} mm")

    print(f"\n=== FEATURE / OPERATION -> TOOL ===")
    def feat(g):
        return re.sub(r"_\d+$", "", g or "")
    for label, key in [
        ("geometry group (feature)", lambda o: feat(o["geometry_group"])),
        ("operation name", lambda o: re.sub(r"_\d+$", "", o["op_name"] or "")),
        ("feature + operation", lambda o: (feat(o["geometry_group"]),
                                           re.sub(r"_\d+$", "", o["op_name"] or ""))),
    ]:
        h = cond_entropy([(key(o), o["tool"]) for o in ops])
        print(f"  H(tool | {label:<24s}) = {h:.3f} bits  "
              f"({100*(1-h/H_tool):.1f}% reduction)")

    print(f"\n=== FEATURE FAMILIES (suffix stripped) ===")
    fam = collections.Counter(feat(o["geometry_group"]) for o in ops)
    for k, c in fam.most_common():
        print(f"  {c:8,}  {100*c/len(ops):5.1f}%  {k}")


if __name__ == "__main__":
    main()
