import collections, csv, glob, json, os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/vasusilawat/Desktop/stpd/data/MachinePlan-10K"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derived")

NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
FIELDS = {
    "diameter":     re.compile(r"\(D\) Diameter\s*=\s*" + NUM),
    "corner_radius": re.compile(r"\(CR\) Corner Radius\s*=\s*" + NUM),
    "length":       re.compile(r"\(L\) Length\s*=\s*" + NUM),
    "point_angle":  re.compile(r"\(PA\) Point Angle\s*=\s*" + NUM),
    "point_length": re.compile(r"\(PL\) Point Length\s*=\s*" + NUM),
    "flute_length": re.compile(r"\(FL\) Flute Length\s*=\s*" + NUM),
    "taper_angle":  re.compile(r"\(TA\) Taper Angle\s*=\s*" + NUM),
    "num_flutes":   re.compile(r"Number of Flute\s+(\d+)"),
}
RE_TOOL_SUB = re.compile(r"Template Subtype:\s*(.+?)\s*$", re.M)
RE_TOOL_TYPE = re.compile(r"Tool Type :\s*(.+?)\s*$", re.M)
RE_TOOL_MAT = re.compile(r"Tool Material Name :\s*(.+?)\s*$", re.M)
RE_OPTYPE = re.compile(r"Operation Type\s+(.+?)\s*$", re.M)
RE_OPNAME = re.compile(r"Operation Name\s+(.+?)\s*$", re.M)
RE_TEMPLATE = re.compile(r"Template Type:\s*(.+?)\s*$", re.M)
RE_GEOMGRP = re.compile(r"Geometry Group\s+(.+?)\s*$", re.M)
RE_METHOD = re.compile(r"Method Group\s+(.+?)\s*$", re.M)
RE_ORDER = re.compile(r"Order Group\s+(.+?)\s*$", re.M)
RE_FEED_CUT = re.compile(r"^Cut\s+" + NUM + r"\s*\(MMPM\)", re.M)
RE_STOCK_PART = re.compile(r"Part Stock\s*=\s*" + NUM)
RE_STOCK_FLOOR = re.compile(r"Floor Stock\s*=\s*" + NUM)
RE_INTOL = re.compile(r"Intol\s*=\s*" + NUM)
RE_QUERY = re.compile(r"-+ Tool Query -+\s*\nQuery\s+(.*?)(?=\n\s*\n|\n-{5})",
                      re.S)


def g(rx, txt, cast=float):
    m = rx.search(txt)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except ValueError:
        return None


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*_details.txt")))
    files = [f for f in files
             if not os.path.basename(f).startswith("workpiece")]
    print(f"found {len(files):,} operation cards")
    if not files:
        sys.exit("no operation *_details.txt found — check the path")

    tools = {}
    op_rows = []
    queries = collections.Counter()
    for i, f in enumerate(files, 1):
        txt = open(f, errors="replace").read()
        pid = os.path.basename(os.path.dirname(f))
        stem = os.path.basename(f)[: -len("_details.txt")]
        seq, tool = stem.split("_", 1)

        tblock = txt.split("Tool  Information", 1)
        tb = tblock[1] if len(tblock) > 1 else txt
        rec = {k: g(rx, tb) for k, rx in FIELDS.items()}
        rec["tool"] = tool
        m = RE_TOOL_SUB.search(tb)
        rec["catalog_desc"] = m.group(1) if m else None
        m = RE_TOOL_TYPE.search(tb)
        rec["tool_type"] = m.group(1) if m else None
        m = RE_TOOL_MAT.search(tb)
        rec["tool_material"] = m.group(1) if m else None
        prev = tools.get(tool)
        if prev is None:
            tools[tool] = rec
        elif prev != rec:
            tools.setdefault(tool + "  [INCONSISTENT]", rec)

        q = RE_QUERY.search(txt)
        qs = re.sub(r"[ \t]+", " ",
                    re.sub(r"\n", "", q.group(1))).strip() if q else None
        if qs:
            queries[qs] += 1

        op_rows.append({
            "part_id": pid, "seq": int(seq), "tool": tool,
            "op_name": (RE_OPNAME.search(txt).group(1)
                        if RE_OPNAME.search(txt) else None),
            "op_type": (RE_OPTYPE.search(txt).group(1)
                        if RE_OPTYPE.search(txt) else None),
            "template": (RE_TEMPLATE.search(txt).group(1)
                         if RE_TEMPLATE.search(txt) else None),
            "order_group": (RE_ORDER.search(txt).group(1)
                            if RE_ORDER.search(txt) else None),
            "method_group": (RE_METHOD.search(txt).group(1)
                             if RE_METHOD.search(txt) else None),
            "geometry_group": (RE_GEOMGRP.search(txt).group(1)
                               if RE_GEOMGRP.search(txt) else None),
            "feed_cut_mmpm": g(RE_FEED_CUT, txt),
            "part_stock_mm": g(RE_STOCK_PART, txt),
            "floor_stock_mm": g(RE_STOCK_FLOOR, txt),
            "intol_mm": g(RE_INTOL, txt),
            "tool_query": qs,
        })
        if i % 10000 == 0:
            print(f"  parsed {i:,}", flush=True)

    with open(os.path.join(OUT, "tools.csv"), "w", newline="") as fh:
        cols = ["tool", "tool_type", "catalog_desc", "tool_material",
                "diameter", "corner_radius", "length", "flute_length",
                "point_angle", "point_length", "taper_angle", "num_flutes"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in sorted(tools):
            w.writerow(tools[t])

    with open(os.path.join(OUT, "op_details.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(op_rows[0].keys()))
        w.writeheader(); w.writerows(op_rows)

    print(f"\n=== TOOL LIBRARY: {len(tools)} distinct tools ===")
    kinds = collections.Counter(v["tool_type"] for v in tools.values())
    for k, c in kinds.most_common():
        print(f"  {c:5d}  {k}")

    print("\n=== ORDER GROUPS (NX's own operation ordering buckets) ===")
    for k, c in collections.Counter(r["order_group"] for r in op_rows).most_common():
        print(f"  {c:8,}  {k}")
    print("\n=== METHOD GROUPS ===")
    for k, c in collections.Counter(r["method_group"] for r in op_rows).most_common():
        print(f"  {c:8,}  {k}")
    print("\n=== GEOMETRY GROUPS (top 25) — the machining FEATURE each op targets ===")
    gg = collections.Counter(r["geometry_group"] for r in op_rows)
    print(f"  {len(gg)} distinct")
    for k, c in gg.most_common(25):
        print(f"  {c:8,}  {k}")
    print(f"\n=== TOOL QUERIES: {len(queries):,} distinct selection rules ===")
    for k, c in queries.most_common(5):
        print(f"  {c:8,}  {k[:150]}")

    print(f"\nwrote {OUT}/tools.csv and op_details.csv")


if __name__ == "__main__":
    main()
