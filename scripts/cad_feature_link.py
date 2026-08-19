import collections, csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_parse import parse

ROOT = "/Users/vasusilawat/Desktop/stpd/data/MachinePlan-10K"
DER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "derived")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

strip = lambda s: re.sub(r"_\d+$", "", s or "")


def main():
    ops = list(csv.DictReader(open(os.path.join(DER, "op_details.csv"))))
    tools = {t["tool"]: t for t in
             csv.DictReader(open(os.path.join(DER, "tools.csv")))}

    by_part = collections.defaultdict(list)
    for o in ops:
        by_part[o["part_id"]].append(o)

    parts = sorted(by_part)[:N]
    print(f"checking {len(parts):,} parts\n")

    rows = []
    for pid in parts:
        f = os.path.join(ROOT, pid, pid + ".stp")
        if not os.path.exists(f):
            continue
        try:
            p = parse(f)
        except Exception as e:
            print(f"  !! parse failed {pid}: {e}")
            continue
        o = by_part[pid]
        gg = {g["geometry_group"] for g in o}
        n_chamfer_feat = len({g for g in gg if strip(g) == "FG_CHAMFER_SURFACE"})
        n_hole_feat = len({g for g in gg if strip(g) == "STEP1HOLE"})
        n_na_plane = sum(1 for x in p.faces
                         if x["kind"] == "plane" and x.get("axis_aligned") is False)
        radii = sorted(f["radius"] for f in p.hole_cylinders())
        fillets = sorted(f["radius"] for f in p.fillet_cylinders())
        tds = sorted({float(tools[g["tool"]]["diameter"])
                      for g in o if tools.get(g["tool"], {}).get("diameter")})
        rows.append({
            "pid": pid, "faces": len(p.faces),
            "na_plane": n_na_plane, "chamfer_feat": n_chamfer_feat,
            "cyl": len(radii), "n_cyl_dia": len(set(radii)),
            "fillet": len(fillets), "n_fillet_dia": len(set(fillets)),
            "hole_feat": n_hole_feat,
            "dias": [round(2 * r, 3) for r in radii],
            "fillet_dias": [round(2 * r, 3) for r in fillets],
            "tool_dias": tds,
        })

    print(f"parsed {len(rows):,} parts OK\n")

    print("=== H1: non-axis-aligned planes <-> chamfer features ===")
    agree = sum(1 for r in rows if r["na_plane"] == r["chamfer_feat"])
    both0 = sum(1 for r in rows if r["na_plane"] == 0 == r["chamfer_feat"])
    print(f"  exact match: {agree:,}/{len(rows):,} ({100*agree/len(rows):.1f}%)"
          f"   [of which both zero: {both0:,}]")
    d = collections.Counter(r["na_plane"] - r["chamfer_feat"] for r in rows)
    print("  (n_planes - n_chamfer_features):",
          "  ".join(f"{k:+d}:{v}" for k, v in sorted(d.items())[:9]))

    print("\n=== H2: CLOSED cylindrical faces <-> hole features ===")
    agree2 = sum(1 for r in rows if r["cyl"] == r["hole_feat"])
    print(f"  closed-cyl count == n hole features: "
          f"{agree2:,}/{len(rows):,} ({100*agree2/len(rows):.1f}%)")
    d = collections.Counter(r["cyl"] - r["hole_feat"] for r in rows)
    print("  (n_closed_cyl - n_hole_features):",
          "  ".join(f"{k:+d}:{v}" for k, v in sorted(d.items())[:11]))
    print(f"  partial (fillet) cylinders seen: "
          f"{sum(r['fillet'] for r in rows):,} across "
          f"{sum(1 for r in rows if r['fillet']):,} parts")

    print("\n=== H3: cylinder diameters vs tool diameters used ===")
    hit = tot = 0
    errs = []
    for r in rows:
        for d_ in set(r["dias"]):
            if not r["tool_dias"]:
                continue
            tot += 1
            best = min(r["tool_dias"], key=lambda t: abs(t - d_))
            errs.append(d_ - best)
            if abs(d_ - best) < 0.51:
                hit += 1
    if tot:
        errs.sort()
        print(f"  cylinder diameters within 0.5 mm of a tool used: "
              f"{hit:,}/{tot:,} ({100*hit/tot:.1f}%)")
        print(f"  signed error (cyl_dia - nearest_tool_dia): "
              f"p10 {errs[int(.1*len(errs))]:+.3f}  median "
              f"{errs[len(errs)//2]:+.3f}  p90 {errs[int(.9*len(errs))]:+.3f} mm")

    print("\n=== B-REP SIZE ===")
    fc = sorted(r["faces"] for r in rows)
    print(f"  faces per part: min {fc[0]}  median {fc[len(fc)//2]}  "
          f"p95 {fc[int(.95*len(fc))]}  max {fc[-1]}")


if __name__ == "__main__":
    main()
