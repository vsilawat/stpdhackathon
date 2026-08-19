#!/usr/bin/env python3
"""Mine per-feature operation chains from derived/opdetails.csv to derive chain-prediction rules.
Groups rows by (part, geometry_group) -- one machined feature -- and characterizes the op chain,
tool sequence and diameters for each feature family (holes, pockets, chamfers, spot drills, boring).
Writes derived/chains.csv and prints the analysis used to build the decision rule set."""
import csv, json, math, re, statistics as st
from collections import Counter, defaultdict

OPDETAILS, SEQUENCES = 'derived/opdetails.csv', 'derived/sequences.jsonl'
OUT_CSV = 'derived/chains.csv'

def header(title): print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")

def base(name): return re.sub(r'_\d+$', '', name)
def gprefix(g): return re.sub(r'_\d+$', '', g)
def load_rows(path=OPDETAILS): return list(csv.DictReader(open(path)))

def load_volumes(path=SEQUENCES):
    vols = {}
    for line in open(path):
        d = json.loads(line)
        vols[d['part']] = {o['full']: o.get('vol', 0.0) for o in d['ops']}
    return vols

def group_rows(rows):
    groups = defaultdict(list)
    for r in rows: groups[(r['part'], r['geometry_group'])].append(r)
    for rs in groups.values(): rs.sort(key=lambda x: int(x['op_index']))
    return groups

def nonspot(rs): return [r for r in rs if not r['op_name'].startswith('SPOT_DRILL')]
def final_diameter(rs): return float((nonspot(rs) or rs)[-1]['tool_diameter_mm'])
def has_op(rs, sub): return any(sub in r['op_name'] for r in rs)
def depth_proxy(rs, vols, part):
    fd = final_diameter(rs)
    tot = sum(vols.get(part, {}).get(r['op_name'], 0.0) for r in rs)
    return tot / (math.pi / 4 * fd * fd) if fd else 0.0

def build_chain_rows(groups):
    out = []
    for (part, gg), rs in groups.items():
        out.append({
            'part': part, 'geometry_group': gg, 'group_prefix': gprefix(gg),
            'chain': '>'.join(base(r['op_name']) for r in rs),
            'tool_types': '>'.join(r['tool_type'] for r in rs),
            'tool_diams': '>'.join(r['tool_diameter_mm'] for r in rs),
            'first_op_index': rs[0]['op_index'], 'last_op_index': rs[-1]['op_index'],
            'final_diameter': f"{final_diameter(rs):.3f}",
        })
    return out

def write_csv(out, path=OUT_CSV):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

def stats_line(label, xs):
    xs = [x for x in xs if x is not None]
    if not xs: return f"{label}: n=0"
    return f"{label}: n={len(xs)} min={min(xs):.2f} median={st.median(xs):.2f} max={max(xs):.2f}"

def diam_ratios(rs):
    ns = nonspot(rs)
    ds = [float(r['tool_diameter_mm']) for r in ns]
    return [round(ds[i] / ds[i + 1], 3) for i in range(len(ds) - 1) if ds[i + 1]]

# ---------- section a/b: STEP1HOLE chain characterization ----------
def analyze_holes(groups, vols):
    header("(a) STEP1HOLE chains: top chains, final_diameter/ratio, THROUGH/BLIND, CENTER/SOLID")
    hg = {k: v for k, v in groups.items() if gprefix(k[1]) == 'STEP1HOLE'}
    print(f"total STEP1HOLE groups: {len(hg)}")
    chains = defaultdict(list)
    for (part, gg), rs in hg.items():
        chains['>'.join(base(r['op_name']) for r in rs)].append((part, rs))
    ranked = sorted(chains.items(), key=lambda x: -len(x[1]))
    cum = 0
    for chain, items in ranked[:40]:
        cum += len(items)
        fds = [final_diameter(rs) for _, rs in items]
        ratios = [r for _, rs in items for r in diam_ratios(rs)]
        thru = sum(1 for _, rs in items if has_op(rs, 'THROUGH'))
        blind = sum(1 for _, rs in items if has_op(rs, 'BLIND'))
        center = sum(1 for _, rs in items if has_op(rs, 'CENTER'))
        solid = sum(1 for _, rs in items if has_op(rs, 'FROM_SOLID'))
        print(f"\n[{len(items):5d}] (cum {cum}/{len(hg)} = {100*cum/len(hg):.1f}%) {chain}")
        print(f"   {stats_line('final_diameter', fds)}")
        print(f"   THROUGH={thru} BLIND={blind} CENTER={center} FROM_SOLID={solid}")
        if ratios: print(f"   {stats_line('pilot/next diameter ratio', ratios)}")

    header("(b) Is chain a deterministic function of final_diameter? (rounded to 0.1mm)")
    byd = defaultdict(Counter)
    for (part, gg), rs in hg.items():
        d = round(final_diameter(rs), 1)
        byd[d]['>'.join(base(r['op_name']) for r in rs)] += 1
    ambiguous = sum(1 for d in byd if len(byd[d]) > 1)
    print(f"distinct final_diameter values: {len(byd)}, ambiguous (>1 chain): {ambiguous}")
    print("diameter breakpoints (dominant chain changes as diameter rises, min n=15):")
    prev_dom = None
    for d in sorted(byd):
        c = byd[d]; tot = sum(c.values())
        if tot < 15: continue
        dom, domn = c.most_common(1)[0]
        if dom != prev_dom:
            print(f"   d>={d:5.1f}: dominant -> {dom}  ({domn}/{tot} = {100*domn/tot:.0f}%)")
        prev_dom = dom

    header("(b2) Depth proxy (L/D = group volume / final-diam cross-section) vs GUN_DRILL usage")
    buckets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 10**6]
    tab = defaultdict(lambda: [0, 0])
    for (part, gg), rs in hg.items():
        ld = depth_proxy(rs, vols, part) / final_diameter(rs)
        gun = has_op(rs, 'GUN')
        for i in range(len(buckets) - 1):
            if buckets[i] <= ld < buckets[i + 1]:
                tab[(buckets[i], buckets[i + 1])][1 if gun else 0] += 1
                break
    for k in sorted(tab):
        nogun, gun = tab[k]; tot = nogun + gun
        hi = k[1] if k[1] < 10**6 else 'inf'
        pct = f"no_gun={100*nogun/tot:5.1f}%  gun={100*gun/tot:5.1f}%"
        print(f"   L/D {k[0]:>3}-{hi}: n={tot:5d}  {pct}")
    return hg

# ---------- section c: pocket-family groups ----------
def analyze_pockets(groups):
    header("(c) Pocket-family groups: STEP1POCKET, POCKET_*, SLOT_*, CORNER_NOTCH_*")
    skip = ('STEP1HOLE', 'FG_CHAMFER_SURFACE')
    prefixes = sorted({gprefix(gg) for (_, gg) in groups if gprefix(gg) not in skip})
    for prefix in prefixes:
        gs = {k: v for k, v in groups.items() if gprefix(k[1]) == prefix}
        if not gs: continue
        chains, nops, diams = Counter(), Counter(), []
        for rs in gs.values():
            chains['>'.join(base(r['op_name']) for r in rs)] += 1
            nops[len(rs)] += 1
            diams.append(final_diameter(rs))
        print(f"\n-- {prefix}: n_groups={len(gs)}  ops_per_group_hist={dict(sorted(nops.items()))}")
        for c, n in chains.most_common(8): print(f"     [{n:5d}] {c}")
        print(f"   {stats_line('final tool diameter', diams)}")
        if prefix == 'STEP1POCKET':
            byd = defaultdict(Counter)
            for rs in gs.values():
                byd[round(final_diameter(rs))]['>'.join(base(r['op_name']) for r in rs)] += 1
            print("   diameter breakpoints (dominant chain vs rounded-mm diameter, min n=15):")
            prev = None
            for d in sorted(byd):
                c = byd[d]; tot = sum(c.values())
                if tot < 15: continue
                dom, domn = c.most_common(1)[0]
                if dom != prev:
                    pct = f"{domn}/{tot}={100*domn/tot:.0f}%"
                    print(f"     d>={d:4.0f}mm: dominant -> {dom} ({pct})")
                prev = dom

# ---------- section d: FG_CHAMFER_SURFACE ----------
def analyze_chamfer(groups, rows):
    header("(d) FG_CHAMFER_SURFACE: ops/group, tool/diameter, chamfer groups vs AREA_MILL ops")
    cg = {k: v for k, v in groups.items() if gprefix(k[1]) == 'FG_CHAMFER_SURFACE'}
    nops = Counter(len(rs) for rs in cg.values())
    tool_types = Counter(r['tool_type'] for rs in cg.values() for r in rs)
    diams = Counter(round(float(r['tool_diameter_mm']), 1) for rs in cg.values() for r in rs)
    print(f"n_groups={len(cg)}  ops_per_group={dict(nops)}  tool_type={dict(tool_types)}")
    print(f"diam_dist={dict(diams)}")
    by_part = defaultdict(list)
    for r in rows: by_part[r['part']].append(r)
    mism, hist = 0, Counter()
    for part, rs in by_part.items():
        am = sum(1 for r in rs if base(r['op_name']) == 'AREA_MILL')
        ch = len({r['geometry_group'] for r in rs
                  if gprefix(r['geometry_group']) == 'FG_CHAMFER_SURFACE'})
        hist[(am, ch)] += 1
        if am != ch: mism += 1
    print(f"parts where AREA_MILL count != FG_CHAMFER_SURFACE group count: {mism}/{len(by_part)}")
    top_hist = dict(sorted(hist.items(), key=lambda x: -x[1])[:15])
    print(f"histogram of (AREA_MILL ops, chamfer groups) per part: {top_hist}")

# ---------- section e: SPOT_DRILLING ----------
def analyze_spot(hole_and_pocket_groups):
    header("(e) SPOT_DRILLING: which hole/pocket chains include a spot drill")
    has_spot_fd, no_spot_fd, ratios, lead_when_no_spot = [], [], [], Counter()
    for rs in hole_and_pocket_groups.values():
        fd = final_diameter(rs)
        spots = [r for r in rs if r['op_name'].startswith('SPOT_DRILL')]
        if spots:
            has_spot_fd.append(fd)
            ratios.append(float(spots[0]['tool_diameter_mm']) / fd if fd else None)
        else:
            no_spot_fd.append(fd)
            lead_when_no_spot[base(rs[0]['op_name'])] += 1
    print(f"  {stats_line('final_diameter WITH spot drill', has_spot_fd)}")
    print(f"  {stats_line('final_diameter WITHOUT spot drill', no_spot_fd)}")
    print(f"  {stats_line('spot_drill_diam / final_diam ratio', ratios)}")
    sd_diams = Counter(round(float(r['tool_diameter_mm']), 1)
                        for rs in hole_and_pocket_groups.values() for r in rs
                        if r['op_name'].startswith('SPOT_DRILL'))
    print(f"  spot drill tool diameter distribution (mm): {dict(sd_diams)}")
    print(f"  leading op when NO spot drill present: {dict(lead_when_no_spot)}")

# ---------- section f: BORING_REAMING ----------
def analyze_boring(groups):
    header("(f) BORING_REAMING: which chains end in boring")
    chains, diams = Counter(), []
    for rs in groups.values():
        if any(r['template_subtype'] == 'BORING_REAMING' for r in rs):
            chains['>'.join(base(r['op_name']) for r in rs)] += 1
            diams.append(final_diameter(rs))
    for c, n in chains.most_common(10): print(f"  [{n:4d}] {c}")
    print(f"  {stats_line('final diameter of bored features', diams)}")

def main():
    rows = load_rows()
    vols = load_volumes()
    groups = group_rows(rows)
    write_csv(build_chain_rows(groups))
    print(f"wrote {OUT_CSV} ({len(groups)} feature groups)")
    hg = analyze_holes(groups, vols)
    analyze_pockets(groups)
    analyze_chamfer(groups, rows)
    hp_prefixes = ('STEP1HOLE', 'STEP1POCKET')
    hole_and_pocket = {k: v for k, v in groups.items() if gprefix(k[1]) in hp_prefixes}
    analyze_spot(hole_and_pocket)
    analyze_boring(groups)

if __name__ == '__main__':
    main()
