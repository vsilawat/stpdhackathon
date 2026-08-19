from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

DATA = Path("data/MachinePlan-10K")
OUT = Path("derived/sequences.jsonl")

def base_name(name: str) -> str: return name.rsplit("_", 1)[0] if name[-1].isdigit() else name

def load_parts() -> list[dict]:
    parts = []
    for part_dir in sorted(DATA.glob("featured_part_*")):
        f = part_dir / f"{part_dir.name}_operations.json"
        if not f.exists(): continue
        d = json.loads(f.read_text())
        ops = sorted(d["operations"], key=lambda o: o["sequence_number"])
        parts.append({"part": part_dir.name, "ops": [
            {"name": base_name(o["name"]), "full": o["name"], "tool": o.get("tool_name"),
             "vol": o.get("volume_removed_mm3")} for o in ops]})
    return parts

def write_jsonl(parts: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for p in parts: fh.write(json.dumps(p) + "\n")

def analyze_counts(parts: list[dict]) -> None:
    counts = Counter()
    for p in parts:
        for op in p["ops"]: counts[op["name"]] += 1
    print(f"\n=== 1. Distinct base names ({len(counts)}) ===")
    for name, c in counts.most_common(): print(f"  {name:45s} {c:6d}")

def part_positions(p: dict) -> dict[str, list[int]]:
    pos = defaultdict(list)
    for i, op in enumerate(p["ops"]): pos[op["name"]].append(i)
    return pos

def precedence_analysis(parts: list[dict]) -> dict[tuple[str, str], float]:
    pair_before = Counter()
    pair_total = Counter()
    for p in parts:
        pos = part_positions(p)
        names = list(pos.keys())
        for a, b in combinations(names, 2):
            amax, amin = max(pos[a]), min(pos[a])
            bmax, bmin = max(pos[b]), min(pos[b])
            if amax < bmin: pair_before[(a, b)] += 1
            elif bmax < amin: pair_before[(b, a)] += 1
            pair_total[frozenset((a, b))] += 1
    print("\n=== 2. Precedence consistency ===")
    consistent, variable = [], []
    frac = {}
    for key in pair_total:
        a, b = tuple(key)
        total = pair_total[key]
        ab = pair_before[(a, b)]
        ba = pair_before[(b, a)]
        if total < 5: continue
        pab = ab / total
        pba = ba / total
        frac[key] = (a, b, pab, pba, total)
        if pab >= 0.99 or pba >= 0.99: consistent.append((a, b, pab, pba, total))
        elif 0.40 <= pab <= 0.60: variable.append((a, b, pab, pba, total))
    print(f"  {len(consistent)} pairs with >=99% consistent precedence (n>=5 co-occurrences):")
    for a, b, pab, pba, total in sorted(consistent, key=lambda x: -x[4]):
        winner, pct = (a, pab) if pab >= pba else (b, pba)
        loser = b if winner == a else a
        print(f"    {winner:45s} before {loser:45s}  {pct*100:5.1f}%  (n={total})")
    print(f"\n  {len(variable)} pairs that are genuinely variable (40-60%):")
    for a, b, pab, pba, total in sorted(variable, key=lambda x: -x[4]):
        print(f"    {a:45s} vs {b:45s}  {pab*100:5.1f}% / {pba*100:5.1f}%  (n={total})")
    return frac

def block_contiguity(parts: list[dict]) -> None:
    print("\n=== 3. Block contiguity (is each base name a single contiguous run?) ===")
    runs_total = Counter()
    runs_contig = Counter()
    for p in parts:
        pos = part_positions(p)
        for name, idxs in pos.items():
            if len(idxs) < 2: continue
            runs_total[name] += 1
            idxs_sorted = sorted(idxs)
            contiguous = all(idxs_sorted[i + 1] - idxs_sorted[i] == 1 for i in range(len(idxs_sorted) - 1))
            if contiguous: runs_contig[name] += 1
    for name in sorted(runs_total, key=lambda n: -runs_total[n]):
        tot = runs_total[name]
        c = runs_contig[name]
        print(f"  {name:45s} {c}/{tot} = {c/tot*100:5.1f}% contiguous")

def within_group_ordering(parts: list[dict]) -> None:
    print("\n=== 4. Within-part ordering of repeated same-name ops (by volume_removed) ===")
    asc, desc, neither, total = 0, 0, 0, 0
    for p in parts:
        pos = part_positions(p)
        for idxs in pos.values():
            if len(idxs) < 2: continue
            idxs_sorted = sorted(idxs)
            vols = [p["ops"][i]["vol"] for i in idxs_sorted]
            if any(v is None for v in vols): continue
            total += 1
            if all(vols[i] <= vols[i + 1] for i in range(len(vols) - 1)): asc += 1
            elif all(vols[i] >= vols[i + 1] for i in range(len(vols) - 1)): desc += 1
            else: neither += 1
    print(f"  groups with >=2 same-name ops: {total}")
    if total: print(f"  ascending volume:  {asc:6d} ({asc/total*100:5.1f}%)")
    if total: print(f"  descending volume: {desc:6d} ({desc/total*100:5.1f}%)")
    if total: print(f"  neither monotone:  {neither:6d} ({neither/total*100:5.1f}%)")

def tool_grouping(parts: list[dict]) -> None:
    print("\n=== 5. Tool grouping (minimize tool changes?) ===")
    actual_changes, min_changes, random_changes, n_multi_tool = 0, 0, 0, 0
    rng = random.Random(0)
    for p in parts:
        tools = [op["tool"] for op in p["ops"]]
        if len(set(tools)) < 2: continue
        n_multi_tool += 1
        actual_changes += sum(tools[i] != tools[i + 1] for i in range(len(tools) - 1))
        min_changes += len(set(tools)) - 1
        shuffled = tools[:]
        rng.shuffle(shuffled)
        random_changes += sum(shuffled[i] != shuffled[i + 1] for i in range(len(shuffled) - 1))
    print(f"  parts with >1 distinct tool: {n_multi_tool}")
    if n_multi_tool:
        print(f"  actual tool changes (total):    {actual_changes}  (avg {actual_changes/n_multi_tool:.2f}/part)")
        print(f"  minimum possible (total):       {min_changes}  (avg {min_changes/n_multi_tool:.2f}/part)")
        print(f"  random shuffle expectation:      {random_changes}  (avg {random_changes/n_multi_tool:.2f}/part)")
        print(f"  actual is {(actual_changes-min_changes)/max(random_changes-min_changes,1)*100:.1f}% of the way from optimal to random")

def topo_order(consistent_pairs: list[tuple[str, str, float]], all_names: set[str]) -> list[str]:
    edges = defaultdict(set)
    indeg = defaultdict(int)
    for a, b in consistent_pairs:
        if b not in edges[a]:
            edges[a].add(b)
            indeg[b] += 1
    for n in all_names: indeg.setdefault(n, 0)
    order = []
    avail = sorted([n for n in all_names if indeg[n] == 0])
    remaining = set(all_names)
    while avail:
        avail.sort()
        n = avail.pop(0)
        order.append(n)
        remaining.discard(n)
        for m in edges[n]:
            indeg[m] -= 1
            if indeg[m] == 0 and m in remaining: avail.append(m)
    order.extend(sorted(remaining - set(order)))
    return order

def global_order(parts: list[dict], frac: dict) -> None:
    print("\n=== 6. Derived GLOBAL ORDER (topological sort of >=99% precedence pairs) ===")
    all_names = set()
    for p in parts:
        for op in p["ops"]: all_names.add(op["name"])
    strong_edges = []
    for a, b, pab, pba, _total in frac.values():
        if pab >= 0.99: strong_edges.append((a, b))
        elif pba >= 0.99: strong_edges.append((b, a))
    order = topo_order(strong_edges, all_names)
    for i, n in enumerate(order): print(f"  {i:2d}. {n}")
    print("\n  SPOT_DRILL immediately-precedes-its-drill check:")
    check_spot_drill_adjacency(parts)
    superblock_analysis(parts)

MILLING_FAMILY = {"AREA_MILL", "MILL_CORNER_NOTCH_RECTANGULAR", "MILL_CORNER_NOTCH_ROUND",
    "MILL_CORNER_NOTCH_STRAIGHT", "MILL_CORNER_NOTCH_U", "MILL_RECTANGULAR_POCKET",
    "MILL_OPEN_POCKET", "MILL_FREE_SHAPED_POCKET", "MILL_SLOT", "MILL_RECTANGULAR_SLOT",
    "MILL_SLOT_ROUND_PART", "MILL_SLOT_PARTIAL_OBROUND", "MILL_THROUGH_HOLE_FROM_SOLID_MATERIAL",
    "MILL_BLIND_HOLE_FROM_SOLID_MATERIAL", "MILL_ROUGH_BLIND_HOLE_CONTOUR",
    "MILL_FINISH_BLIND_HOLE_FLAT_BOTTOM", "MILL_FREE_SHAPED_THROUGH"}
DRILLING_FAMILY = {"SPOT_DRILL", "DRILL_BLIND_HOLE_INTO_CENTER", "DRILL_THROUGH_HOLE_INTO_CENTER",
    "DRILL_TO_ENLARGE_THROUGH_HOLE", "DRILL_TO_ENLARGE_BLIND_HOLE", "GUN_DRILL_THROUGH_HOLE",
    "GUN_DRILL_BLIND_HOLE", "INDEXABLE_INSERT_DRILL_BLIND_HOLE_FROM_SOLID",
    "INDEXABLE_INSERT_DRILL_THROUGH_HOLE_FROM_SOLID", "SPADE_DRILL_TO_ENLARGE_THROUGH_HOLE",
    "BORE_BLIND_HOLE", "DRILL_THROUGH_HOLE_FROM_SOLID_MATERIAL"}

def superblock_analysis(parts: list[dict]) -> None:
    print("\n  Milling-family vs drilling-family superblock check (the two families explain")
    print("  most of the 'variable' pairs above -- they are each internally well-ordered,")
    print("  but which family goes first is close to a coin flip):")
    mill_first = drill_first = interleaved = tot = 0
    for p in parts:
        posm = [i for i, o in enumerate(p["ops"]) if o["name"] in MILLING_FAMILY]
        posd = [i for i, o in enumerate(p["ops"]) if o["name"] in DRILLING_FAMILY]
        if not posm or not posd: continue
        tot += 1
        mmax, mmin, dmax, dmin = max(posm), min(posm), max(posd), min(posd)
        if mmax < dmin: mill_first += 1
        elif dmax < mmin: drill_first += 1
        else: interleaved += 1
    if tot:
        print(f"    parts with both families present: {tot}")
        print(f"    milling block entirely before drilling block:  {mill_first:5d} ({mill_first/tot*100:5.1f}%)")
        print(f"    drilling block entirely before milling block:  {drill_first:5d} ({drill_first/tot*100:5.1f}%)")
        print(f"    interleaved (neither is a clean superblock):   {interleaved:5d} ({interleaved/tot*100:5.1f}%)")

def check_spot_drill_adjacency(parts: list[dict]) -> None:
    total_spot, immediately_before_a_drill = 0, 0
    drill_like = {"DRILL_THROUGH_HOLE_INTO_CENTER", "DRILL_BLIND_HOLE_INTO_CENTER",
                  "DRILL_TO_ENLARGE_THROUGH_HOLE", "DRILL_TO_ENLARGE_BLIND_HOLE",
                  "GUN_DRILL_THROUGH_HOLE", "SPADE_DRILL_TO_ENLARGE_THROUGH_HOLE"}
    for p in parts:
        names = [op["name"] for op in p["ops"]]
        for i, n in enumerate(names):
            if n != "SPOT_DRILL": continue
            total_spot += 1
            if i + 1 < len(names) and names[i + 1] in drill_like: immediately_before_a_drill += 1
    if total_spot:
        print(f"    SPOT_DRILL instances: {total_spot}, immediately followed by a drilling op: "
              f"{immediately_before_a_drill} ({immediately_before_a_drill/total_spot*100:.1f}%)")

def common_patterns(parts: list[dict]) -> None:
    print("\n=== 7. Top 20 most common base-name sequences ===")
    seqs = Counter(tuple(op["name"] for op in p["ops"]) for p in parts)
    for seq, c in seqs.most_common(20): print(f"  n={c:5d}  {' -> '.join(seq)}")

def main() -> None:
    parts = load_parts()
    print(f"Loaded {len(parts)} parts, {sum(len(p['ops']) for p in parts)} total operations")
    write_jsonl(parts)
    analyze_counts(parts)
    frac = precedence_analysis(parts)
    block_contiguity(parts)
    within_group_ordering(parts)
    tool_grouping(parts)
    global_order(parts, frac)
    common_patterns(parts)

if __name__ == "__main__": main()
