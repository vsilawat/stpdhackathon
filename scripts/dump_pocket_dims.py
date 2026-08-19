from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def do_part(args) -> str | None:
    pid, dia = args
    try:
        from shapely.geometry import Polygon
        from machineplan import brep, features
        part = brep.load(f"data/MachinePlan-10K/{pid}/{pid}.stp")
        found = features.extract(part)
        if len(found.pockets) != 1: return None
        pk = found.pockets[0]
        poly = Polygon([(x, y) for x, y, _ in brep.outline(part, pk.faces[0])])
        r = poly.minimum_rotated_rectangle.exterior.coords
        import math
        e1 = math.dist(r[0], r[1]); e2 = math.dist(r[1], r[2])
        return json.dumps({"part": pid, "dia": dia, "kind": pk.kind, "fr": pk.fillet_radius or 0.0,
                           "depth": pk.depth, "area": pk.area, "open": pk.open_sides or 0,
                           "w": min(e1, e2), "l": max(e1, e2)})
    except Exception:
        return None

def main() -> int:
    feats = {r["part"]: len(r["pockets"]) for r in map(json.loads, open("derived/features.jsonl"))}
    jobs = []
    for r in csv.DictReader(open("derived/chains.csv")):
        if not r["group_prefix"].startswith("STEP1POCKET"): continue
        if feats.get(r["part"]) != 1: continue
        dias = [float(v) for v in r["tool_diams"].split(">") if v]
        if dias: jobs.append((r["part"], dias[-1]))
    print(len(jobs), "jobs", flush=True)
    with mp.Pool(max(mp.cpu_count() - 2, 2)) as pool, open("derived/pocket_dims.jsonl", "w") as f:
        for k, res in enumerate(pool.imap_unordered(do_part, jobs, chunksize=8)):
            if res: f.write(res + "\n")
            if (k + 1) % 500 == 0: print(k + 1, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
