from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from machineplan import brep, dataset, features

OUT = Path("derived/features.jsonl")

def hole_row(h: features.Hole) -> dict:
    return {"x": h.x, "y": h.y, "d": h.diameter, "depth": h.depth,
            "through": h.through, "bottom_z": h.bottom_z, "mouth_z": h.mouth_z}

def pocket_row(p: features.Pocket) -> dict:
    return {"floor_z": p.floor_z, "depth": p.depth, "area": p.area, "kind": p.kind,
            "open_sides": p.open_sides, "fillet_radius": p.fillet_radius, "corners": p.corners}

def chamfer_row(c: features.Chamfer) -> dict:
    return {"width": c.width, "angle_deg": c.angle_deg, "n_faces": len(c.faces)}

def scan(part_dir: Path) -> dict:
    try:
        found = features.extract(brep.load(dataset.step_file(part_dir)))
    except Exception as error:  # noqa: BLE001
        return {"part": part_dir.name, "error": f"{type(error).__name__}: {error}"}
    return {"part": part_dir.name, "stock": list(found.stock),
            "top_z": found.top_z, "bottom_z": found.bottom_z,
            "holes": [hole_row(h) for h in found.holes],
            "pockets": [pocket_row(p) for p in found.pockets],
            "chamfers": [chamfer_row(c) for c in found.chamfers]}

def main() -> int:
    parts = dataset.part_dirs()
    print(f"scanning {len(parts)} parts...", flush=True)
    start = time.time()
    OUT.parent.mkdir(exist_ok=True)
    rows, failed = [], 0
    with Pool() as pool:
        for n, row in enumerate(pool.imap_unordered(scan, parts, chunksize=25), 1):
            rows.append(row)
            failed += "error" in row
            if n % 2000 == 0: print(f"  {n}/{len(parts)}  {time.time()-start:.0f}s  failed={failed}", flush=True)
    rows.sort(key=lambda r: r["part"])
    with OUT.open("w") as fh:
        for row in rows: fh.write(json.dumps(row) + "\n")
    ok = [r for r in rows if "error" not in r]
    n_holes = sum(len(r["holes"]) for r in ok)
    n_pockets = sum(len(r["pockets"]) for r in ok)
    n_chamfers = sum(len(r["chamfers"]) for r in ok)
    print(f"\ndone in {time.time()-start:.0f}s   parts={len(rows)}  ok={len(ok)}  failed={failed}")
    print(f"chamfers={n_chamfers}  pockets={n_pockets}  holes={n_holes}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
