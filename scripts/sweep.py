from __future__ import annotations

import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from machineplan import brep, dataset, features

# Dataset_Description.pdf s6.6
PUBLISHED = {"chamfers": 16280, "pockets": 18484, "holes": 23322}
PUBLISHED_POCKET_TYPES = {"center": 5329, "corner": 5155, "edge": 5131, "slot": 2869}
PUBLISHED_THROUGH = 11602
PUBLISHED_BLIND = 11720
OUT = Path("derived/sweep.jsonl")

def base_name(name: str) -> str: return name.rsplit("_", 1)[0] if name[-1].isdigit() else name

def scan(part_dir: Path) -> dict:
    try:
        found = features.extract(brep.load(dataset.step_file(part_dir)))
    except Exception as error:  # noqa: BLE001
        return {"part": part_dir.name, "error": f"{type(error).__name__}: {error}"}
    ops = dataset.load_operations(part_dir)
    return {"part": part_dir.name, **found.counts,
            "pocket_types": Counter(p.kind for p in found.pockets),
            "through": sum(h.through for h in found.holes),
            "blind": sum(not h.through for h in found.holes),
            "stock": list(found.stock), "n_ops": len(ops),
            "op_names": Counter(base_name(o["name"]) for o in ops)}

def report(rows: list[dict]) -> None:
    ok = [r for r in rows if "error" not in r]
    print(f"{'feature':10s} {'found':>8s} {'published':>10s} {'diff':>8s} {'ratio':>7s}")
    for key, want in PUBLISHED.items():
        got = sum(r[key] for r in ok)
        print(f"{key:10s} {got:8d} {want:10d} {got-want:+8d} {got/want:7.3f}")
    types: Counter = Counter()
    for r in ok: types.update(r["pocket_types"])
    print(f"\n{'pocket':10s} {'found':>8s} {'published':>10s} {'diff':>8s}")
    for key, want in PUBLISHED_POCKET_TYPES.items():
        print(f"{key:10s} {types.get(key, 0):8d} {want:10d} {types.get(key, 0)-want:+8d}")
    through, blind = sum(r["through"] for r in ok), sum(r["blind"] for r in ok)
    print(f"\n{'through':10s} {through:8d} {PUBLISHED_THROUGH:10d} {through-PUBLISHED_THROUGH:+8d}")
    print(f"{'blind':10s} {blind:8d} {PUBLISHED_BLIND:10d} {blind-PUBLISHED_BLIND:+8d}")

def main() -> int:
    parts = dataset.part_dirs()
    print(f"scanning {len(parts)} parts...", flush=True)
    start = time.time()
    OUT.parent.mkdir(exist_ok=True)
    rows, failed = [], 0
    with Pool() as pool, OUT.open("w") as fh:
        for n, row in enumerate(pool.imap_unordered(scan, parts, chunksize=25), 1):
            rows.append(row)
            failed += "error" in row
            fh.write(json.dumps(row, default=dict) + "\n")
            if n % 2000 == 0: print(f"  {n}/{len(parts)}  {time.time()-start:.0f}s  failed={failed}", flush=True)
    print(f"\ndone in {time.time()-start:.0f}s   ok={len(rows)-failed}  failed={failed}\n")
    report(rows)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
