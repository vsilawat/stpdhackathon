from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA = Path("data/MachinePlan-10K")
GT = Path("data/MachinePlan-10K-gt")
OUT = Path("submission/hard_ptp")

def do_part(part_id: str) -> str:
    import json

    from machineplan import brep, features, ptp
    from machineplan import plan as planmod
    try:
        part = brep.load(str(DATA / part_id / f"{part_id}.stp"))
        found = features.extract(part)
        ops = planmod.plan(found)
        tools = {"part_id": part_id, "summary": {"number_of_operations": len(ops)},
                 "operations": [{"operation_number": i + 1, "tool_type": o.tool_type or "end_mill",
                                 "tool_diameter_mm": round(o.tool_diameter or 10.0, 2)}
                                for i, o in enumerate(ops)]}
        (OUT.parent / "hard_tools" / f"{part_id}_tools.json").write_text(json.dumps(tools, indent=1))
        pd = OUT / part_id
        for f in pd.glob("*.ptp"): f.unlink()
        for i, o in enumerate(ops):
            (pd / f"{part_id}_operation_{i + 1:02d}.ptp").write_text(ptp.emit(part_id, o, found, part))
        return f"ok {part_id}"
    except Exception as e:
        return f"ERR {part_id} {e!r}"

def main() -> int:
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"): os.environ[v] = "1"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    parts = sorted(p.name for p in GT.iterdir() if p.is_dir())
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4]
    holdout = holdout[::max(len(holdout) // n, 1)][:n]
    with mp.Pool(8) as pool:
        for msg in pool.imap_unordered(do_part, holdout):
            if msg.startswith("ERR"): print(msg, flush=True)
    print("done", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
