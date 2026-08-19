from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA = Path("data/MachinePlan-10K")
OUT = Path("submission/hard_ptp")

def do_part(part_id: str) -> str:
    from machineplan import brep, features, ptp
    from machineplan import plan as planmod
    try:
        part = brep.load(str(DATA / part_id / f"{part_id}.stp"))
        found = features.extract(part)
        ops = planmod.plan(found)
        pd = OUT / part_id; pd.mkdir(parents=True, exist_ok=True)
        for i, o in enumerate(ops):
            (pd / f"{part_id}_operation_{i + 1:02d}.ptp").write_text(ptp.emit(part_id, o, found, part))
        import json
        tools = {"part_id": part_id, "summary": {"number_of_operations": len(ops)},
                 "operations": [{"operation_number": i + 1, "tool_type": o.tool_type or "end_mill",
                                 "tool_diameter_mm": round(o.tool_diameter or 10.0, 2)}
                                for i, o in enumerate(ops)]}
        (OUT.parent / "hard_tools" / f"{part_id}_tools.json").write_text(json.dumps(tools, indent=1))
        return "ok"
    except Exception:
        return f"ERR {part_id} {traceback.format_exc(limit=1)}"

def main() -> int:
    parts = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    nerr = 0
    with mp.Pool(processes=max(mp.cpu_count() - 2, 2)) as pool:
        for k, msg in enumerate(pool.imap_unordered(do_part, parts, chunksize=8)):
            if msg != "ok": nerr += 1; print(msg, flush=True)
            if (k + 1) % 1000 == 0: print(f"[{k + 1}/{len(parts)}] errs={nerr}", flush=True)
    print(f"done, {nerr} errors", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
