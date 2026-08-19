from __future__ import annotations

import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA = Path("data/MachinePlan-10K")
OUT = Path("submission")

def do_part(part_id: str) -> str:
    from machineplan import brep, features, ipw, mesh, ptp
    from machineplan import plan as planmod
    try:
        part = brep.load(str(DATA / part_id / f"{part_id}.stp"))
        found = features.extract(part)
        ops = planmod.plan(found)
        seq = {"part_id": part_id, "summary": {"number_of_operations": len(ops)},
               "operations": [{"operation_number": i + 1, "o1": o.o1, "o2": o.o2}
                              for i, o in enumerate(ops)]}
        (OUT / "easy" / f"{part_id}_sequence.json").write_text(json.dumps(seq, indent=1))
        tools = {"part_id": part_id, "summary": {"number_of_operations": len(ops)},
                 "operations": [{"operation_number": i + 1, "tool_type": o.tool_type or "end_mill",
                                 "tool_diameter_mm": round(o.tool_diameter or 10.0, 2)}
                                for i, o in enumerate(ops)]}
        (OUT / "hard_tools" / f"{part_id}_tools.json").write_text(json.dumps(tools, indent=1))
        md = OUT / "medium" / part_id; md.mkdir(parents=True, exist_ok=True)
        for i, sol in enumerate(ipw.ipws(part, found, ops)):
            mesh.save_stl(sol, md / f"{part_id}_operation_{i + 1:02d}.stl")
        pd = OUT / "hard_ptp" / part_id; pd.mkdir(parents=True, exist_ok=True)
        for i, o in enumerate(ops):
            (pd / f"{part_id}_operation_{i + 1:02d}.ptp").write_text(ptp.emit(part_id, o, found, part))
        return f"ok {part_id} {len(ops)}"
    except Exception:
        return f"ERR {part_id} {traceback.format_exc(limit=2)}"

def done(part_id: str) -> bool:
    f = OUT / "easy" / f"{part_id}_sequence.json"
    if not f.exists(): return False
    n = json.loads(f.read_text())["summary"]["number_of_operations"]
    d = OUT / "hard_ptp" / part_id
    return d.is_dir() and len(list(d.glob("*.ptp"))) == n

def main() -> int:
    for d in ("easy", "hard_tools", "medium", "hard_ptp"): (OUT / d).mkdir(parents=True, exist_ok=True)
    parts = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    todo = [p for p in parts if not done(p)]
    print(f"{len(todo)}/{len(parts)} to generate", flush=True)
    nerr = 0
    with mp.Pool(processes=max(mp.cpu_count() - 2, 2)) as pool:
        for k, msg in enumerate(pool.imap_unordered(do_part, todo, chunksize=4)):
            if msg.startswith("ERR"): nerr += 1; print(msg, flush=True)
            if (k + 1) % 200 == 0: print(f"[{k + 1}/{len(todo)}] errs={nerr}", flush=True)
    print(f"done: {len(todo)} parts, {nerr} errors", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
