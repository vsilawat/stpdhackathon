from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

def vocab_type(sub: str, typ: str) -> str:
    s = sub.lower()
    if typ == "Chamfer Mill": return "chamfer_mill"
    if typ == "Spot Drill": return "spot_drill"
    if typ == "Boring Tool": return "boring_tool"
    if typ.startswith("Milling"): return "end_mill"
    if "insert" in s: return "insert_drill"
    if "gun" in s: return "gun_drill"
    if "spade" in s: return "spade_drill"
    return "twist_drill"

def main() -> int:
    gt = defaultdict(list)
    for r in csv.DictReader(open("derived/opdetails.csv")):
        gt[r["part"]].append((int(r["op_index"]), vocab_type(r["tool_template_subtype"], r["tool_type"]),
                              float(r["tool_diameter_mm"])))
    parts = sorted(gt)
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4]
    both = t_ok = d_ok = n = 0; nlen = 0
    per = defaultdict(lambda: [0, 0, 0])
    for pid in holdout:
        f = Path(f"submission/hard_tools/{pid}_tools.json")
        if not f.exists(): continue
        ops = json.loads(f.read_text())["operations"]
        g = sorted(gt[pid])
        nlen += len(ops) == len(g)
        for k in range(min(len(ops), len(g))):
            _, gtyp, gdia = g[k]
            mtyp, mdia = ops[k]["tool_type"], ops[k]["tool_diameter_mm"]
            n += 1
            tm, dm = mtyp == gtyp, abs(mdia - gdia) / gdia <= 0.02
            t_ok += tm; d_ok += dm; both += tm and dm
            per[gtyp][0] += 1; per[gtyp][1] += tm; per[gtyp][2] += dm
    print(f"parts={len(holdout)} len_match={nlen / len(holdout):.3f} ops={n}")
    print(f"type={t_ok / n:.4f} dia2%={d_ok / n:.4f} BOTH={both / n:.4f}")
    for t, (c, tv, dv) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"  {t:14s} n={c:6d} type={tv / c:.3f} dia={dv / c:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
