from __future__ import annotations

import json
import multiprocessing as mp
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data/MachinePlan-10K")
m_dia = re.compile(r"\(D\) Diameter\s*=\s*([\d.]+)")
m_pa = re.compile(r"\(PA\) Point Angle\s*=\s*([\d.]+)")
m_name = re.compile(r"Object name: (\w+?)(?:_\d+)?\s")

def do(pid: str) -> list:
    out = []
    for f in (DATA / pid).glob("*_details.txt"):
        t = f.read_text(errors="ignore")
        dm, pm, nm = m_dia.search(t), m_pa.search(t), m_name.search(t)
        if dm and pm and nm: out.append((nm.group(1), round(float(dm.group(1)), 2), float(pm.group(1))))
    return out

def main() -> int:
    parts = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    table: dict = defaultdict(Counter)
    with mp.Pool(8) as pool:
        for res in pool.imap_unordered(do, parts, chunksize=50):
            for name, d, pa in res: table[(name, d)][pa] += 1
    pure = tot = 0
    out = {}
    for k, c in table.items():
        n = sum(c.values()); tot += n
        pa, cnt = c.most_common(1)[0]
        pure += cnt
        out[f"{k[0]}|{k[1]}"] = pa
    print(f"keys={len(out)} purity={pure / tot:.4f} rows={tot}")
    json.dump(out, open("derived/point_angles.json", "w"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
