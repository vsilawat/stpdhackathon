from __future__ import annotations

import csv
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

GT = Path("data/MachinePlan-10K-gt")
OUT = Path("derived/ptp_cycles.csv")
CYCLE = re.compile(r"G(81|73|85) G98 Z(-?[\d.]+) F([\d.]+)\.?(?: Q(-?[\d.]+))? R(-?[\d.]+)")
XY = re.compile(r"X(-?[\d.]+) Y(-?[\d.]+)")
ZC = re.compile(r"G43 Z(-?[\d.]+)")
Z1 = re.compile(r"G1 Z(-?[\d.]+) F([\d.]+)")
ZONLY = re.compile(r"^N\d+ Z(-?[\d.]+)$")

def parse_ptp(path: Path) -> dict | None:
    text = path.read_text()
    m = re.search(r"\((\w+) , TOOL", text)
    if not m: return None
    row = {"op": m.group(1), "seq": int(path.name[:3])}
    if mm := XY.search(text): row["x"], row["y"] = float(mm.group(1)), float(mm.group(2))
    if mm := ZC.search(text): row["zclear"] = float(mm.group(1))
    if mm := CYCLE.search(text):
        row["cycle"] = {"81": "G81", "73": "G73", "85": "G85"}[mm.group(1)]
        row["z"], row["f"] = float(mm.group(2)), float(mm.group(3))
        row["q"] = float(mm.group(4)) if mm.group(4) else ""
        row["r"] = float(mm.group(5))
    elif "GUN_DRILL" in row["op"]:
        zs = [float(v) for v in Z1.findall(text) for v in [v[0]]]
        pecks = [float(v) for v in ZONLY.findall(text)]
        first = Z1.search(text)
        row["cycle"] = "GUN"
        if first: row["z1"], row["f"] = float(first.group(1)), float(first.group(2))
        row["z"] = min(zs + pecks) if zs or pecks else ""
        row["npecks"] = text.count("G4 X")
    else: return None
    return row

def scan(item):
    part, feats = item
    rows = []
    for p in sorted((GT / part).glob("[0-9]*.ptp")):
        r = parse_ptp(p)
        if r is None: continue
        r["part"] = part
        r["top_z"] = feats["top_z"]
        hs = feats["holes"]
        if hs and "x" in r:
            h = min(hs, key=lambda h: abs(h["x"] - r["x"]) + abs(h["y"] - r["y"]))
            if abs(h["x"] - r["x"]) + abs(h["y"] - r["y"]) < 0.5:
                r |= {"hole_d": h["d"], "hole_depth": h["depth"], "hole_through": int(h["through"]),
                      "hole_bottom": h["bottom_z"], "hole_mouth": h["mouth_z"]}
        rows.append(r)
    ops_on_hole: dict[tuple, list] = {}
    for r in rows:
        if "hole_d" in r: ops_on_hole.setdefault((r["x"], r["y"]), []).append(r)
    for chain in ops_on_hole.values():
        chain.sort(key=lambda r: r["seq"])
        names = ">".join(c["op"] for c in chain)
        for i, r in enumerate(chain): r["chain"], r["chain_pos"] = names, i
    return rows

COLS = ["part", "seq", "op", "cycle", "x", "y", "z", "r", "q", "f", "z1", "npecks", "zclear",
        "top_z", "hole_d", "hole_depth", "hole_through", "hole_bottom", "hole_mouth", "chain", "chain_pos"]

def main() -> int:
    feats = {r["part"]: r for r in map(json.loads, open("derived/features.jsonl")) if "error" not in r}
    parts = sorted(feats)[:: int(sys.argv[1]) if len(sys.argv) > 1 else 7]
    with Pool() as pool:
        all_rows = [r for rows in pool.imap_unordered(scan, [(p, feats[p]) for p in parts], chunksize=20)
                    for r in rows]
    with OUT.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in all_rows: w.writerow({c: r.get(c, "") for c in COLS})
    print(f"{len(all_rows)} cycle rows from {len(parts)} parts -> {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
