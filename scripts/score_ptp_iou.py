from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GT = Path("data/MachinePlan-10K-gt")
SUB = Path("submission/hard_ptp")
CYCLE = re.compile(r"G(?:81|73|85) G98 Z(-?[\d.]+) F[\d.]+(?: Q[\d.]+)? R(-?[\d.]+)")
ZMOVE = re.compile(r"^N\d+ Z(-?[\d.]+) F[\d.]+", re.MULTILINE)
XY0 = re.compile(r"G0 G90 X(-?[\d.]+) Y(-?[\d.]+)")
OPRE = re.compile(r"^\((\w+) *, *TOOL", re.MULTILINE)
TOK = re.compile(r"([XYZ])(-?[\d.]+)")
RARC = re.compile(r" R(-?[\d.]+)")

def _arc_pts(p0, p1, R, cw):
    import math
    x0, y0, z0 = p0; x1, y1, z1 = p1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d < 1e-9: return []
    h2 = R * R - (d / 2) ** 2
    h = math.sqrt(h2) if h2 > 0 else 0.0
    ux, uy = dx / d, dy / d
    s = (-1.0 if cw else 1.0) * (1.0 if R >= 0 else -1.0)
    cx, cy = (x0 + x1) / 2 - s * h * uy, (y0 + y1) / 2 + s * h * ux
    a0, a1 = math.atan2(y0 - cy, x0 - cx), math.atan2(y1 - cy, x1 - cx)
    if cw:
        while a1 >= a0 - 1e-12: a1 -= 2 * math.pi
    else:
        while a1 <= a0 + 1e-12: a1 += 2 * math.pi
    r, n = abs(R), max(int(abs(a1 - a0) / 0.2), 1)
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n),
             z0 + (z1 - z0) * i / n) for i in range(1, n)]

def parse(path: Path):
    """Return (op, kind, data): drill -> (x, y, z, r); mill -> list of xyz cut points (modal coords)."""
    t = path.read_text()
    m = OPRE.search(t)
    op = m.group(1) if m else "?"
    c = CYCLE.search(t)
    xy = XY0.search(t)
    if c and xy:
        return op, "drill", (float(xy.group(1)), float(xy.group(2)), float(c.group(1)), float(c.group(2)))
    pts, cur, mode = [], {}, 0
    has_xy_cut = False
    for line in t.splitlines():
        if not line.startswith("N"): continue
        if " G4 " in line or line.rstrip().endswith("G4"): continue  # dwell: X is seconds, not a coordinate
        toks = dict(TOK.findall(line))
        for g, m in ((" G0 ", 0), (" G1 ", 1), (" G2 ", 2), (" G3 ", 3)):
            if g in line or line.rstrip().endswith(g.strip()): mode = m
        if mode == 0:
            cur.update((k, float(v)) for k, v in toks.items())
            if pts and pts[-1] is not None: pts.append(None)
            continue
        if not toks: continue
        prev = (cur.get("X"), cur.get("Y"), cur.get("Z"))
        cur.update((k, float(v)) for k, v in toks.items())
        if {"X", "Y", "Z"} <= cur.keys():
            newp = (cur["X"], cur["Y"], cur["Z"])
            rm = RARC.search(line)
            if mode >= 2 and rm and None not in prev and ("X" in toks or "Y" in toks):
                pts.extend(_arc_pts(prev, newp, float(rm.group(1)), mode == 2))
            pts.append(newp)
            if "X" in toks or "Y" in toks: has_xy_cut = True
    if not has_xy_cut and xy:
        zs = ZMOVE.findall(t)
        if zs: return op, "drill", (float(xy.group(1)), float(xy.group(2)), float(zs[-1]),
                                    float(zs[0]) if len(zs) > 1 else float(zs[-1]) + 10)
    return op, "mill", pts

def densify(pts, step=1.0):
    segs, cur = [], []
    for q in pts:
        if q is None:
            if cur: segs.append(cur); cur = []
        else: cur.append(q)
    if cur: segs.append(cur)
    out = []
    for s in segs:
        p = np.array(s, dtype=float)
        if len(p) < 2:
            out.extend(p); continue
        for a, b in zip(p[:-1], p[1:]):
            n = max(int(np.linalg.norm(b[:2] - a[:2]) / step), 1)
            for i in range(n): out.append(a + (b - a) * (i / n))
        out.append(p[-1])
    return np.array(out, dtype=float).reshape(-1, 3)

def heightfield(kind, data, radius, x0, y0, nx, ny, ztop, cell=1.0):
    hf = np.full((nx, ny), ztop)
    rr = max(int(np.ceil(radius / cell)), 1)
    dy, dx = np.meshgrid(np.arange(-rr, rr + 1), np.arange(-rr, rr + 1))
    disk = (dx * dx + dy * dy) * cell * cell <= radius * radius
    ox, oy = dx[disk], dy[disk]
    if kind == "drill":
        x, y, z, _ = data
        pts = np.array([[x, y, z]])
    else:
        pts = densify(data)
        if not len(pts): return hf
    for x, y, z in pts:
        ix, iy = int(round((x - x0) / cell)), int(round((y - y0) / cell))
        gx, gy = ix + ox, iy + oy
        ok = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
        np.minimum.at(hf, (gx[ok], gy[ok]), z)
    return hf

def op_iou(a, b, ra, rb):
    """a, b: (kind, data). Column IoU over a shared grid."""
    def bounds(k, d, r):
        if k == "drill": return d[0] - r, d[0] + r, d[1] - r, d[1] + r, d[2], d[3]
        p = np.array([q for q in d if q is not None], dtype=float)
        if not len(p): return None
        return p[:, 0].min() - r, p[:, 0].max() + r, p[:, 1].min() - r, p[:, 1].max() + r, p[:, 2].min(), p[:, 2].max()
    ba, bb = bounds(*a, ra), bounds(*b, rb)
    if ba is None or bb is None: return 0.0
    x0, x1 = min(ba[0], bb[0]), max(ba[1], bb[1])
    y0, y1 = min(ba[2], bb[2]), max(ba[3], bb[3])
    ztop = max(ba[5], bb[5]) + 5.0
    nx, ny = int(x1 - x0) + 2, int(y1 - y0) + 2
    if nx * ny > 4_000_000: return -1.0
    ha = heightfield(a[0], a[1], ra, x0, y0, nx, ny, ztop)
    hb = heightfield(b[0], b[1], rb, x0, y0, nx, ny, ztop)
    inter = np.clip(ztop - np.maximum(ha, hb), 0, None).sum()
    union = np.clip(ztop - np.minimum(ha, hb), 0, None).sum()
    return float(inter / union) if union > 0 else 0.0

def main(n_sample: int = 30) -> int:
    gtd = defaultdict(dict)
    for r in csv.DictReader(open("derived/opdetails.csv")):
        gtd[r["part"]][int(r["op_index"])] = float(r["tool_diameter_mm"])
    parts = sorted(p.name for p in GT.iterdir() if p.is_dir())
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4]
    holdout = holdout[::max(len(holdout) // n_sample, 1)][:n_sample]
    per_part, all_ious = [], defaultdict(list)
    for pid in holdout:
        sd = SUB / pid
        if not sd.is_dir(): continue
        gt_files = sorted((GT / pid).glob("[0-9]*.ptp"))
        my_files = sorted(sd.glob("*_operation_*.ptp"))
        mytools = {o["operation_number"]: o["tool_diameter_mm"]
                   for o in json.loads((Path("submission/hard_tools") / f"{pid}_tools.json").read_text())["operations"]}
        ious = []
        for k in range(max(len(gt_files), len(my_files))):
            if k >= len(gt_files) or k >= len(my_files):
                ious.append((("MISSING", "?"), 0.0)); continue
            gop, gkind, gdata = parse(gt_files[k])
            mop, mkind, mdata = parse(my_files[k])
            gd = gtd[pid].get(k + 1, 10.0)
            md = mytools.get(k + 1, 10.0)
            v = op_iou((gkind, gdata), (mkind, mdata), gd / 2, md / 2)
            ious.append(((gop, mop), v))
        vals = [v for _, v in ious if v >= 0]
        if vals: per_part.append(sum(vals) / len(vals))
        if len(gt_files) != len(my_files): continue  # op-count misses pollute the per-op view
        for k, ((gop, mop), v) in enumerate(ious):
            if 0 <= v < 0.7: print(f"   LOW op{k + 1:02d} {gop} vs {mop}: {v:.3f}")
            if v >= 0: all_ious[gop.rsplit("_", 1)[0] if gop.split("_")[-1].isdigit() else gop].append(v)
        print(f"{pid}: mean={np.mean(vals):.3f} nops gt={len(gt_files)} mine={len(my_files)}", flush=True)
    print(f"\nOVERALL mean per-part IoU: {np.mean(per_part):.4f} over {len(per_part)} parts")
    for op in sorted(all_ious, key=lambda o: -len(all_ious[o])):
        v = all_ious[op]
        print(f"  {op:48s} n={len(v):4d} mean={np.mean(v):.3f} frac>=0.9={np.mean(np.array(v) >= 0.9):.2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
