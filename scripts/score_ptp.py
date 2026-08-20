from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

GT = Path("data/MachinePlan-10K-gt")
SUB = Path("submission/hard_ptp")
CYCLE = re.compile(r"G(81|73|85) G98 Z(-?[\d.]+) F[\d.]+(?: Q([\d.]+))? R(-?[\d.]+)")
XY = re.compile(r"G0 G90 X(-?[\d.]+) Y(-?[\d.]+)")
GUNZ = re.compile(r"^N\d+ Z(-?[\d.]+) F125\.$", re.MULTILINE)

def parse(path: Path):
    t = path.read_text()
    m = re.search(r"^\((\w+) *, *TOOL", t, re.MULTILINE)
    op = m.group(1) if m else "?"
    xy = XY.search(t)
    c = CYCLE.search(t)
    if c: return op, xy and (float(xy.group(1)), float(xy.group(2))), ("G" + c.group(1), float(c.group(2)), float(c.group(4)), float(c.group(3)) if c.group(3) else None)
    g = GUNZ.findall(t)
    if g: return op, xy and (float(xy.group(1)), float(xy.group(2))), ("GUN", float(g[-1]), None, None)
    return op, xy and (float(xy.group(1)), float(xy.group(2))), None

def main(n_sample: int = 50) -> int:
    parts = sorted(p.name for p in GT.iterdir() if p.is_dir())
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4]
    holdout = holdout[::max(len(holdout) // n_sample, 1)][:n_sample]
    stats = defaultdict(Counter)
    zerr = defaultdict(list)
    for pid in holdout:
        sd = SUB / pid
        if not sd.is_dir(): continue
        gt_ops, my_ops = defaultdict(list), defaultdict(list)
        for f in (GT / pid).glob("[0-9]*.ptp"):
            op, xy, cyc = parse(f)
            if xy and cyc: gt_ops[op].append((xy, cyc))
        for f in sd.glob("*.ptp"):
            op, xy, cyc = parse(f)
            if xy and cyc: my_ops[op].append((xy, cyc))

        def basename(o): return o.rsplit("_", 1)[0] if o.split("_")[-1].isdigit() else o
        my_by_base = defaultdict(list)
        for op, items in my_ops.items(): my_by_base[basename(op)].extend(items)
        for op, items in gt_ops.items():
            base = basename(op)
            for (gx, gy), (gc, gz, gr, gq) in items:
                cands = [(abs(x - gx) + abs(y - gy), c) for (x, y), c in my_by_base.get(base, [])]
                if not cands or min(cands)[0] > 0.5:
                    stats[base]["missing"] += 1; continue
                mc, mz, mr, mq = min(cands)[1]
                stats[base]["n"] += 1
                stats[base]["cyc_ok"] += mc == gc
                stats[base]["z_ok"] += abs(mz - gz) < 0.01
                stats[base]["r_ok"] += gr is None or mr is None or abs(mr - gr) < 0.01
                zerr[base].append(mz - gz)
    for op in sorted(stats, key=lambda o: -stats[o]["n"]):
        s, ze = stats[op], zerr.get(op, [0])
        n = max(s["n"], 1)
        print(f"{op:48s} n={s['n']:4d} miss={s['missing']:3d} cyc={s['cyc_ok']/n:.3f} "
              f"z={s['z_ok']/n:.3f} r={s['r_ok']/n:.3f} dz_mean={sum(ze)/max(len(ze),1):+.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 50))
