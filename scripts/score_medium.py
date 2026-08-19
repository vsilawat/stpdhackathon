from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from machineplan import brep, features, ipw, mesh
from machineplan import plan as planmod

GT = Path("data/MachinePlan-10K-gt")
BANDS = [(0.999, 35), (0.99, 25), (0.98, 20), (0.95, 15), (0.90, 10)]
def band_pts(iou: float) -> int: return next((p for t, p in BANDS if iou >= t), 0)

def gt_stls(part_id: str) -> list[Path]:
    return sorted(p for p in (GT / part_id).glob("[0-9]*.stl") if not p.name.startswith("000_"))

def score_part(part_id: str) -> tuple[int, int, list[float]]:
    part = brep.load(f"data/MachinePlan-10K/{part_id}/{part_id}.stp")
    found = features.extract(part)
    ops = planmod.plan(found)
    stls = gt_stls(part_id)
    pred = ipw.ipws(part, found, ops)
    ious = []
    for k in range(min(len(pred), len(stls))):
        gt_m = mesh.load_stl(stls[k])
        ious.append(mesh.iou(pred[k], gt_m))
    return len(ops), len(stls), ious

def main(n_sample: int = 30) -> int:
    parts = sorted(p.name for p in GT.iterdir() if p.is_dir())
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4][::len(parts) // 5 // n_sample][:n_sample]
    pts = Counter(); count_ok = 0; all_ious = []
    for pid in holdout:
        try: np_, ng, ious = score_part(pid)
        except Exception as e:
            print(f"{pid}: ERROR {e}"); continue
        count_ok += np_ == ng
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        all_ious.extend(ious)
        for i in ious: pts[band_pts(i)] += 1
        print(f"{pid}: ops {np_} vs gt {ng}  mean_iou={mean_iou:.5f}  min={min(ious, default=0):.5f}")
    n = len(all_ious)
    print(f"\ncount match {count_ok}/{n_sample}  ops scored {n}  mean_iou={sum(all_ious)/max(n,1):.5f}")
    print("pts bands:", dict(sorted(pts.items(), reverse=True)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
