from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from machineplan import plan as planmod


def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def f1(pred: list, true: list) -> float:
    p, t = Counter(pred), Counter(true)
    tp = sum(min(p[k], t[k]) for k in p)
    return 2 * tp / (len(pred) + len(true)) if pred or true else 1.0

def load_gt(path: Path = Path("derived/opdetails.csv")) -> dict[str, list[tuple[str, str]]]:
    gt: dict[str, list[tuple[int, str, str]]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            gt.setdefault(row["part"], []).append((int(row["op_index"]), row["template_type"], row["template_subtype"]))
    return {part: [(o1, o2) for _, o1, o2 in sorted(rows)] for part, rows in gt.items()}

def load_features(path: Path = Path("derived/features.jsonl")) -> dict[str, dict]:
    return {row["part"]: row for line in path.open() if (row := json.loads(line))}

def predict(row: dict) -> list[tuple[str, str]]:
    ops = planmod.plan_from_row(row)
    return [(op.o1, op.o2) for op in ops]

def main() -> int:
    gt, feats = load_gt(), load_features()
    parts = sorted(gt)
    holdout = [p for i, p in enumerate(parts) if i % 5 == 4]
    lev_sum = f1_sum = n = exact = 0
    band = Counter()
    for part in holdout:
        true = gt[part]
        pred = predict(feats[part])
        lev = levenshtein(pred, true) / max(len(pred), len(true), 1)
        score = f1(pred, true)
        lev_sum += lev; f1_sum += score; n += 1; exact += lev == 0
        band[min(int(lev * 10), 5)] += 1
    print(f"holdout n={n}  mean_lev={lev_sum/n:.4f}  mean_f1={f1_sum/n:.4f}  exact={exact/n:.3f}")
    print("lev bands (x0.1):", dict(sorted(band.items())))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
