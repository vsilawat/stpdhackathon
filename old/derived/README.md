# derived/ — generated tables and models

Everything here is **regenerated from `data/`** by the scripts; nothing is
hand-edited. It is committed so the repository is inspectable without first
downloading 11 GB, and so results are reproducible against a fixed snapshot.

| file | size | produced by | contents |
|---|---|---|---|
| `operations.csv` | 17 MB | `analyze.py` | one row per machining operation (91,702): part, sequence, name, type, tool, times, path lengths, volumes |
| `parts.csv` | 406 KB | `analyze.py` | one row per part (10,000): operation count, tool changes, total times |
| `op_details.csv` | 37 MB | `tool_library.py` | one row per operation card: template, order/method/geometry group, feed rate, stock, tolerances, and the literal NX tool-selection query |
| `tools.csv` | 45 KB | `tool_library.py` | the 431-tool catalog with geometry (diameter, flute length, point angle, flutes, material) |
| `baseline_model.json` | 579 KB | `baseline.py fit` | learned chain lookups, tool maps, operation metadata, and the train/val split |
| `block_order.json` | 4.7 MB | `block_order.py` | the drill-first-vs-mill-first classifier (boosted stumps + linear ensemble) |
| `o1o2_map.json` | 2 KB | (built inline) | NX operation name → the organizers' `(o1, o2)` categories. All 29 names map to exactly one pair. |
| `tool_type_map.json` | 190 B | `export_hard.py` | NX `(Type, SubType)` code → the organizers' 8 tool types |
| `predictions/` | 40 MB | `baseline.py submit` | 10,000 plans in our **internal** format, used by `evaluate.py` and `check_predictions.py` |

## Internal format vs submission format

`predictions/` is *not* the competition deliverable. It uses our own schema
(fine-grained NX operation names, catalog tool IDs, per-step times) and exists
for internal measurement. The actual deliverable is in `submission/`, which
uses the organizers' coarse `o1`/`o2` vocabulary and their required field
names.

Keeping both is deliberate: the internal format retains detail the submission
format discards, which is what makes the per-category diagnostics possible.

## Regenerating

```bash
python3 scripts/analyze.py && python3 scripts/tool_library.py
python3 scripts/baseline.py fit && python3 scripts/block_order.py
python3 scripts/baseline.py fit && python3 scripts/baseline.py submit derived/predictions
```

Deterministic — no seeds, no randomness. The same input reproduces byte-identical output.
