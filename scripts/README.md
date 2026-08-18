# scripts/ — the pipeline

Standard library only. No third-party dependencies, including for reading
STEP files. Full run is about a minute end to end.

## Run order

```bash
python3 scripts/extract.py         # unpack the dataset          (~5 min, once)
python3 scripts/analyze.py         # -> derived/operations.csv, parts.csv
python3 scripts/tool_library.py    # -> derived/tools.csv, op_details.csv
python3 scripts/baseline.py fit    # -> derived/baseline_model.json
python3 scripts/block_order.py     # -> derived/block_order.json
python3 scripts/baseline.py fit    # refit so chains see the ordering model
python3 scripts/export_easy.py     # -> submission/easy/
python3 scripts/export_hard.py     # -> submission/hard/
python3 scripts/score_official.py  # validate + score Easy
python3 scripts/score_hard.py      # score Hard tool selection
```

## What each file does

| file | role |
|---|---|
| `step_parse.py` | Dependency-free STEP AP214 reader. The parts use only `PLANE` and `CYLINDRICAL_SURFACE` faces bounded by `LINE` and `CIRCLE` edges, so a ~200-line parser suffices and OpenCASCADE is not needed. Also classifies full vs partial cylinders and measures axial extent. |
| `extract.py` | Unpacks the archive, skipping the redundant ASCII-STL duplicates (35 GB → 6.4 GB). |
| `analyze.py` | Flattens every process plan to CSV; reports operation/tool vocabularies, ordering structure, and per-part statistics. |
| `tool_library.py` | Parses the 91,702 operation cards into a tool catalog and a per-operation parameter table, including the literal NX tool-selection `Query` strings. |
| `tool_rules.py` | Decodes those queries. Establishes that the tool is a deterministic function of the query (H(tool \| query) = 0.13 bits). |
| `cad_feature_link.py` | Validates CAD geometry against the CAM labels — the three hypotheses linking faces to features. |
| `baseline.py` | The predictor. `fit` / `predict <part_id>` / `submit <dir>`. Contains feature detection, chain lookup with back-off, tool resolution, and sequencing. |
| `block_order.py` | Predicts whether a plan starts with drilling or milling. Pure-Python logistic regression and gradient-boosted stumps; keeps whichever wins on held-out data. |
| `export_easy.py` | Writes the Easy submission (`<part>_sequence.json`). |
| `export_hard.py` | Writes the Hard tool submission (`<part>_tools.json`). |
| `score_official.py` | Runs the organizers' validator over every file, then scores Easy against the rubric. |
| `score_hard.py` | Scores Hard tool selection against the rubric. |
| `evaluate.py` | Internal metrics on the fine-grained (pre-mapping) predictions. |
| `check_predictions.py` | Self-consistency checks on generated files. Mutation-tested: 12 injected defects, all detected. |

## Notes

- `CHAIN_ALPHA` (default `1.0`) trades edit distance against operation-count
  accuracy when choosing a representative operation chain. The rubric weights
  both equally. Set to `0` for the pure minimum-edit-distance choice.
- Train/validation split is deterministic: every fifth part by sorted ID goes
  to validation. No randomness anywhere in the pipeline, so runs reproduce
  exactly.
- Hyperparameters for `block_order.py` are selected on an inner split of the
  *training* set, never on validation, so reported accuracy stays honest.
