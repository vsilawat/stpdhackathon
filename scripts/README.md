# Scripts

All scripts run from the repo root: `.venv/bin/python scripts/<name>.py`. They fall into five
stages; the main path is **dump → mine → train → generate → score**. Scripts marked *experiment*
are one-off analyses/ablations kept for provenance and are not on the submission path.

## Pipeline order (reproduce from scratch)

```bash
python scripts/dump_features.py        # B-Rep features for all parts   -> derived/features.jsonl
python scripts/mine_details.py         # GT op/tool table               -> derived/opdetails.csv, tool_library.csv
python scripts/mine_sequences.py       # GT op sequences + order stats  -> derived/sequences.jsonl
python scripts/mine_chains.py          # per-feature op chains          -> derived/chains.csv
python scripts/mine_pockets.py         # pocket <-> GT op matching      -> derived/pocket_train.csv
python scripts/mine_ptp.py             # GT canned-cycle table          -> derived/ptp_cycles.csv
python scripts/mine_point_angles.py    # drill point-angle lookup       -> derived/point_angles.json
python scripts/train_chains.py         # hole-chain classifier          -> derived/chain_tree12.pkl
python scripts/train_tools.py          # per-op tool-diameter RFs       -> derived/tool_dia_models.pkl
python scripts/train_pocket_models.py  # pocket tool + kind models      -> derived/pocket_tool_model.pkl, pocket_kind_model.pkl
python scripts/train_order.py          # block-order + hole-mill models -> derived/order_model.pkl, hm_model.pkl
python scripts/generate_submission.py  # all four tracks                -> submission/
python scripts/score_easy.py           # holdout scorers (i%5==4 split)
python scripts/score_tools.py
python scripts/score_medium.py
python scripts/score_ptp.py
```

`generate_submission.py` flags: `--data=<dir>` (default `data/MachinePlan-10K`), `--out=<dir>`
(default `submission`), `--force` (regenerate parts that already look complete). The test-set
submission was produced with `--data=data/test30 --out=test_submission`.

## Index

| script | stage | what it does |
|---|---|---|
| `dump_features.py` | dump | Runs the B-Rep recognizer on every STEP file; one JSON row per part (stock, holes, pockets, chamfers) → `derived/features.jsonl` |
| `dump_pocket_dims.py` | dump | Recomputes single-pocket w/l from the outline polygon, paired with GT final tool diameter → `derived/pocket_dims.jsonl` |
| `mine_details.py` | mine | Parses every GT `*_details.txt` into a flat op/tool table → `derived/opdetails.csv`, `derived/tool_library.csv` |
| `mine_sequences.py` | mine | GT op-sequence analysis: precedence, block contiguity, top sequences → `derived/sequences.jsonl` |
| `mine_chains.py` | mine | Groups ops by machined feature; characterizes op/tool/diameter chains per feature family → `derived/chains.csv` |
| `mine_pockets.py` | mine | Matches GT pocket-mill ops to detected pockets via `.ptp` XY-in-polygon; emits training rows → `derived/pocket_train.csv` |
| `mine_ptp.py` | mine | Parses GT `.ptp` canned cycles (Z/R/Q/F, gun-drill blocks) joined to detected holes → `derived/ptp_cycles.csv` |
| `mine_point_angles.py` | mine | Drill point-angle lookup keyed by `op_name\|diameter` → `derived/point_angles.json` |
| `mine_order_laws.py` | mine | Tests hand-written block-order laws against GT; prints analysis only |
| `train_chains.py` | train | Whole-chain classifier (hole features → op-chain string), tree vs forest on holdout → `derived/chain_tree12.pkl` |
| `train_tools.py` | train | One RF per op name predicting tool-diameter class on the discrete tool grid → `derived/tool_dia_models.pkl` |
| `train_pocket_models.py` | train | Production pocket models: RF+HistGB diameter ensemble + pocket-op-kind RF → `derived/pocket_tool_model.pkl`, `pocket_kind_model.pkl` |
| `train_order.py` | train | Two soft-voting ensembles: drilling-block-first and hole-mill placement → `derived/order_model.pkl`, `hm_model.pkl` |
| `train_pocket_tools.py` | train | **Superseded — do not run.** Earlier single-pocket RF; overwrites `pocket_tool_model.pkl` with a payload the current loader cannot read |
| `generate_submission.py` | generate | Main driver: STEP → features → plan → all four tracks per part, process pool with resume |
| `regen_ptp.py` | generate | Regenerates only the `hard_ptp` + `hard_tools` tracks (after `ptp.emit` changes) |
| `score_easy.py` | score | Easy track on holdout: normalized Levenshtein, multiset F1, exact-match, band histogram |
| `score_medium.py` | score | Medium track: builds IPW solids, mesh IoU vs GT STLs, maps to rubric bands (sample size = argv[1]) |
| `score_tools.py` | score | Hard-tools track: positional tool type + diameter-within-2% vs GT, per-type breakdown |
| `score_ptp.py` | score | Hard-PTP track: matches drill cycles by op + XY, reports cycle/Z/R match rates |
| `score_ptp_iou.py` | score | Stronger PTP scorer: arc-aware toolpath parser, swept-volume heightfield, per-op volumetric IoU |
| `chain_decomp.py` | experiment | Chain decomposition + optimal hole↔chain assignment; rules vs trees vs hybrid → `derived/chain_subtrees.pkl` |
| `sweep.py` | experiment | Feature-detector calibration vs the counts published in Dataset_Description.pdf §6.6 → `derived/sweep.jsonl` |
| `mine_flat.py` | experiment | Checks detected flat-bottom blind holes against GT hole-mill op counts |
| `order_exp.py` | experiment | Feature-set ablation for the block-order classifiers |
| `pocket_exp.py` | experiment | Model ablation for pocket tool diameter (RF vs HGB vs ensemble) |
| `dia_breakdown.py` | experiment | Per-op-name diameter error breakdown with top confusions |
| `oracle_easy.py` | experiment | Upper-bound ablation: oracle block-order choice vs the classifier |
| `tools_variant.py` | experiment | In-memory tools scorer straight from the planner, for fast iteration |
| `regen_ptp_sample.py` | experiment | `regen_ptp` on a small holdout sample for a fast iterate-and-score loop |

Notes: every train/score script uses the same holdout convention (parts sorted, `index % 5 == 4`).
Three small artifacts consumed by `src/machineplan/` (`chain_dia_lookup.json`,
`gun_pilot_lookup.json`, `prebore_mill_dia.pkl`) plus `chain_tree.pkl` were produced by earlier
interactive sessions and are shipped as data (see the top-level README for `chain_tree.pkl`).
