# CAM process-plan prediction from CAD — MachinePlan-10K

Predicts a CNC machining plan (operation sequence, tool selection, timing) from
a CAD part file. Scope: 2.5-axis planar milling.

Dataset: [MachinePlan-10K](https://doi.org/10.5281/zenodo.21653081) —
10,000 parts, 91,702 machining operations, generated in Siemens NX.

## Submission (organizers' format)

The competition splits 100 points across three tracks. We target Easy and Hard;
Medium is deliberately skipped (see below).

| Track | Points | Output | Status |
|---|---|---|---|
| Easy | 20 | operation sequence (`<part>_sequence.json`) | **14/20**, 10,000/10,000 files valid |
| Medium | 35 | IPW mesh per operation (`.stl`) | **not attempted** |
| Hard | 45 | tool type + diameter (20) and G-code toolpath (25) | tools **~9.3/20**, G-code not attempted |

Medium was skipped by choice: it is scored by geometric overlap with a floor at
IoU 0.90, below which it scores zero, so a partial attempt is worth nothing.

```bash
python3 scripts/export_easy.py      # -> submission/easy/<part>_sequence.json
python3 scripts/export_hard.py      # -> submission/hard/<part>_tools.json
python3 scripts/score_official.py   # validates all files + Easy rubric score
python3 scripts/score_hard.py       # Hard tool-selection rubric score
```

Both exports are checked with the organizers' own `validate_submission.py`
(imported directly, run over all 10,000 files): **20,000/20,000 valid**.

### Easy track: 14/20

| metric | value | points |
|---|---|---|
| mean normalized Levenshtein | 0.2040 | 6/10 |
| mean F1 | 0.9321 | 8/10 |

Averaging per-part points instead of banding the mean gives **15.76/20**; the
rubric does not say which aggregation it uses.

The Levenshtein band boundary is 0.20 and we finish at 0.2040 — close, but on
the wrong side. Perfect block ordering would reach 0.1376, so the headroom
exists; we captured about 70% of it and the last stretch did not come.

### Hard track: tool selection

Tool type correct at the same position: **64.4%** (~10.2/20). When the type is
right the diameter is essentially exact — **median relative error 0.00%** — so
tool *knowledge* is solved and the score is limited by sequence alignment, not
by tool choice.

### The shared bottleneck: block order

Both tracks are dominated by one decision: does the plan start with drilling or
with milling? NX keeps each kind in a contiguous block (93.3% of parts) but the
order between them is near a coin flip (58/42). Gradient-boosted stumps on
geometric features predict it at **87.8%** (57.5% majority baseline).

The decisive feature was `min_hole_depth` — the depth of the *shallowest* hole
— with a single-feature AUC of 0.88. Max hole depth, which we tried first, is
far weaker at 0.69.

Progress from this one classifier:

| | Levenshtein | Hard tool type |
|---|---|---|
| always drill-first | 0.352 | 58.4% |
| logistic, 14 features (78.8%) | 0.250 | — |
| + strong features (86.3%) | 0.222 | 61.6% |
| + gradient boosting (87.8%) | 0.214 | 62.7% |
| + medoid chain selection | 0.211 | 63.4% |
| + tuned + plan-composition features + ensemble (89.2%) | **0.204** | **64.4%** |

Perfect block ordering would reach 0.1376, so roughly 30% of the original gap
remains.

`CHAIN_ALPHA` (default 1.0) trades a little edit distance for better operation
counts: the pure medoid minimises edit distance but runs ~5% short on chain
length, which costs F1. The rubric weights both equally.

**Medoid chain selection**: when several operation chains were seen for the
same feature, we now pick the one minimising expected edit distance rather than
the most frequent one. The plain mode is biased toward short chains, which made
us systematically under-predict operation counts.

## Quick start

```bash
python3 scripts/extract.py                    # unpack the dataset (~6.4 GB)
python3 scripts/analyze.py                    # build derived/operations.csv, parts.csv
python3 scripts/tool_library.py               # build derived/tools.csv, op_details.csv
python3 scripts/baseline.py fit               # train the rule-based baseline
python3 scripts/baseline.py submit derived/predictions   # predict all 10,000 parts
python3 scripts/evaluate.py                   # score on held-out parts
```

No third-party dependencies — standard library only. Runs in ~20 s for all
10,000 parts.

## Approach

The dataset's labels come from NX's deterministic knowledge base, so much of the
pipeline is a *rule engine*, not a learning problem. The baseline reproduces the
rules we confirmed exactly and falls back on learned priors only where geometry
extraction is still unsolved.

**Confirmed exact (reimplemented directly):**

| fact | evidence |
|---|---|
| chamfer features == non-axis-aligned planar faces | 1500 / 1500 parts, zero deviation |
| every chamfer → one `AREA_MILL` with chamfer mill `UGT0205_001` | 20,067 / 20,067 operations |
| holes == closed (360°) cylindrical faces | matches G-code drill positions in 84.7% of parts |
| hole diameter == diameter of the tool that cut it | median error 0.000 mm |
| operation name → tool class `(Type, SubType)` | 29 / 29 operations, 100% |
| tool is determined by NX's recorded tool query | H(tool \| query) = 0.13 bits; 96.7% unique |

**Recovered by aligning labels to geometry:**

- Each drilling / cylinder-milling step is matched to the *specific hole it
  cuts*, using the X/Y positions in its G-code. This yields true per-hole
  operation chains instead of guessing from diameter.
- Pockets, slots and notches are found by two complementary detectors:
  **corner-fillet clusters** (same radius, height, depth) and **floor faces**
  (a horizontal face that is not the outermost on its side is the bottom of
  something cut into the part). Fillets alone miss through-slots, which are
  open at both ends; floors alone over-count, because blind holes have flat
  bottoms too. Combined, feature count is exact on **85.0%** of parts and
  within one on **99.9%**.
- The **number of rounded corners identifies the feature shape**, which is
  what selects the operation: 1 → corner notch (73%), 2 → slot open at one
  end (68%), 3 → open pocket (71%), 4 → enclosed pocket (78%). The fillet
  radius is also the endmill radius, which fixes the tool.

**Learned prior (the remaining weak link):**

- `(diameter, through/blind, depth band) → operation chain`, with back-off to
  less specific keys for unseen combinations.

## Results (2,000 held-out parts)

| metric | score |
|---|---|
| chamfer step count exactly right | **100.0%** |
| chamfer contour steps F1 | **100.0%** |
| step F1 (right operations, order-free) | **85.1%** |
| pocket/slot/notch steps F1 | **85.2%** |
| tool F1 | 70.0% |
| drilling steps F1 | 63.6% |
| milled-hole steps F1 | 30.3% |
| sequence similarity | 53.4% |
| whole plan exactly right | 18.4% |

Mean absolute step-count error: **0.87** operations (median 0, p90 0).

Progress on step F1: 54.7% → 67.3% (G-code hole alignment) → 75.2%
(fillet clustering) → 76.2% (floor-face detection) → **85.1%**
(corner count as a shape discriminator).

## Where the remaining error is

1. **Tool F1 (70.2%) lags step F1 (85.1%), and this is largely a hard limit.**
   Of the steps where we name the operation correctly, 78.7% also get the
   right tool. We tried re-deriving the tool from the feature's measured
   dimensions instead of memorising it, and it did *not* help
   (66.5%; the hybrid that only re-derives on a fallback recovers 70.2%).

   The reason: NX's tool query gates on the hole diameter *before* the cut,
   not the final one. For enlarging operations the query's lower diameter
   bound matches a tool used earlier on the same part in 28.7% of cases —
   it is reading an **in-process state that the finished CAD model does not
   contain**. Residual uncertainty is H(tool | operation, hole diameter)
   = 0.93 bits, and adding position-in-chain only reduces it to 0.86.

   Copying a coherent tool *sequence* from a matched training example beats
   predicting tools independently, because the sequence implicitly encodes
   those intermediate states. Further gains here most likely require
   predicting the in-process workpiece, not a better tool rule.
2. **Milled holes (`CylinderMilling`) sit at 30.3%** — only 3.8% of steps, so
   low priority, but they are currently the worst-served category.
3. **Drilling chains at 63.6%.** Holes of the same diameter and depth still
   receive different chains in the source data.
4. **Drill-block vs mill-block order is near-random** (56 / 44 split in the
   data), which caps sequence similarity regardless of everything else. Not
   worth optimising.

## Repo layout

```
.
├── README.md                  this file — approach, evidence, results
├── SUMMARY_FOR_TEAM.md        non-technical summary for teammates
│
├── scripts/                   the pipeline (stdlib only)  — see scripts/README.md
│   ├── step_parse.py            dependency-free STEP AP214 reader
│   ├── extract.py               unpack the archive, skipping ASCII-STL dupes
│   ├── analyze.py               process-plan vocabularies and ordering structure
│   ├── tool_library.py          tool catalog + NX tool-selection queries
│   ├── tool_rules.py            decodes the queries; tool == f(query)
│   ├── cad_feature_link.py      validates CAD geometry against the CAM labels
│   ├── baseline.py              the predictor (fit / predict / submit)
│   ├── block_order.py           drill-first vs mill-first classifier
│   ├── export_easy.py           -> submission/easy/
│   ├── export_hard.py           -> submission/hard/
│   ├── score_official.py        organizers' validator + Easy rubric score
│   ├── score_hard.py            Hard tool-selection rubric score
│   ├── evaluate.py              internal fine-grained metrics
│   └── check_predictions.py     self-consistency checks (mutation-tested)
│
├── derived/                   generated tables and models — see derived/README.md
│   ├── operations.csv           91,702 operations
│   ├── op_details.csv           operation cards incl. tool-selection queries
│   ├── tools.csv                the 431-tool catalog
│   ├── parts.csv                per-part summary
│   ├── baseline_model.json      learned lookups + train/val split
│   ├── block_order.json         the ordering classifier
│   ├── o1o2_map.json            NX operation name -> (o1, o2)
│   └── predictions/             10,000 plans, internal format
│
├── submission/                the deliverable — see submission/README.md
│   ├── easy/                    10,000 x <part>_sequence.json
│   └── hard/                    10,000 x <part>_tools.json
│
├── data/                      NOT IN GIT (11 GB) — see data/README.md
└── reference/                 NOT IN GIT — the organizers' repo, clone it:
```

Two directories are not committed, for reasons other than preference:

```bash
# the dataset: 11 GB, beyond GitHub's limits (see data/README.md)
python3 scripts/extract.py     # after downloading the Zenodo archive

# the organizers' repo: not ours to redistribute, 21 MB of slide decks,
# and it carries its own .git
git clone https://github.com/athulcd/ASME-CIE-Student-Hackathon-2026-Problem-1.git reference
```

Everything else — scripts, generated tables, trained models, and all 20,000
submission files — is committed, so the repository can be inspected and the
submission reviewed without downloading anything.

## Known data issues

- ~2% of operation transitions show in-process workpiece volume *increasing*
  slightly (max 471 mm³ against ~1.1 × 10⁷ mm³ parts). This is STL tessellation
  noise, not corrupt data — clamp to zero if using volume deltas as a target.
- 9 of the 29 operation types occur fewer than 15 times in the whole dataset
  and are effectively unlearnable.
- Part IDs run to `featured_part_11416` with gaps; there are exactly 10,000
  folders. Do not assume ID == index.
- `*_text.stl.txt` files are ASCII duplicates of the binary `.stl` meshes
  (~28 GB of the 35 GB uncompressed) and can be skipped entirely.
