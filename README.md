# ASME-CIE-Student-Hackathon-2026-Problem-1

Dataset: https://doi.org/10.5281/zenodo.21653081  
Kickoff Slides: [Kickoff PPT](Hackathon_Kickoff_Problem1.pptx)   
Data Tutorial: [Tutorial PPT](Tutorial_Hackathon_Problem1.pptx)      
Rubrics: [Rubrics](Rubrics.pdf)   
Dataset Description: [Dataset Description](Dataset_Description.pdf)

## Final submission (all four tracks)

The current pipeline (`src/machineplan/`, documented in [METHODS.md](METHODS.md))
generates every track for all 10,000 parts; all 40,000 files pass the
organizers' `validate_submission.py` with zero errors.

| Track | Points | Holdout result |
|---|---|---|
| Easy | 20 | mean normalized Levenshtein **0.098**, set-F1 **0.955**, exact sequence 70.2% |
| Medium | 35 | mean IPW IoU **0.996** |
| Hard: tools | 20 | positional tool type 84.9%, diameter within 2%: 76.8% |
| Hard: toolpath | 25 | mined post-processor grammar; estimated swept-volume IoU **0.80** on a 200-part sample |

```bash
.venv/bin/python scripts/generate_submission.py   # writes submission/{easy,medium,hard_tools,hard_ptp}
.venv/bin/python scripts/score_easy.py            # holdout scorers
.venv/bin/python scripts/score_tools.py
.venv/bin/python scripts/score_medium.py
.venv/bin/python scripts/score_ptp.py
```

An earlier standard-library baseline (Easy 0.204, tools 64.4%, Medium skipped)
was superseded by this pipeline; its scripts remain under `scripts/` for
provenance but are not part of the submission path.

Note: four trained model pickles exceed GitHub's 100 MB file limit and are
gitignored (`derived/chain_tree.pkl`, `chain_tree12.pkl`, `tool_dia_models.pkl`,
`pocket_tool_model.pkl`). The committed `submission/` files were generated with
them; to regenerate from scratch, retrain via the `scripts/train_*.py` scripts.
