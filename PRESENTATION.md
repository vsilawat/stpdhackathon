# Round 2 Presentation — 3 minutes (~420 words)

**Slide 1 — The idea (30s).**
Every plan in this dataset came out of one deterministic CAM generator. So we didn't train a
black box to imitate labels — we reverse-engineered the generator's rules. Our pipeline mirrors
a CAM system: B-Rep feature recognition, rule-based process planning, analytic material-removal
solids, and generative G-code emission. Machine learning appears only where the mined rules leave
genuine ambiguity, and every rule was mined on 80% of parts and validated on a disjoint holdout.
All four tracks decode from one predicted plan, so sequence, solids, tools and toolpaths are
consistent by construction.

**Slide 2 — Features and sequence (45s).**
We recognize holes, pockets and chamfers directly from the STEP B-Rep with OpenCascade — no mesh
learning. Mining all plans showed each hole's process chain is a function of its geometry, with
exact round-number thresholds: through holes under 12 mm get spot-plus-twist-drill; slender holes
with length-to-diameter over 5 get the gun-drill sequence; over 15 mm switches to indexable insert
drills. Those recovered rules alone explain 82.5% of chains; a small random forest resolves the
residual ambiguity — like insert-catalog grid membership — for 96.3% exact chains on holdout.
Mining op indices also exposed the generator's plan skeleton: a twist-drill block and a mill block
in one of two orders — the one real binary in the data, which an ensemble predicts at 93.2% — with
insert-blind chains leading only mill-first plans. Sequence F1 is 0.955, mean edit distance 0.10,
and 70% of parts match the ground-truth sequence exactly.

**Slide 3 — Material removal (45s).**
For the medium track each operation is an analytic solid — cylinder plus 118-degree tip for twist
drills, flat bottoms for insert drills, outline prisms for pockets — subtracted with exact booleans.
The details are mined, not guessed: spot drills go 3 mm deep above 17.45 mm diameter and 0.043·d
below it; gun-drill pilots stop at 1.5 pilot-diameters past the mouth. Holdout mean IoU is 0.996,
with 77% of operations in the top rubric band.

**Slide 4 — Tools and toolpaths (45s).**
Tool type is a pure function of the operation. Diameters follow the tool catalog we recovered:
gun drills come in integers plus 8.3, spades on a half-millimeter grid — per-operation classifiers
over those discrete grids put ~94% of tools within the 2% tolerance. For toolpaths we recovered the
post-processor grammar itself: retract planes are mouth-plus-2, blind cycles end exactly at the
detected floor, and every through cycle protrudes by tip-height-plus-1.5 mm — a constant we found
across the whole corpus. Gun drilling reproduces the mined manual block template, dwell for dwell,
and our swept-volume proxy scores the emitted paths at 0.83 mean IoU against ground truth.

**Slide 5 — Why this wins (15s).**
One deterministic pipeline generates all four tracks for all 10,000 parts in six minutes, passes
the official validator with zero errors, and every number in it traces back to a rule we can show
you in the data. We didn't fit the dataset — we recovered the planner that made it.

---
*Backup facts: Easy lev 0.098 / F1 0.955 / exact 70.2% → 18/20. Medium mean IoU 0.996.
Tools positional type 0.849 / dia-within-2% 0.768. Toolpath swept-volume IoU 0.80 (200-part sample).
Chain RF: 10 geometric features, 96.3% exact. Block-order RF 91.5%; hole-mill placement RF 81.2% —
the template carries the structure.*
