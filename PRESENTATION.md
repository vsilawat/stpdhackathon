# Round 1 — 3-minute presentation

Target: **2:50 spoken**, leaving ~10s buffer. Script below is ~440 words at a
measured 155 wpm. Optional cuts are marked `[CUT IF LONG]`.

Rubric context this is written against:
- Round 1 requires the video + technical submission + a brief methods description.
- Round 2 scores originality, effectiveness/efficiency, documentation, generalizability, and **scientific rigor** ("reverse engineering or hacky solutions will be penalized").
- Section 4 requires **disclosing LLM use**. Covered on the closing slide.

---

## Slide outline

| # | Time | Slide | Visual |
|---|---|---|---|
| 1 | 0:00–0:15 | The problem | One part render (`isometric_shaded.png`) → arrow → a short operation list |
| 2 | 0:15–0:45 | The key idea | Two-column: "learn the mapping" ✗ vs "reconstruct the rulebook" ✓ |
| 3 | 0:45–1:30 | The discovery | **The money slide.** Four pocket diagrams showing 1 / 2 / 3 / 4 corner fillets, each labelled with the feature it implies |
| 4 | 1:30–2:05 | Results | Rubric table + "20,000/20,000 files valid" |
| 5 | 2:05–2:40 | Measured limits | Two panels: the in-process-state diagram, and the 82%/9% volume split |
| 6 | 2:40–3:00 | What we'd take forward + disclosure | One line of lesson, then the LLM disclosure block |

**Slide 3 is the one to spend design effort on.** It carries the originality
score. Draw the four cases as simple top-down outlines with the fillets
highlighted in a contrasting colour.

---

## Script

### Slide 1 — The problem `[0:00–0:15]`

> Given a CAD model of a metal part, predict how to actually machine it —
> which operations, in what order, and with which cutting tools.
> We worked from MachinePlan-10K: ten thousand parts and ninety-one thousand
> machining operations, generated in Siemens NX.

### Slide 2 — The key idea `[0:15–0:45]`

> One observation shaped everything we did. These labels weren't written by a
> person — they came from NX's deterministic knowledge base. So the target
> isn't fuzzy human judgment. It's a rulebook.
>
> So we set out to reconstruct the rules we could verify exactly, and to learn
> only the parts we genuinely couldn't. The result has no machine learning in
> most of its stages, and no third-party dependencies at all — including our
> own STEP reader. It plans all ten thousand parts in twenty seconds on a
> laptop.

### Slide 3 — The discovery `[0:45–1:30]`

> Here's the idea we're proudest of.
>
> A milling cutter is round. It physically cannot cut a sharp internal corner —
> it always leaves a fillet of exactly its own radius. So the corners it leaves
> behind are a fingerprint of what was cut.
>
> Count them, and you know the feature. One corner is a notch in the edge of
> the block. Two is a slot open at one end. Three, an open pocket. Four, a
> fully enclosed pocket. Each one needs a completely different operation.
>
> We were already measuring those corners. We simply weren't counting them.
> Adding that single number took our pocket and slot accuracy from forty-six
> percent to eighty-five.
>
> The same physical reasoning handles chamfers: on a 2.5-axis part, a chamfer
> is the only flat face that isn't axis-aligned. That test is correct on
> fifteen hundred out of fifteen hundred parts. `[CUT IF LONG]`

### Slide 4 — Results `[1:30–2:05]`

> Against the organizers' rubric, the Easy track scores fourteen out of twenty:
> an F1 of 0.93 on the operations, and a normalized edit distance of 0.204.
> Tool selection scores about ten out of twenty — and when we name the
> operation correctly, the tool diameter is exact, to a median relative error
> of zero.
>
> All twenty thousand of our submission files pass the organizers' own
> validator.

### Slide 5 — Measured limits `[2:05–2:40]`

> We also measured where we *can't* go, and why.
>
> Tool selection is capped. To choose a drill, NX looks at how wide the hole
> already is partway through the job. That's an in-process state — and it
> doesn't exist in the finished CAD model we're given as input. We measured the
> residual uncertainty at 0.93 bits, and our attempts to beat it made results
> worse, not better.
>
> We also chose not to attempt the mesh and G-code tracks. Eighty-two percent
> of all material removed is pockets and slots; drilling is only nine. Those
> tracks score zero below a threshold — so submitting just the drilling half,
> which we could do well, would have scored zero. We measured that before
> deciding, not afterwards.

### Slide 6 — What we'd take forward `[2:40–3:00]`

> One caution against our own judgment. We initially wrote off operation
> ordering as a coin flip. Testing it properly turned it into one of our
> largest gains — eighty-nine percent accurate, driven by the depth of the
> shallowest hole.
>
> The lesson: when the data comes from a rule engine, the wins come from
> finding structure that is already there.

**On-screen disclosure block (read only the first line aloud):**

> Claude Opus 5 (Anthropic) was used to write code and documentation.
> All submitted outputs were generated by our own deterministic pipeline;
> no model produced the easy, medium, or hard outputs directly.

---

## Delivery notes

- **Pace.** 155 wpm is brisk but natural. Rehearse Slide 3 slowest — it carries
  the originality score and the four-case list needs beats between items.
- **The strongest 12 seconds** are "We were already measuring those corners. We
  simply weren't counting them." Pause before it.
- **Say a number, show a number.** Every figure spoken should be on screen.
- **Don't apologise for the skipped tracks.** Frame them as a measured decision
  — that reads as engineering judgment, which Round 2 scores. The 82%/9% split
  is the evidence; lead with it.
- If you overrun, cut the chamfer sentence on Slide 3 first, then compress
  Slide 1 to a single line.

## Rigor framing (important)

Rubric 3.2 penalizes "reverse engineering or hacky solutions." Our approach
reconstructs NX's rules, so state the defence plainly — one line on Slide 2 or
in the written methods description:

> Rules are learned from the training labels only. At inference the pipeline
> reads **nothing but the CAD file** — no toolpaths, no operation cards, no
> ground-truth metadata. This is ordinary supervised learning; the rules are
> the model.

This is true of the code as written (`predict()` parses only the `.stp`), and
it is worth saying explicitly before a judge wonders.

## Written methods description

The rubric asks for a brief methods description alongside the video. `README.md`
already covers approach, confirmed-exact facts with evidence, results, known
data issues, and limitations. Suggested additions if you want it standalone:
a one-paragraph literature note (MFCAD / MFCAD++ / BrepMFR do feature
*recognition*; this problem is process *planning*, which is a harder target),
and a generalizability paragraph — the parts are stock blocks minus features
with planar and cylindrical faces only, so the fillet-counting rule would need
extending for freeform geometry.
