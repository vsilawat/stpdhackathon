# Where we're at — plain English

> **Update (Aug 19, post-deadline submission):** this document describes the
> earlier standard-library baseline. The final pipeline (`src/machineplan/`,
> see [METHODS.md](METHODS.md)) attacks **all four tracks**: the pocket-outline
> capability flagged below as the blocker was built with OpenCascade B-Rep
> analysis, so Medium and the G-code track are attempted honestly — Easy edit
> distance 0.098, Medium mean IoU 0.996 with 77% of ops in the top band, tools
> 84.9%/76.8%, toolpath swept-volume IoU ~0.80. The constant-mesh shortcut
> measured below stayed unexploited; it now serves as the calibration baseline
> in METHODS.md showing our Medium number reflects real modelling.

## The problem

Given a 3D model of a metal part, predict how a CNC machine should actually
make it: which machining steps, in what order, and with which cutting tools.

We're working from a public dataset of **10,000 parts**, each paired with the
complete "correct answer" — 91,702 machining steps in total — produced by
Siemens NX, a professional manufacturing software package.

## What we've built

A program that reads a 3D part file and writes out a full machining plan. It
covers all 10,000 parts in about **20 seconds**, never crashes, and needs no
special software installed — just plain Python. That last point matters: there
is nothing that can break on someone else's laptop the night before the
deadline.

We train it on 8,000 parts and score it on 2,000 it has never seen.

## How well it works

We now submit in the organizers' official format, and score against their own
rubric. **All 20,000 submission files pass their validator.**

| Track | Available | Ours |
|---|---|---|
| Easy — the sequence of machining steps | 20 | **14** |
| Hard — tool type and size | 20 | **~10** |
| Hard — G-code toolpaths | 25 | not attempted |
| Medium — 3D shape after each step | 35 | not attempted |

Roughly **24 of 100**, banked and validated.

Underlying quality, on parts the system has never seen:

| What we measure | Score |
|---|---|
| Getting the bevelled-edge steps right | **100%** |
| Getting the pocket / slot / notch steps right | **85%** |
| Getting the right machining steps overall | **93%** |
| Getting the order right | 80% |
| Getting an entire plan perfectly right | 27% |

For 9 parts out of 10 we predict the *exact number* of machining steps.

## Why the numbers keep going up

This is the important part, and it's not what you'd expect.

**We haven't made the program bigger or more sophisticated. Every improvement
came from noticing something already sitting in the data that we weren't
reading.** Progress on the headline number:

| Change | Score |
|---|---|
| First working version | 55% |
| Matched each drilling step to the specific hole it drills | 67% |
| Found pockets by the rounded corners the cutter leaves behind | 75% |
| Also found slots by looking for their floors | 76% |
| **Counted the rounded corners to tell what shape the feature is** | **85%** |
| Predicted whether to drill first or mill first | **93%** |

The last one is the best illustration. The software that generated this data
follows a rulebook. Rather than teaching a program to guess the rulebook, we
worked out what the rules actually are.

A cutting tool is round, so it physically *cannot* cut a sharp inside corner —
it always leaves a rounded one. That means the rounded corners are a
fingerprint of what was cut:

- **1 rounded corner** → a notch cut out of the block's corner
- **2** → a slot that's open at one end
- **3** → an open pocket
- **4** → a fully enclosed pocket

Each of those needs a completely different machining operation. We were already
measuring those corners — we just weren't counting them. Adding that one number
took pocket accuracy from 46% to 85% in a single change.

## Some things we got wrong along the way

Worth recording, because they're the interesting bits:

- We assumed the tools would be the hard part. They're not — the tool is
  essentially *looked up*, not guessed, once you know the feature's size.
- We assumed "open pockets" were our blind spot. They weren't: only 29 exist
  in the whole dataset. The real gap was ordinary rectangular **slots**, which
  are invisible to the corner trick because they have no corners at all.
- We spent a chunk of time trying to improve tool selection and **it didn't
  work** — see below. That's a genuine limit, not a bug.
- We twice declared something "not worth pursuing" and were wrong once. Block
  ordering was written off as random; testing it properly turned it into one of
  our biggest gains. The other call — skipping the 3D-mesh tracks — we checked
  with hard numbers before committing.

## What's still missing

**Tool selection (70%) has hit a real ceiling.** We tried to improve it and the
attempt made things slightly worse. The reason is genuinely interesting: to pick
the right drill, the original software looks at *how wide the hole already is
partway through the job* — a state that simply doesn't exist in the finished
3D model we're given. It's not that our method is weak; the information isn't
in the input.

**Ordering is our nearest miss.** The scoring uses bands, and we finished at
0.204 where 0.200 would have earned 2 more points. Frustrating, but honest.

Worth flagging: earlier we concluded ordering was unpredictable — whether the
machine drills first or mills first looked like a coin flip (58/42). **That was
wrong, and we only found out by testing it properly.** It turns out to be
predictable at 89% from the part's geometry, and the single most useful clue is
the depth of the *shallowest* hole. Fixing this cut our ordering error by 40%
and improved the tool track at the same time.

**We deliberately skipped two tracks.** Medium (35 points) and the G-code half
of Hard (25 points) are both scored on 3D volume overlap with unforgiving
cut-offs — below the threshold they score *zero*, so a partial attempt is worth
nothing. Both need one missing capability: working out the exact outline of each
pocket. That single piece of geometry would unlock 60 points, but it is several
days of work, not one.

Notably, 82% of all the metal removed is pockets and slots, and only 9% is
drilling — so "do the easy half of the G-code first" would have scored zero.
That measurement is why we stopped rather than starting it.

## Is there a shortcut to the tracks we skipped?

We checked, because 35 points is a lot to leave on the table.

The 3D-shape track is scored by how closely our predicted shape-after-each-step
matches the real one. It turns out machining only removes **4.5% of the metal
block** — so every intermediate shape is nearly identical to every other one.
Submitting the *same* shape for every step would score 15-20 of the 35 points
without modelling anything at all.

**We are not doing that.** It uses none of the actual sequence, it passes off
the finished part as a work-in-progress state, and the rubric explicitly
penalises this kind of thing. It is recorded in our notes as a quirk of the
scoring, not as a plan.

Doing it honestly needs one capability we do not have: the exact outline of
each pocket. That is a known, solved problem in the literature (a 1988 method
called the Attributed Adjacency Graph), but building it plus the 3D geometry
work on top is several days, not one.

## Bottom line

We have a complete, validated, submittable system with a safety margin before
the deadline. Every file passes the organizers' own format checker.

The remaining gaps are measured and understood — we know which are worth
chasing, which are capped by information that simply isn't in the input, and
which are dead ends. That understanding is itself part of the submission.
