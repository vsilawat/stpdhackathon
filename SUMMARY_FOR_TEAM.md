# Where we're at — plain English

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
- We spent a chunk of today trying to improve tool selection and **it didn't
  work** — see below. That's a genuine limit, not a bug.

## What's still missing

**Tool selection (70%) has hit a real ceiling.** We tried to improve it and the
attempt made things slightly worse. The reason is genuinely interesting: to pick
the right drill, the original software looks at *how wide the hole already is
partway through the job* — a state that simply doesn't exist in the finished
3D model we're given. It's not that our method is weak; the information isn't
in the input.

**Ordering (53%) is capped too.** Whether the machine drills first or mills
first is close to a coin flip in the source data — 56/44 — with nothing in the
part file predicting which. Not worth more effort.

**We deliberately skipped two tracks.** Medium (35 points) and the G-code half
of Hard (25 points) are both scored on 3D volume overlap with unforgiving
cut-offs — below the threshold they score *zero*, so a partial attempt is worth
nothing. Both need one missing capability: working out the exact outline of each
pocket. That single piece of geometry would unlock 60 points, but it is several
days of work, not one.

Notably, 82% of all the metal removed is pockets and slots, and only 9% is
drilling — so "do the easy half of the G-code first" would have scored zero.
That measurement is why we stopped rather than starting it.

## Bottom line

We have a complete, validated, submittable system with a safety margin before
the deadline. Every file passes the organizers' own format checker.

The remaining gaps are measured and understood — we know which are worth
chasing, which are capped by information that simply isn't in the input, and
which are dead ends. That understanding is itself part of the submission.
