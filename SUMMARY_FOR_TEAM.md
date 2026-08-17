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

| What we measure | Score |
|---|---|
| Getting the bevelled-edge steps right | **100%** |
| Getting the pocket / slot / notch steps right | **85%** |
| Getting the right machining steps overall | **85%** |
| Picking the right cutting tools | 70% |
| Getting the drilling steps right | 64% |
| Getting an entire plan perfectly right | 18% |

For 9 parts out of 10 we now predict the *exact number* of machining steps.

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

**Drilling (64%) is the best remaining opportunity**, for anyone with time.

## Bottom line

We have a complete, working, submittable system with a solid safety margin
before the deadline, currently at **85%** on the main measure. The remaining
gaps are understood and documented — we know which are worth chasing and which
are dead ends.
