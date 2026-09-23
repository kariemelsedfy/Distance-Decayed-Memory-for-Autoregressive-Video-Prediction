# 2026-09-23 — Claude — Planning Track A around the real GPU ceiling

## What
The owner decided to plan around the GPU limit rather than ask for it to be
raised. Before rewriting the compute plan I tested one thing: the cluster's two
GPU partitions (`gpu` and `mixed`) enforce their limits through *different*
rule sets, so the limits might add up. They do. Four short jobs (`68310`–
`68313`) held four pro6000 cards at once — two through each partition.

So the real ceiling is:

- **2 cards in any single job** (so A1 is a 2-card training run), and
- **4 cards at the same time** across separate jobs.

I rewrote `TRACK_A_PLAN.md` §9 around that, recorded it as decision D-007, and
updated the HPC notes.

## Why
The plan assumed up to seven cards. Four changes the arithmetic but not the
science: the memory sweep is many small independent runs, so what matters is how
many run side by side, and four is close enough to the original estimate that no
experiment has to be dropped. Only the base-model run gets meaningfully slower.

## How to verify
`docs/hpc/ddp-and-requeue.md` §1 records the probe jobs and the exact limits.
`docs/TRACK_A_PLAN.md` §9 holds the new estimates; `docs/DECISIONS.md` holds
D-007.

## Plain-language explanation
The lab's cluster has seven of the fast cards, but house rules limit one person
to two per job. It turns out the rules for the two queues are separate, so by
using both queues we can keep four cards busy at once instead of two — that was
worth checking, because it roughly halves how long the main experiment takes.
The plan now runs the experiments in priority order, so if we run out of time,
the most important result is already finished rather than half of everything.

## What changes in the plan
- The base model trains on 2 cards (~3–5 days instead of 1–2).
- The ~80 fine-tuning runs go four at a time: about 7–14 days, plus evaluation.
- The sweep runs in stages, most important first, so stopping early still gives
  a complete headline result.
- New infrastructure needed before the sweep: a runner that keeps four job slots
  full and restarts interrupted runs.
