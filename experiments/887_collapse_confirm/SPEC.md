# 887/collapse-confirm — does the epsilon target-mix effect survive 3 seeds and 300 epochs?

**PRE-REGISTRATION, written and hashed BEFORE launch.** Staged in scratchpad because **the factory
is ARMED** — no planner writes to `tasks/`, `DECISIONS.md`, `QUEUE.md` or `LEDGER.md`, and no
commits, until disarm. Lands as `experiments/887_collapse_confirm/SPEC.md` then.

## What this is for

`885/hold-collapse-screen` found, at **1 seed and 40 epochs**: search DEPTH is irrelevant to the
hold collapse (16× sims range moved `pct_hold` 0.13pp, non-monotone), while the TASK_812 epsilon
target-mix held it at a plateau (`pct_hold` 90.97 vs 99.02 baseline) and kept the RAW pre-mix
`mcts_visit_entropy` at its epoch-1 value (0.1554 vs 0.0246, 6.3×).

That is a screen, and it was explicitly labelled one. Two things could still be true:

1. **The plateau is a one-seed accident.** n=1 on every arm.
2. **40 epochs is too short to distinguish "prevented" from "slowed".** The baseline reached 99% by
   epoch 34; D was flat 90.97–92.17 over epochs 5–40 — but 40 epochs is **2%** of the 2000-epoch
   run the rung-1 battery would have been.

This row settles both, or withdraws the screen. **It is also the gate on re-specifying rung 1:**
`experiments/884_rung1/manifest_884_battery.txt` must NOT be launched until this reports, because
as written it reproduces arm A.

## Arms — 300 epochs each, otherwise byte-identical to the 884 smoke recipe

| arm | seeds | change |
|---|---|---|
| **A** (baseline control) | 0, 1, 2 | no eps. **Required, not optional** — A was only ever measured to epoch 40, so without it a null result cannot be attributed. |
| **D** (constant mix) | 0, 1, 2 | `--epsilon-greedy 0.25` |
| **E** (decay probe) | 0 | `--epsilon-greedy 0.25 --epsilon-schedule --epsilon-init 0.25 --epsilon-final 0.05 --epsilon-decay-epochs 150` |

**Why E is only 1 seed:** it is a probe, not an arm under test. A CONSTANT mix caps final policy
sharpness by construction — 25% uniform mass is permanently in the targets — so if D works, the
deployable form is almost certainly a decay, and E asks whether decaying re-opens the collapse. It
earns its own seeded arm only if D confirms.

7 jobs, ~14 s/epoch measured ⇒ ≈8 GPU-hours over 3 GPUs ≈ **2.7 h wall-clock**.

## Pre-registered readouts — bands fixed NOW, read at epoch 300

- **(C1) CONFIRMATION** requires BOTH halves: **D** `pct_hold` < 95% on **≥2/3 seeds**, **AND** **A**
  `pct_hold` ≥ 95% on **≥2/3 seeds**. If A does not collapse at 300 either, the 885 screen was a
  40-epoch artifact and it is WITHDRAWN, regardless of what D does.
- **(C2) STABILITY, not merely delay**: D's `pct_hold` over the final 50 epochs must show no upward
  trend exceeding **+0.02 pp/epoch** — the rate at which it would still reach 95% inside a
  2000-epoch run. This is the bar the 40-epoch screen could not test.
- **(C3) DOES NON-COLLAPSE BUY ANYTHING — reported, never a gate** (D17). Median `agent_return` and
  `alpha` over the final 50 epochs, D vs A. The screen showed only that D does not *degenerate*;
  D's min return was −25.4% vs A's −22.6%, so exploration has a short-term cost. If D is
  non-degenerate AND no better on return, that is a real and publishable negative: it would mean
  the collapse is a symptom, not the disease.
- **(NULL)** D and A within **1.0pp** at epoch 300 ⇒ the 40-epoch effect does not persist; screen
  withdrawn, and the next question moves to the reward/value structure.
- **E is descriptive only** — reported against D and A, no band, single seed.

**Honest limit:** 300 epochs is 15% of the planned run, not 100%. A pass here licenses a re-specced
rung-1 battery; it does not license a claim about what any arm converges to.

## Framework epoch

E-a..E-d. `--vector-weight-rebalance` PRESENT on every line (asserted — its omission is what voided
879 rung 3). E-e (877) and E-f (878) still queued; E-f changes execution semantics, so any score
read from this family is provisional. This row reads BEHAVIOUR (`pct_hold`, the two entropies) and
`agent_return`, not net@21, so E-f bears on C3 only.

## Checkpoint economics

`--snapshot-every 100` ⇒ 3 snapshots/seed × 7 jobs ≈ 1.9 GB; srv has 159 GB free. `snapshot_ep<N>`
is the one naming scheme the shipped GC reader resolves (D35), so these are genuinely scored. No
deletion tooling is run against them — the `--apply` freeze STANDS.
