# 885/hold-collapse-screen — is the training-lane hold collapse SEARCH-driven?

**PRE-REGISTRATION, written and hashed BEFORE launch.** Staged in scratchpad (factory ARMED);
lands as `experiments/885_collapse_screen/SPEC.md` at disarm.

**This is a DIAGNOSTIC SCREEN, not a verdict.** One seed per arm. The repo invariant is ≥3 seeds
before any promotion claim, and nothing here promotes anything — the screen's only job is to say
which hypothesis is worth a seeded battery. Any arm that wins gets re-run at 3 seeds before it is
believed.

## What prompted it

`884/rung1` step-0 smoke (40 epochs, seed 0, weight-rebalance env ON, cost0, hold-bias 0.0):

| epoch | 1 | 3 | 10 | 20 | 34 | 40 |
|---|---|---|---|---|---|---|
| `pct_hold` | 82.3% | 95.6% | 98.3% | 98.8% | 99.0% | 99.0% |
| `policy_entropy` | 0.6909 | 0.4641 | 0.1720 | 0.0948 | 0.0786 | 0.0695 |
| `mcts_visit_entropy` | 0.1545 | 0.0870 | — | — | — | 0.0246 |

Crosses 95% hold at **epoch 3**, plateaus ~99% from epoch 34. **S2 FAILED ⇒ the 3-seed rung-1
battery was NOT launched** (the gate did its job: 9 GPU-minutes instead of 24 GPU-hours).

It also separates two things `828s3/env-wall` had conflated. **No bankruptcy anywhere** (min
`agent_return` −22.6%, not −100%) — so E-a's weight-rebalance env genuinely fixed the step-0 cash
cascade — **and the hold collapse happens anyway.** The cash wall was a co-occurring defect, not
the cause of the degeneracy. Costs are zero here, so it is not fee-driven either, and
`--vector-hold-bias 0.0` was specifically the arm that still had a live policy pre-fix.

## Hypothesis

**H: the collapse is SEARCH-driven — the MCTS visit distribution sharpens onto HOLD and the policy
distills it.** The codebase already names this loop: `train_jax_fast.py:1253-1257` calls it the
TASK_812 "death-spiral triad" and computes `mcts_visit_entropy` on the RAW visit distribution
*before* any mixing "so the diagnostic sees true search collapse", with `--epsilon-greedy` provided
to "sever MCTS-sharpening -> policy-sharpening".

The smoke is consistent with it: visit entropy sits BELOW policy entropy throughout and falls to
**0.0246**, against a per-asset maximum of `ln(3) = 1.0986` — i.e. the search is near-deterministic
per asset, and 99% of those deterministic choices are HOLD. The policy target IS the visit
distribution, so a collapsed search hands the policy a collapsed target.

**This is the training-lane twin of `879/rung3-distill-remeasure`'s control**, where the search
stack was measured SUBTRACTIVE at eval time: argmax −0.564 vs search+v̂-leaf −2.166 on identical
weights, Δ1.602, hold flipping 5.4% → 94.1%. Same direction, same magnitude of behaviour change,
opposite end of the pipeline. If this screen confirms H, "the search is the problem" stops being
two separate observations and becomes one finding.

## Arms — 40 epochs, seed 0, one GPU each, everything else byte-identical to the 884 smoke

| arm | change | tests |
|---|--------|-------|
| **A** | *(already run)* `--sims 32`, no eps | baseline |
| **B** | `--sims 8` | less search ⇒ shallower collapse? |
| **C** | `--sims 128` | more search ⇒ deeper/faster collapse? |
| **D** | `--sims 32 --epsilon-greedy 0.25` | the codebase's own designed intervention: mix uniform into the TARGETS, severing the loop while leaving the search untouched |

B and C are the causal test (dose-response); D is the mechanism test (break the link, does the
collapse stop). D matters more than B/C: dose-response could also be produced by anything that
scales with compute, whereas D changes ONLY the target-mixing and nothing about the search itself.

## Pre-registered readouts — bands fixed now

Read at **epoch 40**, on `pct_hold` and `mcts_visit_entropy`:

- **(P1) DOSE-RESPONSE — search implicated**: `pct_hold` ordered `B < A < C`, with **B-to-C spread
  ≥ 2.0 percentage points**, and `mcts_visit_entropy` ordered `B > A > C`. Both orderings must
  hold; one without the other is reported as mixed, not as support.
- **(P2) MECHANISM — the loop is severable**: arm D reaches `pct_hold < 95%` at epoch 40, i.e. it
  passes the S2 bar the baseline failed. This is the outcome that would matter, because it names a
  fix rather than only a cause.
- **(NULL) not search-driven**: all four arms within **1.0 percentage point** of each other at
  epoch 40. Then the collapse lives in the reward/value structure, not the search, and the next
  probe is `--reward-hybrid --reward-attributable` at cost0 — where HOLD may simply be *correct*
  for a policy that has found no edge, which would be a statement about the problem rather than a
  bug.
- **Anything else**: reported as inconclusive with the numbers, and no lane is re-opened or closed
  on it.

Honest limit, stated in advance: 40 epochs is a screen. A difference at epoch 40 is a difference in
*collapse rate*, and does not establish what any arm converges to at 2000. No arm here promotes
anything to the ladder; the winner earns a 3-seed battery, nothing more.

## Framework epoch

E-a..E-d, same as the 884 smoke. `--vector-weight-rebalance` PRESENT on every arm (asserted — its
omission is what voided 879 rung 3). E-e (877) and E-f (878) still queued; E-f changes execution
semantics, so any score read from this family is provisional. That does not affect this screen,
which reads behaviour (`pct_hold`, entropies) and not net@21.
