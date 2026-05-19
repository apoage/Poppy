# TASK_759 — Planning-Side Pivot

**Status**: SPEC DRAFT (2026-05-19, post-Bella class closure)
**Owner**: workstation (spec) → server (execution)
**Depends on**: 758 verdict closing Bella class (DONE 2026-05-19)
**Reference**: lore/notes_2026-05-19.md §149

## Context

After 14 paired evaluations with 0 promotions, today closed the entire
Bella mechanism class (740, 757, 758 all negative; 758 catastrophic
at −16.54pp). The only positive paired signals in the campaign come
from **planning-side reformulations**:

- 746 value_oracle n-step value target: +0.72pp paired 5-seed
- 734 flow_matching policy loss: +0.20pp paired 5-seed

Both marginal but real. Encoder-side mechanisms are confirmed
anti-correlated with Alpha across 5+ attempts. The pivot is:
**stop poking encoder, start poking planning**.

Two untouched planning-side surfaces:

| Surface | Current value | Why untouched |
|---|---|---|
| `mcts_sims` | 32 (unchanged since inception) | Never swept. MuZero's whole edge supposedly comes from search. |
| `dirichlet_alpha`, `exploration_fraction` | 0.3 / default | Standard MuZero defaults, untuned for trading. |
| Value-target deeper (TASK_756) | nstep + td_discount=0.95 in production | Deprioritized when 746 paired ceiling looked low; re-opens now. |
| Policy-head architecture | factored, temp=0.1 in production | Only temperature ever touched. |

## Goal

Find at least one planning-side mechanism that paired-3-seed
**clears mean Δα ≥ +1pp with no seed < −3pp** (light bar; if it
clears, extend to 5-seed for promotion).

## Probes

### P1 — MCTS rollouts sweep (PRIMARY)

**Hypothesis**: At `mcts_sims=32` the search horizon is shallow.
Doubling or quadrupling rollouts should sharpen value estimates and
policy targets at the cost of training wall. Untouched surface →
highest EV.

**Design**:
- 3 variants: `mcts_sims ∈ {32 (baseline), 64, 128}` at b=128.
- Paired 3-seed each (seeds 7, 13, 23).
- Reuse `runs/741_mint/eval_baseline_seed_*` (these are
  `mcts_sims=32` already).
- Variant runs: `runs/759_mcts_sims/sims_{64,128}_seed_{7,13,23}/`.

**Wall**:
- `sims=64`: ~2.5× train step → ~7.5h/seed → 22.5h paired (or
  2-wide on server: ~11h wall).
- `sims=128`: ~5× train step → ~15h/seed → 45h paired (2-wide:
  ~22.5h wall).
- Recommended: launch `sims=64` first as fast probe. If positive,
  commission `sims=128` as confirmation.

**Caveats**:
- C1 — train-time MCTS dominates wall. Verify gen/mcts/train split
  in first epoch log; if mcts > 3× train, abort and pick smaller
  sweep.
- C2 — search inference cost at deployment scales linearly with
  rollouts. If `sims=128` promotes, deployment latency budget
  matters for live trading.
- C3 — `mcts_discount=0.95` matches `td_discount` per TASK_723.
  Don't change this. Only `num_simulations`.

**Promotion gate (P1)**:
- Paired mean Δα ≥ +1pp AND no seed < −3pp at 3-seed → extend
  to 5-seed.
- If extension keeps mean ≥ +3pp with no seed < −5pp → **first
  campaign promotion**. Bloom-trigger.

### P2 — Value-target small probe (SECONDARY)

**Hypothesis**: 746's n-step (+0.72pp paired) signal is real but
small. A single mixture point — combining nstep with
Monte-Carlo-style full-return — might amplify the planning-side
gradient without overhead.

**Design** (cut-down TASK_756):
- ONE mixture: `value_target = 0.7 · nstep(td_discount=0.95) +
  0.3 · MC(actual_episode_return)`.
- Paired 3-seed at b=128 (seeds 7, 13, 23).
- Variant runs: `runs/759_value_mix/vmix_seed_{7,13,23}/`.

**Wall**: ~4h/seed → 12h paired (2-wide: ~6h wall).

**Caveats**:
- C1 — MC return target needs episode-end signal. Verify
  `train_phase2.py` carries a `done` flag through batch construction
  before launching.
- C2 — 0.7/0.3 mixture is a guess; if this probe is positive but
  marginal, sweep mixture weight as P2b.

**Promotion gate (P2)**:
- Same as P1.

### P3 — RESERVED for failure case

If both P1 and P2 close negative, P3 is policy-head architectural
sweep (depth / width / regularization). Spec it as TASK_760 only
then; don't queue speculatively.

## Execution order

1. **Workstation smoke** (P1 sims=64): 10-step run on workstation
   to confirm wall scaling matches estimate. Abort if mcts >
   3× train.
2. **Server P1 launch** (sims=64 paired 3-seed). 2-wide.
3. While P1 runs: workstation smoke P2 (one seed, 10 epochs) to
   confirm MC-mixture mechanic is wired.
4. **Server P2 launch** (vmix paired 3-seed). 2-wide.
5. P1 sims=128 conditional on sims=64 result.

## What is NOT in this task

- No Bella anywhere. Class is closed.
- No encoder-side anything. Confirmed anti-correlated.
- No Titans variant beyond MAG. If MAC/MAL ever reopens, separate task.
- No new baseline runs. Pair against existing `runs/741_mint/eval_baseline_seed_*`.

## Files to touch

- `phase2_config.py` — already has `mcts_sims` field, no edit needed
  for P1. P2 needs new field `value_target_mc_mix: float = 0.0`
  default.
- `train_phase2.py` — add `--mcts-sims` and `--value-target-mc-mix`
  CLI flags (assumed already exists for sims; verify).
- `mcts_trader/jax_agent/trainer.py` — value-target computation
  needs MC mixture branch for P2.

## Review protocol

Per session protocol: subagent reviews this spec → confirms
understanding + caveats → I approve → subagent implements with
per-step smoke + commit pattern. Two-gate.

P1 may not need code changes (CLI flag exists). P2 needs ~50 LOC
in trainer + config + CLI.
