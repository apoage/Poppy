# Pathway — reasoning ML trader over OHLC

**Role**: the line design. The goal is not one build; it is a LADDER OF EXPERIMENTS, each
falsifiable, each cheap relative to the rung above it, each gating the GPU spend above it.
Planner-owned (this doc, the E-ladder, the ledger); factory executes the rows it emits.
Written 2026-07-08 on operator directive: "design an actual pathway … by definition a lot of
experiments."

## Definition of working (pre-registered, fixed)

A reasoning trader WORKS when, on unseen years with honest costs (fee16/slip5), ≥3 seeds, it
- beats the 819 linear trend book (net@21 Sharpe **+0.97** — the bar),
- holds positions (no HOLD collapse; 821 died at 0/1670),
- is BEAR-positive (BEAR22 in-window),
then survives the gate ladder (all-regime → unseen → paper → live). Nothing promotes past a
gate on one seed or one window. `trader/` stays alive solely to keep this bar honest.

## The one lesson of experiments 000–780

Six months of single-mechanism tweaks (encoder, classifier sharpening, entropy, reward shaping,
Bella distill…) were null-or-marginal — **on the scalar-action stack**. The single structural
defect found (scalar action space, `jax_trading_env.py:1154`) explained the planner's rational
HOLD collapse. Rule derived from that record:

> **Every experiment must attack a named STRUCTURAL hypothesis (H1–H5). Knob-turns are not
> experiments; they are noise with a GPU bill.**

## Hypothesis map — where reasoning can beat a fixed rule

| # | Hypothesis | Evidence for | Falsified if |
|---|-----------|--------------|--------------|
| H1 | **Breadth**: one decision = per-asset VECTOR, EV = Σ(64 edges) > noise → learnable/plannable | linear book harvests exactly this aggregate (+0.97); scalar EV was sub-noise | vector planner still collapses to HOLD with a verified action-conditional model (E1 passing) |
| H2 | **Regime conditioning**: value function modulates exposure by regime → beats the fixed-parameter rule where it is weakest (CI on the rule spans −0.01) | rule has zero regime awareness; BEAR22 is in-sample | learned sizing never beats fixed sizing on regime-split evals |
| H3 | **Cross-asset structure**: joint action exploits correlation/lead-lag the per-asset rule ignores | factored head + joint value already in the net | joint-action ablation == independent-per-asset ablation |
| H4 | **Cost-aware planning**: multi-step search nets trades, cuts turnover → wins after costs | search sees fees in the dynamics; rule pays every rebalance | planner turnover/cost ≥ rule's at equal gross |
| H5 | **Teacher validity**: valid imprint = the linear book / self-play; oracle = measurement instrument ONLY | oracle-as-teacher = diagnosed disease vector (anti-correlation, bull-blindness) | linear-book imprint also anti-correlates (would implicate BC itself, not the teacher) |
| H6 | **ARMED 2026-07-08** (measured gap 0.327 > 0.2) — **Action granularity**: the {0..4} per-asset set can't express the book's continuous weights → weight-ladder actions. CONDITIONAL: the +29.7 ceiling lives in the same discrete space, so discreteness doesn't cap the agent below the bar — becomes a build task only if the trained agent plateaus inside [floor_inenv +0.42, ref +0.75] | 830/E0-REAL: in-env replay floor +0.424 vs engine ref +0.751 | trained agent clears the engine ref without weight-ladder actions |

## The rung ladder

Cost ascends; **each rung's VERDICT gates the GPU spend of the rung above** (code for a higher
rung may land earlier — the factory builds ahead; training waits).

```
R0 instrument   E0 corridor: linear book + oracle bound through the vector env      CPU-hours
R1 search       E1 true-model search probe · E2 sampling-scheme ablation            ≤1 GPU-day
R2 learnability E3 action-conditional reward probe · E4 value-ranking probe ·       ≤1 GPU-day
                E5 linear-book imprint + can search beat its teacher                    each
R3 full loop    828 Stage 3 training variants, ranked by R1/R2 verdicts             server, wks
R4 graduation   walk-forward unseen years · cost stress · Bella regime stress ·     scheduled
                gate ladder to paper
```

- **R0 — E0 (TASK_830)**: replay the 819 linear book through `step_vector` → env validity +
  in-env baseline; compute the per-asset oracle bound under the same cost model (corridor top).
  Every later result is reported as a position inside **[linear, oracle]**.
  **D6 corridor semantics (2026-07-08)**: DUAL FLOOR — rung experiments grade against the
  env-native hold-shares in-env floor (same action space the agent lives in); the **+0.97 engine
  bar is unchanged as the R4 graduation bar**. The real-panel run reports both plus the
  EXPRESSIBILITY GAP; a gap >0.2 net@21 Sharpe activates H6 (action granularity).
- **R1 — E1/E2 (TASK_831)**: run Sampled-MuZero search with the SIMULATOR as dynamics + true
  reward (no learned nets). Isolates search capacity from learning: if K-sample search can't find
  ≥ linear with a *perfect* model, Stage-3 training is pointless until the sampler is fixed.
  E2 ablates the sampling scheme (independent per-asset draws vs linear-prior mixture vs dithered
  current book; K ∈ {8,16,32}).
  **Reinterpretation (831 fixture smoke, 2026-07-08)**: E1 measures PROPOSAL CAPACITY only —
  with a perfect value even the uniform sampler reaches the floor at K≥16, so PASS is
  necessary-but-not-sufficient and its kill-power targets a catastrophically-broken sampler.
  The load-bearing R1 deliverable is the **E2 scheme×K ranking** (fixture cheapest reacher:
  dithered-book|K8). The discriminative "can a LEARNED selector find it" question lives at
  E5/R3, as designed.
- **R2 — E3/E4/E5**: E3 = the D2 variance-attribution probe (reward head must MOVE with the
  action input; the Stage-B head was 60× action-blind) — hard gate inside 828 Stage 3. E4 = does
  the value net rank states above a momentum baseline. E5 (TASK_832) = imprint from the LINEAR
  BOOK's trajectories (the valid teacher, per H5), then measure whether search IMPROVES on its
  teacher — the smallest possible "reasoning > rule" test.
- **R3**: 828 Stage 3/4 as specced — but variant choice (sampler, K, imprint-vs-scratch) comes
  from R1/R2 verdicts, not taste.
- **R4**: existing gate ladder; Bella's resident role (D5) = regime STRESS GENERATOR at this
  rung — never a teacher, never load-bearing.

## Experiment protocol (the card)

Every experiment, before launch:
1. `experiments/<id>/SPEC.md` (≤1 page): hypothesis (H#), metric, **pre-registered PASS/FAIL**,
   seeds, compute budget, kill date. No post-hoc metric shopping.
2. Runner = a SMALL module in `experiments/<id>/` importing the shared harness
   (`tools/benchmark_easy.py` scorecard / `runs/821_planner_daily/eval_pass1.py` rollout
   pattern) — token-size mandate: never thread experiment code through the monster files.
3. Factory implements + fixture-smokes; the real-data run is a SCHEDULED server/CPU job.
4. Result → one line in `experiments/LEDGER.md` (verdict + corridor position), PASS or FAIL.
   Nulls are results. Task banner updated; planner regrooms the queue.

Budgets: R0–R2 ≤1 GPU-day each; ≤2 concurrent per host; workstation = smokes under the
allocation contract. Leak-free, cost-honest, ≥3 seeds for any promotion claim — always.

## Kill / pivot logic

- E0 fails → the vector env itself is wrong; fix before anything.
- E1 fails after E2's ablations → Sampled-MuZero sampling design is the defect; redesign the
  action-proposal mechanism (that becomes the new H) before any training.
- E3 fails → reward architecture, not training budget, is the blocker (827's lesson).
- E5 shows search can't beat its own teacher with a working model → H1 weakens materially;
  the honest answer starts looking architectural, and the pathway pivots to a new structural
  hypothesis — recorded, not rationalized.
- Fixed rule stays the fallback shippable at every rung; the corridor keeps everyone honest
  about how much of the oracle gap any rung actually closed.

## Current wiring

828 St1 → **E0** → 828 St2 → **E1/E2** → 828 St3 (E3 inside, E4 alongside) → **E5** → 828 St4
→ R4. Queue rows in `tasks/QUEUE.md`; specs TASK_830/831/832 + TASK_828 stages; ledger at
`experiments/LEDGER.md`.

**Position (2026-07-08, post factory run 1 + scheduled 830)**: 828 S1+S2 SHIPPED (tags
`task828-stage1/2`); factory claims independently re-verified (selftest, 13+11 tests, defaults
OFF, ledger↔JSON). **R0 DONE**: official corridor net@21 **[+0.424, +29.698]**, engine ref
+0.751 (this window), BEAR22 ref +1.049; fixture daily==db daily confirmed; H6 armed
(conditional). R1 real sweep = CPU/scheduled (GPUs ceded to the factory); its E1 verdict opens
the Stage-3 TRAINING gate. Next factory rows: 829 (retire Foxglove), then Stage-3 CODE.
Stage-4 grading note: "beat the bar" = beat the ENGINE REF on the eval window (the +0.97 was
819's window; this window's ref is +0.751 net@21).
