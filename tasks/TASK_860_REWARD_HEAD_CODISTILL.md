# TASK_860 — Reward-head co-distill (D24 ruled: "seed co-distill"; D19 opt (c) with exact targets)

**Status**: READY 2026-07-26 (D24 RULED — operator).
**Size: S/M. Risk: medium-low** (extends the shipped 857 recipe inside `experiments/857_vhat_distill/`;
canon training paths untouched; the only new unfrozen params are the reward projection).

## Why (LEDGER 857/vhat-corridor — FAIL 3/3, fork condition met)
The deployable v̂-leaf search ranks by `Q = r_model + γ·v̂`. The 857 distill lifted v̂-alone
(+0.26/+0.38/+0.10 held-out per-state) and deployed-Q lifted in all 3 real-panel seeds
(+0.24/+0.23/+0.15) — but the corridor read FAILED 3/3 (best −4.23 vs floor +0.424) because the
FROZEN 832 reward head is a poor r0 estimator (corr(r_model, true r0) ≈ 0.27) and caps the
deployed ranking far below steering grade. Attack the weakest term directly: train the reward
head on true-r0 counterfactual targets ALONGSIDE the value imprint. A better r_model benefits
both the search lane (Q) and any future ExIt targets.

## Scope (all inside the 857 module — `distill.py` + eval; no new experiment dir)
1. **Dataset**: the counterfactual generator already computes `G = r0 + γ·V` per (state,
   candidate); additionally emit the `r0` component as its own target column (same walks, same
   RAM-only storage — near-zero extra compute).
2. **Co-distill recipe** (`--codistill-reward`, default OFF = shipped 857 behavior bit-identical):
   unfreeze the `dynamics_dense2` reward-projection columns (optax multi_transform partition
   gains one leg); loss = value CE (as shipped) + λ·reward CE on the true-r0 targets (PopArt/
   two-hot exactly as the shipped trainer's reward loss form; `--reward-loss-weight` default 1.0).
   Policy head + encoder stay FROZEN.
3. **Calibration instrument** (the 849 form, held-out per-state Pearson): add an `r_model`
   vs true-r0 column next to the existing v̂-alone and deployed-Q legs — the direct measure of
   what this task fixes (baseline ≈ +0.27 pooled; report held-out per-state).
4. **Fixture smoke**: end-to-end tiny run; `--shuffle-targets` falsification must still FAIL it;
   OFF-flag run byte-identical to shipped 857 outputs.

## Acceptance (gates)
- (i) OFF = bit-identical (the shipped 857 fixture deliverables reproduce byte-for-byte);
  selftest green.
- (ii) Fixture: with `--codistill-reward`, r_model calibration IMPROVES over the frozen baseline
  and deployed-Q improves over the value-only distill (direction, not a bar — fixture n is tiny).
- (iii) Medium panel (correctness + regression + fail-closed).
- **The REAL bars are the planner's post-ship leg (pre-registered here): deployed-Q held-out
  per-state ≥ +0.5 (the 857 bar, now on the co-distilled Q), then the SAME corridor read
  (floor +0.424 bar; +8.02 true-leaf ceiling; −4.23 = the value-only best-seed reference).**

## Out of scope
Policy-head training; encoder anything; changing the search operator (frozen at the sweep
winner); OOS (routes through the 858-extended canon separately after the in-sample corridor).
