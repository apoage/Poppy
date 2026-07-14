# Handoff 2026-07-14 (#11) — Factory run 9: TASK_844 (short-bias postmortem probe)

Previous: `lore/handoff_2026-07-12_factory-run8-weight-close-fee.md` (#10)
Branch: `master` · Working dir: `/home/lukas/Trader` · Role: ORCHESTRATOR
Nothing pushed.

## What shipped

| Tag | Commit | Task | Panel |
|-----|--------|------|-------|
| `task844-short-bias-probe` | `b53ecfe` | **TASK_844** short-bias postmortem probe — read-only diagnostic instrument (`experiments/844_short_bias/`, zero canon diffs, D13-independent) | MEDIUM. regression clean; **2 MAJOR FIXED** + 1 MINOR + 1 NIT→planner |

## What it is
Diagnoses where the systematic ~10-short standing book of the p1b/p1c policies is carried (value locus / policy prior / realized-reward asymmetry). Three probes: mirror-book value Δ=V(open-long)−V(open-short); flat-book policy sell/buy marginals; rollout long-vs-short leg PnL decomposition. Reuses 834 (fail-loud restore + zero-logit walk) + 835 (row timestamps) read-only via importlib; weight-env flags via `--config-json`. Ships the instrument + a fixture-smoke plumbing proof on an untrained model — **real numbers are the SCHEDULED planner run** on p1c s0/s1(+s2) + p1b s1 control.

## The panel + fixes (the substance of this run)
Both MAJORs were instrument-integrity defects; the mechanics (sign-mirror exactness, determinism, fail-loud restore, per-leg PnL formula) were already sound and left untouched.
- **MAJOR 1 — probe-1 mirror-Δ uncalibrated.** `VectorActionEmbedding.vec_act_proj` (muzero_model.py:306-314) has independently-initialized BUY vs SELL columns → a FRESH untrained model already trips per-checkpoint |t|~4 (correctness judge: seeds 1→+4.10, 7→−3.83). The within-walk-state t-test conflates a *learned* short-preference with the architecture's arbitrary zero-point on a single checkpoint — and the probe reads only 2-4 trained checkpoints. The flow-verifier's 20-init pool (aggregate mean +0.02/t 0.87, per-init |t| to 6.5) independently corroborated the mechanism. FIXED: a matched fresh-init null + `z=(Δ_ckpt−null_mean)/null_std` gating (null bootstrapped over inits); raw Δ kept as an UNCALIBRATED reference. Orchestrator re-verified robustness: seeds 0/7/8 now give calibrated z +0.10/−0.85/−1.12, all pass (seed 8 was a hard smoke FAIL pre-fix).
- **MAJOR 2 — probe-3 mislabel.** It counted ALL closes (limit/TP-SL/margin) but claimed to be the EXACT env `reward_attributable` term (env captures `prev_active_attr` post-limit/margin/TP-SL, pre-action, at :1191). FIXED: relabeled as a SUPERSET + added an `action_attributed` subset (market SELL-closes-long / BUY-closes-short) that now gates `credit_misassign`, so it can't fire on PnL training never credited (smoke: all-close L=23 vs action-attributed L=13).
- **MINOR** — the smoke's fixed `TOL=0.5` false-alarmed on ~10% of fresh seeds → retargeted to the calibrated z.
- **NIT → planner** — `open_count` likewise counts limit-fill opens (parallel to MAJOR2 on the open side); harmless today (`credit_misassign` only needs `n_short_open>0`).

## Notable operational detail
The first fix-pass agent spawned but returned with 0 tool uses (transient failure) — I confirmed the tree was untouched and re-dispatched; the retry completed cleanly. (Run 8 had hit the 2am-Prague session-token limit mid-panel; this run was clear.)

## State of the line (planner-owned)
- **D13 is OPEN (operator wall)**: phase-1c verdict is SPLIT — the weight env is FIXED (first complete trained-policy rollouts, zero bankruptcies) but from-scratch self-play is below floor 2/2 with the systematic short bias. The pre-registered D10 ladder-widening trigger fired; the planner recommends REDIRECTING to the E5/832 imprint→ExIt path (overriding a pre-registration → operator call). TASK_843 is GATED on D13; H6-widening + 832-compat both wait on it.
- **D12** (weight close+reopen cost model) — ratified accept (per git log ef8f913).
- Wedge #3 stack captured (XLA dispatch hang; orbax-lock hypothesis refuted) — the 841 watchdog follow-up.

## Git state
```
b53ecfe TASK_844: short-bias postmortem probe (read-only diagnostic instrument)   [tag: task844-short-bias-probe]
800f015 p1c s2 record folded: split verdict confirmed 3/3; battery closed clean under watchdog   [planner]
f3e9b3b p1c verdicts: env FIXED / policy below floor with short bias; D13 OPEN; TASK_844 postmortem seeded   [planner]
ef8f913 Wedge #3 stack captured; D12 ratified accept; TASK_843 seeded gated   [planner]
```
Working tree: clean after the QUEUE + this handoff commit. Factory: DISARMED. Nothing pushed.

## Next action
Factory idle — Ready EMPTY behind the D13 wall. The highest-value unblock is the **D13 ruling** (redirect to E5/832 imprint→ExIt, or follow the D10 ladder-widening). Once answered: if REDIRECT → a small factory task threads the 840 weight flag through 832 + imprint-sanity fixture smoke (843); if FOLLOW-D10 → H6 granularity widening. TASK_844's real postmortem run is planner-scheduled and will inform which locus (value/policy/credit) the bias lives in — worth reading before the D13 fork is locked. Resume the factory with "armed" once D13 is answered or a new D13-independent row lands.
