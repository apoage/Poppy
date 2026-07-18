# TASK_849 — Read B: leaf-eval autopsy (why does search selection reject the in-set prior?)

> **STATUS (2026-07-18): SHIPPED as `task849-leaf-eval-autopsy`.** Read-only leaf-eval autopsy
> probe `experiments/849_leaf_autopsy/leaf_probe.py` — per-state selection trace (K candidates
> with prior mass / r̂ / v̂ / Q=r̂+γv̂ / Gumbel score / visits / selected? + TRUE one-step env
> reward calibration column) + aggregate report (rank-dist of the in-set prior under each signal,
> signal-vs-true-reward correlation, selected-vs-prior churn); both `policy-prior` and
> `dithered-book` schemes. Zero canon diffs; selftest 8/8. Correctness + flow-verifier panel:
> 1 MAJOR (rank column corrupted by batched-vs-unbatched float32 noise — in-set prior ranked via a
> separate M=1 recompute, one-directional prior-worse bias, produced impossible rank K+1) + 1 MINOR
> (is_policy_argmax index vs vector equality) FIXED + re-verified clean. Acceptance (i)–(iv) PASS
> (selection cross-check 100% vs the 835 operator — traces the real path, not a fork).
> **Remainder → planner: the real run** (frozen `runs/832_e5/ckpt_imprint_full`, fee16 ExIt flags,
> both schemes) is scheduled/planner-owned — its numbers decide value-repair vs reward-repair vs
> selection-mechanics as a fresh DECISIONS row (D18 / Amendment 6).

**Status**: READY 2026-07-18 (D18 planner ruling under D17's decided tree; Amendment 6 FAIL branch).
**Size: S/M. Risk: low** (read-only instrument over frozen ckpts; zero training-path code;
reuses the 835/848 eval scaffolding).

## Why (LEDGER 832/E5-autopsy → 832/E5-inversion)
With the policy-argmax vector GUARANTEED in every candidate set (TASK_847, verified 40/40),
search-eval of the frozen imprint still scores **−7.84 net@21 vs +0.839 policy-argmax** — the
same churn as policy-blind dithered-book (−6.92). The selection layer rejects the in-set prior:
the value/reward leaf evaluation (and/or Gumbel scoring mechanics) ranks churny candidates above
it. Which signal misranks, and by how much, is unmeasured — that measurement decides whether the
fix is value-target repair, reward-model repair, or selection mechanics (gumbel_scale /
Q-normalization), and it must land BEFORE any further training spend (Amendment 6).

## Scope — one instrument, per-state selection trace
1. New probe (e.g. `experiments/849_leaf_autopsy/leaf_probe.py`), reusing 835's model/env
   loaders + `check_checkpoint_arch`: walk the policy-argmax trajectory of a given ckpt
   (deterministic, the 848 operator) and at every Nth decision state (default N=25, ~67 states)
   run the search's candidate generation + scoring WITHOUT stepping the env, dumping per state:
   - the K candidates (scheme flag pass-through; run both `policy-prior` and `dithered-book`),
     flagged: which is the policy-argmax, which are explore/dither slots;
   - per candidate: policy prior mass, reward-model r̂, value-net leaf v̂ (post-dynamics),
     Q = r̂ + γ·v̂, Gumbel score at selection, final visit count, selected? ;
   - TRUE one-step env reward per candidate (env step on a copy — the calibration column).
2. Aggregate report (JSON + printed table): (a) rank distribution of the policy-argmax
   candidate under each signal (prior / r̂ / v̂ / Q / visits) — WHERE the prior loses;
   (b) correlation of each signal vs the true one-step reward across candidates (the
   834-style calibration, now per-candidate); (c) selected-candidate churn stats (per-asset
   flip fraction vs the prior vector).
3. Run matrix (the instrument's scheduled real run, planner-owned): frozen
   `runs/832_e5/ckpt_imprint_full`, fee16 env flags = the ExIt set, both schemes.

## Acceptance (gates)
- (i) Fixture smoke: end-to-end on the fixture ckpt/panel, schema emitted, all dumped Q/v̂/r̂
  finite, the policy-argmax candidate present and flagged in 100% of dumped states.
- (ii) Fail-loud: arch mismatch (reuse `check_checkpoint_arch`), empty panel, scheme typo.
- (iii) Sanity cross-check: summed selected-candidate visit counts reproduce the 835 operator's
  choice on ≥95% of dumped states (the probe must trace the REAL selection path, not a fork).
- (iv) Zero canon diffs; selftest green; small/medium panel (correctness lens).

## Out of scope
Any fix (value retargeting, gumbel retuning, reward repair) — that is the NEXT decision, taken
on this instrument's numbers; any training; the OOS read (runs separately on the 848 operator).
