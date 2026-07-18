# TASK_847 — Policy-prior proposal scheme for the vector search (D17; the read-A inversion)

**Status**: READY 2026-07-18 (D17 ruled — operator).
**Size: M. Risk: medium** (search proposal layer only; canon scalar path untouched; all existing
schemes remain selectable — new scheme is a new `--vector-proposal-scheme` value, default
UNCHANGED so every existing invocation is bit-identical).

## Why (evidence chain — LEDGER 832/E5-exit, 832/E5-erosion, 832/E5-autopsy)
The dithered-book proposal scheme is POLICY-BLIND: it perturbs the CURRENT BOOK, so the policy's
preferred action vector is never among the K candidates and the prior cannot express itself
through search. Demonstrated with zero training: the SAME imprint checkpoint scores **+0.839
net@21 under policy-argmax and −6.92 under the search operator** (tier0: +1.449 vs −0.60 — the
visit distribution is ~noise around the dither even costless; fees multiply it). Training on
those visit targets annihilates the +0.839 prior by ep100 (erosion curve) — the mechanism behind
the universal HOLD/sparse attractor (p1b, p1c, warm-started ExIt, 3× convergence). Dithered-book
was E1/E2-validated ONLY under a perfect value; it does not transfer to a learned stack with a
strong prior.

## Scope
1. New proposal scheme `policy-prior` for the Sampled-MuZero vector search (the K-candidate
   generator used by both training self-play and 835 eval):
   - candidate 0 = the policy-argmax vector (per-asset argmax of the policy marginals) —
     GUARANTEED in every candidate set;
   - candidates 1..K−1−N = i.i.d. samples from the policy's per-asset marginals at temperature
     `--vector-proposal-temp` (default 1.0);
   - last N = exploration slots retained from the existing dither machinery
     (`--vector-proposal-explore-slots`, default 2 of K=16; dither params reuse the existing
     flags).
   Dedup candidates; keep the K/sims interface, Gumbel selection, and visit-target extraction
   untouched.
2. Wire through `--vector-proposal-scheme policy-prior` everywhere the scheme enum exists
   (training CLI, 835 eval, 834/836 config plumbing) — value-add flags only, no default changes.
3. Selftest: candidate-set property test (argmax vector present; temperature monotonicity;
   explore slots honored; dedup).

## Acceptance (gates)
- (i) **The read-A inversion (fixture-scale variant for the factory + the real read scheduled
  planner-side): 835 search-eval of the FROZEN imprint ckpt with `policy-prior` must PRESERVE
  the prior.** Factory fixture leg: on the fixture panel, search-eval net@21 with the new scheme
  ≥ (policy-argmax replay − 0.10) on the same rows. The real-panel binding leg (bar: ≥ +0.74,
  i.e. +0.839 − 0.10) is the planner's scheduled run, same discipline as 846/(iv).
- (ii) Canon + existing schemes BIT-IDENTICAL: default invocations (dithered-book et al.)
  reproduce their pins; `--vector-proposal-scheme policy-prior` is the only entry point.
- (iii) Selftest 7/7 + the new property tests; medium panel (correctness + regression lenses).

## Out of scope
The short ExIt re-try (planner-scheduled after this ships and the real-panel leg passes);
leaf-eval/value-target changes (read B territory, only if the fixed proposals still erode);
retuning K/sims.
