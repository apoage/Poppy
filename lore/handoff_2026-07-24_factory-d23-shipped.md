# Factory handoff — 2026-07-24 — D23 Ready queue shipped (856 · 857 · 855)

Dark-factory implementer run. Cleared the entire D23 Ready queue. Neutral technical record.

## Shipped

| Tag | Task | Commit | Verification outcome |
|-----|------|--------|----------------------|
| `task856-decision-rule-probe` | 856 — decision-rule probe + net_dependent grading fix | `861ea63` | correctness+regression panel; fix pass 1 MAJOR (top-m vacuous in collapse regime) + 1 MINOR + 1 NIT, independently re-verified via mutation test |
| `task857-vhat-leaf-distill` | 857 — counterfactual v̂ imprint → deployable v̂-leaf search | `15a80f5` | correctness+regression+fail-closed panel, THREE fix rounds; regression clean throughout; released owner-resolved the seed-fragile gate directly |
| `task855-flag-hygiene-minors` | 855 — value_shift fail-loud + 835 baseline refresh | `a19c528` | correctness/flag-wiring panel; fix: honest ledger date + ST11 pin |

All default/canon paths bit-identical; selftest 11/11 → 12/12 across the run. `sampled_search.py` byte-untouched by 856/857 (857's operator lives entirely in the eval harness).

## 856 — decision-rule probe (D23 step 1)
- `--decision-rule {argmax,threshold,top-m}` (+ `--decision-tau`, `--decision-m`) on the 848 policy-replay operator, to extract the trade signal per-asset argmax rounds away in the ExIt2 collapse regime. Default `argmax` bit-identical (sole delta = additive `net_dependent` key).
- Scope 2: `operator_net_dependent()` classifier + unconditional `net_dependent` result key + loud WARNING on zero-learned-component operators (fail-loud grading hygiene).
- Panel MAJOR: `top-m` originally assigned the FULL argmax (incl. HOLD) to selected assets → `top-m == argmax == 0 trades` in the collapse regime (the probe's exact target), silently vacuating a sweep leg. Fixed to assign the best NON-HOLD action (selected assets always trade). Mutation-tested guard.
- **Planner remainder**: the real rule-grid sweep (ep300 ckpts × {argmax,threshold-τ,top-m} vs floor bar +0.424) — decides tuning-vs-architecture for ExIt2 distillation. Spec prose line ~29 (`M=64+τ=0 ⇒ argmax`) is superseded by the top-m fix (now `⇒ threshold τ=0`) — correct the line.

## 857 — v̂-leaf distill (D23 step 2, the deployable-search path)
- Deployable `--leaf-eval vhat-distilled` = mechanically the canon model leaf (value head + dynamics, no lookahead) on a distilled checkpoint; `sampled_search.py` untouched; `net_dependent: true` even at pw0. Value comes entirely from the distilled ckpt.
- `experiments/857_vhat_distill/distill.py`: counterfactual dataset `(state, candidate, true G=r0+γ·V)` reusing the 851 true-env leaf; value-head imprint (dynamics+value gradients only, policy/reward frozen) warm-started from `ckpt_imprint_full`.
- Panel (3 rounds) hardened the calibration INSTRUMENT (the planner's real gate runs the same `calibration()`): (i) headline/gate on the 849 per-state HELD-OUT Pearson (not inflated pooled); (ii) calibrate the DEPLOYED quantity `Q=r_model+γ·v̂`, not v̂-in-isolation; non-hollow smoke gates. Round-3 caught a seed-fragile absolute floor → retuned to a fail-closed-safe `deployed-Q > −0.25` non-degeneracy floor (below the legit noise band, above target-shuffle sabotage ~−0.49), verified by the `--shuffle-targets` falsification hook (FAILS + writes nothing) vs normal (PASSES).
- **DECISIONS D24 (seeded, OPEN)** — the load-bearing finding: v̂ improves on its own axis (849 instrument +0.2331→+0.3416) but the DEPLOYED-Q ranking does NOT lift at fixture scale (+0.125→+0.115, within-noise/seed-dependent {+0.115,−0.014,−0.119,+0.491} across seeds 0–3) because the FROZEN 832 reward head is a poor r0 estimator (corr≈0.27). Clearing the v̂-alone ≥+0.5 bar does not alone prove the deployable operator ranks like the true env.
- **Planner remainder**: the real-data corridor read of `--leaf-eval vhat-distilled` (warm-start `ckpt_imprint_full`) vs floor +0.424 / +8.02 true-leaf ceiling, reporting BOTH v̂-alone and deployed-Q calibration — the program's FIRST deployable-operator corridor read. D24 = the read-then-decide fork on whether the reward head gets co-distilled next (extend the counterfactual dataset with true-r0 targets, unfreeze the `dynamics_dense2` reward-projection columns — D19 opt (c) territory). Recommendation: run the corridor AS-IS first (cheap; the fixture deployed-Q gap is 12 states); rebuild the reward head only if the real panel confirms the miss.
- Carried MINORs (disclosed): shuffle fails gate (i) before (ii) in ordering (fail-closed-safe; (ii) independently proven to bite by round-3 isolation); the −0.25 floor cannot distinguish a good distill from a passive no-op at fixture scale (fundamental frozen-reward-head limit; the real verdict is the corridor read).

## 855 — flag-hygiene MINORs (filler)
- `value_shift` fail-loud guard (sibling to 854's `value_scale`): raises on the inert combo (`leaf_eval='true-env'` + no v̂ tail + `value_shift != 0.0`; default 0.0). ST11 extended to pin both sibling guards.
- 835 `results_fixture_smoke.json` refreshed to absorb the additive metadata keys from 851/854/856. Additive only — scored values byte-identical, tiers-SHA `ca72a38c7462ce24` unchanged, honest `ledger_row` date 2026-07-24 (panel caught the initial 2026-07-18 revert as provably false).

## Tree / ops notes
- One untracked file left in the tree: `lore/handoff_2026-07-23_e5-diagnosis-closed-planner.md` — authored by a concurrent PLANNER session (appeared mid-run; not the factory's lane). Left untouched; the planner owns it.
- No pushes. All work committed + tagged on `master`.
