# TASK_861 — Search-teacher imprint (D26 ruled: park the search-leaf lane, test the better-teacher × proven-recipe combination)

**Status**: READY 2026-07-27 (D26 RULED — operator: "prepare dark factory tasks").
**Size: M. Risk: medium-low** (two flag-gated extensions to shipped modules; canon defaults
bit-identical; zero training in the factory half — real legs are planner-scheduled).

## Why (LEDGER 860/codistill-corridor + 832/E5-inversion3 + 832/E5-oos-real)

The program now holds two proven-but-unmarried facts:

1. **The best teacher ever measured is the search**: the true-env-leaf operator at prior-weight
   0.0 scores **+8.024 net@21** on the canonical panel (LEDGER 832/E5-inversion3) — but its leaf
   is an R-step LOOKAHEAD, so it can never trade live, and two distillation rounds trying to give
   it a deployable learned leaf failed structurally (857 best −4.23, 860 best −4.02 vs floor
   +0.424; lane PARKED per D26).
2. **The best deployable recipe ever measured is the BC imprint**: hard per-step weight-vector
   targets via the 832 weight-native recipe produced +0.839 in-sample, **+0.898 OOS (Δ+2.33)**,
   walk-forward consistent both quarters — but its teacher was only the ~floor-grade trend book
   (+0.542).

The one untested strong combination: **use the +8.02 search as an OFFLINE teacher for the 832
imprint recipe.** Run the search operator ONCE over the panel, record the action vector it
CHOOSES at each decision, and BC-imprint the policy on those hard chosen-vector targets — the
exact recipe that produced +0.839. This is explicitly NOT ExIt2's visit-distribution CE
(LEDGER 832/E5-exit2: distribution CE + per-asset argmax rounds to 100% HOLD); the targets here
are the same semantic object 832 already imprints successfully — one weight vector per decision.

## Teacher operator of record (PIN EXACTLY — do not retune)

The +8.024 configuration, verbatim from `runs/832_e5/results_e5inv854_db_pw0_832_e5__ckpt_imprint_full.json`:
`--operator search --vector-proposal-scheme dithered-book --vector-k 16 --sims 32
--vector-hold-bias 3.0 --vector-dither-scale 1.0 --proposal-temp 1.0 --proposal-explore-slots 2
--prior-weight 0.0 --leaf-eval true-env --leaf-rollout-r 3` (no v̂ tail), gumbel_scale 3.0,
deterministic, weight-rebalance env flags as in that artifact, checkpoint
`runs/832_e5/ckpt_imprint_full`. Any deviation is a different experiment and must fail loud.

## Scope (two files, both flag-gated OFF = bit-identical)

1. **Collection hook** — `experiments/835_stage4_eval/eval_vector.py` gains
   `--collect-chosen-vectors <out>`: during a search-operator walk, record per decision step
   (index/timestamp) the CHOSEN action vector — the same semantic object the env executes and the
   832 weight-native BC loss consumes (record the realized book too if cheap, diagnostics only) —
   plus a provenance header: checkpoint path, full operator block (as in the results JSON), db
   path, window, panel-frame provenance, decision count. **Pure recording**: with the flag ON the
   walk numbers must be IDENTICAL to the flag-OFF run (no RNG consumption, no behavior change);
   flag absent = byte-identical module behavior.
2. **Imprint target source** — `experiments/832_linear_imprint/imprint.py` gains
   `--teacher-vectors <file>`: replaces the replay-teacher target stream with the collected
   chosen vectors; EVERYTHING else is the exact shipped recipe (weight-native mapping, same BC
   loss/optimizer/iters/sanity-replay). **Fail-loud provenance gate**: window / panel frame /
   env-flag block in the collection file must match the imprint run's config — refuse on any
   mismatch. Default OFF = shipped 832 bit-identical.
3. **Fixture smoke, end-to-end**: collect on the fixture (tiny window) → imprint (small iters) →
   sanity replay runs. Include a falsification hook (shuffled/permuted teacher vectors must FAIL
   the smoke gate). **860 lesson applies**: the gate must be a high-SNR quantity (train-set
   agreement/fit vs the never-shuffled targets), NOT a small-n held-out metric that inverts
   per-seed at fixture scale.

## Acceptance (gates)

- (i) Both flags OFF = bit-identical (canonical pins / selftests for both modules green).
- (ii) Fixture end-to-end: collection → imprint → sanity replay completes; the collection walk's
  results are IDENTICAL with and without the flag (recording perturbs nothing).
- (iii) Provenance fail-loud demonstrated (mismatched window and mismatched frame both refused).
- (iv) Falsification: imprint on shuffled vectors fails its gate; legit run passes; seed-swept
  (the 860 fail-closed pattern).

## Real bars (planner post-ship legs, PRE-REGISTERED here)

1. **Teacher pass** on the real panel, window pinned `--end 2025-10-04` (post-858-backfill this
   yields **1673 decisions, not the canonical 1670** — flagged in LEDGER 860/codistill-corridor;
   floor references nominal). One walk, deterministic; sanity check: walk net@21 reproduces
   ≈ +8.02 (it is the same command + a recorder).
2. **Imprint** per the exact `jobs_imprint_full.manifest` recipe (fresh init, iters 2000,
   weight-native) with `--teacher-vectors` from leg 1.
3. **In-sample bar**: sanity replay (policy-argmax) net@21 **> +0.839** (the 832 imprint
   reference), above floor BOTH tiers, zero bankruptcies. Expectation management: the argmax
   policy will NOT reproduce the search's +8.02 (timing lives in the lookahead); the question is
   whether a better teacher lifts the proven recipe above its current mark.
4. **OOS** (only if leg 3 clears): the 858-extended window 2025-10-05 → 2026-03-31, policy-argmax
   + on-window teacher floor, then the walk-forward quarters per
   `runs/858_oos/WALKFORWARD_PREREG.md`. Per D17: benchmark reads only — NO gate status.

## Out of scope

ExIt/self-play of any kind; further search-lane distillation (lane PARKED per D26); retuning the
teacher operator; changes to the OOS/floor mechanics; any training spend in the factory half.
