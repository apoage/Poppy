# Dark factory run 16 handoff — 2026-07-23

**One task shipped, clean.** TASK_854 (D21). Queue exhausted for factory work afterward.

## Shipped

| Tag | Task | Verification | Commit |
|-----|------|--------------|--------|
| `task854-prior-weight-selection` | TASK_854 — `--prior-weight` hook on Gumbel candidate scoring + `--value-scale` fail-loud | 4-member panel (correctness + regression + flag-wiring + flow-verifier), NO BLOCKER/MAJOR, no fix pass; all fixture numbers reproduced ×4 | `f5807f6` |

### What TASK_854 is
D21's last learned delta vs the E1 selection (LEDGER 832/E5-inversion2). `--prior-weight <float>`
scales the **learned policy prior term in candidate generation** (dithered-book/iid/policy-prior):
- **default 1.0 = byte-identical to HEAD** (static Python guard `if self.prior_weight != 1.0` — no
  ops added, no RNG consumed; tiers-SHA `ca72a38c7462ce24` reproduces + 25 canon pin tests pass);
- **0.0 = pure-Q = the E1 selection** (candidate draw invariant to `policy_logits`).

ONE implementation feeds BOTH consumers — 835 eval (`eval_vector.py`) and self-play target
generation (`train_jax_fast.py` via `cli.py`/`config.py`) — so eval and target-gen score
candidates identically. Scope 2: `--value-scale` made fail-loud in its provably-inert combo
(`leaf_eval=true-env` + no v̂ tail + `value_scale!=1.0`); stays wired where it bites.

### Design deviation (LOAD-BEARING — panel-cleared, planner should note)
The literal spec formula `score = prior_weight·logits + g + σ(Q)` names mctx's internal **root
`prior_logits`**. But in this vector search that prior is the **uniform empirical Sampled-MuZero
prior** (`_empirical_prior` → zeros), so scaling it is a genuine no-op — the exact silent-inert-flag
hazard this task exists to kill. The learned policy prior enters the true-env-leaf loop **only
through candidate generation** (proposal `policy_logits`), which IS Gumbel-max candidate scoring.
The implementer gated the prior there instead. The correctness judge independently traced
`_empirical_prior` → zeros and confirmed `prior_weight=0.0` reduces all proposal branches to
policy-invariant candidate draws. All four panel members reproduced the fixture numbers exactly.

### Fixture acceptance (i) — directional PASS
`ckpt_smoke`, true-env leaf, `--no-leaf-vhat-tail`, hb3.0, seed 0, 2025-09-14..2025-12-12, 240min, 8 assets:
- policy-argmax replay: net@21 **−6.778** (reproduces the 851 baseline)
- search `--prior-weight 0.0`: net@21 **+17.162** (clears the directional bar; no bankruptcy, no HOLD-collapse, full rollouts)
- search `--prior-weight 1.0`: net@21 **+8.316** — **non-inert**: removing the learned prior moves the number toward the E1 anchor, exactly as D21's isolation predicts (the imprint prior votes for churn, pct_hold 94.5% → 74.8%).

## Planner-owned remainder (pre-registered — the real deliverable)
1. **Real binding leg** — frozen `runs/832_e5/ckpt_imprint_full`, `--prior-weight 0.0`, true-env
   leaf, no tail, hb3.0, bar **net@21 ≥ +0.74 tier21** (honest expectation: E1-class, well above).
   **A MISS FALSIFIES the isolation** → no retune-and-retry → new DECISIONS row on the residual
   candidates (visit-vs-Q final selection; dithered-book trajectory path-dependence). This is the
   scheduled server/workstation run — the factory does not launch it.
2. **PASS ⇒** small eval sweep (`--prior-weight {0, 0.25, 0.5}` × R {3, 6}, frozen imprint) picks
   the ExIt operator → the Amendment-6 ExIt re-try RE-ARMS with it (3 seeds × 300 ep,
   prior-survival read, D11 watchdog).

## Two out-of-scope MINORs carried to planner (both non-blocking, flagged by the panel)
1. **`value_shift` sibling inert-flag hazard** (flag-wiring judge): `value_shift` shares the exact
   same silent-inert property as `value_scale` in the true-env/no-tail combo (also enters only via
   `root_value` → affine → argmax-invariant). TASK_854 scope named only `--value-scale`. If the
   planner wants the hazard fully closed, it's a one-line extension of the existing fail-loud
   condition to `value_shift != 0.0` — a trivial follow-up task.
2. **Stale committed smoke baseline** (regression + flow-verifier): `results_fixture_smoke.json`
   predates the `prior_weight` operator-metadata key (and the older 851 `leaf_eval`/tail keys). This
   is ADDITIVE metadata only — the load-bearing `tiers` block reproduces byte-for-byte, so acceptance
   (ii) parity holds. A one-line refresh of the committed artifact would make its operator block
   current; not required, out of scope for the ship.

## Queue state after this run
- **Ready: EMPTY.** TASK_854 was the sole Ready row; now bannered SHIPPED.
- **D22 (OPEN)** — the only open decision — is the minute-backfill-vs-scoped-benchmark ops/data
  call for the binding OOS benchmark. It is explicitly an **ops decision that does NOT block the
  854/ExIt critical path**, and it is not a factory-lane code task (no code to write; it's a
  human data-pipeline choice). Left untouched.
- **Gated rows** need gates that are not open (Stage-4 eval = scheduled server run; 843 already
  moved to Ready and shipped in a prior run).
- Stopped per the stop condition: everything remaining needs a human decision (D22) or is
  planner/scheduled work (the 854 binding leg + the ExIt re-try).

## Notes
- Clean tree, nothing pushed. Tag `task854-prior-weight-selection` created locally; the operator
  pushes after review.
- No concurrent planner edits were present in the working tree at commit time; staged the 8 lane
  files explicitly regardless.
