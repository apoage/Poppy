# Factory run 14 handoff — 2026-07-21 (851 shipped, 852 reconciled-and-stopped)

Both Ready rows resolved. Zero OPEN decisions on entry; one seeded on exit (**D20**). Clean
tree, nothing pushed. TASK_851 is a tagged ship; TASK_852 is a finding-style commit (STOP, not
a ship — no tag, no CHANGELOG entry, per this repo's convention).

## Shipped / resolved

| Commit | Tag | Task | Outcome |
|--------|-----|------|---------|
| `41561d4` | `task851-true-env-leaf` | TASK_851 — true-env leaf rollouts (D19 opt (a)) | correctness + regression judges + flow-verifier, **2 rounds**: 2 MAJOR fixed + independently re-verified; both fixture acceptance legs PASS; zero regressions; selftest 10/10 |
| `c105038` | — (finding, not tagged) | TASK_852 — daily/minute reconciliation | correctness + fail-closed + regression judges + flow-verifier (4-member, task's own mandated fail-closed lens): **1 BLOCKER + 2 MINOR** fixed + independently re-verified; task's own FLOOR-leg acceptance still FAILS on a residual → **STOP, DECISIONS D20 seeded OPEN** |

## TASK_851 — true-env leaf rollouts

`--leaf-eval {model,true-env}` on `SampledMuZeroVectorSearch` (default `model` = canon
bit-identical). `true-env` steps a copy of the true env forward `--leaf-rollout-r` steps
(default 3) per candidate leaf instead of scoring through the action-blind learned value head
(LEDGER **849/leaf**: v̂ correlates +0.03 with true one-step reward — the measured root cause of
the HOLD/churn attractor behind preflight/p1b/p1c/the ep100 imprint erosion). Reuses the
`experiments/831_search_probe` operator (E1: +5.29 net@21 with a true-env leaf vs the shipped
model-leaf's −6.92) — wiring it into the shipped path, not reinventing it. Threaded into both
training self-play and the 835 eval path.

- **Fixture inversion (i) PASS**: policy-argmax replay net@21 −6.778 vs search+true-env
  **+6.801** — directionally reproduces the real E1/832-autopsy gap.
- **E1-consistency anchor (ii) PASS**: fresh-init net scores below its floor under the model
  leaf (−10.287 vs floor −3.437) but above it under the true-env leaf (+2.198).
- **2 MAJOR found + fixed + independently re-verified** (2 review rounds): (1)
  `_true_env_leaf_values()` didn't dispatch to `batch_step_vector_unique` for unique-per-env
  (3D) prices — an unguarded crash on `--leaf-eval true-env --unique-data-per-env`; fixed with
  a real dispatch mirroring `self_play.py`'s existing pattern + a new permanent regression test
  (ST10, confirmed non-vacuous). (2) the fixture-inversion headline numbers didn't independently
  reproduce at first review — **root-caused to a concurrent TASK_852 edit transiently mutating
  the same shared `eval_vector.py` file during that review window** (see incident note below,
  not nondeterminism in this task); resolved by pinning the exact invocation/seed/checkpoint
  command to `experiments/LEDGER.md` + a committed artifact (830/831/847/849 convention),
  independently re-reproduced from a cold checkpoint rebuild to 3 decimal places.
- **Cost guard**: true-env R=3 measured *faster* than the model leaf on CPU (4.6–5.1ms vs
  11.7–12.8ms median) — the K-candidate rollout vmaps once instead of up to 32 serial NN
  forward passes through the mctx tree.
- **Default byte-identical**: selftest 10/10; 835 `--smoke` diff shows only new
  provenance/config-echo fields, zero numeric drift.

**Planner-owned, post-ship (NOT launched)**: real inversion read on the frozen
`runs/832_e5/ckpt_imprint_full` (bar net@21 ≥ +0.74) → PASS re-arms the Amendment-6 ExIt
re-try (3 seeds × 300 ep, `--leaf-eval true-env`) — the first training run where search can see
*why* trading pays. Distilling the true-env leaf back into v̂ (D19 opt (c)) stays out of scope.

## TASK_852 — daily/minute reconciliation

Diagnosed the divergence from LEDGER **850/eqgate** (floor Δ=0.2565, "lookahead-shaped").

- **Coverage**: 27/64 assets have real `ohlcv_daily` rows in the 2021–2025 overlap; 37/64 have
  zero. **Lookahead hypothesis REFUTED**: shifting the daily panel ±1 day moves the delta to
  0.2497/0.2590 (from baseline 0.2565) — neither direction collapses it, so the "+1-day
  alignment shift" read in 850/eqgate was wrong. **Root cause**: the daily loader
  forward-filled the 37 zero-coverage symbols from a nonexistent anchor, producing fake
  flat-zero price columns (silent densification) — isolating to the same 27 real symbols on
  both sides collapses the gap to 0.0555 (**78% of the original gap was this bug**).
- **Fix** (daily loader + gate orchestration only, minute path byte-identical): active-mask
  filter drops zero-coverage symbols instead of densifying (`load_daily_prices`, now also
  requiring `close IS NOT NULL AND != 0`, closing a placeholder-row loophole); the FLOOR leg
  intersects the comparison universe against real coverage before building either panel, and
  now prints which symbols were excluded, not just a count.
- **The BLOCKER (fixed) — caught by verification, not the implementer's own pass**: the
  active-mask filter reactivated a previously-dead pad-by-cycling branch in
  `create_daily_panel` — real assets were silently duplicated across panel columns to refill
  width whenever the filter under-supplied, doubling their effective exposure. Only guarded
  inside `run_equivalence_gate`'s own FLOOR leg; every other caller — including TASK_852's own
  named "Scheduled real run after PASS" command — was exposed. **3 of the 4 panel members
  independently found and converged on this**, one reproducing it live on the exact scheduled
  window (28/64 dropped → would have double-exposed 28 symbols). Fixed to fail loud instead
  (clear `RuntimeError` naming the exact remedy), independently re-confirmed by the orchestrator
  via a direct reproduction of the same scenario.
- **Residual FAIL**: even after the fix, FLOOR leg |Δ|=0.0555 — 2.75× the 0.02 tolerance.
  Leading unexplained candidates: calendar-day-union daily axis (T=1704) vs minute's positional
  stride (T=1701) are different discretizations; small intra-series gaps in a couple of the 27
  covered majors (BCH/USD, XRP/USD). **STOP per the task's own acceptance (ii) fallback —
  DECISIONS D20 seeded OPEN**, not forced closed and not guessed past.

**Process incident (handled, not a code defect in either task)**: mid-review, TASK_851's
correctness judge ran `git checkout --` on `eval_vector.py` (misreading a pre-existing `M`
marker as its own), wiping TASK_852's concurrent uncommitted fix from disk. The judge restored
its own captured diff, but TASK_852's fix had to be reapplied by its own implementer (confirmed
identical numbers on the confirmation re-run). Both tasks' final commits were hand-split from
the shared working-tree file via a byte-verified patch split (`git apply --cached`, never
touching the working tree) rather than risking further git surgery — recombining both isolated
patches against HEAD reproduces the pre-split file exactly. **Lesson for future runs**: when two
factory-loop tasks land in the same file concurrently, warn every downstream verification agent
explicitly against destructive git operations (done for TASK_852's panel, not proactively for
TASK_851's first pass — that's what caused the incident).

## Where it stands

- Queue Ready is now EMPTY. One fresh wall: **D20** (OPEN) — the TASK_852 residual. Binding OOS
  re-run (`832/E5-oos` on `--panel-source daily`) stays gated on it.
- TASK_851's real binding leg + the Amendment-6 ExIt re-try are the planner's scheduled
  post-ship chain (see above) — independent of D20, can run in parallel.
- Both real runs are genuinely different tracks now: 851 unblocks the *search* fix path; 852's
  residual blocks the *OOS benchmark* path. Neither depends on the other going forward.

## Questions that unblock the highest-value track

1. **Rule D20** — audit intra-series gaps within the 27 covered daily symbols first (SQL only,
   no training spend, per the seeded recommendation); if that closes the 0.0555 residual the
   way the coverage fix closed the 0.21, `--panel-source daily` ships and the binding OOS
   re-run launches. If not, re-base the FLOOR tolerance for cross-source comparisons or keep
   the OOS binding read on `--panel-source minute`.
2. **Schedule TASK_851's real inversion read** (frozen `ckpt_imprint_full`, bar ≥ +0.74) — PASS
   re-arms the Amendment-6 ExIt re-try, the first training run where search can see why trading
   pays. Independent of (1); can run in parallel.

Neither was launched by the factory (expensive real-data runs are scheduled, not polled
in-loop — invariant). Tree clean, nothing pushed.
