# Factory run 31 handoff — TASK_886 HALTED on the 3-cycle rule, TASK_877 SHIPPED

**Date:** 2026-08-10/11 · **Harness:** Claude Code (Opus 5, 1M context) · Subagents per
`.dark-factory/config.json` model policy: implement on `sonnet`, fix passes escalated to
`opus`, final-cycle panel on `opus`.

## Outcome

| Row | Result |
|---|---|
| TASK_886 — universe-pin certification | **HALTED** on the 3-cycle rule. Code in `git stash`. **D38 OPEN.** |
| TASK_877 — XBT↔BTC alias fold | **SHIPPED** as `task877-xbt-alias-fold` (`72c0c87`) |

Nineteen subagent runs total: for 886 a 5-lens panel, a 4-lens re-verify and a 3-lens `opus`
re-verify; for 877 a 3-lens panel, a 2-lens re-verify and a narrow final verification.

---

# TASK_877 — SHIPPED

Kraken's `XBT` is `BTC`. `BTCCHF` (id 1085, frame slot 23) and `XBTCHF` (id 1175) are the same
instrument under two spellings — **48,780 shared timestamps agreeing to max relative close
difference 0.0** — and the resolver's slash-only normalization left 92,639 real rows unread
inside the exact window `BTCCHF` was recorded as "genuinely sparse".

**Measured:** gap-window coverage 667 → **92,637** (UTC); frozen-interior cells for that span
**182 → 0**, exact. Program-wide frozen-interior **1,012 → 830**, NOT-REAL **9.7334% → 9.5635%**,
genuine sparse set **5 → 4**. Scope 4 answered concretely: the fold does **not** move the
coverage ranking — zero symbols enter or exit the top-64 set in either window.

**Three MAJORs, and the first one is the interesting one.** The agreement gate had no
minimum-evidence floor: a merge sharing 0 or 1 timestamps was blessed exactly as confidently as
one sharing 48,780. The obvious fix was wrong — a per-window floor was *measured* to refuse
**seven already-shipped folds**, because each twin's rows begin precisely where the original's
stop. Identity is a property of the ids, not of the requested window, so the floor falls back to
the pair's whole history. `MIN_ALIAS_OVERLAP_TS = 64`, derived in code rather than chosen: 11×
under the binding fixture minimum (720), 167× under the real DB's (10,657).

The second: `runs/868_asset_identity_repair/duplicate_sweep.json` — tracked, cited by path in the
868 CHANGELOG entry, and literally this task's own Scope 2 output — still asserted "NOT
XBT-folding (different feeds)" and 55 groups. Regenerated to 56 / 14 live pairs. **The 868
CHANGELOG entry now understates the file it cites.**

The third: this task's own docstring rewrite falsely claimed the daily loaders apply the union.
They resolve by literal `assets.symbol` match and never invoke the alias machinery. Corrected and
**disclosed** per gate (iv): frame slot 11 `/USDC/USD` (id 996, 0 daily rows) reads INACTIVE while
its twin `USDCUSD` (id 1145) holds 721 days — the minute path unions the identical slot correctly.
Pre-existing to 868; wiring the daily loaders moves daily-pinned numerics, so it is a separate row.

**A local-time ghost, caught and killed.** `harvest_gate_i.py` resolved its window with naive
`.timestamp()` (local) while the suite used UTC — which is why two independent lenses reported
cell counts one apart. Now migrated: the last known TASK_869/D31(a3) survivor. The true pre-877
UTC interior is **1,012, not the 1,011 carried since 868**; 868's own figure is preserved verbatim
rather than silently restated.

**Two corrections to the task file itself**, both by re-measurement: XDG→DOGE has **zero**
collisions (not "stub-only" — no DOGE-spelled row exists in either DB), and its 697-row figure was
naive-local (667 UTC).

---

# TASK_886 — HALTED

## The finding: the row welded two concerns, and only one ever failed

TASK_886 carried (A) the **certification contract** — earn back the positive universe claim that
TASK_883R made structurally impossible — and (B) hardening the **checkpoint load path**, which
arrived when D37 was folded in as "item 0".

**Every BLOCKER in all three cycles was in (B).** (A) was never the problem.

### (A) — built, and verified harder than anything this program has shipped

`verified: true` binds to the LOADED parameter fingerprint AND the provenance's own recorded
checkpoint path AND the realized panel AND `n_padded == 0`. Two independent `opus` fail-closed
panels attacked it and could not forge it; both cycle-1 reproductions (foreign weights beside an
honest provenance; a forged provenance swapping slots 0/1) now refuse with no artifact. The
predicate was also proven **non-vacuous** — a synthesized matching fingerprint certifies and emits
one coherent `UNIVERSE CERTIFIED` sentence.

Also built: D37's load detector via **object identity** (flax's `restore_checkpoint` returns its
`target` unchanged on a miss) — 0/204 leaves aliased across four real checkpoints, 15 real
checkpoints loaded with zero false refusals; hole-1 resolution (all three `--checkpoint` spellings
produce an identical `universe_pin`); the mechanism-typed record; gate 0c; a named `n_assets`
refusal listing 17/204 mismatched leaves in place of a bare `flax.errors.ScopeParamShapeError`.
Byte-identity proven on **12 A/B pairs** against a HEAD worktree. All six suites exit 0.

### (B) — three cycles, each dying inside the previous cycle's own fix

1. **Numeric equality as the load detector.** `_trees_allclose(params_before, loaded.params)`
   treats "params unchanged" as proof of a no-op load. A checkpoint saved from a fresh-init
   `init_seed=0` model legitimately equals the grading trainer's init — and this program uses that
   pattern deliberately. It broke `experiments/861_search_teacher_imprint/smoke_e2e.py` and
   `experiments/835_stage4_eval/scorecard_smoke.py` (TASK_862's own acceptance gate). Measured A/B:
   HEAD exit 0 at net@21 +18.372, candidate exit 1. The refusal message even named the leaf it had
   found while claiming nothing loaded.
2. **The hang guard covered 5 named files; the loader opens the payload.** Three strace-proven
   infinite hangs on a FIFO at `checkpoint_0/d/<hash>`, `checkpoint_0/ocdbt.process_0/d/<hash>`,
   and `checkpoint_0/array_metadatas/process_0`. Same incident class TASK_883R already paid to fix,
   ~1,700 lines away in the same file.
3. **`realpath` guard vs `abspath` loader.** Cycle 2's realpath fix made `_resolve_checkpoint_dirs`
   resolve symlinks while `load_checkpoint_into` still passes `os.path.abspath`. `abspath` collapses
   `..` lexically; `realpath` resolves symlinks first — so a `--checkpoint` crossing a symlink and
   then `..` names **two different directories**. Proven end-to-end: the preflight blesses the clean
   copy, the loader opens the poisoned one and hangs (exit 125), and `check_checkpoint_arch` reads
   checkpoint A's metadata while the run grades checkpoint B. This repo ships exactly those symlinks
   (`jax_ckpt_fast`, `jax_ckpt_easy` → `/mnt/scratch`).

Finding 3 is this row's own thesis — *argv is not evidence* — re-opened one layer down: a guard
verdict about checkpoint A stamped onto a run that graded checkpoint B.

## Two findings that outrun the row

- **`verified: true` is unreachable for every checkpoint that exists today.** Nothing binds a
  training frame to weights, so the contract requires a `param_fingerprint` recorded *inside* the
  provenance — and no emitter writes one. Fail-closed and consistent with D36 ("ship nothing that
  can BLESS") and D37 ("only a recorded fingerprint makes the class falsifiable"), but it means the
  positive claim stays unearnable until an emitter row lands. Needs a ruling.
- **The D37 fail-open is not confined to the eval harness.** `train_jax_fast.py:544`
  (`--load-checkpoint` resume), `experiments/857_vhat_distill/distill.py:632` (`--warm-start`) and
  **`real_trade_jax.py:145` (live)** all print "loaded successfully" on a bad path. A multi-hour
  training resume can silently discard its checkpoint. Higher consequence than the case this row
  was fixing.

Smaller, recorded in D38: `_is_fixture_db` keys the LEDGER caveat on a *path*, so a byte-identical
fixture COPY ships "Scheduled-run numbers; PASS verdict applied by the planner" — strictly worse
than HEAD for that spelling.

## Process notes

- **The panel earned its cost.** Cycle 1's flow-verifier verified all 8 acceptance gates PASS while
  the regression lens simultaneously measured two shipped suites going red. The gates were written
  to the spec; the spec did not anticipate a guard that refuses honest work. A measured suite
  failure outranks a gate-level pass, and that is what caught it.
- **Independent convergence is the strongest signal available.** Correctness and fail-closed
  reached cycle 1's decisive BLOCKER by different routes and different reproductions.
- **A judge ran `git stash`/`git checkout` in place on the live working tree** while four other
  agents read it, making `eval_vector.py` flicker between HEAD and candidate. Caught by the
  flow-verifier, which froze a verified snapshot rather than trusting a single read; the
  orchestrator held a diff snapshot throughout and confirmed no loss. Subsequent briefs forbade it
  and required `git worktree add` instead. Worth putting in the judge base prompt.
- **Two HEAD-control traps, both hit and both documented:** `fixtures/trading_fixture.db` and
  `runs/831_ext/` are gitignored and absent from a fresh worktree (produces a false `teacher_floor`
  diff); and repo roots must be normalized on *both* sides when diffing artifacts, or `--db` and
  `out_pointer` manufacture differences.

## Invariants

`trading_data.db` never opened — size 26245337088, mtime still 2026-07-26 20:47:32. Fixture never
regenerated — sha256 `b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa918206cf9`,
79310848 B, independently re-verified by every agent at the end of its own run. No GPU jobs. Nothing
committed to code; nothing pushed.

## State at end of run

- Tree clean. The halted 886 work is `stash@{0}` (the TASK_883 halt stash is now `stash@{1}`).
- `tasks/DECISIONS.md`: **D38 OPEN** with the split recommendation and both outrunning findings.
- `tasks/QUEUE.md`: run-31 grooming note; row 4 (886) marked HALTED with "do not retake as
  written"; row 5 (877) struck through as shipped.
- Shipped: `task877-xbt-alias-fold` (`72c0c87`). Nothing pushed — the owner pushes after review.
- **Ready and takeable: TASK_878** (failed-fill semantics) → **TASK_881** (checkpoint protection
  model). Both were left for the next run deliberately: 878 rewrites eval-core execution semantics
  with every parity gate in the repo riding on it and carries a 3-arm ablation, and 881 is an M row
  in the lane that has already consumed nine factory cycles. Each needs a full run's worth of
  cycles; starting one at the tail of this session would have risked leaving it half-built.

## Why the daily-loader gap deserves its own row

Both halves of this run converged on the same shape from different directions. 886 died three
times on "which bytes on disk are we actually talking about"; 877's last MAJOR was "the docstring
says the union is applied here, and it is not". In both cases the code was fine and the *claim*
about the code was wrong — and in both cases a reader would have believed the claim. The daily
loaders are now honestly documented as alias-blind, but nothing pins that disclosure: no suite
asserts the `/USDC/USD` state, so a future backfill can stale it silently. That, plus the
`history.py:238-242` blanket `replace('BTC','XBT')` (which can resolve a request to the wrong
`asset_id` through a last-writer-wins dict), is the natural next row on this lane.
