# Factory run 31 handoff — TASK_886 HALTED on the 3-cycle rule

**Date:** 2026-08-10/11 · **Harness:** Claude Code (Opus 5, 1M context) · Subagents per
`.dark-factory/config.json` model policy: implement on `sonnet`, fix passes escalated to
`opus`, final-cycle panel on `opus`.

## Outcome

Nothing shipped. `TASK_886` halted after three failed fix→verify cycles; code preserved in
`git stash` ("TASK_886 … HALTED on the 3-cycle rule (D38)"). Tree clean, nothing committed to
code, nothing pushed. **D38 seeded OPEN.**

Twelve subagent verifications ran: a 5-lens panel (correctness · fail-closed · regression ·
security · flow-verifier), a 4-lens re-verify, and a 3-lens `opus` re-verify.

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

- Tree clean; the halted work is `stash@{0}` (the TASK_883 halt stash is now `stash@{1}`).
- `tasks/DECISIONS.md`: **D38 OPEN** with the split recommendation and both outrunning findings.
- `tasks/QUEUE.md`: run-31 grooming note; row 4 marked HALTED with "do not retake as written".
- Ready rows untouched and takeable: **TASK_877** (XBT alias fold) → **TASK_878** (failed-fill
  semantics) → **TASK_881** (checkpoint protection model).
