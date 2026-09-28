# TASK_922 — PROTECTED.txt self-declaration convention (881E's forward item)

**Status**: IMPLEMENTED 2026-09-17 (interactive; uncommitted, not deployed). Focused TASK_922
and protection regression suites: **543 passed, 1 skipped, 1 deselected** on gpu-server;
independent flow verification and read-only review completed. Scope qualifications and
verification limits below.
Original proposal: 2026-09-06, from run 56's 881E ship — zero run dirs eligible for deletion
is D72's pre-registered outcome, not a defect.
**Size XS. Risk: LOW — one small writer hook + one reader clause; the deleter is under the
standing `--apply` freeze (D34/D35) which this row does NOT touch.**

## What

New-run convention: every run dir writes its own `PROTECTED.txt` at creation (name, task,
date, one-line reason), and the 881E deleter reads it as `covers:`-grade authority —
self-declared protection instead of grandparented manifests. As runs age, the zero-eligible
property then holds by construction rather than by 881i-era manifests that eventually rot.

## Gates

- **G1**: a run dir with `PROTECTED.txt` is refused by the deleter's dry-run; without it,
  normal `covers:` rules apply. Demonstrated both ways (the D83(i) lens: plant a dir whose
  PROTECTED.txt is malformed — the reader must fail LOUD, not silently treat as unprotected).
- **G2**: existing run dirs untouched (their protection rides the shipped 881E manifests);
  no `--apply` semantics change; freeze untouched.
- **G3**: the writer hook is one call in the run-dir creation path — no behavior change to
  any existing run (flag-off default where a hook cannot be inert).

## Implementation and verification — 2026-09-17

- `mcts_trader/training/run_declaration.py::create_protected_run_dir` creates a new directory
  and exclusive UTF-8 comment-only declaration (name, task, UTC date, reason). Existing
  directories and declarations remain untouched; invalid metadata and write errors fail loud.
- `train_jax_fast.py` replaces its checkpoint-directory creation with one helper call.
  **Coverage is this creation path only, not all independent experiment launchers.** The
  trainer has no task-id field, so its metadata explicitly says `task: unspecified`; no
  task identity is inferred from run names and no configuration schema was expanded.
- `tools/ckpt_gc.py` shares the new refusal between offline and in-run consumers. Empty or
  comment-only own `PROTECTED.txt` means self-protected. Malformed, unreadable, and wrong-shaped
  markers explicitly refuse even when another manifest grants valid consent. Ancestors are
  considered only through the deepest configured root. Existing directive-bearing own
  manifests keep their 881E consent semantics; merely having that historical filename is
  **not** an unconditional refusal. This qualifies G1's literal wording to preserve existing
  consent behavior. `citations=None` remains the existing explicit binding-off escape hatch.
- The `--apply` freeze and deletion algorithm are unchanged. No existing run directories,
  real checkpoints, fixture DBs, or deployed code were modified.

Verification against base `f0fa8ff`, candidate uncommitted:

- TASK_922 + TASK_881E pytest suites: **144 passed**, independently repeated on gpu-server
  in isolated copies with `safe_run`, 4G RAM cap. No JAX computation needed for these suites.
- Writer → mock checkpoint tree → CLI dry-run: marker present refused with zero planned
  deletions; removing only the marker made the same consented tree plan two deletions.
  Malformed text and invalid UTF-8 refused. All dry-run input bytes and mtimes were preserved.
- Mutation control on a separate source copy: disabling `_run_declaration_refusal` made
  **7 marker tests fail**. `--apply` without override still exited **2**.
- Scoped independent review: no concrete correctness findings. Syntax and whitespace checks
  passed. Full trainer startup was not exercised (hook placement is checked via AST).
- Full `trader selftest` was attempted by the flow verifier but did **not complete**: its
  staged snapshot omitted `experiments/835_stage4_eval/eval_vector.py`; CUDA initialization
  also encountered the server's existing driver/library mismatch. This is not a passing
  full regression suite. No server remediation was attempted.

Final expanded regression run (TASK_922, 881E, 881 protection model, 881G grammar,
881I ingestion): **543 passed, 1 skipped, 1 deselected** in 63.13s. The isolated server
snapshot includes the genuine `f0fa8ff` HEAD objects and tracked manifest corpus.
`test_self_referential_symlinked_dir_is_not_a_false_refusal` is deselected because its
real-corpus assertion requires gitignored checkpoint directories absent from the copy;
it failed on that missing prerequisite in the preceding run, not a measured code defect.
Earlier expanded runs exposed an unnecessary refactor of the pinned own-manifest reader;
that reader is now byte-identical to HEAD, and the added refusal check is additive. A duplicate
source-text mutation anchor was also removed; the unchanged 881E mutation guard now passes.
No regression assertions or allowlists were weakened. Final source is in server scratch
`/tmp/task922-verify-i8kIvS`.

Fresh independent mutation evidence: `/var/tmp/task922-fresh-d3903e64/`: seven marker tests
passed, all seven failed when only the new refusal was disabled in a copy, then all seven
passed after byte-exact restoration; 144 focused tests passed there too. That check preceded
the final equivalently-behaving, additive-only integration correction above.

Server evidence: `/var/tmp/task922-flow-verifier/` (`commands.json`, `01-marker.log`,
`01-marker-hashes.json`, `02-marker-removed.log`, `03-malformed.log`, `mutation.log`,
`focused.log`). This is an implemented, focused-verified candidate, not a factory release.
