# Factory run 34 — 2026-08-16

Three rows taken in parallel on disjoint surfaces: **TASK_886** (rank 1, certification contract),
**TASK_891** (filler, `tools/deploy/`), **TASK_893** (filler, the D41 measurement).

| Row | Outcome | Ref |
|---|---|---|
| TASK_893 | **SHIPPED** — 2 cycles | `c1e556d` · `task893-prefix-mask-measurement` |
| TASK_891 | **HALTED** on the 3-cycle rule — nothing shipped, D42 seeded | `685fcf6` (untagged, a STOP) |
| TASK_886 | **SHIPPED** — 3 cycles | `a0da06b` · `task886-universe-pin-certification` |

| TASK_890 | **SHIPPED** — 2 cycles, with acceptance (i) explicitly UNMET; D43 seeded | `2804eb7` · `task890-distill-open-gate` |

Three OPEN decisions now sit on the planner's desk: **D41** (ruleable — its measurement landed), **D42**
(891's halt), **D43** (new — the 857 smoke-assert).

---

## TASK_890 — the CLI runs again, and repairing it exposed something nobody could see

The gate is applied per D40, the in-process path is byte-identical when it is inert, and the artifact can
no longer claim work it did not do. Two defects the panel caught are worth carrying:

1. **The counts were computed before the mask was forwarded.** Reverting one kwarg at the `run()` call site
   still printed `force-HOLD 8 of 240 … applied to the counterfactual candidates` and still wrote
   `n_force_hold_decision_cells: 8` — while zero candidates were gated. That is D40's own diagnosis,
   "absence read as a value", recreated one level below where it ruled. Counts now accumulate **on the
   force-HOLD branch itself**, so no number is producible without the application.
2. **The test could not see the defect it existed to catch.** It never called `run()` — it built the mask
   itself and handed it to `collect_pass`, exercising machinery that was already correct. Two reviewers
   independently rebuilt the real regression and the test stayed **11/11 green**. It now drives the real
   CLI in a subprocess and grades the artifact.

**`8 of 240` understated the effect ~15×.** Gating a WALKED action diverges the portfolio trajectory from
that decision onward: `decisions_with_ANY_changed_G` is **29/30 per teacher pass**, 0/30 per search pass —
**58 of 120** decisions, roughly half the training set, against 3.3% of raw cells. Measured by the
flow-verifier, then re-derived by the implementation to an exact match. Both figures now ship.

**ACCEPTANCE (i) IS UNMET, deliberately.** The documented bare invocation now runs the whole pipeline and
then fails its **own** `--smoke-assert` (v̂-alone `+0.1251 → −0.3953` at seed 0). An unpack-and-discard arm
fails a *different* check; at **seed 1 both arms fail the same one**. So the instrument is noisy at fixture
scale and the gate is **not** orthogonal to it. **Thresholds were not re-tuned and `SPEC.md` was not
touched.** The CLI had been dead since 868/874 — it had never once executed past the arity bug, so this
gate's state was simply unknowable before today. **D43** asks the only question that matters: *was this gate
ever green post-868, on ≥3 seeds?* Nobody can currently answer it, and until someone does, **no 857 number
should be quoted from this path.**

---

## TASK_886 — the program may now certify a universe, and it states the limit of what it knows

Contract only, per D38 — the load path stayed 888R/889's. `verified: true` is **reachable and unforgeable**:
five lenses plus a fresh re-verification could not produce it through path spellings, document forgery,
TOCTOU, type confusion, 500K-char/NUL payloads, or 15 hand-built adversarial conjunct combinations.
Acceptance (v) byte-identical vs a HEAD worktree (17 keys, 0 differing); 861's `i_canon_835` and
`results_pin` canaries green across all three cycles.

**The 883R assertion flips were treated as guilty until proven otherwise, and came back earned.** A judge
established it by training a checkpoint from scratch *outside* the harness and measuring fingerprint
equality on real training output — not by reading the diff. Structurally the flips also went 2 assertions →
4 while keeping every prior one, with every refusal gate untouched: the opposite shape from accommodation.

**The finding that matters most, and the one to carry forward.** `param_fingerprint` hashes parameter
shape/dtype/bytes and has **zero dependence on symbol names** — weights do not encode which asset a slot
means. So reversing `frame_symbols` in a checkpoint's own provenance with the fingerprint intact still
certified, moved **net@21 −1.7963 → −1.6575**, and printed *"slot identity is bound to the checkpoint
actually LOADED."* Found by running the real flow, not the suite. The structural close belongs to **a new
row on 889's surface** — record a `frame_symbols` digest inside the checkpoint at SAVE time — after which
`_certification_verdict` gains a fifth conjunct and the `NOT PROVEN` half of the sentence can be deleted.

**Then the fix for that overclaim contained a smaller overclaim and a hang** — worth recording as its own
lesson. The `.sha256` pin check added to partially close it (a) opened every `*.sha256` in the directory
with no `stat.S_ISREG` guard, so a FIFO named *anything* hung every future certification of every document
there — D38's cycle-2 hang class recreated in new code, while the sibling function in the same file stats
first *for exactly that reason* — and (b) claimed a matching pin meant tampering was "ruled out", when the
pin sits behind the same write permission as the document. The cycle-2 implementer's own addendum named it
precisely: *"I closed a BLOCKER about an overclaiming certification sentence and introduced a smaller
overclaim in the replacement."* **A claim about DETECTION needs the same "what is NOT proven" treatment as
a claim about BINDING.**

**Two guards were being enforced by accident.** `_certification_verdict`'s `mechanism == "none"` check was
consulted only for the *reason string* after the success branch had already returned; the invariant held
solely through non-local coupling to another function's dict literal, and deleting the branch changed no
test outcome. The same audit found `order_exact` in the identical state. Both are now live conjuncts (an
allowlist, not `!= "none"`, because a record with **no** `mechanism` key also certified), pinned by direct
predicate probes rather than through the producer's coupling. **"A guard whose removal keeps the suite
green is not a guard" earned its keep twice in one row.**

**Planner remainders.** (1) The save-time `frame_symbols` digest (889's surface). (2) `runs/` is gitignored,
so D39's "committed pins" are committed *operationally*, not in the `git ls-files` sense — a genuinely
independent pin has to move somewhere tracked. (3) **A HEAD-worktree A/B on `eval_vector.py` needs BOTH
`fixtures/trading_fixture.db` AND `runs/831_ext` symlinked in.** A missing `831_ext` silently nulls every
`per_window` floor/ceiling while `sharpe_full` still matches, and reads as a `tiers` regression that is not
real — this cost one false alarm during the run. That is `feedback_head_control_worktree_rig` with one more
dependency in it.

---

## TASK_893 — the D41 measurement landed, and D41 is now ruleable

**Numbers for the ruling.** 4933 prefix rows / 39 assets · **4904 decision cells inside FULL =
7.3793%** (4518 on the scored grid) · concentrated BULL21/BEAR22/REC23, none after 2023 · masking
every prefix cell inactive moves the bar by **exactly 0.000000** (net@21 `+0.9673047900277524` both
sides) · **0** as-shipped positions ever land inside a prefix. So D41 option (a)'s condition — "IF
the prefix rows are provably never traded" — is **met**.

**Two riders the ruling must not skip.**

1. **The Δ=0 does not license (a) on its own.** It is a *structural corollary* of `present_mask`'s
   `warm=30`, not an independent test: returns inside a bit-identical prefix are exactly zero, so
   `first_active` can never precede the prefix boundary. The instrument is provably responsive (a
   planted position moves it `+0.000749`, and that positive control now ships as a permanent
   regression test that fails loudly if it ever goes vacuous) — but the zero is guaranteed *for this
   strategy construction only*. **Any consumer reading `close_d`/`logc` without routing through
   `present_mask` gets no protection, and this instrument would not catch it.** (a) leaves that trap
   for the next consumer; (b) closes it.
2. **The docstring claim is DISPROVEN, not merely imprecise.** SOL's row 167 carries a price from
   **59,760 s in its future**; XDG's row 0 from 420 s. That sentence cannot survive any option.

**Second defect, surfaced and deliberately NOT ruled.** `build_close_aligned`
(`runs/819_single_asset/portfolio.py`) admits bars only where `timestamp >= grid_start`, so **16
majors that DO have real earlier history** still receive a 1-row backward fill. This moved the count
4917 → 4933; all 16 land on row 0, outside FULL, so no scored number moves. It lives in a
planner-owned prototype and was left alone — it wants its own row or a rider on D41.

**How the cross-check got fixed, because the lesson generalises.** Cycle 1 reported "26/39 exact,
13/39 +1, 0 unexplained". Every one of the 26 agreements was a *zero-prefix* asset (trivial), every
asset with anything to check disagreed, and an auto-classifier stamped any `+1` gap as "explained"
without testing the mechanism. **Two judges reached opposite conclusions about whether that
explanation held** — one verified the timestamps and called it sound, the other checked the *values*
and called it a rationalisation. Neither had the mechanism. It was settled by a value anchor:
`close_d[0] = 40.23` is exactly the close of the bar at `2021-06-17 15:32:00Z`, while the last row
carrying that value is stamped `2021-06-16 22:56:00Z`. The docstring's premise fails because the npz
is built on a **one-minute** grid, filled there, then subsampled `[::1440]`, so the fill's source row
almost never lands on the daily grid. Paths A/B now agree **39/39 exact, 29/29 on nonzero prefixes**;
the npz-run detector is demoted to Path C and reported AMBIGUOUS, never as agreement.

---

## TASK_891 — HALTED, D42 seeded

**The trap is still live on gpu-server.** Nothing is worse than HEAD (every failing spelling behaves
exactly as pre-891), but the defect that cost two launches (861 chain, 879 rung-3) is **not fixed**.

Three cycles, three panels, one shape — the escaping spelling got more exotic each time:
`env python` → `\python` → `PATH= python`.

- **c1**: whole-`trader_env/bin` PATH prepend + leading-token refusal. Bypassed by
  `env`/`nohup`/`time`/`bash -c`/`&&`/`PYTHONPATH=` spellings, which then received the interpreter the
  guard had just rejected; the blanket prepend silently swapped `pip` 23.0.1 → 26.0.1 on rows that
  never died at HEAD, labelling them `passthrough`; the version probe was unbounded.
- **c2**: three-class model + word-level regex. `\python` and `python$EMPTY` escaped the strip class;
  `echo python && pip --version` re-triggered the `pip` collision through the new `embedded` class.
- **c3** (orchestrator-directed redesign): a **python-only shim dir** — exactly `python` + `python3` as
  `exec` wrappers, *not* symlinks. That distinction was measured, not assumed: a symlink shim makes
  CPython take `sys.prefix` from the symlink's own directory, miss `pyvenv.cfg`, and silently load the
  base pyenv's site-packages **while passing every version check**. It closed both prior MAJORs
  structurally (A/B job-log diff exactly one hunk, `pip` stays 23.0.1, 11 independently-rebuilt
  mutations with 0 survivors, suites 109/109 + 16/16 + 18/18, selftest 12/12).

**What halted it.** A manifest row that assigns `PATH=` itself (`PATH=/dir python …`, `env -i PATH=…
python …`) locally shadows the injected shim — bash performs the per-command assignment *before* the
path search. The job runs the ambient interpreter, exits `rc=0`, and prints a **false**
`interpreter=venv-path:…` label, which also hides it from a pre-launch `--dry-run` review.
Orchestrator-reproduced first-hand (`AMBIENT_RAN` vs a `SHIM_RAN` control).

**D42(a) recommends** re-cutting to an honest, bounded contract and shipping c3's shim behind it, with
two small changes: never print a resolution claim the code cannot back, and re-derive the
documented-limits list from the actual mechanism. Code is in `stash@{0}` with a full resume note.

---

## Where the run stopped, and why

Remaining Ready rows: **878** (rank 3), **892** (rank 4, gated behind 878), **881** (rank 5, deliberately
last). None is blocked in the mechanical sense — but I stopped rather than open 878, for a reason the
planner should weigh rather than just accept:

**878 rewrites eval-core execution semantics with every parity gate in the repo riding on it, and the queue
itself marks it "fourth and alone."** More importantly, two of the three decisions this run seeded bear
directly on it. **D41** is about what a fabricated bar means in a data path; **878** is about what a
no-print bar means in eval-core execution. Both are the same underlying question — *what is true when a bar
has no real print* — and 878 also re-specifies exactly the cell TASK_890 just changed under D40. Ruling D41
before 878 is built is cheap; discovering afterwards that they disagree is not. D39 records that four of the
last five planner boundary errors were inside this one lane.

---

## Two process findings worth keeping

**1. A judge ran `git stash` on the live tree and stripped a concurrent lane's work.** With three
implementers in flight, the TASK_891 correctness judge stashed to get a HEAD diff, hit a tool timeout
before restoring, and briefly removed the TASK_886 lane's uncommitted `eval_vector.py`. It restored
the file and I verified it byte-for-byte against the stash (all 424 insertions intact), but it left a
stray `WIP on master` stash and staged index entries behind. This is exactly
`feedback_head_control_worktree_rig`, and prompt-level guidance was not enough — I made
"never `git stash`/`checkout`/`reset` on the live tree; use `git worktree`" a **hard constraint block
at the top of every subsequent judge brief**, and there were no further incidents across nine more
agents. **Recommend making it a standing clause in the `dark-factory-judge` agent definition**, since
parallel lanes are now normal for this factory.

**2. Disagreeing judges are a signal to measure, not to average.** Twice this run two lenses reached
opposite conclusions on a load-bearing fact (the 893 `+1` offset; the severity of 891's cycle-2
regex gap). Both times the resolution came from the orchestrator running a direct measurement rather
than adjudicating the arguments — and both times *neither* judge had the actual mechanism. The
factory's "objective checks outrank judge opinion" rule earned its keep.

**3. "A guard whose removal keeps the suite green is not a guard" fired FOUR times in one run,** across
three rows, and it is now the highest-yield question this factory asks:
- 886: `mechanism == "none"` was consulted only for a reason string; deleting it changed no test outcome.
- 886: `order_exact` was held solely by another function happening to `raise`.
- 890: the artifact's force-HOLD counts were computed *before* the mask was forwarded, so they printed
  identically with the gate disconnected.
- 890: the regression test never called the function whose wiring it was testing, and stayed 11/11 green
  against the real defect.

The common shape is not a missing check — it is **a check that lives somewhere other than where the claim is
made**. Recommend promoting *"disable this guard and prove the suite goes red, at the REAL drop site"* to a
standing clause in the judge brief rather than a per-row instruction; it caught all four.

**4. Running the documented invocation beats running the suite.** Run 33's lesson repeated exactly three
times: 886's symbol-order BLOCKER, 890's test-blindness BLOCKER, and 890's acceptance-(i) failure were all
found by executing the real CLI, and all three were invisible to green suites.

**5. Parallel-lane hygiene — second incident, different shape.** A judge ran `distill.py --tag judge_default`
with no output redirection, which would have overwritten the **git-tracked**
`experiments/857_vhat_distill/results_fixture_smoke.json`. Verified no damage landed. Combined with the
`git stash` incident, the pattern is that verification agents need explicit **"do not write into the repo"**
fencing, not just "do not edit files" — a default `--out` landing on a tracked artifact is a write nobody
thinks of as an edit. Also noted: a gitignored `experiments/857_vhat_distill/ckpt_smoke/` written by an
earlier verification run, the same silently-rebasable shared-artifact class as the run-25 fixture incident.

---

## Housekeeping / open ops items

- **`stash@{1}` is a stray `WIP on master`** left by the judge incident. Its content is superseded
  (the tree carries the same or newer `eval_vector.py`, and `run_queue.sh` now lives in `stash@{0}`).
  I did not drop it — dropping another agent's snapshot is destructive and stash indices are
  load-bearing in D38/D39 — but **the planner should verify and drop it**, because it shifts every
  index the older decision rows refer to.
- `tools/deploy/` still needs redeploying to gpu-server after any future 891 ship (as with TASK_833).
- The watchdog suite (`--stall-timeout 3 --stall-grace 2`) is a **pre-existing load-sensitive flake**,
  reproduced at HEAD in a clean worktree (10/18, 15/18, 15/18) as well as on candidates. Not caused by
  this run; untouched. Worth its own row if it keeps costing verification time.
