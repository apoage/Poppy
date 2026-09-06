# Factory run 56 — handoff (2026-09-06)

Planner/implementer session, dark factory ARMED. THREE rows shipped. Base HEAD at start:
`e49bea0`, tree clean. The run crossed midnight; row 4 is dated 2026-09-07.

## Shipped

| Commit | Tag | Row | Cycles | Suite |
|---|---|---|---|---|
| `3461b15` | `task913-write-hardening` | 3 — recorder write-hardening | 2 | 907: 304 -> 326 |
| `f665c87` | `task881e-manifest-authority-binding` | 10 — TASK_881E | 3 (the limit) | ckpt: 1F/453P -> 545P |
| (see below) | `task921-guard-mechanism-fixes` | 4 — TASK_921 | 3 (the limit) | 907: 326 -> 330 |

`python -m trader selftest` = 17/17 at every checkpoint. `fixtures/trading_fixture.db` sha256
`b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa918206cf9` — verified unchanged four times
across the run, including after two accidental `--apply` invocations (below).

## Row 3 — write-hardening (`experiments/907_live_fill_probe/`)

Implements `WRITE_HARDENING_SPEC.md` in full. `CaptureWriter._write` now serialises JSON on the
calling (asyncio loop) thread and `put_nowait`s onto ONE bounded `queue.Queue`; ONE daemon worker
thread does `write()`/`flush()`/`os.fsync()`. Gap/meta get a priority fsync, never waited on;
everything else gets one `flush()` per record and at most one `fsync()` per `fsync_interval_s`
(default 30.0, `time.monotonic()`). `GAP_REASONS` 7 -> 9, additively (`emission_silence`,
`writer_overflow`).

**The BLOCKER, and why it is the run's most important artifact.** An overflow episode active at
`close()` lost BOTH `meta_end` and its `writer_overflow` disclosure **silently** — the 893 failure
mode (missing data presented as observed time) reintroduced through a new door. `_enqueue` only
resolved an episode when a NEW record was admitted, and `close()`'s own `meta_end` went through the
same gate, so it was folded into the drop count; `close()` then enqueued only the `_STOP` sentinel,
which triggers nothing. All 13 tests cycle 1 shipped were green over it.

Orchestrator reproduction, pre-fix (`queue_maxsize=4`, worker parked in `fsync`, 12 `write_book`):
```
record types on disk: ['meta_start', 'gap', 'book', 'book']
HAS meta_end: False | HAS writer_overflow disclosure: False
```
Post-fix, same driver with the worker NEVER released (the harsher case):
```
gap reasons: ['disconnect', 'writer_overflow']
overflow gap: start_ts=102.0 end_ts=111.0 detail='...10 record(s) dropped...coalesced...'
meta_end: False | writer_overflow: True
```
plus three log lines naming the undrained depth, the session drop count, and "closed_cleanly will
read False". `meta_end`'s absence is correct there and is now LOUD rather than silent.

Two deliberate, measured divergences from the spec's literal text, bannered on the file:
1. `_PRIORITY_RECORD_TYPES` has FOUR members — `meta_resume` included. §2.1's own rationale for
   `meta_start` applies verbatim (same `_declaration_fields()` call). Masking note: `fsync` syncs the
   whole fd, so the following resume gap normally covers it; the real hole is the branch where NO
   resume gap is written (`last_ts is None` — the previous session crashed before recording any data
   event), measured at ZERO fsyncs at session start.
2. §2.0.2's overflow disclosure also fires on the `close()` path (the BLOCKER above).

Worst-case `close()` wall time is now `2 x close_join_timeout_s`, measured at 2.001s for a 1.0s
timeout.

Remainders: `_abort_construction()`'s live-worker branch is unreachable from its only call site
(by the time `_worker_error` is set the worker has exited) and untested — dead defensive code today.
`RecorderConfig.__post_init__` does not range-check `heartbeat_timeout_s`, so
`--heartbeat-timeout-s nan` is caught only in the writer.

## Row 10 — TASK_881E (`tools/ckpt_gc.py`, `mcts_trader/training/safe_checkpoint.py`)

D72's authority binding: a `covers:`/`protect:` grant is honoured only where the run dir's own
881i-ingested manifest names the citing experiment back.

**Gate (ii) enumeration**, reproduced independently three times, dry-run: REFUSE **15 -> 18**;
the three dirs losing standing are `runs/832_e5` (cited by
`experiments/832_linear_imprint/PROTECTED.txt`), `runs/861_imprint` and `runs/861_ref` (both cited
by `experiments/861_search_teacher_imprint/PROTECTED.txt`). **Provenance (D83(ii)):** measured on
`workstation` at HEAD `3461b15`, where `jax_ckpt_fast` is a BROKEN SYMLINK
(`/mnt/scratch/...` vs the real automount `/run/mnt/scratch/...`) contributing
0 dirs; passing the resolved mount as a root gives OFF 21 -> ON 24, same 3-item delta.

**Zero run dirs are eligible for the scored-deletion path on this corpus** (0 of 18, or 24 with the
mount resolved, carry their own manifest). This is D72's pre-registered outcome, not a defect —
`tasks/DECISIONS.md:102` says verbatim: *"Undeclared/legacy dirs (no manifest) refuse — that is the
status quo under the freeze, i.e. manual operator delete; refusing more, never less"* and *"expect
the diff to be large-by-construction ... and ENUMERATED."* The `--apply` freeze is untouched and
stays behind 880's D35 gate, so the deleter was already frozen regardless.

**The BLOCKER.** Cycle 1's `--no-bind-authority` escape hatch was reachable by the bare abbreviation
`--n` — `argparse.ArgumentParser` at `ckpt_gc.py:3899` is constructed without `allow_abbrev=False`,
and `--no-bind-authority` is the only long option starting with `n`. Proven by an actual deletion of
two checkpoints on a scratch tree; invisible to all 28 tests cycle 1 shipped. The same file had
already hardened its sibling D64 override against exactly this with three independent defenses
(`ckpt_gc.py:3843-3895`, with the hazard analysed in a comment at `:3721-3724`). **The recurrence,
not the individual hole, was the defect** — both gates now share one spelling discipline.

Orchestrator's direct verification of the fixed gate:
```
--n / --no / --no-b / --no-bind / --no-bind-authorit -> EFFECTIVE bind=True  (argparse said False)
--no-bind-authority                                  -> EFFECTIVE bind=False
--no-bind-authority x2  |  full + abbrev  |  default  -> EFFECTIVE bind=True
```

Cycle 3 then closed a defect in that fix's OWN guard: the anti-recurrence meta-test asserted the
shared helper was CALLED, not that its result was USED. A lens constructed a gate that calls the
helper, ignores the answer, and fully reintroduces the bypass — and the meta-test passed. The guard
is now behavioural (drive each gate with an abbreviated argv, assert the fail-closed answer), with
an AST completeness test forcing any new `argv`-taking function into the contract table, and the old
AST check kept and honestly relabelled a cheap smoke.

Also fixed: a pre-existing RED at `e49bea0`
(`test_apply_behaviour_is_parity_with_head_on_a_scratch_tree`, whose HEAD-control leg was never armed
with the D64 override — `_head_source()` is `git show HEAD:tools/ckpt_gc.py`, so with a clean tree
both legs were the same source and only the arming differed), plus the second-order version of the
same trap where it would have gone red again once `_head_source()` began returning this commit.
Order-dependent owning-root selection fixed: first-match -> longest prefix.

Sanctioned pin change: `_d64_apply_refusal_reason` added to `_881D_KNOWN_REWRITES`. Verified not a
loosening — the digest pin in `test_ckpt_881_protection_model.py::D64_GATE_DIGESTS` is independent
and strictly tighter (catches additions, not only removals); all four digests independently
recomputed and matching, and mutations move them. Re-pinned to `f5eccefb…` with both prior digests
preserved in the comment.

Remainders: `--bind-authority` (ON-direction) remains freely abbreviable — harmless, sets the
default. `_safety_flag_occurrences` reads argv positionally and cannot distinguish an option name
from a value spelled like one; proven non-exploitable in the fail-open direction (argparse consumes
a registered flag token as the flag). `--roots` still lacks a `_dedupe_nested_scan_dirs` equivalent.

## Row 4 — TASK_921 (`test_streaming_reader.py`, no production code)

Rebuilds the two guards run 53 shipped broken. `git diff 3461b15 -- .../capture.py` is EMPTY.

**G2 re-base, recorded:** the task text pins `capture.py` byte-identical to `db8ed48`; that died when
row 3 shipped as `3461b15`. Verified the write-hardening diff touched the WRITER path only
(`read_capture`/`_pair_coverage`/`_last_event_ts` unchanged, `test_streaming_reader.py` still 2/2 at
`3461b15`), so 921's substance was unaffected and G2 was re-based rather than the row being blocked.

**Guard 1.** Run 53's version rebound `builtins.open` (intercepts zero `io.open` calls). The obvious
replacement — intercept `os.open` — was MEASURED not to work either: `builtins.open`/`io.open`/
`Path.open` never call the Python-level `os.open` name, because CPython's `_io.FileIO` issues
`openat`/`read` from C. **This corrected the orchestrator's own brief.** The guard now traces the
child (`strace -f -y -e trace=openat,open,read,pread64,readv,preadv,preadv2`) and asserts one open
and `bytes_read == file_bytes`. It fails LOUD on a zero-match parse (`n_opens == 1`, not `>= 1`) —
proven by planting a `(0,0)` parse, i.e. the exact run-53 "certifies on having measured nothing"
shape. Stated reach limits: mmap, `sendfile`/`splice`/`copy_file_range`, pre-attach inherited fd,
no-ptrace hosts.

**Guard 2.** `MAX_CHILD_BASELINE_BYTES` 64 MiB -> 24 MiB (silent-band deflation 158% -> 36% of
headroom at N=500,000). Gate is N-aware `min(flat, floor + headroom*n)` AND direction-aware: the
N-derived limit binds only where the bound is fail-open; where deflation only pushes toward RED the
flat ceiling is an explicitly-labelled smoke check. Forced by measurement — the poison child's
`types.ModuleType` shim costs ~2.70 MB (REAL 13,377,536 B vs POISON 16,072,704 B; independently
re-measured 13,312,000 / 16,068,608), so N=60,000 is both the smallest N and the highest baseline,
and a naive N-derived limit there clears it by only 1.046x (a flake).

**Two D83(i) findings against this row's own guards**, both closed:
1. The derivation named N=500,000 as binding while the file's own poison runs at N=60,000, where
   deflation is 207.93 B/event = 302% of headroom.
2. The resulting assert `used < headroom` is UNSATISFIABLE wherever the N-derived branch binds, since
   `used == headroom` identically there. Crossover N ~= 181,361. A second unsatisfiable assert
   (`== flat` for every fail-open row) was found alongside it.

**A rejected orchestrator suggestion, recorded because the implementer was right.** The orchestrator
proposed relaxing to `used <= headroom`. That is a tautology: given `ceiling = min(flat, floor +
headroom*n)`, it holds identically in both branches (confirmed, 0 counterexamples in 100,000 random
N). The shipped form is per-branch — strict `<` in the flat branch, equality within 1e-6 B/event in
the N-derived branch — graded against the derivation's own constants rather than against the
helper's return value, so a helper that stops implementing the derivation is caught.

**Residual:** `_HAVE_STRACE` is fail-open-TO-SKIP; a ptrace-less host silently skips both structural
guards (a skip, never a false green). Probe strengthened this cycle to trace a real read with `-y`
through the guards' own parser. `gpu-server` verified to have `/usr/bin/strace` with
`ptrace_scope=1` (permits tracing descendants; strace forks its own child), so the guard runs where
suites belong. Converting skip -> loud failure is separate hardening.
`CHILD_BASELINE_CEILING_CROSSOVER_EVENTS` is hardcoded 181,361, not recomputed — could go stale.

## The run's cross-cutting lesson — D83(i) earned its keep on day one

The rule was ruled 2026-09-06 (this morning). It fired FOUR times in this run, in four unrelated
places, and in every case the author's own review was green:

1. 907 cycle 1 — 13 green tests over a `close()` path that silently lost records.
2. 881E cycle 1 — 28 green tests over a safety flag disarmed by a 2-character abbreviation.
3. 881E cycle 2 — a meta-test asserting a helper was CALLED, not that its result was USED.
4. 921 cycle 1 — a ceiling derivation that named the wrong binding N.
5. 921 cycle 2 — an assertion that is unsatisfiable in exactly the regime the mechanism was added to
   cover, in a guard written to FIX a guard whose prose overstated its reach.

Every one is the same shape: a guard whose reach falls short of what its prose certifies. None was
caught by its author; all five were caught by an adversarial lens told to plant the poison where the
mechanism is BLIND rather than where it is strong. Instance 5 is the sharpest argument for the rule:
even a fix written in direct response to D83(i), by an escalated model, with the rule quoted in its
brief, produced another instance — and it was caught by planting the realistic case (adding a
plausible future row) rather than by reading the code.

**The orchestrator's own briefs were wrong twice, both caught by implementers rather than accepted:**
`os.open` interception cannot work at all (CPython issues the syscalls from C), and `used <= headroom`
is a tautology. An implementer that pushes back on its brief is worth more than one that complies.

D83(ii) also paid: the 921 finding surfaced only because a lens re-derived the arithmetic across
EVERY N present in the file rather than the two the docstring cited. The numbers were right; the
claim about which one was binding was wrong.

## Process findings for the operator

**1. Two accidental `--apply` runs against the real tree. Root cause is the harness, not the agents.**
Two different judges, given an explicit "NEVER `--apply` against real `runs/`" instruction, both ran
it against the live tree. Both disclosed it immediately and accurately. **The Bash tool's cwd resets
to `/home/lukas/Trader` between calls**, so any subprocess or shell command without an explicit
`cd`/`cwd=` runs against the real tree regardless of where the agent believes it is working.

Damage: NONE, verified independently by the orchestrator both times — checkpoint dirs intact
(`runs/832_e5` 1 ckpt, `runs/861_imprint` 3, `runs/861_ref` 3), `reclaimed 0.00 GB across 0
checkpoints`, fixture sha unchanged. The only side effect was `--apply`'s implied `--write-index`
rewriting three gitignored `checkpoints_index.json` files. Note the irony worth recording: the first
breach ran WITH `--no-bind-authority`, i.e. with the binding disarmed, and still deleted nothing —
because those dirs carry no manifest and were refused anyway.

Recommendation: briefs for any agent touching a deleter should state the cwd-reset fact explicitly,
not just the prohibition. That was added to later briefs this run and no third breach occurred.

**2. `jax_ckpt_fast` is a broken symlink on `workstation`** — points at `/mnt/scratch/...`
while the real automount is `/run/mnt/scratch/...`. Anything scanning checkpoints silently
sees 6 fewer dirs than exist. Not fixed (mount-layout question, operator's call).

**3. A subagent reported HEAD as `e49bea0` after `3461b15` was committed and tagged.** Verified false
(`git rev-parse --short HEAD` = `3461b15`, tag present). It had almost certainly run `git` from a
scratch copy. Standing lesson: verify a subagent's claim about repo state before acting on it.

**4. A stale detached worktree from a previous session** sits at
`/tmp/claude-1000/-home-lukas-Trader/ad3d49d4-.../scratchpad/wt914` (`git worktree list`). Not this
session's; left alone.

## Queue state at handoff

Rows 3, 4 and 10 struck SHIPPED. **Next takeable: TASK_904B (row 11) -> TASK_908 (row 12) ->
TASK_906 (row 13).**

Open decisions unchanged by this run: **D81(a)/(b), D84, D80(ii)/(iii), D43, RO02.** None was
touched; no new decision row was seeded. One candidate was considered and WITHDRAWN: "881E leaves
the deleter inert" is not an operator question, because D72 already ruled it (`refusing more, never
less`) and the `--apply` freeze is independent.

Operational follow-up, queue work rather than a decision: for coverage to be non-empty again, run
dirs must start writing their own `PROTECTED.txt`/`SPEC.md`. Writing them into the three named dirs
is an operator action per D72 ("a mismatch is a FINDING to report").
