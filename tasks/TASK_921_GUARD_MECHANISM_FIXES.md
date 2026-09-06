> **STATUS (2026-09-07): SHIPPED as `task921-guard-mechanism-fixes`** (factory run 56, which
> crossed midnight). Both guards in `test_streaming_reader.py` rebuilt. Suite
> `experiments/907_live_fill_probe/` 326 -> **330 passing**; `test_streaming_reader.py` 6 passed,
> 0 skipped; `python -m trader selftest` 17/17. Three cycles (the limit), two adversarial lenses.
>
> **G2 RE-BASED (recorded because the task text is now stale):** G2 said "`capture.py`
> byte-identical to `db8ed48`". That pin died when the write-hardening row shipped as `3461b15`
> and rewrote `capture.py`'s WRITE path (555 lines). Verified that diff touched the writer path
> ONLY — `read_capture`/`_pair_coverage`/`_last_event_ts` unchanged — so 921's substance was
> unaffected and G2 was re-based to `3461b15`. **Met: `git diff 3461b15 -- .../capture.py` is
> EMPTY. This row changed no production code.**
>
> **Guard 1 (structural, single-traversal).** The brief's own proposed fix — intercept `os.open` —
> was measured NOT to work: `builtins.open`/`io.open`/`Path.open` never call the Python-level
> `os.open` name, because CPython's `_io.FileIO` issues `openat`/`read` from C. The guard instead
> runs the measurement child under `strace -f -y -e trace=openat,open,read,pread64,readv,preadv,
> preadv2` and asserts exactly one open and `bytes_read == file_bytes`. It fails LOUD on a
> zero-match parse (`n_opens == 1`, not `>= 1`) — verified by planting a `(0,0)` parse result, which
> is the precise run-53 "certifies on having measured nothing" shape this row existed to kill.
> **Stated reach limits (G3), in the file's own prose:** mmap-backed reads, `sendfile`/`splice`/
> `copy_file_range`, an fd inherited before tracing attached, and hosts without working ptrace.
>
> **Guard 2 (RSS cost).** `MAX_CHILD_BASELINE_BYTES` re-derived 64 MiB -> **24 MiB**: silent-band
> deflation drops from 158% of honest-to-bound headroom to 36% at N=500,000. The gate is now
> N-AWARE (`_max_child_baseline_bytes(n) = min(flat 24 MiB, floor + headroom*n)`) and
> DIRECTION-AWARE — the N-derived limit binds only where the bound is fail-open (`measured < BOUND`,
> where deflation can hide a regression); where it is fail-closed (`measured > BOUND`, deflation
> only pushes toward RED) the flat ceiling stays an explicitly-labelled smoke check.
>
> **Why direction-aware rather than simply N-aware:** the poison child loads `capture.py` through a
> `types.ModuleType`/`compile`/`exec` shim costing ~2.70 MB extra baseline (REAL 13,377,536 B vs
> POISON 16,072,704 B; independently re-measured at 13,312,000 / 16,068,608, delta 2.63 MB). So
> N=60,000 is simultaneously the SMALLEST N and the HIGHEST baseline in the file, and a naive
> N-derived limit there would have cleared the poison child by 1.046x — a guaranteed flake.
>
> **Two D83(i) findings against this row's own guards, both closed:** (1) the derivation comment
> named N=500,000 as the binding N while the file's own poison test runs at N=60,000, where
> deflation is 207.93 B/event = 302% of headroom; (2) the resulting derivation test asserted
> `used < headroom`, which is UNSATISFIABLE wherever the N-derived branch binds, since
> `used == headroom` identically there — so the "holds by construction for any future N" claim was
> false in exactly the regime the mechanism was added to cover. A second unsatisfiable assert
> (`== flat` for every fail-open row) was found and fixed alongside it. Crossover is N ~= 181,361.
>
> **Note on the fix that was NOT taken:** the orchestrator suggested relaxing to `used <= headroom`.
> The implementer rejected it as a tautology and was right — given `ceiling = min(flat, floor +
> headroom*n)`, `used <= headroom` holds identically in both branches (confirmed: 0 counterexamples
> in 100,000 random N). The shipped form asserts strict `<` in the flat branch and exact equality
> within 1e-6 B/event in the N-derived branch, graded against the derivation's own constants rather
> than against the helper's return value.
>
> **Known residual, named:** `_HAVE_STRACE` is fail-open-TO-SKIP — a host without working ptrace
> silently skips both structural guards (never a false green; a skip, not a pass). The probe was
> strengthened this cycle to trace a real read with `-y` through the guards' own parser, but
> converting skip -> loud failure on hosts that are supposed to have strace is a separate hardening
> step, out of this row's scope. Verified `gpu-server` has `/usr/bin/strace` with `ptrace_scope=1`
> (permits tracing descendants; strace forks its own child), so the guard runs where suites belong.
> Also: `CHILD_BASELINE_CEILING_CROSSOVER_EVENTS` is a hardcoded 181,361 rather than recomputed,
> and could go stale in prose if the constants move.

# TASK_921 — the two 914 guard mechanism fixes (D82(b) follow-up row)

**Status**: PROPOSED 2026-09-06 (planner; the mechanism half of D82(b) — the prose half
shipped with `db8ed48` / tag `task914-streaming-reader`).
**Size XS/S. Risk: LOW — one test file + its measurement harness; production `capture.py`
is FROZEN (shipped, byte-identity-pinned; touch nothing outside tests).**

## The two mechanisms (both measured holes in `test_streaming_reader.py`, receipts in
`lore/handoff_2026-09-05_factory_run_53.md` and now transcribed in the shipped file's
own caveats)

1. **fd-layer interception for the structural guard.** Rebinding the `builtins.open` NAME
   intercepts zero `io.open` calls (`io.open` IS `builtins.open`; `pathlib.Path.open`
   holds a direct import-time reference) — measured counterexample `rig_stream_io`:
   a genuinely double-traversing reader certifies "traversals=1.000", n_opens=1, PASSED.
   Fix at the layer every Python path must cross: intercept `os.open`/`os.open_fd`
   (fd-level), count per-fd read bytes for the capture file's inode, and assert total
   bytes read across ALL fds == file size (a second traversal through ANY API re-opens or
   re-reads and cannot hide). Non-vacuity: re-plant `rig_stream_io` and the `list(...)`
   materialization as poisons — both must FAIL RED under the new guard.
2. **Baseline-independent per-event cost, or a re-derived ceiling.** `VmHWM(after) −
   VmHWM(baseline)` deflates by the child's own pre-baseline high-water mark; the 64 MiB
   ceiling only catches baselines ABOVE it, leaving a silent 13 MB–64 MiB band in which a
   duplicated full-file pass still passes (measured: 166.35/158.62/154.47 B/event at
   58.2/62.3/64.4 MB baselines; 108.6 B/event = 158% of the honest-to-bound headroom at
   the ceiling). Either (a) measure the reader's cost in a way that cannot be deflated
   (e.g. `smaps` rollup sampled around the call, or child-internal peak WITH the baseline
   included and the ceiling applied to the MEASURED figure's deflation bound), or (b)
   re-derive `MAX_CHILD_BASELINE_BYTES` from the deflation arithmetic so the silent band
   is smaller than the honest-to-bound headroom, and state the arithmetic in the file.

## Gates (binary)

- **G1**: both poisons (rig_stream_io double-traversal; duplicated raw-line full-file pass
  at a ~60 MB child baseline) FAIL RED under the fixed guards — demonstrated, not argued
  (plant them in the PR/tests).
- **G2**: the shipped 2/2 tests still pass; full `experiments/907` suite green;
  `capture.py` byte-identical to `db8ed48`.
- **G3**: the file's claims match its measured reach — the D83(i) lens check (a guard that
  cannot reach a defect it claims to catch is a MAJOR, per the new conduct rule).

## Out of scope

`capture.py` (frozen), the estimands, any bound VALUE changes not derived from the
deflation arithmetic (190 stands unless (2b) re-derives it).
