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
