# Factory run 12 handoff — 2026-07-18

Dark-factory implementer run. Cleared both Ready rows (TASK_847, TASK_848 — the two
D17-ruled instrument builds). Zero OPEN decisions consumed; none seeded. Everything
remaining is planner-scheduled real-panel work.

## Shipped

| Task | Tag / commit | What | Verification |
|------|--------------|------|--------------|
| TASK_847 | `task847-policy-prior-proposals` / `873effe` | `policy-prior` proposal scheme for the Sampled-MuZero vector search (D17 (1), the read-A inversion). New `--vector-proposal-scheme policy-prior` (default UNCHANGED). Slot 0 = guaranteed legality-masked policy-argmax vector; slots 1..K−1−N = i.i.d. per-asset draws at `--vector-proposal-temp`; last N = dithered explore slots. Wired through training CLI, 835 eval, 834/836 plumbing. | Medium panel + fix re-verify. Canon pin `22b0c7f5…` bit-identical; 15 tests; selftest 8/8 (new ST8). Flow-verifier confirmed the mechanism on a fixture copy: slot0==policy-argmax 40/40; dithered-book contains the argmax vector 0/40 — the exact policy-blindness D17 diagnoses. 5 findings fixed+re-verified. |
| TASK_848 | `task848-policy-replay-instrument` / `371e7c5` | Policy-argmax replay instrument (D17 (2)/Track 2, the OOS-read mechanics). New `--operator {search,policy-argmax}` (default search = byte-identical) on the 835 eval + `--start/--end` window freedom + on-window `--teacher-floor`. New JSON: `operator.mode`, `teacher_floor_inenv`, `corridor_position`. | Medium+ panel (correctness/fail-closed/regression) + 2 fix re-verifies. Default search byte-identical (net@21 −43.972…); round-trip byte-exact vs imprint (gross −6.409/net@21 −6.778/hist [9,388,3,0,0]); selftest 8/8. 1 BLOCKER + 2 MAJOR + MINORs all fixed+re-verified. Only `experiments/835_stage4_eval/` touched. |

## Decision-relevant notes (no new DECISIONS rows — all resolved in-lane)

- **847 acceptance (i) numeric leg is NOT fixture-reproducible.** The real imprint ckpt is
  64-asset/real-panel and cannot map onto the 8-symbol fixture minute slice; on an untrained
  fixture smoke model there is no prior/value to preserve, so no scheme "preserves" it. The
  fixture-verifiable MECHANISM leg is confirmed (argmax guaranteed at slot 0; dithered-book never
  offers it). The binding NUMERIC leg (search-eval of the frozen imprint ckpt with `policy-prior`
  preserves the prior, net@21 ≥ +0.74 = +0.839 − 0.10) is the planner's scheduled real-panel run,
  same discipline as 846/(iv).
- **848 BLOCKER found + fixed: leak-free invariant was at risk.** A malformed `--start`/`--end`
  (e.g. `2025/09/01` with slashes) was coerced to a full-DB sentinel (start→epoch 0), silently
  folding the imprint's IN-SAMPLE window into an "OOS" generalization verdict. Now strict-parsed
  and fail-loud. Worth the planner knowing the exact OOS command uses `YYYY-MM-DD` (dashes).
- **848 pre-existing silent-degeneracy found (affected the OLD search path too):** a wrong
  `--h-cycles`/`--l-cycles` silently ran the wrong TRM net (weights are shared across recursion
  cycles, so no shape error). A `check_checkpoint_arch` guard (reads `trainer_metadata.json`) now
  covers both operators. It fails-OPEN on MISSING metadata (legacy metadata-less ckpts must still
  run under search) and WARN-continues on corrupt/non-dict metadata. Any past 835 run with
  mismatched cycles would have been silently invalid — heads-up for the planner.
- **848 order-code teacher-floor is intentionally out of scope.** The weight-native ladder teacher
  walked through the order env is incoherent; the fix fail-closed-rejects `--teacher-floor` without
  `--vector-weight-rebalance` (mirrors imprint.py:479-482) rather than adding an order-code mapping.
  The real D17 OOS ckpt is weight-native and always passes `--vector-weight-rebalance`, so its
  round-trip stays coherent WITH the floor.

## Where it stands / planner-owned next steps

Both instruments are ready; the science is now scheduled real-panel work the factory does not run:

1. **847 real-panel read-A inversion** (the binding numeric acceptance leg): 835 search-eval of the
   frozen imprint ckpt (`runs/832_e5/ckpt_imprint_full`) with `--vector-proposal-scheme policy-prior`,
   bar net@21 ≥ +0.74. This is the exact test that fails at −6.92 today under dithered-book.
2. **848 real OOS read** (Track 2, BENCHMARK only per the operator D17 correction — no deployment-gate
   status): `--operator policy-argmax --checkpoint runs/832_e5/ckpt_imprint_full --start 2025-09-01
   --end <latest DB day> --teacher-floor --vector-weight-rebalance`, tiers {0,21}. Pre-registered:
   imprint tracks teacher OOS if net@21 ≥ teacher_floor − 0.10 → LEDGER 832/E5-oos. A PASS routes to
   the full benchmark sequence (canonical scorecard, regime windows, walk-forward), and only then the
   4-gate ladder — never straight to deploy.
3. **Short ExIt re-try** with the fixed operator (D17 (1)): after 847's real-panel leg passes, ≤300 ep
   (the erosion window) warm-started from the imprint ckpt at fee16. If the policy survives its own
   training loop, resume the DEC-2 ladder; if it still erodes, read B (value/reward leaf-eval) is the
   next locus.

## Ops

- Clean tree, both tags local, nothing pushed (operator pushes after review per the deploy contract).
- No GPU jobs launched (both were code-only fixture-scale builds). No real-data runs touched.
- Factory disarmed at end of run.
