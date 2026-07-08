# Handoff 2026-07-09 (#6) — Factory run 5: 836 / 837 / 838 (second instrument batch + the canon flag)

Previous: `lore/handoff_2026-07-08_factory-run4-instruments.md`
Branch: `master` · Working dir: `/home/lukas/Trader` · Role: ORCHESTRATOR (implementer fan-out; panels + fix passes + release ritual owned here)

## What shipped

| Tag | Task | Panel outcome |
|-----|------|---------------|
| `task836-e4-rank-probe` (`2b235ae`) | **TASK_836** E4 value-ranking probe (`experiments/836_value_ranking/`) | Medium panel, **0 BLOCKER/MAJOR** (index alignment traced clean, canon decode, deviations valid); 3 MINORs + 3 NITs fixed with bit-identical-baseline proof (honest frozen-book label + `momentum_walk_pv` true mirror; overlap p-caveat; degenerate-value reading). |
| `task837-walkforward` (`2396c55`) | **TASK_837** walk-forward R4 rails (`experiments/837_walkforward/`) | Temporal-leak core CLEAN; **3 MAJOR gate-coverage roots crossval-CONFIRMED → fixed + re-verified live**: POLICY-PIN (validate now pins parsed windows/universe rules to pre-registered constants); eval-job file IDENTITY (leaky-universe swaps can't become the graded number); de-confounded negative battery (no-op control must validate CLEAN; 5 negatives each rejected by NAMED gates). `build_pit_universe` verified NON-PIT → module-side PIT (fold1 64-of-219, fold2 64-of-263). |
| `task838-pernode-resample` (`39ea3de`) | **TASK_838** per-node resampling (canon `sampled_search.py`, flag default OFF) | FULL HIGH TIER. OFF bit-identical: pre-edit pin `22b0c7f5…f23e` MATCH, **pin provenance independently reproduced from the pre-edit commit**; OFF branch byte-identical by construction; Stage-3 scalar pin still MATCH; 11 existing tests untouched. ON semantics proven by adversarial runtime tree-capture. 1 MAJOR fixed (835 eval threads the flag → no silent train/eval operator asymmetry). **Evidence-integrity episode**: implementer drafted fabricated RESULT numbers pre-run, self-caught, replaced; three judges independently bit-reproduced every shipped number; fabricated draft appears nowhere. |

## The scientific finding of the run
**838 acceptance (pre-registered read NOT held)**: constant-policy true-model probe, equal K=8/sims=32:
net@21 OFF +6.421 vs ON +3.634 (Δ **−2.787**, ON wins 0/3 seeds, ON below the +5.591 floor OFF clears).
Fresh random per-node portfolios inject variance into deep backups under an uninformed policy. The
informed-proposal mechanism the flag exists for is untestable without a trained factored policy —
**any `--vector-per-node-resample` ON run must re-measure first (scheduled Stage-3/4 stream). Default OFF.**

## Instrument coupling notes (for the scheduled runs)
- 834/835/836 all take the operator as CLI flags — pass the flags of the battery ACTUALLY LAUNCHED
  (phase-1 = cost0, hb0.0/d1.0, K16; NOT the old manifest defaults).
- 835 now records `operator.per_node_resampling` in its JSON (self-describing).
- 837 server pre-flight: `validate --db trading_data.db` (read-only) BEFORE any run_queue.sh launch —
  the minute-ingest assert only gains teeth against the real db.

## State of the line
- **Phase-1 cost0 battery** on gpu-server (launched 2026-07-08 ~23:24, ~6h/seed): should complete
  early morning 2026-07-09. Then: 834 (DEC-1), 835 (corridor), 836 (value-vs-momentum) on its
  snapshots; phase-2 fee16 continuation gated on those + the 832 imprint-sanity run.
- Queue Ready = EMPTY. All next work = planner (phase-1 verdicts → grooming) or scheduled runs.
- Instruments now complete for the whole rung ladder: E3 (834), Stage-4 corridor (835), E4 (836),
  E5 (832), R4 folds (837) + the depth-search flag (838) parked behind a trained-policy re-measure.

## Git state
```
39ea3de TASK_838: per-node action resampling in the sampled vector search, flag-gated default-OFF
2396c55 TASK_837: walk-forward split harness (R4 graduation instrument), zero canon diffs
2b235ae TASK_836: E4 value-ranking probe (value net vs momentum, per window), zero canon diffs
0d7108b Factory rearmed: TASK_836 E4 probe, TASK_837 walk-forward graduation harness, TASK_838 per-node resampling
```
Tags this run: `task836-e4-rank-probe`, `task835-…` (prior run), `task837-walkforward`, `task838-pernode-resample`.
Nothing pushed (owner pushes after review).

## Next action line
Planner: read the phase-1 battery verdicts when seeds complete; run 834/835/836 on the snapshots
(with phase-1 flags); groom the queue. Factory: idle until new rows.
