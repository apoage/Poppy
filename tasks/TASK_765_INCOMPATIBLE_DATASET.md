# TASK_765 — Incompatible-Dataset Pivot (Iris-DEX block-level OHLCV)

**Status**: STRATEGIC DRAFT — operator stated openness ("im already open to full retrain on incompatible dataset if it would give some good results"). Decision deferred pending TASK_762/763/764 outcomes, but data-prep groundwork can begin in parallel.
**Created**: 2026-05-20.
**Estimated wall**: ~2 weeks first attempt (data prep + Phase 1 + Phase 2).

## Hypothesis

Current stack uses daily OHLCV on a 64-asset CEX-listed universe. Iris-DEX (`~/Iris-DEX/`) produces per-block (12s) OHLCV from on-chain Reth archive node — **7200× the bar frequency**, different universe (DEX-listed tokens), different microstructure (continuous liquidity from AMM swaps rather than CEX matching).

Possible reasons for fundamental different performance:
1. Higher bar frequency → more samples per regime → tighter oracle labels.
2. DEX universe has different return distribution (more fat-tailed, more event-driven).
3. On-chain orderbook proxy (AMM curve) gives genuinely new signal vs CEX OHLCV.
4. Different inductive bias — model may need fewer planning horizons but more pattern-recognition.

**Falsifiable**: train Phase 1 + Phase 2 on Iris-DEX bars; if model produces positive Alpha on its own benchmark (no comparison to current campaign), the dataset swap is a viable product direction. If null/negative, the inductive bias didn't transfer and the bottleneck is architectural, not data.

## Why this isn't a peer experiment to 762/763/764

This is **not** a paired Δα test against the current campaign. The benchmark periods, oracle definition, universe, and Phase 1 imprint all change. It's a **standalone alpha-positive question**:

> Does the same architecture (Foxglove + Poppy) trained from scratch on different data yield a deployable trader on that data's universe?

If yes, we have a parallel product line (DEX trader) — possibly more aligned with the long-term apothecary thesis (per `prompt-lib/bella_context.md`). If no, the architecture is the bottleneck and we need to redesign.

## Phases

### Phase A — Data pipeline groundwork (~1 week wall)

Per `prompt-lib/defi_data_pipeline.md`:
1. Iris-DEX block-level OHLCV ingestion → parallel SQLite DB (`iris_dex_data.db` or similar).
2. Universe selection: top N DEX tokens by liquidity over rolling window. Hyperparameter; start N=32 to keep model size manageable.
3. Oracle re-derivation at new bar interval. Daily k=4/64 → block-level k=? Need to fix horizon H first (current daily H is ~5 days = 5 bars; block-level H equivalent = 5 × 7200 = 36000 bars, infeasible — pick a sensible new H, e.g. 1 hour = 300 bars).
4. Benchmark periods re-anchored. Current `config/benchmark_periods.json` is daily. Need block-level periods of equivalent calendar length.

**This phase can run in parallel with TASK_762/763/764** on workstation/Iris-DEX side. Server still works on the CLI experiments.

### Phase B — Phase 1 imprint on Iris-DEX (~3 days wall)

`train_easy.py --bella` workflow on new data. SegmentPool may need refactor for block-level cadence. Stages:
1. SegmentPool spot-check on new DB.
2. Phase 1 imprint with 3 seeds.
3. Sanity diagnostic: classifier Bal.Acc per regime (regimes must be redefined for block-level data).

### Phase C — Phase 2 a1b1 on new Phase 1 imprint (~3 days wall)

Stage-B with canon a1b1 hyperparameters on new Phase 1 ckpts. 3-seed paired against itself (no cross-dataset comparison meaningful).

### Phase D — Standalone benchmark + decision (~1 day wall)

Eval on new block-level benchmark periods. Decision criteria:
- Mean Alpha > 0 across new periods AND 4/6 (or equivalent) periods positive → **viable parallel product**.
- Mean Alpha ≤ 0 → architecture is the bottleneck; reconsider entire stack.

## Risk

Highest. Breaks every assumption:
- Code: SegmentPool, oracle, benchmark periods all need block-cadence rework.
- Data: depends on Iris-DEX pipeline maturity (per `prompt-lib/defi_data_pipeline.md`).
- Architectural: model may not have right inductive bias for high-frequency data (current attention windows assume slow regimes).
- Time: ~2 weeks per attempt vs ~1 day for CLI experiments.

**Mitigation**: treat Phase A as a sunk-cost investment in parallel — even if Phase B/C don't run, the Iris-DEX SQLite pipeline is reusable for Aconite (MEV) and other future projects per `prompt-lib/defi_data_pipeline.md`.

## Decision points

Operator authoritative on:
1. **Launch trigger**: now (parallel to 762/763/764) or after the CLI experiments close?
2. **Scope**: full Phase A→D pipeline or just Phase A data-prep, decide Phase B+ after CLI experiments close?
3. **Product framing**: is this a Poppy variant (same code, different data) or a new project (e.g. "Poppy-Lite" or rename)?

Operator stance per 2026-05-20 session: "open to full retrain on incompatible dataset if it would give some good results" — suggests willingness to launch Phase A at least, even before CLI experiments close.

## References

- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` Direction 4.
- `lore/notes_2026-05-20.md` §151.
- `~/Iris-DEX/` — block-level OHLCV pipeline (separate repo).
- `prompt-lib/defi_data_pipeline.md` — data prep groundwork.
- `prompt-lib/bella_context.md` — universe/thesis fit.
- `docs/MEV/` — Aconite docs (peer project that benefits from same data pipeline).
