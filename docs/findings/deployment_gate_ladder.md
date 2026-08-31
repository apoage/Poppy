# deployment gate ladder

> User-stated gating ladder from experiment success to live trading — four sequential gates, no shortcuts

*Migrated from local agent memory 2026-08-31 (`project_deployment_gate_ladder.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
Deployment gating ladder (user-stated 2026-06-10). A promoted experiment (e.g. a green
`tasks/` TASK_804) does NOT go live. The path is strictly sequential:

1. **Positive trading on historical data, ALL regimes** — not just the canonical benchmark
   periods; all 17 SegmentPool regimes must show positive performance.
2. **New (unseen) historical data** — out-of-sample periods never used in training or
   benchmark selection.
3. **Paper trading** — live data feed, simulated execution (`mcts_trader/exchange/` paper
   simulator). This is the gate where the streaming/incremental token annotator becomes a
   hard prerequisite (live path consumes one bar at a time; windowed `annotate_ohlcv` is
   not enough). Spec refinement 2026-06-11: it must be a **sealed-flag** annotator (emit
   per-bar "causally sealed" status — a token deep enough behind the merge frontier cannot
   be consumed by future merges). This supersedes the interim fixed-lag-3 input convention
   with zero-skew-by-construction, lag-adaptive serving.
4. **Live trading** (`real_trade_jax.py`).

**Why:** prevents the benchmark→live operator-mismatch class ([benchmark_live_gap](benchmark_live_gap.md))
from recurring at deployment scale.

**How to apply:** when any experiment promotes, the next step is gate 1 (all-regime
historical), never deployment. Schedule the incremental annotator build before gate 3,
not before gate 4.
