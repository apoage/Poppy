# benchmark gap

> Training shows 75% trade agreement but benchmark shows 12-33% — root cause is equal-weight imitation, not pipeline bug

*Migrated from local agent memory 2026-08-31 (`project_benchmark_gap.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
Foxglove training achieves 85% balanced accuracy (75% trade, 96% hold) but benchmark shows 48% (12-33% trade, 94% hold).

**Why:** The benchmark pipeline is correct (verified 2026-03-20). The gap is caused by:
1. Pure imitation treats all trades equally — a +3% trade and a +0.01% trade get same gradient
2. The model learns "this is a trade situation" but not "WHICH specific asset at THIS moment"
3. Cross-asset selection (k=4 out of 64) is the hard problem — requires subtle relative-strength signals

**Why:** Benchmark value MSE (~291) is a metric bug, NOT a model problem. Training predicts remaining PnL over 400-step windows (range 0-5). Benchmark measures over 10,000-step trajectories (range 0-587). Different scales.

**How to apply:** Don't treat benchmark MSE as meaningful until it's rescaled. Focus on agree_trade per period as the real quality metric. TASK_713 (return-weighted loss) addresses the root cause.
