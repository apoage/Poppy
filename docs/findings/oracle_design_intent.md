# oracle design intent

> Operator design intent (2026-07-05): the foresight oracle (path_solver) is a MEASUREMENT INSTRUMENT — upper bound on extractable future profit in the data — never meant to trade or teach. Its later repurposing as a cloning teacher for the trader is what 'brought a lot of problems'.

*Migrated from local agent memory 2026-08-31 (`project_oracle_design_intent.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
Operator clarification (2026-07-05, verbatim intent): "oracle's purpose was to measure maximum
possible future that is in data .. never trade itself .. later used as teacher for trader which
brought a lot a lot of problems."

Three oracle roles, only one of them sound:
1. **Instrument / upper bound** — valid as designed. The ~20k% / ~1024 net-profitable
   trades-per-episode figures are capacity-of-the-data measurements, not a strategy.
2. **Cloning teacher (BC / supervised imprint targets)** — the contamination source. A
   foresight-optimal policy's actions are largely unpredictable from the ex-ante information
   set; imitating it means fitting noise. This is the clean mechanism behind
   [classifier_alpha_anti_correlation](classifier_alpha_anti_correlation.md) (better teacher-imitation = worse alpha,
   3-way confirmed) and plausibly [bull_blindness](bull_blindness.md). The −85% blind-clone result and
   model recall ≈0.02 ([collapse_fixed_predictability_wall](collapse_fixed_predictability_wall.md)) are consistent.
3. **Eval yardstick for a planner trained on its own (absolute-PnL) reward** — fine.

**Provenance of the big oracle numbers (2026-07-05 audit):** there is NO clean "20k%" artifact —
it is a memory of a family of bug-inflated figures (the 50,000% oracle_α was a path-solver 4×
leverage bug: k=4 SUMMED not averaged; earlier 5000-7000% figures also artifacts). Honest oracle
measurements: 65-136% per 400-bar episode (k=4, cost=0) and +754%/ep net for the selective 42bp
zigzag diagnostic. Still enormous vs models — the instrument reading stands, just not at 20k%.

**Why it matters:** any future training design should use the oracle for roles 1/3 only, or
mask teacher labels to the ex-ante-predictable subset (TASK_775 oracle-mask / TASK_784
smoothed-oracle specs existed for exactly this — believed never run; see 2026-07-05 forensic
audit). "MCTS cannot trade" was concluded mostly under role-2 training regimes + differential
reward; see [verdict_contamination_audit](verdict_contamination_audit.md) if/when written.
