# bull blindness

> Bull failure caused by oracle class imbalance (93.75% HOLD from k=4/64), NOT encoder or search. Encoder preserves direction (cosine 0.38 bull/bear), but policy can't learn 6% minority BUY/SELL signal through standard CE loss.

*Migrated from local agent memory 2026-08-31 (`project_bull_blindness.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
Bull/rally regimes fail across ALL model variants — root cause identified via encoder audit (2026-04-17).

**Root cause:** Oracle selects k=4 of 64 assets → 93.75% HOLD by construction. BUY vs SELL swing between bull and bear is only ~1.2 percentage points. Standard cross-entropy fits the HOLD marginal and ignores the per-asset BUY/SELL conditional. `trade_loss_weight=60` isn't enough.

**What's NOT the problem:**
- Encoder: 10+ channels carry signed direction, latent cosine bull/bear = 0.38
- Search: factored search improved 3/6→4/6, but bull unchanged
- Oracle quality: oracle makes +265% in bull (it CAN trade bull correctly)

**Fix direction:** Rebalance BC loss — mask HOLD or per-class weight (w_hold=1/16, w_buy=w_sell=1.0). Or add auxiliary direction-prediction head for dense directional gradient. See `tasks/TASK_725_BULL_EXPERIMENTS.md`.

**How to apply:** Don't add encoder channels (already directional). Don't tune search params. Focus on the loss function's class weighting. The signal exists in the latent — the policy just needs stronger gradient on minority classes to learn the conditional.

**Update (2026-07-05, forensic pass):** confirmed the oracle is direction-SYMMETRIC (path_solver
ranks by |forward return|, sign → long/short; +265% in bull stands) — bull blindness is NOT label
direction asymmetry. The class-imbalance mechanism above is one facet of the deeper problem now
named in [oracle_design_intent](oracle_design_intent.md): cloning a foresight-optimal teacher at all. Where
differential (beat-market) reward was active, its per-regime math adds a second bull-hostile term
(short-in-bear pays ~2×move, long-in-bull pays ~0). Clean planner test = TASK_821.

**Update (2026-05-13):** TASK_737 4-cell single-seed ablation suggested rank-only / no_aux_solo had the LEAST bull blindness. **OVERTURNED 2026-05-14 by TASK_743 multi-seed validation** (5 seeds, AVG Alpha +3.61%, range -4.4 to +7.8% = 12.2pp spread). The "rank head reduces bull blindness, aux head worsens it" refinement was single-seed noise, not signal. **Bull blindness remains unchanged across all known Stage-B head configurations.** Loss-rebalancing remains the most likely lever; investigate next via dense_sign oracle (TASK_739) or Mint per-asset injection (TASK_741) — both must be multi-seeded (≥3) from the start to filter ~5pp noise. See `lore/notes_2026-05-13.md` §99.
