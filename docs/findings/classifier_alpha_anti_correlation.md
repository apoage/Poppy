# classifier alpha anti correlation

> Three independent experiments (737 no_aux_solo, 741 mint_xs, 738 legacy_27ch) confirmed: mechanisms that lower training loss + raise Bal.Acc on this Stage-B stack reliably collapse Alpha via trade-rate explosion. Bottleneck is downstream of the encoder. Stop trying classifier-sharpening interventions; look for planning-side fixes (like 734 flow_matching).

*Migrated from local agent memory 2026-08-31 (`project_classifier_alpha_anti_correlation.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**Pattern (3+ confirmations as of 2026-05-15)**:

Any mechanism that significantly lowers Stage-B training loss and/or raises Bal.Acc/dir_acc on this stack produces:
- trade rate explosion (often 10–100× more trades than baseline)
- Sharpe sign-flip (positive → strongly negative)
- per-period Δα negative in most regimes, especially trending (bull/ftx_crash/etf_approval)
- aggregate Δα strongly negative, all seeds

## Confirmed instances

| Run | Δ loss | Δ Bal.Acc | Δα | Trade rate (sample) |
|---|---:|---:|---:|---|
| 741 mint_xs | −0.81 | +3.29pp | **−6.98pp** | normal → +× |
| 738 legacy_27ch | −0.98 | +7.48pp | **−18.44pp** | 136 → 13,246 (bull s7) |
| 737 no_aux_solo (single-seed read) | mod | mod | +6.21 → +3.61 on 5-seed | within noise |

## Implication

The bottleneck is **NOT** the encoder representation, **NOT** the cross-sectional feature stack, **NOT** the per-asset signal density. It is somewhere in:
- value/policy head aggressiveness scaling
- BC loss weighting in Stage-B
- trade gating / softmax temperature at inference
- the planning-side use of the model (MCTS depth, simulations, etc.)

A "better classifier" makes the policy more confident, and on this setup more confidence → more trades → more losses.

## Counter-example (mechanism that works)

| Run | Δ loss | Δ Bal.Acc | Δα | Trade rate |
|---|---:|---:|---:|---|
| 734 flow_matching | +0.06 (flat) | +1.36pp (flat) | **+3.38pp** | toward selectivity (−76 to −79%) |

Flow-matching does NOT make the model "look smarter" by training metrics, but it improves Alpha because it touches **dynamics** (planning), not the encoder/classifier. See [flow_matching_promising](flow_matching_promising.md).

## How to apply

- **Stop proposing mechanisms framed as "make encoder/classifier better"** (cross-sectional features, denser oracles, per-asset signal injection, dir_acc ceiling breakers). Three failures is enough evidence to skip Tier-A for these.
- **Look at the planning side**: dynamics loss variants, value-target reformulation, MCTS search hyperparameters, policy entropy regularization, trade gating.
- **Trust paired Δα over 3+ seeds.** If Bal.Acc goes up but Δα is flat/negative, the mechanism is dangerous — it shows the "sharpening trap" signature.
- **Bull-blindness** (`project_bull_blindness.md`): the root cause is here, not in the encoder. Mechanisms that pre-empt bull-regime over-trading should be the priority over mechanisms that try to "detect bull better" — the model can already detect, it just over-acts.

Related: [task741_result](task741_result.md), [flow_matching_promising](flow_matching_promising.md), [bull_blindness](bull_blindness.md).
