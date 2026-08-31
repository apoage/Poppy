# task741 result

> Mint per-asset post-pool injection (xs scalars from channels 27-31) trains BETTER (loss −0.8, Bal.Acc +3.29pp) but TRADES WORSE (mean ΔAlpha −6.98pp across 3 seeds, all 3 negative; bull/ftx_crash/etf_approval lose 8-11pp Alpha each). Mint xs is abandoned as a Stage-B improvement. Classic loss-vs-deployment divergence — same pattern as TASK_737 (heads can sharpen classifier and hurt deployment metric).

*Migrated from local agent memory 2026-08-31 (`project_task741_result.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**Verdict (2026-05-14 from paired 3-seed eval)**: Mint xs is a CLEAR
REGRESSION on Alpha. Mechanism is abandoned for production Stage-B.

## Quantitative

| Metric | Δ AVG (mint − base) | Per-seed Δ | Verdict |
|---|---|---|---|
| Bal.Acc | +3.29pp | +7.78, −0.25, +2.34 | Mint = better classifier |
| Alpha | **−6.98pp** | **−14.62, −1.22, −5.08** | Mint = worse trader (all 3 negative) |
| beats-random | 6/6 either way | — | model not broken |
| Training loss | ~0.8 lower every seed | — | Mint clearly fits the loss better |

Crosses TASK_741 SPEC's "clear regression" threshold: mean(Δα) < −2pp,
all 3 deltas negative.

Per-period: Mint loses 8-11pp Alpha in **bull regime, ftx_crash,
etf_approval** specifically (rally/trend regimes). Sideways and bear
less affected.

## Why this is interesting

Same failure mode as TASK_737: adding a head/mechanism that sharpens
the classifier *can hurt deployment metrics* if it makes the policy
over-confident in the wrong direction during trending regimes. The
training loss / Val Agree is NOT a reliable proxy for deployment Alpha
on this setup.

## Why per-asset xs hurts (hypothesis, not verified)

The xs scalars (channels 27-31: ranks, beta, idio-vol, xs momentum)
already exist in the encoder input. Mint injects them again via FiLM on
the post-pool surface, effectively giving them *double weight*. In
rally regimes where these scalars correlate strongly with returns, the
policy over-trades on weak per-asset signal — sharper directional bets
that turn out wrong in trending markets.

## How to apply

- **Do NOT promote Mint xs.** Default `--per-asset-injection none`
  stays canonical Stage-B.
- **Do NOT run the fallback diagnostic** (`--per-asset-injection both`,
  xs + cheat) as production work — it's research-only (cheat is
  forbidden in production checkpoints) and won't ship a trading model.
  If curiosity-driven, queue separately at low priority.
- **The 87% Val Agree ceiling target (TASK_733)** is now confirmed: even
  a mechanism that BREAKS the ceiling (Mint achieves it by giving more
  per-asset signal) does NOT translate to better Alpha. The bottleneck
  is **somewhere else than Val Agree** — likely the
  policy/value heads' aggressiveness scaling or the BC loss weighting,
  not the encoder representation.
- Trust **paired Δα over 3 seeds** as the success metric going forward.
  Don't promote on Bal.Acc or loss alone.

See `runs/741_mint/result.md` for full per-seed/per-period table.
See `lore/notes_2026-05-13.md` §106 for the full narrative.
