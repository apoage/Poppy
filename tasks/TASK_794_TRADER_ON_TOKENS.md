# TASK_794 — Stage ③: Trader-on-tokens integration + efficiency payoff

**Status**: BLOCKED on TASK_793
**Created**: 2026-06-02 · **Parent**: TASK_789 · **Depends on**: TASK_791/792/793

## Goal

Feed the **existing** minute-candle trader/MCTS stack (Poppy / `train_jax_fast.py` / the Stage-A/B
pipeline) **tokens + token embeddings** instead of raw candles. Cross-asset mixing happens **here**,
as it does today — this is where the 64 assets are decided jointly. The token layer is a front-end
swap, not a new trader.

## The two payoffs to prove

1. **Quality** — does a trader on tokens **match or beat** the candle baseline (760 a1b1) on the
   canonical scorecard (eval-MCTS sims=16 joint, paired vs `runs/760_train_mcts_sims/...`, gate +3pp /
   no seed < −5pp)? This is the only stage with a trading metric.
2. **Efficiency / filtering** — the trader works on a **compressed token stream** and **skips candle
   runs that aren't viable tokens** (noise dropped). Measure: throughput (steps/sec, tokens/sec vs
   candles/sec), data volume processed, and that dropping non-viable runs does **not** cost alpha. This
   is the operator's "faster, because we ignore candles we don't see as viable tokens."

## Integration notes

- Reuses: trading env + reward, `benchmark_easy.py` eval, oracle for BC/imprint, MCTS search loop.
  Only the front-end (candles → tokens+embeddings) changes.
- **Composes with B (TASK_788):** B's revived ExIt loop is the natural host; the token front-end drops
  in there once B proves the loop trains stably.
- Cross-asset attention / cross-sectional channels (TASK_738, load-bearing −18pp if removed) operate on
  token embeddings at the decision head — i.e. the existing cross-asset machinery, fed tokens.

## Acceptance

- Paired alpha ≥ candle baseline within the gate, AND a measured throughput/efficiency win with no
  alpha loss from noise-filtering. A clean tie on alpha **with** the efficiency win is still a win
  (cheaper training, denoised input, a reusable language).

## Multi-view = multilingual (operator 2026-06-02)

The Stage-1 panel yields several token languages; per the multilingual decision (TASK_793), the trader
consumes them through the **one shared multilingual embedding**, not N parallel stovepipes. The trader
sees a unified, cross-language-aligned representation of each bar-position. This keeps the
diversity-of-views benefit without N× the model.

## Open forks

- **multi-view fusion** at the decision head — single fused embedding (default) vs per-language heads
  vs cross-view attention over languages. (Deferred until the panel's survivors are known from 792.)
- granularity of token→trader interface (token ids + embeddings vs embeddings only),
- whether non-viable runs are dropped entirely vs mapped to a null/"flat" token the trader sees.
