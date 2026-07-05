---
status: accepted
date: 2026-04-23
deciders: Lukas
---

# 0006. Stage B requires `--value-weight ≥ 0.5` — value training is mandatory for ExIt

## Status

Accepted.

## Context

For six consecutive Stage-B runs (728c, 729c, 730, 732, 734, 735) the agent's `policy_loss` was pinned at `1.61 ≈ log(5)` — exactly the cross-entropy of a uniform 5-way per-asset distribution. The auxiliary heads (direction, rank, aux_price) trained successfully and dropped their losses. dir_acc still stalled at 0.57. No pivot moved it.

Diagnosis (`lore/notes_2026-04-22.md §49`): `--value-weight 0.0` had been set on every Stage B run since 729c (originally as an ablation to isolate policy/direction). Over the pivot cycle the setting became load-bearing without anyone noticing. The closed-loop failure mode:

1. Value head untrained → value predictions diverge from the drifting (encoder-fine-tuned) latent.
2. MCTS backs up garbage values → garbage Q-values → near-uniform visit distributions.
3. Policy target (MCTS visits) ≈ uniform → cross-entropy stays at `log(5)` → policy gradient ≈ 0.
4. Aux heads train the latent fine, but their gradient does not flow into the policy head's *target*. Latent improves; policy never moves.

We were not running ExIt. We were running supervised BC + aux heads with a constant uniform MCTS target.

The fix was a single flag: `--value-weight 0.5` (TASK_736). The result (`lore/notes_2026-04-22.md §52`):
- `policy_loss`: 1.612 → 1.572 (broke the uniform trap by E49).
- `value_loss`: ~50 → 3.56 (14× drop).
- `consistency_loss`: 8.8 → 3.70 (2.4× drop).
- Benchmark: Alpha **+3.87 % AVG**, **6/6 beats-random** — first qualitative inflection in 700 tasks. Alpha bear regime +21.72 %, ftx_crash +17.70 %.
- Action distribution moved from "trade 60 % random direction" to using the full 5-action space including limit orders (10 % LSELL, 9 % LBUY).

## Decision

`--value-weight ≥ 0.5` is **non-negotiable** for any Stage-B run. TASK_736 is the new baseline. All prior Stage-B benchmarks (728c through 735) are compromised experiments — their hypotheses about heads / encoder / oracle say nothing meaningful because ExIt was disabled.

Recommended starting weight: **0.5**. Higher (1.0) risks value dominating total loss 5–10× given the initial value_loss magnitude; lower (0.25) is acceptable as a fallback if value explodes early. `0.0` is forbidden in production runs.

## Consequences

Gain:
- ExIt actually closes — MCTS produces meaningful targets, policy learns, the full 5-action space is used.
- A new known-good baseline (TASK_736) for warmstarting future Stage B work.
- Bal.Acc is decoupled from trading skill: TASK_736's Bal.Acc *dropped* (34 % → 26.5 %) while Alpha turned positive (`lore/notes_2026-04-22.md §52`). High Bal.Acc on prior runs reflected HOLD-dominance, not skill.

Give up:
- Six prior Stage B runs need re-evaluation if their hypotheses still matter — some were re-tested in TASK_737, others remain open questions.
- Initial 5–10 epochs need a NaN watch on value_loss (PopArt stats from Phase 1 may not match Stage B's initial state).

## Alternatives considered

- **Keep value_weight=0.0 and force policy with stronger BC.** That was effectively what 727/728/728b/728c had been doing — produced "near-uniform with symmetric noise" or "always-HOLD" policies. Rejected by data.
- **Drop BC entirely (`--bc-weight-min 0.0`) instead of fixing value.** Tried later as B1 of TASK_733's A1+B1 stacked test (`lore/notes_2026-05-05.md §69`) — within noise of TASK_736 reference. Did not supersede TASK_736.

## References

- `lore/notes_2026-04-22.md §49` — full diagnosis (the "ExIt loop closed on itself" analysis).
- `lore/notes_2026-04-22.md §50–§52` — TASK_736 launch, mid-run, completion + benchmark.
- `lore/notes_2026-04-24.md §60` — TASK_737 ablation reaffirms the value-weight fix is necessary; "minimal" config (value_weight=0.5 + direction head only) was best of the suite at +7.06 % Alpha.
- `tasks/TASK_736_VALUE_WEIGHT_FIX.md`.
