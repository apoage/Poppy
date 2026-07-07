# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-07-07**.

## Ready — take from the top

| # | Task | Size | Note |
|---|------|------|------|
| 1 | `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 1) | M | vector env step `step_vector(state, action_vec)` in `mcts_trader/env/jax_trading_env.py`, flag-gated `--vector-action`, canon scalar path bit-identical; equivalence test = vector action == equivalent scalar-step sequence. Risk: high (touches the canon env). |
| 2 | `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 2) | L | Sampled-MuZero search over K sampled action vectors from the factored policy head (`muzero_model.py:256`). Needs Stage 1 banner. Risk: high. |
| 3 | `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 3) | L | training loop + factored policy targets, `--vector-action` threaded through train_jax_fast.py/config/self_play. Needs Stage 2 banner. Risk: high. |

## Filler — anytime, parallel-safe

- (none groomed yet — repo prune 2026-07-07 cleared the hygiene backlog)

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 4 eval) | Stages 1–3 shipped; real-data training/eval is a SCHEDULED server run, not a factory-loop job |
| `tasks/TASK_824_SERVER_HONEST_REMEASURE.md` | server job (gpu-server), scheduled outside the factory; DECISIONS **D1** |
| `tasks/TASK_825_CROSSASSET_TREND_CTA.md` | DECISIONS **D1** open |
| `tasks/TASK_826_ONCHAIN_SIGNAL_PROBE.md` | DECISIONS **D1** + **D4** open; hosted-RPC/API key missing |

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.
