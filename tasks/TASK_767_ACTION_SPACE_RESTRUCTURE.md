# TASK_767 — Action Space Restructure (Drop Limit Orders)

**Status**: GEMINI-PROPOSED (operator review pending)
**Created**: 2026-05-21
**Estimated wall**: ~12h server (Phase 2 only)

## Hypothesis
The current 320-dim action space (64 assets × 5 actions: Sell, Hold, Buy, LimitSell, LimitBuy) dilutes policy capacity and expands the MCTS search branching factor exponentially. Limit orders rely on simulator assumptions that often fail to generalize. Reducing the action space to 192-dim (Sell, Hold, Buy) will condense the policy's probability mass, reduce over-trade noise, and improve value head accuracy by focusing solely on directional market orders.

## Code prerequisites
- `mcts_trader/env/real_trade_jax.py` (or equivalent): add `--disable-limit-orders` CLI flag to mask or remove actions 3 and 4 from the step function.
- `train_phase2.py`: dynamically set action space dimensions based on config, or simply mask the policy logits for actions 3 and 4 to `-inf` before softmax.
- Line estimate: ~40 lines across environment and Phase 2 config.

## Variants
| Cell | `--disable-limit-orders` | Seeds |
|---|---|---|
| baseline | False (5 actions) | s7, s13, s23 |
| 3-action | True (3 actions) | s7, s13, s23 |

## Promotion gate
Paired Δα ≥ +3pp under eval-MCTS sims=16 joint. 
*Mechanism-specific early-exit gate*: The 3-action variant must show a reduction in the raw number of executed trades compared to the baseline, confirming that the policy was previously bleeding capacity into limit order noise.

## Risk
- *Risk*: The oracle may currently rely heavily on limit orders for its `+265%` bull regime returns. 
- *Mitigation*: Run a quick offline check on the Phase 1 oracle labels to see what percentage of actions are Limit vs Market. If the oracle uses < 10% limit orders, this intervention is extremely safe.

## Why this comes before/after others
Position this **after TASK_762 (Reward Shaping)** but **before TASK_763 (Entropy Reg)**. 
Reward shaping is the cheapest and most direct fix for over-trading. If reward shaping fails, the action space restructure physically removes the model's ability to express complex/noisy trade intents, which might be a better fix than soft entropy regularization (763). It is entirely Phase 2-only, so it precedes the Phase 1-breaking TASK_764.

## References
- `project_classifier_alpha_anti_correlation.md` — over-trade collapse signature.
- `project_single_mechanism_exhausted.md` — justification for Phase 2 structural interventions.
