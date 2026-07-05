# Proposal: Advanced Agent Actions

This document outlines two proposed features to enhance the agent's capabilities: an expanded action space to include limit orders and a multi-step execution model using "Action Points".

## 1. Expanded 5-Action Space for Limit Orders

This proposal expands the agent's action set to allow for more nuanced order types beyond simple market buys and sells.

### Current Action Space
- For each asset, the agent can choose one of 3 actions: `[Market Sell, Hold, Market Buy]`.
- Total action space: `3 * n_assets`.

### Proposed Action Space
- The action set would be expanded to 5 actions per asset: `[Market Sell, Market Buy, Hold, Limit Sell, Limit Buy]`.
- Total action space: `5 * n_assets`.

### Implementation Details
- **Dynamic Limit Price:** The price for a `Limit Sell` or `Limit Buy` order would not be a fixed offset. Instead, it would be dynamically calculated based on the model's own **certainty**.
  - A high-confidence prediction might result in a less aggressive limit price (or even a market order).
  - A low-confidence prediction could place a more opportunistic limit order further from the current price.
  - This could be derived from the policy or value head outputs of the model.

- **Fee Structure:** To properly incentivize the model, the environment should simulate maker-taker dynamics.
  - `Market` orders would incur standard (taker) fees.
  - `Limit` orders, by providing liquidity, could incur lower or even negative fees (maker rebates), reflecting real-world exchange fee structures.

## 2. "Action Points" for Multi-Action Steps

This proposal allows the agent to take multiple actions within a single environment time step, enabling more complex maneuvers like "asset jumping" (selling one asset and buying another simultaneously).

### Concept
Instead of a single, massive combinatorial action space, the agent is given a fixed number of "Action Points" (e.g., 4) per turn. It executes actions sequentially until it runs out of points, only after which the environment's clock advances.

### Execution Flow
1. At environment time `T`, the agent observes the market state.
2. The agent performs an MCTS search and selects `Action 1`.
3. The environment executes `Action 1` instantly. The portfolio state changes, but the market data remains at time `T`.
4. The agent observes the new state (updated portfolio, same market prices) and selects `Action 2`.
5. This loop repeats `N` times, for the number of Action Points.
6. After all actions are executed, the environment's clock advances to `T+1`.

### Advantages
- **Linear Complexity:** This design avoids a combinatorial explosion. The MCTS search space remains `5 * n_assets` at each decision point. The computational cost scales linearly with the number of action points (e.g., `4 * MCTS_compute_time`).
- **Path Dependency:** The agent can make decisions that are dependent on each other within the same turn. For example, selling an asset could free up cash that informs the size of a subsequent buy action.

## Open Questions & Research Directions

As you noted, the core planning capability of the MCTS search remains effective because the action space at any given decision point is not exploding. The most interesting question raised by this design is:

> How would a sequence of rapid, intra-step actions imprint on the model's latent space representation (`h`)?

The model would need to learn the immediate consequences of its actions on its own portfolio state, and how to best use its sequence of "action points" to position itself optimally for the *next* market time step. This is a fascinating research path that moves beyond simple single-step decisions.
