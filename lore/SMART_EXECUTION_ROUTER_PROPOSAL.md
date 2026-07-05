# Proposal: Smart Execution Router

This document outlines the concept for a "Smart Execution Router" (or Trade Execution Optimizer), an intermediate layer designed to translate the agent's high-level strategic intent into efficient, robust, and atomic market orders.

## 1. Problem Statement

As the agent's capabilities grow to include multi-action sequences (via "Action Points"), a new problem emerges: **trade fragility**.

An agent's strategic goal, such as "rotate from Asset A to Asset B," might be expressed as a sequence of simple commands:
1.  `SELL Asset A for USD`
2.  `BUY Asset B with USD`

When using market orders, this is mostly reliable. However, when using **limit orders**, this sequence becomes fragile:
- The `LIMIT SELL A/USD` order might fill, but the `LIMIT BUY B/USD` order might not if the market for B moves away.
- **Result:** The portfolio is left in an unintended state (holding USD) instead of achieving the strategic goal (holding Asset B).

This exposes the agent to unwanted risk and diverges its actual portfolio from its intended state.

## 2. Proposed Solution: A Smart Execution Router

The solution is to introduce a new layer between the agent's decision-making (`Execution Layer`) and the portfolio accounting (`Environment Layer`).

### Architectural Placement

```
┌──────────────────────────────────┐
│         Execution Layer          │
│   (Agent decides on a sequence   │
│    of 4 abstract actions)        │
└──────────────────────────────────┘
                 │
                 ▼ (List of 4 desired actions)
┌──────────────────────────────────┐
│    **Smart Execution Router**    │  <-- NEW LAYER
│  (Optimizes the action sequence) │
└──────────────────────────────────┘
                 │
                 ▼ (Optimized list of VectorBT orders)
┌──────────────────────────────────┐
│        Environment Layer         │
│  (VectorBT processes the orders) │
└──────────────────────────────────┘
```

### Core Responsibilities

The router's job is to analyze the entire sequence of actions for a given timestep and apply heuristics to optimize them before execution.

#### Heuristic 1: Trade Path Optimization (Atomic Swaps)

This is the primary function of the router.

- **Input:** A sequence like `[SELL A/USD, BUY B/USD]`.
- **Logic:**
    1. The router identifies this as a potential `A -> USD -> B` rotation.
    2. It queries the market data to check if a direct `A/B` pair exists.
    3. If a direct pair exists, it cancels the two USD-based orders and synthesizes a single, atomic order on the `A/B` pair.
- **Output:** A single `SELL A for B` order.
- **Benefit:** This makes the rotation atomic. It either succeeds or fails as a single unit, preventing the portfolio from getting stuck in the intermediate USD state.

#### Heuristic 2: Order Consolidation

- **Input:** A sequence that might contain redundant actions, like `[BUY BTC/USD, ..., BUY BTC/USD]`.
- **Logic:** The router detects multiple orders for the same asset and consolidates them into a single, larger order.
- **Output:** One larger `BUY BTC/USD` order.
- **Benefit:** Reduces transaction fees and simplifies the final order list.

#### Heuristic 3 (Future Work): Liquidity-Aware Routing

- **Logic:** For a rotation like `A -> B`, the router could analyze multiple paths (`A -> USD -> B` vs. `A -> C -> B`) and choose the one with the best liquidity or lowest overall fees.

### Example Workflow

1.  **Agent's Intent:** The agent, using its 4 action points, decides on the following sequence for a single timestep:
    1.  `LIMIT SELL 1 BTC/USD` (at $50,100)
    2.  `HOLD`
    3.  `LIMIT BUY 15 ETH/USD` (at $3,340)
    4.  `HOLD`

2.  **Router Receives Actions:** The `step` function passes this list of 4 desired actions to the Smart Execution Router.

3.  **Router Analyzes:**
    - It sees a `SELL BTC` followed by a `BUY ETH`.
    - It recognizes this as a `BTC -> ETH` rotation.
    - It checks the market and finds a liquid `ETH/BTC` pair.

4.  **Router Optimizes:**
    - It calculates that `1 BTC` is approximately `15 ETH` at current prices (`50100 / 3340 ≈ 15`).
    - It cancels the two original limit orders.
    - It creates a single new order: `LIMIT BUY 15 ETH/BTC` at a price of `0.0666` (`3340 / 50100`).

5.  **Final Execution:** The router outputs a list containing just this one optimized order to the `vectorbt` portfolio handler.

## Implementation Plan

This feature should be implemented **after** the foundational multi-action and native `vectorbt` limit order mechanics are complete and validated. It is a powerful enhancement that builds upon that stable base.
