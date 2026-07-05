# TASK_689: Fresh Improvements & Implementation Strategy

**Date**: 2026-02-21
**Source**: Codebase Analysis & Research
**Context**: Enhancing "Agent State Channels" to break the 62% accuracy plateau.

---

## 1. Direct State Injection via Fast Simulation (vs Reconstruction)

**The Problem**: The original task proposes *reconstructing* the agent state (positions, PnL) from `oracle_selections` and `returns`.
- **Flaw**: This reconstruction is brittle. It assumes we can perfectly infer entry prices and position directions from a binary mask. It fails to account for transaction costs, cash limits, or margin calls that the real environment (`jax_trading_env.py`) handles.
- **Codebase Reality**: We already have a high-performance JAX environment (`mcts_trader/env/jax_trading_env.py`) that tracks this perfectly.

**The Improvement**:
Instead of approximating state, **run the actual JAX environment** using the oracle's action sequence to generate the exact state trajectory.

**Implementation**:
In `train_easy.py` `_generate_episode_data`:
1. Get `optimal_path` from the solver.
2. Convert path to discrete actions (already done via `optimal_path_to_policy`).
3. **NEW**: Run `jax_trading_env.batch_step` (or `scan`) with these actions to produce a sequence of `TradingState` objects.
4. Extract exact `pos_active`, `pos_entry`, `cash`, `cumulative_return` from this trajectory.
5. Pass these exact values to the encoder.

**Benefit**: Zero ambiguity. The model sees *exactly* what the portfolio would look like if it followed the oracle.

---

## 2. Missing Critical Channels: Cash & Global Capacity

**The Problem**: The current proposal adds per-asset channels (Position, PnL, Heat). It completely ignores **Global State**.
- The agent doesn't know if it has Cash. It might learn "Buy signal + Good PnL = Buy More", failing to realize it's bankrupt.
- It doesn't know global utilization (e.g., "I already hold 20 assets, I shouldn't buy a 21st").

**The Improvement**: Add **Global Scalar Channels** (broadcasted to all assets).

| Channel | Name | Definition | Logic |
|---------|------|------------|-------|
| 31 | `cash_utilization` | `1.0 - (cash / portfolio_value)` | "How invested am I?" (0=All Cash, 1=Fully Invested) |
| 32 | `capacity_usage` | `n_positions / max_positions` | "Can I open new slots?" |
| 33 | `margin_health` | `equity / maintenance_margin` | "Am I close to liquidation?" (Critical for shorting) |

**Implementation**:
- In `jax_encoder.py`, accept these scalars.
- Tile them to shape `(window_size, n_assets)` (or `(n_assets,)` for compressed).
- Concatenate with the per-asset channels.

---

## 3. Robustness via "State Corruption" (Denoising)

**The Problem**: "Teacher Forcing" bias.
- **Training**: Model sees *Perfect* Oracle State (e.g., "I am holding AAPL and up 5%").
- **Inference**: Model sees *Its Own* Imperfect State (e.g., "I am holding AAPL" (actually not), or "I am flat" (actually holding)).
- **Risk**: The model learns to trust the state channel blindly. If the state channel says "Held", it outputs "Sell/Hold". If it says "Flat", it outputs "Buy". At inference, if it makes one mistake, the state channel becomes a lie, and the model hallucinates.

**The Improvement**: **State Dropout / Corruption**.
During training, with probability $p$ (e.g., 20%), **corrupt** the state channels fed to the model.
- **Bit Flip**: Invert `pos_active` (tell the model it's holding when it's flat, and vice versa).
- **Noise**: Add Gaussian noise to `unrealized_pnl`.
- **Zeroing**: Zero out the entire state channel (forcing it to rely on price action).

**Why it works**:
This forces the model to treat state as a *feature*, not a *command*. It learns: "Usually if state says Hold, I Hold. But if price action looks terrible, I should check my conviction." It effectively trains a **Denoising Autoencoder** component within the policy.

---

## 4. Multi-Scale Heat (Decay Engineering)

**The Problem**: A single decay rate ($0.95$) is arbitrary.
- Fast strategies need quick cooldowns ($0.8$).
- Trend strategies need slow memory ($0.99$).

**The Improvement**: Use **3 Heat Channels** with fixed timescales.
- `heat_fast` (decay=0.80, $\approx 3$ steps)
- `heat_med`  (decay=0.95, $\approx 14$ steps)
- `heat_slow` (decay=0.99, $\approx 70$ steps)

**Benefit**: The model can attend to the timescale that matters for the specific asset/regime without us guessing the magic number.

---

## 5. Architectural Integration (V3 minGRU)

**The Problem**: How to feed 30+ channels into the V3 encoder without exploding param count?
- Current V3: `Dense(channels -> embed_dim)`.
- If channels grow from 27 -> 35+, the input projection grows slightly, but it's negligible.

**The Improvement**: **Channel Grouping**.
- Group 1: Market Data (OHLCV) -> `Embed_Market` (32 dim)
- Group 2: Portfolio State (Pos, PnL, Heat) -> `Embed_State` (16 dim)
- Concatenate: `[Embed_Market, Embed_State]` -> 48 dim -> minGRU.

**Benefit**: Explicitly separates "What the market is doing" from "What I am doing". Allows the model to process them hierarchically. (This is a "Nice to Have", simple concatenation is likely sufficient for now).

---

## Summary of Action Plan

1.  **Modify `jax_encoder.py`**:
    *   Update `EncoderConfig` to support `n_channels=34` (27 market + 4 asset state + 3 global).
    *   Update `encode_observation_with_state` to accept `TradingState` object directly.
    *   Implement scalar broadcasting for global channels.

2.  **Modify `train_easy.py`**:
    *   In `_generate_episode_data`, run `jax_trading_env.scan` using the oracle selections to get the ground-truth `TradingState` trajectory.
    *   Implement **State Corruption** (randomly flipping state bits before encoding).

3.  **Config**:
    *   Add `state_noise_prob = 0.2` to `EasyConfig`.
