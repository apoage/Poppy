gemini
# Ceiling Analysis 002: The Bandwidth Crisis 

**Date**: 2026-02-20
**Reference**: Builds upon `ceiling_analysis_001.md`
**Focus**: Architectural constraints in State Representation vs. Compute Capacity

---

## 1. The "1 Float Per Asset" Absurdity

The most critical finding from the code review is the information density of `state_dim=64`.
- **System**: 64 Independent Assets.
- **State**: 64 Dimensions.
- **Ratio**: **1.0 dimension per asset.**

In a continuous state space, 1 dimension *can* represent a scalar (e.g., price), but the latent state must encode:
1. Current Trend (Up/Down)
2. Volatility Regime
3. Momentum / Reversion probability
4. Cross-asset correlations (Sector coupling)
5. Oracle Lookahead hints (imprinted via Value/Policy targets)

Compressing this 5+ dimensional feature vector per asset into **1 dimension** is mathematically impossible without catastrophic information loss. The model is likely learning a "global market factor" (beta) and applying it to all assets, failing to capture idiosyncratic alpha (specific trade setups).

**This explains why:**
- `agree_hold` is high (78%): "Don't trade" is a safe global default.
- `agree_trade` is low (43%): Specific trade signals (Buy Asset 5 vs Asset 7) are lost in the compression noise.

---

## 2. The "Coupling" Trap

The current TRM implementation in `mcts_trader/jax_agent/trm_flax.py` contains a dangerous coupling:

```python
# trm_flax.py
hidden_dim = self.state_dim * 16  # Hardcoded scaling
```

This design decision links **Bandwidth** (state_dim) with **Reasoning Capacity** (hidden_dim/params).
- To fix the bandwidth (e.g., `state_dim=512` for 8 dims/asset), we would explode the model size:
  - `hidden_dim` = 512 * 16 = 8192
  - MLP Params ~ $8192^2$ = 67 Million params *per layer*.
  - Total model > 1 Billion params. **Impossible on 24GB VRAM.**

**Consequence**: We are artificially capped at `state_dim=64` (Small) or `state_dim=128` (Large) merely to keep parameter count in check, starving the model of representation bandwidth.

---

## 3. The "Global Soup" Bottleneck

The V3 Encoder (`trm_encoder_v3.py`) exacerbates this:
1. `minGRU` produces `(Batch, 64 assets, 32 dims)`.
2. `_post_recurrence` flattens this to `2048` dims.
3. `global_proj` smashes `2048` down to `64`.

The TRM then receives this 64-dim "soup". The Policy Head (`output_refiner`) is expected to extract 320 distinct logits (5 per asset) from this soup. It must effectively "invert" the projection, which is impossible if the projection was lossy (which 32:1 compression definitely is).

---

## Strategic Recommendation: "Wide Bus, Narrow Logic"

We must decouple **State Bandwidth** from **Compute Capacity**.

### Implementation Strategy

1. **Decouple Dimensions**:
   Modify `TRMReasoning` to accept an explicit `hidden_dim` argument, rather than calculating it as `state_dim * 16`.

2. **Expand State, Constrain Compute**:
   - `state_dim`: **256** or **512** (4-8 dims per asset).
   - `hidden_dim`: **1024** or **2048** (Keep parameter count similar to current 15M-30M range).

   **Configuration Example (Target)**:
   - `state_dim = 256` (4x bandwidth increase)
   - `hidden_dim = 1024` (Reasonable compute)
   - `H=6, L=8` (Deep reasoning)

   **Parameter Impact**:
   - Encoder: `2048 -> 256` proj (vs 64). Negligible increase.
   - TRM Layers: `(256 + 1024) -> 1024` matrices. roughly same size as current `(64 + 1024) -> 1024`.
   - **Net Result**: drastically higher bandwidth with *same* VRAM/Compute footprint.

### Action Plan

1.  **Refactor `trm_flax.py`**:
    - Add `hidden_dim` parameter to `TRMReasoning` and `TinyRecursiveTradingModel`.
    - Default to `None` (preserve old behavior `state_dim*16` for backward compat if needed, or just break it for V2).
2.  **Refactor `easy_config.py`**:
    - Add `hidden_dim` to config (it's already there! just ignored by TRM internal logic).
    - Ensure `MuZeroNetworks` passes it down.
3.  **Experiment**:
    - Run `state_dim=256`, `hidden_dim=1024`.
    - Compare against `state_dim=64`, `hidden_dim=1024`.

---

## Comparison with Analysis 001

| Feature | Analysis 001 | Analysis 002 (This) |
| :--- | :--- | :--- |
| **Root Cause** | Encoder Bottleneck | **Coupling of Bandwidth & Compute** |
| **Primary Fix** | Widen Encoder (128) | **Decouple dims + Widen State (256+)** |
| **Secondary Fix** | Focal Loss | **Hard Separation of Assets (Architecture)** |
| **Complexity** | Low (Config change) | **Medium (Code refactor)** |
| **Why Better?** | 128 is still only 2 dims/asset. We need more. | Allows 256/512 state without VRAM explosion. |

## Conclusion

The "Ceiling" is a direct result of the `hidden_dim = state_dim * 16` line. It prevents the model from having a "Wide Bus" (State) without also having a "Wide Brain" (Compute). We need a **Wide Bus, Efficient Brain**.

**Recommendation**: Prioritize the Decoupling Refactor (Strategy 6) over simple config tweaks.
