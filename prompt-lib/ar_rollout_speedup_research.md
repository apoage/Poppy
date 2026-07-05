# Research Prompt: Autoregressive Rollout Speed Optimization

## Role

You are an expert in JAX/XLA optimization, sequential computation in ML training pipelines, and time-series model architectures. You are given a training system where autoregressive (AR) state rollout is the dominant computational bottleneck, causing a 5.6x slowdown. Your job is to find the best approach to reduce this cost while preserving training signal quality.

## System Description

### Training Pipeline (Foxglove)

Supervised expert iteration: model learns to imitate an oracle on multi-asset trading decisions.

- **Model**: TRM (Tiny Recursive Model) + MuZero heads, ~50M params (large preset)
- **Input**: `(34, 64, 64)` = (channels, window_size, n_assets)
  - Channels 0-26: Market features (returns, volatility, RSI, etc.)
  - Channels 27-32: State channels (position, PnL, hold_duration, heat_fast, heat_slow, capacity)
  - Channel 33: Model confidence (softmax max from previous step)
- **Output**: 320 policy logits (5 actions × 64 assets) + value head
- **Oracle**: Greedy lookahead path solver (k=4 assets, 20-step lookahead)
- **Sequence length**: 336 timesteps per episode (400 total - 64 window)
- **Hardware**: RTX 3090 (24GB), single GPU

### The Bottleneck: AR State Rollout

State channels (27-33) describe the agent's current trading state: what positions it holds, accumulated PnL, how long positions have been held, trading heat, and confidence. At inference, these are built from the model's own actions — each timestep's output determines the next timestep's state input.

During training, the AR rollout simulates this:

```python
@jax.jit
def _build_model_state_trajectory(market_obs, returns, model_params, model_apply_fn, ...):
    """Sequential rollout: T=336 model forward passes via jax.lax.scan."""

    def scan_step(carry, idx):
        pos, pnl, steps, h_fast, h_slow, prev_conf = carry

        # Build 7 state channels from carry state
        state_ch = jnp.stack([pos, pnl, steps/ws, h_fast, h_slow, capacity])  # (6, n_assets)
        conf_tiled = prev_conf[None, None, :]

        # Concatenate market (27ch) + state (6ch) + confidence (1ch) = 34ch
        obs = jnp.concatenate([market_obs[idx], state_tiled, conf_tiled], axis=0)

        # Full model forward pass (stop_gradient — no backprop through rollout)
        output = jax.lax.stop_gradient(
            model_apply_fn({'params': model_params}, observation=obs[None], mode='initial_inference')
        )
        policy_logits = output['policy_logits'][0]  # (320,)

        # Extract actions, update state (position, PnL, heat, confidence)
        action_types = jnp.argmax(policy_logits.reshape(64, 5), axis=-1)
        # ... state update logic ...

        return new_carry, (state_ch, prev_conf)

    _, (state_trajectory, confidence_trajectory) = jax.lax.scan(
        scan_step, init_state, jnp.arange(T)
    )
    return state_trajectory, confidence_trajectory
```

**Key properties**:
- `stop_gradient` — no gradients flow through the rollout (pure data generation)
- Each step does a full model forward pass: `model_apply_fn({'params': ...}, obs)`
- 336 sequential forward passes per episode (cannot parallelize — step N needs step N-1)
- Called per-episode during buffer refill (20% of buffer refreshed per epoch)
- The rollout is JIT-compiled via `jax.lax.scan`

### Measured Performance Impact

| AR % | Epoch Time | Steps/s (effective) | Relative Cost |
|------|-----------|---------------------|---------------|
| 0% (oracle state) | ~90s | ~1500 | 1.0x |
| 50% | ~325s | ~410 | 3.6x |
| 88% | ~500s | ~270 | 5.6x |
| 89% (current) | ~509s | ~260 | 5.7x |

At 89% AR, 89% of episodes use model-state rollout. Each AR episode requires 336 sequential model forward passes.

### Episode Economics

- Buffer: 32 episodes, refreshed 20% per epoch (~6-7 episodes regenerated)
- Of those, 89% use AR rollout = ~5-6 AR episodes per epoch
- Each AR episode: 336 sequential forward passes of a 50M param model
- Total: ~1800 forward passes per epoch, all sequential within each episode
- Training step itself (400 batched gradient updates): ~90s
- AR rollout overhead: ~420s (83% of epoch time!)

### Why AR Is Necessary

Without AR, state channels come from the oracle (perfect hindsight). At inference, there is no oracle — the model must build state from its own predictions. Training without AR creates a train/test distribution mismatch:
- Model trains with perfect state → learns policy conditioned on perfect state
- At inference, state is built from model's own (imperfect) actions → distribution shift → turtle (0% trades)

The model MUST learn to operate with its own state estimates. AR rollout is the mechanism for this.

## What's Already Been Considered (And Rejected)

These were discussed but not pursued due to bug risk vs. marginal gains at the time. **Revisit them now with fresh eyes** — the speed cost has become the primary training bottleneck:

1. **Verify detached rollout** (`stop_gradient`) — confirmed present
2. **Chunked rollout** — run N steps with frozen state, update, repeat
3. **Sparse rollout** — recompute every M steps, interpolate between
4. **Cached delta** — only recompute where actions changed from last rollout
5. **Reduced precision rollout** — bfloat16 for rollout forward passes
6. **State prediction head** — learn to predict state without rollout

## Research Questions

### 1. Parallelization Within the Sequential Constraint

The 336 steps are sequential per episode, but:
- Could multiple episodes be rolled out in parallel (batch the scan)?
- Currently each episode is processed individually. Can we batch 6 AR episodes into one `jax.lax.scan` call with batch dimension?
- What's the XLA/GPU utilization during the scan? Is each forward pass too small to saturate the GPU?

### 2. Chunked Rollout (Most Promising?)

Instead of 336 sequential steps, chunk into blocks of K steps:
- Run K steps with frozen state → get K observations in parallel
- Update state using final action of chunk → continue
- This converts 336 sequential passes into 336/K sequential chunks of K parallel passes

```
Full AR:    [s1]->[s2]->[s3]->...->[s336]     = 336 sequential
Chunked K=8: [s1..s8]->[s9..s16]->...->[s329..s336] = 42 sequential chunks
```

Questions:
- What chunk size K maximizes throughput? (K=8? K=16? K=32?)
- How much state error accumulates from frozen-state chunks?
- Does the model learn to handle the chunk boundary artifacts?
- Can chunks be different sizes (larger chunks where state is stable, shorter around trades)?

### 3. State Interpolation / Sparse Rollout

Only compute model forward pass every M steps, interpolate state between:
- Position changes rarely (95% of actions are HOLD)
- PnL is a running sum of `position * returns` — can be computed without model
- Heat decays exponentially — known closed form
- Only position changes require model inference

Could we:
- Run full forward pass every M=4-8 steps
- Between forward passes, assume HOLD (position unchanged) and compute PnL/heat analytically
- Only invoke model when state has "expired" or when previous action was non-HOLD?

### 4. Distilled State Predictor

Train a tiny auxiliary network (MLP or linear) to predict state channels at step T given:
- Market observations at step T
- State at step T-K (a few steps back)
- Compression: replace 336 sequential 50M-param forward passes with 336 sequential tiny-MLP passes

Could this be a small RNN that learns the state transition dynamics directly?

### 5. Batched Episode Rollout

Currently rollout is per-episode. Could we:
- Stack 6 episodes on batch dimension
- Run one `jax.lax.scan` with batch_size=6
- XLA might pipeline the forward passes better with larger batch

The model already handles batch dimensions. The scan body would do batched inference.

### 6. Model Distillation for Rollout

Use the full 50M model for training, but a distilled ~5M model for rollout:
- Periodically distill current model → small model
- Use small model for AR state generation (6x fewer params = faster forward pass)
- State quality degrades but: does it matter? State just needs to be "model-like", not perfect

### 7. Asynchronous / Pipelined Rollout

Decouple rollout from training:
- Background thread: continuously rolls out episodes with current params
- Training thread: trains on whatever episodes are in the buffer
- Rollout never blocks training — slight staleness in params is acceptable

Is this already partially happening? The buffer refill is 20% per epoch — could rollout happen fully asynchronously between training steps?

### 8. Reduced Forward Pass for Rollout

The model has multiple heads (policy, value, dynamics, oracle_mask). For rollout, we only need policy logits. Could we:
- Create a rollout-only forward pass that skips value/dynamics/mask heads?
- The TRM's recursive refinement (H=6, L=8 cycles) is most of the compute — can we use fewer cycles for rollout (H=2, L=4)?
- Would a "fast path" inference mode with fewer refinement steps give good enough policy for state generation?

### 9. XLA Compilation Analysis

Is the `jax.lax.scan` efficiently compiled? Potential issues:
- Dynamic shapes inside scan body?
- Unnecessary recompilation per episode?
- Could `jax.checkpoint` reduce memory during scan?
- Is the scan body doing redundant work (re-JITting the model per step)?

### 10. Hybrid Approaches

Combine multiple strategies:
- Batched episodes (5) + chunked rollout (K=8) + reduced refinement (H=2)
- Expected speedup: 5x (batch) × 8x (chunk) × 3x (refinement) = 120x theoretical
- Even with 50% overhead: 60x realistic → 509s becomes ~8.5s
- What's the realistic combined speedup? What combination minimizes quality loss?

## Constraints

- **Hardware**: Single RTX 3090, 24GB VRAM. Training uses ~22GB. Limited headroom.
- **Correctness**: State must be "model-like" — doesn't need to perfectly match full sequential rollout, but should be close enough that the model learns self-consistent behavior.
- **Integration risk**: Changes to the rollout function are high-risk. Subtle bugs in state channels can silently degrade training for hundreds of epochs before detection.
- **No architecture changes to the model itself** — the 50M param TRM+MuZero model is fixed. Only the rollout/inference path can be optimized.
- **JAX/Flax ecosystem** — must use JAX primitives. No PyTorch, no custom CUDA kernels.
- **stop_gradient is mandatory** — no gradient flow through rollout.

## Context Files

Read these files for implementation details:
- `train_easy.py:176-310` — `_build_model_state_trajectory()` (the bottleneck function)
- `train_easy.py:555-610` — AR blend logic and episode generation
- `train_easy.py:317-420` — `OnTheFlyGenerator.__init__()` (buffer management)
- `train_easy.py:1100-1140` — `set_model_params()` (param update for rollout)
- `easy_trainer.py` — training loop (to understand what runs between rollouts)
- `mcts_trader/jax_agent/muzero_model.py` — model architecture (forward pass)
- `mcts_trader/jax_agent/trm_flax.py` — TRM reasoning (H×L recursive refinement)

## Output Format

Provide a **ranked list of approaches** (best first). For each approach:

1. **Strategy name**
2. **Mechanism**: How it reduces AR cost (with pseudocode if useful)
3. **Expected speedup**: Conservative estimate (Nx) with reasoning
4. **Quality impact**: How much does state quality degrade? Scale: None / Minimal / Moderate / Significant
5. **Implementation complexity**: Lines of code, number of files touched, risk of subtle bugs
6. **VRAM impact**: Additional memory needed (if any)
7. **Composability**: Can this be combined with other approaches?

Then provide a **recommended implementation plan**: which approach(es) to implement first, in what order, with what validation steps.

Save your analysis to `prompt-lib/out-ar-01/ar_speedup_analysis.md`.

## Bonus: The Nuclear Option

If none of the above work well enough, consider fundamentally different approaches:
- **Remove state channels entirely** — can the model learn without explicit state? (Use memory / recurrence instead)
- **Learned state embedding** — train an encoder that maps `(market_obs_t, action_history)` → state directly, no sequential rollout needed
- **Offline rollout** — pre-roll all episodes once per merge, reuse stale rollout for N epochs before re-rolling
- **Teacher-student state** — use oracle state but with random noise injected to simulate model-state imperfections

Evaluate these radical alternatives with the same format.
