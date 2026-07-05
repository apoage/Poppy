# TASK_699: AR Rollout Speedup

**Status**: RESEARCH
**Priority**: 1 (primary training bottleneck)
**Depends on**: None (can implement while training runs)
**Blocks**: Training speed, total epoch budget

---

## Problem

AR rollout is 83% of epoch time. 336 sequential model forward passes per episode via `jax.lax.scan`.

| AR % | Epoch Time | Steps/s | Relative Cost |
|------|-----------|---------|---------------|
| 0% | ~90s | ~1500 | 1.0x |
| 89% | ~509s | ~260 | 5.7x |

Target: reduce 509s → sub-100s while preserving state quality.

---

## Research Results

### Kimi K2

**Source**: Deep research via `prompt-lib/ar_rollout_speedup_research.md`

#### Ranked Approaches

| # | Strategy | Expected Speedup | Quality Impact | Complexity | Composable |
|---|----------|-----------------|----------------|------------|------------|
| 1 | **Batched episode rollout** | 4-6x | None (identical) | Low (~50 LOC) | Yes |
| 2 | **Chunked rollout (K=8)** | 4-6x realistic | Minimal-Moderate | Medium (~150 LOC) | Yes |
| 3 | **bfloat16 rollout** | 1.5-2x | Minimal | Low (~30 LOC) | Yes |
| 4 | **Sparse rollout (M=8)** | 6-8x | Minimal | Medium (~100 LOC) | Yes |
| 5 | **Distilled student model** | 3-5x | Moderate | High (~300 LOC) | Yes |
| 6 | **Async rollout pipeline** | 1.2-1.5x | Minimal (staleness) | High (~200 LOC) | Orthogonal |
| 7 | **Reduced TRM refinement** | 2-3x | Moderate | Low (~20 LOC) | Yes |

#### Recommended Combo (Kimi)

Batched (6x) + bfloat16 (1.5x) + Chunked K=8 (4-6x) = **15-20x theoretical → 509s → ~25-35s**

#### Key Insight: Batched Episodes is Free Lunch

Currently each AR episode runs separately. Stacking 6 episodes into batch dim is mathematically identical, just better GPU utilization. Single 50M model forward pass on `(1, 34, 64, 64)` likely underutilizes RTX 3090.

#### Nuclear Options (Radical)

- Remove state channels entirely → recurrent policy with internal state
- Offline rollout → stale states reused for N epochs
- Learned state predictor → MLP predicts state from recent market obs
- Noisy oracle state → inject noise to simulate model-state imperfections

### Claude (Opus via deep research)

**Source**: Deep research via `prompt-lib/ar_rollout_speedup_research.md`

#### Core Thesis: Don't Optimize the Rollout — Eliminate It

**"The bottleneck is not JAX/XLA inefficiency but an unnecessary coupling between state computation and model inference."**

4 of 5 state channels (position, PnL, hold_duration, heat) are **deterministic functions of actions + market data** — pure accounting, no neural network needed. Only confidence requires model output. Under teacher forcing (using oracle expert actions), the 336-step sequential NN rollout collapses to `jnp.cumsum` — a **100-1000x speedup** on rollout.

#### The Analytical State Argument

```python
# ~15 lines replace 336 sequential 50M-param forward passes:
positions = jnp.cumsum(action_to_position_delta(expert_actions))
pnl = jnp.cumsum(positions * market_returns)
hold_dur, _ = jax.lax.scan(lambda h, a: ((h+1) * (a == HOLD), None), 0, actions)
heat, _ = jax.lax.scan(lambda h, x: (h * 0.95 + x, None), 0.0, heat_inputs)
```

**Risk**: Exposure bias — model trains on oracle-action-derived state but at inference uses its own actions. Mitigations from literature:
- **Scheduled sampling** (Bengio 2015): Start 100% oracle state, linearly increase model-state fraction to 20-30%
- **Professor forcing**: Adversarial alignment of state distributions
- Foxglove is supervised expert iteration → exposure bias is manageable

#### Ranked Approaches (If Rollout Kept)

| Tier | Strategy | Speedup | Quality | Notes |
|------|----------|---------|---------|-------|
| 1 | **Batched scan + bf16 + scan tuning** | 3-5x combined | None | B=6 vmap, bf16 params (fp32 state!), unroll=4, donate_argnums |
| 2 | **Reduced refinement (L=4)** | 1.5-2x additional | Minimal | CALM paper: argmax stabilizes at 1/3-1/2 of layers. >93% argmax agreement |
| 3 | **Chunked rollout (K=8)** | 2-3x additional | Moderate | Speculative decoding analogy. 75% HOLD → most chunks error-free |

#### Key Differences from Kimi/Gemini

- **Reframes the entire problem**: Other two optimize the rollout, Claude says eliminate it via analytical state + teacher forcing. This is the "nuclear option" that Kimi/Gemini rejected, but Claude argues it's actually standard practice in supervised seq2seq
- **More conservative on reduced refinement**: L=4 not L=2 (1.5-2x not 6-10x). Cites CALM paper for argmax stability at 50% depth, not 25%
- **Dismisses async on single GPU**: "JAX dispatches all GPU kernels on single CUDA stream with no user-accessible multi-stream API. Both rollout and training require GPU compute — they cannot overlap." Direct contradiction of Gemini's #3 recommendation
- **Warns on bf16 state accumulation**: "bf16 machine epsilon ~7.8e-3 causes catastrophic drift over 336 cumulative additions (worst-case 260% relative error)" — state MUST stay fp32
- **Dismisses distilled draft model**: Memory-bandwidth bound at B=1 limits actual speedup to ~2x regardless of param count. Reduced refinement achieves same with zero infrastructure
- **Warns on sparse rollout**: `jnp.where`/`lax.select` always executes both branches — no actual compute skipping. `jax.lax.cond` in scan is slower (JAX Issue #5986). Agrees with Gemini's rejection

#### Composability Matrix

| Combination | Realistic Speedup | Risk |
|-------------|------------------|------|
| **Analytical states + bf16 + batched training** | **10-30x** | Low (exposure bias mitigable) |
| Cached rollout (refresh every 5 epochs) + bf16 + batched | 8-20x amortized | Very low |
| **Batched scan + bf16 + L=4 + scan tuning** | **4-8x** | Minimal |
| All sequential optimizations maxed | 5-12x | Moderate (compounding approximations) |

#### Recommended Plan (Claude)

**Phase 1** (1-2 days): bf16 rollout + scan unroll=4 + donate_argnums + batch 6 episodes via vmap → **3-5x**
**Phase 2** (2-3 days): Implement analytical state from expert actions. A/B test: oracle state vs model-conditioned for 5 epochs. If gap <2%, adopt permanently → **10-30x**. If not, scheduled sampling (95% oracle → 25% model over training)
**Phase 3** (if needed): Reduce refinement to L=4 for rollout + chunked K=8 → additional 2-3x

### Gemini 2.5 Pro

**Source**: Deep research via `prompt-lib/ar_rollout_speedup_research.md`

#### Core Diagnosis

The bottleneck is **latency-bound, not compute-bound**. At B=1, each 50M param forward pass doesn't saturate the RTX 3090's SMs. The GPU idles between kernel dispatches waiting for sequential state dependencies. XLA compiles the scan body once as a while-loop — 336 sequential kernel launches where dispatch overhead exceeds compute time.

#### Ranked Approaches

| # | Strategy | Expected Speedup | Quality Impact | Complexity | Composable |
|---|----------|-----------------|----------------|------------|------------|
| 1 | **Batched episode vectorization** | 4.5-5.5x | None (bit-identical) | Low | Excellent |
| 2 | **Reduced TRM refinement (H=2,L=2)** | 6-10x | Minimal (noise = regularization) | Low | Excellent |
| 3 | **Async background rollout** | ~5.7x (latency hiding) | Minimal (staleness) | Moderate | Excellent |
| 4 | **Chunked AR (K=8)** | 3-4x | Moderate (temporal blindness) | High | Moderate |
| 5 | **Sparse rollout (conditional)** | 2-4x | Moderate | Very High | **Extremely poor** |
| 6 | **Head pruning (policy-only)** | 1.1-1.3x | None | Low | High |

#### Key Differences from Kimi

- **Ranks reduced refinement much higher (#2 vs Kimi's #7)** — argues H=2,L=2 is 12x FLOP reduction and noisier states actually help as regularization
- **Warns strongly against sparse rollout** — `jax.lax.cond` in batched context forces full branch if ANY episode triggers, destroying sparsity gains. "Extremely poor composability"
- **Warns against chunked rollout** — "temporal blindness" and "contradictory actions within chunk" are harder to resolve than Kimi suggests. Rates complexity as High
- **Promotes async as #3** — if Phase 1+2 don't suffice, async hides remaining latency completely. Feasible if reduced refinement shrinks VRAM enough for concurrent fwd+bwd

#### Recommended Plan (Gemini)

**Phase 1**: Batched episodes (B=6 via `jax.vmap`) + XLA latency hiding scheduler flag
**Phase 2**: Reduced refinement (H=2,L=2 for rollout) — synergy: less VRAM offsets batching overhead
**Phase 3**: Async producer-consumer (if still needed after 10-15x from Phase 1+2)

Combined target: **10-15x → 509s → ~35-50s**

#### XLA-Specific Recommendations

- Set `XLA_FLAGS='--xla_gpu_enable_latency_hiding_scheduler=true'` to overlap memory transfers with compute
- `stop_gradient` must correctly signal XLA to free intermediate buffers during batched scan
- Monitor for VRAM fragmentation — batched scan + 50M model could spike without proper buffer reuse
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION` if async path needs explicit VRAM partitioning

#### Nuclear Options Assessment

| Option | Speedup | Verdict |
|--------|---------|---------|
| Learned state predictor | >20x | **Reject** — forcing NN to approximate deterministic arithmetic (PnL accumulation) = catastrophic error accumulation over 336 steps |
| Remove state channels | >50x | **Reject** — fundamentally alters paradigm, requires full retrain |
| Offline/stale rollout | ~10x | **Reject** — rapid off-policy divergence as policy updates on frozen states |
| Noisy oracle state | >50x | **Reject** — impossible to engineer noise that maps to model's structural biases |

---

## Consensus (3/3 models)

### Unanimous Agreement

| Strategy | Kimi | Gemini | Claude | Verdict |
|----------|------|--------|--------|---------|
| **Batched episodes** | #1 (4-6x) | #1 (4.5-5.5x) | Tier 1 (2-4x) | **DO FIRST** — all three agree, zero quality risk |
| **bf16 rollout** | #3 (1.5-2x) | Implicit | Tier 1 (1.2-1.5x) | **Easy win** — but fp32 state accumulation mandatory (Claude: 260% drift risk) |
| **Sparse/conditional rollout** | Moderate | **Reject** | **Reject** | **SKIP** — `jnp.where` executes both branches, `lax.cond` breaks batching |
| **Distilled draft model** | Moderate | — | **Reject** | **SKIP** — still bandwidth-bound at B=1, same gain as reduced refinement with 10x complexity |

### Key Divergence: Analytical States vs Optimize-the-Scan

| | Kimi | Gemini | Claude |
|---|------|--------|--------|
| **Core strategy** | Optimize scan (batch + chunk + bf16) | Optimize scan (batch + reduced refinement) | **Eliminate scan** (analytical state + teacher forcing) |
| **Best case** | 15-20x | 10-15x | **10-30x** (rollout cost → ~0) |
| **Risk model** | Bug risk from chunking | State quality from truncation | Exposure bias from teacher forcing |

Claude's analytical state argument is the **most radical but also best-justified** — 4/5 channels are deterministic arithmetic, only confidence needs NN. Kimi/Gemini rejected this class of approach ("noisy oracle state") but Claude reframes it as **standard scheduled sampling from seq2seq literature**, not arbitrary noise injection.

### Reduced Refinement Disagreement

| | Kimi | Gemini | Claude |
|---|------|--------|--------|
| **Rank** | #7 | **#2** | Tier 2 |
| **Target** | H=?, L=? | H=2, L=2 (4 passes) | L=4 (half depth) |
| **Speedup** | 2-3x | 6-10x | 1.5-2x |
| **Quality** | Moderate concern | Minimal (regularization) | Minimal (CALM paper: argmax stable at 50% depth) |

Gemini most aggressive (H=2,L=2 = 4 passes), Claude most conservative (L=4 = 16 passes), Kimi didn't explore deeply.

### Async Pipeline Disagreement

| | Kimi | Gemini | Claude |
|---|------|--------|--------|
| **Verdict** | 1.2-1.5x marginal | #3 (~5.7x latency hiding) | **Reject on single GPU** |

Claude's argument: "JAX dispatches all kernels on single CUDA stream — rollout and training cannot overlap on one GPU." Directly contradicts Gemini. **Claude is likely correct** — JAX doesn't expose multi-stream GPU, so true overlap requires multi-device.

### Final Implementation Order

1. **Batched episodes (vmap B=6)** — unanimous, zero risk, 3-5x
2. **bf16 rollout inference** — trivial, fp32 state mandatory, +1.2-1.5x
3. **scan tuning** (unroll=4, donate_argnums) — free, +1.1-1.3x
4. **Evaluate**: measure actual speedup. If 4-8x achieved → Phase 2
5. **A/B test analytical states** — Claude's big idea. Train 5 epochs with oracle-action-derived state vs model-state. If gap <2% → adopt (10-30x total). If not → scheduled sampling
6. **Reduced refinement (L=4 for rollout)** — if analytical states insufficient, +1.5-2x
7. **Chunked rollout (K=8)** — last resort, highest complexity, +2-3x

---

## Data Safety Analysis

**Critical distinction**: most approaches modify the state distribution the model trains on. State channels ARE training data — changing how they're generated changes what the model learns.

### Truly Safe (Identical Training Signal)

| Approach | Why Safe | Combined |
|----------|----------|----------|
| **Batched vmap (B=6)** | Mathematically identical — just GPU parallelism | |
| **bf16 inference** (fp32 state accum) | Argmax unchanged, rounding below action threshold | |
| **scan tuning** (unroll=4, donate_argnums) | Compilation hints, same computation | |
| **Combined** | | **~4-6x** |

These are the only approaches where the model sees exactly the same data as before. Everything beyond this is a tradeoff.

### Changes State Distribution (Requires A/B Validation)

| Approach | What Changes | Risk |
|----------|-------------|------|
| **Reduced refinement (L=4 or L=2)** | Rollout model makes *different action choices* than full model. Different actions → different positions → different PnL → different heat → **every subsequent timestep's state is different**. Model trains on states generated by a dumber version of itself. | May learn to compensate for errors it won't make at full-refinement inference. Gemini calls it "regularization" — unproven. |
| **Chunked rollout (K=8)** | Model sees stale state within chunks. When it trades mid-chunk, position/PnL/heat for remaining chunk steps are wrong. | Model learns incorrect state→action mappings for ~25% of trade-adjacent timesteps. |
| **Analytical states (teacher forcing)** | Model sees states built from *oracle actions*, not its own. This is exactly what AR was designed to fix — at 0% AR (oracle state) the model trained fine but turtled at inference because it never saw its own state. | Claude's counter: scheduled sampling (blend oracle→model). But this is essentially what AR blend already does. The speedup only applies to the oracle-state fraction — model-state fraction still needs NN rollout. |

### The Bottom Line

The **safe ceiling is ~4-6x** (batch + bf16 + scan tuning). That takes 509s → ~85-125s.

Everything beyond 6x trades state accuracy for speed. Whether that trade works is **empirical, not theoretical** — needs A/B testing:
- Train 20 epochs with modified rollout
- Compare val agreement + agree_trade to baseline
- If degradation < 1% → acceptable
- If degradation > 3% → not worth the speed

---

## Implementation Notes

### Batched Episode Rollout (Priority 1)

Current flow in `train_easy.py`:
```
_generate_episode_data() called per episode
  → _build_model_state_trajectory() called once per AR episode
    → jax.lax.scan over T=336 steps, batch_size=1
```

Proposed: collect 6 episodes worth of `market_obs` and `returns`, stack on batch dim, single scan call.

**File**: `train_easy.py:176-310` (`_build_model_state_trajectory`)
**Validation**: Compare state trajectories to unbatched — must be bit-identical.

### Chunked Rollout (Priority 2)

95% of actions are HOLD → position rarely changes within 8-step window. State drift bounded.

Correctable channels (analytical during frozen chunks):
- PnL: `new_pnl = pnl + position * returns[t]` (exact, no model needed)
- Heat: `h_fast *= decay_fast`, `h_slow *= decay_slow` (exact decay)
- Steps: `steps += 1` (trivial)

Only position requires model inference → error only when model trades during frozen chunk.

### bfloat16 (Priority 3)

RTX 3090 Ampere has native bf16 tensor cores. Model likely memory-bandwidth-bound during single-sample inference. Cast params + obs for rollout only, state updates stay float32.

Already using `--bf16` flag in training — check if rollout path respects it.

---

## Validation Criteria

- [ ] State trajectory divergence < 2% vs baseline full AR
- [ ] No training stability regression over 20 epochs
- [ ] GPU utilization > 80% during rollout (currently likely <30%)
- [ ] Epoch time < 150s at 89% AR (3.5x improvement minimum)
