# TASK_716: Phase 2 — Dynamics Training + MCTS Integration

**Status**: TODO
**Priority**: 0 (next major milestone — Phase 1 imitation ceiling reached)
**Depends on**: TASK_715 (direct-sample pipeline), Phase 1 checkpoint (~E8800+)

---

## Context

Phase 1 (The Cheat) is complete:
- 85% balanced agreement on consistent full-period oracle targets
- 42.7% benchmark trade agreement (held-out data)
- Model trades 18-22% vs oracle's 2-3% — identifies opportunities but can't filter to top-k
- **Imitation ceiling reached**: agree_trade flat at ~42% for 1000+ epochs
- Dynamics layers 100% sparse (blank slate), encoder+TRM 0% sparse (healthy)

Phase 2 (The Dream) trains the dynamics network to predict state transitions, enabling MCTS to simulate multi-step outcomes and filter to top-k trades.

---

## Verified Architecture (from checkpoint E8854+, large preset)

```
state_dim:          128
action_space_size:  320 (5 actions × 64 assets)
value_support:      401 bins over [-10, 10]
value_scale:        2.0 (Phase 1)
hidden_dim:         512 (TRM)
trm_hidden_dim:     2048

Dynamics layers (all 100% sparse, ~3.3M params):
  action_embedding/embedding:     (320, 128)
  dynamics_dense1/kernel:         (256, 1024)   ← concat(latent_128, action_emb_128) → 1024
  dynamics_dense2/kernel:         (1024, 529)   ← 1024 → state_128 + reward_401
  dynamics_ln:                    (1024,)
  dynamics_state_norm:            (128,)
```

### Data flow for MCTS

```
Root:   initial_inference(obs) → latent₀(128), policy₀(320), value₀(401)
Expand: recurrent_inference(latent, action) → next_latent(128), reward(401)
          ↓ stop_gradient
        prediction(next_latent) → policy(320), value(401)
```

**Critical**: BUG-017 stop_gradient on line 276 of muzero_model.py means value/policy gradients do NOT flow through dynamics. Dynamics is trained ONLY via reward loss + consistency loss. This is standard MuZero and correct.

---

## Design: Two-Stage Phase 2

### Stage A: Dynamics Warmup (frozen encoder, dynamics learns transitions)

**Goal**: Dynamics network learns to predict next latent state from (current_latent, action).

**Freeze**: Encoder + TRM + Policy head + Value head (all Phase 1 weights)
**Train**: dynamics_dense1/2, dynamics_ln, dynamics_state_norm, action_embedding (~3.3M params)

**Data source**: Same SegmentPool + direct pipeline (TASK_715). Modified to return consecutive window pairs.

**Losses**:

```python
def dynamics_training_step(params, frozen_params, obs_t, obs_t1, action_t, reward_t):
    # Encode current and next observation (frozen encoder, no gradients)
    # NOTE: Use {'params': ...} wrapper (Flax convention)
    latent_t = jax.lax.stop_gradient(
        model.apply({'params': frozen_params}, observation=obs_t, mode='initial_inference')['latent_state']
    )
    latent_t1 = jax.lax.stop_gradient(
        model.apply({'params': frozen_params}, observation=obs_t1, mode='initial_inference')['latent_state']
    )

    # Predict next state via dynamics (trainable params)
    # recurrent_inference: first positional arg after params is latent_state (mapped to 'observation')
    dynamics_out = model.apply({'params': params}, latent_t, action_t, mode='recurrent_inference')
    pred_latent = dynamics_out['latent_state']
    pred_reward = dynamics_out['reward']  # (batch, 401) categorical

    # Loss 1: Consistency — dynamics predicts the right next latent
    # NOTE: Will have irreducible floor because target latent includes oracle state
    # channel information that dynamics can't access. Target: < 0.05, not 0.0
    consistency_loss = jnp.mean((pred_latent - latent_t1) ** 2)

    # Loss 2: Reward prediction — per-step PnL
    # Scale reward by 100x before binning (matching Poppy's TASK_034 / mctx_search descale)
    # Use _scalar_to_target pattern from easy_trainer.py:115-126
    scaled_reward = jnp.clip(reward_t * REWARD_SCALE, -10.0, 10.0)
    reward_target = scalar_to_categorical(scaled_reward)  # (batch, 401) one-hot
    reward_loss = optax.softmax_cross_entropy(pred_reward, reward_target).mean()

    return consistency_weight * consistency_loss + reward_weight * reward_loss
```

**Optimization — latent cache for consecutive pairs**: Instead of encoding obs_t and obs_t1
separately (2x encoder cost), pre-encode the full cursor range once and index consecutive
latent pairs. See `easy_trainer.py:139-145` for existing latent cache pattern.

```python
# More efficient: encode once, index pairs
latents_all = jax.lax.stop_gradient(
    model.apply({'params': frozen_params}, observation=obs_batch, mode='initial_inference')['latent_state']
)  # (N, state_dim)
latent_t = latents_all[:-1]   # (N-1, state_dim)
latent_t1 = latents_all[1:]   # (N-1, state_dim) — consecutive pairs for free
```

**Consecutive pairs**: For each sample position at cursor c, we need:
- `obs_t`: observation at cursor c → encode → latent_t
- `obs_t1`: observation at cursor c+1 → encode → latent_t1 (target)
- `action_t`: oracle action at decision point c + ws - 1
- `reward_t`: per-step PnL at decision point c + ws - 1

**Reward computation**: `reward_t = sum(oracle_position * returns[c+ws-1]) / n_assets - transaction_costs_for_trades`. Precompute during `_make_slot` from oracle selections and returns.

**Initialization**: Explicit He normal for Dense kernels (`kernel_init=nn.initializers.he_normal()`), normal(0, 0.02) for Embed. More stable than Flax defaults for cold-start dynamics. Create fresh model, extract dynamics keys, merge into Phase 1 params.

**Duration**: ~200-500 epochs. Monitor consistency loss → should drop steadily toward <0.01.

**Learning rate**: 3e-4 for dynamics. Frozen layers excluded from optimizer.

### Stage B: MCTS Self-Play (ExIt loop on historical data)

**Goal**: Use MCTS search to generate improved policy targets. Train policy head to match MCTS-selected actions.

**NOTE**: Stage B replaces the original 3-stage plan. The review correctly identified that:
1. Stage B (value fine-tuning) is blocked by stop_gradient in recurrent_inference
2. Stage C's "train policy to match visit counts" is circular if misunderstood

**Correct ExIt formulation**:

```
For each market state from SegmentPool:
  1. Encode observation → latent (frozen encoder)
  2. Run MCTS search (32 sims) using trained dynamics + value head
     → MCTS evaluates 32 hypothetical action sequences
     → Visit counts concentrate on best actions
  3. MCTS selects action: a_mcts = argmax(visit_counts)
  4. Compute actual reward from executing a_mcts on real market data

Training targets:
  - Policy: cross_entropy(policy_logits, MCTS_visit_distribution)
    This is NOT circular because:
    - MCTS uses dynamics to simulate FUTURES (multi-step lookahead)
    - Policy alone can't simulate futures — it's single-step
    - Visit counts reflect multi-step value, not just policy prior
    - Training policy → visit_counts teaches it the RESULT of planning
    → This is standard AlphaZero/MuZero policy improvement

  - Value: TD bootstrap from MCTS root value
    v_target = r_immediate + γ * V(s_next)  where V comes from MCTS search
    This improves value estimation beyond Phase 1's oracle remaining PnL

  - Dynamics: Continue consistency loss (ongoing calibration)
    As encoder/value change slightly, dynamics must stay aligned

  - Oracle BC (decaying): Keep oracle imitation as regularizer
    bc_loss = cross_entropy(policy_logits, oracle_action) * bc_weight
    bc_weight decays from 0.05 → 0.005 over 5000 steps
    Prevents catastrophic forgetting of Phase 1 trade intuition
```

**Why MCTS visit counts are NOT circular**:
The policy outputs a single-step action distribution. MCTS uses dynamics to simulate 5+ steps ahead, evaluates with value head, and concentrates visits on actions that lead to good MULTI-STEP outcomes. The policy can't do this alone — it has no dynamics. Training the policy to match visit counts distills the multi-step planning knowledge into a single-step predictor. This is the core of Expert Iteration.

**The top-k filtering mechanism**: MCTS `max_num_considered_actions=16` limits tree expansion to the top-16 policy priors. The search evaluates these 16 candidates over 32 simulations. Visit counts will further concentrate on the best 4-8 actions. The policy learns from these concentrated counts → becomes more selective over training.

**Data flow**: Continue using SegmentPool (historical data). No replay buffer needed — we're doing on-policy ExIt with historical states, not true self-play.

```python
# Per training step in Stage B:
obs = encode_from_pool(cursor)  # Same TASK_715 direct pipeline
latent = frozen_encoder(obs)

# MCTS search (uses dynamics from Stage A)
search = MuZeroMCTXSearch(model.apply, num_simulations=32, use_gumbel=True,
                          max_num_considered_actions=16, epsilon_greedy=0.1)
actions, visit_counts, stats = search(params, rng, obs_batch, action_mask=None)
# visit_counts: (batch, 320) — concentrated on best actions by MCTS
# stats['root_value']: (batch,) — scalar value from search (already in [-10,10] scale)

# Get model outputs at root
root_out = model.apply({'params': params}, observation=obs_batch, mode='initial_inference')

# Policy loss: distill MCTS multi-step planning into single-step policy
policy_loss = optax.softmax_cross_entropy(root_out['policy_logits'], visit_counts).mean()

# Value loss: categorical CE against MCTS root value (NOT scalar MSE)
# Convert scalar root_value to categorical target (same as Phase 1 value training)
value_target = scalar_to_categorical(jnp.clip(stats['root_value'], -10.0, 10.0))
value_loss = optax.softmax_cross_entropy(root_out['value'], value_target).mean()

# Dynamics loss: ongoing consistency calibration (same as Stage A)
dynamics_loss = consistency_loss(params, frozen_encoder_params, obs_t, obs_t1, action_t)

# Oracle BC regularizer: prevents forgetting Phase 1 trade intuition (decaying)
bc_loss = optax.softmax_cross_entropy(root_out['policy_logits'], oracle_targets).mean()

total_loss = (policy_weight * policy_loss +
              value_weight * value_loss +
              consistency_weight * dynamics_loss +
              bc_weight * bc_loss)
```

**Unfreeze strategy for Stage B**:
- Policy head: unfrozen (learning from MCTS targets)
- Value head: unfrozen (learning from MCTS root values)
- Dynamics: unfrozen (ongoing consistency calibration)
- Encoder: frozen initially, optionally unfreeze at 10x lower LR after 200 epochs if stable

**Duration**: 1000+ epochs. Monitor benchmark alpha — this is where we expect it to turn positive.

---

## State Channels in MCTS

**Issue**: In Phase 1, observations include 7 state channels (position, PnL, hold_dur, heat×2, capacity, confidence) computed from the oracle trajectory. During MCTS tree expansion, there's no oracle — the model must handle state channels differently.

**Solution for Stage A**: Use oracle state channels (same as Phase 1 training). Dynamics learns transitions in the latent space that corresponds to oracle-state observations.

**Solution for Stage B**: Two options:
1. **Oracle state channels** (simpler): Continue using precomputed oracle state at the ROOT observation. Internal MCTS nodes don't see state channels — they operate in latent space only via dynamics. The root observation still has oracle state, which is fine because we're training on historical data where oracle state is known.
2. **Zero state channels** at root: Force model to work without oracle state. This matches inference conditions where no oracle exists. More honest but may regress initially.

**Recommendation**: Start with oracle state (option 1) for stability. Switch to zero state (option 2) after Stage B has stabilized, to ensure inference robustness.

---

## Reward Scaling

Phase 1 used `value_scale=2.0` for oracle remaining PnL (range 0-5, mapped to [-10, 10] support).

Poppy uses `REWARD_SCALE=100` for per-step rewards (range 0.001-0.05 raw, × 100 → 0.1-5.0).

MCTS search descales: `reward / 100.0` (mctx_search.py line 110).

**For Stage A**: Train reward prediction with `REWARD_SCALE=100`. Per-step rewards are tiny (~0.001) and need 100× to spread across 401 bins.

**For Stage B**: MCTS root value comes from the value head which outputs in [-10, 10] range. TD targets should be in the same scale. `v_target = (r * 100) / 100 + γ * V(s_next)` — the reward descaling cancels out within the search.

---

## MCTS Performance

From Poppy benchmarks: ~1000 recurrent inferences/sec on RTX 3090 for state_dim=128.

32 sims = 1 initial + 31 recurrent = ~31ms per observation.
Batch of 256: ~8s (parallelized across batch).
Batch of 1024: ~31s (may need to reduce batch_size for Stage B).

**Recommendation**: Use batch_size=256 for Stage B (MCTS step). Each epoch has 400 steps × 256 = 102k samples. With MCTS at ~8s/step + training at ~0.2s/step ≈ 8.2s/step → 3280s/epoch (~55 min). Acceptable.

---

## Implementation Plan

### Checkpoint Compatibility (VERIFIED)

Phase 1 checkpoint contains ALL keys needed for Phase 2:
- `trm/encoder/` + `trm/reasoning/` — trained (0% sparse) → freeze in Stage A
- `action_embedding/`, `dynamics_*` — zero (100% sparse) → re-initialize with random init
- No missing keys, no shape mismatches, no architecture changes needed

Loading Phase 1 into Phase 2: load checkpoint → replace zero dynamics keys with He init → train.

### Files to create
- `train_phase2.py` — **Separate entry point** for Phase 2 (NOT modifying train_easy.py).
  Clean separation: Phase 1 continues to work as-is, Phase 2 is a new root script.
  Imports from existing modules: EasyConfig, SegmentPool, MuZeroNetworks, mctx_search.
- `phase2_config.py` — Phase 2 config (extends EasyConfig with consistency_weight, reward_weight,
  bc_weight, freeze flags, mcts_sims, etc.)

### Files to modify
- `SegmentPool._make_slot()` in `train_easy.py` — Precompute per-step rewards for dynamics training.
  Small addition, backward compatible (Phase 1 ignores the new key).

### Files to use as-is (import, don't modify)
- `easy_trainer.py` — Phase 1 trainer (reference for Phase 2 trainer structure)
- `mcts_trader/jax_agent/muzero_model.py` — dynamics/recurrent_inference already implemented
- `mcts_trader/jax_agent/mctx_search.py` — MCTS search ready
- `mcts_trader/jax_agent/popart.py` — Value normalization (optional)
- `train_easy.py` — SegmentPool, OnTheFlyGenerator (data pipeline reused)

---

## Execution Order

```
Stage A: Dynamics Warmup (~200-500 epochs, ~3-8 hours)
  1. Create phase2_trainer.py with dynamics_loss
  2. Add --phase 2 mode to train_easy.py
  3. Modify data pipeline for consecutive window pairs (obs_t, obs_t1)
  4. Precompute per-step rewards in _make_slot
  5. Initialize dynamics weights (fresh Flax init, merge with Phase 1 params)
  6. Train: freeze encoder+policy+value, train dynamics only
  7. Monitor consistency_loss → plateau = done

Stage B: MCTS ExIt (~1000+ epochs, ongoing)
  8. Integrate mctx_search into training loop
  9. Add ExIt loss (policy→visit_counts, value→root_value)
  10. Add oracle BC regularizer (decaying weight)
  11. Unfreeze policy + value heads
  12. Monitor benchmark alpha → should turn positive
  13. Gradually reduce oracle BC weight
  14. Optionally unfreeze encoder at 10x lower LR
```

---

## Key Parameters

| Parameter | Stage A | Stage B |
|-----------|---------|---------|
| Encoder LR | 0 (frozen) | 0 initially, 3e-5 after 200ep |
| Dynamics LR | 3e-4 | 1e-4 (decay) |
| Value LR | 0 (frozen) | 3e-4 |
| Policy LR | 0 (frozen) | 3e-4 |
| consistency_weight | 1.0 | 0.5 |
| reward_weight | 1.0 | 1.0 |
| policy_weight | 0 | 1.0 |
| value_weight | 0 | 0.5 |
| bc_weight | 0 | 0.05 → 0.005 (decay over 5000 steps) |
| MCTS sims | 0 | 32 |
| max_considered_actions | — | 16 |
| epsilon_greedy | — | 0.1 |
| batch_size | 1024 | 256 (MCTS bottleneck) |
| REWARD_SCALE | 100 | 100 |

---

## Success Criteria

### Stage A
- Consistency loss < 0.01 (dynamics predicts next latent accurately)
- Reward prediction loss < 0.5 (per-step reward calibrated)
- Dynamics weights no longer 100% sparse (should be 0-5% sparse)

### Stage B
- **Benchmark alpha turns positive** (even 1/6 periods = breakthrough)
- Model trade rate drops from 18-22% toward 5-10% (MCTS top-k filtering)
- agree_trade improves beyond 42.7% (MCTS finds better trades)
- beats_random > 3/6 consistently

---

## Risks

1. **Dynamics drift over long unrolls**: Consistency loss at every step + state normalization mitigates. Start with 5-step unroll max in MCTS, increase as dynamics stabilizes.

2. **MCTS with bad dynamics**: Early Stage B, dynamics predictions are noisy. Low sims (8) initially, ramp to 32+ as consistency loss drops. Oracle BC regularizer prevents catastrophic forgetting.

3. **Representation collapse on encoder unfreeze**: Monitor consistency loss. If it spikes when encoder unfreezes, re-freeze immediately.

4. **MCTS speed bottleneck**: 32 sims × 256 batch ≈ 8s/step. Acceptable for 55 min/epoch. If too slow, reduce sims to 16 (still better than no search).

5. **Reward scale**: Per-step rewards ~0.001 need 100× scaling. If reward loss doesn't converge, try 50× or 200× and monitor bin utilization.

---

## Checkpoint Strategy

```
phase1_final.pkl           ← Best Phase 1 checkpoint (E8800+)
phase2_stageA_best.pkl     ← Lowest consistency loss
phase2_stageB_epoch*.pkl   ← Periodic saves during MCTS training
phase2_alpha_positive.pkl  ← First checkpoint with positive alpha (milestone!)
```

---

## Subtask Breakdown

Each subtask is a self-contained unit with its own implementation, validation, and gate criterion. Do NOT proceed to the next subtask until the gate passes.

### TASK_716a: Infrastructure — train_phase2.py + phase2_config.py

**Scope**: Create the skeleton entry point and config. No training logic yet — just arg parsing, checkpoint loading, model creation, and data pipeline setup.

**Deliverables**:
- `train_phase2.py`: CLI entry, loads Phase 1 checkpoint, creates model, imports SegmentPool
- `phase2_config.py`: Extends EasyConfig with Phase 2 params

**Validation**:
```bash
# Must pass before proceeding:
python train_phase2.py --checkpoint jax_ckpt_easy_A/branch_A_historical/phase1_best.pkl --dry-run
# Expected output:
#   Loaded Phase 1 checkpoint (epoch XXXX, 49.59M params)
#   Dynamics layers: 9/9 zero (ready for init)
#   Encoder layers: 0 zero (trained)
#   SegmentPool: 17 regimes loaded
#   Dry run complete — ready for Stage A
```

**Gate**: Script loads checkpoint, verifies dynamics=zero + encoder=trained, loads pool. No crashes.

---

### TASK_716b: Dynamics Initialization

**Scope**: Replace zero dynamics weights with proper He init. Verify the model can run `recurrent_inference` without NaN.

**Deliverables**:
- Function `initialize_dynamics(params, rng_key)` that replaces zero keys with He normal init
- Smoke test: run `recurrent_inference(latent, action)` on random input → no NaN, shapes correct

**Validation**:
```python
# In train_phase2.py --test-init:
params = initialize_dynamics(phase1_params, rng)
latent = jnp.ones((1, 128))  # dummy
action = jnp.array([42])      # dummy
out = model.apply({'params': params}, latent, action, mode='recurrent_inference')
assert out['latent_state'].shape == (1, 128)
assert out['reward'].shape == (1, 401)
assert not jnp.any(jnp.isnan(out['latent_state']))
print(f"recurrent_inference OK: latent std={out['latent_state'].std():.4f}")
```

**Gate**: `recurrent_inference` produces valid outputs with initialized dynamics. No NaN.

---

### TASK_716c: Dynamics Loss + Training Loop (Stage A core)

**Scope**: Implement the dynamics training step — consistency loss + reward loss. Freeze encoder. Train for 10 epochs and verify loss decreases.

**Deliverables**:
- `dynamics_train_step(params, frozen_params, obs_t, obs_t1, action_t, reward_t)` — JIT-compiled
- Modified data pipeline: `generate_batch_direct` returns consecutive pairs (obs_t, obs_t1)
- Precomputed per-step rewards in SegmentPool slots
- Training loop that logs consistency_loss and reward_loss per epoch

**Validation**:
```bash
# Run 10 epochs of dynamics training:
python train_phase2.py --checkpoint ... --stage A --epochs 10
# Expected output:
#   E0: consistency=X.XX reward=X.XX  (high — random dynamics)
#   E9: consistency=Y.YY reward=Y.YY  (lower — dynamics learning)
# Verify: consistency_loss at E9 < consistency_loss at E0 (monotonic decrease)
```

**Gate**: Consistency loss decreases over 10 epochs. Reward loss does not diverge. No NaN.

---

### TASK_716d: Stage A Full Run + Validation

**Scope**: Run Stage A to convergence (~200-500 epochs). Validate dynamics quality.

**Deliverables**:
- Full Stage A training run with checkpointing
- Dynamics quality test: encode 1000 consecutive windows, predict next latent via dynamics, measure MSE vs actual next latent
- Reward prediction quality: compare predicted step rewards vs actual on validation data

**Validation**:
```bash
python train_phase2.py --checkpoint ... --stage A --epochs 500
# Then validate:
python train_phase2.py --checkpoint phase2_stageA_best.pkl --validate-dynamics
# Expected output:
#   Dynamics consistency MSE: < 0.01
#   Reward prediction MSE: < 0.5
#   5-step unroll drift: < 0.05 (dynamics stable over short sequences)
```

**Gate**: consistency < 0.01, reward MSE < 0.5, 5-step drift < 0.05. Dynamics is a usable world model.

---

### TASK_716e: MCTS Integration Smoke Test

**Scope**: Wire up mctx_search with the Stage A dynamics. Run MCTS on a single batch and verify it produces valid actions and visit counts.

**Deliverables**:
- MCTS search configured with Stage A checkpoint
- Smoke test: 1 batch of 16 observations, 32 sims, verify output shapes + no NaN

**Validation**:
```python
# In train_phase2.py --test-mcts:
search = MuZeroMCTXSearch(model.apply, num_simulations=32, use_gumbel=True)
actions, visit_counts, stats = search(params, rng, obs_batch)
assert actions.shape == (16,)
assert visit_counts.shape == (16, 320)
assert jnp.all(visit_counts >= 0)
assert jnp.allclose(visit_counts.sum(axis=-1), 1.0, atol=0.01)
print(f"MCTS OK: root_value mean={stats['root_value'].mean():.4f}")
print(f"Visit entropy: {-(visit_counts * jnp.log(visit_counts + 1e-8)).sum(-1).mean():.2f}")
```

**Gate**: MCTS produces valid actions, normalized visit counts, no NaN. Visit entropy > 0 (not collapsed).

---

### TASK_716e2: JAX-Native Data Pipeline (lax.scan port)

**Scope**: Port the SegmentPool data feeding from Python-orchestrated (numpy→GPU per step) to a fully JIT'd `lax.scan` loop with prices pre-loaded on GPU. This eliminates the Python overhead that made Foxglove 10-50x slower than Poppy.

**Why here**: Stage B (MCTS ExIt) runs 32 MCTS sims per step at ~8s/step for batch=256. If the data pipeline adds another 1-2s of Python overhead per step, that's 12-25% wasted. With thousands of Stage B epochs ahead, this compounds to days of wasted GPU time. Fix it before the long run.

**Current bottleneck**:
```
Python: SegmentPool.sample_positions() → numpy cursors
Python: slot['hm'].cache_data[c_min:c_max+ws] → numpy slice
Python→GPU: jnp.array(price_slice) → transfer
GPU: jit_batch_encode_temporal → encode
GPU: jit train_step → gradient
Back to Python for next step
```

**Target architecture**:
```
Startup: pre-load prices for active slots onto GPU (JAX arrays)
         pre-load oracle targets, state channels onto GPU
GPU-only: lax.scan over training steps
  Per step: index into GPU-resident prices → encode → train
  No Python in the loop, no CPU→GPU transfers
```

**Key challenges**:
1. **Multi-slot indexing**: 17 slots with different price arrays. Can't stack (different lengths). Options:
   - Pad all to max length → single (17, max_T, 64, 5) array on GPU (~17 × 330MB = 5.6GB VRAM)
   - Or: process one slot at a time in the scan, round-robin
2. **Dynamic cursor advancement**: `lax.scan` needs static shapes. Cursor state can be carried as scan state.
3. **Slot rotation**: When a slot exhausts, need to load new data. This MUST break out of the scan (disk I/O). Use `jax.lax.cond` or `host_callback` for rare rotation events.
4. **VRAM budget**: RTX 3090 has 24GB. Model params ~200MB. MCTS tree ~500MB. Prices: depends on approach. Budget ~5-8GB for data.

**Deliverables**:
- GPU-resident price arrays for active slots (padded or chunked)
- GPU-resident oracle targets + state channels (from precomputation)
- `jax_training_loop(params, slot_data, cursors, rng, n_steps)` — single JIT'd function
- Fallback to Python path when slot rotation needed

**Validation**:
```bash
python train_phase2.py --checkpoint ... --stage B --benchmark-speed --epochs 5
# Compare:
#   Python pipeline: X.Xs/step
#   JAX pipeline: Y.Ys/step
#   Speedup: X/Y (target: 3-5x)
```

**Gate**: JAX pipeline achieves >3x speedup over Python pipeline on Stage B training steps. No accuracy regression (same loss values within tolerance).

**Note**: This is an optimization subtask. If it proves too complex (multi-slot VRAM management), can be deferred — Stage B still works with the Python pipeline, just slower. But the speed gain compounds over 1000+ epochs, so it's worth the effort.

---

### TASK_716f: Stage B ExIt Training Loop

**Scope**: Implement the full ExIt loop — MCTS generates targets, model trains on them. With oracle BC regularizer.

**Deliverables**:
- `exit_train_step(params, obs, oracle_action, mcts_visit_counts, mcts_root_value)` — JIT-compiled
- Training loop: for each step, run MCTS → get targets → train → log metrics
- Oracle BC weight with configurable decay
- Benchmark integration (every 100 epochs)

**Validation**:
```bash
python train_phase2.py --checkpoint phase2_stageA_best.pkl --stage B --epochs 100 --benchmark-every 50
# Expected output after 100 epochs:
#   Policy loss decreasing (model learning from MCTS targets)
#   Value loss decreasing (TD bootstrap improving)
#   Dynamics consistency stable (not drifting)
#   BC loss present but small (regularizer working)
#
# Benchmark at E50 and E100:
#   Compare agree_trade vs Phase 1 baseline (42.7%)
#   Compare model_trade_rate (should decrease toward 5-10%)
```

**Gate**: Policy loss decreases. Model trade rate in benchmark drops below 15% (MCTS filtering working). No loss divergence.

---

### TASK_716g: Stage B Extended Run + Alpha Hunt

**Scope**: Run Stage B for 1000+ epochs. Hunt for positive alpha.

**Deliverables**:
- Extended training with periodic benchmarks
- Dashboard integration (CSV metrics, DSe compatible)
- Checkpoint at first positive alpha period

**Validation**:
```
Benchmark shows:
  - alpha > 0 on at least 1/6 periods
  - model_trade_rate < 10% (selective trading)
  - agree_trade > 45% (better than Phase 1)
  - beats_random > 3/6
```

**Gate**: First positive alpha on any benchmark period. This is the Phase 2 milestone.
