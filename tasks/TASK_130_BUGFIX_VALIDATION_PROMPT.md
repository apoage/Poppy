# TASK_130: Bug Fix Validation Audit

**Priority**: HIGH
**Status**: PENDING VALIDATION
**Auditor**: [GEMINI/GLM/KIMI]

---

## Mission

You are auditing 16 bug fixes applied to a JAX-based MuZero reinforcement learning trading system. Your job is to **validate each fix is correctly implemented** - check the actual code, verify the logic, and identify any remaining issues.

**You are an AUDITOR. Write findings and suggest solutions. Do NOT edit code.**

---

## Files to Audit

Read these files in order:

```
1. mcts_trader/jax_agent/trainer.py
2. mcts_trader/jax_agent/expert_buffer.py
3. mcts_trader/jax_agent/replay_buffer.py
4. mcts_trader/jax_agent/muzero_model.py
5. mcts_trader/jax_agent/path_solver.py
6. mcts_trader/training/self_play.py
7. mcts_trader/training/config.py
8. mcts_trader/training/cli.py
9. mcts_trader/training/setup.py
10. mcts_trader/env/jax_trading_env.py
11. mcts_trader/env/price_cache.py
12. train_jax_fast.py
```

---

## Bug Fixes to Validate

### Phase 1 Critical Fixes (7 bugs)

#### BUG-001: BC Confidence Masked Mean
**File**: `trainer.py` (search for `bc_loss` or `_compute_bc_loss`)
**Problem**: BC loss was computed as `jnp.mean(loss * confidence)` which dilutes gradients when confidence is sparse
**Expected Fix**:
```python
# WRONG (dilutes with zeros):
bc_loss = jnp.mean(cross_entropy * confidence)

# CORRECT (masked mean):
bc_loss = jnp.sum(cross_entropy * confidence) / (jnp.sum(confidence) + 1e-8)
```
**Validate**:
- [ ] Find the BC loss computation
- [ ] Confirm it uses sum/sum pattern, NOT simple mean
- [ ] Check epsilon guard on denominator

#### BUG-002: PER Importance Weights Not Applied
**File**: `trainer.py` (search for `per_weights` in BC loss section)
**Problem**: PER weights from expert_buffer were computed but never used in loss
**Expected Fix**: BC loss should multiply by both confidence AND per_weights
```python
bc_loss = jnp.sum(cross_entropy * confidence * per_weights) / (jnp.sum(confidence) + 1e-8)
```
**Validate**:
- [ ] Check if `per_weights` is passed to loss function
- [ ] Confirm it's multiplied into BC loss
- [ ] Verify expert_buffer returns per_weights in sample batch

#### BUG-003/004: PopArt Weight Correction Disabled
**File**: `trainer.py` (search for `popart` or `weight_correction` or `_update_popart`)
**Problem**: PopArt weight correction is incompatible with categorical value heads (applies to scalar heads only)
**Expected Fix**: Weight correction should be DISABLED or skipped
```python
# Either removed entirely, or:
if not self.use_categorical_value:
    # apply weight correction
    pass
# For categorical: skip weight correction, keep normalization only
```
**Validate**:
- [ ] Find PopArt update logic
- [ ] Confirm weight correction is disabled/removed for categorical heads
- [ ] Normalization (mean/std tracking) should still work

#### BUG-005: MCTS Visit Count Normalization
**File**: `self_play.py` (search for `policy_array` or visit counts)
**Problem**: MCTS visit counts may not sum to 1.0, breaking cross-entropy loss
**Expected Fix**: Defensive normalization before storing
```python
policy_array = policy_array / (np.sum(policy_array) + 1e-8)
```
**Validate**:
- [ ] Find where MCTS policy is stored in trajectory
- [ ] Confirm normalization is applied
- [ ] Check it handles edge case of all-zero visits

#### BUG-006: MuZero Consistency Loss Missing
**File**: `trainer.py` (search for `consistency_loss` or `dynamics` loss)
**Problem**: MuZero should have consistency loss to prevent dynamics drift during K-step unrolling
**Expected Fix**:
```python
# In K-step unroll:
predicted_latent = dynamics_fn(prev_latent, action)
true_latent = representation_fn(next_obs)  # or from stored latents
consistency_loss = jnp.mean((predicted_latent - jax.lax.stop_gradient(true_latent))**2)
```
**Validate**:
- [ ] Find MuZero unroll logic
- [ ] Confirm consistency loss is computed
- [ ] Check stop_gradient on target
- [ ] Verify it's added to total loss with appropriate weight (~0.5)

#### BUG-033: Alpha-Relative Scale Mismatch
**File**: `trainer.py` (search for `alpha` or `symlog` in reward scaling)
**Problem**: Rewards use symlog but were compared to linear alpha, causing 100x scale mismatch
**Expected Fix**: Apply symlog to max_alpha before division
```python
# WRONG:
scaled_reward = symlog(reward) / max_alpha  # linear alpha

# CORRECT:
scaled_reward = symlog(reward) / symlog(max_alpha)  # both in symlog space
```
**Validate**:
- [ ] Find alpha-relative reward scaling
- [ ] Confirm both numerator AND denominator use symlog
- [ ] Check for division guards

#### BUG-034: Phantom Rewards Past Termination
**File**: `self_play.py` (search for reward summation or `n_valid`)
**Problem**: Rewards were summed past episode termination (done=True)
**Expected Fix**: Only sum rewards up to first done index
```python
n_valid = jnp.argmax(dones) + 1  # or similar
total_reward = jnp.sum(rewards[:n_valid])
```
**Validate**:
- [ ] Find episode reward computation
- [ ] Confirm it respects done flags
- [ ] Check edge case handling (no done, all done)

---

### Phase 2 High Severity Fixes (8 bugs)

#### BUG-007: Demo Priority Scale Mismatch
**File**: `expert_buffer.py` (search for priority computation)
**Problem**: Demo priorities (~0.9 confidence) were << agent TD-error priorities (~1-10)
**Expected Fix**: Scale by max_priority
```python
priority = confidence * self.max_priority + boost  # Scales to agent range
```
**Validate**:
- [ ] Find demo priority assignment
- [ ] Confirm scaling by max_priority or similar
- [ ] Check boost value is reasonable

#### BUG-009: Demo Zone Sampling Not Stratified
**File**: `expert_buffer.py` (search for `_sample_from_range` or sampling logic)
**Problem**: Used `np.random.choice` which doesn't respect priority properly
**Expected Fix**: Stratified sampling (segment tree style)
```python
# Divide total priority into n segments
# Sample one index from each segment
segment_size = total_priority / batch_size
for i in range(batch_size):
    target = (i + random()) * segment_size
    # Binary search for index with cumulative priority >= target
```
**Validate**:
- [ ] Find sampling method
- [ ] Confirm stratified approach (segments, not uniform random)
- [ ] Check it covers full priority range

#### BUG-010: Replay Buffer Recent-Only Sampling
**File**: `replay_buffer.py` (search for sampling or `heuristic_window`)
**Problem**: Only sampled from most recent ~0.2% of buffer
**Expected Fix**: Mixed recent/uniform sampling
```python
# 50% from recent, 50% from full history
recent_indices = sample_recent(n // 2)
uniform_indices = sample_uniform(n // 2)
indices = concatenate([recent_indices, uniform_indices])
```
**Validate**:
- [ ] Find main sampling method
- [ ] Confirm mixed strategy (not recent-only)
- [ ] Check ratio is reasonable (50/50 or configurable)

#### BUG-011: Weight Decay Not Configurable
**Files**: `config.py`, `cli.py`, `setup.py`
**Problem**: Weight decay was hardcoded, not configurable via CLI
**Expected Fix**: Add CLI flag and pass through
```python
# config.py
weight_decay: float = 1e-4

# cli.py
parser.add_argument('--weight-decay', type=float, default=1e-4)

# setup.py - pass to optimizer
optimizer = optax.adamw(learning_rate=lr, weight_decay=config.weight_decay)
```
**Validate**:
- [ ] Check config.py has weight_decay field
- [ ] Check cli.py has --weight-decay argument
- [ ] Check setup.py passes it to optimizer

#### BUG-013: Global Grad Clip Too High
**Files**: `config.py`, `cli.py`, `trainer.py`
**Problem**: Hardcoded to 10.0, should be configurable and lower (5.0)
**Expected Fix**: Add CLI flag, default 5.0
```python
# config.py
global_grad_clip: float = 5.0

# trainer.py
optax.clip_by_global_norm(config.global_grad_clip)
```
**Validate**:
- [ ] Check config.py has global_grad_clip field
- [ ] Check default is 5.0 (not 10.0)
- [ ] Check trainer.py uses config value (not hardcoded)

#### BUG-015: Unused Target Network (Dead Code)
**File**: `trainer.py` (search for `target_params`)
**Problem**: target_params was defined, initialized, updated every 100 steps, but NEVER USED
**Expected Fix**: Remove entirely
```python
# REMOVE all of:
# - target_params in TrainState
# - target_params initialization
# - target_params update logic (every N steps)
```
**Validate**:
- [ ] Search for `target_params` in trainer.py
- [ ] Confirm it's removed from TrainState
- [ ] Confirm no update logic remains
- [ ] Confirm no references remain

#### BUG-017: Missing stop_gradient Before Prediction
**File**: `muzero_model.py` (search for `recurrent_inference` or prediction head)
**Problem**: Gradients from value/policy heads were corrupting dynamics/encoder
**Expected Fix**: Block gradients before prediction heads
```python
def recurrent_inference(self, latent, action):
    next_latent = self.dynamics(latent, action)

    # Block gradients - standard MuZero
    prediction_input = jax.lax.stop_gradient(next_latent)

    value = self.prediction.value(prediction_input)
    policy = self.prediction.policy(prediction_input)
    reward = self.prediction.reward(prediction_input)

    return next_latent, value, policy, reward
```
**Validate**:
- [ ] Find recurrent_inference or dynamics unroll
- [ ] Confirm stop_gradient before prediction heads
- [ ] Check it's applied to the input, not the output

#### BUG-035: Bear Market Burn Sign Error
**File**: `jax_trading_env.py` (search for `market_burn` or opportunity cost)
**Problem**: Negative market returns led to negative burn (became reward via symlog)
**Expected Fix**: Clamp to non-negative
```python
market_burn = jnp.maximum(market_return - risk_free_rate, 0.0)
# In bear market, holding cash has no opportunity cost
```
**Validate**:
- [ ] Find market_burn computation
- [ ] Confirm clamping with jnp.maximum(..., 0.0)
- [ ] Check symlog application is correct

---

### TASK_128 Hang Fixes (6 items)

#### HANG-001: PriceCache Reentrant Lock Deadlock (ROOT CAUSE)
**File**: `price_cache.py` (search for `Lock` or `_lock`)
**Problem**: `threading.Lock()` deadlocks when `add_episode()` calls `flush()` while holding lock
**Expected Fix**: Use RLock
```python
# WRONG:
self._lock = threading.Lock()

# CORRECT:
self._lock = threading.RLock()
```
**Validate**:
- [ ] Find lock initialization in __init__
- [ ] Confirm it's RLock, not Lock
- [ ] Check add_episode and flush both use same lock

#### HANG-002: Path Solver Dynamic Shape JIT
**File**: `path_solver.py` (search for `greedy_lookahead_path` or `max_len`)
**Problem**: Dynamic sequence length T caused JIT recompilation per unique length
**Expected Fix**: Pad to static max_len
```python
@partial(jax.jit, static_argnames=['k', 'lookahead', 'max_len'])
def greedy_lookahead_path(returns, ..., max_len=512):
    T = returns.shape[0]
    padded = jnp.zeros((max_len, n_assets)).at[:T].set(returns)
    # scan over max_len
    # slice results back to T at end
```
**Validate**:
- [ ] Check max_len is in static_argnames
- [ ] Confirm padding to fixed length
- [ ] Confirm slicing output back to T
- [ ] Check DEFAULT_MAX_PATH_LEN constant exists

#### HANG-003: Missing block_until_ready
**File**: `self_play.py` (search for `block_until_ready`)
**Problem**: Device-to-host transfers could hang without explicit sync
**Expected Fix**: Add before host access
```python
jax.block_until_ready(trajectory_batch['rewards'])
# Then do np.array() or other host operations
```
**Validate**:
- [ ] Find block_until_ready call
- [ ] Confirm it's before device-to-host transfers
- [ ] Check it's on a key array (rewards or similar)

#### HANG-004: Periodic Cache Clearing
**File**: `train_jax_fast.py` (search for `clear_caches`)
**Problem**: JAX cache buildup over many epochs
**Expected Fix**: Clear every N epochs
```python
if epoch > 0 and epoch % 50 == 0:
    jax.clear_caches()
```
**Validate**:
- [ ] Find jax.clear_caches() call
- [ ] Confirm it's periodic (every 50 epochs or similar)
- [ ] Check it's in main training loop

#### HANG-005: Replay Buffer JIT Helper
**File**: `replay_buffer.py` (search for `_update_buffer_slice`)
**Problem**: Outside JIT, `.at[].set()` creates full copies
**Expected Fix**: JIT-compiled helper
```python
@jax.jit
def _update_buffer_slice(buffer, indices, values):
    return buffer.at[indices].set(values)
```
**Validate**:
- [ ] Find _update_buffer_slice function
- [ ] Confirm @jax.jit decorator
- [ ] Check it's used for buffer updates

#### HANG-006: Path Solver Output Slicing
**File**: `path_solver.py` (search for `OptimalPath` return)
**Problem**: Initial fix returned padded 512-length arrays, breaking downstream
**Expected Fix**: Slice back to original T
```python
return OptimalPath(
    selections=all_masks[:T],
    indices=all_indices[:T],
    cumulative_pnl=cumulative_pnl[:T],
    ...
)
```
**Validate**:
- [ ] Find OptimalPath return statement
- [ ] Confirm all arrays are sliced to [:T]
- [ ] Check T is captured before padding

---

## Output Format

Create a validation report with this structure:

```markdown
# TASK_130: Bug Fix Validation Report

**Auditor**: [Your name]
**Date**: 2026-02-01
**Status**: [PASS/FAIL/PARTIAL]

## Summary
- Total bugs checked: 16
- Correctly fixed: X
- Issues found: Y
- Not found/unclear: Z

## Detailed Findings

### BUG-001: BC Confidence Masked Mean
**Status**: [PASS/FAIL/NOT FOUND]
**Location**: trainer.py:XXX
**Finding**: [Describe what you found]
**Issue**: [If FAIL, describe the problem]
**Suggested Fix**: [If FAIL, provide code]

[Repeat for each bug...]

## Critical Issues Requiring Immediate Attention
1. ...
2. ...

## Recommendations
1. ...
2. ...
```

---

## Checklist Before Submitting

- [ ] Read all 12 files listed
- [ ] Validated all 16 bug fixes
- [ ] Documented line numbers for each finding
- [ ] Provided code snippets for any issues found
- [ ] Created summary with pass/fail counts
- [ ] Listed critical issues separately

**Write your report to**: `tasks/TASK_130_VALIDATION_REPORT.md`
