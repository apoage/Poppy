# TASK_673: Multipass Oracle for Soft Policy Targets

**Status**: PROPOSED → REVIEWED
**Priority**: HIGH
**Created**: 2026-02-15
**Reviewed**: 2026-02-15
**Blocked By**: None
**Blocks**: Training plateau breakthrough

---

## Problem Statement

Current oracle sees only **one valid path** through the market:
- At each timestep, oracle selects exactly k assets (k=1 currently)
- The model is trained to match that *exact* pick with one-hot targets
- But often 5-10 assets have similar expected returns
- **Result**: Model is punished for picking asset 12 instead of asset 7, even when both are equally profitable

The model needs to learn **"find a profitable path and follow signals"**, not **"replicate this exact path."** When the oracle's choice between two similar assets is essentially a coin flip, that coin flip becomes noise in the gradient.

The 1% oracle trade rate creates extreme class imbalance. Even with adaptive weighting (TASK_149), the model receives sparse and noisy gradient signal on actual trade decisions. Agreement plateaus at ~56%.

---

## Solution: Average Policies, Not Selections

**Key insight**: Run oracle N times with noise → N different binary paths → convert each to a one-hot policy → **average the N one-hot policies** → one truly soft distribution.

This avoids the fatal flaw in the original plan: averaging binary *selections* then passing them through `optimal_path_to_policy` produces garbage because `jnp.where` binarizes any non-zero value. By averaging *after* the policy conversion, each pass produces a correct one-hot policy through the existing state-aware logic, and the average is a proper soft distribution.

### Before (Binary - Single Path)
```
Oracle: "BUY asset 7" → [0, HOLD, BUY, 0, 0] per-asset one-hot
Model picks asset 12 (also profitable) → WRONG, full penalty
```

### After (Soft - Multiple Paths)
```
10 passes with noise:
- 6 say BUY asset 7    → [0, HOLD, BUY, 0, 0] for asset 7
- 3 say BUY asset 12   → [0, HOLD, BUY, 0, 0] for asset 12
- 1 says HOLD all       → [0, HOLD, 0, 0, 0] everywhere

Average: Asset 7 BUY=0.6, Asset 12 BUY=0.3, both HOLD~0.1
Model picks asset 12 → gets partial credit (0.3 vs 0.6 for optimal)
Model picks asset 55 (not profitable) → near-zero credit
```

### Why This Works

1. **Multiple valid paths are captured** - model learns "which assets are tradeable" not "which exact asset"
2. **Gradient signal is proportional to quality** - near-optimal picks get partial credit
3. **Existing loss function works unchanged** - `softmax_cross_entropy(logits, soft_target)` is mathematically equivalent to KL-divergence + constant, which is the correct loss for soft distributions
4. **State-aware logic preserved** - each pass runs the full open/close/maintain transition logic (TASK_141) independently, so directional correctness is maintained per-pass
5. **Ambiguous markets produce ambiguous targets** - when no asset is clearly best, the soft target reflects that uncertainty, teaching the model confidence calibration

---

## Implementation

### File: `mcts_trader/jax_agent/path_solver.py`

Add after `batched_greedy_lookahead_path` (~line 213):

```python
# =============================================================================
# TASK_673: Multipass Oracle → Soft Policy Targets
# =============================================================================

@partial(jax.jit, static_argnames=['k', 'lookahead', 'max_len', 'n_passes',
                                    'n_actions_per_asset'])
def multipass_oracle_policy(
    returns: jnp.ndarray,
    key: jnp.ndarray,
    k: int = 1,
    lookahead: int = 20,
    transaction_cost: float = 0.001,
    min_alpha_threshold: float = 0.001,
    max_len: int = DEFAULT_MAX_PATH_LEN,
    n_passes: int = 10,
    noise_scale: float = 0.1,
    n_actions_per_asset: int = 5,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    TASK_673: Multipass oracle producing genuinely soft policy targets.

    Algorithm:
      1. Run greedy oracle N times, each with Gaussian noise on returns
      2. Convert each binary path → one-hot policy via state-aware logic
      3. Average the N one-hot policies → soft policy distribution

    This preserves the correctness of optimal_path_to_policy per-pass
    (no binarization bugs) while producing soft targets from the ensemble.

    Args:
        returns: (T, n_assets) log returns (REAL, not noisy)
        key: JAX PRNG key
        k: Max assets to hold per step (default 1, matches oracle_k)
        lookahead: Oracle lookahead steps (default 20)
        transaction_cost: Cost per trade as fraction
        min_alpha_threshold: Min return to consider trading
        max_len: Static max sequence length for JIT (TASK_128)
        n_passes: Number of noisy oracle passes (default 10)
        noise_scale: Noise std as fraction of global return std
        n_actions_per_asset: Actions per asset (5: Sell/Hold/Buy/LimitSell/LimitBuy)
        temperature: Softmax temperature for policy conversion

    Returns:
        soft_policy: (T, n_assets * n_actions_per_asset) averaged soft targets
        avg_cumulative_pnl: (T,) mean cumulative PnL across passes
        avg_total_alpha: scalar mean final alpha
    """
    T, n_assets = returns.shape

    # Global noise scaling (matched to overall return magnitude)
    return_std = jnp.maximum(jnp.std(returns), 1e-6)

    def single_pass(subkey):
        # 1. Perturb returns with Gaussian noise
        noise = jax.random.normal(subkey, returns.shape) * noise_scale * return_std
        noisy_returns = returns + noise

        # 2. Solve binary path on noisy returns
        path = greedy_lookahead_path(
            noisy_returns,
            k=k,
            lookahead=lookahead,
            transaction_cost=transaction_cost,
            min_alpha_threshold=min_alpha_threshold,
            max_len=max_len,
        )

        # 3. Compute prev_selections for state-aware oracle (TASK_141)
        #    Uses REAL returns for position direction, path selections for position state
        prev_sels = jnp.zeros((T, n_assets))
        prev_sels = prev_sels.at[1:].set(
            jnp.where(
                path.selections[:-1] > 0.5,
                jnp.where(returns[:-1] > 0, 1.0, -1.0),
                0.0
            )
        )

        # 4. Convert binary path → one-hot policy (state-aware, correct per-pass)
        policy = optimal_path_to_policy(
            path.selections,
            returns,  # REAL returns for action direction
            prev_selections=prev_sels,
            n_actions_per_asset=n_actions_per_asset,
            temperature=temperature,
        )

        return policy, path.cumulative_pnl, path.total_alpha

    # 5. Split key, vmap over all passes
    keys = jax.random.split(key, n_passes)
    all_policies, all_pnls, all_alphas = jax.vmap(single_pass)(keys)
    # all_policies:  (n_passes, T, 320)
    # all_pnls:      (n_passes, T)
    # all_alphas:    (n_passes,)

    # 6. Average → genuinely soft policy distribution
    soft_policy = jnp.mean(all_policies, axis=0)       # (T, 320)
    avg_cumulative_pnl = jnp.mean(all_pnls, axis=0)    # (T,)
    avg_total_alpha = jnp.mean(all_alphas)              # scalar

    return soft_policy, avg_cumulative_pnl, avg_total_alpha


@partial(jax.jit, static_argnames=['k', 'lookahead', 'max_len', 'n_passes',
                                    'n_actions_per_asset'])
def batched_multipass_oracle_policy(
    returns_batch: jnp.ndarray,
    key: jnp.ndarray,
    k: int = 1,
    lookahead: int = 20,
    transaction_cost: float = 0.001,
    min_alpha_threshold: float = 0.001,
    max_len: int = DEFAULT_MAX_PATH_LEN,
    n_passes: int = 10,
    noise_scale: float = 0.1,
    n_actions_per_asset: int = 5,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    TASK_673: Batched multipass oracle for N episodes in parallel.

    Double-vmap: outer over N episodes, inner over n_passes per episode.
    With N=8 episodes and n_passes=10, runs 80 oracle solves in one JIT call.

    Args:
        returns_batch: (N, T, n_assets) batched log returns
        key: JAX PRNG key
        ... (same as multipass_oracle_policy)

    Returns:
        soft_policies: (N, T, n_assets * n_actions_per_asset) batched soft targets
        avg_pnls: (N, T) batched mean cumulative PnL
        avg_alphas: (N,) batched mean final alpha
    """
    N = returns_batch.shape[0]
    keys = jax.random.split(key, N)

    vmapped_fn = jax.vmap(
        lambda r, subkey: multipass_oracle_policy(
            r, subkey,
            k=k, lookahead=lookahead,
            transaction_cost=transaction_cost,
            min_alpha_threshold=min_alpha_threshold,
            max_len=max_len, n_passes=n_passes,
            noise_scale=noise_scale,
            n_actions_per_asset=n_actions_per_asset,
            temperature=temperature,
        )
    )
    return vmapped_fn(returns_batch, keys)
```

**No changes to `optimal_path_to_policy`** - it works correctly on binary inputs per-pass. The softness comes from averaging *after* conversion.

---

### File: `easy_config.py`

Add to `EasyConfig` dataclass (after `state_aware_oracle` field, ~line 58):

```python
    # TASK_673: Multipass oracle for soft policy targets
    oracle_multipass: bool = False         # Disabled by default (backwards compat)
    oracle_n_passes: int = 10              # Number of noisy oracle passes
    oracle_noise_scale: float = 0.1        # Noise as fraction of return std
```

Default `False` so existing configs and checkpoints work unchanged.

---

### File: `train_easy.py`

**Import** (update line 55-58):

```python
from mcts_trader.jax_agent.path_solver import (
    greedy_lookahead_path, optimal_path_to_policy,
    batched_greedy_lookahead_path, batched_optimal_path_to_policy,
    multipass_oracle_policy, batched_multipass_oracle_policy,  # TASK_673
)
```

**Single episode path** - replace lines 228-265 in `_generate_episode_data`:

```python
        # Compute oracle (GPU JIT)
        log_returns = jnp.log(prices_jax[1:] / jnp.maximum(prices_jax[:-1], 1e-8))

        # TASK_673: Multipass oracle (soft targets) or single-pass (hard targets)
        if getattr(self.config, 'oracle_multipass', False):
            self.jax_rng, subkey = jax.random.split(self.jax_rng)
            policy_targets, cumulative_pnl, _ = multipass_oracle_policy(
                log_returns,
                subkey,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
                n_passes=self.config.oracle_n_passes,
                noise_scale=self.config.oracle_noise_scale,
            )
        else:
            optimal_path = greedy_lookahead_path(
                log_returns,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
            )

            # TASK_141: Compute prev_selections for state-aware oracle
            T, n_assets = log_returns.shape
            if getattr(self.config, 'state_aware_oracle', False):
                prev_selections = jnp.zeros((T, n_assets))
                prev_selections = prev_selections.at[1:].set(
                    jnp.where(optimal_path.selections[:-1] > 0.5,
                              jnp.where(log_returns[:-1] > 0, 1.0, -1.0),
                              0.0)
                )
            else:
                prev_selections = None

            policy_targets = optimal_path_to_policy(
                optimal_path.selections,
                log_returns,
                prev_selections=prev_selections,
                n_actions_per_asset=5,
                temperature=1.0,
            )
            cumulative_pnl = optimal_path.cumulative_pnl

        # TASK_670 FIX: Compute REMAINING PnL (future value), not cumulative
        remaining_pnl = cumulative_pnl[-1] - cumulative_pnl
```

Note: When `oracle_multipass=True`, the state-aware logic runs inside `multipass_oracle_policy` per-pass (always enabled, hardcoded). The `state_aware_oracle` config flag only affects the `else` branch.

**Batched episode path** - replace lines 328-364 in `_generate_batch_episodes`:

```python
        # Compute log returns (N, T-1, n_assets)
        log_returns = jnp.log(prices_jax[:, 1:] / jnp.maximum(prices_jax[:, :-1], 1e-8))

        # TASK_673: Multipass oracle (soft targets) or single-pass (hard targets)
        if getattr(self.config, 'oracle_multipass', False):
            self.jax_rng, subkey = jax.random.split(self.jax_rng)
            policy_targets, cumulative_pnl_batch, _ = batched_multipass_oracle_policy(
                log_returns,
                subkey,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
                n_passes=self.config.oracle_n_passes,
                noise_scale=self.config.oracle_noise_scale,
            )
            # policy_targets: (N, T, 320) - already soft policy
            # cumulative_pnl_batch: (N, T)
        else:
            # Batched oracle - single GPU call for all N episodes!
            optimal_paths = batched_greedy_lookahead_path(
                log_returns,
                k=self.config.oracle_k,
                lookahead=self.config.oracle_lookahead,
                transaction_cost=self.config.transaction_cost,
                min_alpha_threshold=getattr(self.config, 'min_alpha_threshold', 0.0),
            )

            # TASK_141: Compute prev_selections for state-aware oracle (batched)
            N, T_log, n_assets_batch = log_returns.shape
            if getattr(self.config, 'state_aware_oracle', False):
                prev_selections_batch = jnp.zeros((N, T_log, n_assets_batch))
                prev_selections_batch = prev_selections_batch.at[:, 1:, :].set(
                    jnp.where(optimal_paths.selections[:, :-1, :] > 0.5,
                              jnp.where(log_returns[:, :-1, :] > 0, 1.0, -1.0),
                              0.0)
                )
            else:
                prev_selections_batch = None

            policy_targets = batched_optimal_path_to_policy(
                optimal_paths.selections,
                log_returns,
                prev_selections_batch=prev_selections_batch,
                n_actions_per_asset=5,
                temperature=1.0,
            )
            cumulative_pnl_batch = optimal_paths.cumulative_pnl

        # TASK_670 FIX: Compute REMAINING PnL (future value), not cumulative
        remaining_pnl = cumulative_pnl_batch[:, -1:] - cumulative_pnl_batch  # (N, T)
```

---

## Shape Verification

Traced through the actual code, all shapes check out:

```
Input:
  returns:         (T, 64)     T ≈ 399 (episode_length - 1)
  key:             (2,)

single_pass(subkey):
  noise:           (T, 64)     gaussian * noise_scale * return_std
  noisy_returns:   (T, 64)     returns + noise
  path.selections: (T, 64)     binary {0.0, 1.0} from greedy_lookahead_path
  prev_sels:       (T, 64)     {-1.0, 0.0, 1.0} state-aware positions
  policy:          (T, 320)    per-asset softmax, near one-hot per pass

vmap(single_pass)(keys):       keys = (10, 2)
  all_policies:    (10, T, 320)
  all_pnls:        (10, T)
  all_alphas:      (10,)

Output:
  soft_policy:     (T, 320)    mean → genuinely soft distribution
  avg_pnl:         (T,)
  avg_alpha:       scalar

Batched (N=8):
  returns_batch:   (8, T, 64)
  Double vmap: 8 × 10 = 80 oracle solves per JIT call
  soft_policies:   (8, T, 320)
  avg_pnls:        (8, T)
  avg_alphas:      (8,)
```

---

## Memory & Compute Budget

### Per Episode (single_pass vmapped)
```
Intermediate per pass:
  noisy_returns:    T × 64 × 4B = 100 KB
  scan state:       max_len × 64 × 4B = 128 KB (padded to 512)
  policy:           T × 320 × 4B = 500 KB

× 10 passes (vmapped, all live simultaneously):
  ~7 MB per episode

Output (averaged):
  soft_policy:      T × 320 × 4B = 500 KB
```

### Batched (8 episodes × 10 passes)
```
Peak VRAM: ~60 MB (80 parallel oracle solves)
Acceptable: Current training uses 2-4 GB VRAM
```

### Compute
```
Single oracle:    ~1-2ms (T=400, n=64, k=1, lookahead=20)
10 passes vmap:   ~5-10ms (parallelized by XLA)
8 episodes vmap:  ~15-30ms (double vmap)
Current budget:   ~100ms per refill (oracle is not the bottleneck)
```

### JIT Compilation
```
First call: 30-90s (nested vmap + scan, XLA traces deeply)
Cached in .xla_cache, subsequent calls instant
n_passes is static_argname: changing it triggers recompilation
→ Use fixed n_passes=10, never change during training
```

---

## Downstream Compatibility

### Loss Function (easy_trainer.py:142-176) - NO CHANGES NEEDED

```python
# Current loss: per-asset softmax cross-entropy
per_asset_ce = optax.softmax_cross_entropy(policy_reshaped, targets_reshaped)
```

`softmax_cross_entropy(logits, soft_target)` = `KL(soft_target || softmax(logits))` + `H(soft_target)`

The entropy term `H(soft_target)` is constant w.r.t. parameters → same gradients as KL divergence. This is exactly the right loss for soft targets. No changes needed.

### TASK_149 Adaptive Weighting (easy_trainer.py:154-176) - MINOR CONSIDERATION

```python
is_trade = targets_reshaped[:, :, 1] < 0.5  # NOT HOLD
```

With soft targets, `targets_reshaped[:, :, 1]` is now continuous (e.g., 0.3 instead of 0.0 or 0.997). The `< 0.5` threshold still works: if HOLD probability < 0.5, it's classified as a trade signal. This correctly captures the multipass consensus:
- 10/10 passes say HOLD → prob_hold ≈ 0.997 → `< 0.5` = False → classified as HOLD
- 6/10 passes say BUY → prob_hold ≈ 0.4 → `< 0.5` = True → classified as TRADE
- 3/10 passes say BUY → prob_hold ≈ 0.7 → `< 0.5` = False → classified as HOLD (weak signal, correct)

### Agreement Metric (easy_trainer.py:194-216) - NO CHANGES NEEDED

```python
pred_action_type = jnp.argmax(policy_reshaped, axis=-1)     # (batch, 64)
oracle_action_type = jnp.argmax(targets_reshaped, axis=-1)  # (batch, 64)
```

`argmax` of soft target = the most-frequently-selected action across passes. Agreement measures "did the model pick the consensus action?" This is the right metric for soft targets — if 6/10 passes say BUY and the model says BUY, that's correct.

### Value Head (remaining_pnl) - HANDLED

Multipass returns averaged `cumulative_pnl` → averaged `remaining_pnl` = expected future value under ensemble uncertainty. This is the correct target for the value head.

---

## Testing Plan

### Unit Test 1: Soft policies are genuinely soft

```python
def test_multipass_produces_soft_policies():
    """Averaged policies should NOT be one-hot."""
    T, n_assets = 100, 64
    returns = jax.random.normal(jax.random.PRNGKey(0), (T, n_assets)) * 0.01
    key = jax.random.PRNGKey(42)

    soft_policy, _, _ = multipass_oracle_policy(
        returns, key, k=1, lookahead=20, n_passes=10
    )

    # Shape check
    assert soft_policy.shape == (T, n_assets * 5)

    # Valid probability distribution per asset
    reshaped = soft_policy.reshape(T, n_assets, 5)
    assert jnp.allclose(jnp.sum(reshaped, axis=-1), 1.0, atol=1e-5)

    # Should have intermediate values (not all near 0 or 1)
    max_probs = jnp.max(reshaped, axis=-1)  # (T, 64)
    # At least some assets should have max_prob < 0.95 (disagreement between passes)
    assert jnp.any(max_probs < 0.95)
```

### Unit Test 2: Deterministic with same key

```python
def test_multipass_deterministic():
    T, n_assets = 50, 64
    returns = jax.random.normal(jax.random.PRNGKey(0), (T, n_assets)) * 0.01
    key = jax.random.PRNGKey(42)

    p1, _, _ = multipass_oracle_policy(returns, key, k=1, n_passes=10)
    p2, _, _ = multipass_oracle_policy(returns, key, k=1, n_passes=10)
    assert jnp.allclose(p1, p2)
```

### Unit Test 3: Different keys produce different results

```python
def test_multipass_varies_with_key():
    T, n_assets = 50, 64
    returns = jax.random.normal(jax.random.PRNGKey(0), (T, n_assets)) * 0.01

    p1, _, _ = multipass_oracle_policy(returns, jax.random.PRNGKey(1), k=1, n_passes=10)
    p2, _, _ = multipass_oracle_policy(returns, jax.random.PRNGKey(2), k=1, n_passes=10)
    assert not jnp.allclose(p1, p2)
```

### Unit Test 4: Batched version matches single

```python
def test_batched_matches_single():
    T, n_assets = 50, 64
    r1 = jax.random.normal(jax.random.PRNGKey(0), (T, n_assets)) * 0.01
    r2 = jax.random.normal(jax.random.PRNGKey(1), (T, n_assets)) * 0.01
    batch = jnp.stack([r1, r2])

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 2)

    # Single
    p1, _, _ = multipass_oracle_policy(r1, keys[0], k=1, n_passes=10)
    p2, _, _ = multipass_oracle_policy(r2, keys[1], k=1, n_passes=10)

    # Batched
    bp, _, _ = batched_multipass_oracle_policy(batch, key, k=1, n_passes=10)

    assert jnp.allclose(p1, bp[0], atol=1e-5)
    assert jnp.allclose(p2, bp[1], atol=1e-5)
```

### Unit Test 5: High noise → more disagreement, zero noise → hard targets

```python
def test_noise_scale_effect():
    T, n_assets = 50, 64
    returns = jax.random.normal(jax.random.PRNGKey(0), (T, n_assets)) * 0.01
    key = jax.random.PRNGKey(42)

    # Zero noise: all passes identical → near one-hot output
    p_hard, _, _ = multipass_oracle_policy(returns, key, k=1, n_passes=10, noise_scale=0.0)
    reshaped_hard = p_hard.reshape(T, n_assets, 5)
    assert jnp.mean(jnp.max(reshaped_hard, axis=-1)) > 0.98

    # High noise: passes disagree → softer output
    p_soft, _, _ = multipass_oracle_policy(returns, key, k=1, n_passes=10, noise_scale=0.5)
    reshaped_soft = p_soft.reshape(T, n_assets, 5)
    assert jnp.mean(jnp.max(reshaped_soft, axis=-1)) < jnp.mean(jnp.max(reshaped_hard, axis=-1))
```

### Integration Test: Training loop with soft targets

```python
def test_training_converges_with_multipass():
    """Full training loop works with soft targets, loss decreases."""
    config = EasyConfig(oracle_multipass=True, oracle_n_passes=10, epochs=5)
    config.name = "test_multipass"
    trainer = create_trainer(config)

    losses = []
    for epoch in range(5):
        metrics = train_epoch(trainer, epoch)
        losses.append(metrics['total_loss'])
        assert jnp.isfinite(metrics['total_loss'])

    # Loss should decrease (not plateau immediately)
    assert losses[-1] < losses[0]
```

---

## Config Defaults & Presets

```python
# Default: disabled (backwards compatible)
oracle_multipass: bool = False

# When enabled:
oracle_n_passes: int = 10     # 10 is enough (diminishing returns beyond ~10)
oracle_noise_scale: float = 0.1  # 10% of return std

# e-dojo preset override (in orchestrator config):
# oracle_multipass=True, oracle_n_passes=10, oracle_noise_scale=0.1
```

---

## Rollout Plan

### Phase 1: Implementation (1 file + 1 config change + 1 integration)
1. Add `multipass_oracle_policy` + `batched_multipass_oracle_policy` to `path_solver.py`
2. Add 3 config fields to `easy_config.py`
3. Update `train_easy.py` two code paths (single + batched) with `if oracle_multipass` gate
4. Update import line in `train_easy.py`

### Phase 2: Validation
1. Run unit tests (5 tests above)
2. Smoke test: `python train_easy.py --preset small --epochs 5` with multipass on
3. Compare single-pass vs multipass soft target distributions visually (spot check)
4. Verify JIT compilation succeeds and is cached on second run

### Phase 3: A/B Training Experiment
1. Run 500 epochs with `oracle_multipass=False` (baseline)
2. Run 500 epochs with `oracle_multipass=True` (experiment)
3. Compare: agreement curves, agree_trade, agree_hold, loss curves
4. If multipass wins: enable by default, update presets

---

## Success Criteria

1. **Valid soft distributions**: Per-asset probabilities sum to 1.0, intermediate values exist
2. **No training regression**: Loss converges at least as fast as baseline
3. **Improved agreement**: Break 56% balanced accuracy plateau, target 60%+
4. **Better trade precision**: Model picks *any* profitable asset, not just the oracle's arbitrary pick
5. **Compute overhead**: <2× slowdown on oracle step (target: 10× oracle, still <30ms)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Noise too high → model learns noise | Training degradation | Start noise_scale=0.1, tune down. Test 5 above catches this. |
| JIT compilation slow on first call | 30-90s startup delay | One-time cost, cached in .xla_cache |
| Double vmap memory explosion | OOM on batched path | N=8 × n_passes=10 = 80 passes × ~0.7MB each ≈ 60MB. Safe. |
| Soft targets confuse model early | Slow initial learning | Can disable with `oracle_multipass=False`, revert any time |
| Adaptive weight (TASK_149) interacts poorly | Trade weight miscalibrated | The `< 0.5` threshold on soft hold_prob works correctly (see Downstream Compatibility) |

---

## Future Extensions (Not in scope)

- **Noise annealing**: Start with `noise_scale=0.2` (explore), decay to `0.05` (exploit) over training. Would need curriculum integration.
- **Per-asset noise scaling**: Use `jnp.std(returns, axis=0)` instead of global std. Prevents volatile assets from dominating.
- **Variable lookahead per pass**: Different passes use lookahead=10/15/20/30. Captures multi-horizon signals. Requires `lookahead` to not be `static_argname` (perf impact).
- **Continuous agreement metric**: Track `mean(soft_target[argmax(prediction)])` as "how much credit the oracle gives the model's pick." Better signal than binary argmax agreement.

---

## References

- TASK_141: State-aware oracle (open/close/maintain logic) - preserved per-pass
- TASK_149: Adaptive class balancing - compatible with soft targets
- TASK_151: Weight warmup ramping - unaffected
- TASK_128: Static max_len JIT fix - used inside greedy_lookahead_path per-pass
- TASK_670: Remaining PnL value targets - uses averaged PnL from multipass
- Hinton et al. 2015: "Distilling the Knowledge in a Neural Network" - soft targets generalize better
