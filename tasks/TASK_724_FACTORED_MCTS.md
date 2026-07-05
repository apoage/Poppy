# TASK_724: Factored Per-Asset Search — Replace Joint 320-Action MCTS

**Status**: IMPLEMENTED
**Priority**: HIGH — structural fix for action space mismatch
**Depends on**: TASK_723 (reward scaling, discount match, BC floor)
**Blocks**: Bull market alpha improvement beyond what TASK_723 achieves

---

## Problem Statement

Foxglove uses Gumbel MuZero MCTS over 320 joint actions (64 assets × 5 action types). Even with TASK_723's `max_considered_actions=64`, MCTS treats "BUY asset 7" and "SELL asset 23" as competing alternatives in a single distribution. But per-asset decisions are mostly independent — the oracle selects k=4 assets to trade, with each asset's action chosen independently.

With 320 joint actions and 64 considered, MCTS explores 20% of the action space. It cannot discover the optimal 4-asset combination (C(64,4) ≈ 635K possibilities). The search is fundamentally mismatched to the factored structure of the problem.

**Solution**: Replace joint MCTS with factored per-asset evaluation. For each of 64 assets, evaluate all 5 actions independently using 1-step dynamics lookahead + value prediction. This gives every asset's every action a fair evaluation — 100% coverage of the per-asset action space.

---

## Design Overview

### Current Flow (Joint MCTS)
```
obs → initial_inference → latent_t, policy(320), value_t
     → mctx.gumbel_muzero(320 actions, 64 sims, 64 considered)
     → visit_counts(320)  [global distribution, sums to 1.0]
     → policy CE: softmax_cross_entropy(policy_logits(320), visits(320))
```

### New Flow (Factored Per-Asset Search)
```
obs → initial_inference → latent_t, policy(320), value_t
     → for each asset i, action a: Q(i,a) = R(i,a) + γ * V(next_state(i,a))
       (batched as one forward pass of B×320 recurrent_inference calls)
     → per-asset softmax: π_target(i,:) = softmax(Q(i,:) / temperature)
     → policy CE: per_asset_softmax_cross_entropy(policy(64,5), targets(64,5))
```

### Key Differences
| Aspect | Joint MCTS | Factored Search |
|--------|-----------|-----------------|
| Actions evaluated | 64 of 320 (20%) | 320 of 320 (100%) |
| Per-asset coverage | ~1 action/asset | All 5 actions/asset |
| Search depth | 1-4 steps (tree) | 1 step (lookahead) |
| Cross-asset interaction | Implicit in joint action | Independent (assumes others HOLD) |
| Policy loss | Global softmax CE (320) | Per-asset softmax CE (64×5) |
| Compute | ~38s/epoch (64 sims) | ~76s/epoch (full reasoning) or ~8s (value-head only) |

---

## Architecture: `FactoredPerAssetSearch`

### New File: `mcts_trader/jax_agent/factored_search.py`

```python
"""
TASK_724: Factored per-asset search for trading action selection.

Instead of running MCTS over 320 joint actions, evaluates each asset's
5 actions independently via 1-step dynamics lookahead + value prediction.

Key insight: per-asset decisions are mostly independent. The oracle selects
k=4 assets to trade, with each asset's action chosen based on its own
expected return. Joint MCTS can't discover this factored structure.

Interface matches MuZeroMCTXSearch for drop-in replacement in train_phase2.py.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any
from functools import partial


class FactoredPerAssetSearch:
    """
    Per-asset 1-step lookahead search.

    For each (asset, action) pair:
        1. Compose joint action index: j = asset * 5 + action
        2. Run dynamics: next_latent, reward = dynamics(latent, j)
        3. Run value prediction on next_latent
        4. Q(asset, action) = reward + discount * value(next_latent)

    Returns per-asset softmax policy targets (B, 320) for training,
    same interface as MuZeroMCTXSearch.
    """

    def __init__(
        self,
        model_apply_fn: Callable,
        n_assets: int = 64,
        n_actions_per_asset: int = 5,
        discount: float = 0.95,
        temperature: float = 1.0,
        epsilon_greedy: float = 0.0,
        sub_batch_size: int = 4096,
        value_scale: float = 1.0,
        value_shift: float = 0.0,
    ):
        """
        Args:
            model_apply_fn: Model's apply function (MuZeroNetworks.apply)
            n_assets: Number of tradeable assets (64)
            n_actions_per_asset: Actions per asset (5: Sell, Hold, Buy, LimitSell, LimitBuy)
            discount: Reward discount (MUST match td_discount from training)
            temperature: Softmax temperature for Q → policy conversion.
                         1.0 = proportional to Q-values. Lower = sharper. Higher = softer.
            epsilon_greedy: Mix uniform into training target (0.0 = pure Q-values)
            sub_batch_size: Max elements per forward pass (VRAM management).
                           With state_dim=128, H=6, L=8: ~670MB per 4096 elements.
            value_scale: PopArt std for value denormalization (set per-epoch)
            value_shift: PopArt mean for value denormalization (set per-epoch)
        """
        self.model_apply_fn = model_apply_fn
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset
        self.n_total_actions = n_assets * n_actions_per_asset  # 320
        self.discount = discount
        self.temperature = temperature
        self.epsilon_greedy = epsilon_greedy
        self.sub_batch_size = sub_batch_size
        self.value_scale = value_scale
        self.value_shift = value_shift

        # Pre-compute action indices: for each asset i, actions are [i*5, i*5+1, ..., i*5+4]
        # Shape: (320,) — maps flat index to joint action index (identity for our encoding)
        self._all_actions = jnp.arange(self.n_total_actions, dtype=jnp.int32)

        # JIT-compile the search
        self._search_fn = jax.jit(self._search, static_argnames=['self'])

    def _compute_q_values(
        self,
        params: dict,
        latent: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute Q-values for all (asset, action) pairs via 1-step lookahead.

        For each of 320 joint actions:
            next_state, reward = dynamics(latent, action)
            value = prediction(next_state).value  (includes full H×L reasoning)
            Q = reward + discount * value

        Args:
            params: Model parameters
            latent: (B, state_dim) latent states from initial_inference

        Returns:
            q_values: (B, n_assets, n_actions_per_asset) Q-values
            rewards: (B, n_assets, n_actions_per_asset) immediate rewards
        """
        from . import get_value_support
        support = get_value_support()  # linspace(-10, 10, 401)

        B = latent.shape[0]
        n_total = self.n_total_actions  # 320

        # Expand latent: (B, dim) → (B, 320, dim) → (B*320, dim)
        latent_expanded = jnp.repeat(latent, n_total, axis=0)  # (B*320, dim)

        # Action indices: [0,1,...,319, 0,1,...,319, ...] for B repetitions
        actions_expanded = jnp.tile(self._all_actions, B)  # (B*320,)

        # --- Sub-batched recurrent_inference ---
        # Full recurrent_inference = dynamics + reasoning(H×L) + value_head
        # With B=256: B*320 = 81920 forward passes. Sub-batch for VRAM.
        total = B * n_total
        sub_batch = self.sub_batch_size

        all_rewards = []
        all_values = []

        for start in range(0, total, sub_batch):
            end = min(start + sub_batch, total)
            lat_chunk = latent_expanded[start:end]
            act_chunk = actions_expanded[start:end]

            out = self.model_apply_fn(
                {'params': params},
                lat_chunk,       # observation arg (mapped to latent_state in recurrent mode)
                act_chunk,       # action arg
                mode='recurrent_inference'
            )

            # Reward: categorical → scalar (NO PopArt, rewards are not normalized)
            reward_probs = jax.nn.softmax(out['reward'], axis=-1)
            reward_scalar = jnp.sum(reward_probs * support, axis=-1)  # (chunk,)
            all_rewards.append(reward_scalar)

            # Value: categorical → scalar + PopArt denormalization
            value_probs = jax.nn.softmax(out['value'], axis=-1)
            value_raw = jnp.sum(value_probs * support, axis=-1)  # (chunk,)
            value_denorm = value_raw * self.value_scale + self.value_shift
            all_values.append(value_denorm)

        rewards_flat = jnp.concatenate(all_rewards)   # (B*320,)
        values_flat = jnp.concatenate(all_values)      # (B*320,)

        # Q = R + γ * V
        q_flat = rewards_flat + self.discount * values_flat  # (B*320,)

        # Reshape: (B*320,) → (B, 64, 5)
        q_values = q_flat.reshape(B, self.n_assets, self.n_actions_per_asset)
        rewards = rewards_flat.reshape(B, self.n_assets, self.n_actions_per_asset)

        return q_values, rewards

    def _q_to_policy(
        self,
        q_values: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Convert per-asset Q-values to policy targets via temperature-scaled softmax.

        Centers Q-values per asset (subtract mean) before softmax for numerical
        stability and scale independence. This ensures the softmax operates on
        the relative advantage of each action, not absolute Q-value magnitude.

        Args:
            q_values: (B, n_assets, 5) Q-values

        Returns:
            policy_targets: (B, 320) per-asset softmax probabilities
                           (each group of 5 sums to 1.0, total sums to 64.0)
        """
        # Center per asset — softmax only sees advantage over mean
        q_centered = q_values - q_values.mean(axis=-1, keepdims=True)

        # Temperature-scaled softmax per asset
        targets = jax.nn.softmax(q_centered / self.temperature, axis=-1)  # (B, 64, 5)

        # ε-greedy mixing (same as MuZeroMCTXSearch)
        if self.epsilon_greedy > 0:
            uniform = jnp.ones_like(targets) / self.n_actions_per_asset  # 1/5 uniform
            targets = (1 - self.epsilon_greedy) * targets + self.epsilon_greedy * uniform

        # Flatten: (B, 64, 5) → (B, 320)
        return targets.reshape(-1, self.n_total_actions)

    def _search(
        self,
        params: dict,
        rng: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """
        Run factored per-asset search.

        Same interface as MuZeroMCTXSearch._search for drop-in replacement.

        Args:
            params: Model parameters
            rng: JAX random key (unused — search is deterministic given params)
            observation: (B, C, W, N) encoded observations

        Returns:
            actions: (B,) selected joint actions (argmax of per-asset Q-advantages)
            policy_probs: (B, 320) per-asset policy targets for training
            search_stats: dict with diagnostic metrics
        """
        # 1. Initial inference — get latent state and root value
        root_out = self.model_apply_fn(
            {'params': params},
            observation,
            method='initial_inference'
        )
        latent = root_out['latent_state']         # (B, state_dim)
        root_value_logits = root_out['value']      # (B, 401)

        # Root value (PopArt denormalized) — same as MuZeroMCTXSearch
        from . import get_value_support
        support = get_value_support()
        root_value_raw = jnp.sum(jax.nn.softmax(root_value_logits, axis=-1) * support, axis=-1)
        root_value = root_value_raw * self.value_scale + self.value_shift

        # 2. Compute Q-values for all 320 (asset, action) pairs
        q_values, rewards = self._compute_q_values(params, latent)  # (B, 64, 5)

        # 3. Convert Q-values to per-asset policy targets
        policy_probs = self._q_to_policy(q_values)  # (B, 320)

        # 4. Select actions: per-asset argmax of Q-advantage, then compose joint action
        q_advantage = q_values - q_values[:, :, 1:2]  # Advantage over HOLD (action 1)
        best_per_asset = jnp.argmax(q_values, axis=-1)  # (B, 64) — best action per asset
        # Joint action = first asset's best action (for stats; actual training uses policy_probs)
        # More useful: asset with highest advantage gets its action selected
        max_advantage_asset = jnp.argmax(jnp.max(q_advantage, axis=-1), axis=-1)  # (B,)
        actions = max_advantage_asset * self.n_actions_per_asset + \
                  best_per_asset[jnp.arange(latent.shape[0]), max_advantage_asset]

        # 5. Diagnostics
        search_stats = {
            'root_value': root_value,
            'q_mean': jnp.mean(q_values),
            'q_std': jnp.std(q_values),
            'q_advantage_mean': jnp.mean(jnp.max(q_advantage, axis=-1)),
            'reward_mean': jnp.mean(rewards),
            'n_active_assets': jnp.mean(jnp.sum(jnp.argmax(q_values, axis=-1) != 1, axis=-1)),
        }

        return actions, policy_probs, search_stats

    def __call__(
        self,
        params: dict,
        rng: jnp.ndarray,
        observation: jnp.ndarray,
        action_mask: jnp.ndarray = None,   # Ignored — factored search doesn't use masks
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Convenience method matching MuZeroMCTXSearch interface."""
        return self._search_fn(params, rng, observation)
```

---

## Critical Implementation Details

### 1. Reward Handling — No PopArt, No /100

The reward from `recurrent_inference` is categorical logits over [-10, 10]. Convert to scalar via softmax + dot product with support. Do NOT apply PopArt (rewards are not normalized) and do NOT divide by 100 (TASK_723 Fix 1).

```python
# CORRECT (in _compute_q_values):
reward_probs = jax.nn.softmax(out['reward'], axis=-1)
reward_scalar = jnp.sum(reward_probs * support, axis=-1)  # Raw, no PopArt

# WRONG (old MCTS had both of these bugs):
# reward_scalar = self._categorical_to_scalar(output['reward']) / 100.0
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ applies PopArt AND /100
```

### 2. Value Handling — Yes PopArt

The value head outputs PopArt-normalized categorical logits. Convert to scalar, then denormalize:

```python
value_raw = jnp.sum(value_probs * support, axis=-1)    # Normalized scalar
value_denorm = value_raw * self.value_scale + self.value_shift  # Real scale
```

`value_scale` and `value_shift` are set per-epoch from PopArt state in `train_stage_b`, same as current MCTS.

### 3. Discount — Must Match td_discount

The `discount` parameter MUST be `config.td_discount` (0.95 after TASK_723), NOT `mcts_discount`. The Q-value formula `Q = R + γ * V` requires the same γ that the value function was trained with.

### 4. Sub-Batching for VRAM

With B=256, `recurrent_inference` is called on B×320 = 81,920 latent states. Each goes through full H=6, L=8 reasoning with hidden_dim=2048.

Peak VRAM per sub-batch element: ~160KB (latent + hidden activations + intermediates).
- Sub-batch of 4096: ~656MB peak activation VRAM (safe on RTX 3090)
- Sub-batch of 8192: ~1.3GB (still safe, faster)
- Sub-batch of 16384: ~2.6GB (pushing it with other allocations)

Default `sub_batch_size=4096` is conservative. Increase if VRAM headroom allows.

### 5. Per-Asset Policy Targets

The output `policy_probs` has shape (B, 320) but is structured as 64 groups of 5 that each sum to 1.0. The TOTAL sum is 64.0, NOT 1.0. This is different from MCTS visit distributions (which sum to 1.0 globally).

This requires changing the **policy loss** in `make_stage_b_train_step` from global softmax CE to per-asset softmax CE. See Fix section below.

### 6. JIT Compilation

The `_search` method uses `jax.jit` with `static_argnames=['self']`. The sub-batching loop inside `_compute_q_values` uses Python `range()` — this means the loop is unrolled at trace time. If `B * 320 / sub_batch_size` varies between calls, it triggers recompilation. To avoid this, ensure batch size is constant across epochs (it is — `stage_b_batch_size=256` is fixed).

**Potential issue**: The `for start in range(0, total, sub_batch)` loop has `total = B * 320`. With B=256 and sub_batch=4096: `81920/4096 = 20` iterations. This is a fixed number for constant B, so JIT traces once and caches.

If sub_batch_size doesn't evenly divide total, the last chunk is smaller. `model.apply` handles variable batch sizes, and JAX will pad/mask internally. This may cause one extra trace for the smaller chunk. To avoid: set `sub_batch_size` to a divisor of `B * 320`. With B=256: `81920 / 4096 = 20` (exact). With B=128: `40960 / 4096 = 10` (exact). Good.

### 7. `method='initial_inference'` vs Positional Args

In MuZeroNetworks.__call__, when `mode='recurrent_inference'`, the first positional arg `observation` is mapped to `latent_state`:
```python
elif mode == 'recurrent_inference':
    return self.recurrent_inference(observation, action)
```

So calling `model.apply({'params': params}, latent_chunk, act_chunk, mode='recurrent_inference')` correctly passes `latent_chunk` as `latent_state` and `act_chunk` as `action`. This is the same pattern used by `mctx_search.py` line 191.

---

## Changes to Training Loop

### File: `train_phase2.py` — `train_stage_b()`

Replace MCTS creation (around line 980) with a conditional:

```python
# After TASK_723 fixes are applied:
if config.use_factored_search:
    from mcts_trader.jax_agent.factored_search import FactoredPerAssetSearch
    search = FactoredPerAssetSearch(
        model_apply_fn=model.apply,
        n_assets=config.n_assets,
        discount=config.td_discount,          # MUST match value training discount
        temperature=config.factored_temperature,
        epsilon_greedy=config.epsilon_greedy,
        sub_batch_size=config.factored_sub_batch,
    )
    logger.info(f"  Factored per-asset search (T={config.factored_temperature}, "
                f"ε={config.epsilon_greedy}, sub_batch={config.factored_sub_batch})")
else:
    from mcts_trader.jax_agent.mctx_search import MuZeroMCTXSearch
    search = MuZeroMCTXSearch(
        model_apply_fn=model.apply,
        num_simulations=config.mcts_sims,
        max_num_considered_actions=config.max_considered_actions,
        discount=config.mcts_discount,
        use_gumbel=config.use_gumbel,
        epsilon_greedy=config.epsilon_greedy,
    )
```

The per-epoch PopArt update (line 1073-1075) stays the same — both search types use `value_scale` and `value_shift`:

```python
if popart_state is not None:
    search.value_scale = float(jnp.sqrt(popart_state.var + 1e-8))
    search.value_shift = float(popart_state.mean)
```

The MCTS call (line 1094) stays the same — same interface:

```python
_actions, visit_counts, stats = search(params, search_key, obs)
```

### File: `train_phase2.py` — `make_stage_b_train_step()`

**Change the policy loss from global softmax CE to per-asset softmax CE.**

This is critical: factored search targets are per-asset (each group of 5 sums to 1.0), not global (all 320 sum to 1.0). Using global CE would treat all 320 actions as competing alternatives, destroying the factored structure.

```python
# BEFORE (global CE, around line 735):
policy_loss = optax.softmax_cross_entropy(policy_logits, mcts_visits).mean()

# AFTER (per-asset CE):
if use_factored_search:
    # Per-asset CE: each asset's 5 actions are an independent classification
    policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
    target_reshaped = mcts_visits.reshape(-1, n_assets, n_actions_per_asset)
    policy_loss = optax.softmax_cross_entropy(policy_reshaped, target_reshaped).mean()
else:
    # Joint CE for traditional MCTS (visits sum to 1.0 globally)
    policy_loss = optax.softmax_cross_entropy(policy_logits, mcts_visits).mean()
```

**Important**: The BC loss already uses per-asset CE (line 790-792). So with factored search, both policy loss and BC loss use the same per-asset structure. This is correct — they should be measuring the same thing.

**Also important**: `use_factored_search` must be captured in the JIT closure (via `config.use_factored_search`) as a static boolean. Since it doesn't change during training, it won't cause recompilation.

---

## Changes to Config

### File: `phase2_config.py`

Add these fields to `Phase2Config`:

```python
# === Factored Per-Asset Search (TASK_724) ===
use_factored_search: bool = False  # True = factored, False = joint MCTS (default for compat)
factored_temperature: float = 1.0  # Q-value → policy softmax temperature
factored_sub_batch: int = 4096     # Sub-batch size for VRAM management
```

### File: `train_phase2.py` — `parse_args()`

Add CLI args:

```python
# Factored search (TASK_724)
parser.add_argument('--factored', action='store_true',
                    help='Use factored per-asset search instead of joint MCTS (TASK_724)')
parser.add_argument('--factored-temperature', type=float, default=None,
                    help='Q-value softmax temperature (default: 1.0)')
parser.add_argument('--factored-sub-batch', type=int, default=None,
                    help='Sub-batch size for factored search (default: 4096)')
```

And in the override section:

```python
if args.factored:
    config.use_factored_search = True
if args.factored_temperature is not None:
    config.factored_temperature = args.factored_temperature
if args.factored_sub_batch is not None:
    config.factored_sub_batch = args.factored_sub_batch
```

---

## Compute Budget

### Per-Step Cost (B=256)

| Component | Joint MCTS (64 sims) | Factored (full reasoning) | Factored (value-head only) |
|-----------|---------------------|--------------------------|---------------------------|
| Initial inference | 1× (B=256) | 1× (B=256) | 1× (B=256) |
| Recurrent calls | 256×64 = 16,384 | 256×320 = 81,920 (20 sub-batches) | 256×320 = 81,920 |
| Per-call cost | dynamics + full reasoning | dynamics + full reasoning | dynamics + value head only |
| Estimated time | ~95ms | ~200ms | ~10ms |

### Per-Epoch Cost (400 steps)

| Mode | Time/epoch | vs Current MCTS (38s) |
|------|-----------|----------------------|
| Joint MCTS (64 sims, TASK_723) | ~38s | 1.0× |
| Factored (full reasoning, sub=4096) | ~80s | 2.1× |
| Factored (value-head only) | ~4s | 0.1× |

**Recommendation**: Start with full reasoning (accurate Q-values). If too slow, add a `factored_fast: bool = True` option that skips reasoning and applies the value head directly to dynamics-predicted latents. The value head was trained on reasoning-refined latents, so accuracy degrades — but 1-step lookahead may be forgiving enough.

### VRAM Budget

Peak VRAM per sub-batch (full reasoning, state_dim=128, hidden_dim=2048, H=6, L=8):
- Latent states: 4096 × 128 × 4B = 2.1MB
- Reasoning hidden: 4096 × 2048 × 4B = 33.6MB
- Multiple intermediates (~6 active tensors): ~200MB
- Gradient-free (inference): no backward pass storage

Total per sub-batch: ~200MB. With 20 sub-batches processed sequentially, peak is ~200MB (not cumulative). Safe on RTX 3090 (24GB).

---

## Testing & Validation

### Smoke Test (5 min)

```python
# In --test-mcts mode or standalone:
from mcts_trader.jax_agent.factored_search import FactoredPerAssetSearch

search = FactoredPerAssetSearch(
    model_apply_fn=model.apply,
    n_assets=64,
    discount=0.95,
    temperature=1.0,
)
search.value_scale = 2.39  # Example PopArt std
search.value_shift = 2.47  # Example PopArt mean

# Dummy observation
obs = jnp.ones((4, 34, 64, 64))  # Small batch for testing
rng = jax.random.PRNGKey(0)

actions, policy_probs, stats = search(params, rng, obs)

# Shape checks
assert actions.shape == (4,)
assert policy_probs.shape == (4, 320)

# Per-asset structure: each group of 5 should sum to 1.0
probs_reshaped = policy_probs.reshape(4, 64, 5)
per_asset_sums = probs_reshaped.sum(axis=-1)  # (4, 64)
assert jnp.allclose(per_asset_sums, 1.0, atol=1e-5), f"Per-asset sums: {per_asset_sums[0, :5]}"

# Q-values should vary (not collapsed)
assert stats['q_std'] > 0.01, f"Q-values collapsed: std={stats['q_std']}"

# Should have non-HOLD actions for some assets
assert stats['n_active_assets'] > 0, "All assets choosing HOLD"

print(f"Q mean={stats['q_mean']:.3f}, std={stats['q_std']:.3f}")
print(f"Advantage mean={stats['q_advantage_mean']:.3f}")
print(f"Active assets (non-HOLD): {stats['n_active_assets']:.1f}/64")
```

### Training Gate (50 epochs)

| Metric | Target | Rationale |
|--------|--------|-----------|
| policy_loss (per-asset CE) | Decreasing, < 1.5 by E50 | Factored targets are learnable |
| value_loss | Continuing to decrease | Value unaffected by search change |
| q_advantage_mean | > 0.1 | Search can distinguish actions |
| n_active_assets | 2-10 average | Model trades selectively, not all-HOLD |
| total_loss | Monotonic decrease | Overall convergence |

### Benchmark Gate

Run `tools/benchmark_easy.py` on E50 checkpoint. Target:
- Bull alpha > -15% (improved from -22%)
- Model trade rate > 5% (improved from 3.1%)
- At least 4/6 positive alpha

---

## Interaction with Other TASK_723 Fixes

| TASK_723 Fix | Still needed? | Why |
|-------------|---------------|-----|
| Fix 1 (reward descaling) | **YES** — factored search reads rewards from `recurrent_inference` which still has the `/100` bug | Fix 1 must be applied to `mctx_search.py` AND reward handling must be correct in factored_search.py |
| Fix 2 (discount match) | **YES** — factored search uses `config.td_discount` as its discount | Consistency is still required |
| Fix 3 (max_considered=64) | **Not needed** if `use_factored_search=True` (bypasses mctx entirely) | Only relevant for fallback to joint MCTS |
| Fix 4 (BC floor) | **YES** — BC regularizer still operates alongside factored policy targets | Oracle anchoring remains valuable |
| Fix 5 (value weight) | **YES** — value quality drives Q-value accuracy in factored search | More important than before — Q-values depend directly on V(s') |
| Fix 6 (slot rotation) | **YES** — data diversity is orthogonal to search method | |
| Fix 7 (benchmark MSE) | **YES** — diagnostic fix, independent | |

**Important**: Fix 1's reward bug exists in `mctx_search.py:_recurrent_fn` which is NOT used by factored search. Factored search calls `model.apply(..., mode='recurrent_inference')` which goes through `MuZeroNetworks.recurrent_inference()` → gets raw reward logits. The factored search handles reward conversion internally (no PopArt, no /100). So Fix 1 is only needed if you want joint MCTS as a fallback.

---

## Future Extensions

### Multi-Step Factored Search (TASK_724b, if needed)

1-step lookahead may be sufficient if the value head is accurate. If not:

1. Get per-asset Q-values from 1-step lookahead
2. Select top-k=4 assets by Q-advantage: `A(i) = max_a Q(i,a) - Q(i, HOLD)`
3. Compose joint action: selected assets take best action, rest HOLD
4. Run dynamics with composite action → new latent at depth 2
5. Repeat 1-4 at depth 2 → gives 2-step Q-values for top-k assets
6. Backpropagate: `Q_2step(i,a) = R_1(i,a) + γ * R_2(composite) + γ² * V(depth_2)`

This naturally handles the k-of-N selection (only top-k assets get multi-step evaluation) and reuses the existing dynamics model (composite joint action = valid input).

### Value-Head-Only Fast Mode (TASK_724c)

Skip full H×L reasoning in `_compute_q_values`. Apply value head directly to dynamics-predicted latent states (3 Dense layers, no reasoning cycles). ~20× faster but less accurate. Add as `factored_fast: bool = False` config option.

---

## File Summary

| File | Action | Changes |
|------|--------|---------|
| `mcts_trader/jax_agent/factored_search.py` | **NEW** | `FactoredPerAssetSearch` class (~150 lines) |
| `train_phase2.py` | MODIFY | Conditional search creation in `train_stage_b`, per-asset CE in `make_stage_b_train_step`, CLI args |
| `phase2_config.py` | MODIFY | 3 new config fields |
| `mcts_trader/jax_agent/__init__.py` | MODIFY | Export `FactoredPerAssetSearch` (optional, for test scripts) |
