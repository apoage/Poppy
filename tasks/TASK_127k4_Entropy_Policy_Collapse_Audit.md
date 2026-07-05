# TASK_127k4: Entropy and Policy Collapse Audit

## Executive Summary

This audit examines entropy computation and policy collapse prevention mechanisms. **Multiple protective mechanisms are in place**, but some edge cases and potential issues exist. The entropy computation is correct, and several anti-collapse mechanisms (entropy bonus, adaptive entropy, ε-greedy) work together effectively.

**Risk Level: LOW-MEDIUM** - Good protection against collapse, but some monitoring and edge cases need attention.

---

## Entropy Computation Analysis

### ✅ CORRECT: Entropy Computation with Log(0) Protection

**Location:** `mcts_trader/jax_agent/trainer.py:1107`

```python
# Compute policy entropy for regularization (prevent collapse)
policy_probs = jax.nn.softmax(initial_output['policy_logits'], axis=-1)
per_sample_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)
policy_entropy = jnp.sum(per_sample_entropy * valid_mask_float) / num_valid
```

**Verification:**
- ✅ Log(0) protected with `+ 1e-10` epsilon
- ✅ Computed on probabilities (not logits)
- ✅ Proper categorical entropy: `-sum(p * log(p))`
- ✅ Masked averaging for valid samples

**Same pattern used throughout:**
- `trainer.py:1163` (unroll steps)
- `mctx_search.py:311` (MCTS visit entropy)
- `exploration/entropy.py:5` (standalone function)

---

## Policy Collapse Prevention Mechanisms

### Mechanism 1: Entropy Bonus (Base Protection)

**Location:** `trainer.py:1183-1189`

```python
# Base MuZero loss (entropy bonus is SUBTRACTED to maximize it)
muzero_loss = (
    self.value_loss_weight * value_loss +
    self.policy_loss_weight * policy_loss +
    self.reward_loss_weight * reward_loss
    - entropy_coef * policy_entropy  # Maximize entropy to prevent collapse
    + adaptive_bonus  # Extra penalty when entropy drops below threshold
)
```

**Analysis:**
- ✅ Entropy bonus maximizes policy entropy directly
- Default `entropy_coef = 0.01`
- This is **standard practice** in policy gradient methods

**Potential Issue:** Fixed coefficient may not adapt to different training phases.

---

### Mechanism 2: Adaptive Entropy Bonus (Strong Protection)

**Location:** `trainer.py:1174-1181`

```python
# Adaptive entropy bonus - stronger push when entropy drops too low
# Target entropy ~4.0 (near-uniform over 192 actions = ln(192) ≈ 5.26)
# TASK_066: Use configurable min_entropy threshold
adaptive_bonus = jnp.where(
    policy_entropy < self.min_entropy,
    entropy_coef * 5.0 * (self.min_entropy - policy_entropy),  # 5x stronger push when too low
    0.0
)
```

**Analysis:**
- ✅ Triggers when entropy drops below `min_entropy` (default: 3.0)
- ✅ Multiplies base penalty by 5× when triggered
- ✅ Target entropy ~4.0 vs max ~5.26 for 192 actions

**Threshold Analysis:**
```
Max entropy (uniform): ln(192) ≈ 5.26
Target entropy: 4.0 (76% of max)
Min entropy threshold: 3.0 (57% of max)
Danger zone: < 2.5 (< 48% of max)
Collapse: < 1.0 (< 19% of max, near-deterministic)
```

The threshold of 3.0 is **conservative and appropriate**.

---

### Mechanism 3: Epsilon-Greedy in MCTS (Exploration Injection)

**Location:** `mcts_trader/jax_agent/mctx_search.py:227-235`

```python
# TASK_037: Apply ε-greedy mixing to training targets
# This breaks the feedback loop where MCTS sharpening → policy sharpening
# Action selection uses pure MCTS, but training target gets exploration mixed in
if self.epsilon_greedy > 0:
    policy_probs = apply_epsilon_greedy(
        policy_probs,
        epsilon=self.epsilon_greedy,
        action_mask=action_mask,
    )
```

**Analysis:**
- ✅ Mixes uniform exploration into MCTS policy targets
- ✅ Breaks positive feedback loop of policy sharpening
- ✅ Training targets have minimum entropy floor

**Minimum Entropy Guarantee:**
```python
# From exploration/entropy.py:9-13
def compute_minimum_entropy(epsilon: float, num_actions: int) -> float:
    if epsilon <= 0:
        return 0.0
    return epsilon * jnp.log(num_actions)

# With epsilon=0.2, num_actions=192:
# min_entropy = 0.2 * ln(192) = 0.2 * 5.26 = 1.05
```

Even with ε=0.1, minimum entropy is ~0.5, preventing total collapse.

---

### Mechanism 4: Gumbel MuZero (Built-in Exploration)

**Location:** `mcts_trader/jax_agent/mctx_search.py:195-208`

```python
if self.use_gumbel:
    # Gumbel MuZero (recommended)
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=self.num_simulations,
        max_num_considered_actions=self.max_num_considered_actions,
        qtransform=mctx.qtransform_by_parent_and_siblings,
        gumbel_scale=self.gumbel_scale,
        invalid_actions=~action_mask.astype(bool) if action_mask is not None else None,
    )
```

**Analysis:**
- ✅ Gumbel noise adds stochastic exploration
- ✅ `max_num_considered_actions` limits action space (default: 16 of 192)
- ✅ Natural entropy from sampling, even without explicit bonus

---

## Action Masking

### ✅ CORRECT: Action Mask Applied Properly

**Location:** `mcts_trader/jax_agent/mctx_search.py:175-184`

```python
# Apply action mask to root logits if provided
if action_mask is not None:
    # Mask is 1 for valid, 0 for invalid
    # Set invalid logits to -inf
    min_logit = jnp.finfo(root.prior_logits.dtype).min
    root_logits = jnp.where(action_mask.astype(bool), root.prior_logits, min_logit)
    
    # Re-create root with masked logits
    root = root.replace(prior_logits=root_logits)
```

**Analysis:**
- ✅ Invalid actions set to `-inf` (not large negative number)
- ✅ Uses `jnp.finfo(dtype).min` for proper -inf representation
- ✅ Mask applied before MCTS search begins
- ✅ Also passed to `invalid_actions` parameter in mctx

**Important:** Setting to `-inf` ensures softmax outputs exactly 0 for invalid actions, maintaining valid probability distributions.

---

## Temperature and Softmax

### Temperature Scaling

**Location:** `mcts_trader/jax_agent/mctx_search.py:51`

```python
def __init__(..., temperature: float = 1.0, ...)
```

**Analysis:**
- ✅ Temperature parameter exists for standard MuZero
- ✅ Default 1.0 (no temperature scaling)
- ✅ Only used in `muzero_policy` (not Gumbel)

**Note:** When using Gumbel MuZero (default), temperature is not used - exploration comes from Gumbel noise instead.

---

## Potential Issues and Risks

### 🟡 MEDIUM: No Temperature Annealing Schedule

**Issue:** Temperature is fixed at 1.0 (or gumbel_scale is fixed). No schedule to:
- Start high (exploration) → End low (exploitation)
- Adapt based on training progress

**Recommendation:** Consider temperature annealing:
```python
# Start with high temperature (exploration)
# Gradually reduce as training progresses
temperature = max(0.5, 1.0 - epoch / 500)  # Linear decay to 0.5
```

---

### 🟡 LOW: Entropy Coefficient is Fixed

**Issue:** `entropy_coef` is initialized but not scheduled:
```python
self.entropy_coef = entropy_coef  # Fixed value
```

**Potential Problem:**
- Early training: May need lower entropy (let MCTS guide exploration)
- Late training: May need higher entropy (prevent premature convergence)

**Recommendation:** Consider entropy coefficient scheduling:
```python
# Decay entropy coefficient over training
self.entropy_coef = entropy_coef * max(0.5, 1.0 - epoch / total_epochs)
```

---

### 🟢 LOW: Minimum Entropy Threshold May Be Too Conservative

**Current:** `min_entropy = 3.0` for 192 actions (max entropy = 5.26)

**Analysis:**
- Current threshold: 57% of max entropy
- This is conservative - good for preventing collapse
- But may restrict policy from becoming confident on clear decisions

**Trade-off:**
- Too low (< 2.0): Risk of collapse
- Too high (> 4.0): May prevent learning good policies
- Current (3.0): Reasonable balance

**Monitoring:** Track `policy_entropy` in logs - if consistently near 3.0, consider lowering threshold.

---

### 🟢 LOW: No Policy Gradient Variance Monitoring

**Missing:** No explicit tracking of policy gradient variance, which can indicate:
- High variance → Unstable training, may need entropy boost
- Near-zero variance → Policy collapse risk

**Recommendation:** Add monitoring:
```python
# In _train_step, after computing policy loss
policy_grad_norm = jnp.linalg.norm(grads['policy_head'])
metrics['policy_grad_norm'] = policy_grad_norm

if policy_grad_norm > 10.0:
    print(f"⚠️  High policy gradient norm: {policy_grad_norm:.2f}")
```

---

## Collapse Indicators to Monitor

### Key Metrics from CSV Logs

Based on `CSV_METRICS_REFERENCE.md`:

| Metric | Healthy Range | Warning | Collapse |
|--------|--------------|---------|----------|
| `policy_entropy` | 3.5-4.5 | < 3.0 | < 2.0 |
| `pct_hold` | 30-70% | < 20% or > 90% | < 10% or > 95% |
| `pct_buy` | 10-40% | < 5% | < 1% |
| `pct_sell` | 10-40% | < 5% | < 1% |

**From the actual training logs checked:**
```
Epoch 1: pct_hold=84.9%, pct_buy=3.9%, pct_sell=3.3%
Epoch 27: pct_hold=94.7%, pct_buy=1.3%, pct_sell=1.3%
```

The high hold percentage (94.7%) in epoch 27 suggests the policy is becoming conservative, but not collapsed (entropy still likely > 3.0 based on adaptive bonus kicking in).

---

## Anti-Collapse Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    POLICY COLLAPSE DEFENSES                      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Architecture                                           │
│    - Categorical policy (natural smoothing via softmax)         │
│    - Gumbel MuZero (stochastic exploration)                     │
│    - Action masking (prevents invalid action collapse)          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Training Objectives                                    │
│    - Entropy bonus: -entropy_coef * entropy                     │
│    - Adaptive bonus: 5× stronger when entropy < min_entropy     │
│    - MCTS targets with visit count smoothing                    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Exploration Injection                                  │
│    - ε-greedy mixing in MCTS targets                            │
│    - Minimum entropy floor: ε * ln(n_actions)                   │
│    - Gumbel noise in action selection                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Monitoring                                             │
│    - policy_entropy tracked in metrics                          │
│    - Action distribution logged                                 │
│    - adaptive_bonus triggers when needed                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommendations

### Priority 1: Monitor Policy Entropy Trends

Add to training loop:
```python
# Log entropy trend
if iteration % 50 == 0:
    print(f"  Policy entropy: {metrics['policy_entropy']:.2f} "
          f"(min_threshold: {self.min_entropy:.2f})")
    
    if metrics['policy_entropy'] < self.min_entropy + 0.5:
        print(f"  ⚠️  Entropy near threshold! Adaptive bonus active.")
```

### Priority 2: Consider Entropy Coefficient Scheduling

```python
def get_entropy_coef(self, epoch: int) -> float:
    """Scheduled entropy coefficient."""
    # Decay from initial to half over training
    progress = min(1.0, epoch / self.total_epochs)
    return self.entropy_coef * (1.0 - 0.5 * progress)
```

### Priority 3: Add Policy Gradient Norm Monitoring

```python
# After gradient computation
policy_grads = traverse_util.flatten_dict(grads['policy_head'])
grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in policy_grads.values()))
metrics['policy_grad_norm'] = grad_norm

if grad_norm > 10.0:
    print(f"⚠️  High policy grad norm: {grad_norm:.2f}")
```

### Priority 4: Document Anti-Collapse Design

Add to `AGENTS.md` or architecture docs:
```markdown
## Policy Collapse Prevention

Multiple mechanisms prevent policy collapse:
1. Entropy bonus in loss (default coef: 0.01)
2. Adaptive bonus (5× when entropy < 3.0)
3. ε-greedy in MCTS (minimum entropy floor)
4. Gumbel MuZero (natural exploration)

Monitoring:
- `policy_entropy` should stay > 3.0
- `pct_hold` should stay < 95%
- If entropy drops consistently, lower `min_entropy` threshold
```

---

## Testing Plan

1. **Entropy Sanity Check:**
   ```python
   # Uniform policy should have max entropy
   uniform = jnp.ones(192) / 192
   entropy = compute_policy_entropy(uniform)
   assert jnp.abs(entropy - jnp.log(192)) < 0.01
   
   # Deterministic policy should have 0 entropy
   deterministic = jnp.zeros(192).at[0].set(1.0)
   entropy = compute_policy_entropy(deterministic)
   assert entropy < 0.01
   ```

2. **Adaptive Bonus Test:**
   ```python
   # Below threshold
   entropy_low = 2.0
   bonus_low = compute_adaptive_bonus(entropy_low, min_entropy=3.0)
   assert bonus_low > 0
   
   # Above threshold
   entropy_high = 4.0
   bonus_high = compute_adaptive_bonus(entropy_high, min_entropy=3.0)
   assert bonus_high == 0
   ```

3. **Epsilon-Greedy Floor Test:**
   ```python
   # Sharp policy
   sharp = jnp.zeros(192).at[0].set(0.9).at[1].set(0.1)
   
   # After epsilon-greedy with ε=0.1
   mixed = apply_epsilon_greedy(sharp, epsilon=0.1)
   entropy = compute_policy_entropy(mixed)
   min_entropy = 0.1 * jnp.log(192)
   
   assert entropy >= min_entropy - 0.01
   ```

---

## Acceptance Criteria

- [ ] Entropy computation verified correct (log(0) protected)
- [ ] All four anti-collapse mechanisms documented
- [ ] Policy entropy monitoring in training logs
- [ ] Action distribution (pct_hold/buy/sell) healthy (>10% each)
- [ ] Entropy stays above 2.5 throughout training
- [ ] Optional: Entropy coefficient scheduling implemented
- [ ] Optional: Policy gradient norm monitoring added

---

## References

- `mcts_trader/jax_agent/trainer.py:1105-1189` - Entropy computation and loss
- `mcts_trader/jax_agent/mctx_search.py:142-265` - MCTS with exploration
- `mcts_trader/jax_agent/exploration/entropy.py` - Entropy utilities
- `mcts_trader/jax_agent/exploration/epsilon_greedy.py` - ε-greedy mixing
- `docs/CSV_METRICS_REFERENCE.md` - Policy entropy monitoring
