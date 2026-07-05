# TASK_125: Causal Expert Imprinting (Retrospective AMAX)

**Date:** 2026-01-30
**Status:** OPEN
**Priority:** HIGH
**Depends On:** TASK_124 (alpha bugs fixed)

## Problem Statement

The current AMAX (Optimal Action Imprint) feature in TASK_119 **leaks future information** by design:

```python
# trainer.py:1086 - Uses FUTURE returns during training
returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]  # ALL returns
optimal_path = compute_optimal_path(returns_for_amax[0])  # Hindsight-optimal
imprint_loss = compute_imprint_loss(policy, optimal_targets)  # Train toward future!
```

This teaches the agent "what would have been optimal IF you knew the future" - which is useless (or harmful) for real trading because:
1. It trains the agent to recognize patterns that only exist in hindsight
2. On real data, this causes overfitting to noise
3. The agent learns to "predict" unpredictable events

**However**, expert imprinting IS valuable if done **causally** - using only information available at decision time.

---

## Goal

Implement **Causal Expert Imprinting** that:
1. Never leaks future information during training
2. Teaches the agent patterns from COMPLETED historical trades
3. Provides a "soft guide" toward profitable behaviors without cheating
4. Can be safely used on both synthetic curriculum AND historical data

---

## Current Implementation Analysis

### Files Involved

| File | Role | Lines |
|------|------|-------|
| `mcts_trader/jax_agent/trainer.py` | AMAX loss computation | 1075-1147 |
| `mcts_trader/jax_agent/path_solver.py` | Optimal path algorithms | Full file |
| `mcts_trader/training/self_play.py` | Price sequence collection | 794-803 |
| `mcts_trader/jax_agent/replay_buffer.py` | Stores log_price_sequence | 329 |

### Current AMAX Flow (BROKEN - Future Leak)

```
Episode N starts
    ↓
Agent observes state[t], takes action
    ↓
Batch created with log_price_sequence (contains t+1, t+2, ..., T)
    ↓
trainer.py:1086 - Compute returns_for_amax (uses ALL future)
    ↓
trainer.py:1088 - Compute optimal_path (hindsight)
    ↓
trainer.py:1142 - imprint_loss toward future-optimal actions
    ↓
Agent learns "patterns" that require future knowledge ← BUG!
```

---

## Proposed Solution: Retrospective Expert Buffer

### Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           TRAINING PIPELINE                 │
                    └─────────────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         │                              │                              │
         ▼                              ▼                              ▼
   ┌───────────┐                ┌───────────────┐            ┌─────────────────┐
   │ Episode N │                │ Expert Buffer │            │   Trainer       │
   │ (Current) │                │ (Past Ep's)   │            │                 │
   └─────┬─────┘                └───────┬───────┘            └────────┬────────┘
         │                              │                              │
         │ Agent trades                 │ Contains                     │ Computes losses:
         │ normally (NO                 │ (obs, optimal_action)        │
         │ imprint loss)                │ from episodes                │ 1. RL loss (causal)
         │                              │ 0, 1, ..., N-1               │ 2. BC loss from buffer
         │                              │                              │    (also causal!)
         ▼                              │                              │
   ┌───────────┐                        │                              │
   │ Episode N │                        │                              │
   │ COMPLETES │                        │                              │
   └─────┬─────┘                        │                              │
         │                              │                              │
         │ NOW compute                  │                              │
         │ optimal path                 │                              │
         │ (hindsight OK -              │                              │
         │  episode done!)              │                              │
         │                              │                              │
         ▼                              │                              │
   ┌───────────────┐                    │                              │
   │ Add to Expert │────────────────────┘                              │
   │ Buffer        │                                                   │
   └───────────────┘                                                   │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘
```

### Key Insight: Temporal Separation

The trick is **temporal separation**:
- During Episode N: NO imprint loss (agent explores freely)
- After Episode N ends: Compute optimal path (hindsight is NOW valid)
- During Episode N+1: BC loss samples from buffer (past episodes only)

This is **causally valid** because:
- The expert demonstrations come from COMPLETED episodes
- At training time for Episode N+1, we're learning from Episodes 0..N
- No information from the current episode's future is used

---

## Critical Refinements (Review Feedback)

### 1. Action Space Clarification - NOT A BLOCKER

The review identified a potential mismatch between 192 (config default) and 320 (path_solver).

**Actual Status:** The JAX training pipeline uses **320 actions** consistently:
- `train_jax_fast.py:310`: `action_space_size = n_assets * 5`
- `setup.py:431`: `action_space_size = n_assets * 5`
- `path_solver.py:401`: `n_actions_per_asset: int = 5` (default)
- `trainer.py:1131`: calls with `n_actions_per_asset=5`

The `trading_config.py` values (192, n_actions_per_asset=3) are **stale/unused** in the JAX path.

**Verification:**
```python
# All of these are 320:
assert train_jax_fast.action_space_size == 320
assert model.action_space_size == 320
assert optimal_path_to_policy(..., n_actions_per_asset=5) outputs (T, 320)
```

### 2. Memory: Lazy Reconstruction - ALREADY ADDRESSED

The review correctly identified that storing 50k full observations (10.4 GB) would OOM.

**Our design already uses lazy reconstruction** (see "Memory-Efficient Buffer Architecture" section):
- Store 32-byte **indices** (episode_id, step, action, etc.)
- Keep raw prices in **memory-mapped PriceCache** on disk
- Reconstruct observations **on-demand** during sampling

Memory usage:
```
Old design: 50k × 207KB = 10.4 GB  ❌ OOM
Our design: 50k × 32B  = 1.6 MB   ✅ Fits easily
```

### 3. Curriculum Reset Handling - MUST ADD

**Risk:** Imprinting on L0 (easy sine waves) while learning L5 (complex) teaches unsafe habits.

**Solution:** Add `clear()` and `reset_for_level()` methods to ExpertBuffer:

```python
class UnifiedIndexBuffer:
    # ... existing code ...

    def clear(self):
        """Clear all demos and agent experience. Called on major curriculum changes."""
        with self._lock:
            self.demo_count = 0
            self.agent_count = 0
            self.agent_write_idx = self.demo_capacity
            self.sum_tree[:] = 0
            self.min_tree[:] = np.inf
            self.max_priority = 1.0
            print(f"[ExpertBuffer] Cleared all {self.total_capacity} entries")

    def reset_for_level(self, new_level: int, keep_fraction: float = 0.0):
        """
        Reset buffer for new curriculum level.

        Args:
            new_level: The new curriculum level (0-14)
            keep_fraction: Fraction of demos to retain (0.0 = clear all)
        """
        with self._lock:
            if keep_fraction <= 0:
                self.clear()
                return

            # Keep top-performing demos based on episode_return
            keep_count = int(self.demo_count * keep_fraction)
            if keep_count > 0:
                # Sort by episode_return, keep best
                returns = self.indices[:self.demo_count]['episode_return']
                keep_indices = np.argsort(returns)[-keep_count:]

                # Compact demos to front
                for new_idx, old_idx in enumerate(sorted(keep_indices)):
                    if new_idx != old_idx:
                        self.indices[new_idx] = self.indices[old_idx]
                        # Update priority trees
                        priority = self.indices[new_idx]['priority']
                        self._update_priority(new_idx, priority)

                # Clear remaining demo slots
                for idx in range(keep_count, self.demo_count):
                    self._update_priority(idx, 0.0)

            self.demo_count = keep_count

            # Always clear agent buffer on level change
            self.agent_count = 0
            self.agent_write_idx = self.demo_capacity
            for idx in range(self.demo_capacity, self.total_capacity):
                self._update_priority(idx, 0.0)

            print(f"[ExpertBuffer] Reset for L{new_level}: kept {keep_count} demos, cleared agent buffer")

    def tag_regime(self, regime_tag: int):
        """Tag all recent additions with current market regime."""
        # Called by curriculum manager when regime detected
        pass
```

**Integration with curriculum manager:**
```python
# In dojo orchestrator or train_jax_fast.py
def on_level_change(old_level: int, new_level: int, expert_buffer: UnifiedIndexBuffer):
    """Called when curriculum advances to new level."""

    if new_level == 0:
        # Starting fresh - clear everything
        expert_buffer.clear()
    elif new_level > old_level + 2:
        # Big jump - clear most, keep top 10%
        expert_buffer.reset_for_level(new_level, keep_fraction=0.1)
    else:
        # Gradual progression - keep top 25%
        expert_buffer.reset_for_level(new_level, keep_fraction=0.25)
```

### 4. DS24 Metrics - MUST HAVE (Not "Should Have")

**Required metrics for debugging:**

```python
@dataclass
class ExpertMetrics:
    """Metrics logged to DS24 every training step."""

    # Buffer stats
    demo_count: int
    demo_fill_ratio: float
    agent_count: int
    avg_demo_confidence: float
    avg_demo_return: float

    # Learning signal quality
    expert_acceptance_rate: float  # How often argmax(policy) == expert_action
    bc_loss: float                 # BC loss value
    bc_weight: float               # Current BC weight (after decay)
    q_filter_active_ratio: float   # If using Q-filter: fraction where demo > policy

    # Distribution health
    regime_distribution: Dict[int, int]  # Count per regime tag
    return_percentiles: Tuple[float, float, float]  # 25th, 50th, 75th

def compute_expert_metrics(
    buffer: UnifiedIndexBuffer,
    batch: Dict[str, np.ndarray],
    policy_logits: jnp.ndarray,
    bc_loss: float,
    bc_weight: float,
) -> ExpertMetrics:
    """Compute metrics for DS24 logging."""

    # Expert acceptance: does policy agree with expert?
    policy_actions = np.argmax(policy_logits, axis=-1)
    expert_actions = batch['optimal_actions']
    is_demo = batch['is_demo']

    demo_mask = is_demo.astype(bool)
    if demo_mask.sum() > 0:
        acceptance = (policy_actions[demo_mask] == expert_actions[demo_mask]).mean()
    else:
        acceptance = 0.0

    # Buffer stats
    stats = buffer.get_stats()
    demo_entries = buffer.indices[:buffer.demo_count]

    return ExpertMetrics(
        demo_count=buffer.demo_count,
        demo_fill_ratio=stats['demo_fill_ratio'],
        agent_count=buffer.agent_count,
        avg_demo_confidence=float(demo_entries['confidence'].mean()) if len(demo_entries) > 0 else 0.0,
        avg_demo_return=float(demo_entries['episode_return'].mean()) if len(demo_entries) > 0 else 0.0,
        expert_acceptance_rate=float(acceptance),
        bc_loss=bc_loss,
        bc_weight=bc_weight,
        q_filter_active_ratio=0.0,  # Computed if using Q-filter
        regime_distribution=dict(zip(*np.unique(demo_entries['regime_tag'], return_counts=True))),
        return_percentiles=tuple(np.percentile(demo_entries['episode_return'], [25, 50, 75])) if len(demo_entries) > 0 else (0, 0, 0),
    )

# In training loop:
def train_step(...):
    # ... compute losses ...

    metrics = compute_expert_metrics(buffer, batch, policy_logits, bc_loss, bc_weight)

    # Log to DS24
    return {
        'expert/demo_count': metrics.demo_count,
        'expert/acceptance_rate': metrics.expert_acceptance_rate,
        'expert/bc_loss': metrics.bc_loss,
        'expert/bc_weight': metrics.bc_weight,
        'expert/avg_confidence': metrics.avg_demo_confidence,
        'expert/avg_return': metrics.avg_demo_return,
        # ... other metrics
    }
```

### 5. Reconstruction Pipeline Clarification

The review noted that `self_play.py` discards observations to save RAM.

**Our design handles this:**

1. **Price arrays are NOT discarded** - stored in PriceCache (memory-mapped)
2. **Episode data flow:**
   ```
   Episode completes
       ↓
   Raw prices already in PriceCache (from env setup)
       ↓
   Compute optimal path from returns
       ↓
   Store (episode_id, step, optimal_action) indices
       ↓
   On sample: reconstruct from PriceCache → encode on GPU
   ```

3. **Integration with self_play.py:**
   ```python
   # After episode completion in self_play.py
   def on_episode_complete(
       episode_id: int,
       price_data: np.ndarray,  # Already in PriceCache
       returns: np.ndarray,
       expert_buffer: UnifiedIndexBuffer,
   ):
       # Compute hindsight-optimal path
       optimal_path = compute_optimal_path(returns)

       # Store indices (NOT full observations)
       for step in range(len(returns)):
           expert_buffer.add_demo(
               episode_id=episode_id,
               step=step,
               action=0,  # Actual action taken (not used for demos)
               optimal_action=optimal_path.actions[step],
               reward=returns[step].sum(),
               confidence=optimal_path.step_confidence[step],
               episode_return=optimal_path.total_return,
           )
   ```

---

## Implementation Plan

### Phase 1: Expert Buffer Infrastructure

#### 1.1 Create ExpertBuffer Class

```python
# New file: mcts_trader/jax_agent/expert_buffer.py

@dataclass
class ExpertSample:
    """Single expert demonstration."""
    observation: np.ndarray      # (27, window, n_assets) or encoded
    optimal_action: int          # Action index (0-319)
    confidence: float            # How "optimal" was this action? (0-1)
    context_return: float        # What return did this trade achieve?

class ExpertBuffer:
    """Buffer of hindsight-optimal (observation, action) pairs from past episodes."""

    def __init__(
        self,
        max_size: int = 50000,
        min_confidence: float = 0.5,  # Only store high-confidence demonstrations
        recency_weight: float = 0.1,  # Weight recent episodes more
    ):
        self.buffer = deque(maxlen=max_size)
        self.min_confidence = min_confidence
        self.recency_weight = recency_weight
        self.episode_count = 0

    def add_completed_episode(
        self,
        observations: np.ndarray,    # (T, 27, window, n_assets)
        returns: np.ndarray,         # (T, n_assets) - NOW COMPLETE
        actions_taken: np.ndarray,   # (T,) - what agent actually did
        method: str = 'greedy',
    ):
        """
        Called AFTER episode ends - computes optimal path and stores demonstrations.

        This is causally valid because the episode is COMPLETE.
        """
        # Compute hindsight-optimal path (safe - episode done!)
        optimal_path = compute_optimal_path(returns, method=method)

        # Convert to action indices
        optimal_actions = optimal_path_to_actions(optimal_path.selections, returns)

        # Store (obs, optimal_action) pairs with confidence scores
        for t in range(len(observations)):
            # Confidence = how much alpha this action captured
            step_alpha = compute_step_alpha(optimal_path, t)

            if step_alpha > self.min_confidence:
                sample = ExpertSample(
                    observation=observations[t],
                    optimal_action=optimal_actions[t],
                    confidence=step_alpha,
                    context_return=float(optimal_path.cumulative_pnl[t]),
                )
                self.buffer.append(sample)

        self.episode_count += 1

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample expert demonstrations for behavioral cloning."""
        if len(self.buffer) < batch_size:
            return None

        # Weighted sampling: recent + high-confidence preferred
        samples = self._weighted_sample(batch_size)

        obs = np.stack([s.observation for s in samples])
        actions = np.array([s.optimal_action for s in samples])
        weights = np.array([s.confidence for s in samples])

        return obs, actions, weights
```

#### 1.2 Integration Points

```python
# In self_play.py - after episode collection
def collect_trajectories(...):
    # ... existing code ...

    # NEW: After episode completes, add to expert buffer
    if expert_buffer is not None and episode_complete:
        expert_buffer.add_completed_episode(
            observations=trajectory['observations'],
            returns=trajectory['returns'],  # Now complete!
            actions_taken=trajectory['actions'],
        )
```

```python
# In trainer.py - compute_loss()
def compute_loss(self, batch, expert_buffer=None):
    # ... existing RL loss ...

    # NEW: Causal BC loss from past episodes
    bc_loss = 0.0
    if expert_buffer is not None and self.bc_weight > 0:
        expert_batch = expert_buffer.sample_batch(self.bc_batch_size)
        if expert_batch is not None:
            obs, actions, weights = expert_batch
            # Compute policy for expert observations
            expert_policy = self.model.forward(obs)
            # Cross-entropy toward optimal actions, weighted by confidence
            bc_loss = weighted_cross_entropy(expert_policy, actions, weights)
            bc_loss = self.bc_weight * bc_loss

    total_loss = rl_loss + bc_loss
```

### Phase 2: Signal Pattern Mining (Optional Enhancement)

Instead of just storing (obs, action) pairs, mine **recurring patterns**:

```python
class PatternMiner:
    """Extracts recurring profitable patterns from expert buffer."""

    def mine_entry_patterns(self, expert_buffer, min_occurrences=10):
        """
        Find observation patterns that frequently preceded profitable actions.

        Returns: List of (pattern_embedding, profitable_action, frequency)
        """
        # Cluster observations that led to profitable buys
        buy_observations = [s.observation for s in expert_buffer
                          if s.optimal_action % 5 == 2]  # Buy actions

        # Use simple clustering or learned embeddings
        patterns = cluster_observations(buy_observations)

        return patterns

    def compute_pattern_similarity(self, current_obs, patterns):
        """
        Score how similar current observation is to known profitable patterns.

        Can be used as auxiliary input to the model.
        """
        similarities = [cosine_sim(current_obs, p.embedding) for p in patterns]
        return np.array(similarities)
```

### Phase 3: Hindsight Experience Replay (HER) Extension

Relabel "failed" trajectories with alternative goals:

```python
def her_relabel(trajectory, achieved_return):
    """
    Hindsight Experience Replay for trading.

    If a trade targeted +5% but achieved +2%, relabel:
    "This IS an optimal trajectory for a +2% target"
    """
    # Find the step where +2% was achieved
    returns = np.cumsum(trajectory['rewards'])
    achieved_steps = np.where(returns >= achieved_return)[0]

    if len(achieved_steps) > 0:
        optimal_exit = achieved_steps[0]

        # Relabel: actions up to optimal_exit are "correct"
        relabeled = {
            'observations': trajectory['observations'][:optimal_exit+1],
            'optimal_actions': trajectory['actions'][:optimal_exit],
            'target_return': achieved_return,
        }
        return relabeled

    return None
```

---

## Configuration Options

```python
# New config in TrainingConfig
@dataclass
class CausalExpertConfig:
    """Configuration for causal expert imprinting."""

    # Expert buffer settings
    expert_buffer_size: int = 50000
    min_confidence_threshold: float = 0.5

    # BC loss weight scheduling
    bc_weight_initial: float = 0.1
    bc_weight_final: float = 0.01
    bc_weight_decay_epochs: int = 100

    # What to store in buffer
    store_mode: str = 'top_quantile'  # 'all', 'top_quantile', 'profitable_only'
    top_quantile: float = 0.25  # Store top 25% by alpha capture

    # Pattern mining (optional)
    enable_pattern_mining: bool = False
    pattern_update_interval: int = 10  # Re-mine every N episodes

    # HER relabeling (optional)
    enable_her: bool = False
    her_goals: List[float] = [0.01, 0.02, 0.05]  # Return targets to relabel for

    # Curriculum integration
    synthetic_only: bool = True  # Only use expert buffer on synthetic data
    real_data_mode: str = 'disabled'  # 'disabled', 'pattern_only', 'full'
```

---

## Comparison: Old AMAX vs Causal Expert

| Aspect | Old AMAX | Causal Expert |
|--------|----------|---------------|
| **Information** | Uses future returns from CURRENT episode | Uses past COMPLETED episodes only |
| **Causality** | BROKEN - leaks future | VALID - temporal separation |
| **When computed** | During training iteration | After episode ends |
| **What it teaches** | "Do this because future is X" | "This pattern historically worked" |
| **Safe for real data?** | NO - causes overfitting | YES - learns generalizable patterns |
| **Computational cost** | Low (in training loop) | Medium (buffer + extra forward pass) |

---

## Success Criteria

### Must Have
1. [ ] ExpertBuffer class implemented and tested
2. [ ] Integration with self_play.py for episode completion
3. [ ] BC loss in trainer.py using expert buffer
4. [ ] No future information leakage (verified by ablation)
5. [ ] Works on synthetic curriculum (L0-L5)

### Should Have
6. [ ] Configurable confidence thresholds
7. [ ] BC weight scheduling (high early, low late)
8. [ ] Logging of expert buffer statistics
9. [ ] Memory-efficient storage (compressed observations)

### Nice to Have
10. [ ] Pattern mining for recurring signals
11. [ ] HER relabeling for failed trajectories
12. [ ] Visualization of learned patterns

---

## Verification Tests

### Test 1: Causality Check

```python
def test_no_future_leak():
    """Verify expert buffer only contains past episode data."""
    expert_buffer = ExpertBuffer()

    # Run episode 0
    trajectory_0 = run_episode(env, agent)
    expert_buffer.add_completed_episode(trajectory_0)

    # During episode 1, buffer should only have episode 0 data
    for step in episode_1:
        expert_batch = expert_buffer.sample_batch(32)
        # All samples should be from episode 0, not episode 1
        assert all(sample.episode_id < 1 for sample in expert_batch)
```

### Test 2: Learning Signal Quality

```python
def test_expert_improves_learning():
    """Expert buffer should improve learning on synthetic data."""
    # Train without expert buffer
    baseline_alpha = train(epochs=100, expert_buffer=None)

    # Train with expert buffer
    expert_alpha = train(epochs=100, expert_buffer=ExpertBuffer())

    # Expert should help (especially early)
    assert expert_alpha > baseline_alpha * 0.9  # At least not harmful
```

### Test 3: No Overfitting on Historical

```python
def test_no_overfitting_historical():
    """On historical data, expert buffer shouldn't cause memorization."""
    # Train on 2024-01 with expert buffer
    train_alpha = train(data='2024-01', expert_buffer=ExpertBuffer())

    # Evaluate on 2024-02 (unseen)
    test_alpha = evaluate(data='2024-02')

    # Should generalize, not memorize
    assert test_alpha > -10  # Not catastrophic
```

---

## Research Prompt for External Models

```
TASK: Design causal expert imprinting for trading RL

CONTEXT:
We have a MuZero-based trading agent trained on financial time series.
We want to use hindsight-optimal demonstrations to guide learning
WITHOUT leaking future information.

CONSTRAINTS:
1. At decision time t, agent can only see observations up to t-1
2. Hindsight-optimal path requires knowing returns from t to T (future)
3. Training must be causally valid - no future leak

CURRENT APPROACH:
- After each episode COMPLETES, compute hindsight-optimal path
- Store (observation, optimal_action) pairs in "Expert Buffer"
- During training, sample from buffer (past episodes only)
- Behavioral cloning loss toward optimal actions

QUESTIONS FOR RESEARCH:
1. What are best practices for mixing RL loss with BC loss?
   - Fixed ratio? Scheduled decay? Adaptive weighting?
   - Literature: DAgger, GAIL, demonstrations in RL

2. How to select which demonstrations to store?
   - All? Top quantile by profitability? Diverse set?
   - Literature: Prioritized Experience Replay, hindsight goal selection

3. How to handle distribution shift?
   - Expert buffer reflects past market conditions
   - Current market may be different
   - Literature: Domain adaptation, continual learning

4. Can we extract generalizable "patterns" instead of raw obs?
   - Cluster similar profitable setups
   - Learn pattern embeddings
   - Literature: Representation learning, prototype networks

5. What about HER (Hindsight Experience Replay)?
   - Relabel failed trades with alternative goals
   - "This was optimal for +2% target even if +5% failed"
   - Literature: HER (Andrychowicz 2017), goal-conditioned RL

6. Theoretical guarantees?
   - Does this satisfy "no lookahead" constraint?
   - Can we prove convergence with mixed RL+BC?
   - Literature: Imitation + RL theory

SPECIFIC SEARCH QUERIES:
- "combining imitation learning with reinforcement learning"
- "hindsight experience replay finance"
- "causal representation learning time series"
- "behavioral cloning weight scheduling"
- "expert demonstrations reinforcement learning"
- "DAgger algorithm financial trading"
- "goal-conditioned RL trading"

EXPECTED OUTPUT:
1. Recommended approach with literature support
2. Potential failure modes and mitigations
3. Hyperparameter guidance (BC weight, buffer size, etc.)
4. Implementation considerations for JAX/Flax
```

---

## Research Findings (from docs/imprinting/)

### Theoretical Validation: NO CAUSALITY VIOLATION

**Key insight:** Training on hindsight-optimal demonstrations computed AFTER episode completion uses future information only to construct better *training targets*—the learned policy π(a|s) conditions solely on current state at inference.

This mirrors HER's proof (Andrychowicz et al., 2017): "The goal being pursued influences the agent's actions but NOT the environment dynamics."

### Q-Filter: Recommended BC Weight Strategy

Instead of fixed or decaying BC weight, use **Q-Filter**:

```python
# Apply BC loss ONLY when demo action is better than current policy
L_BC = Σ||π(s_i) - a_i||² · 𝟙[Q(s_i, a_i) > Q(s_i, π(s_i))]
```

This automatically:
- Dominates early training (when policy is worse than demos)
- Diminishes as policy exceeds demonstrator quality
- Prevents BC from limiting improvement beyond expert

### Hyperparameter Recommendations

| Parameter | Value | Notes |
|-----------|-------|-------|
| BC weight λ_BC | 1.0 → 0.01 | Decay over 50% of training (or use Q-filter) |
| Demo buffer size | 10⁴-10⁵ | Permanent, boosted priority |
| Agent buffer size | 10⁵-10⁶ | FIFO eviction |
| Demo samples/batch | 128/1024 (~12.5%) | Mixed minibatch |
| HER relabeling k | 4-8 | Future strategy |
| PER α (priority exp) | 0.6 | |
| PER β (IS weight) | 0.4 → 1.0 | Anneal over training |
| Demo priority boost ε_D | 0.001 | |
| Recency decay λ | 0.01 | Half-life ~70 days |

### Demonstration Selection: Diversity Over Volume

**RDES finding:** A diverse buffer of moderately successful trades OUTPERFORMS a homogeneous buffer of best trades.

```python
# Selection criteria
- Store top 60-70% quantile by return (NOT top 10%)
- K-means clustering on state representations
- Each cluster contributes proportionally
- Use PER with demo priority boost
```

### Dual Buffer Architecture (DDPGfD Style)

```
┌─────────────────────────────────────────────────────────┐
│ Demo Buffer R_D (10⁴-10⁵)                               │
│ - Permanent storage                                      │
│ - Boosted sampling priority (ε_D = 0.001)               │
│ - Hindsight-optimal demonstrations                       │
├─────────────────────────────────────────────────────────┤
│ Agent Buffer R (10⁵-10⁶)                                │
│ - FIFO eviction                                          │
│ - Self-play experience                                   │
│ - Can promote good trajectories to demo buffer          │
└─────────────────────────────────────────────────────────┘

Minibatch: 128 demo + 896 agent = 1024 transitions
```

### HER for Trading: Return-Target Relabeling

```python
def relabel_trading_trajectory(trajectory, k=4):
    """HER with continuous return goals."""
    relabeled = []
    for i, (s, a, r, s_next, goal) in enumerate(trajectory):
        relabeled.append((s, a, r, s_next, goal))  # Original

        # Future strategy: sample k achieved returns from AFTER current step
        future_returns = [traj[j].achieved_return for j in range(i+1, len(trajectory))]
        for new_goal in random.sample(future_returns, min(k, len(future_returns))):
            new_r = 0 if current_return >= new_goal else -1
            relabeled.append((s, a, new_r, s_next, new_goal))
    return relabeled
```

**Multi-dimensional goals:** goal = {return_target: 0.05, max_drawdown: 0.02, holding_period: 5}

### Distribution Shift: Regime-Aware Management

**Three complementary strategies:**

1. **Regime tagging:** Label demos with detected regime (bull/bear/sideways/high-vol)
   - Sample matching regime with 3x probability boost
   - Maintain 20% cross-regime sampling for robustness

2. **Recency weighting:** `w(t) = exp(-λ · age_days)` with λ≈0.01
   - Half-life ~70 trading days
   - Combine: `final_weight = recency × regime_match × performance`

3. **Elastic Weight Consolidation (EWC):** Protect learned patterns
   - `L = L_new + λ Σ_i F_i(θ_i - θ*_i)²`
   - Prevents forgetting bull patterns while learning bear behavior

### Causal TCN Encoder Architecture

```python
class CausalTCNEncoder(nn.Module):
    """Temporal Convolutional Network with CAUSAL (left-only) padding."""
    hidden_dim: int = 128
    num_layers: int = 4
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):  # x: [batch, time, features]
        for i in range(self.num_layers):
            dilation = 2 ** i
            pad = (self.kernel_size - 1) * dilation
            x_padded = jnp.pad(x, ((0,0), (pad,0), (0,0)))  # LEFT-PAD ONLY!
            x = nn.Conv(self.hidden_dim, (self.kernel_size,),
                       kernel_dilation=(dilation,))(x_padded)
            x = nn.gelu(x)
        return x[:, -1, :]  # Final embedding
```

**Critical:** Left-padding ensures output at time t depends ONLY on inputs ≤t.

### LUPI: Learning Using Privileged Information

The expert has access to privileged space X* (future prices), student only has X (current data).

```
Expert sees:   (x, x*, y)  where x* = future prices, y = optimal action
Student sees:  (x, y)      and must find latent precursors in x that predict y

The mapping f(x) approximates f*(x*) by learning CAUSAL features.
```

### DQfD Loss Function

```python
L_Total = L_TD + λ₁·L_n_step + λ₂·L_BC + λ₃·L_L2

# With Large Margin Classification for BC:
L_BC(s, a*) = max_a[Q(s,a) + l(a*,a)] - Q(s, a*)
# Where l(a*,a) = 0 if a=a*, else margin δ

# Forces Q(expert_action) to exceed ALL other actions by margin δ
```

### Conservative Q-Learning (CQL) for OOD Safety

```python
# Penalize Q-values of out-of-distribution actions
L_CQL = α · Σ_s [ log Σ_a exp(Q(s,a)) - Q(s, a_expert) ]
```

This pushes down values of non-expert actions → "Risk Aversion" prior.

### Failure Modes and Mitigations

| Failure Mode | Mechanism | Mitigation |
|--------------|-----------|------------|
| BC limiting improvement | High BC weight prevents exceeding demos | Q-Filter |
| Covariate shift | Policy visits OOD states | DAgger online collection |
| Demo overfitting | Memorizes limited data | L2 (10⁻⁵), dropout, early stop |
| Regime mismatch | Past demos mislead current | Regime tagging + recency + EWC |
| Hindsight goal gap | Training/deploy goal mismatch | Goal coverage curriculum |
| Model error | Dynamics diverge | EfficientZero consistency loss |

### MuZero Integration: Reanalyse

```
Stored trajectory → Current network runs MCTS → Improved targets
                                ↓
                    Policy learns to exceed original demos
                                ↓
                    At deployment: MCTS uses only current obs
```

**ROSMO alternative:** One-step lookahead achieves 5.6% compute of full MCTS with competitive performance.

---

## Related Literature

### Core Papers
1. **DAgger** (Ross et al., 2011) - Dataset Aggregation for imitation learning
2. **HER** (Andrychowicz et al., 2017) - Hindsight Experience Replay
3. **GAIL** (Ho & Ermon, 2016) - Generative Adversarial Imitation Learning
4. **DQfD** (Hester et al., 2018) - Deep Q-learning from Demonstrations
5. **DDPGfD** (Nair et al., 2018) - Overcoming Exploration with Demonstrations

### Relevant for Trading
6. **Decision Transformer** (Chen et al., 2021) - Sequence modeling for RL
7. **MuZero Unplugged** (Schrittwieser et al., 2021) - Offline MuZero
8. **Conservative Q-Learning** (Kumar et al., 2020) - Safe offline learning
9. **PPO-HER for Portfolio** (Zhang & Xie, 2025) - HER in trading

### Causality and Representation
10. **LUPI** (Vapnik & Izmailov, 2015) - Learning Using Privileged Information
11. **TF-C** (NeurIPS 2022) - Time-Frequency Consistency
12. **Proto-RL** (Yarats et al., 2021) - Prototype-based exploration

---

## Implementation Order (Approved)

### Phase 1: Core Infrastructure ✅ COMPLETE
| Task | Status | Files |
|------|--------|-------|
| Create `expert_buffer.py` with `UnifiedIndexBuffer` | ✅ | `mcts_trader/jax_agent/expert_buffer.py` |
| Implement `PriceCache` with memory-mapping | ✅ | `mcts_trader/jax_agent/price_cache.py` |
| Create `observation_encoder.py` for encoding | ✅ | `mcts_trader/jax_agent/observation_encoder.py` |
| Test memory usage (<60 MB buffers) | ✅ | `tools/test_expert_buffer.py` |

**Test Results (2026-01-30):**
- Memory at scale: 51.15 MB (4459x reduction from 228 GB)
- Sample batch: ~5 ms
- Encode batch: ~1 ms (after JIT warmup)

### Phase 2: Training Integration
| Task | Status | Files |
|------|--------|-------|
| Add BC loss with Q-filter to trainer | ⬜ | `mcts_trader/jax_agent/trainer.py` |
| Hook episode completion in self_play.py | ⬜ | `mcts_trader/training/self_play.py` |
| Add DS24 metrics logging | ⬜ | `trainer.py`, `train_jax_fast.py` |

```python
# Required DS24 metrics:
'expert/demo_count'
'expert/acceptance_rate'  # argmax(policy) == expert_action
'expert/bc_loss'
'expert/bc_weight'
'expert/avg_confidence'
'expert/avg_return'
```

### Phase 3: Curriculum Integration
| Task | Status | Files |
|------|--------|-------|
| Add curriculum reset callback | ⬜ | `train_jax_fast.py` |
| Integration test on L0→L1→L2 | ⬜ | |

```python
# train_jax_fast.py curriculum callback
if new_level > old_level:
    expert_buffer.reset_for_level(new_level, keep_fraction=0.25)
```

### Phase 4: Optional Enhancements
| Task | Status | Files |
|------|--------|-------|
| HER relabeling for failed trades | ⬜ | `expert_buffer.py` |
| Pattern mining for recurring setups | ⬜ | New module |

---

## Readiness Assessment (2026-01-30)

| Component | Status | Notes |
|-----------|--------|-------|
| UnifiedIndexBuffer | ✅ IMPLEMENTED | `mcts_trader/jax_agent/expert_buffer.py` |
| PriceCache | ✅ IMPLEMENTED | `mcts_trader/jax_agent/price_cache.py` |
| ObservationEncoder | ✅ IMPLEMENTED | `mcts_trader/jax_agent/observation_encoder.py` |
| Unit Tests | ✅ PASSING | `tools/test_expert_buffer.py` |
| BC loss with Q-filter | ✅ Ready | DQfD/DDPGfD patterns in doc |
| Integration points | ⚠️ Phase 2 | self_play.py, trainer.py hooks |
| Curriculum callbacks | ⚠️ Phase 3 | reset_for_level() integration |
| DS24 metrics | ⚠️ Phase 2 | Logging hooks |

**Phase 1: COMPLETE** ✅
- Memory: 51 MB (4459x reduction)
- Performance: 6ms/batch

**Quality: 9/10** - Well-researched, detailed specification
**Readiness: 8/10** - Core implemented and tested
**Risk: Low** - Core approach sound, validated by literature

---

## Notes

- Keep old AMAX code but **disabled by default** - useful for comparison
- Expert buffer should be **saveable/loadable** for curriculum continuity
- Consider **memory mapping** for large buffers (>100k samples)
- Log buffer statistics: size, avg confidence, episode diversity

---

## JAX/Flax Implementation Patterns (Extracted from Reference Code)

### 1. Large Margin Classification Loss (from DQfD)

Source: `docs/Hindsight-optimal/DQfD-master/deep_q_networks.py:222-225`

```python
# JAX/Flax adaptation of DQfD large margin loss
def compute_large_margin_loss(
    q_values: jnp.ndarray,      # (batch, num_actions)
    expert_actions: jnp.ndarray, # (batch,) - action indices
    margin: float = 0.8
) -> jnp.ndarray:
    """
    Forces Q(expert_action) > Q(other_actions) + margin.

    The expert action should be the maximum when we add margin to all OTHER actions.
    """
    batch_size, num_actions = q_values.shape

    # Create margin mask: 0 for expert action, margin for others
    # one_hot with on_value=0, off_value=margin
    margin_mask = jax.nn.one_hot(expert_actions, num_actions)
    margin_mask = margin * (1 - margin_mask)  # margin for non-expert, 0 for expert

    # Large margin target: max over (Q + margin_mask)
    large_margin_targets = jnp.max(q_values + margin_mask, axis=1)

    # Expert Q-values
    expert_q = jnp.sum(q_values * jax.nn.one_hot(expert_actions, num_actions), axis=1)

    # Loss: expert should exceed the margin-adjusted maximum
    # L = |expert_q - large_margin_targets| or max(0, large_margin_targets - expert_q)
    loss = jnp.abs(expert_q - large_margin_targets)

    return loss
```

### 2. Q-Filter for BC Loss (from DDPGfD)

Source: `docs/imprinting/compass_artifact_*.md`

```python
def compute_qfiltered_bc_loss(
    policy_logits: jnp.ndarray,  # (batch, num_actions)
    expert_actions: jnp.ndarray,  # (batch,)
    q_values: jnp.ndarray,        # (batch, num_actions)
) -> jnp.ndarray:
    """
    BC loss only applied when Q(expert_action) > Q(policy_action).

    This auto-decays BC influence as policy improves beyond demonstrations.
    """
    # Current policy's chosen actions
    policy_actions = jnp.argmax(policy_logits, axis=-1)

    # Q-values for expert and policy actions
    q_expert = jnp.sum(q_values * jax.nn.one_hot(expert_actions, q_values.shape[-1]), axis=-1)
    q_policy = jnp.sum(q_values * jax.nn.one_hot(policy_actions, q_values.shape[-1]), axis=-1)

    # Q-filter mask: only apply BC when expert is better
    q_filter_mask = (q_expert > q_policy).astype(jnp.float32)

    # Standard BC loss (cross-entropy)
    bc_loss = optax.softmax_cross_entropy_with_integer_labels(policy_logits, expert_actions)

    # Apply Q-filter
    filtered_bc_loss = bc_loss * q_filter_mask

    return filtered_bc_loss
```

### 3. Prioritized Experience Replay with Demo Protection (from DDPGfD)

Source: `docs/Hindsight-optimal/DDPGfD-main/replay_memory.py:200-210`

```python
@dataclass
class DualReplayBuffer:
    """
    Dual buffer architecture: protected demo buffer + FIFO agent buffer.

    Demo transitions are NEVER overwritten (protected zone).
    """
    demo_buffer_size: int = 50000    # Fixed demo capacity
    agent_buffer_size: int = 500000  # FIFO agent capacity

    def __post_init__(self):
        self.demo_count = 0
        self.agent_start_idx = self.demo_buffer_size
        self.next_agent_idx = self.agent_start_idx

        # Storage
        total_size = self.demo_buffer_size + self.agent_buffer_size
        self.observations = np.zeros((total_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((total_size,), dtype=np.int32)
        self.rewards = np.zeros((total_size,), dtype=np.float32)
        self.priorities = np.ones((total_size,), dtype=np.float32)

        # Priority trees for efficient sampling
        self.sum_tree = SumTree(total_size)
        self.min_tree = MinTree(total_size)

    def add_demo(self, obs, action, reward, confidence):
        """Add to protected demo zone."""
        if self.demo_count >= self.demo_buffer_size:
            return  # Demo buffer full - don't overwrite

        idx = self.demo_count
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward

        # Demos get priority boost
        priority = confidence + 0.001  # ε_D boost
        self.priorities[idx] = priority
        self.sum_tree.update(idx, priority ** 0.6)  # α = 0.6
        self.min_tree.update(idx, priority ** 0.6)

        self.demo_count += 1

    def add_agent(self, obs, action, reward, td_error):
        """Add to FIFO agent zone."""
        idx = self.next_agent_idx

        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward

        priority = np.abs(td_error) + 1e-6
        self.priorities[idx] = priority
        self.sum_tree.update(idx, priority ** 0.6)
        self.min_tree.update(idx, priority ** 0.6)

        # FIFO: wrap around but SKIP demo zone
        self.next_agent_idx += 1
        if self.next_agent_idx >= self.demo_buffer_size + self.agent_buffer_size:
            self.next_agent_idx = self.agent_start_idx  # Skip back over demo zone

    def sample_batch(self, batch_size, demo_ratio=0.125, beta=0.4):
        """
        Sample mixed batch: demo_ratio from demos, rest from agent buffer.

        Default: 128 demo + 896 agent = 1024 batch
        """
        n_demo = int(batch_size * demo_ratio)
        n_agent = batch_size - n_demo

        # Sample from demo zone (proportional to priority)
        demo_indices = self._sample_proportional(0, self.demo_count, n_demo)

        # Sample from agent zone
        agent_indices = self._sample_proportional(
            self.agent_start_idx,
            self.next_agent_idx,
            n_agent
        )

        indices = np.concatenate([demo_indices, agent_indices])

        # Importance sampling weights
        weights = self._compute_is_weights(indices, beta)

        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'weights': weights,
            'indices': indices,
            'is_demo': indices < self.demo_buffer_size,
        }
```

### 4. Segment Tree for Efficient PER (from DDPGfD)

Source: `docs/Hindsight-optimal/DDPGfD-main/replay_memory.py:50-167`

```python
class SumTree:
    """
    Binary segment tree for O(log n) priority sum queries and updates.

    Stores priorities in leaves, internal nodes store partial sums.
    """
    def __init__(self, capacity):
        # Capacity must be power of 2
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        # Tree: 2*capacity nodes (1-indexed, leaves at capacity:2*capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    def update(self, idx, priority):
        """Update priority at leaf idx."""
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority

        # Propagate up
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] = self.tree[2*tree_idx] + self.tree[2*tree_idx + 1]
            tree_idx //= 2

    def sum(self, start=0, end=None):
        """Sum priorities in range [start, end)."""
        if end is None:
            end = self.capacity
        return self._reduce(start, end - 1, 1, 0, self.capacity - 1)

    def _reduce(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce(start, end, 2*node, node_start, mid)
        elif mid + 1 <= start:
            return self._reduce(start, end, 2*node + 1, mid + 1, node_end)
        else:
            return (self._reduce(start, mid, 2*node, node_start, mid) +
                    self._reduce(mid + 1, end, 2*node + 1, mid + 1, node_end))

    def find_prefixsum_idx(self, prefixsum):
        """
        Find leaf idx where cumsum(priorities[0:idx]) <= prefixsum.

        Used for proportional sampling.
        """
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = left + 1
        return idx - self.capacity
```

### 5. MuZero Reanalyse Buffer Structure (from MuZero-unplugged)

Source: `docs/Hindsight-optimal/Muzero-unplugged-main/replay_buffer.py:267-342`

```python
@dataclass
class DemonstrationBuffer:
    """
    Permanent buffer for hindsight-optimal demonstrations.

    Unlike ReanalyseBuffer, this NEVER accepts new games from self-play.
    Only loads from pre-computed demonstrations.
    """
    max_size: int = 50000
    keep_after_sample: bool = True  # Don't delete after sampling

    def __post_init__(self):
        self.buffer = []
        self.priorities = []

    def save_game(self, game):
        """Disabled - demonstrations are loaded, not generated."""
        pass  # Intentionally no-op

    def load_demonstrations(self, path: str):
        """Load pre-computed hindsight-optimal demonstrations."""
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)

        # Compute priorities based on episode return
        for game in self.buffer:
            episode_return = sum(game.rewards)
            self.priorities.append(max(episode_return, 0.1))

    def sample_game(self):
        """Sample a demonstration game for reanalyze."""
        if len(self.buffer) == 0:
            return None

        # Priority-weighted sampling
        probs = np.array(self.priorities) / sum(self.priorities)
        idx = np.random.choice(len(self.buffer), p=probs)

        game = self.buffer[idx]
        if not self.keep_after_sample:
            self.buffer.pop(idx)
            self.priorities.pop(idx)

        return game


class HighestRewardBuffer:
    """Keep only the highest-reward episodes for expert demonstrations."""

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []

    def save_game(self, game):
        episode_return = sum(game.rewards)

        if len(self.buffer) == 0:
            self.buffer.append((episode_return, game))
        elif episode_return > self._min_return():
            self.buffer.append((episode_return, game))
            self.buffer.sort(key=lambda x: x[0], reverse=True)
            if len(self.buffer) > self.max_size:
                self.buffer.pop()

    def _min_return(self):
        return self.buffer[-1][0] if self.buffer else float('-inf')
```

### 6. Consistency Loss (from EfficientZero)

Source: `docs/Hindsight-optimal/EfficientZeroV2-main/ez/utils/loss.py:16-22`

```python
def cosine_similarity_loss(f1: jnp.ndarray, f2: jnp.ndarray) -> jnp.ndarray:
    """
    Self-supervised consistency loss between predicted and observed representations.

    Used to align dynamics model predictions with actual observations.
    """
    # L2 normalize
    f1 = f1 / (jnp.linalg.norm(f1, axis=-1, keepdims=True) + 1e-5)
    f2 = f2 / (jnp.linalg.norm(f2, axis=-1, keepdims=True) + 1e-5)

    # Negative cosine similarity (we minimize, so negate)
    return -(f1 * f2).sum(axis=-1)


def compute_consistency_loss(
    model,
    observations: jnp.ndarray,    # (batch, T, obs_dim)
    actions: jnp.ndarray,          # (batch, T)
    unroll_steps: int = 5
) -> jnp.ndarray:
    """
    EfficientZero-style consistency loss.

    Compare dynamics model predictions to actual encoded observations.
    """
    batch_size = observations.shape[0]

    # Encode initial observation
    state = model.representation(observations[:, 0])

    total_loss = jnp.zeros(batch_size)

    for t in range(unroll_steps):
        # Dynamics model predicts next state
        state, _ = model.dynamics(state, actions[:, t])

        # Encode actual next observation (no gradient)
        gt_state = jax.lax.stop_gradient(model.representation(observations[:, t + 1]))

        # Project both for comparison
        proj_pred = model.projection(state)
        proj_gt = model.projection(gt_state)

        total_loss += cosine_similarity_loss(proj_pred, proj_gt)

    return total_loss / unroll_steps
```

### 7. HER Relabeling for Trading (JAX-compatible)

Source: Adapted from `docs/imprinting/compass_artifact_*.md`

```python
def her_relabel_trading(
    trajectory: Dict[str, jnp.ndarray],
    k: int = 4,
    strategy: str = 'future'
) -> List[Dict[str, jnp.ndarray]]:
    """
    Hindsight Experience Replay for trading.

    Relabels failed trades with achieved returns as successful goals.

    Args:
        trajectory: {observations, actions, returns, target_return}
        k: number of alternative goals to sample
        strategy: 'future' (sample from future steps) or 'episode' (sample from any step)

    Returns:
        List of relabeled trajectories
    """
    T = len(trajectory['actions'])
    achieved_returns = trajectory['returns']  # (T,) cumulative returns at each step

    relabeled = [trajectory]  # Keep original

    for t in range(T):
        if strategy == 'future':
            # Sample from steps AFTER current (HER-future)
            future_indices = list(range(t + 1, T))
        else:
            # Sample from entire episode (HER-episode)
            future_indices = list(range(T))

        if len(future_indices) == 0:
            continue

        # Sample k achieved returns as alternative goals
        sample_k = min(k, len(future_indices))
        sampled_indices = np.random.choice(future_indices, sample_k, replace=False)

        for goal_idx in sampled_indices:
            new_goal = achieved_returns[goal_idx]

            # Reward: sparse binary
            # 1 if we achieved the goal, 0 otherwise
            achieved_at_t = achieved_returns[t]
            new_reward = 1.0 if achieved_at_t >= new_goal else 0.0

            relabeled_traj = {
                'observations': trajectory['observations'][:t+1],
                'actions': trajectory['actions'][:t+1],
                'target_return': new_goal,
                'reward': new_reward,
            }
            relabeled.append(relabeled_traj)

    return relabeled
```

### 8. Combined BC + RL Loss Function (DQfD Style)

```python
def compute_combined_loss(
    model: nn.Module,
    params: dict,
    batch: Dict[str, jnp.ndarray],
    expert_batch: Dict[str, jnp.ndarray],
    config: CausalExpertConfig,
    training_step: int
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """
    DQfD-style combined loss: TD + n-step + BC + L2.

    L = L_TD + λ₁·L_n_step + λ₂·L_BC + λ₃·L_L2
    """
    # ====== 1. TD Loss (standard RL) ======
    q_values = model.apply(params, batch['observations'])
    target_q = batch['rewards'] + config.gamma * batch['next_q_values']
    td_loss = optax.huber_loss(
        q_values[jnp.arange(len(batch['actions'])), batch['actions']],
        target_q
    )

    # ====== 2. N-step TD Loss ======
    if config.n_step > 1:
        n_step_return = batch['n_step_returns']
        n_step_q = batch['n_step_next_q'] * (config.gamma ** config.n_step)
        n_step_target = n_step_return + n_step_q
        n_step_loss = optax.huber_loss(
            q_values[jnp.arange(len(batch['actions'])), batch['actions']],
            n_step_target
        )
    else:
        n_step_loss = jnp.zeros_like(td_loss)

    # ====== 3. Expert BC Loss ======
    if expert_batch is not None and config.bc_weight > 0:
        expert_q = model.apply(params, expert_batch['observations'])

        if config.use_q_filter:
            # Q-Filter: only apply BC when demo is better
            bc_loss = compute_qfiltered_bc_loss(
                expert_q,
                expert_batch['actions'],
                expert_q
            )
        else:
            # Large margin classification loss
            bc_loss = compute_large_margin_loss(
                expert_q,
                expert_batch['actions'],
                margin=config.large_margin
            )

        # Weight by demo confidence
        bc_loss = bc_loss * expert_batch['confidence']
    else:
        bc_loss = jnp.zeros(1)

    # ====== 4. L2 Regularization ======
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    # ====== 5. BC Weight Schedule ======
    # Decay BC weight over training
    bc_weight = config.bc_weight_initial * jnp.exp(
        -config.bc_decay_rate * training_step / config.bc_decay_steps
    )
    bc_weight = jnp.maximum(bc_weight, config.bc_weight_final)

    # ====== Combined Loss ======
    total_loss = (
        td_loss.mean() * config.td_weight +
        n_step_loss.mean() * config.n_step_weight +
        bc_loss.mean() * bc_weight +
        l2_loss * config.l2_weight
    )

    metrics = {
        'td_loss': td_loss.mean(),
        'n_step_loss': n_step_loss.mean(),
        'bc_loss': bc_loss.mean(),
        'bc_weight': bc_weight,
        'l2_loss': l2_loss,
        'total_loss': total_loss,
    }

    return total_loss, metrics
```

### 9. Causal TCN Encoder (Flax)

```python
import flax.linen as nn

class CausalTCNBlock(nn.Module):
    """Single causal TCN block with residual connection."""
    hidden_dim: int
    kernel_size: int
    dilation: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        # x: (batch, time, features)

        # Causal padding: LEFT ONLY
        pad = (self.kernel_size - 1) * self.dilation
        x_padded = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)), mode='constant')

        # Dilated causal conv
        h = nn.Conv(
            self.hidden_dim,
            (self.kernel_size,),
            kernel_dilation=(self.dilation,),
            padding='VALID'
        )(x_padded)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not training)

        # Second conv in block
        h_padded = jnp.pad(h, ((0, 0), (pad, 0), (0, 0)), mode='constant')
        h = nn.Conv(
            self.hidden_dim,
            (self.kernel_size,),
            kernel_dilation=(self.dilation,),
            padding='VALID'
        )(h_padded)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not training)

        # Residual connection (project if needed)
        if x.shape[-1] != self.hidden_dim:
            x = nn.Dense(self.hidden_dim)(x)

        return x + h


class CausalTCNEncoder(nn.Module):
    """
    Causal Temporal Convolutional Network for pattern encoding.

    CRITICAL: Uses LEFT-ONLY padding to ensure output[t] depends ONLY on inputs[0:t].
    This maintains causality - no future information leaks.
    """
    hidden_dim: int = 128
    num_layers: int = 4
    kernel_size: int = 3
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        # x: (batch, time, features)

        # Initial projection
        h = nn.Dense(self.hidden_dim)(x)

        # Stack of causal TCN blocks with increasing dilation
        for i in range(self.num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            h = CausalTCNBlock(
                hidden_dim=self.hidden_dim,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout_rate=self.dropout_rate
            )(h, training=training)

        # Return final timestep embedding
        return h[:, -1, :]  # (batch, hidden_dim)
```

### 10. Expert Buffer Integration with Training Loop

```python
def train_step_with_expert(
    state: TrainState,
    batch: Dict,
    expert_buffer: DualReplayBuffer,
    config: CausalExpertConfig,
    rng: jax.random.PRNGKey
):
    """
    Single training step with expert demonstrations.

    Samples from both agent buffer and expert buffer,
    computes combined loss, updates model.
    """
    rng, expert_rng = jax.random.split(rng)

    # Sample expert batch
    expert_batch = expert_buffer.sample_batch(
        batch_size=config.expert_batch_size,
        demo_ratio=1.0,  # All from demo zone
        beta=config.per_beta
    )

    def loss_fn(params):
        return compute_combined_loss(
            model=state.model,
            params=params,
            batch=batch,
            expert_batch=expert_batch,
            config=config,
            training_step=state.step
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Gradient clipping
    grads = optax.clip_by_global_norm(grads, config.max_grad_norm)

    # Update
    state = state.apply_gradients(grads=grads)

    # Update priorities for sampled transitions
    td_errors = metrics.get('td_errors', None)
    if td_errors is not None:
        expert_buffer.update_priorities(
            batch['indices'],
            np.abs(td_errors) + 1e-6
        )

    return state, metrics
```

---

## Memory-Efficient Buffer Architecture

### The Problem: Full Observations Don't Fit

```python
# Current observation shape
obs_shape = (27, 60, 64)  # channels × window × assets
obs_bytes = 27 * 60 * 64 * 4  # = 414,720 bytes (~400KB) per sample

# Naive buffer sizes
demo_buffer_50k = 50_000 * 400KB = 20GB   # ❌ Won't fit
agent_buffer_500k = 500_000 * 400KB = 200GB  # ❌ Impossible
```

### Solution: Index-Based Storage + On-Demand Reconstruction

Store **indices** (12 bytes) instead of **observations** (400KB). Reconstruct on-demand from a shared price cache.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MEMORY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    CPU MEMORY                                 │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  PRICE CACHE (Memory-Mapped)                           │  │  │
│  │  │  - Raw price data: (N_episodes, T_max, 64_assets, 5)   │  │  │
│  │  │  - OHLCV per asset per timestep                        │  │  │
│  │  │  - Memory-mapped from disk (~2-10GB on disk)           │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  │                                                               │  │
│  │  ┌─────────────────────┐    ┌─────────────────────┐          │  │
│  │  │  DEMO INDEX BUFFER  │    │  AGENT INDEX BUFFER │          │  │
│  │  │  (Protected Zone)   │    │  (FIFO Zone)        │          │  │
│  │  │                     │    │                     │          │  │
│  │  │  50k × 32 bytes     │    │  500k × 32 bytes    │          │  │
│  │  │  = 1.6 MB           │    │  = 16 MB            │          │  │
│  │  └─────────────────────┘    └─────────────────────┘          │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  PRIORITY TREES (SumTree + MinTree)                    │  │  │
│  │  │  550k entries × 8 bytes × 2 trees = 8.8 MB             │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              │ Sample indices                       │
│                              │ Reconstruct batch                    │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    GPU MEMORY (24GB)                          │  │
│  │                                                               │  │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐    │  │
│  │  │  MODEL              │  │  BATCH TENSORS              │    │  │
│  │  │  params: 60MB       │  │  obs: 1024 × 400KB = 400MB  │    │  │
│  │  │  grads: 60MB        │  │  targets: ~50MB             │    │  │
│  │  │  optim: 120MB       │  │  activations: ~500MB        │    │  │
│  │  │  ─────────────      │  │  ─────────────────────      │    │  │
│  │  │  Total: ~250MB      │  │  Total: ~1GB                │    │  │
│  │  └─────────────────────┘  └─────────────────────────────┘    │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │  JIT CACHE + MCTS + ENVS: ~2-3GB                        │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  │  TOTAL GPU: ~4-5GB (with large safety margin!)               │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Structures

#### 1. Index Entry (32 bytes per sample)

```python
import numpy as np
from dataclasses import dataclass

# Compact index entry - stores WHERE to find data, not the data itself
INDEX_DTYPE = np.dtype([
    ('episode_id', np.uint32),      # 4 bytes - which episode
    ('step', np.uint16),            # 2 bytes - which step in episode
    ('action', np.uint16),          # 2 bytes - action taken (0-319)
    ('optimal_action', np.uint16),  # 2 bytes - hindsight-optimal action
    ('reward', np.float16),         # 2 bytes - immediate reward
    ('confidence', np.float16),     # 2 bytes - how optimal (for demos)
    ('episode_return', np.float16), # 2 bytes - total episode return
    ('regime_tag', np.uint8),       # 1 byte - market regime (0-7)
    ('is_demo', np.uint8),          # 1 byte - from demo buffer?
    ('timestamp', np.uint32),       # 4 bytes - when added (for recency)
    ('priority', np.float32),       # 4 bytes - current priority
    ('_padding', np.uint32),        # 4 bytes - alignment to 32
])
# Total: 32 bytes per entry

# Buffer sizes
DEMO_CAPACITY = 50_000      # 50k × 32 = 1.6 MB
AGENT_CAPACITY = 500_000    # 500k × 32 = 16 MB
# Combined: ~18 MB (vs 220 GB with full observations!)
```

#### 2. Price Cache (Memory-Mapped)

```python
import numpy as np
from pathlib import Path

class PriceCache:
    """
    Memory-mapped price data for on-demand observation reconstruction.

    Stored on disk, accessed like array. OS handles paging.
    Only actively-used pages loaded into RAM.
    """

    def __init__(
        self,
        cache_path: Path,
        max_episodes: int = 10_000,
        max_steps: int = 1000,
        n_assets: int = 64,
        n_features: int = 5,  # OHLCV
    ):
        self.cache_path = cache_path
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.n_assets = n_assets
        self.n_features = n_features

        # Shape: (episodes, steps, assets, features)
        self.shape = (max_episodes, max_steps, n_assets, n_features)

        # Memory-map the file (creates if not exists)
        self.prices = np.memmap(
            cache_path / 'prices.dat',
            dtype=np.float32,
            mode='r+' if (cache_path / 'prices.dat').exists() else 'w+',
            shape=self.shape
        )

        # Episode metadata (small, kept in RAM)
        self.episode_lengths = np.zeros(max_episodes, dtype=np.uint16)
        self.episode_start_dates = np.zeros(max_episodes, dtype=np.uint32)
        self.next_episode_id = 0

    def add_episode(self, prices: np.ndarray, start_date: int) -> int:
        """
        Add episode price data to cache.

        Args:
            prices: (T, n_assets, n_features) price data
            start_date: Unix timestamp of episode start

        Returns:
            episode_id for referencing this data
        """
        if self.next_episode_id >= self.max_episodes:
            raise RuntimeError("Price cache full - increase max_episodes or clear old data")

        episode_id = self.next_episode_id
        T = prices.shape[0]

        self.prices[episode_id, :T] = prices
        self.episode_lengths[episode_id] = T
        self.episode_start_dates[episode_id] = start_date

        self.next_episode_id += 1

        # Flush to disk periodically
        if episode_id % 100 == 0:
            self.prices.flush()

        return episode_id

    def get_observation_window(
        self,
        episode_id: int,
        step: int,
        window_size: int = 60,
    ) -> np.ndarray:
        """
        Reconstruct observation window from price cache.

        Returns:
            (window_size, n_assets, n_features) array
        """
        start_idx = max(0, step - window_size + 1)
        end_idx = step + 1

        window = self.prices[episode_id, start_idx:end_idx]

        # Pad if near episode start
        if start_idx == 0 and (step - window_size + 1) < 0:
            pad_size = window_size - window.shape[0]
            padding = np.tile(window[0:1], (pad_size, 1, 1))
            window = np.concatenate([padding, window], axis=0)

        return window

    def get_batch_observations(
        self,
        episode_ids: np.ndarray,
        steps: np.ndarray,
        window_size: int = 60,
    ) -> np.ndarray:
        """
        Vectorized batch observation reconstruction.

        Args:
            episode_ids: (batch_size,) episode indices
            steps: (batch_size,) step indices

        Returns:
            (batch_size, window_size, n_assets, n_features) array
        """
        batch_size = len(episode_ids)
        obs = np.zeros(
            (batch_size, window_size, self.n_assets, self.n_features),
            dtype=np.float32
        )

        for i, (ep_id, step) in enumerate(zip(episode_ids, steps)):
            obs[i] = self.get_observation_window(ep_id, step, window_size)

        return obs
```

#### 3. Unified Index Buffer with Protected Demo Zone

```python
import numpy as np
from typing import Optional, Tuple, Dict
import threading

class UnifiedIndexBuffer:
    """
    Memory-efficient replay buffer using index storage.

    Layout:
    [0 ... demo_capacity-1] = Protected demo zone (never overwritten)
    [demo_capacity ... total_capacity-1] = FIFO agent zone

    Memory usage:
    - 50k demos × 32 bytes = 1.6 MB
    - 500k agent × 32 bytes = 16 MB
    - Priority trees = ~9 MB
    - Total: ~27 MB (vs 220 GB with full observations!)
    """

    def __init__(
        self,
        price_cache: PriceCache,
        demo_capacity: int = 50_000,
        agent_capacity: int = 500_000,
        window_size: int = 60,
        per_alpha: float = 0.6,
        per_beta_init: float = 0.4,
        per_beta_final: float = 1.0,
        per_beta_steps: int = 100_000,
    ):
        self.price_cache = price_cache
        self.demo_capacity = demo_capacity
        self.agent_capacity = agent_capacity
        self.total_capacity = demo_capacity + agent_capacity
        self.window_size = window_size

        # PER parameters
        self.per_alpha = per_alpha
        self.per_beta = per_beta_init
        self.per_beta_init = per_beta_init
        self.per_beta_final = per_beta_final
        self.per_beta_steps = per_beta_steps
        self.update_count = 0

        # Index storage (tiny!)
        self.indices = np.zeros(self.total_capacity, dtype=INDEX_DTYPE)

        # Counters
        self.demo_count = 0
        self.agent_count = 0
        self.agent_write_idx = demo_capacity  # Start after demo zone

        # Priority trees (CPU-side, efficient sampling)
        tree_capacity = 1
        while tree_capacity < self.total_capacity:
            tree_capacity *= 2
        self.sum_tree = np.zeros(2 * tree_capacity, dtype=np.float64)
        self.min_tree = np.full(2 * tree_capacity, np.inf, dtype=np.float64)
        self.tree_capacity = tree_capacity

        self.max_priority = 1.0

        # Thread safety
        self._lock = threading.Lock()

    # ==================== ADDING SAMPLES ====================

    def add_demo(
        self,
        episode_id: int,
        step: int,
        action: int,
        optimal_action: int,
        reward: float,
        confidence: float,
        episode_return: float,
        regime_tag: int = 0,
    ) -> bool:
        """Add to protected demo zone. Returns False if full."""
        with self._lock:
            if self.demo_count >= self.demo_capacity:
                return False

            idx = self.demo_count
            self._set_entry(
                idx, episode_id, step, action, optimal_action,
                reward, confidence, episode_return, regime_tag,
                is_demo=1
            )

            # Demos get priority boost
            priority = confidence + 0.001
            self._update_priority(idx, priority)

            self.demo_count += 1
            return True

    def add_agent(
        self,
        episode_id: int,
        step: int,
        action: int,
        reward: float,
        td_error: float,
        episode_return: float = 0.0,
        regime_tag: int = 0,
    ):
        """Add to FIFO agent zone."""
        with self._lock:
            idx = self.agent_write_idx

            self._set_entry(
                idx, episode_id, step, action, action,  # optimal = action for agent
                reward, 0.0, episode_return, regime_tag,
                is_demo=0
            )

            # Priority from TD error
            priority = np.abs(td_error) + 1e-6
            self._update_priority(idx, priority)

            # FIFO: advance and wrap within agent zone
            self.agent_write_idx += 1
            if self.agent_write_idx >= self.total_capacity:
                self.agent_write_idx = self.demo_capacity

            self.agent_count = min(self.agent_count + 1, self.agent_capacity)

    def _set_entry(
        self, idx, episode_id, step, action, optimal_action,
        reward, confidence, episode_return, regime_tag, is_demo
    ):
        """Set index entry fields."""
        entry = self.indices[idx]
        entry['episode_id'] = episode_id
        entry['step'] = step
        entry['action'] = action
        entry['optimal_action'] = optimal_action
        entry['reward'] = reward
        entry['confidence'] = confidence
        entry['episode_return'] = episode_return
        entry['regime_tag'] = regime_tag
        entry['is_demo'] = is_demo
        entry['timestamp'] = self.update_count

    # ==================== PRIORITY TREE OPERATIONS ====================

    def _update_priority(self, idx: int, priority: float):
        """Update priority in sum and min trees."""
        priority_alpha = priority ** self.per_alpha
        self.indices[idx]['priority'] = priority
        self.max_priority = max(self.max_priority, priority)

        # Update sum tree
        tree_idx = idx + self.tree_capacity
        self.sum_tree[tree_idx] = priority_alpha
        while tree_idx > 1:
            tree_idx //= 2
            self.sum_tree[tree_idx] = (
                self.sum_tree[2 * tree_idx] +
                self.sum_tree[2 * tree_idx + 1]
            )

        # Update min tree
        tree_idx = idx + self.tree_capacity
        self.min_tree[tree_idx] = priority_alpha
        while tree_idx > 1:
            tree_idx //= 2
            self.min_tree[tree_idx] = min(
                self.min_tree[2 * tree_idx],
                self.min_tree[2 * tree_idx + 1]
            )

    def _sample_proportional(self, n: int, start_idx: int, end_idx: int) -> np.ndarray:
        """Sample n indices proportional to priority in [start_idx, end_idx)."""
        indices = np.zeros(n, dtype=np.int32)

        # Get total priority in range (approximate for simplicity)
        total = self.sum_tree[1]  # Root = total sum
        segment = total / n

        for i in range(n):
            # Sample uniformly within segment
            s = np.random.uniform(segment * i, segment * (i + 1))

            # Find leaf with this cumsum
            idx = 1
            while idx < self.tree_capacity:
                left = 2 * idx
                if self.sum_tree[left] > s:
                    idx = left
                else:
                    s -= self.sum_tree[left]
                    idx = left + 1

            buffer_idx = idx - self.tree_capacity

            # Ensure in valid range
            buffer_idx = np.clip(buffer_idx, start_idx, end_idx - 1)
            indices[i] = buffer_idx

        return indices

    # ==================== SAMPLING ====================

    def sample_batch(
        self,
        batch_size: int = 1024,
        demo_ratio: float = 0.125,  # 128/1024 from demos
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Sample mixed batch from demo and agent zones.

        Returns dict with observations reconstructed on-demand.
        """
        total_available = self.demo_count + self.agent_count
        if total_available < batch_size:
            return None

        # Anneal beta
        self.update_count += 1
        self.per_beta = min(
            self.per_beta_final,
            self.per_beta_init +
            (self.per_beta_final - self.per_beta_init) *
            (self.update_count / self.per_beta_steps)
        )

        with self._lock:
            # Determine sample counts
            n_demo = min(int(batch_size * demo_ratio), self.demo_count)
            n_agent = batch_size - n_demo
            n_agent = min(n_agent, self.agent_count)
            n_demo = batch_size - n_agent  # Adjust if not enough agent samples

            # Sample indices
            if n_demo > 0:
                demo_indices = self._sample_proportional(n_demo, 0, self.demo_count)
            else:
                demo_indices = np.array([], dtype=np.int32)

            if n_agent > 0:
                agent_indices = self._sample_proportional(
                    n_agent,
                    self.demo_capacity,
                    self.demo_capacity + self.agent_count
                )
            else:
                agent_indices = np.array([], dtype=np.int32)

            indices = np.concatenate([demo_indices, agent_indices])

            # Get entries
            entries = self.indices[indices]

            # Compute importance sampling weights
            priorities = entries['priority'] ** self.per_alpha
            min_priority = self.min_tree[1]  # Global min
            if min_priority == 0:
                min_priority = 1e-6

            # P(i) = priority_i / sum(priorities)
            total_priority = self.sum_tree[1]
            probs = priorities / total_priority

            # w_i = (N * P(i))^(-beta), normalized by max
            N = total_available
            weights = (N * probs) ** (-self.per_beta)
            max_weight = (N * min_priority / total_priority) ** (-self.per_beta)
            weights = weights / max_weight

            # Reconstruct observations from price cache
            observations = self.price_cache.get_batch_observations(
                entries['episode_id'],
                entries['step'],
                self.window_size
            )

        return {
            'observations': observations,          # (batch, window, assets, features)
            'actions': entries['action'],          # (batch,)
            'optimal_actions': entries['optimal_action'],  # (batch,)
            'rewards': entries['reward'].astype(np.float32),
            'confidence': entries['confidence'].astype(np.float32),
            'is_demo': entries['is_demo'].astype(bool),
            'weights': weights.astype(np.float32),
            'indices': indices,
            'regime_tags': entries['regime_tag'],
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities after training step."""
        with self._lock:
            for idx, td_error in zip(indices, td_errors):
                priority = np.abs(td_error) + 1e-6
                self._update_priority(idx, priority)

    # ==================== STATS ====================

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        return {
            'demo_count': self.demo_count,
            'demo_capacity': self.demo_capacity,
            'demo_fill_ratio': self.demo_count / self.demo_capacity,
            'agent_count': self.agent_count,
            'agent_capacity': self.agent_capacity,
            'agent_fill_ratio': self.agent_count / self.agent_capacity,
            'total_samples': self.demo_count + self.agent_count,
            'per_beta': self.per_beta,
            'max_priority': self.max_priority,
            'memory_mb': (
                self.indices.nbytes / 1e6 +
                self.sum_tree.nbytes / 1e6 +
                self.min_tree.nbytes / 1e6
            ),
        }
```

#### 4. Observation Encoder (Compute Features On-Demand)

```python
import jax
import jax.numpy as jnp

class ObservationEncoder:
    """
    Encode raw price windows into model input features.

    Called on-demand after batch sampling.
    Runs on GPU for efficiency.
    """

    def __init__(
        self,
        n_channels: int = 27,
        window_size: int = 60,
        n_assets: int = 64,
    ):
        self.n_channels = n_channels
        self.window_size = window_size
        self.n_assets = n_assets

    @partial(jax.jit, static_argnums=(0,))
    def encode_batch(
        self,
        prices: jnp.ndarray,  # (batch, window, assets, 5) OHLCV
    ) -> jnp.ndarray:
        """
        Convert raw OHLCV to 27-channel encoded observation.

        Returns: (batch, n_channels, window, assets)
        """
        batch_size = prices.shape[0]

        # Extract OHLCV
        open_p = prices[..., 0]   # (batch, window, assets)
        high_p = prices[..., 1]
        low_p = prices[..., 2]
        close_p = prices[..., 3]
        volume = prices[..., 4]

        # Normalize by first close (relative prices)
        first_close = close_p[:, 0:1, :]  # (batch, 1, assets)
        first_close = jnp.maximum(first_close, 1e-8)

        norm_open = open_p / first_close
        norm_high = high_p / first_close
        norm_low = low_p / first_close
        norm_close = close_p / first_close

        # Returns
        log_returns = jnp.log(close_p[:, 1:] / jnp.maximum(close_p[:, :-1], 1e-8))
        log_returns = jnp.pad(log_returns, ((0,0), (1,0), (0,0)))  # Pad first step

        # Volume normalization (rolling z-score)
        vol_mean = jnp.mean(volume, axis=1, keepdims=True)
        vol_std = jnp.std(volume, axis=1, keepdims=True) + 1e-8
        norm_volume = (volume - vol_mean) / vol_std

        # Technical indicators
        # SMA ratios
        sma_5 = self._rolling_mean(close_p, 5)
        sma_20 = self._rolling_mean(close_p, 20)
        sma_ratio_5 = close_p / jnp.maximum(sma_5, 1e-8)
        sma_ratio_20 = close_p / jnp.maximum(sma_20, 1e-8)

        # Volatility (rolling std of returns)
        volatility = self._rolling_std(log_returns, 20)

        # RSI approximation
        gains = jnp.maximum(log_returns, 0)
        losses = jnp.maximum(-log_returns, 0)
        avg_gain = self._rolling_mean(gains, 14)
        avg_loss = self._rolling_mean(losses, 14) + 1e-8
        rsi = avg_gain / (avg_gain + avg_loss)

        # High-low range
        hl_range = (high_p - low_p) / jnp.maximum(close_p, 1e-8)

        # Stack channels: (batch, channels, window, assets)
        channels = [
            norm_open, norm_high, norm_low, norm_close,  # 4
            log_returns,                                   # 1
            norm_volume,                                   # 1
            sma_ratio_5, sma_ratio_20,                    # 2
            volatility,                                    # 1
            rsi,                                           # 1
            hl_range,                                      # 1
        ]
        # 11 channels so far, add more as needed to reach 27

        # Lagged returns (1, 2, 5, 10, 20 step)
        for lag in [1, 2, 5, 10, 20]:
            lagged = jnp.roll(log_returns, lag, axis=1)
            lagged = lagged.at[:, :lag].set(0)
            channels.append(lagged)
        # +5 = 16 channels

        # Cross-asset momentum rank
        rank = self._cross_asset_rank(log_returns)
        channels.append(rank)
        # +1 = 17 channels

        # Pad to 27 channels with zeros or repeat
        while len(channels) < 27:
            channels.append(jnp.zeros_like(norm_close))

        # Stack and transpose to (batch, channels, window, assets)
        encoded = jnp.stack(channels[:27], axis=1)

        return encoded

    def _rolling_mean(self, x, window):
        """Simple rolling mean."""
        kernel = jnp.ones(window) / window
        # Convolve along time axis
        return jax.lax.conv_general_dilated(
            x[:, :, :, None],  # (batch, time, assets, 1)
            kernel[None, :, None, None],  # (1, window, 1, 1)
            window_strides=(1,),
            padding=[(window-1, 0)],
            dimension_numbers=('NHWC', 'OHWI', 'NHWC')
        )[:, :, :, 0]

    def _rolling_std(self, x, window):
        """Rolling standard deviation."""
        mean = self._rolling_mean(x, window)
        sq_mean = self._rolling_mean(x ** 2, window)
        var = sq_mean - mean ** 2
        return jnp.sqrt(jnp.maximum(var, 1e-8))

    def _cross_asset_rank(self, returns):
        """Rank assets by return at each timestep."""
        # Returns percentile rank (0-1)
        sorted_idx = jnp.argsort(returns, axis=-1)
        ranks = jnp.argsort(sorted_idx, axis=-1)
        return ranks / (returns.shape[-1] - 1)
```

#### 5. Complete Training Loop Integration

```python
def train_epoch_memory_efficient(
    state: TrainState,
    buffer: UnifiedIndexBuffer,
    encoder: ObservationEncoder,
    config: CausalExpertConfig,
    rng: jax.random.PRNGKey,
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Memory-efficient training epoch.

    Observations are reconstructed on-demand, never stored in full.
    """
    metrics_sum = {}
    n_batches = config.batches_per_epoch

    for batch_idx in range(n_batches):
        rng, batch_rng = jax.random.split(rng)

        # 1. Sample indices and reconstruct observations (CPU → GPU transfer)
        batch = buffer.sample_batch(
            batch_size=config.batch_size,
            demo_ratio=config.demo_ratio
        )

        if batch is None:
            continue

        # 2. Move raw prices to GPU and encode
        prices_gpu = jax.device_put(batch['observations'])
        encoded_obs = encoder.encode_batch(prices_gpu)

        # 3. Prepare batch tensors
        train_batch = {
            'observations': encoded_obs,
            'actions': jax.device_put(batch['actions']),
            'optimal_actions': jax.device_put(batch['optimal_actions']),
            'rewards': jax.device_put(batch['rewards']),
            'confidence': jax.device_put(batch['confidence']),
            'is_demo': jax.device_put(batch['is_demo']),
            'weights': jax.device_put(batch['weights']),
        }

        # 4. Training step
        state, step_metrics = train_step(state, train_batch, batch_rng)

        # 5. Update priorities
        td_errors = np.array(step_metrics.get('td_errors', np.zeros(len(batch['indices']))))
        buffer.update_priorities(batch['indices'], td_errors)

        # Accumulate metrics
        for k, v in step_metrics.items():
            if k != 'td_errors':
                metrics_sum[k] = metrics_sum.get(k, 0) + v

    # Average metrics
    metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    metrics.update(buffer.get_stats())

    return state, metrics
```

### Memory Summary

| Component | Size | Location |
|-----------|------|----------|
| Index buffers (550k entries) | 17.6 MB | CPU RAM |
| Priority trees | 8.8 MB | CPU RAM |
| Price cache (10k episodes) | ~12 GB | Disk (memory-mapped) |
| Model + optimizer | 250 MB | GPU |
| Batch tensors (1024 samples) | ~500 MB | GPU |
| JIT cache + envs | ~2 GB | GPU |
| **GPU Total** | **~3 GB** | |
| **CPU RAM** | **~100 MB** | |
| **Disk** | **~12 GB** | |

### Key Benefits

1. **550k samples in 18 MB** vs 220 GB with full observations
2. **On-demand reconstruction** - only batch data on GPU
3. **Memory-mapped price cache** - OS handles paging
4. **Protected demo zone** - demonstrations never overwritten
5. **Efficient priority sampling** - O(log n) with segment trees
6. **GPU-accelerated encoding** - JIT-compiled feature computation

---

## VRAM Considerations

### Memory Budget (for 24GB VRAM) - REVISED

| Component | Estimated Size | Notes |
|-----------|---------------|-------|
| Model + gradients + optimizer | ~250MB | 15M params |
| Batch tensors | ~500MB | 1024 × encoded obs |
| JIT cache | ~1-2GB | Compiled kernels |
| MCTS trees | ~1GB | If using MCTS |
| Parallel envs | ~500MB | 8-16 envs |
| **Total GPU** | **~4-5GB** | Plenty of headroom! |

### CPU/Disk Usage

| Component | Size | Notes |
|-----------|------|-------|
| Index buffers | ~18MB | In-memory |
| Priority trees | ~9MB | In-memory |
| Price cache | ~12GB | On disk, memory-mapped |

---

*Created: 2026-01-30*
*Last Updated: 2026-01-30*
