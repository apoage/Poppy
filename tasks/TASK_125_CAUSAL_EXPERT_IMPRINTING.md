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

## Related Literature

### Core Papers
1. **DAgger** (Ross et al., 2011) - Dataset Aggregation for imitation learning
2. **HER** (Andrychowicz et al., 2017) - Hindsight Experience Replay
3. **GAIL** (Ho & Ermon, 2016) - Generative Adversarial Imitation Learning
4. **Demonstrations in Deep RL** (Hester et al., 2018) - DQfD

### Relevant for Trading
5. **Decision Transformer** (Chen et al., 2021) - Sequence modeling for RL
6. **Offline RL** (Levine et al., 2020) - Learning from fixed datasets
7. **Conservative Q-Learning** (Kumar et al., 2020) - Safe offline learning

### Causality in ML
8. **Causal Inference in Statistics** (Pearl, 2009)
9. **Time Series Causal Discovery** (Runge et al., 2019)

---

## Implementation Timeline

| Phase | Tasks | Estimated Effort |
|-------|-------|------------------|
| 1 | ExpertBuffer class, basic integration | 1-2 sessions |
| 2 | BC loss in trainer, config options | 1 session |
| 3 | Testing and verification | 1 session |
| 4 | Pattern mining (optional) | 1-2 sessions |
| 5 | HER extension (optional) | 1 session |

---

## Notes

- Keep old AMAX code but **disabled by default** - useful for comparison
- Expert buffer should be **saveable/loadable** for curriculum continuity
- Consider **memory mapping** for large buffers (>100k samples)
- Log buffer statistics: size, avg confidence, episode diversity

---

*Created: 2026-01-30*
*Last Updated: 2026-01-30*
