# TASK_119: Optimal Hindsight Path Solver

## Status: PHASE 1 COMPLETE - Light Implementations Done

### Implementation Summary

**File:** `mcts_trader/jax_agent/path_solver.py`

| Method | Time (T=1000) | VRAM | Trades/step | Use Case |
|--------|---------------|------|-------------|----------|
| `greedy_lookahead_path` | 24ms | ~200MB | 3.8 | Fast expert demos |
| `soft_topk_path` | 48ms | ~300MB | 0.5 | Differentiable loss |
| `sinkhorn_topk_path` | 461ms | ~300MB | 3.5 | Balanced, pure JAX |

All methods return `OptimalPath` with:
- `selections`: (T, n) soft/hard selection weights
- `indices`: (T, k) selected asset indices
- `cumulative_pnl`: (T,) cumulative PnL curve
- `total_alpha`: Final captured alpha
- `trade_count`: Number of trades

**Target was 0.1-0.5s → Achieved 24-461ms**

## Goal

Compute the **actual optimal trade sequence** given hindsight returns, not just the upper bound. This provides:
- Expert demonstrations for imitation learning
- Replay buffer signal ("what should I have done?")
- Tactical ground truth, not just ceiling

---

## Why This Matters

**TASK_118 gives us:** `max_alpha ≈ Σ top_k(|returns|)` - a ceiling
**This task gives us:** The actual path to get close to that ceiling

```
TASK_118: "You could have made $1000"
TASK_119: "Here's the exact trades to make $950"
```

The path is valuable for:
1. **Imitation learning**: Train policy to mimic optimal hindsight
2. **Replay ranking**: Prioritize episodes where agent deviated from optimal
3. **Counterfactual analysis**: "If you had held 2 more steps..."
4. **Feature extraction**: Optimal path reveals market structure

---

## The NP-Hard Problem

True optimal portfolio allocation is NP-hard:
- State space: 2^n possible portfolios (n=64 assets)
- Constraints: position limits, transaction costs, capital
- T timesteps → O(2^n × T) naive DP

**But we have advantages:**
- Small k (4-10 positions max) → reduced state
- Transaction costs favor holding → sparse transitions
- Approximate is fine (90% of optimal >> random)

---

## Candidate Algorithms

### 1. Greedy with Lookahead
```python
def greedy_lookahead(returns, k, lookahead=5, cost=0.001):
    """Greedy: at each step, pick action maximizing next L steps."""
    portfolio = set()
    trades = []

    for t in range(T - lookahead):
        # Score each possible action (enter/exit each asset)
        best_action, best_value = None, -inf
        for action in possible_actions(portfolio, k):
            value = simulate(action, returns[t:t+lookahead], cost)
            if value > best_value:
                best_action, best_value = action, value

        portfolio = apply(portfolio, best_action)
        trades.append(best_action)

    return trades
```
- **Complexity:** O(T × n × L) with pruning
- **Quality:** Good for trending markets, may miss reversals

### 2. Dynamic Programming with Beam Search
```python
def beam_dp(returns, k, beam_width=100, cost=0.001):
    """DP with beam pruning - keep top-B states per step."""
    # State: frozenset of held assets
    beam = {frozenset(): 0.0}  # state -> cumulative value

    for t in range(T):
        next_beam = {}
        for state, value in beam.items():
            for action in possible_actions(state, k):
                new_state = apply(state, action)
                new_value = value + reward(state, action, returns[t], cost)
                if new_state not in next_beam or new_value > next_beam[new_state]:
                    next_beam[new_state] = new_value

        # Prune to top-B
        beam = dict(sorted(next_beam.items(), key=lambda x: -x[1])[:beam_width])

    return backtrack(beam)
```
- **Complexity:** O(T × B × n × k)
- **Quality:** Near-optimal for reasonable B

### 3. MCTS Hindsight Search
```python
def mcts_hindsight(returns, k, sims=1000):
    """Use MCTS to search for optimal path (ironic but effective)."""
    # Root: empty portfolio at t=0
    # Actions: enter/exit/hold
    # Reward: known returns (no uncertainty!)

    # Since returns are known, this is just tree search
    # UCB guides exploration of trade sequences
    pass
```
- **Complexity:** O(sims × depth)
- **Quality:** Anytime algorithm, improves with more sims

### 4. Integer Linear Programming
```python
def ilp_optimal(returns, k, cost=0.001):
    """Exact solution via ILP (slow but optimal)."""
    # Variables: x[t,i] ∈ {0,1} = holding asset i at time t
    # Objective: Σ x[t,i] * returns[t,i] - cost * |x[t,i] - x[t-1,i]|
    # Constraints: Σ_i x[t,i] <= k for all t

    # Use CVXPY with GLPK or Gurobi
    pass
```
- **Complexity:** Exponential worst case, but solvers are smart
- **Quality:** Exact optimal

### 5. Reinforcement Learning (offline)
```python
def offline_rl_optimal(returns, k):
    """Train small RL agent on single episode with known returns."""
    # Observation: returns[t:t+window], current_portfolio
    # Action: trade decisions
    # Reward: realized PnL

    # Train for 1000 epochs on single episode
    # Agent learns optimal policy for this specific sequence
    pass
```
- **Complexity:** O(epochs × T)
- **Quality:** Approximate, but learns nuances

---

## Proposed Approach: Hierarchical

1. **Fast approximation** (for all episodes):
   - Greedy with lookahead=10
   - O(T × n × 10) ≈ 1ms per episode

2. **Refined solution** (for top episodes):
   - Beam DP with width=1000
   - O(T × 1000 × n × k) ≈ 100ms per episode

3. **Exact solution** (for analysis):
   - ILP for selected episodes
   - Verify approximation quality

---

## Output Format

```python
@dataclass
class OptimalPath:
    trades: List[Trade]           # Sequence of (time, asset, action)
    portfolio_trajectory: Array   # (T, n_assets) holdings
    cumulative_pnl: Array         # (T,) realized PnL curve
    total_alpha: float            # Final captured alpha
    efficiency: float             # captured / theoretical_max

@dataclass
class Trade:
    time: int
    asset: int
    action: str  # 'enter' | 'exit'
    price: float
    alpha_captured: float  # Return from this trade
```

---

## Integration Points

### 1. Replay Buffer Enhancement
```python
# In replay buffer, store optimal path alongside trajectory
buffer.add(
    trajectory=agent_trajectory,
    optimal_path=compute_optimal_path(returns),
    regret=optimal_path.total_alpha - agent_pnl,
)

# Prioritize high-regret episodes
priority = regret ** priority_exponent
```

### 2. Auxiliary Loss
```python
# Train policy to predict optimal action at each step
optimal_actions = path_to_actions(optimal_path)
imitation_loss = cross_entropy(policy_logits, optimal_actions)
total_loss = rl_loss + 0.1 * imitation_loss
```

### 3. Counterfactual Features
```python
# Add to observation: "optimal is holding X, you're holding Y"
obs['optimal_portfolio'] = optimal_path.portfolio_trajectory[t]
obs['regret_so_far'] = optimal_path.cumulative_pnl[t] - agent_pnl[t]
```

---

## Research Questions

1. **How close is greedy to optimal?** Benchmark on synthetic data
2. **Does imitation learning help?** A/B test with/without
3. **Is per-episode solving worth the cost?** vs. just using top-k ceiling
4. **Can we learn a path predictor?** NN that outputs approximate optimal path

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Greedy vs ILP gap | < 10% |
| Path computation time | < 10ms for greedy |
| Agent improvement with imitation | +5% alpha capture |
| Replay prioritization benefit | +10% sample efficiency |

---

## References

- Optimal execution: Almgren-Chriss, Bertsimas-Lo
- Hindsight optimization: HIRO, Goal-conditioned RL
- Imitation learning: DAgger, GAIL
- Beam search in RL: MuZero, AlphaFold

---

## Research Prompts

### Prompt 1: Bleeding Edge Hindsight Optimization (2024-2025)
```
Find the most recent papers (2024-2025) on hindsight optimization in RL:

Looking for:
- Hindsight Experience Replay (HER) extensions
- Goal-conditioned RL with hindsight relabeling
- Counterfactual policy optimization
- Return-conditioned sequence models (Decision Transformer variants)

Specifically interested in:
- JAX/Flax implementations
- O(T) or O(T log T) algorithms
- Differentiable hindsight (can backprop through optimal path)
- Applications to trading/portfolio optimization

Need: sub-2-second computation for T=1000, n=64 assets
```

### Prompt 2: Fast Combinatorial Optimization in JAX
```
What are the fastest JAX-compatible algorithms for combinatorial
optimization over sequences?

Setup:
- T=1000 timesteps, n=64 items
- Select k=4 items at each step
- Transition cost between steps
- Known rewards (hindsight)

Looking for:
- Differentiable combinatorial solvers (recent advances)
- Gumbel-softmax for discrete optimization
- Straight-through estimators
- Relaxation + rounding techniques
- JAX implementations of Hungarian algorithm, assignment problems

Target: 1-2 second solve time, must be jit-able
```

### Prompt 3: Hindsight as Auxiliary Loss (Data Leakage Done Right)
```
Research on using hindsight/future information as auxiliary training signal:

The "data leakage" paradox:
- Future info at train time = cheating
- BUT: if model learns to PREDICT what hindsight would say...
- ...it's learning market structure, not memorizing

Papers on:
- Auxiliary future prediction losses
- Hindsight credit assignment
- Temporal difference with privileged information
- Teacher-student with oracle teacher
- Information asymmetric training

Key question: How to use hindsight signal without model
memorizing specific sequences?
```

### Prompt 4: Tree Search Pruning (MuZero/AlphaZero Advances)
```
Latest advances in tree search pruning for planning (2024-2025):

Interested in:
- Value-based pruning (cut branches below threshold)
- Policy-based pruning (ignore low-probability actions)
- Learned pruning functions
- Progressive widening improvements
- Proof-number search adaptations

For context: MuZero-style planning where:
- State space is continuous (portfolio state)
- Action space is large (320 actions)
- Known dynamics (hindsight returns)
- Need to search efficiently

JAX implementations preferred. Target: 1000 simulations in <1s
```

### Prompt 5: Differentiable Dynamic Programming
```
Recent work on differentiable dynamic programming:

Looking for:
- Soft/differentiable DP (entropy-regularized)
- Implicit differentiation through argmax
- Structured prediction with DP
- Differentiable shortest path algorithms
- JAX implementations (e.g., jax.scipy optimizers)

Application: want to backprop through optimal path computation
so model learns to predict paths that would be optimal.

Key papers:
- Differentiable MPC
- OptNet / differentiable QP layers
- Soft attention as differentiable selection
```

### Prompt 6: Sequence-to-Sequence for Optimal Paths
```
Can we frame optimal path finding as seq2seq?

Input: returns sequence (T, n_assets)
Output: trade sequence (variable length)

Relevant architectures:
- Pointer networks for combinatorial optimization
- Set transformer for portfolio selection
- Decision Transformer for trajectory optimization
- Trajectory Transformer
- Autoregressive path generation

Question: Can a transformer learn to output near-optimal
trade sequences in O(T) forward pass? Would be much faster
than explicit search.
```

### Prompt 7: Optimal Transport for Portfolio Transitions
```
Can optimal transport theory help with portfolio transitions?

Framing:
- Current portfolio = source distribution over assets
- Target portfolio = destination distribution
- Transition cost = trading cost
- Find minimum cost transport plan

Recent advances:
- Sinkhorn algorithm (differentiable OT)
- Neural OT
- Unbalanced optimal transport
- JAX implementations (ott-jax library)

Could OT give us optimal rebalancing paths?
```

### Prompt 8: Self-Play with Hindsight Opponents
```
Can we use hindsight oracle as opponent in self-play?

Setup:
- Agent plays against "hindsight oracle" that knows future
- Oracle provides upper bound on performance
- Agent learns to minimize regret vs oracle
- Similar to: expert iteration, AlphaGo self-play

Papers on:
- Regret minimization in games
- Oracle-guided policy improvement
- Adversarial training with privileged information
- Minimax regret objectives

This frames the problem as a game, not just optimization.
```

---

## Notes

The key insight: even a "lucky run" path is valuable signal because:
1. It shows what WAS possible (proof of feasibility)
2. It reveals market structure (when to trade)
3. It provides contrastive learning signal (your path vs optimal)

Not perfect tactics, but strong supervision signal.

---

## Potential Impact (Why This Matters)

**If we get 1-2s solve time:**

| Benefit | Mechanism |
|---------|-----------|
| **Better than random seed** | Guided exploration vs blind sampling |
| **Loss stabilization** | Consistent target signal across regimes |
| **Smarter tree cutting** | Prune branches far from optimal |
| **Sample efficiency** | Learn from near-optimal demos, not random walks |

**The "data leakage" opportunity:**
- Yes, hindsight is "cheating" in a sense
- But if model learns to PREDICT optimal paths...
- ...it's extracting market microstructure
- Key: model shouldn't memorize paths, should learn patterns

**Risk mitigation:**
- Train on path FEATURES, not exact sequences
- Use as soft target (KL divergence), not hard labels
- Validate on held-out time periods
- Monitor for overfitting to specific regimes

**Astronomical upside:**
- Current: agent explores randomly, hopes to find good paths
- With oracle: agent sees what good paths LOOK LIKE
- Difference between "find needle in haystack" vs "here's what needles look like"

---

## Research Findings (2026-01-26)

### Key Result: Target 1-2s IS ACHIEVABLE

**OTT-JAX + jax.lax.scan = 0.1-0.5s for T=1000, n=64, k=4**

### Recommended Implementation Stack

| Component | Library | Complexity | Time |
|-----------|---------|------------|------|
| **Top-k Selection** | OTT-JAX soft sort | O(n²) per step | 1-5ms/step |
| **Sequential DP** | jax.lax.scan | O(T) | Native JAX |
| **Transition penalty** | Quadratic/L1 | O(1) | Trivial |
| **Hard rounding** | jax.lax.top_k | O(n log k) | <1ms |

### Best Algorithm: Gumbel Top-k + OTT-JAX

```python
@jax.jit
def solve_sequential_selection(rewards, transition_weight=0.1, k=4, epsilon=0.05):
    """
    rewards: [T, n] known reward matrix (hindsight optimization)
    Returns: [T, n] soft selection probabilities
    """
    from ott.tools.soft_sort import sort as ott_sort

    def step(prev_selection, reward_t):
        # Soft top-k via OTT-JAX
        base_selection = ott_sort(reward_t, topk=k, epsilon=epsilon)

        # Penalize switching from previous selection
        transition_penalty = transition_weight * jnp.abs(base_selection - prev_selection)
        adjusted_scores = reward_t - jnp.sum(transition_penalty)

        selection = ott_sort(adjusted_scores, topk=k, epsilon=epsilon)
        return selection, selection

    init_selection = jnp.zeros(n)
    _, all_selections = jax.lax.scan(step, init_selection, rewards)
    return all_selections
```

### Tree Search: Gumbel MuZero (mctx library)

**Breakthrough finding**: Gumbel MuZero needs only **2-16 simulations** for 320 actions!

```python
import mctx

policy_fn = jax.jit(lambda p, k, r: mctx.gumbel_muzero_policy(
    params=p,
    rng_key=k,
    root=r,
    recurrent_fn=recurrent_fn,
    num_simulations=100,  # May not need 1000!
    max_num_considered_actions=32,  # Sample 32 of 320
    gumbel_scale=1.0
))
```

**Performance**: `gumbel_muzero_policy` is **3x faster** than `muzero_policy`.

### Hindsight Learning: JaxGCRL

**22x faster** than PyTorch implementations!

```bash
pip install jaxgcrl
python run.py sac --use_her  # Native HER support
```

- 10M environment steps in 10 minutes on single GPU
- Implements CRL (Contrastive RL), SAC+HER, TD3+HER

### Differentiable Hindsight: IDOC (O(T) complexity!)

**IDOC** uses Implicit Function Theorem - no backprop through solver iterations:

| Method | Time | Memory |
|--------|------|--------|
| IDOC | **O(T)** | O(T) |
| Naive autodiff | O(T²) | O(T²) |

Achieves **50-1000x improvement** over REINFORCE in sample efficiency.

### Decision Transformers for Trading

Return-conditioned portfolio generation:

- **TACR** (AAMAS 2023): Outperforms SOTA on Sharpe ratio
- **FTRL** (Neurocomputing 2025): Beats 14 traditional + 9 RL methods
- **RL-TVDT**: Sharpe 1.48, **2.65x portfolio returns** on CSI-300

### Data Leakage Done Right (LUPI Framework)

**Key insight**: Architectural separation prevents memorization.

1. **Asymmetric Actor-Critic**: Critic sees future, actor sees past only
2. **Generalized Distillation**: Teacher trains on privileged, student on regular
3. **Information Bottleneck**: Compress structure, not raw sequences

Vapnik's LUPI proves: O(1/√n) → O(1/n) convergence acceleration!

### Optimal Transport: Limited for Execution

OTT-JAX Sinkhorn is great for **differentiable sorting**, but OT has limits:
- Fixed costs (real market impact is size-dependent)
- Continuous mass (real shares are discrete)
- Static (real markets evolve during execution)

**Verdict**: Use for soft top-k selection, NOT for actual execution paths.

---

## Recommended Implementation Plan

### Phase 1: Fast Soft Solver (0.1-0.5s target)
```python
# 1. Install OTT-JAX
pip install ott-jax

# 2. Implement soft top-k with transition penalty
# 3. Use jax.lax.scan for temporal unrolling
# 4. Hard round for final discrete solution
```

### Phase 2: Gumbel MuZero Integration
```python
# 1. Use mctx.gumbel_muzero_policy
# 2. Sample 32 of 320 actions (max_num_considered_actions)
# 3. Start with 100 simulations, tune down
```

### Phase 3: Hindsight as Training Signal
```python
# 1. Asymmetric critic (sees optimal path)
# 2. Actor learns from critic's value estimates
# 3. Soft KL divergence loss, not hard imitation
```

---

## Key Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| **OTT-JAX** | Differentiable sorting/selection | `pip install ott-jax` |
| **mctx** | Gumbel MuZero tree search | `pip install mctx` |
| **JaxGCRL** | Goal-conditioned RL + HER | `pip install jaxgcrl` |
| **flashbax** | JIT-compatible replay buffers | `pip install flashbax` |
| **SynJAX** | Differentiable DP (CRF, DTW) | DeepMind EMNLP 2023 |

---

## Papers to Read

1. **Gumbel MuZero** (ICLR 2022) - 2-16 sims sufficient
2. **IDOC** (NeurIPS 2023) - O(T) differentiable hindsight
3. **JaxGCRL** (ICLR 2025) - 22x speedup HER
4. **Learning by Cheating** (CoRL 2020) - Asymmetric training
5. **Decision Transformer** (NeurIPS 2021) - Return-conditioned

---

## Benchmark Targets (Updated)

| Metric | Original Target | Research Finding |
|--------|-----------------|------------------|
| Solve time | 1-2s | **0.1-0.5s achievable** |
| MCTS simulations | 1000 | **100 sufficient** (Gumbel) |
| Greedy vs optimal gap | <10% | **0.004-5%** with neural |
| Training speedup | - | **22x** with JAX |
