# Combining imitation learning with MuZero for hindsight-optimal trading

**Your proposed approach is theoretically sound and does NOT violate causality.** Training on hindsight-optimal demonstrations computed after episode completion uses future information only to construct better *training targets*—the learned policy π(a|s) conditions solely on current state at inference. This mirrors how HER (Andrychowicz et al., 2017) proved that goal relabeling is valid because environment dynamics P(s'|s,a) are goal-independent. MuZero Unplugged's Reanalyse mechanism provides the direct framework: MCTS generates improved targets from stored trajectories, while the deployed policy uses only current observations. The literature strongly supports combining BC loss with RL loss using adaptive Q-filtering and scheduled decay, with extensive empirical validation from DQfD, DDPGfD, and BC-SAC.

## Mixing BC and RL losses requires adaptive weighting with Q-filtering

The core insight from **DQfD** (Hester et al., 2018) and **Overcoming Exploration with Demonstrations** (Nair et al., 2018) is that fixed ratios work poorly—BC loss should dominate early training but must diminish as the policy improves beyond demonstrator quality. Three mechanisms achieve this:

**Q-Filter** (recommended primary approach): Apply BC loss only when the demonstration action is better than the current policy: `L_BC = Σ||π(s_i) - a_i||² · 𝟙[Q(s_i,a_i) > Q(s_i,π(s_i))]`. This automatically prevents BC from limiting improvement beyond demonstrator performance while maintaining stability during early training. Nair et al. showed this enables policies to significantly exceed expert performance on robotic manipulation.

**Scheduled decay** provides a simpler alternative: exponentially decay BC weight λ_BC from **1.0 to 0.01** over the first 50% of training (DAgger uses β_i ≤ (1-α)^(i-1)). The rationale is straightforward—imitation bootstraps exploration early, then RL dominates for improvement.

**Adaptive weighting** (most sophisticated) adjusts BC weight based on the difference between current return and moving average return (arXiv:2210.13846). Increase BC weight when performance drops; decrease when exceeding baseline. RPI (ICLR 2024) formalizes this by interleaving IL and RL based on online performance estimates.

| Approach | λ_BC Range | When to Use |
|----------|------------|-------------|
| Q-Filter | Dynamic (0-1) | Policy should exceed demos |
| Exponential decay | 1.0→0.01 | Known good demonstrations |
| Adaptive | 0.1-1.0 | Non-stationary environments |
| Fixed | 0.5-1.0 | Stable baseline |

**Combined loss function** (DQfD-style): `L = L_TD + λ₁L_n-step + λ₂L_BC + λ₃L_L2` with λ₁=λ₂=1.0, λ₃=10⁻⁵. Pre-train for ~75k gradient steps on demonstrations before mixing RL updates.

## Demonstration selection benefits from diversity over volume

The literature contradicts the intuition to store only top-performing demonstrations. **RDES** (Wang et al., 2024) shows that diversity in state-action coverage matters more than raw performance—a diverse buffer of moderately successful trades outperforms a homogeneous buffer of best trades.

**Recommended buffer architecture**: Maintain dual buffers following DDPGfD. A permanent demonstration buffer R_D (10⁴-10⁵ transitions) stores hindsight-optimal experiences with boosted sampling priority. The agent buffer R (10⁵-10⁶ transitions) uses FIFO eviction. Sample **128 demo transitions + 896 agent transitions = 1024 minibatch** per update.

**Selection criteria for trading demonstrations**:
- Store top **60-70% quantile** by realized return (not top 10%)—this preserves diversity while filtering disasters
- Apply k-means clustering on state representations; ensure each cluster contributes proportionally
- Use Prioritized Experience Replay with demo priority boost ε_D=0.001: `p_i = δ²_i + λ|∇_aQ|² + ε + ε_D`

**Dynamic Experience Replay** (Luo & Li, 2020) periodically refreshes the demonstration zone with successful agent-generated trajectories, allowing the "expert" quality to evolve as the agent improves. This is particularly valuable for trading where market regimes shift.

## Adapting HER for profit-target relabeling is highly effective

**HER's future strategy with k=4-8 relabeled goals** consistently outperforms all alternatives (original paper tested k=4 and k=8 extensively). For trading, treat return targets as goals: if a trade aimed for +5% but achieved +2%, relabel with goal=+2% for the hindsight experience.

**Trading-specific HER implementation**:
```python
# Goal space: continuous return targets [-0.10, +0.15]
# Sparse binary reward: r=0 if achieved_return ≥ goal, else r=-1

def relabel_trading_trajectory(trajectory, k=4):
    relabeled = []
    for i, (s, a, r, s_next, goal) in enumerate(trajectory):
        relabeled.append((s, a, r, s_next, goal))  # Original
        
        # Future strategy: sample k states visited AFTER current timestep
        future_returns = [traj[j].achieved_return for j in range(i+1, len(trajectory))]
        for new_goal in random.sample(future_returns, min(k, len(future_returns))):
            new_r = 0 if current_return >= new_goal else -1
            relabeled.append((s, a, new_r, s_next, new_goal))
    return relabeled
```

**Multi-dimensional goals** extend this naturally: goal = {return_target: 0.05, max_drawdown: 0.02, holding_period: 5}. Each dimension can be independently relabeled based on achieved values.

**PPO-HER for Portfolio Optimization** (Zhang & Xie, 2025) directly validates this approach, combining TimeSformer for cross-asset attention with HER relabeling for sample efficiency. Their framework achieved significant improvements over non-HER baselines in portfolio allocation.

## Distribution shift requires regime-aware buffer management

Financial markets exhibit concept drift (P(Y|X) changes) more than pure covariate shift—the same technical setup may warrant opposite actions in different regimes. Three complementary strategies address this:

**Regime-tagged demonstrations** label each stored experience with detected regime (bull/bear/sideways/high-vol). Use HMM or TimesNet for regime detection. During training, sample demonstrations matching current detected regime with **3x probability boost**, while maintaining **20% cross-regime sampling** for robustness.

**Exponential recency weighting** downweights older demonstrations: `w(t) = exp(-λ · age_days)` with λ≈0.01 (half-life ~70 trading days). Combine with regime similarity: `final_weight = recency_weight × regime_match_score × performance_weight`.

**Elastic Weight Consolidation** (Kirkpatrick et al., 2017) protects learned trading patterns when adapting to new regimes: `L = L_new + λ Σ_i F_i(θ_i - θ*_i)²` where F_i is the Fisher Information for weight importance. This prevents catastrophic forgetting of bull market patterns while learning bear market behavior.

**Buffer refresh protocol**: Maintain a "core buffer" (regime-invariant patterns like stop-loss execution) that never expires. Refresh regime-specific demonstrations every **20-50 trading days** or upon detected regime change. Use CLEAR (Rolnick et al., 2019) mixing ratio of 70% recent / 30% historical for replay.

## Pattern extraction through causal temporal convolutions

**Causal TCN encoders** (Temporal Convolutional Networks with dilated causal convolutions) are the recommended architecture for pattern embeddings. Critical implementation detail: use **left-padding only** to ensure output at time t depends exclusively on inputs ≤t.

**Recommended encoder architecture for JAX/Flax**:
```python
class CausalTCNEncoder(nn.Module):
    hidden_dim: int = 128
    num_layers: int = 4
    kernel_size: int = 3
    
    @nn.compact
    def __call__(self, x):  # x: [batch, time, features]
        for i in range(self.num_layers):
            dilation = 2 ** i
            pad = (self.kernel_size - 1) * dilation
            x_padded = jnp.pad(x, ((0,0), (pad,0), (0,0)))  # Causal: left-pad only
            x = nn.Conv(self.hidden_dim, (self.kernel_size,), 
                       kernel_dilation=(dilation,))(x_padded)
            x = nn.gelu(x)
        return x[:, -1, :]  # Final embedding
```

**Contrastive pre-training** using TF-C (Time-Frequency Consistency, NeurIPS 2022) learns robust representations without labels. Create positive pairs via jittering and scaling augmentations; use InfoNCE loss. This pre-training enables downstream pattern clustering.

**Prototype-based pattern recognition** (Snell et al., 2017): After contrastive pre-training, compute prototype embeddings as the mean of each profitable pattern cluster. New patterns classify by Euclidean distance to prototypes—this naturally handles few-shot learning of rare but valuable setups.

**Proto-RL** (Yarats et al., 2021) unifies representation learning with exploration: prototypes summarize exploratory experience while serving as basis for state encoding. For trading, prototypes could represent canonical patterns (breakout, mean-reversion, momentum) with the agent learning goal-conditioned policies for each.

## Theoretical guarantees confirm causal validity

**The proposed approach satisfies the no-lookahead constraint formally.** The mathematical justification has three components:

**Information flow separation**: During training, the model learns θ from {(s_t, a_t, target_t)} where targets are computed with hindsight. At inference, π_θ(a|s) conditions only on current state and learned parameters—no future information flows to decisions.

**HER's causal proof** (Andrychowicz et al., 2017): "The goal being pursued influences the agent's actions but NOT the environment dynamics." Therefore trajectories can be replayed with arbitrary goals using off-policy learning. HIPI (NeurIPS 2020) formalizes this as inverse RL: hindsight relabeling asks "for what task was this trajectory optimal?"

**MuZero Unplugged validation**: The Reanalyse mechanism computes MCTS-improved targets on stored trajectories. The neural network learns to predict these improved targets. At deployment, MCTS uses only current observation and learned model—achieving state-of-the-art on the RL Unplugged benchmark without any special offline RL modifications.

**Convergence guarantees** for IL+RL: CSIL (NeurIPS 2023) proves KL-regularized BC equals game-theoretic IRL objective plus constant under appropriate conditions. BC-SAC demonstrates convergence by using BC for in-distribution states and RL signals for out-of-distribution states. Sample complexity with demonstrations is O(demos + RL_samples), substantially better than pure RL.

## Failure modes and critical mitigations

| Failure Mode | Mechanism | Mitigation |
|--------------|-----------|------------|
| BC limiting improvement | High BC weight prevents exceeding demos | Q-Filter: BC only when Q_demo > Q_policy |
| Covariate shift | Policy visits states outside demo distribution | DAgger-style online collection; GAIL for adversarial matching |
| Demonstration overfitting | Network memorizes limited demo data | L2 regularization (λ=10⁻⁵), dropout, early stopping |
| Regime mismatch | Demos from past regime mislead current trading | Regime tagging + recency weighting + EWC |
| Hindsight goal distribution gap | Training goals differ from deployment goals | Ensure goal coverage; use curriculum from easy to hard targets |
| Model error (MuZero) | Learned dynamics diverge from true dynamics | EfficientZero's self-supervised consistency loss |

**Critical implementation check**: Verify no future leakage by ensuring all features computed at time t use only data from times ≤t-1. Common bugs include using closing prices when only open/high/low are available, or technical indicators computed with centered (non-causal) windows.

## Hyperparameter guidance and implementation checklist

**Core hyperparameters**:
- BC weight λ_BC: Start 1.0, decay to 0.01 over 50% of training (or use Q-filter)
- Demo buffer size: 10⁴-10⁵ transitions
- Agent buffer size: 10⁵-10⁶ transitions
- Demo samples per minibatch: 128/1024 (~12.5%)
- HER relabeling k: 4-8 (future strategy)
- PER α (priority exponent): 0.6
- PER β (importance sampling): 0.4→1.0 annealed
- Demo priority boost ε_D: 0.001
- Recency decay λ: 0.01 (half-life ~70 days)
- EWC penalty λ: 400-1000 (tune on validation)

**JAX/Flax implementation notes**:
- Use **Flax** for neural networks (Haiku deprecated)
- **Optax** for gradient processing: `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))`
- **RLax** from DeepMind for TD-learning primitives
- Implement causal convolutions with explicit left-padding (jnp.pad with ((0,0), (pad,0), (0,0)))
- Use `jax.jit` for training steps; `jax.vmap` for batched MCTS rollouts
- PureJaxRL patterns achieve 4000x speedups through full vectorization

**MuZero-specific considerations**:
- Store demonstrations in replay buffer alongside self-play
- Reanalyse computes improved MCTS targets from any stored trajectory
- Use EfficientZero's self-supervised consistency loss for sample efficiency
- For offline-heavy training, consider ROSMO's one-step lookahead (5.6% compute of full MCTS, competitive performance)

## Conclusion

The approach of computing hindsight-optimal paths after episode completion and training via behavioral cloning toward optimal actions is well-supported by theory and practice. The key insight—that hindsight information improves *training signal quality* without affecting *inference causality*—enables learning from every trade as if it were optimal for some target. Combine this with Q-filtered BC loss decay, diverse demonstration selection via clustering, regime-aware buffer management, and causal TCN pattern encoders implemented in JAX/Flax. The MuZero Unplugged framework provides the direct integration path: demonstrations enter the replay buffer, Reanalyse generates improved targets, and the policy learns to exceed demonstrator performance while maintaining strict causal validity at deployment.