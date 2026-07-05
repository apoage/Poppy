# TASK_139: GRPO Integration from DeepSeek-R1

**Status**: Research Complete / Implementation Pending
**Priority**: High
**Created**: 2026-02-04

## Source Paper

**Location**: `/home/lukas/Trader/docs/research/arXiv-2501.12948v2/`
**Title**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**Main file**: `sn-article.tex`

---

## Executive Summary

DeepSeek-R1 demonstrates that **pure RL without supervised fine-tuning** can develop sophisticated reasoning through GRPO (Group Relative Policy Optimization). Key finding: rule-based rewards + group-relative advantages = emergent complex behavior without reward hacking.

**Relevance to Poppy**: GRPO is simpler than PPO, requires no critic network, and aligns with our oracle-based training philosophy.

---

## Key Techniques to Adopt

### 1. GRPO Algorithm (HIGH PRIORITY)

**Paper Reference**: Section 2.1, Equations 1-3 (`sn-article.tex:217-232`)

**Core Equations**:

```
Objective (Eq. 1):
J_GRPO(θ) = E[1/G * Σ(min(ratio * A_i, clip(ratio, 1-ε, 1+ε) * A_i) - β * D_KL)]

where ratio = π_θ(o_i|q) / π_θ_old(o_i|q)

KL Divergence (Eq. 2):
D_KL = π_ref(o|q)/π_θ(o|q) - log(π_ref(o|q)/π_θ(o|q)) - 1

Advantage (Eq. 3):
A_i = (r_i - mean(r_1...r_G)) / std(r_1...r_G)
```

**Why This Matters**:
- No critic/value network needed (unlike PPO)
- Advantages computed from group statistics (self-normalizing)
- Simpler training loop, fewer hyperparameters
- Proven at scale (671B params)

**Application to Trading**:
```python
# Pseudocode for trading GRPO
def grpo_step(policy, market_state, G=16):
    # Sample G action trajectories from current policy
    trajectories = [policy.rollout(market_state) for _ in range(G)]

    # Compute rule-based rewards (PnL, Sharpe, etc.)
    rewards = [compute_trading_reward(t) for t in trajectories]

    # Group-relative advantage (no learned value function!)
    mean_r, std_r = np.mean(rewards), np.std(rewards)
    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

    # PPO-style clipped objective with group advantages
    loss = compute_clipped_loss(policy, trajectories, advantages)
    return loss
```

### 2. Rule-Based Rewards (VALIDATES CURRENT APPROACH)

**Paper Reference**: Section 2.2 (`sn-article.tex:252-267`)

DeepSeek explicitly **avoids neural reward models** due to reward hacking at scale:

> "We abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks. This decision is predicated on our observation that neural reward models are susceptible to reward hacking during large-scale reinforcement learning."

**Our Equivalent**:
| DeepSeek | Poppy Trading |
|----------|---------------|
| Math correctness | Realized PnL |
| Code test pass | Sharpe ratio |
| Format compliance | Position limits |

**Reward Formula** (`sn-article.tex:261-263`):
```
Reward_rule = Reward_acc + Reward_format
```

**Trading Translation**:
```python
Reward_trading = Reward_pnl + Reward_risk + Reward_constraints
# Where:
#   Reward_pnl = log_returns or cumulative PnL
#   Reward_risk = -max_drawdown or Sharpe component
#   Reward_constraints = penalty for position violations
```

### 3. Training Hyperparameters

**Paper Reference**: Section 2.1 (`sn-article.tex:235`)

```
learning_rate = 3e-6
kl_coefficient = 0.001
sampling_temperature = 1.0
group_size = 16 outputs per question
batch_size = 512 (32 questions × 16 outputs)
max_length = 32768 tokens (early), 65536 (later)
reference_model_update = every 400 steps
```

**Adaptation for Trading**:
```python
# Suggested starting point
GRPO_CONFIG = {
    'lr': 3e-6,
    'kl_coef': 0.001,
    'temperature': 1.0,
    'group_size': 16,        # trajectories per market state
    'batch_questions': 32,   # market states per batch
    'clip_ratio': 10.0,      # They found larger clip helps (Section 3.2.1)
    'ref_update_freq': 400,
}
```

### 4. Large Clip Ratio Discovery

**Paper Reference**: Section 3.2.1 (`sn-article.tex:393-394`)

> "Note that the clip ratio plays a crucial role in training. A lower value can lead to the truncation of gradients for a significant number of tokens, thereby degrading the model's performance"

They use **ε = 10** (vs standard PPO's 0.2). This is a significant finding for stable large-scale RL.

### 5. Multi-Stage Training Pipeline

**Paper Reference**: Figure 2, Section 3 (`sn-article.tex:337-348`)

```
Stage 1: Cold-start SFT (small human-aligned data)
    ↓
Stage 2: RL with rule-based rewards (reasoning focus)
    ↓
Stage 3: Rejection sampling + SFT (quality filter)
    ↓
Stage 4: RL with mixed rewards (helpfulness + safety)
```

**Trading Adaptation**:
```
Phase 1: Oracle imprinting (train_easy.py) - current
    ↓
Phase 2: GRPO with PnL rewards (new)
    ↓
Phase 3: Rejection sampling on best trajectories
    ↓
Phase 4: GRPO with full reward (Sharpe, drawdown, etc.)
```

### 6. Emergent Behaviors

**Paper Reference**: Table 2, Section 2.3 (`sn-article.tex:293-331`)

Key observation: Model naturally learned to:
- Self-reflect ("Wait, let me reconsider...")
- Verify intermediate steps
- Explore alternative approaches

**"Aha Moment"** (`sn-article.tex:306`):
> "Wait, wait. Wait. That's an aha moment I can flag here."

This emerged purely from RL without explicit teaching. Implication: if we give Poppy the right reward signal, it may discover trading strategies we haven't considered.

---

## Implementation Plan

### Phase A: GRPO Core (estimate: 1-2 days)

1. Create `mcts_trader/jax_agent/grpo.py`:
   - Group sampling from policy
   - Advantage computation (Eq. 3)
   - Clipped objective (Eq. 1)
   - KL penalty (Eq. 2)

2. No changes needed to model architecture (works with existing TRM)

### Phase B: Trading Reward Function

1. Formalize reward components:
   ```python
   def trading_reward(trajectory):
       pnl = trajectory.cumulative_pnl[-1]
       sharpe = compute_sharpe(trajectory.returns)
       drawdown = max_drawdown(trajectory.equity_curve)
       violations = count_position_violations(trajectory)

       return (
           0.5 * pnl +
           0.3 * sharpe +
           -0.1 * drawdown +
           -0.1 * violations
       )
   ```

### Phase C: Integration with train_easy.py

Option 1: Add GRPO as Phase 2 after oracle imprinting
Option 2: Create separate `train_grpo.py` for RL-only experiments

### Phase D: Ablation Studies

- Compare GRPO vs current MuZero value-based approach
- Test different group sizes (8, 16, 32)
- Vary clip ratio (0.2, 1.0, 10.0)

---

## Key Equations Reference

| Equation | Location | Purpose |
|----------|----------|---------|
| GRPO Objective | `sn-article.tex:218-225` | Main training loss |
| KL Divergence | `sn-article.tex:226-228` | Regularization |
| Group Advantage | `sn-article.tex:230-232` | No critic needed |
| Rule Reward | `sn-article.tex:261-263` | Verifiable signal |
| Language Consistency | `sn-article.tex:388-390` | Format reward |

---

## Questions to Resolve

1. **Group size for trading**: 16 works for text, but trading trajectories are longer. Start with 16, tune if needed.

2. **Reference model update frequency**: 400 steps in paper. May need adjustment for trading dynamics.

3. **Temperature schedule**: They use 1.0 for exploration, 0.7 later for stability. Trading may need different schedule.

4. **Integration with MCTS**: GRPO is for policy learning. Could combine with MCTS at inference (policy provides priors, MCTS refines).

---

## Success Criteria

- [ ] GRPO implementation matches paper equations
- [ ] Training stable for 10k+ steps
- [ ] Performance >= oracle imprinting baseline
- [ ] No reward hacking observed (PnL gaming)
- [ ] Emergent strategies discovered (not in oracle)

---

## Vision Alignment: DeepSeek-R1 ↔ ALPHAZERO_TRADING_VISION.md

**Key Discovery**: The AlphaZero Trading Vision (written December 19, 2025) independently arrived at the same conclusions as DeepSeek-R1 (published January 2025). This validates the architectural direction.

### Perfect Alignments

| Our Vision (Dec 2025) | DeepSeek-R1 (Jan 2025) | Status |
|----------------------|------------------------|--------|
| "Move 37" emergent tactics | "Aha moment" self-reflection | ✅ Validated |
| Rule-based rewards (PnL, Sharpe) | Rule-based rewards (accuracy) | ✅ Validated |
| Curriculum → Self-improvement | Cold-start SFT → Pure RL | ✅ Same philosophy |
| Multi-stage training pipeline | SFT → RL → Rejection → RL | ✅ Same structure |
| Distillation (Winner-Loser in Self-Play Dojo) | Distilled R1 to 7B/14B/32B | ✅ Proven at scale |

### Where Our Vision Goes Further

| Aspect | DeepSeek-R1 | Our Vision |
|--------|-------------|------------|
| **Opponent** | Verifiable problems (static) | Adversarial Market Agent (dynamic) |
| **Environment** | Problem → Solution | Agent affects market (feedback loop) |
| **Competition** | Single agent | Population-based (8-32 agents) |
| **Expected Emergence** | Self-reflection | Front-running, regime switching, liquidity games |
| **Physics** | None | NeuralOGCM market physics for stability |

### Synthesis: Combined Pipeline

```
Phase 1: Oracle Imprinting (train_easy.py)
    └── BC from greedy oracle - gets baseline policy
    └── STATUS: Current implementation

Phase 2: GRPO Refinement (from DeepSeek-R1)
    └── Pure RL with rule-based rewards (PnL)
    └── No value network, group advantages (Eq. 3)
    └── Uses ε=10 clip ratio
    └── Discovers tactics oracle missed
    └── STATUS: This task

Phase 3: Self-Play Dojo (Our vision, GRPO-powered)
    └── Agent population (8-32)
    └── GRPO for each agent (efficient training)
    └── Adversarial market dynamics
    └── "Move 37" emergence
    └── STATUS: Future (lore/SELF_PLAY_DOJO_VISION.md)

Phase 4: Physics + Transfer (Our vision)
    └── NeuralOGCM stability
    └── Multi-market generalization
    └── STATUS: Future
```

### Key Insight

**DeepSeek provides**: Simpler RL algorithm (GRPO), proven hyperparameters (ε=10, G=16), validation that pure RL works at scale.

**Our vision adds**: Adversarial opponent, population dynamics, market physics - things DeepSeek doesn't address.

**Combined**: GRPO efficiency + Adversarial robustness = potentially stronger than either alone.

### Reference Documents

- Our vision: `lore/ALPHAZERO_TRADING_VISION.md` (Dec 19, 2025)
- Self-play design: `lore/SELF_PLAY_DOJO_VISION.md` (Jan 11, 2026)
- DeepSeek paper: `docs/research/arXiv-2501.12948v2/sn-article.tex`

---

## References

- Main paper: `docs/research/arXiv-2501.12948v2/sn-article.tex`
- Appendix (detailed ablations): `docs/research/arXiv-2501.12948v2/appendix.tex`
- GRPO original: DeepSeekMath paper (cited in Section 2.1)
- Our vision: `lore/ALPHAZERO_TRADING_VISION.md`
- Self-play dojo: `lore/SELF_PLAY_DOJO_VISION.md`
