# Breaking the dual collapse trap in MuZero-style trading agents

Your "dual collapse" phenomenon—where policy entropy collapses to HOLD while the auxiliary predictor learns trivial "no change" outputs—is a **well-documented degenerate equilibrium** in model-based RL. The critical insight: these failures are not independent but form a self-reinforcing feedback loop that standard entropy regularization alone cannot break.

## The feedback loop creates a stable but useless equilibrium

When policy collapses to HOLD, the replay buffer fills exclusively with "no action → no change" transitions. The world model then optimizes perfectly for this data by predicting "no change" (explaining your MSE drop from 0.4 to 0.01). MCTS planning with this trivial dynamics model sees no benefit from non-HOLD actions, reinforcing the collapsed policy. This is a **stable equilibrium**—both components are locally optimal given the other's behavior.

Research from Nathan Lambert's MBRL debugging guide identifies this as "objective mismatch": the world model loss decreases while becoming useless for planning. The 2024 paper "What model does MuZero learn?" confirms that MuZero's learned model "cannot generally evaluate policies accurately, especially those that deviate from the data collection policy"—your collapsed model literally cannot imagine what would happen with BUY/SELL actions.

## Original MuZero lacks entropy regularization entirely

A critical discovery: **the original MuZero and AlphaZero papers do NOT include entropy regularization in their loss function**. They rely instead on Dirichlet noise at the MCTS root and temperature-based action selection. This works for board games where exploration emerges naturally from diverse game states, but fails catastrophically for trading where HOLD creates reward stability that resembles a "safe" option.

MuZero-General GitHub issue #175 explicitly addresses this gap, proposing:
```python
loss = value_loss * weight + reward_loss + policy_loss - (beta * entropy_loss)
```
With **β = 0.01–0.1** as starting range. However, fixed entropy coefficients often fail because the optimal coefficient changes during training—SAC-style automatic entropy tuning with target entropy **H_target = 0.5 × log(num_actions)** provides adaptive control.

## MCTS parameters amplify collapse when misconfigured

Your 320-action space requires careful MCTS tuning. The standard parameters (c₁=1.25, c₂=19652, Dirichlet α=0.03) were designed for Go's ~361 actions—using α=0.03 on 320 actions produces sparse Dirichlet samples that concentrate on few actions, exacerbating collapse.

**Recommended adjustments for large discrete action spaces:**
- **Dirichlet α = 0.1–0.3** (heuristic: α ≈ 10/num_actions produces useful diversity)
- **c_puct = 2.0–4.0** (increased from 1.25 to encourage exploration)
- **Simulations = 25–50** initially (more simulations → sharper policies)
- **Temperature τ = 1.0–1.5** for first 50+ epochs (standard AlphaZero drops to τ→0 too aggressively)

Gumbel MuZero offers an alternative with **guaranteed policy improvement even with 2 simulations**, using Gumbel noise for action selection rather than visit count ratios. This is more robust for limited simulation budgets and large action spaces.

## World model collapse mirrors self-supervised learning failures

Your neural ODE predictor's collapse to "no change" outputs parallels **representation collapse** in self-supervised learning (SimSiam, BYOL). The trivial solution where all predictions equal the mean is a stable attractor without explicit prevention mechanisms.

MuDreamer (2024) found that **batch normalization is crucial** when removing reconstruction loss: "The use of batch normalization is crucial to prevent learning collapse in which the model produces constant or non-informatice hidden states." EfficientZero addresses this with **temporal consistency loss**—forcing predicted next states to match encoded actual observations:

```python
# Predicted state from dynamics model
z_pred = dynamics(z_t, action)
# Encoded actual next observation (with stop-gradient)
z_target = representation(obs_next).detach()
# Cosine similarity loss
ssl_loss = -cosine_similarity(z_pred, z_target).mean()
```

This loss provides gradient signal independent of reward, preventing the "no change" trivial solution by requiring predictions to actually match encoded observations.

## Dormant neurons may compound the prediction collapse

Research from Dohare et al. (Nature 2024) demonstrates that deep networks **lose plasticity** in continual learning settings—neurons become dormant (near-zero output), overcommitted (saturated), or redundant. Your neural ODE head may have experienced plasticity loss where few active neurons all output similar values.

**Diagnostic**: Check the effective rank of your prediction head's output—if predictions cluster tightly regardless of input, plasticity loss is likely. **Continual Backpropagation** (selectively reinitializing least-used units) or **Shrink and Perturb** provides intervention:

```python
def shrink_and_perturb(model, shrink=0.9, perturb=0.01):
    for param in model.parameters():
        param.data *= shrink  # Shrink toward zero
        param.data += perturb * torch.randn_like(param)  # Add noise
```

Apply every 10K–50K steps with shrink=0.9–0.99, perturb=0.001–0.01.

## Breaking the loop requires multi-point intervention

Since the collapse is a stable equilibrium, breaking it at only one point allows the other component to pull it back. Effective solutions target multiple points simultaneously:

**Point 1 — Policy entropy (prevent action collapse)**
- SAC-style automatic entropy tuning with adaptive α
- Random Network Distillation (RND) provides exploration signal independent of world model quality—novel states receive intrinsic bonus regardless of whether the model correctly predicts them
- RND coefficient **0.1–0.5× extrinsic reward** with separate value heads for intrinsic/extrinsic returns

**Point 2 — World model training (prevent trivial predictions)**
- Add temporal consistency loss (SSL) with weight 0.5–1.0
- Use separate replay buffers: world model buffer prioritizes transition diversity, policy buffer prioritizes TD error
- Update world model **10× faster** than policy to ensure dynamics learning leads policy learning

**Point 3 — Coupling between components**
- Freeze world model during policy training phases (alternating optimization)
- Apply stop-gradient to prevent auxiliary head gradients from corrupting shared representations
- Periodic Shrink+Perturb specifically when detecting prediction variance collapse

**Detection triggers for intervention:**
```python
if policy_entropy < 0.5 * target_entropy:
    entropy_coef *= 2.0  # Emergency boost
    
if prediction_variance < threshold:
    shrink_and_perturb(world_model)
    reinitialize(prediction_head)  # Keep representation
```

## Trading-specific considerations exacerbate HOLD bias

Your domain presents unique challenges. Research from transaction-aware RL confirms that "the hold action deals with two opposite states, empty or nonempty position"—HOLD's reward distribution has lower variance than BUY/SELL because it avoids transaction costs and position risk. During early training with noisy value estimates, this makes HOLD appear "safe."

Additionally, trading rewards are sparse and delayed—meaningful signal may only arrive after multi-step position building. Standard MuZero with short planning horizons cannot evaluate such strategies.

**Trading-specific recommendations:**
- Use **asymmetric reward shaping** that penalizes excessive HOLD sequences
- Implement **position diversity bonus** as auxiliary reward
- Consider **Ornstein-Uhlenbeck noise** in action selection (used in PortfolioZero for finance)
- Set **td_steps > max episode length** for sparse reward propagation

## Production systems use population diversity and league training

AlphaStar's approach to preventing collapse relies on **population-based training (PBT)** with explicit diversity mechanisms:
- Main agents train against full league with prioritized matchmaking
- **Exploiter agents** specifically train to find weaknesses in main agents
- Historical checkpoints maintain strategy diversity
- PBT optimizes entropy_coef, learning_rate, and auxiliary_loss_weight jointly

For your trading agent, PBT with **8–16 agents** optimizing entropy coefficient (0.001–1.0), world model learning rate, and intrinsic reward coefficient provides automatic hyperparameter scheduling that responds to collapse signals. Fitness metric should be a "true objective" like Sharpe ratio on held-out periods, not shaped training rewards.

## Recommended implementation priority

**Immediate (break current collapse):**
1. Add RND intrinsic reward (coefficient 0.1–0.5)
2. Implement SAC-style automatic entropy tuning targeting 0.5 × log(320) ≈ 2.9 nats
3. Add batch normalization + LayerNorm to representation network

**Short-term (prevent recurrence):**
1. Implement temporal consistency SSL loss for world model
2. Separate update rates: world model 10× policy updates
3. Apply Shrink+Perturb every 10K steps with shrink=0.95
4. Increase Dirichlet α to 0.15 and c_puct to 2.5

**Long-term (robust training):**
1. Deploy PBT with 8–12 agents
2. Consider Gumbel MuZero for more stable policy improvement
3. Implement action-conditional prediction diversity loss: different actions must produce different predicted futures

## Conclusion

Your coupled collapse represents a well-characterized failure mode where policy and world model converge to a mutually reinforcing but useless equilibrium. The key insight is that **original MuZero lacks the mechanisms needed to prevent this in domains with "safe" default actions and sparse rewards**—entropy regularization, auxiliary SSL losses, and exploration bonuses must be added explicitly. Breaking the loop requires simultaneous intervention at multiple points: maintaining policy entropy through adaptive regularization and intrinsic motivation, preventing world model collapse through consistency losses and plasticity maintenance, and decoupling their training dynamics to prevent mutual reinforcement of trivial solutions.

The trading domain's inherent HOLD bias and sparse rewards make it particularly susceptible. Solutions that work for board games (Dirichlet noise, temperature annealing) are insufficient here—explicit architectural and training modifications from recent research (EfficientZero's SSL, Gumbel MuZero's exploration guarantees, RND's model-independent curiosity) provide the necessary robustness.