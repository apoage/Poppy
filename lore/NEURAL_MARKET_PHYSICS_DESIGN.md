# Neural Market Physics: Adapting NeuralOGCM for Trading

**Inspired by**: arXiv:2512.11525v1 - "NeuralOGCM: Differentiable Ocean Modeling with Learnable Physics"

**Author**: Claude Code
**Date**: December 19, 2025

---

## Core Concept

Adapt NeuralOGCM's **hybrid physics-AI framework** to market prediction, combining:
1. **Differentiable Market Physics Core** - Known market dynamics with learnable parameters
2. **Neural Market Corrector** - Learns residual dynamics (order flow, sentiment, black swans)
3. **ODE Solver Integration** - Fuses both tendencies for next-step prediction

**Key Innovation**: Instead of hand-tuning momentum/volatility/correlation parameters, let them be **learned end-to-end** from market data alongside the neural network.

---

## Architecture Parallel

### Ocean Modeling (NeuralOGCM)
```
∂y/∂t = G(y, F) = G_phys(y; θ_p) + G_neural(y, F; θ_n)

Components:
- G_phys: Advection, Coriolis force, horizontal diffusion
- θ_p: Learnable diffusion coefficients (ν_momentum, ν_tracer)
- G_neural: Encoder-Decoder + STE (multi-head attention)
- F: External atmospheric forcings

Integration:
y_{t+Δt} = y_t + Δt·G_phys + Δt·G_neural
```

### Market Prediction (Proposed)
```
∂p/∂t = G(p, X) = G_physics(p; θ_physics) + G_neural(p, X; θ_neural)

Components:
- G_physics: Momentum, mean reversion, volatility diffusion, correlation dynamics
- θ_physics: Learnable market parameters (momentum α, reversion β, volatility σ, correlation γ)
- G_neural: Encoder-Decoder + STE (multi-head attention for cross-asset interactions)
- X: External features (volume, order flow, sentiment, macro indicators)

Integration:
price_{t+Δt} = price_t + Δt·G_physics + Δt·G_neural
```

---

## Differentiable Market Physics Core

### 1. Price Momentum
**Physics**: Prices carry momentum (Newton's first law analogy)

```python
def momentum_tendency(prices, velocities, α_momentum):
    """
    dp/dt = α·velocity

    Args:
        prices: [batch, n_assets, H, W] current prices
        velocities: [batch, n_assets, H, W] price velocities (dp/dt from last step)
        α_momentum: learnable parameter (momentum strength)

    Returns:
        tendency: [batch, n_assets, H, W]
    """
    # α_momentum is learned (starts ~0.8, converges to optimal value)
    return softplus(α_momentum) * velocities
```

### 2. Mean Reversion
**Physics**: Prices revert to moving average (economic equilibrium)

```python
def mean_reversion_tendency(prices, moving_avg, β_reversion):
    """
    dp/dt = -β·(p - MA)

    Args:
        prices: [batch, n_assets, H, W]
        moving_avg: [batch, n_assets, H, W] (e.g., 20-period MA)
        β_reversion: learnable parameter (reversion strength)

    Returns:
        tendency: [batch, n_assets, H, W]
    """
    deviation = prices - moving_avg
    return -softplus(β_reversion) * deviation
```

### 3. Volatility Diffusion
**Physics**: Random price fluctuations (Brownian motion analogy)

```python
def volatility_tendency(prices, noise, σ_vol):
    """
    dp/dt = σ·ε,  ε ~ N(0,1)

    Args:
        prices: [batch, n_assets, H, W]
        noise: [batch, n_assets, H, W] sampled from N(0,1)
        σ_vol: [n_assets] learnable volatility per asset

    Returns:
        tendency: [batch, n_assets, H, W]
    """
    # Each asset has its own learned volatility
    return softplus(σ_vol).unsqueeze(-1).unsqueeze(-1) * noise
```

### 4. Cross-Asset Correlation Dynamics
**Physics**: Correlated assets influence each other (coupling)

```python
def correlation_tendency(prices, γ_correlation):
    """
    dp_i/dt = Σ_j γ_ij·(p_j - p_i)

    Assets with positive correlation pull each other.

    Args:
        prices: [batch, n_assets, H, W]
        γ_correlation: [n_assets, n_assets] learnable correlation matrix

    Returns:
        tendency: [batch, n_assets, H, W]
    """
    # Compute pairwise differences
    # This is like diffusion in ocean model (horizontal diffusion)
    # Implementation: conv2d with correlation kernels

    batch, n, h, w = prices.shape
    flat_prices = prices.reshape(batch, n, -1)  # [batch, n_assets, H*W]

    # Correlation coupling (learned adjacency matrix)
    coupling = torch.einsum('ij,bjk->bik', softplus(γ_correlation), flat_prices)
    tendency = (coupling - flat_prices).reshape(batch, n, h, w)

    return tendency
```

### 5. Combined Physics Core

```python
class DifferentiableMarketPhysics(nn.Module):
    """
    Learnable market physics core.

    Learnable parameters (θ_physics):
        - α_momentum: momentum strength [1]
        - β_reversion: mean reversion strength [1]
        - σ_vol: volatility per asset [n_assets]
        - γ_correlation: correlation matrix [n_assets, n_assets]
    """

    def __init__(self, n_assets=64):
        super().__init__()

        # Initialize learnable physics parameters
        self.α_momentum = nn.Parameter(torch.tensor(0.8))  # Start at 0.8
        self.β_reversion = nn.Parameter(torch.tensor(0.2))  # Start at 0.2
        self.σ_vol = nn.Parameter(torch.ones(n_assets) * 0.02)  # 2% volatility
        self.γ_correlation = nn.Parameter(torch.eye(n_assets) * 0.1)  # Weak correlation

    def forward(self, state):
        """
        Args:
            state: dict with keys:
                - 'prices': [batch, n_assets, H, W]
                - 'velocities': [batch, n_assets, H, W]
                - 'moving_avg': [batch, n_assets, H, W]
                - 'noise': [batch, n_assets, H, W] (sampled externally)

        Returns:
            G_physics: [batch, n_assets, H, W] total physics tendency
        """
        prices = state['prices']
        velocities = state['velocities']
        moving_avg = state['moving_avg']
        noise = state['noise']

        # Compute each physics component
        T_momentum = momentum_tendency(prices, velocities, self.α_momentum)
        T_reversion = mean_reversion_tendency(prices, moving_avg, self.β_reversion)
        T_volatility = volatility_tendency(prices, noise, self.σ_vol)
        T_correlation = correlation_tendency(prices, self.γ_correlation)

        # Combine (additive tendencies)
        G_physics = T_momentum + T_reversion + T_volatility + T_correlation

        return G_physics
```

---

## Neural Market Corrector

### Architecture (from NeuralOGCM)

```python
class NeuralMarketCorrector(nn.Module):
    """
    Learns residual market dynamics not captured by physics.

    Architecture:
        Encoder → STE (Spatio-temporal Evolution) → Decoder

    Input:
        - state_t: [batch, n_assets, H, W] (prices, volumes, indicators)
        - features_t: [batch, n_features, H, W] (order flow, sentiment, macro)

    Output:
        - G_neural: [batch, n_assets, H, W] correction tendency
    """

    def __init__(self, n_assets=64, n_features=10, latent_dim=256):
        super().__init__()

        # Encoder (downsample state + features)
        self.encoder = ConvEncoder(
            in_channels=n_assets + n_features,
            out_channels=latent_dim,
            hidden_dims=[64, 128, 256]
        )

        # Spatio-Temporal Evolution (multi-head self-attention)
        self.ste = MultiHeadAttention(
            d_model=latent_dim,
            num_heads=8,
            dropout=0.1
        )

        # Decoder (upsample back to full resolution)
        self.decoder = ConvDecoder(
            in_channels=latent_dim,
            out_channels=n_assets,
            hidden_dims=[256, 128, 64]
        )

    def forward(self, state_t, features_t):
        """
        Args:
            state_t: [batch, n_assets, H, W]
            features_t: [batch, n_features, H, W]

        Returns:
            G_neural: [batch, n_assets, H, W]
        """
        # Concatenate state and features
        x = torch.cat([state_t, features_t], dim=1)  # [batch, n_assets+n_features, H, W]

        # Encode to latent space
        z_t, skip_features = self.encoder(x)  # z_t: [batch, latent_dim, h, w]

        # Evolve in latent space (capture long-range spatial dependencies)
        z_t_evolved = self.ste(z_t)  # [batch, latent_dim, h, w]

        # Decode back to tendency space
        G_neural = self.decoder(z_t_evolved, skip_features)  # [batch, n_assets, H, W]

        return G_neural
```

---

## Integration: ODE Solver

```python
class NeuralMarketODE(nn.Module):
    """
    Hybrid physics-AI market model.

    Combines:
        - G_physics: Learnable market physics (momentum, reversion, volatility, correlation)
        - G_neural: Neural corrector (order flow, sentiment, black swans)

    Integration via Forward Euler ODE solver.
    """

    def __init__(self, n_assets=64, n_features=10, dt=1.0):
        super().__init__()

        self.physics_core = DifferentiableMarketPhysics(n_assets=n_assets)
        self.neural_corrector = NeuralMarketCorrector(n_assets=n_assets, n_features=n_features)
        self.dt = dt  # Time step (e.g., 1 minute)

    def forward(self, state_t, features_t):
        """
        One-step prediction: state_{t+Δt} = state_t + Δt·(G_physics + G_neural)

        Args:
            state_t: dict with 'prices', 'velocities', 'moving_avg', 'noise'
            features_t: [batch, n_features, H, W] external features

        Returns:
            state_{t+dt}: dict with updated 'prices'
        """
        # Compute physics tendency
        G_physics = self.physics_core(state_t)  # [batch, n_assets, H, W]

        # Compute neural tendency
        G_neural = self.neural_corrector(state_t['prices'], features_t)  # [batch, n_assets, H, W]

        # Integrate (Forward Euler)
        prices_new = state_t['prices'] + self.dt * (G_physics + G_neural)

        # Update state (compute new velocities for next step)
        velocities_new = (prices_new - state_t['prices']) / self.dt

        state_new = {
            'prices': prices_new,
            'velocities': velocities_new,
            'moving_avg': state_t['moving_avg'],  # Updated externally
            'noise': torch.randn_like(prices_new)  # New noise for next step
        }

        return state_new
```

---

## Training

### Loss Function

```python
def train_step(model, batch):
    """
    Single-step prediction training (same as NeuralOGCM).

    Args:
        model: NeuralMarketODE
        batch: dict with 'state_t', 'features_t', 'state_target'

    Returns:
        loss: MSE between prediction and ground truth
    """
    state_t = batch['state_t']
    features_t = batch['features_t']
    state_target = batch['state_target']  # Ground truth at t+Δt

    # Forward pass
    state_pred = model(state_t, features_t)

    # Compute loss (MSE on prices)
    loss = F.mse_loss(state_pred['prices'], state_target['prices'])

    return loss
```

### Key Training Details (from NeuralOGCM)

- **Optimizer**: AdamW with lr=1e-4, weight_decay=1e-5
- **Batch size**: 2 (limited by memory, each batch has full spatial resolution)
- **Normalization**: Physics core operates in physical space, neural in normalized space
- **End-to-end**: Both θ_physics and θ_neural trained jointly

### Multi-Step Rollout (Autoregressive)

```python
def rollout(model, state_0, features_sequence, n_steps):
    """
    Autoregressive rollout for long-term prediction.

    Args:
        model: NeuralMarketODE
        state_0: Initial state
        features_sequence: [n_steps, batch, n_features, H, W]
        n_steps: Number of steps to predict

    Returns:
        trajectory: [n_steps+1, batch, n_assets, H, W]
    """
    trajectory = [state_0['prices']]
    state = state_0

    for t in range(n_steps):
        features_t = features_sequence[t]
        state = model(state, features_t)
        trajectory.append(state['prices'])

    return torch.stack(trajectory, dim=0)
```

---

## Integration with MuZero Agent

### Option 1: Prediction Head (TASK_05 Integration)

Add the **NeuralMarketODE** as a **prediction head** to the MuZero latent space:

```python
class MuZeroWithMarketPhysics(nn.Module):
    def __init__(self, muzero_model, market_physics_model):
        super().__init__()

        self.muzero = muzero_model  # Existing TRM model
        self.market_physics = market_physics_model  # NeuralMarketODE

    def prediction(self, latent_state):
        """
        Augment MuZero prediction with market physics prediction.

        Args:
            latent_state: [batch, state_dim] from TRM

        Returns:
            policy_logits, value, price_prediction
        """
        # Standard MuZero prediction
        policy_logits, value = self.muzero.prediction(latent_state)

        # Decode latent to market state
        state_t = self.decode_latent(latent_state)

        # Physics-based price prediction
        state_pred = self.market_physics(state_t, features_t=None)
        price_prediction = state_pred['prices']

        return policy_logits, value, price_prediction

    def auxiliary_loss(self, latent_state, target_prices):
        """
        Auxiliary loss for price prediction (improves latent space).
        """
        _, _, price_pred = self.prediction(latent_state)
        loss_pred = F.mse_loss(price_pred, target_prices)
        return loss_pred
```

**Training**:
```python
# Total loss = MuZero loss + auxiliary prediction loss
loss_total = loss_policy + loss_value + loss_reward + λ_pred * loss_prediction
```

**Benefits**:
- Forces latent space to encode "what will happen" (prices)
- Learnable physics parameters improve interpretability
- Long-term stability from physics core
- No black-box hallucinations (physics constraints)

### Option 2: Environment Augmentation

Use **NeuralMarketODE** to **augment the environment** with predicted future states:

```python
class PhysicsAugmentedEnv(TradingEnv):
    def __init__(self, market_physics_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_model = market_physics_model

    def get_observation(self, step_idx):
        """
        Add physics-predicted future states to observation.
        """
        obs = super().get_observation(step_idx)

        # Current market state
        state_t = self.extract_state(step_idx)

        # Predict next 5 steps using physics model
        future_states = self.physics_model.rollout(state_t, n_steps=5)

        # Append to observation
        obs['future_predictions'] = future_states

        return obs
```

**Benefits**:
- Agent sees model's "intuition" about future
- Reduces need for deep MCTS search (physics already extrapolates)
- Can still learn to ignore predictions if unreliable

---

## Expected Benefits

### 1. Long-Term Stability
**Problem**: Pure neural models accumulate errors during autoregressive rollout
**Solution**: Physics core provides stable baseline, neural only corrects residuals

### 2. Physical Plausibility
**Problem**: Black-box models can hallucinate impossible price movements
**Solution**: Physics core enforces momentum/reversion/correlation constraints

### 3. Interpretability
**Problem**: Can't explain why model predicts certain moves
**Solution**: Inspect learned physics parameters (α, β, σ, γ)
- "Model learned α=0.92 → strong momentum in this market regime"
- "γ_BTC,ETH=0.85 → model discovered Bitcoin-Ethereum coupling"

### 4. Sample Efficiency
**Problem**: Need massive data to learn market dynamics from scratch
**Solution**: Physics prior reduces learning burden (only learn corrections)

### 5. Generalization
**Problem**: Overfitting to training regime, fails on new market conditions
**Solution**: Physics parameters adapt via end-to-end learning

---

## Implementation Roadmap

### Phase 1: Core Physics Implementation (3-4 hours)
- [ ] Implement `DifferentiableMarketPhysics` with 4 components
- [ ] Test parameter gradients flow correctly
- [ ] Verify physics tendencies are reasonable (momentum should amplify velocity, etc.)

### Phase 2: Neural Corrector (4-5 hours)
- [ ] Implement `NeuralMarketCorrector` (Encoder-STE-Decoder)
- [ ] Use existing `TRMEncoder` as starting point
- [ ] Add multi-head attention for STE module

### Phase 3: ODE Integration (2-3 hours)
- [ ] Implement `NeuralMarketODE` with Forward Euler solver
- [ ] Create training loop (single-step MSE loss)
- [ ] Test on 1 asset first (sanity check)

### Phase 4: Full System Training (5-6 hours)
- [ ] Train on historical data (64 assets, 30 days)
- [ ] Monitor physics parameter evolution
- [ ] Evaluate rollout stability (10-step, 50-step, 200-step)

### Phase 5: MuZero Integration (6-8 hours)
- [ ] Add as prediction head (TASK_05)
- [ ] Tune auxiliary loss weight λ_pred
- [ ] Benchmark: With physics vs without physics

---

## Potential Challenges

### 1. Spatial Grid Representation
**Ocean model**: Uses 2D spatial grid (latitude × longitude)
**Trading**: Don't have natural 2D grid for assets

**Solutions**:
- A) **Graph representation**: Assets as nodes, correlation as edges
- B) **1D sequence**: Treat assets as sequence, use 1D convolutions
- C) **Correlation grid**: Embed assets in 2D via t-SNE/correlation clustering

**Recommendation**: Start with **1D sequence** (simplest), upgrade to graph later

### 2. Noise Handling
**Ocean model**: Deterministic dynamics (low noise)
**Trading**: High noise, low signal-to-noise ratio

**Solutions**:
- Increase neural corrector capacity (it handles noise)
- Add noise variance as learnable parameter (σ_noise)
- Use multiple rollouts with different noise samples (ensemble)

### 3. Non-Stationarity
**Ocean model**: Physics doesn't change over time
**Trading**: Market regime shifts (bull→bear, low→high volatility)

**Solutions**:
- **Conditional physics**: θ_physics = f(regime_indicator)
- **Meta-learning**: Learn to adapt physics parameters quickly
- **Mixture of experts**: Multiple physics cores, gate selects active one

---

## Next Steps

**Immediate**:
1. Read experiments section (`04_exps.tex`) to understand NeuralOGCM's empirical results
2. Prototype `DifferentiableMarketPhysics` (1D sequence version)
3. Test if physics parameters can be learned from synthetic data (Level 0 sine waves)

**Short-term**:
1. Create TASK_07: "Implement Neural Market Physics Prediction Head"
2. Integrate with existing TRM model
3. Benchmark on historical data

**Long-term**:
1. Extend to graph-based representation (asset correlation network)
2. Add regime-conditional physics
3. Scale to 1000+ assets (cryptocurrency universe)

---

## References

- **NeuralOGCM**: arXiv:2512.11525v1 (Differentiable Ocean Modeling)
- **Neural ODE**: Chen et al. 2018 (Continuous-depth models)
- **Differentiable Physics**: Kochkov et al. 2024 (ML-accelerated CFD)
- **Market Microstructure**: Hasbrouck 2007 (Empirical market dynamics)

---

## Conclusion

The **NeuralOGCM paradigm** is a perfect fit for financial markets:
- **Physics core**: Captures mean reversion, momentum, volatility (market laws)
- **Neural corrector**: Handles order flow, sentiment, black swans (complex patterns)
- **Learnable parameters**: Discover optimal market dynamics from data
- **End-to-end training**: Both components co-optimize

This approach combines the best of physics-based models (stability, interpretability) and neural networks (flexibility, pattern recognition) for stable long-term market prediction.

**Recommendation**: Prototype this as TASK_07 after completing current model scaling work.
