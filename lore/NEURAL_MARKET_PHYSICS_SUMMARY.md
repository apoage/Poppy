# Neural Market Physics - Quick Reference

**Source**: arXiv:2512.11525v1 - NeuralOGCM
**Adapted by**: Claude Code, Dec 19, 2025

---

## Core Equation

```
price_{t+Δt} = price_t + Δt · [G_physics(state; θ_learned) + G_neural(state, features; θ_neural)]
```

**Two components**:
1. **G_physics**: Known market laws with **learnable parameters**
2. **G_neural**: Neural network learning residual dynamics

---

## Learnable Market Physics Components

### 1. Momentum
```python
G_momentum = α · velocity
```
- **α**: Learned momentum strength (starts ~0.8)
- Captures: "Prices in motion stay in motion"

### 2. Mean Reversion
```python
G_reversion = -β · (price - moving_average)
```
- **β**: Learned reversion strength (starts ~0.2)
- Captures: "Prices return to fair value"

### 3. Volatility (Brownian Motion)
```python
G_volatility = σ · noise
```
- **σ**: Learned volatility per asset (starts ~0.02)
- Captures: "Random price fluctuations"

### 4. Cross-Asset Correlation
```python
G_correlation = Σ_j γ_ij · (price_j - price_i)
```
- **γ**: Learned correlation matrix [64×64]
- Captures: "BTC moves → ETH follows"

---

## Neural Corrector (Black Box)

**Architecture**: Encoder → Multi-Head Attention → Decoder

**Captures**:
- Order flow dynamics
- News impact
- Whale movements
- Flash crashes
- Sentiment shifts

**Input**: Current state + external features
**Output**: Correction tendency

---

## Why This Works

**Stability**: Physics core prevents divergence
**Plausibility**: Can't hallucinate impossible dynamics
**Interpretability**: Inspect learned α, β, σ, γ
**Sample Efficiency**: Physics prior → less data needed
**Generalization**: Adapts to regime changes via learned params

---

## Integration with MuZero

### Option 1: Prediction Head (TASK_05)
Add auxiliary loss to force latent space to predict prices:

```python
loss_total = loss_policy + loss_value + loss_reward + λ · loss_price_prediction
```

**Benefit**: Latent space encodes "what will happen" (not just "what to do")

### Option 2: Environment Augmentation
Provide agent with physics-predicted future:

```python
observation = {
    'current_state': state_t,
    'physics_prediction': rollout(state_t, n_steps=5)
}
```

**Benefit**: Reduces MCTS search depth (physics already extrapolates)

---

## Quick Start Prototype

### 1D Sequence Version (Simplest)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketPhysics1D(nn.Module):
    def __init__(self, n_assets=64):
        super().__init__()
        # Learnable parameters
        self.α = nn.Parameter(torch.tensor(0.8))
        self.β = nn.Parameter(torch.tensor(0.2))
        self.σ = nn.Parameter(torch.ones(n_assets) * 0.02)

    def forward(self, prices, velocities, moving_avg, noise):
        # G_momentum
        T1 = F.softplus(self.α) * velocities

        # G_reversion
        T2 = -F.softplus(self.β) * (prices - moving_avg)

        # G_volatility
        T3 = F.softplus(self.σ).unsqueeze(0) * noise

        return T1 + T2 + T3

# Test
model = MarketPhysics1D(n_assets=64)
prices = torch.randn(1, 64)
velocities = torch.randn(1, 64)
moving_avg = torch.randn(1, 64)
noise = torch.randn(1, 64)

G_physics = model(prices, velocities, moving_avg, noise)
print(f"Physics tendency shape: {G_physics.shape}")
print(f"Learned α: {model.α.item():.3f}")
```

---

## NeuralOGCM Experimental Results (Ocean Modeling)

### Performance (RMSE on ocean prediction)

| Model | 10 days | 30 days | 60 days | 120 days |
|-------|---------|---------|---------|----------|
| **NeuralOGCM** | **0.919** | **1.280** | **1.439** | **1.574** |
| U-Net (pure AI) | 0.923 | 1.436 | 1.719 | 1.886 |
| FourCastNet | 0.959 | 1.424 | 2.433 | **3.332** |

**Critical Finding**: Pure AI model (FourCastNet) **diverges catastrophically** at 120 days (2.1× worse than NeuralOGCM)

### Visual Quality (90-day rollout)

- **NeuralOGCM**: Maintains complex spatial structures, accurate magnitudes, no artifacts
- **U-Net**: Over-saturated values, grid-like artifacts, high-frequency noise (hallucinations)

**Conclusion**: Physics core prevents long-term divergence and hallucinations

### Training Setup

- **Hardware**: 8 NVIDIA A100 GPUs
- **Batch size**: 16 (2 per GPU)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Data split**: 1993-2017 train, 2018-2019 val, 2020 test

---

## Implications for Trading

**Problem**: Pure neural trading models accumulate errors over time
- Similar to FourCastNet divergence at 120 days
- Results in hallucinations (impossible price predictions)
- Unstable in autoregressive rollout (multi-step prediction)

**Solution**: Add market physics core
- Learnable momentum (α), reversion (β), volatility (σ), correlation (γ)
- Prevents divergence (physics constraints)
- Neural only corrects residuals (order flow, sentiment, black swans)

**Expected Impact**:
- **Short-term (1-10 steps)**: Similar to pure neural (slight improvement)
- **Long-term (50-200 steps)**: **2× better stability** (based on NeuralOGCM results)
- **Interpretability**: Can inspect learned physics parameters
- **Training**: Faster convergence (physics provides structure)

---

## Expected Training Behavior

**Epoch 1-10**: Neural corrector dominates (physics params not yet tuned)
**Epoch 10-50**: Physics params converge (α, β, σ stabilize)
**Epoch 50+**: Physics handles baseline, neural only corrects anomalies

**Inspect learned params**:
```python
print(f"Momentum: α = {model.α.item():.3f}")
print(f"Reversion: β = {model.β.item():.3f}")
print(f"Volatility (BTC): σ[0] = {model.σ[0].item():.4f}")
```

**What to expect**:
- α should converge to ~0.7-0.9 (strong momentum in crypto)
- β should be small ~0.1-0.3 (weak mean reversion in trending markets)
- σ varies by asset (BTC lower volatility than small-cap altcoins)
- γ_BTC,ETH should be high ~0.8 (strong correlation)

---

## Next Steps

1. **Read experiments** (`04_exps.tex`) - See NeuralOGCM empirical results
2. **Prototype 1D physics** - Test on synthetic Level 0 data
3. **Add to TRM** - Integrate as prediction head
4. **Benchmark** - Compare: TRM vs TRM+Physics

---

## File Locations

- **Full design**: `lore/NEURAL_MARKET_PHYSICS_DESIGN.md`
- **Paper source**: `lore/arXiv-2512.11525v1/`
- **Diary entry**: `lore/DIARY.md` (Dec 19, 2025)
