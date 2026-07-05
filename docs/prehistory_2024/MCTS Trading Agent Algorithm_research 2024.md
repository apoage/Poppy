# MCTS Trading Agent - Algorithm Research & Best Practices

## Executive Summary

This document summarizes the best algorithms and techniques for each module of the MCTS trading system, based on current research (2024).

---

## 1. State Encoding - Best Algorithms

### Primary: HSV Color Encoding
**Why**: Intuitive mapping of market features to color channels, CNN-friendly representation.

| Channel | Encoding | Research Basis |
|---------|----------|----------------|
| **Hue** | Relative price position (0-1, cyclical) | Captures price momentum and mean-reversion |
| **Saturation** | Support/Resistance strength | Technical analysis integration |
| **Value** | Normalized volume | Activity/liquidity signal |

### Alternative: Gramian Angular Field (GAF)
**Paper**: "Deep Learning and Time Series-to-Image Encoding for Financial Forecasting" (IEEE 2020)
**Benefits**: 
- Captures temporal correlation structure
- Works well with pretrained CNN models (ResNet, EfficientNet)
- Can be used as additional channels alongside HSV

```python
# GASF encoding formula
phi = arccos(normalized_price)
GASF[i,j] = cos(phi_i + phi_j)
```

### Cutting Edge: Time2Vec Encoding
**Paper**: "Time2Vec: Learning a Vector Representation of Time" (2019)
**Use Case**: Add learnable periodic time features to capture:
- Time-of-day effects
- Day-of-week patterns
- Seasonal cycles

---

## 2. Neural Network Architecture - Best Algorithms

### Backbone Comparison (for 64x64 image input)

| Architecture | Params | RTX 3090 Speed | Accuracy | Recommendation |
|--------------|--------|----------------|----------|----------------|
| **ResNet-18** | 11M | ★★★★★ | ★★★☆☆ | Development/Testing |
| **EfficientNet-B0** | 5M | ★★★★☆ | ★★★★☆ | Production - Best tradeoff |
| **ConvNeXt-Tiny** | 28M | ★★★☆☆ | ★★★★★ | When accuracy critical |
| **Vision Transformer** | 86M | ★★☆☆☆ | ★★★★☆ | Large dataset only |

### Recommended: Modified EfficientNet with Dual Heads

```python
class TradingNet(nn.Module):
    def __init__(self):
        self.backbone = efficientnet_b0(pretrained=False)
        self.backbone.classifier = nn.Identity()  # Remove classifier
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Policy head (action probabilities)
        self.policy = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 actions
        )
        
        # Value head (state evaluation)
        self.value = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
```

### Attention Enhancement
Add self-attention after CNN backbone for capturing long-range dependencies:
- Use Multi-Head Attention (8 heads recommended)
- Apply after flattening spatial features
- Improves capture of cross-temporal patterns

---

## 3. MCTS Algorithm - Best Variant

### AlphaZero-style PUCT (Predictor UCT)

**Core Formula**:
```
UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

**Key Hyperparameters** (optimized for trading):

| Parameter | AlphaZero Default | Trading Optimized | Reason |
|-----------|-------------------|-------------------|--------|
| `c_puct` | 1.0 | 1.5-2.0 | Higher exploration for non-stationary markets |
| `num_simulations` | 800 | 100-200 | Sufficient for 5 actions |
| `root_dirichlet_alpha` | 0.3 | 0.5 | More exploration at root |
| `discount` | 1.0 | 0.99 | Future rewards slightly discounted |

### Advanced: Entropy-Guided MCTS (EG-MCTS)
**Paper**: "Entropy-Guided Exploration in AlphaZero" (MLMI 2024)
**Key Innovation**: Add information gain term to UCB formula

```python
def enhanced_ucb(node, config):
    Q = node.q_value
    U = config.c_puct * node.prior * sqrt(node.parent.visits) / (1 + node.visits)
    
    # Entropy bonus for information gain
    if config.use_entropy_bonus:
        H = -sum(p * log(p + 1e-8) for p in node.children_priors)
        I = config.entropy_weight * H * sqrt(node.parent.visits) / (1 + node.visits)
        return Q + U + I
    
    return Q + U
```

---

## 4. Training Algorithm - Best Practices

### Self-Play Training (AlphaZero-style)

**Training Loop**:
1. **Self-Play Phase**: Generate N episodes using MCTS
2. **Store Phase**: Add (state, MCTS_policy, value) to replay buffer
3. **Train Phase**: Sample batch, update network
4. **Repeat**

### Critical Optimizations

#### a) Prioritized Experience Replay (PER)
**Paper**: Schaul et al. (2015)
**Priority Formula**: 
```
priority_i = |TD_error_i| + epsilon
P(i) = priority_i^alpha / sum(priority_j^alpha)
```

#### b) Reanalyze Technique (from MuZero Unplugged)
Re-run MCTS on old states with updated network to improve targets:
```python
def reanalyze(old_experience, network):
    new_policy = mcts.search(old_experience.state, network)
    new_value = network.value(old_experience.state)
    return Experience(old_experience.state, new_policy, new_value)
```

#### c) Learning Rate Schedule
```python
# Cosine annealing with warmup
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-4,
    total_steps=num_epochs * steps_per_epoch,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
```

---

## 5. LightZero Framework Integration

**Reference**: OpenDILab/LightZero (NeurIPS 2023 Spotlight)

### Key Components to Borrow

1. **Efficient MCTS Implementation**
   - C++ tree with Python bindings (10x faster)
   - Batched leaf evaluation
   - Root parallelization

2. **Training Pipeline**
   - Async data collection
   - GPU-optimized replay buffer
   - Automatic hyperparameter tuning

3. **Supported Algorithms**
   - AlphaZero (best for discrete actions)
   - MuZero (if learning dynamics needed)
   - EfficientZero (sample efficient)
   - Sampled MuZero (continuous actions)

### Integration Example
```python
from lzero.policy import MuZeroPolicy
from lzero.model import MuZeroModel

config = dict(
    model=dict(
        observation_shape=(3, 64, 64),
        action_space_size=5,
        num_res_blocks=4,
    ),
    mcts=dict(
        num_simulations=200,
        root_dirichlet_alpha=0.3,
    ),
    learn=dict(
        batch_size=128,
        learning_rate=2e-4,
    )
)

model = MuZeroModel(**config['model'])
policy = MuZeroPolicy(model, config)
```

---

## 6. Kraken API Best Practices

### Data Acquisition

**For Historical Data**: Use Trades endpoint (unlimited history)
```python
# GET /0/public/Trades
# Returns 1000 trades per request, paginate with 'since' parameter
# Rate limit: 1 request per second for public endpoints
```

**For Real-time Data**: WebSocket subscription
```python
# Subscribe to OHLC
{
    "event": "subscribe",
    "pair": ["XBT/USD"],
    "subscription": {"name": "ohlc", "interval": 5}
}
```

### Order Execution

**Best Practices**:
1. Always use limit orders for better fills
2. Implement exponential backoff for rate limits
3. Use nonce management for private endpoints
4. Handle partial fills gracefully

```python
# Rate limits for private endpoints
TIER_3_LIMITS = {
    'AddOrder': 60,  # per minute
    'CancelOrder': 60,
    'QueryOrders': 6
}
```

---

## 7. Reward Function Design

### Basic: Log Return
```python
reward = log(1 + position * price_return - transaction_cost)
```

### Advanced: Risk-Adjusted Return
```python
def sharpe_reward(returns, risk_free=0):
    """Rolling Sharpe as reward signal"""
    excess_returns = returns - risk_free
    return excess_returns.mean() / (excess_returns.std() + 1e-8)
```

### Research-Backed: Differential Sharpe Ratio
**Paper**: Moody & Wu (1997)
```python
def differential_sharpe(current_return, prev_A, prev_B, eta=0.001):
    """Online Sharpe ratio approximation"""
    delta_A = current_return - prev_A
    delta_B = current_return**2 - prev_B
    
    new_A = prev_A + eta * delta_A
    new_B = prev_B + eta * delta_B
    
    dsr = (prev_B * delta_A - 0.5 * prev_A * delta_B) / (prev_B - prev_A**2)**(3/2)
    return dsr, new_A, new_B
```

---

## 8. Hardware Optimization Targets

### RTX 3090 Utilization Goals

| Metric | Target | How to Achieve |
|--------|--------|----------------|
| GPU Utilization | >70% | Larger batches, parallel MCTS |
| Memory Usage | 18-20GB | Mixed precision, gradient checkpointing |
| Throughput | 1000+ states/sec | Batched inference |
| Training Speed | 50+ episodes/hour | Async data collection |

### Memory Budget (24GB VRAM)

| Component | Memory |
|-----------|--------|
| Model weights (FP16) | ~100MB |
| Activations (batch=128) | ~2GB |
| Gradients (FP16) | ~100MB |
| Optimizer states | ~200MB |
| MCTS trees (8 parallel) | ~1GB |
| Replay buffer on GPU | ~8GB |
| **Total** | ~12GB |
| **Headroom** | ~12GB |

---

## 9. Evaluation Metrics

### Essential Metrics

| Metric | Formula | Good Value |
|--------|---------|------------|
| **Sharpe Ratio** | (μ - rf) / σ | > 1.0 |
| **Sortino Ratio** | (μ - rf) / σ_downside | > 1.5 |
| **Max Drawdown** | max(peak - trough) / peak | < 20% |
| **Win Rate** | wins / total_trades | > 50% |
| **Profit Factor** | gross_profit / gross_loss | > 1.5 |
| **Calmar Ratio** | annual_return / max_drawdown | > 1.0 |

### Benchmark Comparison
Always compare against:
1. Buy & Hold baseline
2. Simple Moving Average crossover
3. Random trading (sanity check)

---

## 10. Training Curriculum

### Phase 1: Basic Learning (Epochs 1-100)
- Simple reward: raw returns
- Low MCTS simulations: 50
- High exploration: c_puct = 2.0

### Phase 2: Risk Awareness (Epochs 100-500)
- Add Sharpe component to reward
- Increase MCTS: 100
- Medium exploration: c_puct = 1.5

### Phase 3: Production Tuning (Epochs 500+)
- Full reward with risk penalties
- Full MCTS: 200
- Lower exploration: c_puct = 1.0
- Add transaction cost penalties

---

## Quick Reference: Algorithm Choices

| Module | Recommended Algorithm | Alternative |
|--------|----------------------|-------------|
| **State Encoding** | HSV Color | Gramian Angular Field |
| **Backbone** | EfficientNet-B0 | ResNet-18 |
| **MCTS** | PUCT (AlphaZero) | Entropy-Guided MCTS |
| **Training** | Self-Play + PER | MuZero Unplugged |
| **Reward** | Differential Sharpe | Risk-adjusted log return |
| **Optimizer** | AdamW | RAdam |
| **LR Schedule** | OneCycleLR | CosineAnnealing |

---

## References

1. Silver, D. et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero, 2017)
2. Schrittwieser, J. et al. "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero, 2020)
3. Ye, W. et al. "Mastering Atari Games with Limited Data" (EfficientZero, 2021)
4. Niu, Y. et al. "LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios" (NeurIPS 2023)
5. Zheng, R. et al. "Entropy-Guided Exploration in AlphaZero" (MLMI 2024)
6. Bao, W. et al. "Deep Learning and Time Series-to-Image Encoding for Financial Forecasting" (IEEE 2020)
7. Moody, J. & Wu, L. "Optimization of Trading Systems and Portfolios" (1997)
