# MCTS Trading Agent - Algorithm Research & Best Practices (2025 Update)

## Executive Summary

This document updates the research recommendations for November 2025, incorporating breakthrough developments including Tiny Recursive Models (TRM), UniZero, DreamerV3, and advancements in high-frequency crypto trading with deep reinforcement learning.

**Key 2025 Breakthroughs:**
- **TRM (Samsung SAIL Montreal)**: 7M parameter model achieving 45% on ARC-AGI-1 using recursive reasoning
- **UniZero**: Transformer-based latent world model for MCTS, accepted to TMLR 2025
- **DreamerV3**: Published in Nature (April 2025), general-purpose world model across 150+ tasks
- **Robust MCTS**: New variants addressing simulation-to-reality gap (June 2025)
- **High-Frequency Crypto DRL**: Adversarial-robust frameworks achieving 94.3% baseline performance

---

## 1. State Encoding - Best Algorithms (2025 Update)

### Primary: HSV Color Encoding (Unchanged, Still Recommended)
**Why**: Intuitive mapping, CNN-friendly, proven in practice.

| Channel | Encoding | Research Basis |
|---------|----------|----------------|
| **Hue** | Relative price position (0-1, cyclical) | Captures price momentum and mean-reversion |
| **Saturation** | Support/Resistance strength | Technical analysis integration |
| **Value/Lightness** | Normalized volume | Activity/liquidity signal |

### Enhanced Alternative: Multi-Channel GAF + HSV Fusion
**New Research (2025)**: "Two forecasting model selection methods based on time series image feature augmentation" (Nature Scientific Reports, July 2025)

The IFFA (Information Fusion Feature Augmentation) method combines:
- **GASF** (Gramian Angular Summation Field)
- **GADF** (Gramian Angular Difference Field)  
- **MTF** (Markov Transition Field)
- **RP** (Recurrence Plots)

```python
# IFFA-enhanced state encoding
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

class EnhancedStateEncoder:
    def __init__(self, lookback=60):
        self.gasf = GramianAngularField(method='summation')
        self.gadf = GramianAngularField(method='difference')
        self.mtf = MarkovTransitionField(n_bins=8)
        self.rp = RecurrencePlot(threshold='point')
        
    def encode(self, price_series, volume_series):
        # Normalize price
        normalized = (price_series - price_series.min()) / (price_series.max() - price_series.min() + 1e-8)
        
        # Generate all representations
        gasf_img = self.gasf.transform(normalized.reshape(1, -1))[0]
        gadf_img = self.gadf.transform(normalized.reshape(1, -1))[0]
        mtf_img = self.mtf.transform(normalized.reshape(1, -1))[0]
        
        # HSV encoding (your original approach)
        hsv_img = self.encode_hsv(price_series, volume_series)
        
        # Stack as multi-channel input (6 channels total)
        return np.stack([gasf_img, gadf_img, mtf_img, 
                        hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]], axis=0)
```

### NEW: Recursive Feature Enhancement
**Paper**: "Less is More: Recursive Reasoning with Tiny Networks" (Samsung SAIL, Oct 2025)

Key insight: Instead of single-pass encoding, iteratively refine representations:

```python
class RecursiveStateEncoder(nn.Module):
    """TRM-inspired recursive state encoding"""
    def __init__(self, n_recursions=4):
        super().__init__()
        self.n_recursions = n_recursions
        self.refine_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, state_embedding, n_steps=None):
        n_steps = n_steps or self.n_recursions
        z = state_embedding
        for _ in range(n_steps):
            z = z + self.refine_net(z)  # Residual refinement
        return z
```

---

## 2. Neural Network Architecture - Best Algorithms (2025 Update)

### Backbone Comparison (Updated November 2025)

| Architecture | Params | RTX 3090 Speed | Accuracy | 2025 Status |
|--------------|--------|----------------|----------|-------------|
| **EfficientNetV2-S** | 21M | ★★★★☆ | ★★★★☆ | Still excellent for efficiency |
| **ConvNeXt-Tiny** | 28M | ★★★☆☆ | ★★★★★ | **Best for low-data fine-tuning** |
| **ConvNeXt V2** | 28M | ★★★☆☆ | ★★★★★ | Improved with MAE pretraining |
| **VisionMamba** | 25M | ★★★★★ | ★★★★☆ | **NEW: State-space models for efficiency** |
| **DaViT** | 28M | ★★★☆☆ | ★★★★★ | Dual attention - best for diverse inputs |

**2025 Recommendation**: ConvNeXt-Tiny for trading (best low-data fine-tuning per Sept 2025 research)

### NEW: TRM-Style Recursive Architecture for Trading

**Key Innovation**: Recursive refinement dramatically reduces parameters while improving reasoning.

```python
class TinyRecursiveTradingModel(nn.Module):
    """
    Inspired by TRM (Samsung SAIL Montreal, Oct 2025)
    7M params can outperform 671B params LLMs on reasoning tasks
    """
    def __init__(self, state_dim=64, hidden_dim=128, n_actions=5):
        super().__init__()
        
        # Lightweight encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64*4*4, state_dim)
        )
        
        # Recursive reasoning module (the key innovation)
        self.latent_refiner = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # [state, latent] -> hidden
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Output refinement
        self.output_refiner = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Initial embeddings
        self.initial_latent = nn.Parameter(torch.zeros(state_dim))
        self.initial_output = nn.Parameter(torch.zeros(n_actions))
        
        # Value head (separate from recursive policy)
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, x, H_cycles=3, L_cycles=4):
        """
        H_cycles: Number of output improvement cycles (K in paper)
        L_cycles: Number of latent refinement cycles (n in paper)
        """
        batch_size = x.shape[0]
        
        # Encode state
        state = self.encoder(x)
        
        # Initialize latent and output
        latent = self.initial_latent.expand(batch_size, -1)
        output = self.initial_output.expand(batch_size, -1)
        
        # Recursive improvement (TRM core algorithm)
        for h in range(H_cycles):
            # Refine latent given state and current output
            for l in range(L_cycles):
                combined = torch.cat([state, latent], dim=-1)
                latent = latent + self.latent_refiner(combined)  # Residual
            
            # Refine output given latent
            output_input = torch.cat([latent, output], dim=-1)
            output = output + self.output_refiner(output_input)  # Residual
        
        policy = F.softmax(output, dim=-1)
        value = self.value_head(state)
        
        return policy, value
```

### Architecture Comparison: TRM vs Traditional

| Aspect | Traditional CNN+MCTS | TRM-Style Recursive |
|--------|---------------------|---------------------|
| Parameters | 10-50M | 5-10M |
| Forward passes | 1 | K×n (configurable) |
| Memory | Fixed | Adaptive via recursion depth |
| Error correction | None | Built-in via refinement |
| Effective depth | N layers | N × (K×L+1) |

---

## 3. MCTS Algorithm - Best Variant (2025 Update)

### Core: AlphaZero-style PUCT (Still Recommended)

**Core Formula** (unchanged):
```
UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

### NEW 2025 Variants:

#### 1. Robust MCTS (June 2025)
**Paper**: "Online Robust Reinforcement Learning Through Monte-Carlo Planning" (OpenReview)

Key innovation: Power mean backup operator for handling model uncertainty:
```python
def robust_backup(node, config, uncertainty_set):
    """Robust backup with ambiguity set consideration"""
    children_values = [child.q_value() for child in node.children.values()]
    
    # Power mean instead of simple average (more robust to outliers)
    power = config.robustness_power  # Typically -1 to 1
    if power == 0:
        robust_value = np.exp(np.mean(np.log(np.abs(children_values) + 1e-8)))
    else:
        robust_value = np.power(
            np.mean(np.power(np.abs(children_values) + 1e-8, power)), 
            1/power
        )
    
    return robust_value * np.sign(np.mean(children_values))
```

#### 2. Epistemic MCTS (April 2025 Update)
**Paper**: "Epistemic Monte Carlo Tree Search" (arXiv v5)

Accounts for uncertainty in learned models - critical for trading:
```python
class EpistemicMCTS:
    def __init__(self, model_ensemble, config):
        self.models = model_ensemble  # Multiple models for uncertainty
        self.config = config
    
    def compute_epistemic_bonus(self, state):
        """Higher bonus for states with high model disagreement"""
        predictions = [m(state) for m in self.models]
        variance = torch.var(torch.stack(predictions), dim=0)
        return self.config.epistemic_weight * torch.sqrt(variance)
```

#### 3. UniZero-style Transformer MCTS (TMLR 2025)
**Paper**: "UniZero: Generalized and Efficient Planning with Scalable Latent World Models"

Key advancement: Transformer-based latent world model decouples history from state:
```python
class UniZeroStyleMCTS:
    """
    Key insight: Use transformer for long-term memory,
    separate from MCTS state representation
    """
    def __init__(self, config):
        self.world_model = TransformerWorldModel(config)
        self.history_encoder = nn.TransformerEncoder(...)
        
    def search(self, observations_history):
        # Encode full history with transformer (handles POMDP)
        history_embedding = self.history_encoder(observations_history)
        
        # Extract current latent state
        current_state = self.world_model.extract_state(history_embedding)
        
        # Run standard MCTS with enhanced state
        return self._mcts_search(current_state, history_embedding)
```

### Updated Hyperparameters (2025 Trading Optimized)

| Parameter | AlphaZero Default | 2025 Trading Optimized | Change Rationale |
|-----------|-------------------|------------------------|------------------|
| `c_puct` | 1.0 | 2.0 | Higher exploration for non-stationary markets |
| `num_simulations` | 800 | 50-100 | Sufficient for 5 actions, prioritize speed |
| `root_dirichlet_alpha` | 0.3 | 0.5 | More exploration at root |
| `discount` | 1.0 | 0.995 | Slight discount for uncertainty |
| `temperature` | 1.0 | 0.5-1.0 (decay) | Start explorative, become selective |
| **NEW: `robustness_power`** | N/A | -0.5 | For robust backup |
| **NEW: `epistemic_weight`** | N/A | 0.1 | Uncertainty bonus |

---

## 4. Training Algorithm - Best Practices (2025 Update)

### Core: Self-Play Training (Enhanced with 2025 Techniques)

#### NEW: ReZero Techniques (2025)
**Paper**: Accelerates MuZero/EfficientZero training significantly

```python
class ReZeroTrainer:
    """
    Just-in-time reanalyze + Speedy reanalyze
    Can provide 2-3x speedup with same sample efficiency
    """
    def __init__(self, config):
        self.reanalyze_ratio = config.reanalyze_ratio  # 0.25-0.5
        self.temporal_reuse = True  # Use cached search results
        
    def reanalyze_batch(self, batch, model):
        """Re-run MCTS on old states with current model"""
        reanalyzed = []
        for experience in batch:
            if random.random() < self.reanalyze_ratio:
                # Run fresh MCTS with updated model
                new_policy = self.mcts.search(experience.state, model)
                new_value = model.value(experience.state)
                reanalyzed.append(Experience(
                    state=experience.state,
                    policy=new_policy,  # Updated target
                    value=new_value     # Updated target
                ))
            else:
                reanalyzed.append(experience)
        return reanalyzed
```

#### NEW: Recursive Self-Improvement (TRM-Inspired)
```python
class RecursiveTrainingLoop:
    """
    Train model to improve its own outputs through recursion
    """
    def train_step(self, batch):
        # Forward with current recursion depth
        policy_1, value_1 = self.model(batch.states, H_cycles=1, L_cycles=2)
        policy_k, value_k = self.model(batch.states, H_cycles=4, L_cycles=4)
        
        # Distillation loss: learn to predict improved output from fewer cycles
        distill_loss = F.kl_div(
            F.log_softmax(policy_1, dim=-1),
            F.softmax(policy_k.detach(), dim=-1),  # Detach deeper prediction
            reduction='batchmean'
        )
        
        # Standard policy/value losses
        policy_loss = F.cross_entropy(policy_k, batch.target_policy)
        value_loss = F.mse_loss(value_k, batch.target_value)
        
        total_loss = policy_loss + value_loss + 0.1 * distill_loss
        return total_loss
```

#### Enhanced Learning Rate Schedule (2025)
```python
# Cosine annealing with warmup + restarts (found effective for trading)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,          # First restart after 1000 steps
    T_mult=2,          # Double period after each restart
    eta_min=1e-6
)

# Alternative: OneCycleLR with longer warmup
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-4,
    total_steps=num_epochs * steps_per_epoch,
    pct_start=0.15,    # 15% warmup (increased from 10%)
    div_factor=25,
    final_div_factor=10000,
    anneal_strategy='cos'
)
```

---

## 5. LightZero Framework - 2025 Status

**Latest Version**: v0.2.0 (April 2025)

### Key 2025 Updates:
- **UniZero integration** (accepted TMLR June 2025)
- **Multi-Agent MuZero/EfficientZero**
- **C++ MCTS tree** (10x faster than Python)
- **Gymnasium support** (replaces deprecated gym)

### Recommended Configuration for Trading:
```python
from lzero.policy import MuZeroPolicy
from lzero.model import MuZeroModel

trading_config = dict(
    model=dict(
        observation_shape=(6, 64, 64),  # Multi-channel (GASF, GADF, MTF, HSV)
        action_space_size=5,
        num_res_blocks=4,
        norm_type='LN',  # LayerNorm more stable for trading
    ),
    mcts=dict(
        num_simulations=100,  # Reduced for HFT
        root_dirichlet_alpha=0.5,
        root_noise_weight=0.25,
    ),
    learn=dict(
        batch_size=256,
        learning_rate=2e-4,
        reanalyze_ratio=0.25,  # ReZero technique
        policy_loss_weight=1.0,
        value_loss_weight=0.5,  # Reduced - policy more important for trading
    ),
    collect=dict(
        unroll_len=5,
        game_segment_length=200,
    )
)
```

---

## 6. Kraken API Best Practices (2025 Update)

### Rate Limits Summary (Updated August 2025)

**REST API:**
| Tier | Counter Max | Decay Rate | Trading Rate |
|------|-------------|------------|--------------|
| Starter | 15 | 0.33/sec | Limited |
| Intermediate | 20 | 0.5/sec | Moderate |
| Pro | 20 | 1/sec | High |

**WebSocket V2** (Recommended for HFT):
- Level 3 order book: 200 symbols per connection (standard), 500 (pro)
- Subscription rate: 200-500/sec depending on tier
- Reconnection limit: 150 attempts per 10 minutes (Cloudflare)

### Recommended Python SDK
```bash
pip install python-kraken-sdk>=2.3.0  # Latest October 2025
```

```python
from kraken.spot import SpotWSClient
import asyncio

class TradingClient(SpotWSClient):
    """WebSocket V2 client for HFT"""
    
    async def on_message(self, message):
        if message.get("channel") == "ticker":
            await self.process_ticker(message)
        elif message.get("channel") == "executions":
            await self.process_execution(message)
    
    async def place_order(self, symbol, side, quantity, price=None):
        """Place order with rate limit awareness"""
        order_type = "limit" if price else "market"
        await self.send_message({
            "method": "add_order",
            "params": {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "order_qty": quantity,
                "limit_price": price,
                "time_in_force": "IOC"  # Immediate-or-cancel for HFT
            }
        })

# Usage
async def main():
    client = TradingClient(key="...", secret="...")
    await client.start()
    await client.subscribe({"channel": "ticker", "symbol": ["BTC/USD"]})
```

### High-Frequency Trading Considerations (256 trades/day)

With free trades, 256 trades/day = ~10.67 trades/hour = 1 trade every 5.6 minutes.

**Rate Budget Calculation:**
- Each AddOrder: 0-3 points (depends on fill rate)
- Each CancelOrder: 3-8 points
- Decay rate (Pro): 1 point/second
- **Safe limit**: ~1 trade every 3-5 seconds with immediate fills

```python
class RateLimitedTrader:
    def __init__(self, tier='pro'):
        self.counter = 0
        self.max_counter = 180 if tier == 'pro' else 125
        self.decay_rate = 1.0 if tier == 'pro' else 0.5
        self.last_update = time.time()
    
    async def can_trade(self):
        # Update counter with decay
        elapsed = time.time() - self.last_update
        self.counter = max(0, self.counter - elapsed * self.decay_rate)
        self.last_update = time.time()
        
        return self.counter < self.max_counter - 10  # Leave buffer
```

---

## 7. Reward Function Design (2025 Update)

### NEW: Adversarial-Robust Reward (November 2025)
**Paper**: "Adversarial-Robust Deep Reinforcement Learning for High-Frequency Cryptocurrency Trading"

```python
class RobustRewardFunction:
    """
    Combines multiple reward signals with adversarial consideration
    """
    def __init__(self, risk_free_rate=0.0, transaction_cost=0.0005):
        self.rf = risk_free_rate
        self.tc = transaction_cost
        self.sharpe_window = 20
        self.A = 0  # Running mean (Differential Sharpe)
        self.B = 0  # Running second moment
        self.eta = 0.001
        
    def compute_reward(self, position, price_return, position_change):
        """Multi-objective reward with risk awareness"""
        
        # 1. Raw return (with transaction cost)
        raw_return = position * price_return - abs(position_change) * self.tc
        
        # 2. Differential Sharpe Ratio (online approximation)
        dsr = self._differential_sharpe(raw_return)
        
        # 3. Drawdown penalty
        dd_penalty = self._drawdown_penalty()
        
        # 4. Regime-aware scaling
        volatility = self._estimate_volatility()
        regime_scale = 1.0 / (1.0 + volatility)  # Reduce reward in high-vol
        
        # Combined reward
        reward = regime_scale * (
            0.5 * raw_return +      # Short-term profit
            0.3 * dsr +             # Risk-adjusted return
            0.2 * (-dd_penalty)     # Drawdown avoidance
        )
        
        return reward
    
    def _differential_sharpe(self, current_return):
        """Moody & Wu (1997) + 2025 stability improvements"""
        delta_A = current_return - self.A
        delta_B = current_return**2 - self.B
        
        # Update running estimates
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B
        
        # Compute DSR with numerical stability
        variance = self.B - self.A**2
        if variance > 1e-8:
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / (variance ** 1.5)
        else:
            dsr = 0.0
        
        return np.clip(dsr, -1, 1)
```

### Risk-Adjusted Metrics for HFT
```python
class HFTMetrics:
    """Metrics optimized for high-frequency trading"""
    
    @staticmethod
    def calmar_ratio(returns, period=252):
        """Annual return / Max drawdown"""
        annual_return = np.mean(returns) * period
        max_dd = HFTMetrics.max_drawdown(returns)
        return annual_return / (max_dd + 1e-8)
    
    @staticmethod  
    def sortino_ratio(returns, target=0, period=252):
        """Return / Downside deviation (better than Sharpe for trading)"""
        downside = returns[returns < target]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-8
        return (np.mean(returns) - target) * np.sqrt(period) / downside_std
    
    @staticmethod
    def profit_factor(returns):
        """Gross profit / Gross loss"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / (losses + 1e-8)
```

---

## 8. Hardware Optimization (2025 Update)

### RTX 3090 Targets (Updated with 2025 Benchmarks)

| Metric | 2024 Target | 2025 Target | How to Achieve |
|--------|-------------|-------------|----------------|
| GPU Utilization | >70% | >85% | Larger batches, mixed precision, TRM architecture |
| Memory Usage | 18-20GB | 20-22GB | Gradient checkpointing, bf16 |
| Throughput | 1000 states/sec | 5000+ states/sec | TRM (fewer params), C++ MCTS |
| Training Speed | 50 episodes/hour | 200+ episodes/hour | ReZero techniques, async data |

### Memory Budget (Updated for 6-Channel Input)

| Component | Memory (FP16/BF16) |
|-----------|-------------------|
| Model weights (TRM-style, 7M params) | ~14MB |
| Model weights (ConvNeXt-Tiny, 28M params) | ~56MB |
| Activations (batch=256) | ~3GB |
| Gradients | ~same as weights |
| MCTS trees (16 parallel) | ~2GB |
| Replay buffer on GPU (100k experiences) | ~10GB |
| **Total (TRM)** | ~16GB |
| **Headroom (TRM)** | ~8GB |

### Mixed Precision Training (BF16 Recommended for 2025)
```python
# BF16 more stable than FP16 for transformers
scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16 doesn't need scaler

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    policy, value = model(states)
    loss = compute_loss(policy, value, targets)

loss.backward()
optimizer.step()
```

---

## 9. Integration: TRM + MCTS for Trading

### Complete Architecture (Recommended 2025)

```python
class TRMMCTSTrader:
    """
    Combines TRM's recursive reasoning with MCTS planning
    - TRM provides policy/value estimates
    - MCTS provides look-ahead planning
    - Recursive refinement improves both
    """
    
    def __init__(self, config):
        self.config = config
        
        # State encoding
        self.encoder = EnhancedStateEncoder(
            lookback=config.lookback_window,
            use_gasf=True,
            use_mtf=True
        )
        
        # TRM-style model
        self.model = TinyRecursiveTradingModel(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            n_actions=config.n_actions
        )
        
        # MCTS with epistemic uncertainty
        self.mcts = EpistemicMCTS(
            model=self.model,
            num_simulations=config.mcts_simulations,
            c_puct=config.c_puct
        )
        
    def get_action(self, market_data):
        """Get trading action with full pipeline"""
        
        # 1. Encode state
        state = self.encoder.encode(
            market_data['prices'], 
            market_data['volumes']
        )
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        # 2. Get initial policy/value from TRM (multiple recursion depths)
        with torch.no_grad():
            # Quick estimate (for MCTS prior)
            policy_quick, value_quick = self.model(state_tensor, H_cycles=1, L_cycles=2)
            
            # Deep estimate (final decision)
            policy_deep, value_deep = self.model(state_tensor, H_cycles=4, L_cycles=4)
        
        # 3. Run MCTS with TRM as policy/value network
        mcts_policy = self.mcts.search(
            state_tensor,
            prior_policy=policy_quick.cpu().numpy()[0]
        )
        
        # 4. Blend MCTS result with deep TRM prediction
        final_policy = self._blend_policies(
            mcts_policy, 
            policy_deep.cpu().numpy()[0],
            temperature=self.config.temperature
        )
        
        # 5. Select action
        action = np.random.choice(self.config.n_actions, p=final_policy)
        
        return action, {
            'policy': final_policy,
            'value': value_deep.item(),
            'mcts_visits': self.mcts.get_visit_counts()
        }
    
    def _blend_policies(self, mcts_policy, trm_policy, temperature=1.0):
        """Blend MCTS and TRM policies"""
        # Convert to logits
        mcts_logits = np.log(mcts_policy + 1e-8)
        trm_logits = np.log(trm_policy + 1e-8)
        
        # Average logits (gives geometric mean of probabilities)
        blended_logits = 0.6 * mcts_logits + 0.4 * trm_logits
        
        # Apply temperature
        blended_logits = blended_logits / temperature
        
        # Softmax
        exp_logits = np.exp(blended_logits - np.max(blended_logits))
        return exp_logits / exp_logits.sum()
```

---

## 10. High-Frequency Trading Pipeline (2025)

### Complete HFT System Architecture

```python
import asyncio
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingConfig:
    # Model
    state_dim: int = 64
    hidden_dim: int = 128
    n_actions: int = 5  # Strong sell, sell, hold, buy, strong buy
    lookback_window: int = 60
    
    # MCTS
    mcts_simulations: int = 50  # Reduced for HFT
    c_puct: float = 2.0
    
    # TRM
    H_cycles: int = 3
    L_cycles: int = 4
    
    # Trading
    max_position: float = 1.0
    max_trades_per_hour: int = 10  # 240/day
    min_trade_interval: int = 180  # seconds
    
    # Hardware
    device: str = 'cuda:0'

class HFTTradingSystem:
    """Complete high-frequency trading system"""
    
    def __init__(self, config: TradingConfig, api_key: str, api_secret: str):
        self.config = config
        self.trader = TRMMCTSTrader(config)
        self.client = TradingClient(key=api_key, secret=api_secret)
        
        # State tracking
        self.position = 0.0
        self.last_trade_time = 0
        self.daily_trades = 0
        self.portfolio_value = 10000.0
        
        # Market data buffer
        self.price_buffer = []
        self.volume_buffer = []
        
    async def run(self):
        """Main trading loop"""
        await self.client.start()
        await self.client.subscribe({
            "channel": "ohlc",
            "symbol": ["BTC/USD"],
            "interval": 1  # 1 minute
        })
        
        while True:
            try:
                # Wait for new data
                data = await self.client.receive()
                
                if data.get("channel") == "ohlc":
                    await self.process_candle(data)
                    
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1)
    
    async def process_candle(self, data):
        """Process new OHLC candle and potentially trade"""
        candle = data["data"][0]
        
        # Update buffers
        self.price_buffer.append(float(candle["close"]))
        self.volume_buffer.append(float(candle["volume"]))
        
        # Keep buffer size manageable
        if len(self.price_buffer) > self.config.lookback_window * 2:
            self.price_buffer = self.price_buffer[-self.config.lookback_window:]
            self.volume_buffer = self.volume_buffer[-self.config.lookback_window:]
        
        # Check if we have enough data
        if len(self.price_buffer) < self.config.lookback_window:
            return
        
        # Check trade timing constraints
        current_time = time.time()
        if current_time - self.last_trade_time < self.config.min_trade_interval:
            return
        
        # Get trading decision
        market_data = {
            'prices': np.array(self.price_buffer),
            'volumes': np.array(self.volume_buffer)
        }
        
        action, info = self.trader.get_action(market_data)
        
        # Convert action to position change
        target_position = self._action_to_position(action)
        position_change = target_position - self.position
        
        # Execute trade if significant
        if abs(position_change) > 0.1:
            await self.execute_trade(position_change, float(candle["close"]))
    
    def _action_to_position(self, action: int) -> float:
        """Map action index to target position"""
        position_map = {
            0: -1.0,   # Strong sell (full short)
            1: -0.5,   # Sell (half short)
            2: 0.0,    # Hold (neutral)
            3: 0.5,    # Buy (half long)
            4: 1.0     # Strong buy (full long)
        }
        return position_map[action]
    
    async def execute_trade(self, position_change: float, current_price: float):
        """Execute trade with proper risk management"""
        # Calculate order size
        order_value = abs(position_change) * self.portfolio_value
        quantity = order_value / current_price
        
        side = "buy" if position_change > 0 else "sell"
        
        logging.info(f"Executing {side} order: {quantity:.6f} BTC @ ~${current_price:.2f}")
        
        await self.client.place_order(
            symbol="BTC/USD",
            side=side,
            quantity=round(quantity, 6),
            price=None  # Market order for HFT
        )
        
        self.position += position_change
        self.last_trade_time = time.time()
        self.daily_trades += 1
```

---

## 11. Evaluation Metrics (2025 Update)

### Essential Metrics for HFT

| Metric | Formula | Good Value (HFT) |
|--------|---------|-----------------|
| **Sharpe Ratio** | (μ - rf) / σ | > 2.0 (higher bar for HFT) |
| **Sortino Ratio** | (μ - rf) / σ_downside | > 3.0 |
| **Max Drawdown** | max(peak - trough) / peak | < 10% |
| **Win Rate** | wins / total_trades | > 55% |
| **Profit Factor** | gross_profit / gross_loss | > 2.0 |
| **Calmar Ratio** | annual_return / max_drawdown | > 2.0 |
| **Avg Trade Duration** | mean(trade_duration) | 5-30 min (HFT) |
| **Trades per Day** | total_trades / days | 100-256 |

### NEW: Adversarial Robustness Metrics (2025)
```python
class AdversarialMetrics:
    @staticmethod
    def attack_success_rate(model, test_data, epsilon=0.01):
        """Measure model robustness to input perturbations"""
        successful_attacks = 0
        for state, original_action in test_data:
            # FGSM attack
            perturbed = state + epsilon * torch.sign(state.grad)
            new_action = model(perturbed).argmax()
            if new_action != original_action:
                successful_attacks += 1
        return successful_attacks / len(test_data)
    
    @staticmethod
    def performance_under_attack(model, test_data, attack_fn):
        """Compare performance with and without adversarial attacks"""
        clean_returns = evaluate(model, test_data, attack=False)
        attacked_returns = evaluate(model, test_data, attack=True)
        
        degradation = (clean_returns - attacked_returns) / clean_returns
        return {
            'clean_sharpe': sharpe(clean_returns),
            'attacked_sharpe': sharpe(attacked_returns),
            'degradation_pct': degradation * 100
        }
```

---

## 12. Training Curriculum (2025 Update)

### Phase 1: Foundation (Episodes 1-500)
- **Model**: TRM with H_cycles=1, L_cycles=2
- **MCTS**: 25 simulations
- **Reward**: Simple returns only
- **c_puct**: 3.0 (high exploration)
- **Temperature**: 1.0

### Phase 2: Recursive Depth (Episodes 500-2000)
- **Model**: Increase to H_cycles=2, L_cycles=3
- **MCTS**: 50 simulations
- **Reward**: Add Sharpe component
- **c_puct**: 2.0
- **Temperature**: 0.8
- **NEW**: Start ReZero reanalysis (ratio=0.1)

### Phase 3: Full Recursion (Episodes 2000-5000)
- **Model**: Full H_cycles=4, L_cycles=4
- **MCTS**: 100 simulations
- **Reward**: Full reward with drawdown penalty
- **c_puct**: 1.5
- **Temperature**: 0.5 → 0.25
- **ReZero**: ratio=0.25

### Phase 4: Robustness (Episodes 5000+)
- **Add**: Adversarial training
- **Add**: Epistemic MCTS
- **Add**: Domain randomization (synthetic market scenarios)
- **c_puct**: 1.0-1.5
- **Temperature**: 0.25

---

## Quick Reference: Algorithm Choices (2025)

| Module | 2024 Recommendation | 2025 Recommendation | Change Reason |
|--------|---------------------|---------------------|---------------|
| **State Encoding** | HSV Color | HSV + GASF/MTF (6-channel) | Richer temporal representation |
| **Backbone** | EfficientNet-B0 | TRM-style recursive (7M params) | 10x fewer params, better reasoning |
| **MCTS** | PUCT (AlphaZero) | PUCT + Epistemic bonus | Handles model uncertainty |
| **Training** | Self-Play + PER | Self-Play + ReZero + TRM distillation | 2-3x faster |
| **Reward** | Differential Sharpe | Robust multi-objective | Adversarial resilience |
| **Optimizer** | AdamW | AdamW + AdamAtan2 (TRM paper) | More stable for recursion |
| **LR Schedule** | OneCycleLR | CosineAnnealingWarmRestarts | Better for long training |
| **Framework** | LightZero 0.0.5 | LightZero 0.2.0 | UniZero, better C++ MCTS |

---

## References (2025)

1. **Jolicoeur-Martineau, A.** "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871, Oct 2025) - **TRM**
2. **Wang, G. et al.** "Hierarchical Reasoning Model" (arXiv:2506.21734, 2025) - **HRM**
3. **Niu, Y. et al.** "UniZero: Generalized and Efficient Planning with Scalable Latent World Models" (TMLR, June 2025)
4. **Hafner, D. et al.** "Mastering diverse control tasks through world models" (Nature, April 2025) - **DreamerV3**
5. **OpenReview** "Online Robust Reinforcement Learning Through Monte-Carlo Planning" (June 2025) - **Robust MCTS**
6. **Oren, Y. et al.** "Epistemic Monte Carlo Tree Search" (arXiv v5, April 2025)
7. **ResearchSquare** "Adversarial-Robust Deep Reinforcement Learning for High-Frequency Cryptocurrency Trading" (Nov 2025)
8. **Chun & Lee** "Cryptocurrency Futures Portfolio Trading System Using Reinforcement Learning" (Applied Sciences, Aug 2025)
9. **MDPI Sensors** "Gramian Angular Field and CNNs for Real-Time Spectrum Sensing" (June 2025)
10. **Nature Scientific Reports** "Two forecasting model selection methods based on time series image feature augmentation" (July 2025)
11. **OpenDILab** LightZero v0.2.0 (April 2025) - https://github.com/opendilab/LightZero
12. **Samsung SAIL Montreal** TinyRecursiveModels (Oct 2025) - https://github.com/SamsungSAILMontreal/TinyRecursiveModels
13. **Kraken** API Documentation (Aug 2025) - https://docs.kraken.com/api/

---

## Appendix: Quick Start Code

### Minimal TRM+MCTS Trading Agent

```python
# pip install torch lightzero python-kraken-sdk pyts

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Full implementation available in project files
# See: enhanced_agent.py, main.py

if __name__ == "__main__":
    config = TradingConfig()
    system = HFTTradingSystem(
        config=config,
        api_key="YOUR_KEY",
        api_secret="YOUR_SECRET"
    )
    
    # Backtest first
    backtest_results = system.backtest("data/historical_btc.csv")
    print(f"Backtest Sharpe: {backtest_results['sharpe']:.2f}")
    
    # Then run live (paper trading recommended first)
    # asyncio.run(system.run())
```
