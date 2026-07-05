# MCTS Trading Agent - Full System Generation Prompt for Gemini CLI

## Instructions for Use
Copy this entire prompt into Gemini CLI. The prompt is designed to generate a complete, modular MCTS trading system optimized for:
- **Hardware**: 1x RTX 3090 (24GB VRAM), 48GB RAM, Ryzen 5800X
- **Exchange**: Kraken API (spot trading)
- **Architecture**: Scalable to multi-GPU systems

---

## MASTER PROMPT START

```
You are an expert AI systems architect specializing in reinforcement learning, Monte Carlo Tree Search (MCTS), and quantitative trading systems. Generate a complete, production-ready MCTS-based trading agent system.

=============================================================================
HARDWARE CONSTRAINTS (MUST OPTIMIZE FOR)
=============================================================================
- GPU: 1x NVIDIA RTX 3090 (24GB VRAM)
- RAM: 48GB DDR4
- CPU: AMD Ryzen 5800X (8 cores, 16 threads)
- Storage: NVMe SSD (fast I/O)

The code MUST:
1. Achieve >70% GPU utilization during training
2. Use mixed precision (FP16) where beneficial
3. Use efficient batch sizes (target: 128-256 for training)
4. Support future scaling to 4x RTX 3090 with minimal code changes

=============================================================================
PROJECT STRUCTURE (GENERATE ALL FILES)
=============================================================================

mcts_trader/
├── config/
│   ├── __init__.py
│   ├── base_config.py          # Dataclass configurations
│   ├── kraken_config.yaml      # API and trading parameters
│   └── training_config.yaml    # Model and training hyperparameters
├── data/
│   ├── __init__.py
│   ├── kraken_client.py        # Kraken API wrapper (REST + WebSocket)
│   ├── data_fetcher.py         # Historical data download
│   ├── data_preprocessor.py    # Normalization, cleaning
│   ├── state_encoder.py        # Market data → Image encoding (HSV/Gramian)
│   └── replay_buffer.py        # Experience replay with prioritization
├── models/
│   ├── __init__.py
│   ├── backbone.py             # Shared CNN/Transformer backbone
│   ├── policy_head.py          # Action probability output
│   ├── value_head.py           # State value estimation
│   ├── trading_network.py      # Combined dual-head network
│   └── model_utils.py          # Initialization, checkpointing
├── mcts/
│   ├── __init__.py
│   ├── node.py                 # MCTS tree node implementation
│   ├── tree.py                 # Tree management and statistics
│   ├── search.py               # PUCT-based search algorithm
│   ├── parallel_mcts.py        # Batched parallel MCTS
│   └── mcts_config.py          # Search hyperparameters
├── environment/
│   ├── __init__.py
│   ├── trading_env.py          # Gymnasium-compatible trading environment
│   ├── market_simulator.py     # Order execution simulation
│   ├── portfolio.py            # Portfolio tracking and metrics
│   └── reward_functions.py     # Multiple reward formulations
├── training/
│   ├── __init__.py
│   ├── self_play.py            # Self-play game generation
│   ├── trainer.py              # Main training loop
│   ├── distributed_trainer.py  # Multi-GPU DDP training
│   └── curriculum.py           # Progressive difficulty scheduling
├── evaluation/
│   ├── __init__.py
│   ├── backtester.py           # Historical backtesting engine
│   ├── metrics.py              # Sharpe, Sortino, MaxDD, etc.
│   ├── visualizer.py           # Training curves, equity curves
│   └── paper_trader.py         # Live paper trading integration
├── live/
│   ├── __init__.py
│   ├── live_trader.py          # Real-time trading execution
│   ├── risk_manager.py         # Position sizing, stop-losses
│   └── order_manager.py        # Order lifecycle management
├── utils/
│   ├── __init__.py
│   ├── logging_setup.py        # Structured logging
│   ├── gpu_utils.py            # CUDA utilities, memory management
│   └── math_utils.py           # Numerical helpers
├── scripts/
│   ├── download_data.py        # Fetch Kraken historical data
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Backtesting entry point
│   ├── paper_trade.py          # Paper trading entry point
│   └── live_trade.py           # Live trading entry point
├── tests/
│   ├── test_data.py
│   ├── test_mcts.py
│   ├── test_models.py
│   └── test_environment.py
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md

=============================================================================
MODULE 1: CONFIGURATION (config/)
=============================================================================

Generate config/base_config.py with these dataclasses:

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class ActionType(Enum):
    STRONG_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    STRONG_BUY = 4

@dataclass
class DataConfig:
    """Data acquisition and preprocessing configuration"""
    exchange: str = "kraken"
    symbols: List[str] = field(default_factory=lambda: ["XXBTZUSD", "XETHZUSD"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])
    lookback_window: int = 120  # Number of bars in state
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    normalize_method: str = "rolling_zscore"  # or "minmax", "robust"
    rolling_window: int = 100

@dataclass
class StateEncodingConfig:
    """State representation configuration"""
    encoding_method: str = "hsv"  # or "gramian", "recurrence_plot", "raw"
    image_height: int = 64
    image_width: int = 64
    channels: int = 3
    include_volume: bool = True
    include_indicators: bool = True  # RSI, MACD, etc.
    
@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    backbone: str = "resnet"  # or "efficientnet", "convnext", "transformer"
    backbone_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    use_attention: bool = True
    policy_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    value_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    num_actions: int = 5
    dropout: float = 0.2
    use_layer_norm: bool = True

@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search configuration"""
    num_simulations: int = 200  # Simulations per move
    c_puct: float = 1.5  # Exploration constant
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    pb_c_base: int = 19652
    pb_c_init: float = 1.25
    discount: float = 0.997
    value_delta_max: float = 0.01
    
@dataclass  
class TrainingConfig:
    """Training loop configuration"""
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 1000
    episodes_per_epoch: int = 100
    gradient_clip: float = 1.0
    lr_scheduler: str = "cosine"  # or "step", "plateau"
    warmup_steps: int = 1000
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    entropy_bonus: float = 0.01
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    checkpoint_interval: int = 50
    eval_interval: int = 25
    
@dataclass
class ReplayBufferConfig:
    """Experience replay configuration"""
    capacity: int = 100000
    prioritized: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 100000
    
@dataclass
class TradingConfig:
    """Trading environment configuration"""
    initial_balance: float = 10000.0
    max_position_size: float = 1.0  # Fraction of portfolio
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    leverage: float = 1.0
    stop_loss: Optional[float] = 0.05  # 5%
    take_profit: Optional[float] = 0.10  # 10%
    
@dataclass
class KrakenConfig:
    """Kraken API configuration"""
    api_key: str = ""
    api_secret: str = ""
    websocket_url: str = "wss://ws.kraken.com"
    rest_url: str = "https://api.kraken.com"
    rate_limit_per_second: int = 1
    max_retries: int = 3
```

Generate config/training_config.yaml:
```yaml
# Training Configuration for MCTS Trader
# Optimized for RTX 3090 + 48GB RAM

data:
  symbols: ["XXBTZUSD"]
  timeframes: ["5m"]
  lookback_window: 120
  normalize_method: "rolling_zscore"
  
state_encoding:
  encoding_method: "hsv"
  image_height: 64
  image_width: 64
  
model:
  backbone: "resnet"
  backbone_channels: [64, 128, 256, 512]
  use_attention: true
  dropout: 0.2
  
mcts:
  num_simulations: 200
  c_puct: 1.5
  root_dirichlet_alpha: 0.3
  
training:
  batch_size: 128
  learning_rate: 0.0002
  num_epochs: 1000
  episodes_per_epoch: 50
  mixed_precision: true
  gradient_clip: 1.0
  checkpoint_interval: 50
  
replay_buffer:
  capacity: 100000
  prioritized: true
  
trading:
  initial_balance: 10000
  transaction_cost: 0.001
  
hardware:
  device: "cuda:0"
  num_workers: 4
  pin_memory: true
```

=============================================================================
MODULE 2: DATA LAYER (data/)
=============================================================================

Generate data/kraken_client.py:
- Async REST API client using aiohttp
- WebSocket client for real-time data
- Rate limiting with sliding window
- Automatic reconnection
- Methods: get_ohlc(), get_ticker(), get_order_book(), place_order(), cancel_order()

Generate data/data_fetcher.py:
- Download historical OHLCVT data from Kraken
- Use the Trades endpoint to get unlimited history (no 720 bar limit)
- Resample trades to OHLCV at any frequency
- Cache data locally in Parquet format
- Support incremental updates

Key algorithm for unlimited history:
```python
def fetch_all_trades(self, pair: str, since: int = None) -> pd.DataFrame:
    """Fetch entire trade history using pagination"""
    trades_list = []
    url = "https://api.kraken.com/0/public/Trades"
    
    while True:
        params = {"pair": pair, "since": since}
        response = requests.get(url, params=params)
        data = response.json()["result"]
        
        pair_key = [k for k in data.keys() if k != "last"][0]
        new_trades = data[pair_key]
        
        if len(new_trades) <= 1:
            break
            
        trades_list.extend(new_trades)
        since = data["last"]
        time.sleep(1)  # Rate limiting
        
    return self._trades_to_ohlcv(pd.DataFrame(trades_list))
```

Generate data/state_encoder.py with multiple encoding methods:

1. **HSV Encoding** (Primary method):
```python
def encode_hsv(self, data: pd.DataFrame) -> np.ndarray:
    """
    Encode market data to HSV color space
    
    Channels:
    - Hue: Relative price position in range (cyclical, 0-1)
    - Saturation: Support/Resistance strength (0-1)  
    - Value: Normalized volume (0-1)
    
    Returns: (H, W, 3) numpy array
    """
    prices = data['close'].values
    volumes = data['volume'].values
    
    # Normalize price to 0-1 range
    price_min = prices.min()
    price_max = prices.max()
    hue = (prices - price_min) / (price_max - price_min + 1e-8)
    
    # Calculate S/R strength using local extrema
    saturation = self._calculate_sr_strength(prices)
    
    # Z-score normalize volume, then sigmoid to 0-1
    vol_zscore = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
    value = 1 / (1 + np.exp(-vol_zscore))
    
    # Stack into image
    hsv_image = np.stack([hue, saturation, value], axis=-1)
    return self._resize_image(hsv_image)
```

2. **Gramian Angular Field (GAF)** encoding for additional channels:
```python
def encode_gasf(self, series: np.ndarray) -> np.ndarray:
    """Gramian Angular Summation Field"""
    # Normalize to [-1, 1]
    normalized = 2 * (series - series.min()) / (series.max() - series.min() + 1e-8) - 1
    # Compute angle
    phi = np.arccos(np.clip(normalized, -1, 1))
    # Compute GASF
    gasf = np.cos(phi[:, None] + phi[None, :])
    return gasf
```

Generate data/replay_buffer.py:
- Prioritized Experience Replay (PER) with Sum Tree
- Support for multi-step returns
- Efficient sampling with numpy indexing
- Thread-safe for async training

=============================================================================
MODULE 3: NEURAL NETWORK (models/)
=============================================================================

Generate models/backbone.py with ResNet-style architecture:
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class TradingBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.backbone_channels
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = 2 if i > 0 else 1
            self.layers.append(ResidualBlock(channels[i], channels[i+1], stride))
            self.layers.append(ResidualBlock(channels[i+1], channels[i+1]))
        
        if config.use_attention:
            self.attention = nn.MultiheadAttention(channels[-1], num_heads=8, batch_first=True)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.output_dim = channels[-1] * 16
```

Generate models/trading_network.py combining policy + value heads:
```python
class TradingNetwork(nn.Module):
    """
    Dual-head network for MCTS trading
    - Policy head: Action probabilities
    - Value head: State value estimation
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = TradingBackbone(config)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, config.policy_hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(config.policy_hidden_dims[0]) if config.use_layer_norm else nn.Identity(),
            nn.Dropout(config.dropout),
            nn.Linear(config.policy_hidden_dims[0], config.policy_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.policy_hidden_dims[1], config.num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, config.value_hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(config.value_hidden_dims[0]) if config.use_layer_norm else nn.Identity(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_hidden_dims[0], config.value_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.value_hidden_dims[1], 1),
            nn.Tanh()  # Value in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return F.softmax(policy_logits, dim=-1), value
```

=============================================================================
MODULE 4: MCTS IMPLEMENTATION (mcts/)
=============================================================================

Generate mcts/node.py:
```python
@dataclass
class MCTSNode:
    """Node in the MCTS tree"""
    state: np.ndarray
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    prior: float = 1.0
    
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    
    @property
    def q_value(self) -> float:
        """Mean action value"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
    
    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0
    
    def ucb_score(self, config: MCTSConfig) -> float:
        """PUCT score for node selection"""
        if self.visit_count == 0:
            return float('inf')
        
        # Progressive bias based on parent visits
        pb_c = math.log((self.parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
        pb_c *= math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        prior_score = pb_c * self.prior
        value_score = self.q_value
        
        return value_score + prior_score
```

Generate mcts/search.py with PUCT algorithm:
```python
class MCTSSearch:
    """
    Monte Carlo Tree Search with neural network guidance
    Based on AlphaZero's PUCT algorithm
    """
    def __init__(self, config: MCTSConfig, network: TradingNetwork, device: str):
        self.config = config
        self.network = network
        self.device = device
        self.network.eval()
    
    def search(self, root_state: np.ndarray) -> Dict[int, float]:
        """
        Run MCTS simulations and return action probabilities
        """
        root = MCTSNode(state=root_state)
        
        # Add Dirichlet noise for exploration at root
        self._expand_node(root)
        self._add_exploration_noise(root)
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree to leaf
            while node.is_expanded:
                action, node = self._select_child(node)
                search_path.append(node)
            
            # Expansion and evaluation
            value = self._expand_node(node)
            
            # Backpropagation
            self._backpropagate(search_path, value)
        
        # Return visit count distribution
        return self._get_action_probs(root)
    
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = child.ucb_score(self.config)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    @torch.no_grad()
    def _expand_node(self, node: MCTSNode) -> float:
        """Expand node using neural network"""
        state_tensor = torch.from_numpy(node.state).float().unsqueeze(0).to(self.device)
        
        if node.state.ndim == 3:
            state_tensor = state_tensor.permute(0, 3, 1, 2)
        
        policy, value = self.network(state_tensor)
        policy = policy.cpu().numpy()[0]
        value = value.cpu().item()
        
        # Create children for all legal actions
        for action in range(len(policy)):
            if action not in node.children:
                child = MCTSNode(
                    state=node.state,  # State updated on actual move
                    parent=node,
                    action=action,
                    prior=policy[action]
                )
                node.children[action] = child
        
        return value
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """Update statistics along search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value *= -1  # Flip value for opponent (in adversarial setting)
            # For single-player: value *= self.config.discount
```

Generate mcts/parallel_mcts.py for batched inference:
```python
class ParallelMCTS:
    """
    Parallelized MCTS with batched neural network inference
    Key optimization for GPU utilization
    """
    def __init__(self, config: MCTSConfig, network: TradingNetwork, device: str, num_parallel: int = 8):
        self.config = config
        self.network = network
        self.device = device
        self.num_parallel = num_parallel
    
    def batch_search(self, states: List[np.ndarray]) -> List[Dict[int, float]]:
        """Run parallel MCTS on batch of states"""
        roots = [MCTSNode(state=s) for s in states]
        
        # Initial expansion with batched inference
        self._batch_expand(roots)
        
        for _ in range(self.config.num_simulations):
            # Collect leaves from all trees
            leaves_and_paths = []
            for root in roots:
                node = root
                path = [node]
                while node.is_expanded:
                    _, node = self._select_child(node)
                    path.append(node)
                leaves_and_paths.append((node, path))
            
            # Batch expand all leaves
            leaves = [lp[0] for lp in leaves_and_paths]
            values = self._batch_expand(leaves)
            
            # Backpropagate each tree
            for (node, path), value in zip(leaves_and_paths, values):
                self._backpropagate(path, value)
        
        return [self._get_action_probs(root) for root in roots]
    
    @torch.no_grad()
    def _batch_expand(self, nodes: List[MCTSNode]) -> List[float]:
        """Expand multiple nodes with single batched inference"""
        states = np.stack([n.state for n in nodes])
        states_tensor = torch.from_numpy(states).float().to(self.device)
        
        if states_tensor.ndim == 4:
            states_tensor = states_tensor.permute(0, 3, 1, 2)
        
        policies, values = self.network(states_tensor)
        policies = policies.cpu().numpy()
        values = values.cpu().numpy().flatten()
        
        for node, policy in zip(nodes, policies):
            for action in range(len(policy)):
                if action not in node.children:
                    child = MCTSNode(
                        state=node.state,
                        parent=node,
                        action=action,
                        prior=policy[action]
                    )
                    node.children[action] = child
        
        return values.tolist()
```

=============================================================================
MODULE 5: TRADING ENVIRONMENT (environment/)
=============================================================================

Generate environment/trading_env.py (Gymnasium compatible):
```python
class TradingEnvironment(gym.Env):
    """
    Trading environment for MCTS self-play
    
    State: Encoded market image (H, W, C)
    Action: Discrete - [STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY]
    Reward: Log return * position - transaction costs
    """
    def __init__(self, data: pd.DataFrame, config: TradingConfig, encoder: StateEncoder):
        super().__init__()
        self.data = data
        self.config = config
        self.encoder = encoder
        
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(config.image_height, config.image_width, 3),
            dtype=np.float32
        )
        
        # Position mapping: action -> target position fraction
        self.action_to_position = {
            0: -1.0,   # STRONG_SELL
            1: -0.5,   # SELL
            2: 0.0,    # HOLD
            3: 0.5,    # BUY
            4: 1.0     # STRONG_BUY
        }
        
        self.reset()
    
    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = self.encoder.lookback_window
        self.position = 0.0
        self.portfolio_value = self.config.initial_balance
        self.trade_history = []
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        target_position = self.action_to_position[action]
        position_change = target_position - self.position
        
        # Get current and next price
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Calculate returns
        price_return = (next_price - current_price) / current_price
        
        # Apply slippage and transaction costs
        transaction_cost = abs(position_change) * self.config.transaction_cost
        slippage_cost = abs(position_change) * self.config.slippage
        
        # Calculate reward (log return)
        gross_return = self.position * price_return
        net_return = gross_return - transaction_cost - slippage_cost
        reward = np.log(1 + net_return)
        
        # Update portfolio
        self.portfolio_value *= (1 + net_return)
        self.position = target_position
        
        # Record trade
        if position_change != 0:
            self.trade_history.append({
                'step': self.current_step,
                'action': action,
                'position_change': position_change,
                'price': current_price,
                'portfolio_value': self.portfolio_value
            })
        
        # Advance time
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= len(self.data) - 1
        truncated = self.portfolio_value <= self.config.initial_balance * 0.5  # 50% drawdown
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'price': next_price
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step - self.encoder.lookback_window
        window_data = self.data.iloc[start_idx:self.current_step]
        return self.encoder.encode(window_data)
```

=============================================================================
MODULE 6: TRAINING LOOP (training/)
=============================================================================

Generate training/trainer.py:
```python
class MCTSTrainer:
    """
    Main training loop for MCTS trading agent
    
    Training cycle:
    1. Self-play: Generate games using MCTS
    2. Store experiences in replay buffer
    3. Sample batches and update network
    4. Evaluate on validation set
    """
    def __init__(
        self,
        config: TrainingConfig,
        network: TradingNetwork,
        env: TradingEnvironment,
        device: str = "cuda:0"
    ):
        self.config = config
        self.network = network.to(device)
        self.env = env
        self.device = device
        
        # Initialize components
        self.mcts = ParallelMCTS(config.mcts, network, device)
        self.replay_buffer = PrioritizedReplayBuffer(config.replay_buffer)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Metrics
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'episode_rewards': [],
            'portfolio_values': []
        }
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.config.num_epochs):
            # Self-play phase
            experiences = self._self_play_epoch()
            self.replay_buffer.add_batch(experiences)
            
            # Training phase
            if len(self.replay_buffer) >= self.config.batch_size:
                losses = self._training_step()
                self.metrics['policy_loss'].append(losses['policy'])
                self.metrics['value_loss'].append(losses['value'])
            
            # Evaluation
            if epoch % self.config.eval_interval == 0:
                eval_metrics = self._evaluate()
                self._log_progress(epoch, eval_metrics)
            
            # Checkpointing
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
            
            self.scheduler.step()
    
    def _self_play_epoch(self) -> List[Dict]:
        """Generate self-play games"""
        self.network.eval()
        all_experiences = []
        
        for _ in range(self.config.episodes_per_epoch):
            state, _ = self.env.reset()
            episode_experiences = []
            done = False
            
            while not done:
                # MCTS search
                action_probs = self.mcts.search(state)
                
                # Sample action (with temperature)
                actions = list(action_probs.keys())
                probs = np.array([action_probs[a] for a in actions])
                action = np.random.choice(actions, p=probs)
                
                # Step environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                episode_experiences.append({
                    'state': state,
                    'action_probs': action_probs,
                    'reward': reward,
                    'done': done
                })
                
                state = next_state
            
            # Process episode with returns
            processed = self._process_episode(episode_experiences)
            all_experiences.extend(processed)
            
            self.metrics['episode_rewards'].append(sum(e['reward'] for e in episode_experiences))
            self.metrics['portfolio_values'].append(info['portfolio_value'])
        
        return all_experiences
    
    def _training_step(self) -> Dict[str, float]:
        """Single training step on replay buffer"""
        self.network.train()
        
        batch, indices, weights = self.replay_buffer.sample(self.config.batch_size)
        
        states = torch.from_numpy(np.stack([e['state'] for e in batch])).float().to(self.device)
        states = states.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        target_policies = torch.from_numpy(np.stack([e['policy'] for e in batch])).float().to(self.device)
        target_values = torch.from_numpy(np.array([e['value'] for e in batch])).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            pred_policy, pred_value = self.network(states)
            
            # Policy loss (cross-entropy)
            policy_loss = -(target_policies * torch.log(pred_policy + 1e-8)).sum(dim=1)
            policy_loss = (policy_loss * weights).mean()
            
            # Value loss (MSE)
            value_loss = F.mse_loss(pred_value.squeeze(), target_values, reduction='none')
            value_loss = (value_loss * weights).mean()
            
            # Total loss
            total_loss = (
                self.config.policy_loss_weight * policy_loss +
                self.config.value_loss_weight * value_loss
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        # Update priorities
        td_errors = torch.abs(pred_value.squeeze() - target_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        return {
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'total': total_loss.item()
        }
```

=============================================================================
MODULE 7: EVALUATION & BACKTESTING (evaluation/)
=============================================================================

Generate evaluation/backtester.py:
```python
class Backtester:
    """Historical backtesting engine"""
    
    def __init__(self, network: TradingNetwork, config: TradingConfig, device: str):
        self.network = network.to(device)
        self.config = config
        self.device = device
        self.mcts = MCTSSearch(config.mcts, network, device)
    
    def backtest(self, data: pd.DataFrame, encoder: StateEncoder, use_mcts: bool = True) -> Dict:
        """Run backtest on historical data"""
        self.network.eval()
        
        portfolio = Portfolio(self.config.initial_balance)
        metrics = TradingMetrics()
        
        for i in range(encoder.lookback_window, len(data) - 1):
            # Get state
            window = data.iloc[i - encoder.lookback_window:i]
            state = encoder.encode(window)
            
            # Get action
            if use_mcts:
                action_probs = self.mcts.search(state)
                action = max(action_probs, key=action_probs.get)
            else:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                state_tensor = state_tensor.permute(0, 3, 1, 2)
                with torch.no_grad():
                    policy, _ = self.network(state_tensor)
                action = policy.argmax().item()
            
            # Execute trade
            current_price = data.iloc[i]['close']
            next_price = data.iloc[i + 1]['close']
            
            position = self._action_to_position(action)
            portfolio.update(position, current_price, next_price)
            metrics.record(portfolio.value, position, current_price)
        
        return {
            'total_return': portfolio.total_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'num_trades': metrics.num_trades,
            'equity_curve': metrics.equity_curve
        }
```

Generate evaluation/metrics.py with all standard trading metrics:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Average Drawdown
- Win Rate, Profit Factor
- Value at Risk (VaR), Expected Shortfall
- Information Ratio (vs benchmark)

=============================================================================
MODULE 8: LIVE TRADING (live/)
=============================================================================

Generate live/paper_trader.py:
```python
class PaperTrader:
    """
    Paper trading with real-time Kraken data
    No actual orders placed
    """
    def __init__(
        self,
        network: TradingNetwork,
        config: TradingConfig,
        kraken_config: KrakenConfig,
        device: str
    ):
        self.network = network.to(device)
        self.config = config
        self.device = device
        self.mcts = MCTSSearch(config.mcts, network, device)
        
        self.kraken = KrakenClient(kraken_config)
        self.encoder = StateEncoder(config.state_encoding)
        self.portfolio = Portfolio(config.initial_balance)
        
        self.running = False
        self.data_buffer = deque(maxlen=config.lookback_window + 100)
    
    async def run(self, symbol: str, interval: str = "5m"):
        """Start paper trading loop"""
        self.running = True
        
        # Initialize with historical data
        historical = await self.kraken.get_ohlc(symbol, interval, limit=self.config.lookback_window)
        self.data_buffer.extend(historical.to_dict('records'))
        
        # Subscribe to websocket
        async for candle in self.kraken.subscribe_ohlc(symbol, interval):
            if not self.running:
                break
            
            self.data_buffer.append(candle)
            
            if len(self.data_buffer) >= self.config.lookback_window:
                await self._process_candle()
    
    async def _process_candle(self):
        """Process new candle and make trading decision"""
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.data_buffer))
        
        # Encode state
        window = df.tail(self.config.lookback_window)
        state = self.encoder.encode(window)
        
        # MCTS search
        self.network.eval()
        action_probs = self.mcts.search(state)
        action = max(action_probs, key=action_probs.get)
        
        # Update paper portfolio
        current_price = df.iloc[-1]['close']
        position = self._action_to_position(action)
        
        old_position = self.portfolio.position
        self.portfolio.update_position(position, current_price)
        
        # Log trade
        if position != old_position:
            logger.info(f"Paper trade: {old_position} -> {position} @ {current_price}")
            logger.info(f"Portfolio value: {self.portfolio.value:.2f}")
```

=============================================================================
MODULE 9: SCRIPTS & ENTRY POINTS (scripts/)
=============================================================================

Generate scripts/train.py:
```python
"""Training entry point"""
import argparse
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser(description="Train MCTS Trading Agent")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(config)
    
    # Load data
    data = load_training_data(config['data'])
    
    # Create components
    encoder = StateEncoder(StateEncodingConfig(**config['state_encoding']))
    network = TradingNetwork(ModelConfig(**config['model']))
    env = TradingEnvironment(data['train'], TradingConfig(**config['trading']), encoder)
    
    # Create trainer
    trainer = MCTSTrainer(
        config=TrainingConfig(**config['training']),
        network=network,
        env=env,
        device=args.device
    )
    
    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

Generate scripts/download_data.py:
```python
"""Download historical data from Kraken"""
import argparse
import asyncio
from datetime import datetime, timedelta

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="XXBTZUSD")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="data/kraken")
    args = parser.parse_args()
    
    fetcher = KrakenDataFetcher()
    
    # Download all trades (unlimited history)
    trades = await fetcher.fetch_all_trades(
        pair=args.symbol,
        since=datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    )
    
    # Resample to OHLCV at multiple timeframes
    for timeframe in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        ohlcv = fetcher.resample_trades(trades, timeframe)
        ohlcv.to_parquet(f"{args.output}/{args.symbol}_{timeframe}.parquet")
        print(f"Saved {len(ohlcv)} candles for {timeframe}")

if __name__ == "__main__":
    asyncio.run(main())
```

=============================================================================
ADDITIONAL REQUIREMENTS
=============================================================================

Generate requirements.txt:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
gymnasium>=0.29.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
aiohttp>=3.8.0
websockets>=11.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
tensorboard>=2.14.0
tqdm>=4.65.0
python-dotenv>=1.0.0
pytest>=7.4.0
krakenex>=2.1.0
```

Generate README.md with:
- Project overview
- Installation instructions
- Quick start guide
- Configuration options
- Training tips
- Backtesting guide
- Paper trading setup
- Architecture diagram

=============================================================================
CRITICAL OPTIMIZATIONS FOR RTX 3090
=============================================================================

1. **Batch Size**: Start with 128, can increase to 256 if memory allows
2. **Mixed Precision**: Always use FP16 for forward/backward passes
3. **Gradient Accumulation**: If batch size limited by memory
4. **CUDA Streams**: Use multiple streams for overlapping data transfer
5. **Pin Memory**: Enable for DataLoader
6. **Non-blocking Transfers**: Use `.to(device, non_blocking=True)`
7. **Compile Model**: Use `torch.compile()` for PyTorch 2.0+
8. **Efficient Attention**: Use Flash Attention if available

SCALABILITY HOOKS (for future 4x GPU):
```python
# In distributed_trainer.py
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Wrap model
model = DDP(model, device_ids=[rank])
```

=============================================================================
OUTPUT FORMAT
=============================================================================

Generate ALL files with complete, runnable code. Each file should:
1. Have comprehensive docstrings
2. Include type hints
3. Have error handling
4. Be production-ready
5. Include unit tests where appropriate

Start with the core modules and build up to the complete system.
```

---

## PROMPT END

## Usage Notes

1. **For Gemini CLI**: Copy everything between `## MASTER PROMPT START` and `## PROMPT END`

2. **Iterative Generation**: If output is too long, ask Gemini to generate modules sequentially:
   - First: `config/` and `data/` modules
   - Second: `models/` and `mcts/` modules
   - Third: `environment/` and `training/` modules
   - Fourth: `evaluation/` and `live/` modules

3. **Testing**: After generation, test each module independently before integration

4. **Data**: Download Kraken historical data first using the scripts before training
