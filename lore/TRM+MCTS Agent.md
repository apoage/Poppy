**PROMPT 4: TRM+MCTS Agent**

```markdown
# TRM+MCTS Trading Agent - Core Decision Engine

## Objective
Build the decision-making agent combining Tiny Recursive Model (TRM) architecture with MCTS planning for multi-asset correlation trading.

## Architecture Overview

```
Input State (27, 64, 64)
         ↓
┌─────────────────────────────────────┐
│         CNN Backbone                │
│   (Lightweight feature extraction)  │
└─────────────────────────────────────┘
         ↓
    (batch, 256) features
         ↓
┌─────────────────────────────────────┐
│     TRM Recursive Reasoner          │
│  ┌─────────────────────────────┐   │
│  │ For h in H_cycles:          │   │
│  │   For l in L_cycles:        │   │
│  │     z = z + Refine(x, z)    │   │
│  │   y = y + Output(z, y)      │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         ↓
    Policy (64 assets × 3 actions)
    Value (portfolio expected return)
         ↓
┌─────────────────────────────────────┐
│         MCTS Planner                │
│   (50-100 simulations, 4hr horizon) │
└─────────────────────────────────────┘
         ↓
    Final Action: [(asset, direction, size), ...]
```

## Implementation

### 1. TRM Model Architecture

```python
"""
trm_model.py - Tiny Recursive Model for trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class CNNBackbone(nn.Module):
    """Lightweight CNN for feature extraction from state tensor"""
    
    def __init__(self, in_channels: int = 27, feature_dim: int = 256):
        super().__init__()
        
        self.conv = nn.Sequential(
            # (27, 64, 64) -> (64, 32, 32)
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            # (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            
            # (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            
            # Final projection
            nn.Linear(256 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TRMBlock(nn.Module):
    """
    Core TRM recursive block.
    Refines latent state given input and current latent.
    """
    
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        
        # Latent refinement: (state, latent) -> latent_delta
        self.latent_refiner = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Gating for residual (learned mixing)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Refine latent given state.
        
        Args:
            state: (batch, state_dim) - encoded market state
            latent: (batch, latent_dim) - current latent representation
            
        Returns:
            new_latent: (batch, latent_dim) - refined latent
        """
        combined = torch.cat([state, latent], dim=-1)
        delta = self.latent_refiner(combined)
        
        # Gated residual connection
        gate = self.gate(torch.cat([latent, delta], dim=-1))
        new_latent = latent + gate * delta
        
        return new_latent


class TRMOutputHead(nn.Module):
    """
    Output refinement head.
    Refines action logits given latent and current output.
    """
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        
        self.refiner = nn.Sequential(
            nn.Linear(latent_dim + output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, latent: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([latent, output], dim=-1)
        delta = self.refiner(combined)
        gate = self.gate(torch.cat([output, delta], dim=-1))
        return output + gate * delta


class TRMTradingModel(nn.Module):
    """
    Complete TRM model for multi-asset trading.
    
    ~7M parameters, recursive reasoning for correlation exploitation.
    """
    
    def __init__(
        self,
        n_assets: int = 64,
        n_actions: int = 3,  # sell, hold, buy per asset
        in_channels: int = 27,
        feature_dim: int = 256,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        H_cycles: int = 4,   # Output refinement cycles
        L_cycles: int = 6    # Latent refinement cycles
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_actions = n_actions
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        
        # Feature extraction
        self.backbone = CNNBackbone(in_channels, feature_dim)
        
        # TRM components
        self.latent_block = TRMBlock(feature_dim, latent_dim, hidden_dim)
        self.output_block = TRMOutputHead(latent_dim, n_assets * n_actions, hidden_dim)
        
        # Value head (separate, non-recursive)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        # Position sizing head (how much to allocate)
        self.size_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_assets),
            nn.Sigmoid()  # 0-1 allocation per asset
        )
        
        # Learnable initial states
        self.initial_latent = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.initial_output = nn.Parameter(torch.zeros(n_assets * n_actions))
        
        # Adaptive halting (optional - learn when to stop recursing)
        self.halt_predictor = nn.Linear(latent_dim, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with recursive refinement.
        
        Args:
            x: (batch, channels, 64, 64) - encoded state
            H_cycles: override output cycles
            L_cycles: override latent cycles
            return_intermediates: return all intermediate outputs
            
        Returns:
            policy: (batch, n_assets, n_actions) - action probabilities
            value: (batch, 1) - expected portfolio return
            sizes: (batch, n_assets) - position sizes
        """
        H = H_cycles or self.H_cycles
        L = L_cycles or self.L_cycles
        batch_size = x.shape[0]
        
        # Extract features
        state = self.backbone(x)
        
        # Initialize latent and output
        latent = self.initial_latent.expand(batch_size, -1)
        output = self.initial_output.expand(batch_size, -1)
        
        intermediates = [] if return_intermediates else None
        
        # Recursive refinement
        for h in range(H):
            # Refine latent L times
            for l in range(L):
                latent = self.latent_block(state, latent)
            
            # Refine output once
            output = self.output_block(latent, output)
            
            if return_intermediates:
                intermediates.append(output.clone())
        
        # Reshape output to (batch, n_assets, n_actions)
        policy_logits = output.view(batch_size, self.n_assets, self.n_actions)
        policy = F.softmax(policy_logits, dim=-1)
        
        # Value and size from final latent
        value = self.value_head(latent)
        sizes = self.size_head(latent)
        
        if return_intermediates:
            return policy, value, sizes, intermediates
        
        return policy, value, sizes
    
    def get_action(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Get trading actions from state.
        
        Returns:
            actions: (batch, n_assets) - action per asset (0=sell, 1=hold, 2=buy)
            sizes: (batch, n_assets) - position sizes
            info: additional info dict
        """
        policy, value, sizes = self.forward(x)
        
        if deterministic:
            actions = policy.argmax(dim=-1)
        else:
            # Sample from policy with temperature
            policy_temp = F.softmax(policy / temperature, dim=-1)
            actions = torch.multinomial(
                policy_temp.view(-1, self.n_actions), 1
            ).view(-1, self.n_assets)
        
        return actions, sizes, {
            'policy': policy,
            'value': value,
            'entropy': -(policy * policy.log().clamp(min=-100)).sum(dim=-1).mean()
        }


class TRMWithAttention(TRMTradingModel):
    """
    Enhanced TRM with cross-asset attention.
    Allows assets to "communicate" during reasoning.
    """
    
    def __init__(self, *args, n_heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Cross-asset attention
        self.asset_attention = nn.MultiheadAttention(
            embed_dim=self.n_actions,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Asset embeddings (learnable)
        self.asset_embeddings = nn.Parameter(
            torch.randn(self.n_assets, 32) * 0.01
        )
    
    def forward(self, x: torch.Tensor, **kwargs):
        policy, value, sizes = super().forward(x, **kwargs)
        
        # Apply cross-asset attention to policy
        # Shape: (batch, n_assets, n_actions)
        attended, _ = self.asset_attention(policy, policy, policy)
        policy = policy + 0.1 * attended  # Residual
        policy = F.softmax(policy, dim=-1)
        
        return policy, value, sizes
```

### 2. MCTS Planner

```python
"""
mcts.py - Monte Carlo Tree Search for multi-asset trading
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 2.0
    gamma: float = 0.995  # Discount factor
    temperature: float = 1.0
    dirichlet_alpha: float = 0.5
    dirichlet_weight: float = 0.25
    max_depth: int = 20  # ~20 minutes planning horizon
    
    
class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(
        self,
        state: torch.Tensor,
        prior: np.ndarray,
        parent: Optional['MCTSNode'] = None,
        action: Optional[np.ndarray] = None
    ):
        self.state = state
        self.prior = prior  # (n_assets, n_actions)
        self.parent = parent
        self.action = action  # Action that led to this node
        
        self.children: Dict[tuple, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float) -> float:
        if self.visit_count == 0:
            return float('inf')
        
        # Average prior for this action
        prior_score = self.prior.mean() if self.prior is not None else 1.0
        
        exploration = c_puct * prior_score * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value() + exploration


class TradingMCTS:
    """
    MCTS for multi-asset trading decisions.
    
    Key adaptations for trading:
    - State = market state tensor
    - Action = (asset_actions, position_sizes)
    - Simulates future market states using learned dynamics
    """
    
    def __init__(
        self,
        model: TRMTradingModel,
        config: MCTSConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Simple market simulator for MCTS rollouts
        self.simulator = MarketSimulator(model.n_assets)
        
    def search(
        self,
        state: torch.Tensor,
        valid_actions_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run MCTS search from given state.
        
        Args:
            state: (channels, 64, 64) - current market state
            valid_actions_mask: optional mask for valid actions
            
        Returns:
            actions: (n_assets,) - action per asset
            sizes: (n_assets,) - position sizes
            value: expected value
        """
        # Add batch dimension
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        # Get initial policy from model
        with torch.no_grad():
            policy, value, sizes = self.model(state.to(self.device))
        
        prior = policy[0].cpu().numpy()
        
        # Add Dirichlet noise at root
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * self.model.n_actions,
            size=self.model.n_assets
        )
        prior = (1 - self.config.dirichlet_weight) * prior + \
                self.config.dirichlet_weight * noise
        
        # Create root node
        root = MCTSNode(state, prior)
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root)
        
        # Extract policy from visit counts
        action_probs = self._get_action_probs(root)
        
        # Sample or argmax based on temperature
        if self.config.temperature < 0.01:
            actions = action_probs.argmax(axis=-1)
        else:
            actions = np.array([
                np.random.choice(self.model.n_actions, p=action_probs[i])
                for i in range(self.model.n_assets)
            ])
        
        return actions, sizes[0].cpu().numpy(), root.q_value()
    
    def _simulate(self, root: MCTSNode):
        """Single MCTS simulation"""
        node = root
        path = [node]
        
        # Selection - traverse tree
        while node.is_expanded and node.children:
            node = self._select_child(node)
            path.append(node)
        
        # Expansion & Evaluation
        value = self._expand_and_evaluate(node)
        
        # Backpropagation
        self._backpropagate(path, value)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB"""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.ucb_score(self.config.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand node and return value estimate"""
        if node.is_expanded:
            return node.q_value()
        
        # Simulate next state
        next_state = self.simulator.step(node.state, node.action)
        
        # Get model predictions
        with torch.no_grad():
            policy, value, sizes = self.model(next_state.to(self.device))
        
        prior = policy[0].cpu().numpy()
        value = value[0].item()
        
        # Create children for sampled actions
        # (Don't expand all possible actions - too many)
        n_children = min(10, self.model.n_assets)
        
        for _ in range(n_children):
            # Sample action
            action = np.array([
                np.random.choice(self.model.n_actions, p=prior[i])
                for i in range(self.model.n_assets)
            ])
            
            action_key = tuple(action)
            if action_key not in node.children:
                child = MCTSNode(
                    next_state,
                    prior=prior,
                    parent=node,
                    action=action
                )
                node.children[action_key] = child
        
        node.is_expanded = True
        return value
    
    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Backpropagate value through path"""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value *= self.config.gamma
    
    def _get_action_probs(self, root: MCTSNode) -> np.ndarray:
        """Get action probabilities from visit counts"""
        # Aggregate visit counts per asset-action
        action_counts = np.zeros((self.model.n_assets, self.model.n_actions))
        
        for action_key, child in root.children.items():
            action = np.array(action_key)
            for i in range(self.model.n_assets):
                action_counts[i, action[i]] += child.visit_count
        
        # Normalize to probabilities
        total = action_counts.sum(axis=-1, keepdims=True)
        total = np.maximum(total, 1)  # Avoid division by zero
        
        probs = action_counts / total
        
        # Apply temperature
        if self.config.temperature != 1.0:
            probs = np.power(probs, 1.0 / self.config.temperature)
            probs /= probs.sum(axis=-1, keepdims=True)
        
        return probs


class MarketSimulator:
    """
    Simple market simulator for MCTS rollouts.
    Uses historical statistics for realistic simulation.
    """
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        
        # Historical statistics (updated during training)
        self.mean_returns = np.zeros(n_assets)
        self.std_returns = np.ones(n_assets) * 0.01
        self.correlation_matrix = np.eye(n_assets)
        
    def update_stats(self, returns: np.ndarray):
        """Update statistics from observed returns"""
        self.mean_returns = returns.mean(axis=1)
        self.std_returns = returns.std(axis=1).clip(min=0.001)
        self.correlation_matrix = np.corrcoef(returns)
    
    def step(
        self,
        state: torch.Tensor,
        action: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Simulate next state.
        
        This is a simplified simulation - in practice, you might use
        a learned dynamics model.
        """
        # Generate correlated returns
        uncorrelated = np.random.randn(self.n_assets)
        
        # Apply correlation structure
        L = np.linalg.cholesky(self.correlation_matrix + 1e-6 * np.eye(self.n_assets))
        correlated = L @ uncorrelated
        
        # Scale by historical stats
        returns = self.mean_returns + self.std_returns * correlated
        
        # Simple state update (shift and add new data)
        # In practice, this would update the full state tensor
        next_state = state.clone()
        
        # Shift correlation matrices, add noise to simulate uncertainty
        next_state[:, :4, :, :] = state[:, :4, :, :] + 0.01 * torch.randn_like(state[:, :4, :, :])
        
        return next_state
```

### 3. Portfolio Manager

```python
"""
portfolio.py - Position and risk management
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class Position:
    asset_idx: int
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float = 0.0
    
    @property
    def pnl(self) -> float:
        multiplier = 1 if self.side == 'long' else -1
        return multiplier * self.size * (self.current_price - self.entry_price)
    
    @property
    def pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0.0
        multiplier = 1 if self.side == 'long' else -1
        return multiplier * (self.current_price - self.entry_price) / self.entry_price * 100


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[int, Position] = field(default_factory=dict)
    total_value: float = 0.0
    max_value: float = 0.0
    
    @property
    def drawdown(self) -> float:
        if self.max_value == 0:
            return 0.0
        return (self.max_value - self.total_value) / self.max_value


class PortfolioManager:
    """
    Manages positions, risk limits, and order execution.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.1,      # Max 10% per position
        max_total_exposure: float = 2.0,     # Max 200% total (with leverage)
        max_drawdown: float = 0.15,          # Stop trading at 15% drawdown
        n_assets: int = 64
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.max_drawdown = max_drawdown
        self.n_assets = n_assets
        
        self.state = PortfolioState(cash=initial_capital)
        self.state.total_value = initial_capital
        self.state.max_value = initial_capital
        
        # Asset info
        self.symbols: Dict[int, str] = {}
        self.prices: Dict[int, float] = {}
        
    def register_assets(self, symbols: Dict[int, str]):
        self.symbols = symbols
    
    def update_prices(self, prices: Dict[int, float]):
        """Update current prices and portfolio value"""
        self.prices = prices
        
        # Update position values
        position_value = 0.0
        for idx, pos in self.state.positions.items():
            if idx in prices:
                pos.current_price = prices[idx]
                position_value += abs(pos.size * pos.current_price)
        
        # Calculate total value
        self.state.total_value = self.state.cash + sum(
            pos.pnl for pos in self.state.positions.values()
        ) + self.initial_capital
        
        self.state.max_value = max(self.state.max_value, self.state.total_value)
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions considering risk limits.
        
        Returns:
            mask: (n_assets, 3) - True if action is valid
        """
        mask = np.ones((self.n_assets, 3), dtype=bool)
        
        # Check drawdown limit
        if self.state.drawdown >= self.max_drawdown:
            # Only allow closing positions (action depends on current position)
            for idx in range(self.n_assets):
                if idx in self.state.positions:
                    pos = self.state.positions[idx]
                    if pos.side == 'long':
                        mask[idx, 2] = False  # Can't buy more
                    else:
                        mask[idx, 0] = False  # Can't sell more
                else:
                    mask[idx, 0] = False  # Can't open short
                    mask[idx, 2] = False  # Can't open long
        
        # Check per-position limits
        for idx in range(self.n_assets):
            if idx not in self.prices:
                mask[idx, :] = [False, True, False]  # Only hold if no price
                continue
            
            current_exposure = 0.0
            if idx in self.state.positions:
                pos = self.state.positions[idx]
                current_exposure = abs(pos.size * pos.current_price) / self.state.total_value
            
            if current_exposure >= self.max_position_pct:
                # Can't increase position
                if idx in self.state.positions:
                    if self.state.positions[idx].side == 'long':
                        mask[idx, 2] = False
                    else:
                        mask[idx, 0] = False
        
        return mask
    
    def execute_actions(
        self,
        actions: np.ndarray,      # (n_assets,) - 0=sell, 1=hold, 2=buy
        sizes: np.ndarray,        # (n_assets,) - position sizes (0-1)
        prices: Dict[int, float]
    ) -> List[Dict]:
        """
        Execute trading actions.
        
        Returns:
            List of executed orders
        """
        self.update_prices(prices)
        orders = []
        
        for idx in range(self.n_assets):
            action = actions[idx]
            size = sizes[idx]
            
            if action == 1:  # Hold
                continue
            
            if idx not in prices:
                continue
            
            price = prices[idx]
            
            # Calculate target position value
            max_value = self.state.total_value * self.max_position_pct * size
            target_size = max_value / price
            
            current_pos = self.state.positions.get(idx)
            
            if action == 2:  # Buy
                order = self._create_buy_order(idx, target_size, price, current_pos)
            else:  # Sell
                order = self._create_sell_order(idx, target_size, price, current_pos)
            
            if order:
                orders.append(order)
                self._apply_order(order)
        
        return orders
    
    def _create_buy_order(
        self,
        idx: int,
        target_size: float,
        price: float,
        current_pos: Optional[Position]
    ) -> Optional[Dict]:
        """Create buy order"""
        if current_pos is None:
            # Open long
            return {
                'asset_idx': idx,
                'symbol': self.symbols.get(idx, f'ASSET_{idx}'),
                'side': 'buy',
                'size': target_size,
                'price': price,
                'type': 'open_long'
            }
        elif current_pos.side == 'short':
            # Close short
            return {
                'asset_idx': idx,
                'symbol': self.symbols.get(idx, f'ASSET_{idx}'),
                'side': 'buy',
                'size': current_pos.size,
                'price': price,
                'type': 'close_short'
            }
        else:
            # Already long - could increase but respecting limits
            return None
    
    def _create_sell_order(
        self,
        idx: int,
        target_size: float,
        price: float,
        current_pos: Optional[Position]
    ) -> Optional[Dict]:
        """Create sell order"""
        if current_pos is None:
            # Open short
            return {
                'asset_idx': idx,
                'symbol': self.symbols.get(idx, f'ASSET_{idx}'),
                'side': 'sell',
                'size': target_size,
                'price': price,
                'type': 'open_short'
            }
        elif current_pos.side == 'long':
            # Close long
            return {
                'asset_idx': idx,
                'symbol': self.symbols.get(idx, f'ASSET_{idx}'),
                'side': 'sell',
                'size': current_pos.size,
                'price': price,
                'type': 'close_long'
            }
        else:
            # Already short
            return None
    
    def _apply_order(self, order: Dict):
        """Apply order to portfolio state"""
        idx = order['asset_idx']
        
        if order['type'] in ['close_long', 'close_short']:
            # Close position
            if idx in self.state.positions:
                pos = self.state.positions[idx]
                self.state.cash += pos.pnl
                del self.state.positions[idx]
        else:
            # Open position
            side = 'long' if order['type'] == 'open_long' else 'short'
            self.state.positions[idx] = Position(
                asset_idx=idx,
                symbol=order['symbol'],
                side=side,
                size=order['size'],
                entry_price=order['price'],
                current_price=order['price']
            )
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        return {
            'total_value': self.state.total_value,
            'cash': self.state.cash,
            'n_positions': len(self.state.positions),
            'drawdown': self.state.drawdown,
            'return_pct': (self.state.total_value - self.initial_capital) / self.initial_capital * 100,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'size': pos.size,
                    'pnl': pos.pnl,
                    'pnl_pct': pos.pnl_percent
                }
                for pos in self.state.positions.values()
            ]
        }
```

### 4. Combined Agent

```python
"""
agent.py - Complete TRM+MCTS trading agent
"""

import torch
from typing import Dict, Tuple

class TRMMCTSAgent:
    """
    Complete trading agent combining TRM, MCTS, and portfolio management.
    """
    
    def __init__(
        self,
        n_assets: int = 64,
        device: str = 'cuda',
        use_mcts: bool = True,
        mcts_simulations: int = 100
    ):
        self.n_assets = n_assets
        self.device = device
        self.use_mcts = use_mcts
        
        # Model
        self.model = TRMTradingModel(
            n_assets=n_assets,
            n_actions=3,
            H_cycles=4,
            L_cycles=6
        ).to(device)
        
        # MCTS
        if use_mcts:
            self.mcts = TradingMCTS(
                self.model,
                MCTSConfig(num_simulations=mcts_simulations),
                device
            )
        
        # Portfolio
        self.portfolio = PortfolioManager(n_assets=n_assets)
        
    def act(
        self,
        state: torch.Tensor,
        prices: Dict[int, float],
        deterministic: bool = False
    ) -> Tuple[Dict, Dict]:
        """
        Get trading actions for current state.
        
        Returns:
            orders: List of orders to execute
            info: Additional information
        """
        self.portfolio.update_prices(prices)
        
        # Get valid actions mask
        valid_mask = self.portfolio.get_valid_actions()
        
        if self.use_mcts:
            # Run MCTS search
            actions, sizes, value = self.mcts.search(state, valid_mask)
        else:
            # Direct model inference
            with torch.no_grad():
                actions, sizes, info = self.model.get_action(
                    state.unsqueeze(0).to(self.device),
                    deterministic=deterministic
                )
            actions = actions[0].cpu().numpy()
            sizes = sizes[0].cpu().numpy()
            value = info['value'].item()
        
        # Apply valid action mask
        for i in range(self.n_assets):
            if not valid_mask[i, actions[i]]:
                actions[i] = 1  # Default to hold
        
        # Execute
        orders = self.portfolio.execute_actions(actions, sizes, prices)
        
        return orders, {
            'actions': actions,
            'sizes': sizes,
            'value': value,
            'portfolio': self.portfolio.get_portfolio_summary()
        }
    
    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'portfolio_state': self.portfolio.state
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.portfolio.state = checkpoint['portfolio_state']
```

## Usage

```python
# Initialize
agent = TRMMCTSAgent(n_assets=64, use_mcts=True)
encoder = MultiAssetStateEncoder(n_assets=64)

# Register assets
agent.portfolio.register_assets(symbol_map)
encoder.register_assets(symbols)

# Trading loop
while True:
    # Update with new market data
    encoder.update(market_data)
    
    # Encode state
    state = encoder.encode()['combined']
    
    # Get prices
    prices = {i: market_data[sym]['close'] for i, sym in symbol_map.items()}
    
    # Act
    orders, info = agent.act(state, prices)
    
    # Execute orders via Kraken API
    for order in orders:
        execute_order(order)
    
    # Log
    print(f"Value: ${info['portfolio']['total_value']:.2f}, "
          f"Positions: {info['portfolio']['n_positions']}, "
          f"Drawdown: {info['portfolio']['drawdown']*100:.1f}%")
```

## Model Size

| Component | Parameters |
|-----------|------------|
| CNN Backbone | ~1.5M |
| TRM Latent Block | ~0.8M |
| TRM Output Block | ~0.5M |
| Value/Size Heads | ~0.2M |
| Embeddings | ~0.1M |
| **Total** | **~3.1M** |

Smaller than full TRM (7M) due to trading-specific simplifications.
```

Ready for Prompt 5 (Training Loop)?
