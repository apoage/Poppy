**PROMPT 5: Training Loop**

```markdown
# Training Loop - Self-Play and Reinforcement Learning

## Objective
Complete training infrastructure: environment, experience replay, self-play loop, and curriculum learning.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                        │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │ Environment │───▶│  Self-Play  │───▶│  Trainer   │  │
│  │ (Backtest)  │    │   (MCTS)    │    │  (Update)  │  │
│  └─────────────┘    └─────────────┘    └────────────┘  │
│         │                  │                  │         │
│         ▼                  ▼                  ▼         │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │   Replay    │◀───│ Experiences │───▶│ Checkpoint │  │
│  │   Buffer    │    │             │    │   Saver    │  │
│  └─────────────┘    └─────────────┘    └────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Trading Environment

```python
"""
environment.py - Backtesting environment for training
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

class RewardType(Enum):
    SIMPLE = "simple"           # Raw returns
    SHARPE = "sharpe"           # Differential Sharpe
    COMBINED = "combined"       # Full reward function

@dataclass
class EnvConfig:
    n_assets: int = 64
    lookback_window: int = 60       # Minutes of history
    episode_length: int = 240       # 4 hours per episode
    transaction_cost: float = 0.001 # 0.1% per trade
    slippage: float = 0.0005        # 0.05% slippage
    initial_capital: float = 10000.0
    reward_type: RewardType = RewardType.COMBINED
    max_position_pct: float = 0.1
    
class TradingEnvironment:
    """
    Backtesting environment for multi-asset trading.
    Uses historical data for realistic simulation.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],  # symbol -> OHLCV DataFrame
        encoder: 'MultiAssetStateEncoder',
        config: EnvConfig
    ):
        self.data = data
        self.encoder = encoder
        self.config = config
        
        self.symbols = list(data.keys())
        self.n_assets = min(len(self.symbols), config.n_assets)
        
        # Align timestamps across assets
        self._align_data()
        
        # Episode state
        self.current_step = 0
        self.episode_start = 0
        self.positions = np.zeros(self.n_assets)  # -1 to 1
        self.position_sizes = np.zeros(self.n_assets)
        self.cash = config.initial_capital
        self.portfolio_value = config.initial_capital
        
        # Tracking for rewards
        self.returns_history = []
        self.prev_portfolio_value = config.initial_capital
        self.peak_value = config.initial_capital
        
    def _align_data(self):
        """Align all asset data to common timestamps"""
        # Find common timestamps
        all_timestamps = None
        for symbol, df in self.data.items():
            if 'timestamp' in df.columns:
                ts = set(df['timestamp'])
            else:
                ts = set(df.index)
            
            if all_timestamps is None:
                all_timestamps = ts
            else:
                all_timestamps &= ts
        
        self.timestamps = sorted(list(all_timestamps))
        self.max_steps = len(self.timestamps) - self.config.lookback_window - 1
        
        # Create aligned price matrix
        self.prices = np.zeros((len(self.timestamps), self.n_assets))
        self.volumes = np.zeros((len(self.timestamps), self.n_assets))
        
        for i, symbol in enumerate(self.symbols[:self.n_assets]):
            df = self.data[symbol]
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            for t, ts in enumerate(self.timestamps):
                if ts in df.index:
                    self.prices[t, i] = df.loc[ts, 'close']
                    self.volumes[t, i] = df.loc[ts, 'volume']
    
    def reset(self, start_idx: Optional[int] = None) -> torch.Tensor:
        """Reset environment to start of episode"""
        if start_idx is None:
            max_start = self.max_steps - self.config.episode_length
            start_idx = np.random.randint(
                self.config.lookback_window,
                max(self.config.lookback_window + 1, max_start)
            )
        
        self.episode_start = start_idx
        self.current_step = start_idx
        
        # Reset portfolio
        self.positions = np.zeros(self.n_assets)
        self.position_sizes = np.zeros(self.n_assets)
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.prev_portfolio_value = self.config.initial_capital
        self.peak_value = self.config.initial_capital
        self.returns_history = []
        
        # Update encoder with current window
        self._update_encoder()
        
        return self._get_state()
    
    def step(
        self,
        actions: np.ndarray,    # (n_assets,) - 0=sell, 1=hold, 2=buy
        sizes: np.ndarray       # (n_assets,) - position sizes 0-1
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute actions and advance one step.
        
        Returns:
            state: next state tensor
            reward: step reward
            done: episode finished
            info: additional info
        """
        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        
        # Execute trades
        trades = self._execute_trades(actions, sizes, current_prices)
        
        # Calculate portfolio value after price change
        self._update_portfolio_value(next_prices)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Advance step
        self.current_step += 1
        self._update_encoder()
        
        # Check done
        steps_in_episode = self.current_step - self.episode_start
        done = (
            steps_in_episode >= self.config.episode_length or
            self.current_step >= self.max_steps or
            self.portfolio_value < self.config.initial_capital * 0.5  # Stop-loss
        )
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital,
            'drawdown': (self.peak_value - self.portfolio_value) / self.peak_value,
            'n_trades': len(trades),
            'trades': trades
        }
        
        return self._get_state(), reward, done, info
    
    def _execute_trades(
        self,
        actions: np.ndarray,
        sizes: np.ndarray,
        prices: np.ndarray
    ) -> List[Dict]:
        """Execute trades and update positions"""
        trades = []
        
        for i in range(self.n_assets):
            action = actions[i]
            size = sizes[i]
            price = prices[i]
            
            if price <= 0 or action == 1:  # Invalid price or hold
                continue
            
            # Map action to target position
            target_pos = [-1, 0, 1][action]  # -1=short, 0=flat, 1=long
            current_pos = self.positions[i]
            
            if target_pos == current_pos:
                continue
            
            # Calculate position change
            max_position_value = self.portfolio_value * self.config.max_position_pct * size
            target_size = max_position_value / price
            
            # Close existing position
            if current_pos != 0:
                close_value = self.position_sizes[i] * price
                pnl = current_pos * (price - self._get_entry_price(i)) * self.position_sizes[i]
                
                # Apply costs
                cost = close_value * (self.config.transaction_cost + self.config.slippage)
                self.cash += close_value + pnl - cost
                
                trades.append({
                    'asset': i,
                    'type': 'close',
                    'side': 'long' if current_pos > 0 else 'short',
                    'size': self.position_sizes[i],
                    'price': price,
                    'pnl': pnl - cost
                })
                
                self.positions[i] = 0
                self.position_sizes[i] = 0
            
            # Open new position
            if target_pos != 0:
                cost = target_size * price * (self.config.transaction_cost + self.config.slippage)
                
                if self.cash >= target_size * price + cost:
                    self.cash -= target_size * price + cost
                    self.positions[i] = target_pos
                    self.position_sizes[i] = target_size
                    self._set_entry_price(i, price)
                    
                    trades.append({
                        'asset': i,
                        'type': 'open',
                        'side': 'long' if target_pos > 0 else 'short',
                        'size': target_size,
                        'price': price
                    })
        
        return trades
    
    def _update_portfolio_value(self, prices: np.ndarray):
        """Update portfolio value with new prices"""
        self.prev_portfolio_value = self.portfolio_value
        
        position_value = 0.0
        for i in range(self.n_assets):
            if self.positions[i] != 0 and prices[i] > 0:
                entry = self._get_entry_price(i)
                pnl = self.positions[i] * (prices[i] - entry) * self.position_sizes[i]
                position_value += self.position_sizes[i] * entry + pnl
        
        self.portfolio_value = self.cash + position_value
        self.peak_value = max(self.peak_value, self.portfolio_value)
        
        # Track returns
        if self.prev_portfolio_value > 0:
            ret = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
            self.returns_history.append(ret)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on config"""
        if self.config.reward_type == RewardType.SIMPLE:
            return self._simple_reward()
        elif self.config.reward_type == RewardType.SHARPE:
            return self._sharpe_reward()
        else:
            return self._combined_reward()
    
    def _simple_reward(self) -> float:
        """Simple return-based reward"""
        if self.prev_portfolio_value == 0:
            return 0.0
        return (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
    
    def _sharpe_reward(self) -> float:
        """Differential Sharpe Ratio reward"""
        if len(self.returns_history) < 2:
            return self._simple_reward()
        
        returns = np.array(self.returns_history[-20:])  # Last 20 steps
        mean_ret = returns.mean()
        std_ret = returns.std() + 1e-8
        
        # Differential Sharpe (reward for improving Sharpe)
        current_sharpe = mean_ret / std_ret
        
        if len(self.returns_history) > 20:
            prev_returns = np.array(self.returns_history[-21:-1])
            prev_sharpe = prev_returns.mean() / (prev_returns.std() + 1e-8)
            return current_sharpe - prev_sharpe
        
        return current_sharpe * 0.1
    
    def _combined_reward(self) -> float:
        """Full combined reward function"""
        raw_return = self._simple_reward()
        
        # Differential Sharpe
        dsr = 0.0
        if len(self.returns_history) >= 10:
            returns = np.array(self.returns_history[-10:])
            dsr = returns.mean() / (returns.std() + 1e-8) * 0.01
        
        # Drawdown penalty
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        dd_penalty = drawdown ** 2
        
        # Regime scaling (higher reward in volatile markets)
        if len(self.returns_history) >= 5:
            volatility = np.std(self.returns_history[-5:])
            regime_scale = 1.0 + min(volatility * 10, 1.0)
        else:
            regime_scale = 1.0
        
        reward = regime_scale * (
            0.5 * raw_return +
            0.3 * dsr +
            0.2 * (-dd_penalty)
        )
        
        return reward
    
    def _get_state(self) -> torch.Tensor:
        """Get current state tensor"""
        return torch.FloatTensor(self.encoder.encode()['combined'])
    
    def _update_encoder(self):
        """Update encoder with current data window"""
        window_start = self.current_step - self.config.lookback_window
        window_end = self.current_step
        
        for i, symbol in enumerate(self.symbols[:self.n_assets]):
            df = self.data[symbol]
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            window_ts = self.timestamps[window_start:window_end]
            window_data = df.loc[df.index.isin(window_ts)]
            
            self.encoder.update_asset(i, window_data)
    
    # Entry price tracking
    _entry_prices: Dict[int, float] = {}
    
    def _get_entry_price(self, asset_idx: int) -> float:
        return self._entry_prices.get(asset_idx, 0.0)
    
    def _set_entry_price(self, asset_idx: int, price: float):
        self._entry_prices[asset_idx] = price
```

### 2. Experience Replay

```python
"""
replay_buffer.py - Prioritized experience replay
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

class SumTree:
    """Sum tree for O(log n) prioritized sampling"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0
    
    def total(self) -> float:
        return self.tree[0]
    
    def add(self, priority: float, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
    
    def get(self, s: float) -> Tuple[int, float, any]:
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


@dataclass
class Experience:
    state: torch.Tensor
    action: np.ndarray
    action_probs: np.ndarray
    sizes: np.ndarray
    reward: float
    next_state: torch.Tensor
    done: bool
    value_target: float = 0.0


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    Higher TD-error experiences are sampled more frequently.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,     # Prioritization exponent
        beta: float = 0.4,      # Importance sampling
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def add(self, experience: Experience):
        """Add experience with max priority"""
        self.tree.add(self.max_priority ** self.alpha, experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with priorities.
        
        Returns:
            experiences: list of experiences
            indices: tree indices for updating
            weights: importance sampling weights
        """
        experiences = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, exp = self.tree.get(s)
            
            if exp is not None:
                experiences.append(exp)
                indices.append(idx)
                priorities.append(priority)
        
        # Calculate importance sampling weights
        priorities = np.array(priorities)
        probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()
        
        return experiences, np.array(indices), weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries


class SimpleReplayBuffer:
    """Simple uniform replay buffer (fallback)"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], None, np.ndarray]:
        experiences = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        weights = np.ones(len(experiences))
        return experiences, None, weights
    
    def __len__(self) -> int:
        return len(self.buffer)
```

### 3. Trainer

```python
"""
trainer.py - Self-play training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    # Training
    num_episodes: int = 10000
    batch_size: int = 256
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    # Self-play
    mcts_simulations: int = 50
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay_episodes: int = 5000
    
    # Replay
    replay_capacity: int = 100000
    min_replay_size: int = 1000
    updates_per_episode: int = 4
    use_prioritized: bool = True
    
    # Curriculum
    curriculum_phases: int = 4
    episodes_per_phase: int = 2500
    
    # Checkpointing
    checkpoint_interval: int = 100
    eval_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 4


class CurriculumScheduler:
    """Curriculum learning scheduler"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_phase = 0
        
        # Phase configurations
        self.phases = [
            # Phase 1: Foundation
            {'H_cycles': 1, 'L_cycles': 2, 'mcts_sims': 25, 
             'reward_type': RewardType.SIMPLE, 'episode_length': 60},
            # Phase 2: Recursive depth
            {'H_cycles': 2, 'L_cycles': 3, 'mcts_sims': 50,
             'reward_type': RewardType.SHARPE, 'episode_length': 120},
            # Phase 3: Full recursion
            {'H_cycles': 4, 'L_cycles': 4, 'mcts_sims': 75,
             'reward_type': RewardType.COMBINED, 'episode_length': 180},
            # Phase 4: Robustness
            {'H_cycles': 4, 'L_cycles': 6, 'mcts_sims': 100,
             'reward_type': RewardType.COMBINED, 'episode_length': 240}
        ]
    
    def get_phase_config(self, episode: int) -> Dict:
        """Get configuration for current phase"""
        phase = min(episode // self.config.episodes_per_phase, len(self.phases) - 1)
        
        if phase != self.current_phase:
            self.current_phase = phase
            logger.info(f"Advancing to curriculum phase {phase + 1}")
        
        return self.phases[phase]


class MCTSTrainer:
    """
    Main trainer for TRM+MCTS agent.
    """
    
    def __init__(
        self,
        agent: 'TRMMCTSAgent',
        env: TradingEnvironment,
        config: TrainingConfig
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.agent.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.agent.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # LR scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.episodes_per_phase,
            T_mult=2
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Replay buffer
        if config.use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(config.replay_capacity)
        else:
            self.replay_buffer = SimpleReplayBuffer(config.replay_capacity)
        
        # Curriculum
        self.curriculum = CurriculumScheduler(config)
        
        # Metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_returns': [],
            'policy_losses': [],
            'value_losses': [],
            'sharpe_ratios': [],
            'max_drawdowns': []
        }
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.config.num_episodes} episodes")
        
        for episode in range(self.config.num_episodes):
            # Get curriculum config
            phase_config = self.curriculum.get_phase_config(episode)
            
            # Update model cycles
            self.agent.model.H_cycles = phase_config['H_cycles']
            self.agent.model.L_cycles = phase_config['L_cycles']
            
            # Update MCTS simulations
            self.agent.mcts.config.num_simulations = phase_config['mcts_sims']
            
            # Update environment
            self.env.config.episode_length = phase_config['episode_length']
            self.env.config.reward_type = phase_config['reward_type']
            
            # Temperature schedule
            temperature = self._get_temperature(episode)
            self.agent.mcts.config.temperature = temperature
            
            # Run self-play episode
            episode_data = self._run_episode()
            
            # Add to replay buffer
            for exp in episode_data['experiences']:
                self.replay_buffer.add(exp)
            
            # Train if enough data
            if len(self.replay_buffer) >= self.config.min_replay_size:
                for _ in range(self.config.updates_per_episode):
                    losses = self._train_step()
                    self.metrics['policy_losses'].append(losses['policy'])
                    self.metrics['value_losses'].append(losses['value'])
            
            # Track metrics
            self.metrics['episode_rewards'].append(episode_data['total_reward'])
            self.metrics['episode_returns'].append(episode_data['total_return'])
            
            # Evaluate
            if episode % self.config.eval_interval == 0:
                self._evaluate(episode)
            
            # Checkpoint
            if episode % self.config.checkpoint_interval == 0:
                self._save_checkpoint(episode)
            
            # Log
            if episode % 10 == 0:
                self._log_progress(episode)
            
            # Step scheduler
            self.scheduler.step()
    
    def _run_episode(self) -> Dict:
        """Run single self-play episode"""
        state = self.env.reset()
        experiences = []
        total_reward = 0.0
        
        done = False
        while not done:
            # Get action from MCTS
            state_tensor = state.unsqueeze(0).to(self.device)
            actions, sizes, value = self.agent.mcts.search(state_tensor)
            
            # Get policy for experience
            with torch.no_grad():
                policy, _, _ = self.agent.model(state_tensor)
            action_probs = policy[0].cpu().numpy()
            
            # Step environment
            next_state, reward, done, info = self.env.step(actions, sizes)
            total_reward += reward
            
            # Store experience
            exp = Experience(
                state=state,
                action=actions,
                action_probs=action_probs,
                sizes=sizes,
                reward=reward,
                next_state=next_state,
                done=done
            )
            experiences.append(exp)
            
            state = next_state
        
        # Calculate value targets (TD-lambda or Monte Carlo)
        self._compute_value_targets(experiences)
        
        return {
            'experiences': experiences,
            'total_reward': total_reward,
            'total_return': info['return']
        }
    
    def _compute_value_targets(self, experiences: List[Experience]):
        """Compute value targets using TD(lambda)"""
        gamma = 0.995
        lambda_ = 0.95
        
        # Bootstrap from final state if not done
        if not experiences[-1].done:
            with torch.no_grad():
                final_state = experiences[-1].next_state.unsqueeze(0).to(self.device)
                _, bootstrap_value, _ = self.agent.model(final_state)
                bootstrap = bootstrap_value.item()
        else:
            bootstrap = 0.0
        
        # Compute TD(lambda) targets backwards
        running_value = bootstrap
        for exp in reversed(experiences):
            running_value = exp.reward + gamma * (
                lambda_ * running_value + (1 - lambda_) * running_value
            )
            exp.value_target = running_value
    
    def _train_step(self) -> Dict[str, float]:
        """Single training step"""
        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(self.config.batch_size)
        
        # Prepare batch
        states = torch.stack([e.state for e in experiences]).to(self.device)
        action_probs = torch.FloatTensor([e.action_probs for e in experiences]).to(self.device)
        value_targets = torch.FloatTensor([e.value_target for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if self.config.mixed_precision:
            with autocast():
                policy, value, _ = self.agent.model(states)
                
                # Policy loss (KL divergence)
                policy_loss = F.kl_div(
                    torch.log(policy + 1e-8),
                    action_probs,
                    reduction='none'
                ).sum(dim=-1).mean(dim=-1)
                
                # Value loss
                value_loss = F.mse_loss(value.squeeze(), value_targets, reduction='none')
                
                # Weighted loss
                total_loss = (weights * (policy_loss + value_loss)).mean()
            
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.agent.model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy, value, _ = self.agent.model(states)
            
            policy_loss = F.kl_div(
                torch.log(policy + 1e-8),
                action_probs,
                reduction='none'
            ).sum(dim=-1).mean(dim=-1)
            
            value_loss = F.mse_loss(value.squeeze(), value_targets, reduction='none')
            total_loss = (weights * (policy_loss + value_loss)).mean()
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
        
        # Update priorities
        if indices is not None:
            td_errors = (value.squeeze() - value_targets).abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
        
        return {
            'policy': policy_loss.mean().item(),
            'value': value_loss.mean().item()
        }
    
    def _get_temperature(self, episode: int) -> float:
        """Get temperature for exploration"""
        progress = min(episode / self.config.temperature_decay_episodes, 1.0)
        return self.config.temperature_start + progress * (
            self.config.temperature_end - self.config.temperature_start
        )
    
    def _evaluate(self, episode: int):
        """Run evaluation episodes"""
        self.agent.model.eval()
        
        returns = []
        sharpes = []
        drawdowns = []
        
        for _ in range(5):  # 5 eval episodes
            state = self.env.reset()
            episode_returns = []
            
            done = False
            while not done:
                with torch.no_grad():
                    actions, sizes, info = self.agent.model.get_action(
                        state.unsqueeze(0).to(self.device),
                        deterministic=True
                    )
                
                actions = actions[0].cpu().numpy()
                sizes = sizes[0].cpu().numpy()
                
                state, reward, done, info = self.env.step(actions, sizes)
                episode_returns.append(info['return'])
            
            returns.append(info['return'])
            
            # Calculate Sharpe
            if len(episode_returns) > 1:
                ret_arr = np.diff(episode_returns)
                sharpe = ret_arr.mean() / (ret_arr.std() + 1e-8) * np.sqrt(252 * 24)  # Annualized
                sharpes.append(sharpe)
            
            drawdowns.append(info['drawdown'])
        
        self.metrics['sharpe_ratios'].append(np.mean(sharpes))
        self.metrics['max_drawdowns'].append(np.mean(drawdowns))
        
        logger.info(
            f"Eval @ {episode}: Return={np.mean(returns)*100:.2f}%, "
            f"Sharpe={np.mean(sharpes):.2f}, MaxDD={np.mean(drawdowns)*100:.1f}%"
        )
        
        self.agent.model.train()
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state': self.agent.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        path = self.checkpoint_dir / f"checkpoint_{episode}.pt"
        torch.save(checkpoint, path)
        
        # Also save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.agent.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.metrics = checkpoint['metrics']
        
        return checkpoint['episode']
    
    def _log_progress(self, episode: int):
        """Log training progress"""
        recent_rewards = self.metrics['episode_rewards'][-100:]
        recent_returns = self.metrics['episode_returns'][-100:]
        
        lr = self.optimizer.param_groups[0]['lr']
        phase = self.curriculum.current_phase + 1
        
        logger.info(
            f"Episode {episode} | Phase {phase} | "
            f"Reward: {np.mean(recent_rewards):.4f} | "
            f"Return: {np.mean(recent_returns)*100:.2f}% | "
            f"LR: {lr:.6f} | "
            f"Buffer: {len(self.replay_buffer)}"
        )
```

### 4. Paper Trading

```python
"""
paper_trade.py - Paper trading for validation
"""

import asyncio
from datetime import datetime
import pandas as pd

class PaperTrader:
    """
    Paper trading with live data.
    Validates agent before live deployment.
    """
    
    def __init__(
        self,
        agent: 'TRMMCTSAgent',
        encoder: 'MultiAssetStateEncoder',
        kraken_client: 'KrakenWebSocket',
        initial_capital: float = 10000.0
    ):
        self.agent = agent
        self.encoder = encoder
        self.kraken = kraken_client
        
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        self.running = False
    
    async def run(self, duration_hours: float = 24):
        """Run paper trading session"""
        self.running = True
        start_time = datetime.now()
        
        logger.info(f"Starting paper trading for {duration_hours} hours")
        
        # Subscribe to market data
        symbols = list(self.encoder.symbol_to_idx.keys())
        await self.kraken.connect()
        await self.kraken.subscribe_ohlc(symbols, interval=1)
        
        while self.running:
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            if elapsed >= duration_hours:
                break
            
            # Wait for new data
            await asyncio.sleep(60)  # 1-minute intervals
            
            # Update encoder with latest data
            prices = {}
            for symbol in symbols:
                if symbol in self.kraken.latest_prices:
                    prices[symbol] = self.kraken.latest_prices[symbol]
                    self.encoder.update_asset(
                        self.encoder.symbol_to_idx[symbol],
                        self.kraken.ohlcv_buffer[symbol]
                    )
            
            # Get state and act
            state = torch.FloatTensor(self.encoder.encode()['combined'])
            
            # Run agent (no MCTS for speed in paper trading)
            with torch.no_grad():
                actions, sizes, info = self.agent.model.get_action(
                    state.unsqueeze(0),
                    deterministic=True
                )
            
            actions = actions[0].cpu().numpy()
            sizes = sizes[0].cpu().numpy()
            
            # Execute paper trades
            self._execute_paper_trades(actions, sizes, prices)
            
            # Update equity curve
            equity = self._calculate_equity(prices)
            self.equity_curve.append((datetime.now(), equity))
            
            # Log status
            if len(self.equity_curve) % 60 == 0:  # Every hour
                self._log_status()
        
        await self.kraken.close()
        self._generate_report()
    
    def _execute_paper_trades(
        self,
        actions: np.ndarray,
        sizes: np.ndarray,
        prices: Dict[str, float]
    ):
        """Execute paper trades"""
        for idx, (action, size) in enumerate(zip(actions, sizes)):
            symbol = self.encoder.idx_to_symbol.get(idx)
            if symbol is None or symbol not in prices:
                continue
            
            price = prices[symbol]
            
            if action == 1:  # Hold
                continue
            
            # Similar logic to real trading but just track state
            # ... (simplified)
    
    def _calculate_equity(self, prices: Dict[str, float]) -> float:
        """Calculate current equity"""
        equity = self.capital
        
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
                equity += pos.pnl
        
        return equity
    
    def _generate_report(self):
        """Generate paper trading report"""
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        
        # Calculate metrics
        returns = df['equity'].pct_change().dropna()
        
        report = {
            'total_return': (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24),
            'max_drawdown': ((df['equity'].cummax() - df['equity']) / df['equity'].cummax()).max() * 100,
            'n_trades': len(self.trade_history),
            'win_rate': sum(1 for t in self.trade_history if t.get('pnl', 0) > 0) / max(len(self.trade_history), 1) * 100
        }
        
        logger.info("Paper Trading Report:")
        for k, v in report.items():
            logger.info(f"  {k}: {v:.2f}")
        
        # Save report
        with open('paper_trading_report.json', 'w') as f:
            json.dump(report, f, indent=2)
```

## Usage

```python
# Initialize components
config = TrainingConfig(
    num_episodes=10000,
    device='cuda',
    mixed_precision=True
)

# Load data
data = load_market_data()  # From Prompt 2

# Create components
encoder = MultiAssetStateEncoder(n_assets=64)
env = TradingEnvironment(data, encoder, EnvConfig())
agent = TRMMCTSAgent(n_assets=64, use_mcts=True)

# Train
trainer = MCTSTrainer(agent, env, config)
trainer.train()

# Paper trade validation
paper_trader = PaperTrader(agent, encoder, kraken_ws)
asyncio.run(paper_trader.run(duration_hours=24))
```

## Training Timeline

| Phase | Episodes | H/L Cycles | MCTS Sims | Focus |
|-------|----------|------------|-----------|-------|
| 1 | 0-2500 | 1/2 | 25 | Basic patterns |
| 2 | 2500-5000 | 2/3 | 50 | Sharpe optimization |
| 3 | 5000-7500 | 4/4 | 75 | Full reward |
| 4 | 7500-10000 | 4/6 | 100 | Robustness |

Estimated time on RTX 3090: ~48-72 hours for full training.
```

Ready for Prompt 6 (Monitoring & Analysis)?
