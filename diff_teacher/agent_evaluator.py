"""
Agent Evaluator for Adversarial Market Generation.

TASK_110: Phase 2 - Runs TRM+MuZero agent on generated market sequences
and computes PnL for regret calculation.

Usage:
    evaluator = AgentEvaluator("jax_ckpt_fast/latest", model_size="small")
    result = evaluator.evaluate(prices)  # (steps, n_assets) array
    print(f"Agent PnL: {result.actual_pnl}")
"""

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import chex

# Silence JAX warnings
os.environ['JAX_PLATFORM_NAME'] = 'gpu'


@dataclass
class EvaluationResult:
    """Result of evaluating agent on a price sequence."""
    actual_pnl: float                  # Final PnL (negative = loss)
    final_portfolio_value: float       # Final portfolio value
    initial_cash: float                # Starting cash
    return_pct: float                  # Percentage return

    # Action statistics
    n_trades: int                      # Number of trades executed
    n_buys: int                        # Number of buy actions
    n_sells: int                       # Number of sell actions
    n_holds: int                       # Number of hold actions
    hold_rate: float                   # Fraction of holds

    # Position statistics
    max_position_count: int            # Max simultaneous positions
    avg_position_duration: float       # Average holding time

    # Regime info (if available)
    regime: Optional[str] = None

    # Full trajectory (optional, for analysis)
    actions: Optional[List[int]] = None
    portfolio_values: Optional[List[float]] = None
    positions_history: Optional[List[Dict]] = None


class SimpleStateEncoder:
    """
    Simplified state encoder for evaluation.
    Creates (27, n_assets, n_assets) observations from price arrays.

    Unlike FastMultiAssetStateEncoder, this takes a full price sequence
    rather than streaming updates.
    """

    def __init__(self, n_assets: int = 64, window_size: int = 60):
        self.n_assets = n_assets
        self.window_size = window_size

    def encode(
        self,
        prices: np.ndarray,      # (window_size, n_assets) or longer
        step: int,               # Current step in episode
        positions: np.ndarray,   # (n_assets,) position sizes
        cash: float,
        portfolio_value: float,
    ) -> np.ndarray:
        """
        Encode market state into (27, n_assets, n_assets) observation.

        Args:
            prices: Full price sequence (total_steps, n_assets)
            step: Current timestep (where to sample window from)
            positions: Current position sizes per asset
            cash: Current cash
            portfolio_value: Current total portfolio value

        Returns:
            (27, n_assets, n_assets) observation tensor
        """
        n = self.n_assets
        w = self.window_size

        # Get price window ending at current step
        start_idx = max(0, step - w + 1)
        end_idx = step + 1
        window_prices = prices[start_idx:end_idx]  # (w', n_assets)

        # Pad if not enough history
        if len(window_prices) < w:
            pad_len = w - len(window_prices)
            window_prices = np.concatenate([
                np.tile(window_prices[0:1], (pad_len, 1)),
                window_prices
            ], axis=0)

        # Normalize prices (per-asset z-score)
        price_mean = window_prices.mean(axis=0, keepdims=True)
        price_std = window_prices.std(axis=0, keepdims=True) + 1e-8
        norm_prices = (window_prices - price_mean) / price_std  # (w, n)

        # Create 27 channels
        channels = np.zeros((27, n, n), dtype=np.float32)

        # Channels 0-3: Normalized OHLC (just close repeated for simplicity)
        for i in range(4):
            channels[i] = np.tile(norm_prices[-1:].T, (1, n))  # (n, n)

        # Channels 4-7: Log returns at different horizons
        for i, horizon in enumerate([1, 5, 15, 60]):
            h = min(horizon, len(window_prices) - 1)
            if h > 0:
                log_ret = np.log(window_prices[-1] / (window_prices[-1-h] + 1e-8) + 1e-8)
                channels[4 + i] = np.tile(log_ret.reshape(-1, 1), (1, n))

        # Channels 8-11: Volume patterns (zeros since we don't have volume)
        # Channels 8-11 remain zeros

        # Channels 12-15: Price momentum (rolling mean differences)
        for i, period in enumerate([5, 10, 20, 60]):
            p = min(period, len(window_prices))
            if p > 0:
                short_ma = window_prices[-p:].mean(axis=0)
                long_ma = window_prices.mean(axis=0)
                momentum = (short_ma - long_ma) / (long_ma + 1e-8)
                channels[12 + i] = np.tile(momentum.reshape(-1, 1), (1, n))

        # Channels 16-19: Volatility at different horizons
        for i, period in enumerate([5, 10, 20, 60]):
            p = min(period, len(window_prices))
            if p > 1:
                returns = np.diff(window_prices[-p:], axis=0) / (window_prices[-p:-1] + 1e-8)
                vol = returns.std(axis=0)
                channels[16 + i] = np.tile(vol.reshape(-1, 1), (1, n))

        # Channels 20-23: Price percentiles
        for i, pct in enumerate([25, 50, 75, 100]):
            if len(window_prices) > 0:
                percentile = np.percentile(window_prices, pct, axis=0)
                rel_to_current = (window_prices[-1] - percentile) / (percentile + 1e-8)
                channels[20 + i] = np.tile(rel_to_current.reshape(-1, 1), (1, n))

        # Channels 24-26: Position/Portfolio info
        # Channel 24: Position sizes (normalized)
        pos_norm = positions / (np.abs(positions).max() + 1e-8)
        channels[24] = np.tile(pos_norm.reshape(-1, 1), (1, n))

        # Channel 25: Cash ratio
        cash_ratio = cash / (portfolio_value + 1e-8)
        channels[25] = np.full((n, n), cash_ratio)

        # Channel 26: Portfolio return
        initial_cash = 10000.0  # Assumed
        ret = (portfolio_value - initial_cash) / initial_cash
        channels[26] = np.full((n, n), np.clip(ret, -1, 1))

        return channels


class AgentEvaluator:
    """
    Evaluate TRM+MuZero agent on generated market sequences.

    Loads a checkpoint and runs inference to compute actual PnL,
    which is compared to optimal PnL for regret calculation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "small",
        n_simulations: int = 24,
        use_popart: bool = True,
        n_assets: int = 64,
        initial_cash: float = 10000.0,
        fee_bps: float = 10.0,
        window_size: int = 60,
    ):
        """
        Initialize evaluator with trained agent.

        Args:
            checkpoint_path: Path to TRM+MuZero checkpoint
            model_size: small/medium/large (must match checkpoint)
            n_simulations: MCTS simulations per step
            use_popart: Whether PopArt normalization was used
            n_assets: Number of assets (default 64)
            initial_cash: Starting cash for episodes
            fee_bps: Trading fee in basis points
            window_size: Observation window size
        """
        self.checkpoint_path = checkpoint_path
        self.model_size = model_size
        self.n_simulations = n_simulations
        self.use_popart = use_popart
        self.n_assets = n_assets
        self.initial_cash = initial_cash
        self.fee_bps = fee_bps
        self.window_size = window_size

        # Model size configs
        SIZE_CONFIGS = {
            'small':  {'state_dim': 64,  'h_cycles': 2, 'l_cycles': 2},
            'medium': {'state_dim': 96,  'h_cycles': 3, 'l_cycles': 3},
            'large':  {'state_dim': 160, 'h_cycles': 4, 'l_cycles': 4},
        }
        self.config = SIZE_CONFIGS.get(model_size, SIZE_CONFIGS['small'])

        # Lazy load model/trainer (on first evaluate call)
        self._model = None
        self._trainer = None
        self._search = None
        self._params = None

        # State encoder
        self.encoder = SimpleStateEncoder(n_assets=n_assets, window_size=window_size)

        # RNG
        self.rng = jax.random.PRNGKey(42)

    def _load_model(self):
        """Lazy load the model and checkpoint."""
        if self._model is not None:
            return

        # Ensure absolute path
        ckpt_path = os.path.abspath(self.checkpoint_path)
        print(f"Loading agent from {ckpt_path}...")

        # Import here to avoid circular imports
        from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
        from mcts_trader.jax_agent.trainer import MuZeroTrainer
        from mcts_trader.jax_agent.mctx_search import create_mctx_search

        # Try to read checkpoint metadata to determine model architecture
        metadata_path = os.path.join(ckpt_path, 'trainer_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            # Override config from checkpoint if available
            if 'h_cycles' in metadata:
                self.config['h_cycles'] = metadata['h_cycles']
            if 'l_cycles' in metadata:
                self.config['l_cycles'] = metadata['l_cycles']
            if 'state_dim' in metadata:
                self.config['state_dim'] = metadata['state_dim']
            print(f"  Using checkpoint architecture: state_dim={self.config['state_dim']}, "
                  f"H={self.config['h_cycles']}, L={self.config['l_cycles']}")

        action_space_size = self.n_assets * 5  # 5 actions per asset

        self._model = MuZeroNetworks(
            state_dim=self.config['state_dim'],
            action_space_size=action_space_size,
            H_cycles=self.config['h_cycles'],
            L_cycles=self.config['l_cycles'],
            use_prediction_head=False,  # Not needed for inference
            n_assets=self.n_assets,
        )

        self._trainer = MuZeroTrainer(
            model=self._model,
            num_simulations=self.n_simulations,
            use_popart=self.use_popart,
        )

        # Initialize with dummy observation
        dummy_obs = jnp.ones((1, 27, self.n_assets, self.n_assets))
        self._trainer.init_train_state(jax.random.PRNGKey(0), dummy_obs)

        # Load checkpoint
        self._trainer.load_checkpoint(ckpt_path)
        self._params = self._trainer.state.params

        # Create search function
        self._search = create_mctx_search(
            self._model,
            num_simulations=self.n_simulations,
            use_gumbel=True,
        )

        print(f"Agent loaded: {self.model_size} model, {self.n_simulations} sims")

    def evaluate(
        self,
        prices: np.ndarray,
        regime: Optional[str] = None,
        return_trajectory: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate agent on a single price sequence.

        Args:
            prices: (steps, n_assets) price sequence
            regime: Optional regime label for analysis
            return_trajectory: Whether to return full action/value history

        Returns:
            EvaluationResult with PnL and statistics
        """
        self._load_model()

        steps, n_assets = prices.shape
        if n_assets != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} assets, got {n_assets}")

        # Initialize portfolio state
        cash = self.initial_cash
        positions = np.zeros(n_assets, dtype=np.float32)  # Shares held
        entry_prices = np.zeros(n_assets, dtype=np.float32)

        # Tracking
        actions_taken = []
        portfolio_values = [self.initial_cash]
        positions_history = []

        n_buys, n_sells, n_holds = 0, 0, 0
        n_trades = 0
        position_durations = []
        position_start_times = {}
        max_position_count = 0

        # Episode loop
        for step in range(self.window_size, steps):
            self.rng, step_rng = jax.random.split(self.rng)

            # Current prices
            current_prices = prices[step]

            # Portfolio value
            position_values = positions * current_prices
            portfolio_value = cash + position_values.sum()

            # Encode observation
            obs = self.encoder.encode(
                prices, step, positions, cash, portfolio_value
            )
            obs_jax = jnp.array(obs[np.newaxis, ...], dtype=jnp.float32)

            # Action mask (all valid for simplicity)
            mask_jax = jnp.ones((1, self.n_assets * 5), dtype=jnp.int8)

            # MCTS search
            action_jax, policy_jax, stats = self._search(
                self._params, step_rng, obs_jax, mask_jax
            )
            action = int(action_jax[0])

            # Decode action
            asset_idx = action // 5
            action_type = action % 5

            # Execute action
            if action_type == 0:  # Sell
                if positions[asset_idx] > 0:
                    # Sell existing position
                    sell_value = positions[asset_idx] * current_prices[asset_idx]
                    fee = sell_value * self.fee_bps / 10000
                    cash += sell_value - fee

                    # Track duration
                    if asset_idx in position_start_times:
                        duration = step - position_start_times[asset_idx]
                        position_durations.append(duration)
                        del position_start_times[asset_idx]

                    positions[asset_idx] = 0
                    n_trades += 1
                n_sells += 1

            elif action_type == 1:  # Hold
                n_holds += 1

            elif action_type == 2:  # Buy
                # Buy with 10% of portfolio
                buy_value = portfolio_value * 0.1
                if buy_value < cash:
                    fee = buy_value * self.fee_bps / 10000
                    shares = (buy_value - fee) / current_prices[asset_idx]

                    # Update position (average entry if adding)
                    if positions[asset_idx] > 0:
                        old_value = positions[asset_idx] * entry_prices[asset_idx]
                        new_value = shares * current_prices[asset_idx]
                        entry_prices[asset_idx] = (old_value + new_value) / (positions[asset_idx] + shares)
                    else:
                        entry_prices[asset_idx] = current_prices[asset_idx]
                        position_start_times[asset_idx] = step

                    positions[asset_idx] += shares
                    cash -= buy_value
                    n_trades += 1
                n_buys += 1

            elif action_type == 3:  # Limit Sell (treat as hold for simplicity)
                n_holds += 1

            elif action_type == 4:  # Limit Buy (treat as hold for simplicity)
                n_holds += 1

            # Track
            actions_taken.append(action)
            portfolio_values.append(portfolio_value)
            max_position_count = max(max_position_count, (positions > 0).sum())

            if return_trajectory:
                positions_history.append({
                    'step': step,
                    'cash': cash,
                    'positions': positions.copy(),
                    'portfolio_value': portfolio_value,
                })

        # Final portfolio value
        final_prices = prices[-1]
        final_position_values = positions * final_prices
        final_portfolio_value = cash + final_position_values.sum()

        # Compute metrics
        actual_pnl = final_portfolio_value - self.initial_cash
        return_pct = actual_pnl / self.initial_cash * 100

        total_actions = n_buys + n_sells + n_holds
        hold_rate = n_holds / max(total_actions, 1)
        avg_duration = np.mean(position_durations) if position_durations else 0.0

        return EvaluationResult(
            actual_pnl=actual_pnl,
            final_portfolio_value=final_portfolio_value,
            initial_cash=self.initial_cash,
            return_pct=return_pct,
            n_trades=n_trades,
            n_buys=n_buys,
            n_sells=n_sells,
            n_holds=n_holds,
            hold_rate=hold_rate,
            max_position_count=max_position_count,
            avg_position_duration=avg_duration,
            regime=regime,
            actions=actions_taken if return_trajectory else None,
            portfolio_values=portfolio_values if return_trajectory else None,
            positions_history=positions_history if return_trajectory else None,
        )

    def evaluate_batch(
        self,
        prices_batch: np.ndarray,
        regimes: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate agent on multiple price sequences.

        Args:
            prices_batch: (batch, steps, n_assets) price sequences
            regimes: Optional list of regime labels

        Returns:
            List of EvaluationResult
        """
        batch_size = prices_batch.shape[0]
        regimes = regimes or [None] * batch_size

        results = []
        for i in range(batch_size):
            result = self.evaluate(
                prices_batch[i],
                regime=regimes[i] if i < len(regimes) else None,
            )
            results.append(result)

        return results

    def compute_regret(
        self,
        actual_pnl: float,
        optimal_pnl: float,
    ) -> float:
        """
        Compute regret: how much worse agent did vs optimal.

        Regret = (optimal_pnl - actual_pnl) / |optimal_pnl|

        Range: [0, inf) where 0 = matched optimal
        """
        if abs(optimal_pnl) < 1e-6:
            # If optimal is ~0, use absolute difference
            return abs(actual_pnl) if actual_pnl < 0 else 0.0
        return (optimal_pnl - actual_pnl) / abs(optimal_pnl)

    def load_checkpoint(self, path: str):
        """Reload agent from a different checkpoint."""
        self.checkpoint_path = path
        self._model = None
        self._trainer = None
        self._search = None
        self._params = None


def test_agent_evaluator():
    """Quick test of AgentEvaluator."""
    import os

    # Find a checkpoint
    ckpt_dirs = [
        "jax_ckpt_fast/latest",
        "jax_ckpt_fast/checkpoint_1000",
        "checkpoints/level_0/checkpoint_golden",
    ]

    ckpt_path = None
    for d in ckpt_dirs:
        if os.path.exists(d):
            ckpt_path = d
            break

    if ckpt_path is None:
        print("No checkpoint found, skipping test")
        return

    print(f"Testing with checkpoint: {ckpt_path}")

    # Create evaluator
    evaluator = AgentEvaluator(
        checkpoint_path=ckpt_path,
        model_size="small",
        n_simulations=8,  # Fewer for speed
    )

    # Generate random prices
    np.random.seed(42)
    steps = 200
    n_assets = 64

    # Random walk prices
    returns = np.random.randn(steps, n_assets) * 0.01  # 1% daily vol
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    print(f"Evaluating on random prices: {prices.shape}")

    # Evaluate
    result = evaluator.evaluate(prices, regime="random")

    print(f"\nEvaluation Result:")
    print(f"  PnL: ${result.actual_pnl:.2f}")
    print(f"  Return: {result.return_pct:.2f}%")
    print(f"  Trades: {result.n_trades}")
    print(f"  Hold rate: {result.hold_rate:.2%}")
    print(f"  Max positions: {result.max_position_count}")


if __name__ == "__main__":
    test_agent_evaluator()
