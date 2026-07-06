"""
Optimal Solver (Hindsight Oracle) for Market Sequences.

Computes the maximum achievable PnL given perfect foresight.
Used for regret-based curriculum: regret = optimal_pnl - actual_pnl

Algorithms:
    - Peak-Valley: Buy at every local min, sell at every local max
    - Dynamic Programming: Optimal with transaction costs and position limits
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class OptimalResult:
    """Result from optimal solver."""

    total_pnl: float  # Total profit/loss
    n_trades: int  # Number of trades executed
    max_drawdown: float  # Maximum drawdown
    sharpe_approx: float  # Approximate Sharpe ratio
    per_asset_pnl: np.ndarray  # PnL per asset


class OptimalSolver:
    """
    Hindsight optimal solver for trading.

    Computes the maximum possible PnL for a given price sequence
    assuming perfect knowledge of future prices.
    """

    def __init__(
        self,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        max_position: float = 1.0,
        algorithm: str = "peak_valley",
    ):
        """
        Initialize solver.

        Args:
            fee_bps: Transaction fee in basis points
            slippage_bps: Slippage in basis points
            max_position: Maximum position size per asset (fraction of portfolio)
            algorithm: "peak_valley" (fast) or "dp" (handles costs)
        """
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.total_cost_bps = fee_bps + slippage_bps
        self.max_position = max_position
        self.algorithm = algorithm

    def solve(self, prices: np.ndarray) -> OptimalResult:
        """
        Compute optimal PnL for given prices.

        Args:
            prices: (window_size, n_assets) close prices
                   or (window_size, n_assets, 5) OHLCV

        Returns:
            OptimalResult with total PnL and metrics
        """
        # Extract close prices if OHLCV
        if prices.ndim == 3:
            prices = prices[:, :, 3]  # Close prices

        if self.algorithm == "peak_valley" and self.total_cost_bps == 0:
            return self._solve_peak_valley(prices)
        else:
            return self._solve_dp(prices)

    def _solve_peak_valley(self, prices: np.ndarray) -> OptimalResult:
        """
        Peak-valley algorithm (no transaction costs).

        For each asset, sum all positive price movements.
        This is the maximum profit with unlimited trades.

        Time complexity: O(T * N) where T=timesteps, N=assets
        """
        n_steps, n_assets = prices.shape

        per_asset_pnl = np.zeros(n_assets)
        total_trades = 0

        for asset in range(n_assets):
            asset_prices = prices[:, asset]

            # Skip if all zeros or NaN
            if np.all(asset_prices == 0) or np.any(np.isnan(asset_prices)):
                continue

            # Sum all positive price differences
            diffs = np.diff(asset_prices)
            positive_moves = diffs[diffs > 0]
            negative_moves = diffs[diffs < 0]

            # PnL from capturing all positive moves
            pnl = np.sum(positive_moves)

            # Normalize by first price (return percentage)
            if asset_prices[0] > 0:
                pnl = pnl / asset_prices[0]

            per_asset_pnl[asset] = pnl
            total_trades += len(positive_moves) * 2  # Buy + sell for each move

        total_pnl = np.mean(per_asset_pnl)  # Average across assets

        # Compute max drawdown (simplified)
        max_drawdown = self._compute_max_drawdown(prices)

        # Approximate Sharpe (using daily returns assumption)
        returns = np.diff(prices, axis=0) / (prices[:-1] + 1e-10)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-10
        sharpe = mean_return / std_return * np.sqrt(252 * 24 * 60)  # Annualized

        return OptimalResult(
            total_pnl=float(total_pnl),
            n_trades=total_trades,
            max_drawdown=float(max_drawdown),
            sharpe_approx=float(sharpe),
            per_asset_pnl=per_asset_pnl,
        )

    def _solve_dp(self, prices: np.ndarray) -> OptimalResult:
        """
        Dynamic programming with transaction costs.

        State: (timestep, position) where position in {0, 1}
        Transition: Buy, Sell, or Hold

        Time complexity: O(T * N) with constant number of states
        """
        n_steps, n_assets = prices.shape
        cost_mult = 1 - self.total_cost_bps / 10000

        per_asset_pnl = np.zeros(n_assets)
        total_trades = 0

        for asset in range(n_assets):
            asset_prices = prices[:, asset]

            if np.all(asset_prices == 0) or np.any(np.isnan(asset_prices)):
                continue

            # DP states: cash (no position), holding
            cash = 0.0  # Profit when not holding
            holding = -asset_prices[0] * cost_mult  # Profit after buying

            trades = 0

            for i in range(1, n_steps):
                price = asset_prices[i]

                # Options:
                # 1. Stay in cash
                # 2. Buy (from cash to holding)
                # 3. Stay holding
                # 4. Sell (from holding to cash)

                new_cash = max(
                    cash,  # Stay in cash
                    holding + price * cost_mult,  # Sell
                )

                new_holding = max(
                    holding,  # Stay holding
                    cash - price * cost_mult,  # Buy
                )

                # Count trades (approximate)
                if new_cash > cash and holding + price * cost_mult > cash:
                    trades += 1  # Sold
                if new_holding > holding and cash - price * cost_mult > holding:
                    trades += 1  # Bought

                cash = new_cash
                holding = new_holding

            # Final: must be in cash (sell if holding)
            final_pnl = max(cash, holding + asset_prices[-1] * cost_mult)

            # Normalize by initial price
            if asset_prices[0] > 0:
                final_pnl = final_pnl / asset_prices[0]

            per_asset_pnl[asset] = final_pnl
            total_trades += trades

        total_pnl = np.mean(per_asset_pnl)
        max_drawdown = self._compute_max_drawdown(prices)

        returns = np.diff(prices, axis=0) / (prices[:-1] + 1e-10)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-10
        sharpe = mean_return / std_return * np.sqrt(252 * 24 * 60)

        return OptimalResult(
            total_pnl=float(total_pnl),
            n_trades=total_trades,
            max_drawdown=float(max_drawdown),
            sharpe_approx=float(sharpe),
            per_asset_pnl=per_asset_pnl,
        )

    def _compute_max_drawdown(self, prices: np.ndarray) -> float:
        """Compute maximum drawdown across all assets."""
        # Use mean price across assets as proxy
        mean_prices = np.mean(prices, axis=1)

        running_max = np.maximum.accumulate(mean_prices)
        drawdowns = (running_max - mean_prices) / (running_max + 1e-10)

        return float(np.max(drawdowns))


# =============================================================================
# JAX-compatible version for GPU acceleration
# =============================================================================
@jax.jit
def optimal_pnl_jax(prices: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-jitted optimal PnL computation (peak-valley, no costs).

    Args:
        prices: (batch, window_size, n_assets) or (window_size, n_assets)

    Returns:
        (batch,) or scalar optimal PnL
    """
    if prices.ndim == 2:
        prices = prices[None, :, :]
        squeeze = True
    else:
        squeeze = False

    # Sum all positive differences
    diffs = jnp.diff(prices, axis=1)  # (batch, T-1, N)
    positive_moves = jnp.maximum(diffs, 0)

    # Sum per asset, then average
    per_asset_gain = jnp.sum(positive_moves, axis=1)  # (batch, N)

    # Normalize by first price
    first_prices = prices[:, 0, :]  # (batch, N)
    per_asset_return = per_asset_gain / (first_prices + 1e-10)

    # Average across assets
    optimal = jnp.mean(per_asset_return, axis=1)  # (batch,)

    if squeeze:
        optimal = optimal[0]

    return optimal


@jax.jit
def regret_score(
    optimal_pnl: jnp.ndarray,
    actual_pnl: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute regret score.

    Regret = (optimal - actual) / (|optimal| + epsilon)

    Normalized to [0, 1] range approximately.

    Args:
        optimal_pnl: Optimal PnL from hindsight
        actual_pnl: Actual PnL achieved by agent

    Returns:
        Regret score (higher = worse performance)
    """
    diff = optimal_pnl - actual_pnl
    scale = jnp.abs(optimal_pnl) + 1e-6

    # Clip to reasonable range
    regret = jnp.clip(diff / scale, 0.0, 2.0)

    return regret


# =============================================================================
# Batch processing utilities
# =============================================================================
def compute_batch_optimal(
    prices_batch: np.ndarray,
    fee_bps: float = 0.0,
    use_jax: bool = True,
) -> np.ndarray:
    """
    Compute optimal PnL for a batch of price sequences.

    Args:
        prices_batch: (batch, window_size, n_assets) or (batch, window_size, n_assets, 5)
        fee_bps: Transaction fee
        use_jax: Use JAX acceleration

    Returns:
        (batch,) optimal PnL values
    """
    # Extract close prices if OHLCV
    if prices_batch.ndim == 4:
        prices_batch = prices_batch[:, :, :, 3]

    if use_jax and fee_bps == 0:
        return np.array(optimal_pnl_jax(jnp.array(prices_batch)))
    else:
        solver = OptimalSolver(fee_bps=fee_bps)
        results = [solver.solve(prices_batch[i]) for i in range(len(prices_batch))]
        return np.array([r.total_pnl for r in results])


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    import time

    # Test with synthetic price data
    np.random.seed(42)

    n_steps = 400
    n_assets = 64

    # Generate random walk prices
    returns = np.random.randn(n_steps - 1, n_assets) * 0.01
    prices = np.zeros((n_steps, n_assets))
    prices[0] = 100  # Initial price
    prices[1:] = 100 * np.exp(np.cumsum(returns, axis=0))

    print("=" * 60)
    print("Optimal Solver Test")
    print("=" * 60)

    # Test peak-valley (no costs)
    solver_pv = OptimalSolver(algorithm="peak_valley")
    t0 = time.time()
    result_pv = solver_pv.solve(prices)
    t_pv = time.time() - t0

    print(f"\nPeak-Valley (no costs):")
    print(f"  Total PnL: {result_pv.total_pnl:.4f} ({result_pv.total_pnl * 100:.2f}%)")
    print(f"  Trades: {result_pv.n_trades}")
    print(f"  Max Drawdown: {result_pv.max_drawdown:.4f}")
    print(f"  Time: {t_pv * 1000:.2f}ms")

    # Test DP with costs
    solver_dp = OptimalSolver(fee_bps=10, slippage_bps=2, algorithm="dp")
    t0 = time.time()
    result_dp = solver_dp.solve(prices)
    t_dp = time.time() - t0

    print(f"\nDP (12 bps cost):")
    print(f"  Total PnL: {result_dp.total_pnl:.4f} ({result_dp.total_pnl * 100:.2f}%)")
    print(f"  Trades: {result_dp.n_trades}")
    print(f"  Time: {t_dp * 1000:.2f}ms")

    # Test JAX version
    print("\n--- JAX Benchmark ---")
    prices_jax = jnp.array(prices)

    # Warmup
    _ = optimal_pnl_jax(prices_jax)

    # Benchmark
    t0 = time.time()
    for _ in range(100):
        _ = optimal_pnl_jax(prices_jax)
    t_jax = (time.time() - t0) / 100

    print(f"JAX optimal (single): {t_jax * 1000:.3f}ms")

    # Batch benchmark
    batch_size = 32
    prices_batch = np.random.randn(batch_size, n_steps, n_assets) * 0.01
    prices_batch = 100 * np.exp(np.cumsum(prices_batch, axis=1))
    prices_batch_jax = jnp.array(prices_batch)

    # Warmup
    _ = optimal_pnl_jax(prices_batch_jax)

    t0 = time.time()
    for _ in range(100):
        _ = optimal_pnl_jax(prices_batch_jax)
    t_batch = (time.time() - t0) / 100

    print(f"JAX optimal (batch {batch_size}): {t_batch * 1000:.3f}ms")
    print(f"  Per sample: {t_batch / batch_size * 1000:.3f}ms")

    # Test regret
    print("\n--- Regret Test ---")
    optimal = optimal_pnl_jax(prices_batch_jax)
    actual = optimal * 0.7  # Simulate 70% of optimal

    regret = regret_score(optimal, actual)
    print(f"Optimal PnL range: [{optimal.min():.4f}, {optimal.max():.4f}]")
    print(f"Actual PnL range: [{actual.min():.4f}, {actual.max():.4f}]")
    print(f"Regret range: [{regret.min():.4f}, {regret.max():.4f}]")
    print(f"Mean regret: {regret.mean():.4f}")
