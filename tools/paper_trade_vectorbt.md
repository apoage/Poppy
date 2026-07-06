# paper_trade_vectorbt.py

**Paper Trading Audit using VectorBT Environment + QuantStats**

## Purpose

Run paper trading audits using the VectorBT environment with QuantStats reporting. Uses the same environment as training for consistent accounting.

## Usage

```bash
# Basic usage with checkpoint
python tools/paper_trade_vectorbt.py --checkpoint jax_ckpt_fast/default/checkpoint_250

# With synthetic data
python tools/paper_trade_vectorbt.py --checkpoint jax_ckpt_fast/default/checkpoint_250 --synthetic-level 0 --steps 500

# Generate QuantStats report
python tools/paper_trade_vectorbt.py --checkpoint jax_ckpt_fast/default/checkpoint_250 --generate-report --output-dir reports/
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | None | Path to model checkpoint |
| `--n-assets` | 64 | Number of assets |
| `--initial-cash` | 10000 | Starting capital |
| `--fee` | 26 | Trading fee in basis points |
| `--slippage` | 5 | Slippage in basis points |
| `--steps` | 500 | Steps per episode |
| `--synthetic-level` | 0 | Synthetic data level (0-3) |
| `--synthetic-steps` | 10000 | Synthetic data length |
| `--state-dim` | 64 | Model state dimension |
| `--h-cycles` | 4 | TRM H cycles |
| `--l-cycles` | 6 | TRM L cycles |
| `--sims` | 32 | MCTS simulations |
| `--gumbel-scale` | 3.0 | Gumbel scale |
| `--episodes` | 3 | Number of episodes to run |
| `--generate-report` | False | Generate QuantStats HTML report |
| `--output-dir` | reports | Output directory for reports |

## Output

- **Console**: Episode-by-episode performance summary (return, Sharpe, positions, actions)
- **HTML Report**: QuantStats report with Sharpe, Sortino, max drawdown, win rate, volatility

## Example Output

```
======================================================================
PAPER TRADING AUDIT (VectorBT)
======================================================================

Episode 1/3
  Return: +2.54% | Sharpe (approx): 0.85
  Actions: BUY=18, HOLD=73, SELL=9
  Final positions: 5

SUMMARY
======================================================================
Episodes: 3
Avg Return: +1.49%
Std Return: 2.31%
Win Rate: 66.7%
```

## Dependencies

- VectorBTTradingEnv (mcts_trader.env.vectorbt_env)
- MuZeroNetworks (mcts_trader.jax_agent.muzero_model)
- QuantStats (optional, for HTML reports)

## Related Files

- `mcts_trader/env/vectorbt_env.py` - Trading environment
- `train_jax_fast.py` - Training script (uses same environment)
