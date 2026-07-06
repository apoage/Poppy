# visualize_synthetic.py

**Synthetic Data Level Visualization**

## Purpose

Generate preview plots for all synthetic data levels (0-3) to visualize market patterns used in curriculum training.

## Usage

```bash
python tools/visualize_synthetic.py
```

## Output

Creates PNG files in current directory:
- `synthetic_level_0_preview.png` - Sine waves (basic patterns)
- `synthetic_level_1_preview.png` - Lagged pairs (leader/follower)
- `synthetic_level_2_preview.png` - Dip/crash patterns (buy the dip)
- `synthetic_level_3_preview.png` - Noisy trends (realistic noise)

## Synthetic Levels

| Level | Pattern | Training Goal |
|-------|---------|---------------|
| 0 | Pure sine waves | Learn basic buy low/sell high |
| 1 | Lagged pairs | Exploit leader/follower correlation |
| 2 | Dip/crash recovery | Learn mean reversion |
| 3 | Noisy trends | Handle realistic market noise |

## Example

Each output file shows 4 sample assets with 200 steps of price history.

## Dependencies

- matplotlib
- SyntheticMarket (mcts_trader.data.synthetic_market)

## Related Files

- `mcts_trader/data/synthetic_market.py` - Synthetic data generator
- `train_jax_fast.py` - Uses synthetic levels via `--synthetic-level`
