# TASK_682: Real Data Benchmark for Easy Mode

**Status**: IMPLEMENTED
**Priority**: HIGH
**Depends on**: TASK_666 (easy mode), TASK_681 (oracle k=4), TASK_676 (V3 encoder)
**Created**: 2026-02-19

---

## Context

Training is running at 60%+ balanced agreement on synthetic/Bella data (V3+k=4+Bella, 2500 epochs). But we have **zero visibility** into how the model performs on real market data. TASK_666 spec requires "positive alpha on 4/6 benchmark periods" as Phase 1 success criteria. The 6 benchmark periods are defined in `config/benchmark_periods.json` but never evaluated against.

The model needs to prove it can generalize from synthetic/Bella training to real market regimes (bull, bear, sideways, crash, euphoria, low-vol).

---

## Approach

Create `tools/benchmark_easy.py` — a standalone benchmark tool that:
1. Loads real prices from DB for each benchmark period
2. Runs oracle + model on those prices
3. Computes agreement, P&L, alpha, Sharpe per period
4. Prints a formatted report + optional JSON output

Reuse as much as possible from `tools/benchmark_oracle_vs_model.py` (P&L simulation, metrics, model loading) and `mcts_trader/data/history.py` (DB loading with RAM cache).

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `tools/benchmark_easy.py` | **NEW** | Core benchmark: load DB prices, run eval, report |
| `train_easy.py` | **MODIFY** | Add `--benchmark` flag for post-training eval |
| `config/benchmark_periods.json` | READ ONLY | Period definitions (6 periods, 7 days each) |

---

## Implementation Plan

### 1. `tools/benchmark_easy.py` — Core Benchmark

#### 1a. `load_benchmark_prices(period, db_path, n_assets)` → `(T, n_assets)` close prices

```python
def load_benchmark_prices(period: dict, db_path: str = "trading_data.db",
                          n_assets: int = 64) -> Tuple[np.ndarray, List[str]]:
    """Load real close prices for a benchmark period from DB.

    Uses HistoricalMarket with date filtering + RAM cache.

    Returns:
        prices: (T, n_assets) close prices, forward-filled
        symbols: list of symbol names (for reporting)
    """
```

Key logic:
- Create `HistoricalMarket(db_path, n_assets, start_date=period['start_date'], end_date=period['end_date'], use_ram_cache=True)`
- Extract close prices from `cache_data[:, :, 3]` — this gives `(T, n_assets)` directly
- Return prices array + symbol list from `market.assets`
- Log actual asset count and timestamp count (some periods may have fewer than 64 assets)

#### 1b. `run_period_eval(prices, symbols, model, params, config, encoder_config)` → `PeriodResult`

```python
@dataclass
class PeriodResult:
    name: str
    n_steps: int
    n_assets: int
    # Agreement
    balanced_accuracy: float       # 0-1
    agree_trade: float             # 0-1
    agree_hold: float              # 0-1
    # P&L
    oracle_pnl: float             # Total oracle P&L
    model_pnl: float              # Total model P&L
    pnl_capture_ratio: float      # model/oracle
    # Alpha
    buy_hold_return: float        # Buy-and-hold benchmark return
    model_alpha: float            # model_pnl - buy_hold_return
    oracle_alpha: float           # oracle_pnl - buy_hold_return
    # Risk
    model_sharpe: float           # Annualized Sharpe ratio
    max_drawdown: float           # Maximum drawdown %
    # Trading
    oracle_trades: int
    model_trades: int
    # Baseline
    random_pnl: float              # Average P&L of random policy (100 trials)
    # Value head
    value_mse: float               # MSE of model value vs oracle cumulative P&L
```

Key logic:
1. Compute log returns: `returns = log(prices[1:] / max(prices[:-1], 1e-8))`
2. Run oracle with **all config params** (must match training teacher):
   ```python
   greedy_lookahead_path(returns,
       k=config.oracle_k,                             # 4
       lookahead=config.oracle_lookahead,              # 20 (NOT default 5!)
       transaction_cost=config.transaction_cost,       # 0.001
       min_alpha_threshold=config.min_alpha_threshold)  # 0.0005
   ```
3. Convert oracle selections → actions: `oracle_selections_to_actions()` — reuse from `benchmark_oracle_vs_model.py`
4. **Batched encode+infer loop** (encode and infer in same batch to cap VRAM):
   ```python
   BATCH = 512
   all_actions = []
   for start in range(0, n_windows, BATCH):
       batch_starts = window_starts[start:start+BATCH]
       obs = jit_batch_encode_temporal(prices, batch_starts, encoder_config)
       output = model.apply(params, obs, mode='initial_inference')
       actions = np.argmax(output.policy_logits, axis=-1)  # (B, 64) from (B, 320)
       all_actions.append(actions)
   model_actions = np.concatenate(all_actions, axis=0)
   ```
5. **Temporal alignment** (critical — off-by-one = silent failure):
   - `window_starts[i] = i` → obs[i] covers `prices[i:i+64]`
   - Action from obs[i] applies to `returns[i+63]` (the return AFTER the last observed price)
   - Oracle selections and model actions must be aligned to the SAME return indices
   - Assert: `len(model_actions) == len(oracle_actions)` and both index into `returns[63:]`
6. Compute metrics using existing functions:
   - `simulate_oracle_pnl()` and `simulate_model_pnl()` from benchmark_oracle_vs_model.py
   - Balanced accuracy from comparing oracle_actions vs model_actions
7. Compute Sharpe: from per-step model P&L returns, `sharpe = mean(step_returns) / std(step_returns) * sqrt(525960)` (1-min candles, 525960 min/year)
8. Compute buy-and-hold: equal-weight portfolio return = `mean(prices[-1] / prices[window_size]) - 1`
9. Alpha = model cumulative P&L - buy-and-hold return
10. Max drawdown from cumulative P&L curve
11. Random baseline: `evaluate_random_policy(returns_aligned, ...)`
12. Value head MSE: compare model's value output to oracle cumulative P&L at each step. `model.apply()` returns both policy logits and value — compare value to oracle's `cumulative_pnl[i+63]` for each window i.

**P&L normalization note**: Oracle P&L divides by k (`np.sum(abs_returns * mask) / k_count`), measuring per-asset opportunity. Model P&L sums directional returns across all positioned assets (`np.sum(position * returns)`). These are intentionally different measures — `pnl_capture_ratio` compares realized vs opportunity, not like-for-like absolute returns.

#### 1c. `run_full_benchmark(checkpoint_path, periods, db_path)` → `List[PeriodResult]`

Orchestrator that:
1. Loads checkpoint + creates model (reuse `load_checkpoint`, `create_model` from benchmark_oracle_vs_model.py)
2. Loads `config/benchmark_periods.json`
3. For each period: `load_benchmark_prices()` → `run_period_eval()`
4. Prints formatted summary table
5. Returns list of results

#### 1d. `print_benchmark_report(results)` — formatted output

```
╔══════════════════════════════════════════════════════════════════╗
║              Easy Mode Benchmark — Real Market Data              ║
╠══════════════════════════════════════════════════════════════════╣
║ Checkpoint: jax_ckpt_easy/easy_large/phase1_best.pkl            ║
║ Model: 14.59M params, V3 encoder, k=4 oracle                   ║
╚══════════════════════════════════════════════════════════════════╝

┌──────────────┬────────┬────────┬──────────┬──────────┬──────────┬─────────┬────────┐
│ Period       │ Bal.Acc│ Trade% │ OraclePnL│ ModelPnL │ RandomPnL│  Alpha  │ Sharpe │
├──────────────┼────────┼────────┼──────────┼──────────┼──────────┼─────────┼────────┤
│ bull         │ 58.3%  │ 21.4%  │  +3.21%  │  +1.87%  │  -0.32%  │ +0.42%  │  1.23  │
│ bear         │ 55.1%  │ 18.7%  │  +4.56%  │  +0.82%  │  -0.45%  │ +1.15%  │  0.67  │
│ sideways     │ 52.4%  │ 15.2%  │  +0.89%  │  -0.12%  │  -0.28%  │ -0.07%  │ -0.34  │
│ ftx_crash    │ 49.8%  │ 12.1%  │  +5.67%  │  -0.45%  │  -0.51%  │ +0.88%  │ -0.22  │
│ etf_approval │ 57.2%  │ 19.8%  │  +2.34%  │  +1.23%  │  -0.18%  │ +0.31%  │  0.89  │
│ boring_week  │ 51.1%  │ 11.3%  │  +0.45%  │  +0.05%  │  -0.22%  │ +0.02%  │  0.11  │
├──────────────┼────────┼────────┼──────────┼──────────┼──────────┼─────────┼────────┤
│ AVERAGE      │ 54.0%  │ 16.4%  │  +2.85%  │  +0.57%  │  -0.33%  │ +0.45%  │  0.39  │
│ vs Random    │                                        │  6/6 ✓  │         │        │
│ Positive α   │                                                  │  4/6    │        │
└──────────────┴────────┴────────┴──────────┴──────────┴──────────┴─────────┴────────┘

TASK_666 Success: Positive alpha on 4/6 periods ✓ | Beats random on 6/6 ✓
```

#### 1e. CLI

```bash
# Full benchmark (all 6 periods)
python tools/benchmark_easy.py --checkpoint jax_ckpt_easy/easy_large/phase1_best.pkl

# Specific periods
python tools/benchmark_easy.py --checkpoint ... --periods bull bear

# JSON output for programmatic use
python tools/benchmark_easy.py --checkpoint ... --json results.json

# Custom DB path
python tools/benchmark_easy.py --checkpoint ... --db trading_data.db
```

---

### 2. `train_easy.py` — Integration

Add `--benchmark` flag:

```python
parser.add_argument('--benchmark', action='store_true',
                    help='Run benchmark on real data after training completes')
```

At end of training loop (after epoch loop), if `--benchmark`:
```python
if args.benchmark:
    from tools.benchmark_easy import run_full_benchmark
    results = run_full_benchmark(best_checkpoint_path, db_path="trading_data.db")
```

---

### 3. Key Reuse Points

| Function | Source | Reuse |
|----------|--------|-------|
| `HistoricalMarket` | `mcts_trader/data/history.py` | Direct — load prices with date filter |
| `greedy_lookahead_path` | `mcts_trader/jax_agent/path_solver.py` | Direct — compute oracle on real returns |
| `oracle_selections_to_actions` | `tools/benchmark_oracle_vs_model.py` | Direct — convert oracle → action indices |
| `simulate_oracle_pnl` | `tools/benchmark_oracle_vs_model.py` | Direct — oracle P&L |
| `simulate_model_pnl` | `tools/benchmark_oracle_vs_model.py` | Direct — model P&L with position tracking |
| `compute_balanced_metrics` | `tools/benchmark_oracle_vs_model.py` | Direct — balanced accuracy, F1 |
| `jit_batch_encode_temporal` | `mcts_trader/env/jax_encoder.py` | Direct — encode real prices to observations |
| `load_checkpoint`, `create_model` | `tools/benchmark_oracle_vs_model.py` | Direct — model setup |
| `EncoderConfig` | `mcts_trader/env/jax_encoder.py` | Direct — encoder config |
| `get_large_config` | `easy_config.py` | Direct — model config |

---

### 4. Asset Mapping Strategy

**Problem**: DB has varying asset counts per period (587-657 per benchmark_periods.json), model expects exactly 64.

**Solution**:
- `HistoricalMarket(n_assets=64)` already selects top 64 by data availability for the date range
- The model is trained on synthetic/Bella data with arbitrary asset indices — it doesn't learn asset identity, it learns price dynamics
- So using top-64 from each period is fine — asset columns are interchangeable
- Log which assets were selected for transparency

---

### 5. Sharpe Ratio Computation

```python
# Per-step model returns from position tracking
step_returns = []  # append each step's P&L
sharpe = np.mean(step_returns) / max(np.std(step_returns), 1e-10) * np.sqrt(525960)
# 525960 = minutes per year (365.25 * 24 * 60)
```

Need to modify `simulate_model_pnl` to also return per-step returns array (not just total). Create a wrapper: `simulate_model_pnl_detailed()` that returns `(total_pnl, trade_count, step_returns_array)`.

---

### 6. Random Baseline (Hold-Biased)

```python
def evaluate_random_policy(returns: np.ndarray, n_assets: int = 64,
                           transaction_cost: float = 0.001,
                           n_trials: int = 100, seed: int = 42) -> float:
    """Average P&L of hold-biased random policy over n_trials.

    Distribution: 90% hold, 5% buy, 5% sell — mimics a "noise trader"
    that occasionally acts but mostly holds. Uniform random would be
    annihilated by transaction costs (changing positions every minute)
    and provide zero signal.
    """
    rng = np.random.RandomState(seed)
    pnls = []
    for _ in range(n_trials):
        r = rng.random(size=(returns.shape[0], n_assets))
        actions = np.ones((returns.shape[0], n_assets), dtype=np.int32)  # default HOLD
        actions[r < 0.05] = 0   # 5% sell
        actions[r > 0.95] = 2   # 5% buy
        pnl, _ = simulate_model_pnl(actions, returns, transaction_cost)
        pnls.append(pnl)
    return float(np.mean(pnls))
```

Sanity check — if model can't beat a noise trader, something is fundamentally wrong.

---

### 7. Max Drawdown

```python
cumulative = np.cumsum(step_returns)
running_max = np.maximum.accumulate(cumulative)
drawdown = running_max - cumulative
max_drawdown = np.max(drawdown)
```

---

## Data Flow

```
config/benchmark_periods.json
    ↓ (period name, start_date, end_date)
HistoricalMarket(start_date, end_date, n_assets=64)
    ↓ (cache_data[:, :, 3] = close prices)
prices: np.ndarray (T, 64)
    ↓
┌─────────────────────┐     ┌─────────────────────────┐
│ log returns          │     │ jit_batch_encode_temporal│
│ (T-1, 64)           │     │ → (T-64, 27, 64, 64)    │
└──────┬──────────────┘     └──────────┬──────────────┘
       ↓                               ↓
greedy_lookahead_path()         model.apply(obs)
       ↓                               ↓
oracle_selections (T, 64)       policy_logits (T, 320)
oracle_cum_pnl (T,)             → argmax → actions (T, 64)
       ↓                               ↓
simulate_oracle_pnl()           simulate_model_pnl_detailed()
       ↓                               ↓
oracle P&L, trades              model P&L, trades, step_returns
       ↓                               ↓
       └──────── compare ──────────────┘
                    ↓
         PeriodResult (agreement, P&L, alpha, Sharpe, drawdown)
```

---

## DB Schema Reference

```sql
-- OHLCV table (main price data)
CREATE TABLE ohlcv (
    asset_id INTEGER,
    timestamp INTEGER NOT NULL,    -- Unix seconds
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    trade_count INTEGER,
    PRIMARY KEY (asset_id, timestamp)
);

-- Assets table
CREATE TABLE assets (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,  -- e.g., "BTC/USD"
    kraken_symbol VARCHAR(20),
    is_active BOOLEAN DEFAULT 1
);
```

HistoricalMarket RAM cache: `cache_data[timestamp_idx, asset_idx, channel]`
- Channel 0=open, 1=high, 2=low, 3=close, 4=volume
- Forward-filled for gaps

---

## Verification

1. Run benchmark on current best checkpoint:
   ```bash
   python tools/benchmark_easy.py --checkpoint /mnt/scratch/trader_checkpoints/jax_ckpt_easy/easy_large/phase1_best.pkl
   ```
2. Verify all 6 periods load from DB (check asset counts, timestamp counts)
3. Verify oracle P&L is positive on all periods (sanity check — oracle should always profit with hindsight)
4. Compare agreement on real data vs training agreement (~60%) — expect lower on real data
5. Check that Sharpe and alpha numbers are reasonable (not NaN, not astronomical)

---

## Success Criteria (from TASK_666)

- [ ] Oracle agreement > 75% on real benchmark data
- [ ] Value MSE < 0.1 on real benchmark data
- [ ] Positive alpha on at least 4/6 benchmark periods
- [ ] Model beats hold-biased random policy on all 6 periods
- [ ] Trade density > 1% on all periods (non-turtle check)

---

## Adversarial Review (2026-02-19)

8 CRITs raised. 2 partially valid, 6 rejected. Analysis below.

### Rejected (6/8)

| CRIT | Claim | Why Invalid |
|------|-------|-------------|
| **CRIT-1** | Missing transaction costs | `easy_config.py:60` has `transaction_cost=0.001`. Flows: config → `greedy_lookahead_path` (deducted in lookahead scoring + P&L) → `simulate_oracle_pnl` → `simulate_model_pnl`. Both sides already deduct costs. |
| **CRIT-2** | Oracle gets full-period hindsight (marked "CRITICAL") | **Wrong.** `greedy_lookahead_path` uses `lookahead=5` (default, `path_solver.py:42`). At each step t, oracle only sees `returns[t:t+5]` via `jax.lax.dynamic_slice`. NOT the entire future. Reviewer misread the algorithm. |
| **CRIT-3** | Forward-fill creates false overnight signals | **Wrong domain.** This is 24/7 crypto (Kraken). No market close, no overnight gaps. Forward-fill is for individual asset data gaps (thin liquidity altcoins with missing 1-min candles). Reviewer's "BTC closes Friday, opens Sunday" example doesn't apply to crypto. |
| **CRIT-4** | Training/test data leak risk | **Impossible by construction.** Model trains on synthetic + Bella data, never on real historical prices. Zero overlap possible. |
| **CRIT-7** | No statistical significance tests | 6 hand-picked regime periods, not random samples. Classical significance testing doesn't apply. With N=6 regimes, bootstrap CIs add complexity without value. Future v2 consideration. |
| **CRIT-8** | Missing per-regime analysis | The 6 periods ARE the per-regime analysis — specifically chosen as bull, bear, sideways, crash, euphoria, boring. The report table IS per-regime. |

### Accepted — Minor (2/8)

**CRIT-5 (Random baseline)**: Add random policy baseline as sanity check. Cheap to implement — random actions → `simulate_model_pnl` → compare to model. Momentum/mean-reversion are scope creep for v1.

**CRIT-6 (Document candle frequency)**: DB stores 1-minute candles (Kraken API `interval=1`, `benchmark_periods.json`: "1440 steps/day"). The `sqrt(525960)` annualization is correct for 1-min data. Add explicit comment documenting this assumption.

### GLM Changes Applied

1. Add `random_pnl` field to `PeriodResult` — hold-biased random baseline per period
2. Add `evaluate_random_policy(returns, n_trials=100)` function — hold-biased (90/5/5) noise trader
3. Document candle frequency assumption in Sharpe computation comment
4. Add "Model beats random on all periods" to success criteria

### GLM Conclusion

The reviewer's "CRITICAL" finding (CRIT-2) was based on misunderstanding `greedy_lookahead_path` — the oracle is myopic with 5-step lookahead, not omniscient. Rest of plan stands.

---

## Adversarial Review #2: Gemini (2026-02-19)

4 CRITs + 1 MED + 1 LOW + 2 questions. Higher hit rate than GLM — more practical, less theoretical.

### Accepted (4/4 CRITs valid)

| CRIT | Issue | Fix |
|------|-------|-----|
| **CRIT-1** | Random baseline destroyed by transaction costs (uniform random changes position every minute) | Changed to hold-biased (90% hold, 5% buy, 5% sell). Uniform random was a useless floor. |
| **CRIT-2** | Encoding all 10k windows at once wastes VRAM (4.4GB peak) | Changed to batched encode+infer loop (512 windows per batch). Constant VRAM regardless of period length. |
| **CRIT-3** | Off-by-one temporal alignment risk | Added explicit index mapping docs: obs[i] covers prices[i:i+64], action applies to returns[i+63]. |
| **CRIT-4** | Oracle uses default lookahead=5 but training uses lookahead=20 from EasyConfig | Now explicitly passes all config params: `k`, `lookahead`, `transaction_cost`, `min_alpha_threshold`. |

### Accepted — Minor

| Issue | Fix |
|-------|-----|
| **MED-1** (asset underflow) | Low risk (587-657 DB assets per period, requesting 64). If <64: duplicate existing assets to fill (NOT zero-pad — a flat-price column is a dangerous artificial constraint the model never sees in training). |
| **Q1** (turtle threshold) | Added: "trade density > 1% on all periods" to success criteria. |
| **Q2** (P&L normalization) | Documented: oracle divides by k (opportunity), model sums positions (realized). Intentionally different. |

### Rejected

| Issue | Why |
|-------|-----|
| **LOW-1** (Sharpe frequency) | Already addressed in GLM review — 1-min candles confirmed, sqrt(525960) correct. |

### Gemini vs GLM Comparison

- GLM: 8 CRITs, 2 partially valid (25% hit rate). Strong methodology, weak domain knowledge.
- Gemini: 4 CRITs, 4 valid (100% hit rate). Practical engineering focus, caught real implementation risks.
- GLM found the *right categories* (baselines, bias, costs) but wrong specifics. Gemini found *actual bugs* (parameter mismatch, OOM, off-by-one).

---

## Notes

- Benchmark periods are 7 days each = ~10,080 minute candles (1-min from Kraken, 24/7 crypto)
- With window_size=64, that's ~10,016 valid inference windows per period
- Batch inference at 512 windows per batch = ~20 batches per period
- Total benchmark time estimate: ~5 minutes (6 periods × ~50s each)
- The 6 benchmark periods were specifically chosen to NOT overlap with any training data
- No data leak possible: model trains on synthetic/Bella, never real historical prices
- Sharpe annualization: `sqrt(525960)` is correct for 1-minute candle data (525960 = 365.25 × 24 × 60)
