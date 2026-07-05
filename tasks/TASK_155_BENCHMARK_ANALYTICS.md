# TASK_155: Benchmark Analytics - Oracle vs Model Profitability

**Status:** COMPLETE
**Completed:** 2026-02-12
**Priority:** HIGH (Diagnostic tool for training plateau)

**Implementation:** `tools/benchmark_oracle_vs_model.py`

**Objective:** Build comprehensive benchmark tool comparing oracle (hindsight-optimal) vs trained model performance, with profitability tracking and book status.

---

## 1. Problem Statement

Training plateaued at ~60% agreement. Need diagnostics to understand:
1. Is 60% agreement profitable or not?
2. Where does the model fail? (HOLD vs BUY vs SELL)
3. What's the oracle's achievable P&L baseline?
4. Is model's book positive or negative?

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Data Gen    │───▶│   Oracle     │───▶│  Oracle      │      │
│  │  (Bella)     │    │  path_solver │    │  P&L Sim     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                       │               │
│         │                                       ▼               │
│         │                              ┌──────────────┐         │
│         │                              │  COMPARISON  │         │
│         │                              │  - Agreement │         │
│         │                              │  - Confusion │         │
│         │                              │  - P&L Ratio │         │
│         │                              └──────────────┘         │
│         │                                       ▲               │
│         ▼                                       │               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Obs Builder │───▶│   Model      │───▶│  Model       │      │
│  │  (temporal)  │    │  Inference   │    │  P&L Sim     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Components

### 3.1 Checkpoint Loading

Load e-dojo merged checkpoints:

```python
def load_checkpoint(checkpoint_path: str) -> dict:
    """Load model params from merged e-dojo checkpoint."""
    import pickle
    from pathlib import Path

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.suffix == '.pkl':
        with open(ckpt_path, 'rb') as f:
            data = pickle.load(f)
        return data['params']
    else:
        # Orbax checkpoint (fallback)
        import orbax.checkpoint as ocp
        ckptr = ocp.StandardCheckpointer()
        return ckptr.restore(str(ckpt_path))
```

### 3.2 Data Generation

Reuse existing Bella generator or synthetic fallback:

```python
def generate_validation_data(config, n_episodes, episode_length, seed):
    """Generate price data for benchmarking."""
    from diff_teacher.generation.generator import MarketGenerator

    generator = MarketGenerator.load("bella_checkpoints/bella_cnn_latest.pkl")

    prices_batch = []
    key = jax.random.PRNGKey(seed)

    for _ in range(n_episodes):
        key, subkey = jax.random.split(key)
        prices = generator.generate(subkey, length=episode_length + config.window_size, n_assets=64)
        prices_batch.append(prices)

    return np.stack(prices_batch)  # (N, T+window, n_assets)
```

### 3.3 Observation Encoding

Use actual encoder from codebase (NOT `build_temporal_observation` which doesn't exist):

```python
from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_temporal

def setup_encoder(config):
    """Create encoder config for observation building."""
    encoder_config = EncoderConfig(
        n_assets=config.n_assets,
        window_size=config.window_size,
    )
    return encoder_config

def encode_observations(prices, encoder_config):
    """Encode all observations at once (batched).

    Args:
        prices: (T+window, n_assets) price array

    Returns:
        observations: (T, 27, 64, 64) encoded temporal observations
    """
    T = prices.shape[0] - encoder_config.window_size
    window_starts = jnp.arange(T)
    observations = jit_batch_encode_temporal(prices, window_starts, encoder_config)
    return observations  # (T, 27, 64, 64)
```

### 3.4 Oracle Execution (Batched)

Use batched oracle for 10x speedup:

```python
from mcts_trader.jax_agent.path_solver import batched_greedy_lookahead_path, OptimalPath

def run_oracle_batched(returns_batch: jnp.ndarray, config) -> OptimalPath:
    """Run hindsight-optimal path solver on all episodes in parallel.

    Args:
        returns_batch: (N, T, n_assets) batched log returns

    Returns:
        OptimalPath with batched arrays
    """
    return batched_greedy_lookahead_path(
        returns_batch,
        k=config.oracle_k,                    # NOT k_positions
        lookahead=config.oracle_lookahead,    # NOT lookahead
        transaction_cost=config.transaction_cost,
        min_alpha_threshold=config.min_alpha_threshold,
    )
    # Returns batched: selections (N, T, n_assets), cumulative_pnl (N, T), etc.
```

**Note:** Oracle only outputs actions {0=SELL, 1=HOLD, 2=BUY}. LimitSell(3) and LimitBuy(4) are not used by oracle.

### 3.5 Model Inference (Batched)

Efficient batched inference instead of per-timestep:

```python
def run_model_batched(params, observations_batch, model):
    """Batched model inference on all observations at once.

    Args:
        params: Model parameters
        observations_batch: (N, T, 27, 64, 64) encoded observations
        model: MuZeroNetworks instance

    Returns:
        actions: (N, T, 64) action per asset
    """
    B, T = observations_batch.shape[:2]

    # Flatten batch and time dimensions
    flat_obs = observations_batch.reshape(B * T, 27, 64, 64)

    # Single forward pass
    output = model.apply({'params': params}, flat_obs, training=False)
    logits = output['policy_logits']  # (B*T, 320)

    # Per-asset argmax (reshape to 64×5, argmax per row)
    probs = jax.nn.softmax(logits).reshape(B * T, 64, 5)
    actions = jnp.argmax(probs, axis=-1)  # (B*T, 64)

    return actions.reshape(B, T, 64)
```

### 3.6 Trading Simulation (Oracle-Aligned)

P&L calculation aligned with oracle's `abs(return)` method:

```python
from dataclasses import dataclass, field

@dataclass
class TradingBook:
    """Track positions and P&L - aligned with oracle calculation."""
    total_pnl: float = 0.0
    trade_count: int = 0
    step_pnls: list = field(default_factory=list)

    def step_oracle_style(self, selections_mask, returns, transaction_cost=0.001):
        """Oracle-style P&L: capture abs(return) for selected assets.

        This matches path_solver.py:131:
            step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask) / k_count

        Oracle trades optimally in both directions, so we use abs(return).
        """
        k_count = max(np.sum(selections_mask), 1)
        step_pnl = np.sum(np.abs(returns) * selections_mask) / k_count

        # Transaction costs (count changes from previous mask)
        step_pnl -= transaction_cost * np.sum(selections_mask > 0.5)

        self.total_pnl += step_pnl
        self.step_pnls.append(step_pnl)
        return step_pnl

    def step_model_style(self, actions, returns, prev_actions=None, transaction_cost=0.001):
        """Model-style P&L: directional based on action.

        BUY(2) → long → +return if positive
        SELL(0) → short → +return if negative (i.e., -return)
        HOLD(1) → no position → 0
        """
        pnl = 0.0
        n_trades = 0

        for i in range(len(actions)):
            if actions[i] == 2:  # BUY/Long
                pnl += returns[i]
                if prev_actions is not None and prev_actions[i] != 2:
                    n_trades += 1
            elif actions[i] == 0:  # SELL/Short
                pnl -= returns[i]  # Profit when price goes down
                if prev_actions is not None and prev_actions[i] != 0:
                    n_trades += 1

        pnl -= n_trades * transaction_cost
        self.total_pnl += pnl
        self.trade_count += n_trades
        self.step_pnls.append(pnl)
        return pnl
```

### 3.7 Metrics Collection

```python
@dataclass
class BenchmarkMetrics:
    """Collected benchmark metrics."""
    # Agreement
    overall_agreement: float          # Total action match %
    per_action_agreement: dict        # {SELL: %, HOLD: %, BUY: %}

    # Distributions
    oracle_distribution: dict         # {SELL: %, HOLD: %, BUY: %}
    model_distribution: dict          # {SELL: %, HOLD: %, BUY: %}

    # Confusion matrix
    confusion: dict                   # {oracle_action: {model_action: count}}

    # Profitability
    oracle_total_pnl: float
    oracle_avg_episode_pnl: float
    oracle_positive_episodes: int

    model_total_pnl: float
    model_avg_episode_pnl: float
    model_positive_episodes: int

    # Capture ratio
    pnl_capture_ratio: float          # model_pnl / oracle_pnl

    # Temporal
    agreement_by_timestep: list       # Agreement at each step (drift detection)
```

---

## 4. Output Report Format

```
======================================================================
BENCHMARK REPORT: Oracle vs Model Performance
======================================================================

Dataset: 100 episodes × 400 steps

--- AGREEMENT METRICS ---
Overall Agreement: 60.4%

Per-Action Agreement:
  Action    Oracle %    Model %     When Oracle Says X, Model Agrees
  ----------------------------------------------------------------
  SELL      18.2%       12.1%       42.3%
  HOLD      63.5%       78.2%       89.1%
  BUY       18.3%       9.7%        38.7%

--- CONFUSION MATRIX ---
(What model does when oracle says X)

  Oracle↓ Model→     SELL       HOLD        BUY
  -----------------------------------------------
  SELL               42.3%      51.2%       6.5%
  HOLD               3.1%       89.1%       7.8%
  BUY                8.2%       53.1%       38.7%

--- PROFITABILITY ---

  Metric                          Oracle          Model
  --------------------------------------------------------
  Total P&L                       +12.847%        +3.241%
  Avg P&L per Episode             +0.128%         +0.032%
  Positive Episodes               78/100          54/100
  Total Trades                    15,234          8,412

  P&L Capture Ratio: 25.2%
  (Model captures 25.2% of oracle's returns)

--- BOOK STATUS ---
  Oracle Book: POSITIVE (+12.847%)
  Model Book:  POSITIVE (+3.241%)

--- TEMPORAL AGREEMENT ---
  Steps 1-100:    62.1%
  Steps 101-200:  60.8%
  Steps 201-300:  59.4%
  Steps 301-400:  58.9%

--- BALANCED ACCURACY ---
  Macro-F1:             0.412
  Balanced Accuracy:    0.567  (weighted by inverse class frequency)
  Per-Class F1:
    SELL:  0.38
    HOLD:  0.71
    BUY:   0.35

--- TRADE ANALYSIS ---
  Oracle Trades Made:       1,523 / 40,000 steps (3.8%)
  Model Trades Made:        8,412 / 40,000 steps (21.0%)
  Trade Precision:          23.1%  (correct trades / model trades)
  Trade Recall:             84.2%  (correct trades / oracle trades)

  INTERPRETATION:
    - Model over-trades (21% vs oracle 3.8%)
    - But catches 84% of oracle's trades (high recall)
    - Many false positives (low precision)

--- DIAGNOSIS ---
  WARNING: Model is too conservative (HOLD: 78.2% vs oracle 63.5%)
  WARNING: Model struggles with SELL decisions (42.3% agreement)
  WARNING: Model struggles with BUY decisions (38.7% agreement)
  ISSUE: Model defaults to HOLD, missing trading opportunities

======================================================================
```

**Balanced Accuracy Calculation:**

```python
from sklearn.metrics import balanced_accuracy_score, f1_score

def compute_balanced_metrics(oracle_actions, model_actions):
    """Compute class-balanced metrics to handle imbalance."""
    oracle_flat = oracle_actions.flatten()
    model_flat = model_actions.flatten()

    return {
        'balanced_accuracy': balanced_accuracy_score(oracle_flat, model_flat),
        'macro_f1': f1_score(oracle_flat, model_flat, average='macro'),
        'per_class_f1': {
            'SELL': f1_score(oracle_flat == 0, model_flat == 0),
            'HOLD': f1_score(oracle_flat == 1, model_flat == 1),
            'BUY': f1_score(oracle_flat == 2, model_flat == 2),
        }
    }

def compute_trade_metrics(oracle_actions, model_actions):
    """Trade-level precision/recall."""
    # "Trade" = BUY or SELL (not HOLD)
    oracle_trades = (oracle_actions != 1)  # Not HOLD
    model_trades = (model_actions != 1)

    true_positives = np.sum(oracle_trades & model_trades & (oracle_actions == model_actions))
    oracle_trade_count = np.sum(oracle_trades)
    model_trade_count = np.sum(model_trades)

    precision = true_positives / max(model_trade_count, 1)
    recall = true_positives / max(oracle_trade_count, 1)

    return {
        'oracle_trade_count': int(oracle_trade_count),
        'model_trade_count': int(model_trade_count),
        'oracle_trade_rate': oracle_trade_count / oracle_actions.size,
        'model_trade_rate': model_trade_count / model_actions.size,
        'trade_precision': precision,
        'trade_recall': recall,
        'trade_f1': 2 * precision * recall / max(precision + recall, 1e-10),
    }
```

---

## 5. Implementation Files

| File | Purpose |
|------|---------|
| `tools/benchmark_oracle_vs_model.py` | Main benchmark script |
| `tools/benchmark_visualize.py` | Optional: matplotlib plots |

---

## 6. CLI Interface

```bash
# Basic usage
python tools/benchmark_oracle_vs_model.py \
    --checkpoint jax_ckpt_easy_merged/merged_0124_*.pkl

# Full options
python tools/benchmark_oracle_vs_model.py \
    --checkpoint jax_ckpt_easy_merged/merged_0124_*.pkl \
    --episodes 100 \
    --steps 400 \
    --seed 42 \
    --output results/benchmark_20260212.json \
    --verbose
```

---

## 7. Dependencies (Existing)

All components already exist in codebase:

| Component | Location |
|-----------|----------|
| Oracle path solver | `mcts_trader/jax_agent/path_solver.py` |
| TRM model | `mcts_trader/jax_agent/trm.py` |
| Bella generator | `diff_teacher/generation/generator.py` |
| Temporal obs builder | `mcts_trader/env/jax_encoder.py:jit_batch_encode_temporal()` |
| Config | `easy_config.py` |

---

## 8. Enhanced Diagnostics (Review Additions)

### 8.1 Top-K Set Intersection

**Problem:** Hard k=4 cap creates false disagreement. Model picks Asset 12, Oracle picks Asset 5 - both profitable but 0% "agreement".

**Solution:** Measure neighborhood overlap, not exact match.

```python
def topk_intersection(oracle_selections, model_actions, k_oracle=10, k_model=5):
    """
    Check if model's top picks overlap with oracle's wider selection.

    Returns: Intersection ratio (0.0 to 1.0)
    """
    # Oracle's top-10 by absolute return
    oracle_topk = np.argsort(np.abs(oracle_selections))[-k_oracle:]

    # Model's active positions (BUY or SELL, not HOLD)
    model_active = np.where((model_actions == 0) | (model_actions == 2))[0]
    model_topk = model_active[:k_model] if len(model_active) >= k_model else model_active

    # Intersection
    overlap = len(set(oracle_topk) & set(model_topk))
    return overlap / max(len(model_topk), 1)
```

**Metric:** `neighborhood_agreement` - "Model is in the right area even if not exact asset"

### 8.2 Return-Weighted Agreement

**Problem:** High agreement on noise (+0.2% trades), missing big moves (+5% trades).

**Solution:** Weight agreement by return magnitude.

```python
def return_weighted_agreement(oracle_actions, model_actions, returns):
    """
    Agreement weighted by trade importance.

    Formula: Σ(match * |return|) / Σ(|return|)
    """
    matches = (oracle_actions == model_actions).astype(float)
    abs_returns = np.abs(returns)

    # Weight matches by return magnitude
    weighted_matches = np.sum(matches * abs_returns)
    total_weight = np.sum(abs_returns)

    return weighted_matches / max(total_weight, 1e-10)
```

**Metric:** `return_weighted_accuracy` - "Agreement on trades that matter"

### 8.3 P&L Density Analysis

**Problem:** Mean P&L hides distribution shape. Model might have same mean but worse tail.

**Solution:** Compare return distributions.

```python
def pnl_distribution_metrics(oracle_pnls, model_pnls):
    """Compare P&L distribution shapes."""
    return {
        'oracle_mean': np.mean(oracle_pnls),
        'model_mean': np.mean(model_pnls),
        'oracle_std': np.std(oracle_pnls),
        'model_std': np.std(model_pnls),
        'oracle_skew': scipy.stats.skew(oracle_pnls),
        'model_skew': scipy.stats.skew(model_pnls),
        'oracle_p95': np.percentile(oracle_pnls, 95),
        'model_p95': np.percentile(model_pnls, 95),
        'oracle_p5': np.percentile(oracle_pnls, 5),
        'model_p5': np.percentile(model_pnls, 5),
    }
```

**Metric:** Fat-tail similarity - "Does model capture the big moves?"

### 8.4 Branch Comparison Mode

**Problem:** Merged model might be worse than individual branches (destructive merge).

**Solution:** Benchmark all three: Branch_A, Branch_B, Merged.

```bash
# Run comparison
python tools/benchmark_oracle_vs_model.py \
    --checkpoint-a jax_ckpt_easy_A/branch_A_bella/phase1_best.pkl \
    --checkpoint-b jax_ckpt_easy_B/branch_B_bella_inv/phase1_best.pkl \
    --checkpoint-merged jax_ckpt_easy_merged/merged_0124_*.pkl \
    --compare-branches
```

**Output:**
```
--- BRANCH COMPARISON ---
  Metric              Branch A    Branch B    Merged
  ---------------------------------------------------
  Agreement           62.1%       61.8%       60.4%
  P&L Capture         28.3%       27.1%       25.2%
  Positive Episodes   56/100      54/100      52/100

  DIAGNOSIS: Merged < Both branches → FedAvg destroying knowledge
  SUGGESTION: Try TIES merge or increase merge frequency
```

### 8.5 Execution Cost Realism

**Problem:** Fixed `transaction_cost=0.001` ignores slippage.

**Solution:** Integrate realistic execution model.

```python
def realistic_execution_cost(action, price, volume_estimate, is_buy):
    """
    Realistic execution including slippage.

    Based on HistoricalMarket slippage model.
    """
    base_fee = 0.001  # 10 bps

    # Slippage scales with position size and inverse liquidity
    # Simplified: 2-5 bps additional for market orders
    slippage = 0.0002 + 0.0003 * np.random.random()  # 2-5 bps

    total_cost = base_fee + slippage

    # Apply direction
    if is_buy:
        return price * (1 + total_cost)
    else:
        return price * (1 - total_cost)
```

---

## 9. Success Criteria (Implementation)

- [ ] Runs on merged checkpoint without errors
- [ ] Reports overall + per-action agreement
- [ ] Shows confusion matrix
- [ ] Calculates oracle and model P&L
- [ ] Reports book status (positive/negative)
- [ ] Identifies bottleneck (where model fails)
- [ ] Top-K intersection metric (neighborhood agreement)
- [ ] Return-weighted accuracy (big-move capture)
- [ ] Branch comparison mode (merge quality check)
- [ ] P&L distribution analysis (tail behavior)

---

## 10. Expected Insights

The benchmark should answer:

1. **Is 60% agreement enough?**
   - If model book positive → maybe 60% is fine
   - If model book negative → need higher agreement

2. **Where does model fail?**
   - If HOLD agreement high but BUY/SELL low → conservative bias
   - If all low → model not learning patterns

3. **How much alpha is lost?**
   - P&L capture ratio shows efficiency
   - 25% capture = losing 75% of opportunities

4. **Is there temporal drift?**
   - If agreement drops over episode → state tracking issue
   - If consistent → architecture limit

---

## 11. The "Why" Matrix - Findings to Fixes

Comprehensive mapping from benchmark findings to actionable fixes:

| Finding | Diagnosis | Fix |
|---------|-----------|-----|
| **High Agreement / Negative P&L** | Oracle threshold too low | Increase `min_alpha_threshold` from 0.001 to 0.002+ |
| **Low Agreement / Positive P&L** | Model is "selective" | GOOD outcome - model ignores noise, catches signal |
| **High HOLD Agreement only** | Class imbalance | Increase `adaptive_trade_weight` ramp in train_easy.py |
| **Merged < Branch A/B** | Destructive merge | Switch from FedAvg to TIES or Fisher-weighted merge |
| **Model too conservative** | HOLD bias | Weight BUY/SELL loss higher, reduce HOLD reward |
| **Low neighborhood overlap** | Wrong asset selection | Check encoder - not learning cross-asset patterns |
| **Low return-weighted acc** | Missing big moves | Add "magnitude weighting" to loss function |
| **Temporal drift (late steps worse)** | State tracking failure | Check RNN/temporal encoder attention span |
| **High variance in P&L** | Inconsistent behavior | Increase training epochs or reduce learning rate |
| **Branch A >> Branch B** | Inversion hurting | Remove `--invert-regime` or fix inversion logic |
| **Good agreement, bad slippage** | Unrealistic training | Add slippage noise to training data |

---

## 12. Report Output Enhancement

Add these sections to benchmark report:

```
--- ENHANCED METRICS ---
  Neighborhood Agreement (Top-5 in Top-10): 78.3%
  Return-Weighted Accuracy: 71.2%
  Fat-Tail Capture (P95 ratio): 0.82

--- BRANCH COMPARISON ---
  Branch A Agreement: 62.1%  |  P&L: +4.12%
  Branch B Agreement: 61.8%  |  P&L: +3.89%
  Merged Agreement:   60.4%  |  P&L: +3.24%

  DIAGNOSIS: Merge reducing performance by 0.65-0.88%
  → Consider: More frequent merges OR switch to TIES

--- EXECUTION REALISM ---
  Ideal P&L (no slippage):     +3.241%
  Realistic P&L (with slip):   +2.847%
  Slippage Impact:             -0.394% (-12.2%)
```

---

## 13. Next Steps After Benchmark

Based on results:

| Finding | Action |
|---------|--------|
| Model too conservative | Increase BUY/SELL loss weight |
| Low capture but positive | May be acceptable for risk-adjusted |
| Negative model book | Need architecture/training changes |
| Temporal drift | Check RNN/state handling |
| High HOLD agreement only | Class imbalance in training |
| Merged worse than branches | Fix merge strategy |
| Good neighborhood, bad exact | Model learns patterns, needs fine-tuning |

---

## 14. First Benchmark Results (2026-02-12)

**Checkpoint:** `merged_0124` (Epoch 2484, 124 merges)
**Data:** 20 episodes × 200 steps × 64 assets, Synthetic Level 0

```
--- KEY METRICS ---
Overall Agreement:     83.3%  (misleading - oracle 98.4% HOLD)
Balanced Accuracy:     35.2%
Return-Weighted Acc:   82.3%

--- ACTION DISTRIBUTION ---
Oracle: SELL 0.8%, HOLD 98.4%, BUY 0.7%
Model:  SELL 9.2%, HOLD 84.4%, BUY 6.4%

--- TRADE METRICS ---
Oracle Trade Rate:     1.6%
Model Trade Rate:      15.6%
Trade Precision:       1.1%
Trade Recall:          10.8%

--- P&L ---
Oracle P&L:            +97.14%
Model P&L:             +14.94%
Capture Ratio:         15.4%

--- DIAGNOSIS ---
1. Oracle rarely trades (1.6%) - model over-trades (15.6%)
2. When oracle trades, model misses 89% (10.8% recall)
3. Model is HOLD-biased (conservative)
4. POSITIVE book but only 15% of oracle returns
```

**Interpretation:**
- Synthetic Level 0 (sine waves) is very predictable → oracle captures 97% returns
- Model learned to default HOLD, occasionally trades
- 60% training agreement ≠ 60% trading accuracy
- Need to weight SELL/BUY loss higher in training

---

## 15. Technical Corrections Applied

| Section | Original Issue | Fix Applied |
|---------|----------------|-------------|
| §3.1 | Missing checkpoint loading | Added `load_checkpoint()` with pkl/orbax support |
| §3.3 | Referenced non-existent `build_temporal_observation` | Use `jax_encoder.jit_batch_encode_temporal` |
| §3.4 | Used `config.k_positions` | Corrected to `config.oracle_k` |
| §3.4 | Single-episode oracle (slow) | Use `batched_greedy_lookahead_path` (10x faster) |
| §3.5 | Per-timestep model inference (slow) | Batched inference with reshape |
| §3.6 | P&L mismatch with oracle | Aligned with `abs(return)` method from path_solver:131 |
| §4 | Missing balanced metrics | Added Balanced Accuracy, Macro-F1, Per-Class F1 |
| §4 | No trade-level metrics | Added Trade Precision/Recall/F1 |
| §3 | Action mapping undocumented | Noted oracle only uses {0,1,2}, not LimitSell/LimitBuy |

---

## 15. Dependencies Summary

| Component | Actual Location | Purpose |
|-----------|-----------------|---------|
| Encoder | `mcts_trader/env/jax_encoder.py` | `jit_batch_encode_temporal`, `EncoderConfig` |
| Oracle | `mcts_trader/jax_agent/path_solver.py` | `batched_greedy_lookahead_path` |
| Model | `mcts_trader/jax_agent/muzero_model.py` | `MuZeroNetworks` |
| Config | `easy_config.py` | `oracle_k`, `oracle_lookahead`, `window_size` |
| Generator | `diff_teacher/generation/generator.py` | `MarketGenerator.load()` |
