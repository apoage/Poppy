# Session Notes - 2026-01-01

## Summary
Fixed checkpoint compatibility issues that could corrupt long-term training. Created tuning tournament script with live output streaming. Fixed trainer initialization bug. Added market baseline tracking (alpha metric) to properly measure if model beats buy-and-hold.

---

## 1. Tuning Tournament Script

**Location:** `tuning_tournament.py`

Created tournament script to compare stability features across winning hyperparameter configs.

### Structure
```
Round A: 4 Prediction Variations × 3 base configs = 12 runs
  - no_pred:  No prediction loss (baseline)
  - physics:  Physics-only prediction (--physics-only)
  - hybrid:   Physics + neural corrector
  - tree:     Tree search with prediction

Round B: 3 Value Stabilization × 3 configs = 9 runs
  - baseline: Default MSE loss
  - huber:    Huber loss (--value-loss-type huber)
  - popart:   PopArt normalization (--use-popart)

Round C: 3 Warmup Variations × 3 configs = 9 runs
  - none:       No warmup
  - fast:       50 epoch warmup (--sims-warmup 50)
  - slow:       100 epoch warmup (--sims-warmup 100)

Round D: Combined Best = 3 runs
```

### Base Configs (from hyperparameter tournament)
1. `s16_lr10_h2_np` (score=-2.35) - 16 sims, lr=1e-4, h=2, no pred
2. `s16_lr10_h4_p10` (score=-2.73) - 16 sims, lr=1e-4, h=4, pred=1.0
3. `s32_lr10_h2_p3` (score=-3.37) - 32 sims, lr=1e-4, h=2, pred=0.3

### Live Output Streaming Fix
Changed from `subprocess.run(capture_output=True)` to `subprocess.Popen` with line-buffered streaming:
```python
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end='', flush=True)
    output_lines.append(line)
```

---

## 2. Trainer Initialization Bug Fix

### Problem
`AttributeError: 'MuZeroTrainer' object has no attribute 'optimizer'`

### Root Cause
When adding `update_simulations()` method to trainer.py, it was incorrectly placed in the middle of `__init__`, causing optimizer initialization to be inside the new method instead of `__init__`.

### Fix
Moved optimizer, state, and train_step_fn initialization back into `__init__`, placing `update_simulations()` as a separate method after `__init__` completes.

---

## 3. Checkpoint Compatibility Audit (CRITICAL)

### Issues Found

#### Issue 1: PopArt State Not Saved
**Severity:** CRITICAL

The `PopArtNormalizer` running statistics (mean, var, count) were NOT saved to checkpoints:
- On resume: PopArt resets to mean=0, var=1
- First batch applies wrong weight corrections
- Value head weights dramatically rescaled
- Training destabilized

**Fix Applied:**
```python
# In save_checkpoint():
if self.use_popart and self.popart is not None:
    np.savez(popart_path,
        mean=float(self.popart.state.mean),
        var=float(self.popart.state.var),
        count=int(self.popart.state.count),
        ...)

# In load_checkpoint():
data = np.load(popart_path)
self.popart.state = PopArtState(
    mean=jnp.array(data['mean']),
    var=jnp.array(data['var']),
    ...)
```

#### Issue 2: Prediction Head Architecture Mismatch
**Severity:** HIGH

Loading checkpoint without prediction head into prediction-enabled model:
- `latent_to_market` and `price_predictor` weights remain random
- Silent bug: model acts trained but prediction is garbage

**Fix Applied:**
Added architecture compatibility warnings in `load_checkpoint()`:
```python
if not ckpt_pred and self.model.use_prediction_head:
    print("⚠️  WARNING: Checkpoint has NO prediction head, but model has prediction enabled!")
    print("   Prediction weights will be RANDOM (not from checkpoint)")
```

#### Issue 3: Architecture Metadata Saved
**New Behavior:**

Checkpoints now save architecture info for compatibility checking:
```json
{
  "use_prediction_loss": true,
  "use_prediction_head": true,
  "physics_only": false,
  "h_cycles": 4,
  "l_cycles": 6,
  "state_dim": 64
}
```

### What's Safe
- **MCTS simulations**: Runtime parameter, not in model weights
- **Optimizer state**: `--reset-optimizer` flag handles this
- **Replay buffer**: `--clear-buffer` flag handles this
- **physics_only mode change**: Uses same weights, just skips neural corrector

### What Needs Care
| Transition | Risk | Recommendation |
|-----------|------|----------------|
| no_pred → pred | HIGH (random weights) | Train pred from scratch or use pred checkpoint |
| pred → no_pred | LOW (weights ignored) | Safe, just unused capacity |
| h_cycles change | ERROR | Will cause shape mismatch |
| l_cycles change | ERROR | Will cause shape mismatch |
| state_dim change | ERROR | Will cause shape mismatch |
| PopArt on → off | LOW | Statistics not used |
| PopArt off → on | MEDIUM | Starts fresh (shows warning) |

---

## 4. Files Modified

### Modified
- `mcts_trader/jax_agent/trainer.py`
  - Fixed `update_simulations()` method placement (line 187)
  - Added PopArt state save/restore in checkpoints
  - Added architecture compatibility warnings
  - Added metadata fields: use_prediction_head, physics_only, h_cycles, l_cycles, state_dim

- `tuning_tournament.py`
  - Changed subprocess handling for live output streaming
  - Uses Popen with line-buffered stdout

---

## 5. Testing Commands

### Quick training test (verify fix):
```bash
python train_jax_fast.py --synthetic-level=0 --epochs=3 --steps=50 --model-size=small --name=test_fix
```

### Run tuning tournament:
```bash
python tuning_tournament.py --round A
```

### Test checkpoint compatibility warnings:
```bash
# Train without prediction
python train_jax_fast.py --synthetic-level=0 --epochs=5 --name=ckpt_no_pred

# Resume with prediction (should show warning)
python train_jax_fast.py --synthetic-level=0 --epochs=5 --use-prediction-loss --load-checkpoint=jax_ckpt_fast/ckpt_no_pred --name=ckpt_with_pred
```

---

## 6. Adaptive Entropy - Critical Insight

**Observation**: Without `--adaptive-entropy`, policy entropy collapses rapidly even on simple L0 patterns.

**Analysis**: Adaptive entropy is essential for maintaining exploration. Without it, the model converges to suboptimal policies (usually HOLD-dominant) due to value function learning faster than policy exploration.

**Why it's necessary (for now)**:
- Static entropy coef (0.01) insufficient once model starts learning
- Value function learns faster than policy explores
- MCTS with learned values converges to deterministic actions
- Adaptive entropy detects HOLD% > 70% and boosts exploration

**Tournament fix**: Added `--adaptive-entropy` to `tuning_tournament.py` base command.

**Future work**: Investigate why model needs this crutch:
- Is value function overconfident? (try PopArt, Huber)
- Is MCTS too greedy? (try higher temperature, Gumbel noise)
- Is reward signal too sparse? (try denser shaping)

---

## 7. Prediction Head Size Option

Added `--pred-head-size` to control prediction head capacity.

### Usage
```bash
# Small (default) - ~0.5M extra params
python train_jax_fast.py --use-prediction-loss --pred-head-size small

# Medium - ~2M extra params
python train_jax_fast.py --use-prediction-loss --pred-head-size medium

# Large - ~8M extra params
python train_jax_fast.py --use-prediction-loss --pred-head-size large
```

### Size Configurations
| Size | embed_dim | blocks | mlp_dim | n_factors | Est. Params |
|------|-----------|--------|---------|-----------|-------------|
| small | 128 | 2 | 256 | 8 | ~0.5M |
| medium | 256 | 3 | 512 | 16 | ~2M |
| large | 512 | 4 | 1024 | 32 | ~8M |

### Files Modified
- `train_jax_fast.py` - Added `--pred-head-size` CLI arg and size config mapping
- `mcts_trader/jax_agent/muzero_model.py` - Added `pred_embed_dim`, `pred_num_blocks`, `pred_mlp_dim`, `pred_n_factors` params
- `mcts_trader/jax_agent/prediction/market_ode.py` - Added `mlp_dim` param

---

## 8. Market Baseline Tracking (Alpha Metric)

### Problem
Rewards were consistently negative but unclear if model was actually underperforming market or just accumulating per-step noise. With symlog rewards, more steps = more opportunities for negative accumulation even if trading profitably.

### Solution
Added **alpha metric**: `alpha = agent_return - market_return`
- **agent_return**: Cumulative percentage return from trading
- **market_return**: Buy-and-hold benchmark (average price change across assets)
- **alpha > 0**: Model beating market
- **alpha < 0**: Model underperforming market

### Implementation

**Environment tracking (vectorbt_env.py):**
```python
# In reset():
self._initial_prices = np.array([self._current_prices.get(i, 1.0) for i in range(self.n_assets)])
self._cumulative_agent_return = 0.0

# In step():
self._cumulative_agent_return += pct_return

# Market return = average buy-and-hold return
current_prices = np.array([...])
price_returns = (current_prices - self._initial_prices) / self._initial_prices * 100
market_return = float(np.mean(price_returns))
alpha = self._cumulative_agent_return - market_return

info['market_return'] = market_return
info['agent_return'] = self._cumulative_agent_return
info['alpha'] = alpha
```

**Training output (train_jax_fast.py):**
```
Epoch   42 | Reward:   +3.21 | α: +1.52% (Mkt: +2.30%) | ...
```

**CSV logging:**
Added columns: `alpha`, `market_return`, `agent_return`

### Interpretation
- Positive alpha with negative rewards: Model making good decisions despite per-step losses
- Negative alpha with positive rewards: Lucky trades, not systematic edge
- Zero alpha: Model matches market (no skill)

### Files Modified
- `mcts_trader/env/vectorbt_env.py` - Track initial prices, calculate market/agent returns
- `train_jax_fast.py` - Display alpha in output, log to CSV

---

## 9. DS24 Market Baseline Integration

Added market baseline metrics (alpha, market_return, agent_return) to DS24 logging system.

### Changes

**metrics_logger.py:**
- Added columns: `alpha`, `market_return`, `agent_return` to `DS24_COLUMNS`
- Added `log_market_baseline()` method

**training_hooks.py:**
- Added `_log_market_baseline()` method
- Called at Level 1+ (always logged when DS24 enabled)
- Updated level descriptions to reflect market baseline at Level 1

### Updated Level Descriptions
```
Level 0: OFF - No DS24 logging
Level 1: CSV + Market baseline (alpha, market_return, agent_return)
Level 2: Level 1 + TRM intermediates + MCTS planning metrics
Level 3: Level 2 + Physics forward pass
Level 4: Level 3 + Latent space heatmaps + weight monitoring
```

### Usage
Enable with `--ds24-level 1` or higher. Alpha metrics will be logged to DS24 CSV columns.

---

## 10. Current Status

- Tuning tournament restarted (Round A: 12 runs with adaptive entropy)
- Checkpoint compatibility fixes applied
- Prediction head sizing implemented
- Market baseline tracking implemented (environment + train output + CSV)
- DS24 integration for market baseline metrics
- Ready for long-term curriculum training

---

## Next Steps

1. Complete tuning tournament (all rounds)
2. Analyze results for best stability feature combo
3. Run long curriculum with best config
4. Verify checkpoint restoration works correctly across levels
5. Investigate alpha trends - is model learning to beat market or just tracking it?
6. Use DS24 dashboard to visualize alpha over training epochs

---

## 11. Advanced Synthetic Curriculum System (Session 2)

Designed and implemented a complete advanced curriculum system with 10 new synthetic market levels and intelligent checkpoint safety.

### SafeCheckpointManager

**Location:** `mcts_trader/training/safe_checkpoint.py`

Intelligent checkpoint management for hard curriculum levels:

```python
manager = SafeCheckpointManager(
    checkpoint_dir='checkpoints/level_5',
    patience=50,              # Epochs before considering rollback
    rollback_threshold=0.5,   # Performance drop to trigger rollback
    keep_best_n=3,            # Number of best checkpoints to keep
    rolling_window=20         # Window for moving average
)
```

**Key Features:**
- **Save on improvement only**: Uses rolling average, not single-epoch spikes
- **Auto-rollback**: Detects performance degradation and restores best checkpoint
- **Best-N preservation**: Keeps top N checkpoints sorted by metric
- **Golden checkpoint**: First checkpoint saved as ultimate fallback
- **Level-aware settings**: Harder levels = lower patience, more rollbacks allowed

**Rollback Conditions:**
1. Rolling average dropped > `rollback_threshold` from best
2. Waited at least `patience` epochs
3. OR absolute degradation > 2× threshold from level start

### New CLI Arguments (train_jax_fast.py)

```bash
--safe-checkpoints          # Enable SafeCheckpointManager
--rollback-patience 50      # Epochs before rollback considered
--rollback-threshold 0.5    # Performance drop threshold
--max-rollbacks 3           # Maximum rollback attempts per level
--keep-best-n 3             # Checkpoints to keep
--lr-decay-on-rollback 0.5  # LR multiplier on rollback (TODO)
```

### Advanced Synthetic Levels (5-14)

**Location:** `mcts_trader/data/synthetic_market.py`

| Level | Generator | Trading Concept | Mathematical Basis |
|-------|-----------|-----------------|-------------------|
| 5 | pairs_trading | Cointegrated pairs | OU process spread |
| 6 | triangle_arbitrage | Cross-pair mispricing | FX triangles (ABC) |
| 7 | index_arbitrage | Component vs basket | Weighted index + lag |
| 8 | momentum | Trend continuation | 70% continuation prob |
| 9 | mean_reversion | VWAP reversion | OU process |
| 10 | breakout | Consolidation → breakout | Range + expansion |
| 11 | bid_ask_bounce | Spread capture | Bid/ask oscillation |
| 12 | order_imbalance | Volume signals | Imbalance encoding |
| 13 | market_making | Liquidity provision | Noise vs informed |
| 14 | volatility_clustering | GARCH dynamics | GARCH(1,1) |

### Curriculum Config

**Location:** `config/advanced_curriculum.json`

17 stages covering levels 5-14, each with:
- Progressive epochs (300 → 500)
- Progressive steps (500 → 1440)
- Progressive sims (32 → 64)
- Progressive fees (5 → 26 bps)
- Progressive slippage (0 → 5 bps)
- Decreasing thresholds (1.5 → 0.2)
- Per-stage safety settings

**Example stage:**
```json
{
  "stage": "14.2",
  "level": 14,
  "generator": "volatility_clustering",
  "epochs": 500,
  "steps": 1440,
  "sims": 64,
  "fee_bps": 26,
  "slippage_bps": 5,
  "pass_threshold": 0.2,
  "safety": {
    "enabled": true,
    "rollback_patience": 20,
    "rollback_threshold": 0.2,
    "max_rollbacks": 15,
    "keep_best_n": 5
  }
}
```

### dojo_manager.py Integration

Auto-enables safe checkpoints for level >= 5:
```python
if safety_config.get('enabled', False) or level >= 5:
    cmd.append('--safe-checkpoints')
    cmd.extend(['--rollback-patience', str(safety_config.get('rollback_patience', 50))])
    cmd.extend(['--rollback-threshold', str(safety_config.get('rollback_threshold', 0.5))])
    cmd.extend(['--max-rollbacks', str(safety_config.get('max_rollbacks', 5))])
    cmd.extend(['--keep-best-n', str(safety_config.get('keep_best_n', 3))])
```

### Design Documents Created

- `lore/CHECKPOINT_SAFETY_DESIGN.md` - Full SafeCheckpointManager design
- `lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md` - Mathematical foundations for levels 5-14

### Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `mcts_trader/training/__init__.py` | NEW | Module init |
| `mcts_trader/training/safe_checkpoint.py` | NEW | SafeCheckpointManager |
| `mcts_trader/data/synthetic_market.py` | MOD | 10 new generators |
| `train_jax_fast.py` | MOD | Safe checkpoint CLI + integration |
| `dojo_manager.py` | MOD | Safety config passing |
| `config/advanced_curriculum.json` | NEW | 17-stage advanced curriculum |
| `lore/CHECKPOINT_SAFETY_DESIGN.md` | NEW | Design doc |
| `lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md` | NEW | Design doc |

### Usage

```bash
# Single level with safe checkpoints
python train_jax_fast.py --synthetic-level 5 --safe-checkpoints --name pairs_test

# Full advanced curriculum (auto-enables safety)
python dojo_manager.py --curriculum config/advanced_curriculum.json --model-size large --name advanced_v1

# Test a specific generator
python train_jax_fast.py --synthetic-level 14 --epochs 100 --safe-checkpoints --name vol_test
```

---

## 12. Updated Next Steps

1. Complete tuning tournament (all rounds)
2. Analyze results for best stability feature combo
3. Run basic curriculum (levels 0-4) with best config
4. **NEW:** Run advanced curriculum (levels 5-14) with safe checkpoints
5. Monitor rollback frequency - adjust thresholds if too aggressive/passive
6. Validate advanced generators produce learnable patterns

---

## 13. Production Roadmap

### Phase 1: Historic Data Training
Train on real market data until alpha moving averages all exceed 10%:

| Metric | Target | Description |
|--------|--------|-------------|
| α MA30 | > 10% | 30-epoch moving average of alpha |
| α MA50 | > 10% | 50-epoch moving average |
| α MA150 | > 10% | 150-epoch moving average |
| α MA1440 | > 10% | Full day (1440 min) moving average |

**Command:**
```bash
python train_jax_fast.py \
    --db-path trading_data.db \
    --model-size large \
    --safe-checkpoints \
    --name historic_alpha_run
```

**Graduation criteria:** All 4 alpha MAs > 10% sustained

### Phase 2: Paper Trading (Realtime)
Run model in paper trading mode against live market:

| Metric | Target | Duration |
|--------|--------|----------|
| α MA1440 | > 5% | N days continuous |

- No real money at risk
- Validate model generalizes to unseen market conditions
- Monitor for regime changes, black swan handling

### Phase 3: Live Trading (Small Stake)
Initial live deployment:

| Parameter | Value |
|-----------|-------|
| Starting capital | 100 EUR/USD |
| Target profit | 10,000 EUR/USD |
| Risk management | Full TP/SL enabled |

**Success criteria:** Generate 10,000 EUR/USD (100x return)

### Phase 4: Publication & Research
Once profitable:

1. **Publish paper** documenting:
   - TRM+MuZero architecture
   - Curriculum learning approach
   - Safe checkpoint system
   - Real trading results

2. **Train variants:**
   - Different model sizes
   - Different asset classes
   - Multi-exchange deployment

3. **Public challenges:**
   - Open competitions
   - Benchmark datasets
   - Community contributions

### Milestones Summary

```
[Historic Training] → α MAs > 10%
        ↓
[Paper Trading] → α MA1440 > 5% for N days
        ↓
[Live: 100 EUR] → Generate 10,000 EUR
        ↓
[Publish Paper] → Document methodology
        ↓
[Scale & Compete] → Variants + challenges
```

### Important: Spot Trading Only (Initial Phase)

**Live trading MUST use spot markets initially:**
- No margin/leverage
- No futures/perpetuals
- Maximum loss = position size (can't go negative)
- Simpler execution (no funding rates, liquidations)
- Model learns "real" price action, not derivative quirks

Graduate to margin/futures only after proving profitability on spot.

---

## 14. Model Relevance Lifetime (Fun Estimation)

How long before the trained model becomes a expensive random number generator?

### Alpha Decay by Strategy Type

| Strategy Level | Learned Pattern | Expected Lifetime | Why It Dies |
|----------------|-----------------|-------------------|-------------|
| L0-4 (Basic) | Sine, trends, dips | ∞ (synthetic only) | Not real markets |
| L5 Pairs | Cointegration | 3-6 months | Relationships break |
| L6 Triangle Arb | FX mispricings | 2-4 weeks | HFT eats this |
| L7 Index Arb | Basket lag | 1-3 months | ETF flows change |
| L8 Momentum | Trend following | 6-12 months | Crowded trade |
| L9 Mean Rev | VWAP reversion | 3-6 months | Regime dependent |
| L10 Breakout | Range escape | 4-8 months | Vol clustering |
| L11-13 Micro | Spread/imbalance | Days to weeks | Eaten by MMs |
| L14 Volatility | GARCH patterns | 1-3 months | Vol of vol |

### Pessimistic Estimate: "The Model is Dead" Timeline

```
Week 1-2:     "Wow it's printing money!"
Month 1:      "Hmm, alpha dropped to 5%"
Month 2:      "Still positive, just slower"
Month 3:      "Breakeven... market changed?"
Month 4:      "Need to retrain on recent data"
Month 6:      "Original model officially retired"
```

**Half-life of edge: ~2-3 months**

### Optimistic Estimate: "Continuous Learning" Timeline

If we implement online learning / periodic retraining:

```
Month 1-3:    Original model
Month 3-6:    Retrain v2 on recent 3mo data
Month 6-12:   Retrain v3, ensemble with v2
Year 1+:      Rolling retrain every quarter
              Keep best N model versions
              Ensemble voting for robustness
```

**Sustainable edge with maintenance: 1-2 years**

### Reality Check

| Factor | Impact on Lifetime |
|--------|-------------------|
| Market regime change | -50% (bear vs bull) |
| Major macro event | Reset to zero |
| Strategy crowding | -30% per year |
| Exchange rule changes | Unpredictable |
| Our edge is MCTS depth | Longer than simple ML |
| Curriculum breadth | Longer (not single trick) |

### Honest Guess

```
┌─────────────────────────────────────────────────────┐
│  TRAINED MODEL RELEVANCE                            │
│                                                     │
│  Without retraining:     2-4 months                 │
│  With quarterly retrain: 12-18 months               │
│  With online learning:   Indefinite (in theory)    │
│                                                     │
│  Black swan survival:    Flip a coin 🪙             │
└─────────────────────────────────────────────────────┘
```

### The Brutal Truth

> "All models are wrong, but some are useful... for a while."
>
> The market is an adversarial environment. Every edge gets arbitraged away.
> Our advantage: MCTS + curriculum makes the model *adaptable*, not just *fitted*.
>
> Plan for retraining. Budget for model decay. Celebrate while it works.

---

## 15. Continuous Adaptation (Counterpoint)

**Disagreement with decay pessimism:** A System 2 (MCTS/planning) agent SHOULD maintain profitability across regime changes - that's the whole point of planning vs pattern matching.

### Key Insight: Warmup vs Live Learning

| Phase | Purpose | Duration |
|-------|---------|----------|
| Historic 30-day | Learn current asset set | Warmup only |
| Live trading | Continuous encoder updates | Ongoing |
| Checkpoint cycle | Save new state | Every 10 days |

**The 30-day historic alignment isn't the final model** - it's just warmup to:
- Calibrate encoder to current asset universe
- Learn current volatility regime
- Establish baseline value estimates

### Live Continuous Learning Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LIVE TRADING LOOP                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Market Data ──→ Encoder ──→ MCTS Planning ──→ Action  │
│        │              ↑                                 │
│        │              │                                 │
│        └──→ Online Learning (encoder updates) ←─┘       │
│                       │                                 │
│                       ↓                                 │
│              [Checkpoint every 10 days]                 │
│                       │                                 │
│                       ↓                                 │
│              checkpoint_day_10/                         │
│              checkpoint_day_20/                         │
│              checkpoint_day_30/                         │
│              ...                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Why System 2 Survives Regime Changes

| System 1 (Pattern Match) | System 2 (MCTS Planning) |
|--------------------------|--------------------------|
| "I saw this pattern before" | "What happens if I do X?" |
| Dies when patterns change | Adapts via tree search |
| Fitted to past | Plans for future |
| Static after training | Dynamic value updates |

**MCTS doesn't memorize** - it *searches*. New regime = new values propagate through tree.

### Continuous Checkpoint Strategy

```python
# Pseudo-code for live learning
every_10_days:
    # Save current state
    save_checkpoint(f"live_day_{day_count}")

    # Keep rolling window of checkpoints
    keep_best_n_checkpoints(n=5)

    # If performance degraded, can rollback
    if alpha_ma10 < 0:
        load_checkpoint(best_recent_checkpoint)
```

### Revised Lifetime Estimate

```
┌─────────────────────────────────────────────────────────┐
│  SYSTEM 2 AGENT RELEVANCE (REVISED)                     │
│                                                         │
│  Static model (no updates):     2-4 months              │
│  With 10-day checkpoint cycle:  Indefinite              │
│  With encoder online learning:  Self-healing            │
│                                                         │
│  Regime change survival:        HIGH (planning adapts)  │
│  Black swan survival:           MEDIUM (if not OOM)     │
└─────────────────────────────────────────────────────────┘
```

### Implementation TODO

- [ ] Add online encoder updates during live trading
- [ ] Implement 10-day checkpoint cycle
- [ ] Add regime change detector (vol spike, correlation break)
- [ ] Automatic rollback if alpha_ma10 < threshold
- [ ] Ensemble recent N checkpoints for robustness
