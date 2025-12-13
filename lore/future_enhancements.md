# Future Enhancements - MCTS Trading Agent

## 1. Auto-Resample Assets on Training Plateau

### Concept
When training plateaus (loss falls below threshold or stops improving), automatically resample a new set of trading pairs and continue training. This prevents overfitting and exposes the model to diverse market conditions during a single training run.

### Motivation
- **Prevents overfitting**: Model won't memorize specific asset behaviors
- **Curriculum learning**: Gradually expose model to more diverse patterns
- **Efficient training**: No need to manually restart with new datasets
- **Robustness**: Model learns general trading principles across many assets

### Detection Criteria

Trigger asset resampling when **any** of these conditions are met:

#### Option 1: Loss Threshold
```python
if current_loss < 1.0 and epochs_since_resample > 500:
    resample_assets()
```
- Simple and direct
- Loss < 1.0 indicates model has learned current dataset well
- Wait 500 epochs minimum between resamples to avoid thrashing

#### Option 2: Plateau Detection (Better)
```python
if no_improvement_for_N_epochs(N=200) and epochs_since_resample > 500:
    resample_assets()
```
- More robust than fixed threshold
- Detects when model stops learning (plateau)
- Configurable patience (N=200 epochs)

#### Option 3: Reward-Based
```python
if avg_reward_stable_for_N_epochs(N=100) and epochs_since_resample > 500:
    resample_assets()
```
- Focuses on trading performance, not just loss
- More aligned with actual goal (profitable trading)
- Requires tracking reward moving average

### Implementation Plan

#### Phase 1: Detection Infrastructure

**File**: `train_jax_fast.py`

Add plateau detection logic:
```python
class PlateauDetector:
    def __init__(self, patience=200, min_delta=0.01, min_epochs_between=500):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs_between = min_epochs_between

        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.last_resample_epoch = 0

    def should_resample(self, current_epoch, current_loss):
        """Check if we should trigger asset resampling"""
        # Too soon since last resample?
        if current_epoch - self.last_resample_epoch < self.min_epochs_between:
            return False

        # Has loss improved?
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            return False

        # Loss plateaued
        self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            logging.info(f"Plateau detected after {self.epochs_without_improvement} epochs")
            return True

        return False

    def mark_resampled(self, current_epoch):
        """Mark that we just resampled"""
        self.last_resample_epoch = current_epoch
        self.epochs_without_improvement = 0
        # Don't reset best_loss - we want to keep improving globally
```

#### Phase 2: Asset Resampling Function

**File**: `train_jax_fast.py`

Add resampling logic:
```python
def resample_assets(db_path, n_assets=64, exclude_forex=True,
                   date_start=None, date_end=None, min_candles=262800):
    """
    Select new random assets from database.
    Returns: (asset_symbols, min_required_assets)
    """
    import sqlite3
    import random
    from datetime import datetime

    # Build SQL query (similar to train_manager.py)
    conn = sqlite3.connect(db_path)

    date_filter_sql = ""
    params = []
    if date_start or date_end:
        start_ts = int(date_start) if isinstance(date_start, (int, float)) else \
                   int(datetime.strptime(date_start, '%Y-%m-%d').timestamp())
        end_ts = int(date_end) if isinstance(date_end, (int, float)) else \
                 int(datetime.strptime(date_end, '%Y-%m-%d').timestamp())
        date_filter_sql = " AND o.timestamp >= ? AND o.timestamp <= ?"
        params = [start_ts, end_ts]

    query = f"""
        SELECT a.symbol, COUNT(*) as candle_count
        FROM assets a
        JOIN ohlcv o ON a.id=o.asset_id
        WHERE 1=1 {date_filter_sql}
        GROUP BY a.id
        HAVING COUNT(*) >= ?
    """
    params.append(min_candles)

    cursor = conn.execute(query, params)
    candidates = [r[0] for r in cursor.fetchall()]
    conn.close()

    # Filter forex if requested
    if exclude_forex:
        from train_manager import is_forex_pair  # Import helper
        candidates = [p for p in candidates if not is_forex_pair(p)]

    # Sample N assets
    if len(candidates) < n_assets:
        logging.warning(f"Only {len(candidates)} valid assets, requested {n_assets}")
        selected = candidates
    else:
        selected = random.sample(candidates, n_assets)

    logging.info(f"Resampled {len(selected)} new assets: {selected[:5]}...")
    return selected
```

#### Phase 3: Environment Reload

**File**: `mcts_trader/env/wrapper.py`

Add reload capability to FastTradingEnv:
```python
def reload_assets(self, new_asset_symbols, min_required_assets=None):
    """
    Reload environment with new assets without recreating everything.
    Preserves model weights, only changes data source.
    """
    logging.info(f"Reloading environment with {len(new_asset_symbols)} new assets")

    # Reinitialize market with new assets
    self.market = HistoricalMarket(
        db_path=self.market.db_path,  # Preserve DB path
        asset_symbols=new_asset_symbols,
        min_required_assets=min_required_assets or self.market.min_required_assets
    )

    # Update n_assets
    self.n_assets = len(new_asset_symbols)

    # Recreate encoder with new asset count
    self.encoder = MultiAssetStateEncoder(n_assets=self.n_assets)

    # Recreate portfolio
    self.portfolio = PortfolioManager(
        initial_capital=self.initial_capital,
        trading_fee_bps=self.trading_fee_bps,
        slippage_bps=self.slippage_bps,
        mode=self.mode,
        n_assets=self.n_assets
    )
    self.portfolio.register_assets(self.market.symbol_map)

    # Reset to new episode
    return self.reset()
```

#### Phase 4: Integration into Training Loop

**File**: `train_jax_fast.py`

Main training loop modification:
```python
# Initialize plateau detector
plateau_detector = PlateauDetector(
    patience=200,          # No improvement for 200 epochs
    min_delta=0.01,        # Minimum loss decrease to count as improvement
    min_epochs_between=500 # Wait at least 500 epochs between resamples
)

# Training loop
for epoch in range(args.epochs):
    # ... existing training code ...

    # After computing loss
    if plateau_detector.should_resample(epoch, current_loss):
        logging.info(f"🔄 Triggering asset resample at epoch {epoch}")

        # Resample new assets
        new_assets = resample_assets(
            db_path='trading_data.db',
            n_assets=64,  # Or read from config
            exclude_forex=True,
            date_start=args.date_start,  # Preserve date range
            date_end=args.date_end,
            min_candles=262800
        )

        # Reload environment with new assets
        env.reload_assets(new_assets, min_required_assets)

        # Update config tracking
        plateau_detector.mark_resampled(epoch)

        # Log the change
        logging.info(f"📊 Resampled to: {new_assets[:10]}...")

        # Optional: Save checkpoint before continuing
        save_checkpoint(model, optimizer, epoch, f'checkpoint_resample_{epoch}.pkl')
```

### Configuration

Add CLI arguments to control behavior:

**File**: `train_jax_fast.py`
```python
parser.add_argument('--auto-resample', action='store_true',
                   help='Enable automatic asset resampling on plateau')
parser.add_argument('--resample-patience', type=int, default=200,
                   help='Epochs without improvement before resampling')
parser.add_argument('--resample-interval', type=int, default=500,
                   help='Minimum epochs between resamples')
parser.add_argument('--resample-min-delta', type=float, default=0.01,
                   help='Minimum loss decrease to count as improvement')
```

### Usage Example

```bash
# Phase 1 with auto-resampling
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2024-01-01 \
  --date-end 2025-12-12 \
  --min-coverage 0.75 \
  --steps 1440 \
  --epochs 50000 \
  --no-validate \
  --auto-resample \
  --resample-patience 200 \
  --resample-interval 500
```

**Expected behavior**:
- Train on initial 64 random assets
- After ~500-700 epochs, plateau detected
- Resample to new 64 assets
- Continue training seamlessly
- Repeat every ~500-700 epochs until max epochs (50,000)

### Benefits

1. **Exposure to ~640 assets** over 50K epochs (10 resamples × 64 assets)
2. **No manual intervention** required
3. **Prevents overfitting** to specific asset behaviors
4. **Model learns general patterns** applicable to any crypto asset
5. **Efficient use of training time** - no restarts needed

### Monitoring

Add logging to track resamples:
```python
# Log summary at end of training
logging.info("=" * 70)
logging.info("ASSET RESAMPLE SUMMARY")
logging.info("=" * 70)
logging.info(f"Total resamples: {plateau_detector.last_resample_epoch // 500}")
logging.info(f"Assets seen: ~{plateau_detector.last_resample_epoch // 500 * 64}")
logging.info(f"Final loss: {current_loss:.4f}")
logging.info("=" * 70)
```

### Risks & Mitigations

#### Risk 1: Too Frequent Resampling
**Symptom**: Model never converges, keeps restarting
**Mitigation**: `min_epochs_between=500` ensures minimum 500 epochs on each dataset

#### Risk 2: Resampling Breaks Training
**Symptom**: Loss spikes after resample
**Mitigation**: This is expected! Model needs to adapt to new assets. Should recover within 50-100 epochs.

#### Risk 3: Model Forgets Previous Learning
**Symptom**: Performance degrades over time
**Mitigation**:
- Model weights are **preserved** across resamples
- Only the **data** changes, not the network
- Model actually **generalizes better** by seeing diverse data

### Testing Strategy

Before full deployment:

**Test 1**: Quick test with 5 epochs between resamples
```bash
python train_jax_fast.py --epochs 50 --auto-resample --resample-interval 5
```
- Verify resampling triggers
- Check environment reloads successfully
- Confirm no crashes

**Test 2**: Short production test (1000 epochs)
```bash
python train_manager.py --epochs 1000 --auto-resample --resample-patience 100 --resample-interval 200
```
- Should trigger 3-5 resamples
- Verify loss recovers after each resample
- Check checkpoint integrity

**Test 3**: Full production (50K epochs)
- Monitor first 5K epochs closely
- Check logs for resample frequency
- Validate final model on held-out data

### Alternative: Manual Trigger

If auto-detection is unreliable, add manual trigger:
```bash
# Send SIGUSR1 to trigger resample
kill -SIGUSR1 <pid>
```

In training code:
```python
import signal

def handle_resample_signal(signum, frame):
    global should_resample
    should_resample = True
    logging.info("Received resample signal")

signal.signal(signal.SIGUSR1, handle_resample_signal)
```

---

## 2. Transfer Learning Between Phases

### Concept
Save Phase 1 checkpoint and load it to initialize Phase 2 training, rather than starting from scratch.

### Implementation
```python
# End of Phase 1
save_checkpoint(model, optimizer, epoch, 'phase1_final.pkl')

# Start of Phase 2
if args.load_checkpoint:
    load_checkpoint(model, optimizer, args.load_checkpoint)
    logging.info(f"Loaded Phase 1 weights from {args.load_checkpoint}")
```

### Usage
```bash
# Phase 1
python train_manager.py --epochs 10000 ... --save-final phase1_final.pkl

# Phase 2 (load Phase 1 weights)
python train_manager.py --epochs 2000 ... --load-checkpoint phase1_final.pkl
```

---

## 3. Curriculum Learning (Progressive Asset Count)

### Concept
Start with 32 assets (easy to learn), gradually increase to 128 (harder).

### Implementation
```python
asset_schedule = [
    (0, 32),      # Epochs 0-2000: 32 assets
    (2000, 48),   # Epochs 2000-4000: 48 assets
    (4000, 64),   # Epochs 4000-8000: 64 assets
    (8000, 96),   # Epochs 8000-10000: 96 assets
]

for epoch in range(args.epochs):
    # Check if we should increase asset count
    for epoch_threshold, n_assets in asset_schedule:
        if epoch == epoch_threshold and env.n_assets != n_assets:
            logging.info(f"📈 Increasing assets to {n_assets}")
            new_assets = resample_assets(n_assets=n_assets)
            env.reload_assets(new_assets)
```

---

## 4. Validation Split

### Concept
Hold out 10% of temporal data for validation (don't train on it).

### Implementation
```python
# In train_manager.py
if args.validation_split:
    # Use 90% of time range for training
    train_end = start_ts + int((end_ts - start_ts) * 0.9)
    val_start = train_end
    val_end = end_ts

    logging.info(f"Train: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(train_end)}")
    logging.info(f"Val: {datetime.fromtimestamp(val_start)} to {datetime.fromtimestamp(val_end)}")
```

---

## 5. Live Data Integration (Paper Trading)

### Concept
Replace HistoricalMarket with WebSocket stream for real-time paper trading.

### Files to Create
- `mcts_trader/data/live_stream.py` - WebSocket feed wrapper
- `mcts_trader/env/live_env.py` - Live trading environment

### Implementation Sketch
```python
class LiveMarket:
    def __init__(self, asset_symbols):
        self.ws_client = KrakenWebSocketClient(asset_symbols)
        self.current_data = {}

    def step(self):
        # Non-blocking, returns latest data
        self.current_data = self.ws_client.get_latest()

    def get_current_data(self):
        return self.current_data
```

---

## 6. Hyperparameter Optimization

### Concept
Use Optuna or similar to tune:
- Learning rate
- Batch size
- MCTS simulations
- Coverage threshold
- Episode length

### Implementation
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    # Train model with these hyperparams
    final_reward = train_model(lr=lr, batch_size=batch_size)

    return final_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## 7. Multi-Timeframe Training

### Concept
Train on multiple timeframes (1min, 5min, 15min, 1h) simultaneously.

### Implementation
```python
class MultiTimeframeEnv:
    def __init__(self, timeframes=['1m', '5m', '15m', '1h']):
        self.markets = {
            tf: HistoricalMarket(timeframe=tf)
            for tf in timeframes
        }

    def get_observation(self):
        # Combine observations from all timeframes
        obs = {}
        for tf, market in self.markets.items():
            obs[tf] = market.get_current_data()
        return obs
```

---

## Implementation Priority

### High Priority (Implement Soon)
1. ✅ Log filtering (DONE)
2. **Auto-resample on plateau** (this document)
3. Transfer learning between phases

### Medium Priority (Next Month)
4. Validation split
5. Curriculum learning

### Low Priority (Future)
6. Live data integration (after paper trading validation)
7. Hyperparameter optimization
8. Multi-timeframe training

---

## Notes

All enhancements should:
- Be backward compatible (use flags to enable)
- Log clearly what's happening
- Save checkpoints before major changes
- Allow user to disable if unstable

Test each enhancement independently before combining them.
