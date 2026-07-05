# Training Roadmap - Q1 2026

## Current Status

**Running**: L2_full_realism - hardened reality test of kitchen_sink_v2
```bash
python train_jax_fast.py \
    --synthetic-level 2 \
    --load-checkpoint jax_ckpt_fast/L1_kitchen_sink_v2 \
    --probabilistic-fills --borrow-rate 0.03 \
    --synthetic-noise 0.3 --episode-variance 30 \
    --lr-schedule cosine --lr-warmup-epochs 20 \
    ...
```

---

## Phase 1: Pre-Hero Fixes (Before Full Curriculum)

### Complex Tasks to Implement

| Task | Description | Priority |
|------|-------------|----------|
| **TASK_022** | TD(λ) returns - variance reduction | HIGH |
| **TASK_025** | Per-head gradient clipping | MEDIUM |
| **TASK_042** | ReDo dormant neurons | MEDIUM |
| **TASK_043** | ATC representation learning | LOW |

### Validation Checklist

- [ ] **Action Space Review**: Verify all 5 actions per asset are valid
  - Sell (market) - valid for long positions
  - Hold - always valid
  - Buy (market) - valid when cash available
  - Limit Sell - valid for positions
  - Limit Buy - valid when cash available

- [ ] **Real Market Validity**: Check if simulated actions reflect real market behavior
  - Order fill mechanics realistic?
  - Fee/slippage models accurate?
  - Position sizing sensible?

- [ ] **Command Validation**: Ensure no invalid commands are possible
  - Short selling when disabled?
  - Buying with no cash?
  - Selling when no position?

### Curriculum Review

- [ ] **Pace Check**: Levels progress easiest → hardest
- [ ] **Configuration Audit**: Each level's parameters make sense
- [ ] **Promotion Thresholds**: Alpha targets are achievable but challenging
- [ ] **Realism Ramp**: TASK_056 curriculum integration

---

## Phase 2: Hero Foundation Training

### Full Curriculum Run

```
Level 0 (Sine)      → Learn basic timing
Level 1 (Lagged)    → Learn correlation
Level 2 (Dip)       → Learn dip buying
Level 3 (Trends)    → Learn trend following
Level 4 (Rotation)  → Learn sector timing
Level 5+ (Advanced) → Arbitrage, microstructure
```

### Training Parameters
- Model: small (7.3M params)
- Config: kitchen_sink_v2 base
- Realism: Progressive (TASK_056)
- Duration: ~500 epochs per level

---

## Phase 3: Historical Data Training

### map_training_data.py Enhancement

**Goal**: Prepare real market data segments for training

**Current Approach**:
- 180-day segments
- Asset must cover all 180 days to be valid
- Generates training-ready datasets

**Enhancements Needed**:
- [ ] Include delisted pairs from `delistedpairs.csv`
- [ ] Handle gaps in data gracefully
- [ ] Generate segment metadata (volatility, trend, etc.)
- [ ] Create train/val/test splits

### Data Reuse Techniques

Historical data is deterministic - can multiply effective data via:

| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **Time dilation** | Stretch/compress segment tempo | Interpolate OHLCV |
| **Price mirroring** | Flip values around mean | `mirrored = 2*mean - price` |
| **Time reversal** | Play segment backwards | Reverse array, recalc returns |
| **Multi-market** | Add data from other exchanges | Validate pair compatibility |

**CRITICAL**: All transformations must preserve market physics:
- Returns distribution realistic
- Volume relationships sensible
- No zero/constant artifacts
- Math must match real movements

### Data Validation Checklist

- [ ] No collapsing values to zero
- [ ] No unexpected constants
- [ ] All pairs mathematically valid
- [ ] Returns/volatility realistic after transforms
- [ ] Numbers match reality - model learns REAL movements

### Data Constraints

| Metric | Value |
|--------|-------|
| Real data consumption | ~9 hours with 15 envs |
| Database size | 22GB |
| Usable segments | TBD after mapping |
| Potential with reuse | 3-4x effective data |

---

## Phase 4: Live Validation Progression

### Stage 1: Paper Trading (Validation Only)

**Purpose**: System validation, NOT training

- Second PC for paper trading
- Real-time data feed
- Model inference only
- Measure latency and throughput
- Identify synthetic↔real gaps

### Stage 2: Sacrificial Spot Run

**Purpose**: Real money validation with minimal risk

- Small capital allocation
- Spot market only (no leverage)
- **Quick signal**: Will know FAST if model works or not
- Success criteria: Positive alpha over benchmark

### Stage 3: Margin Trading

**Purpose**: Full deployment (only if spot succeeds)

- Switch to margin after spot validation
- Leverage introduces real risk
- Only proceed with proven model

```
Paper Trading → Spot (sacrificial) → Margin
    ↓               ↓                  ↓
 Validate       Quick signal       Full deploy
 system         works/doesn't      if proven
```

---

## Timeline (Tentative)

```
Week 1-2:  L2_full_realism validation
           Complex tasks (TASK_022, 025, 042, 043)
           Action space review

Week 3-4:  Curriculum review and fixes
           Hero foundation training start (L0-L4)

Week 5-6:  Hero training continuation (L5-L10)
           map_training_data.py enhancement

Week 7-8:  Historical data training prep
           Paper trading setup

Week 9+:   Historical data training
           Paper trading validation
```

---

## Notes

- Real data is consumed fast - may need to balance synthetic/real ratio
- Paper trading is exploratory - "we will see"
- Delisted pairs add historical diversity
- Curriculum pace may need adjustment based on L2 results
