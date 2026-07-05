# TASK_061: Benchmark Event Coverage Strategy

## Status: PLANNING

## Problem Statement

Training needs exposure to extreme market events (crashes, rallies, volatility spikes) to learn appropriate behavior. But benchmarks need to be "blind" - testing generalization to unseen data.

**The Paradox:**
- If we exclude all crashes from training → model never learns crash behavior
- If we include all crashes in training → benchmark on crashes isn't blind
- Need to learn the PATTERN TYPE without seeing the SPECIFIC INSTANCE

## Proposed Solution: Cross-Event Holdout

For each event type, maintain multiple instances. Train on some, benchmark on others.

### Event Classification

| Type | Characteristics | Examples |
|------|-----------------|----------|
| **Panic Crash** | -20% to -40% in days, high volume | Luna, FTX, Musk Crash |
| **Rally** | +15% to +40% in week, momentum | ETF, BlackRock, Trump |
| **Volatility Spike** | High range, no clear direction | ATH Nov 2021, Halving |
| **Capitulation Bottom** | Low volume, exhaustion | Nov 2022 bottom |
| **Recovery** | Post-crash bounce | Mar 2023 bank crisis |

### Train/Benchmark Split by Event Type

```
PANIC CRASHES:
  Train: Luna (2022-05-07 → 2022-05-14)
  Benchmark: FTX (2022-11-06 → 2022-11-13)

RALLIES:
  Train: BlackRock (2023-06-15 → 2023-06-22)
  Benchmark: ETF Approval (2024-01-08 → 2024-01-15)

VOLATILITY:
  Train: ATH Peak (2021-11-08 → 2021-11-15)
  Benchmark: Halving (2024-04-17 → 2024-04-24)

RECOVERY:
  Train: Bank Crisis (2023-03-10 → 2023-03-17)
  Benchmark: Musk Crash recovery (2021-05-24 → 2021-05-31)
```

## Implementation

### Phase 1: Event Period Configuration

Extend `config/benchmark_periods.json`:

```json
{
  "benchmark_periods": {
    "ftx_crash": {
      "start": "2022-11-06",
      "end": "2022-11-13",
      "type": "panic_crash",
      "description": "FTX bankruptcy"
    },
    "etf_approval": {
      "start": "2024-01-08",
      "end": "2024-01-15",
      "type": "rally",
      "description": "Spot ETF approval"
    }
  },
  "training_periods": {
    "luna_crash": {
      "start": "2022-05-07",
      "end": "2022-05-14",
      "type": "panic_crash",
      "description": "Luna/UST collapse"
    },
    "blackrock_rally": {
      "start": "2023-06-15",
      "end": "2023-06-22",
      "type": "rally",
      "description": "BlackRock ETF filing"
    }
  },
  "event_types": ["panic_crash", "rally", "volatility", "capitulation", "recovery"]
}
```

### Phase 2: Training Date Exclusion

Modify historical data loading to:
1. Load all data EXCEPT benchmark periods
2. Ensure training periods are weighted/included

```python
def get_training_dates(db_path, config):
    """Get training date ranges excluding benchmark periods."""
    benchmark_periods = config['benchmark_periods']

    # Exclude benchmark windows from training
    excluded_ranges = [
        (p['start'], p['end'])
        for p in benchmark_periods.values()
    ]

    return filter_dates(all_dates, excluded_ranges)
```

### Phase 3: Event-Aware Sampling

During historical training, oversample rare event types:
- Normal market: 70% of samples
- Crash periods: 15% of samples
- Rally periods: 15% of samples

This ensures the model sees enough extreme events without memorizing specific ones.

### Phase 4: Benchmark Analytics

Update DS24 to show:
- Per-event-type performance
- Cross-event generalization (trained on Luna, tested on FTX)
- Event type confusion matrix

## Verification

Model should show:
1. Positive alpha on benchmark crashes (learned crash behavior from training crashes)
2. Positive alpha on benchmark rallies (learned rally behavior from training rallies)
3. Similar performance across train/benchmark events of same type

## Event Database (Verified Dates Needed)

| Event | Start | End | Type | BTC Change | In DB? |
|-------|-------|-----|------|------------|--------|
| Musk Crash | 2021-05-17 | 2021-05-24 | panic_crash | -30% | ? |
| ATH Peak | 2021-11-08 | 2021-11-15 | volatility | ±10% | ? |
| Luna Collapse | 2022-05-07 | 2022-05-14 | panic_crash | -25% | ? |
| FTX Implosion | 2022-11-06 | 2022-11-13 | panic_crash | -22% | ? |
| Cycle Bottom | 2022-11-20 | 2022-11-27 | capitulation | flat | ? |
| Bank Crisis | 2023-03-10 | 2023-03-17 | recovery | +35% | ? |
| BlackRock Filing | 2023-06-15 | 2023-06-22 | rally | +15% | ? |
| ETF Approval | 2024-01-08 | 2024-01-15 | rally | +10% | ? |
| Halving | 2024-04-17 | 2024-04-24 | volatility | flat | ? |
| Trump Pump | 2024-11-04 | 2024-11-11 | rally | +30% | ? |

## Zero-Knowledge Properties

This approach gives "zero knowledge" in the cryptographic sense:
- Model proves it can handle crashes → without seeing the specific crash
- Model proves it can ride rallies → without seeing the specific rally
- Benchmark is truly blind to the specific instance

## Dependencies

- Historical data in `trading_data.db` covering 2021-2024
- `config/benchmark_periods.json` extended with event types
- Data loader modification to exclude benchmark periods
- DS24 analytics for event-type breakdown

## Acceptance Criteria

- [ ] All event periods verified in database
- [ ] Config file extended with train/benchmark split
- [ ] Training excludes benchmark periods
- [ ] Benchmark shows per-event-type metrics
- [ ] Cross-event generalization measured
