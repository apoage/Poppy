# Test Log Filter Fix

## Quick Test

Run this to verify the log spam is gone:

```bash
python train_manager.py \
  --random-subset 32 \
  --exclude-forex \
  --date-start 2024-12-01 \
  --date-end 2024-12-31 \
  --min-coverage 0.8 \
  --steps 200 \
  --epochs 1 \
  --no-validate
```

## Expected Results

### Before Fix (500+ warning lines):
```
[spot_baseline_v1] Unknown currency USDTCHF, using 1.0
[spot_baseline_v1] Unknown currency USDTCHF, using 1.0
[spot_baseline_v1] Unknown currency USDTCHF, using 1.0
... (repeated 500+ times)
```

### After Fix (clean logs):
```
[spot_baseline_v1] Market loaded: 32 assets, 1,191 valid timestamps
[spot_baseline_v1] DEBUG: Min required assets set to 25 (80.0% coverage)
[spot_baseline_v1] Epoch 1 | Reward: +11.79 | Steps: 200 | Self-play: 1.4s
```

## What Changed

**File**: `mcts_trader/agent/portfolio.py`

1. Added USDT/USDC pair mappings (lines 104-109)
2. Added log deduplication (lines 151-156)

## Verify Fix

```bash
# Run test and count warnings
python train_manager.py --random-subset 32 --exclude-forex --date-start 2024-12-01 --date-end 2024-12-31 --min-coverage 0.8 --steps 200 --epochs 1 --no-validate 2>&1 | grep -c "Unknown currency"

# Should be 0 or very low (1-2) if any truly unknown currencies exist
```

If count is still high (>10), the fix didn't work. Check the file was saved correctly.
