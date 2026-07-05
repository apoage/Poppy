# TASK_732: Cross-Sectional Input Channels

**Status**: CONTINGENT — activate after TASK_730/731 if Bal.Acc still sub-40%.
**Priority**: Scope-adjacent. Addresses input-side ceiling, not model side.
**Rationale**: `research/729c_ceiling_alternatives.md` §6; salvageable code from TASK_125.

---

## Gap Identified

`env/jax_encoder.py::_compute_all_channels_temporal` produces 27 channels per asset:
- OHLCV-derived (returns, ranges, volumes, etc.)
- Technical indicators (momentum, RSI-like, BB-like, percentiles)

**All 27 are per-asset features computed independently.** There are ZERO cross-sectional features — nothing that tells asset `i` how it stands relative to the other 63 at the same timestamp.

A human trader's intuition for "buy the best, sell the worst" cannot emerge from features that don't know where an asset ranks. Rank oracle (TASK_730) asks the model to predict cross-sectional structure without giving it cross-sectional inputs. These tasks go together.

---

## Channels to Add

| # | Channel | Definition | Notes |
|---|---------|------------|-------|
| 28 | rank-of-return (1-step) | `argsort` of 1-step return across 64 assets / 63 | Salvage from TASK_125 prototype |
| 29 | rank-of-return (20-step) | Same, cumulative return over window | Matches oracle lookahead |
| 30 | beta-to-universe | Rolling 60-step beta of asset return vs equal-weighted universe return | Captures "moves with market" vs "moves against" |
| 31 | idiosyncratic vol | Residual std after regressing out universe beta | Captures "weird mover" regime |
| 32 | cross-sectional momentum | Asset's return - universe mean | Signed directional delta |

5 new channels → 32 total. Backward-incompatible with all existing checkpoints (channel count changes, first Dense layer shape changes).

---

## Salvage Check

Read `tasks/TASK_125_*.md` first — the rank-of-return channel was prototyped there but never deployed. Reuse existing code if still structurally compatible with current `jax_encoder.py`.

```bash
ls tasks/ | grep -i rank
ls tasks/ | grep TASK_125
```

---

## Implementation Steps

### Step 1 — Channel compute (JAX-pure)
```python
# env/jax_encoder.py
def _compute_cross_sectional_channels(returns_1step, returns_20step, n_assets=64):
    """Compute 5 cross-sectional channels.

    Args:
        returns_1step: (T, n_assets) per-step returns
        returns_20step: (T, n_assets) 20-step cumulative returns
    Returns:
        channels: (T, n_assets, 5)
    """
    # Ranks via argsort of argsort
    rank_1 = jnp.argsort(jnp.argsort(returns_1step, axis=-1), axis=-1) / (n_assets - 1)
    rank_20 = jnp.argsort(jnp.argsort(returns_20step, axis=-1), axis=-1) / (n_assets - 1)

    # Universe mean
    uni_ret = jnp.mean(returns_1step, axis=-1, keepdims=True)
    xs_momentum = returns_1step - uni_ret

    # Rolling beta (60-step window, pre-computed via cumulative moments)
    # ... (use existing rolling window utilities in jax_encoder)
    beta = _rolling_beta(returns_1step, uni_ret, window=60)
    idio_vol = _rolling_idio_std(returns_1step, beta, uni_ret, window=60)

    return jnp.stack([rank_1, rank_20, beta, idio_vol, xs_momentum], axis=-1)
```

### Step 2 — Integrate with `_compute_all_channels_temporal`
Append the 5 channels to the existing 27. Update channel count constant (probably in `mcts_trader/config/` or inline).

### Step 3 — Retrain Phase 1 with new channels
Full retrain required — first encoder Dense layer shape changes. Use the V3 / wider config (or whichever won previous pivots).

```bash
python train_easy.py \
  --periods-file config/foxglove_periods.json --epochs 200 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --output-dir jax_ckpt_easy_phase1_xs32 \
  > jax_ckpt_easy_phase1_xs32/phase1_nohup.log 2>&1 &
```

### Step 4 — Stage B
Standard protocol. Stack on top of whichever TASK_730/731 wins.

---

## Gates

### Phase 1 (V3 + 32 channels)
- Trade-agreement ≥ 75% by E50 (we expect better than 27-channel version — more signal should help imprint).

### Phase 2 E50
- `dir_acc > 0.65` AND Bal.Acc projected > 45% → cross-sectional channels were the missing ingredient. Ship this encoder.

### Benchmark
- Bal.Acc > 50% → **best result in project history**. Archive. Consider this the new baseline for future work.
- Bal.Acc 45-50% → meaningful. Stack next pivot.
- Bal.Acc < 45% → cross-sectional not sufficient; stop chasing features.

---

## Why Reserve for Last

- Full Phase 1 retrain required (12-24 hrs of GPU).
- All existing checkpoints shape-incompatible.
- Most expensive of the three pivots in time and electricity.

Do only after TASK_730 (rank oracle) and TASK_731 (identity ablation) have given their verdicts. If 730 alone wins to ≥45% Bal.Acc, this may not be needed — budget preservation.

---

## Files Modified

| File | Change |
|------|--------|
| `mcts_trader/env/jax_encoder.py` | Add `_compute_cross_sectional_channels`, integrate into `_compute_all_channels_temporal` |
| `mcts_trader/config/` or constants | Channel count 27 → 32 |
| Downstream: encoder input shape unchanged except for the channel dim; no model architecture changes |
