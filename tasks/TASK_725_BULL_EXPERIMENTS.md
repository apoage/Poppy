# TASK_725: Bull Market Fix — Experiment Sweep

**Status**: DEFERRED — superseded by TASK_726 (class-balanced BC loss)
**Priority**: LOW — experiment sweep is the fallback if TASK_726 doesn't work
**Diagnostic finding**: Model response to oracle BUY and oracle SELL is statistically identical (±0.3%). Root cause identified by encoder audit: 93.75% HOLD class imbalance in oracle targets (k=4/64), not encoder or search. See `prompt-lib/out_encoder_audit/findings.md`.

**Note**: Experiments A/B (high BC, pure BC) are still relevant as baselines. Experiments C/F are deprioritized (encoder already preserves direction). Experiment D (return-weighted BC) is subsumed by TASK_726's more targeted class rebalancing approach.

---

## Experiment Protocol

Each experiment:
1. Starts from `jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl` (best checkpoint, 4/6 gate)
2. Runs 200 epochs (~10 hours at 190s/epoch)
3. Benchmarked at E100 and E200
4. Compared against baseline: bull alpha = -24.25%, avg alpha = +5.73%
5. Output dir: `jax_ckpt_easy_phase2_725X/` where X = experiment letter

**Base command template:**
```bash
mkdir -p jax_ckpt_easy_phase2_725X && nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.3 \
  --no-reset-value-head \
  --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 \
  [EXPERIMENT-SPECIFIC FLAGS] \
  --output-dir jax_ckpt_easy_phase2_725X \
  > jax_ckpt_easy_phase2_725X/stage_b_nohup.log 2>&1 &
```

**Benchmark command:**
```bash
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_easy.py --checkpoint jax_ckpt_easy_phase2_725X/phase2_stageB_e100.pkl
```

---

## Experiment A: High BC Weight (Oracle Dominance)

**Hypothesis**: The oracle IS profitable in bull (+265% PnL). If BC weight is high enough, the model will be forced to reproduce oracle behavior including bull BUYs. The factored search targets may be actively harming directional learning.

**Changes**: BC weight only — no code changes.

```bash
# Experiment-specific flags:
--bc-weight-min 0.15 --bc-decay-steps 100000
```

This makes BC 5× stronger (0.15 vs 0.03). bc_decay_steps=100000 means BC stays at 0.15 for the full 200 epochs (200×400=80000 steps < 100000). The oracle signal dominates the policy gradient.

**What to watch**: If policy_loss drops below 1.55 AND bull alpha improves, the oracle has the right signal and the model just wasn't following it. If policy_loss drops but bull stays bad, the oracle itself might not give clear directional signals.

**Risk**: Overfitting to oracle — losing the factored search improvements. But we need to isolate whether the problem is "can't learn from oracle" vs "oracle doesn't teach direction."

---

## Experiment B: Pure BC (No Factored Search)

**Hypothesis**: Factored search targets are direction-agnostic (Q-values similar for BUY and SELL because the dynamics model can't differentiate them). Removing factored search entirely and training with BC only eliminates this noise source.

**Changes**: Disable factored search. This requires a code change.

**File**: `train_phase2.py`, in `train_stage_b()`, around line 1130 where the search is called:

Replace the MCTS/factored search call block with a skip:
```python
# After batch = pipeline.get_batch():
if config.use_factored_search and config.factored_temperature < 0:
    # EXPERIMENT B: Skip search entirely, use oracle targets as policy target
    visit_counts = batch['oracle_targets']  # (B, 320) per-asset oracle distribution
    mcts_value = jnp.zeros(batch_size)  # Dummy — not used with value_weight=0
else:
    # Normal search path
    _actions, visit_counts, stats = search(params, search_key, obs)
    mcts_value = stats['root_value']
```

**Run with**: `--factored-temperature -1` to trigger the skip (hack: negative temp = oracle-only mode).

Actually, simpler approach — just set bc_weight very high and policy_weight very low:

```bash
# Experiment-specific flags (no code change needed):
--bc-weight-min 1.0 --bc-decay-steps 100000
```

With bc_weight=1.0 and policy_weight=1.0 (from the factored targets), BC is 50% of the policy gradient. But factored targets at T=0.3 are near-uniform (CE ≈ 1.60), so their gradient is weak. BC at weight 1.0 with oracle targets (CE ≈ 1.45) dominates.

This effectively becomes BC-only training without code changes.

**What to watch**: If bull alpha improves significantly, the factored search IS the problem (its direction-agnostic targets wash out oracle's directional signal). If no improvement, the model architecture can't learn direction regardless of signal.

---

## Experiment C: Unfreeze Encoder (Learn New Features)

**Hypothesis**: The frozen encoder can't extract trend/direction information from price windows. The TRM reasoning operates on encoder outputs that don't encode "prices are trending up." Unfreezing the encoder with a very low LR lets it adapt features for directional awareness.

**Changes**: Code change to unfreeze encoder immediately (not at a specific epoch).

**File**: `train_phase2.py`, in `create_stage_b_optimizer()` around line 641.

The current mask freezes encoder:
```python
def should_train(path, _leaf):
    if _is_dynamics_key(path):
        return True
    if _is_encoder_key(path):
        return False  # ← CHANGE THIS
    ...
```

Need to either:
1. Add a CLI flag `--unfreeze-encoder` that sets `config.encoder_unfreeze_epoch = 0`
2. Or set `config.freeze_encoder = False` and rebuild optimizer

**Simplest implementation** — add to `parse_args()`:
```python
parser.add_argument('--unfreeze-encoder', action='store_true',
                    help='Unfreeze encoder from epoch 0 (low LR)')
```

And in the overrides section:
```python
if args.unfreeze_encoder:
    config.freeze_encoder = False
    config.encoder_unfreeze_epoch = 0
    if config.encoder_lr == 0.0:
        config.encoder_lr = 3e-5  # Very low LR for fine-tuning
```

Then modify `make_stage_b_mask()` to respect `config.freeze_encoder`:
```python
def should_train(path, _leaf):
    if _is_dynamics_key(path):
        return True
    if _is_encoder_key(path):
        return not config.freeze_encoder  # ← Respect config
    ...
```

Wait — `make_stage_b_mask` doesn't have access to `config`. The config is captured in the closure of `create_stage_b_optimizer`. Need to pass it through.

**Better approach**: Use the existing `encoder_unfreeze_epoch` mechanism. Set it to 0:

```bash
# But there's no CLI flag for this. Add one:
parser.add_argument('--encoder-unfreeze-epoch', type=int, default=None)
# Override:
if args.encoder_unfreeze_epoch is not None:
    config.encoder_unfreeze_epoch = args.encoder_unfreeze_epoch
```

Then in `train_stage_b()`, the existing check at line 1201:
```python
if (config.encoder_unfreeze_epoch is not None and
        epoch == config.encoder_unfreeze_epoch):
    optimizer, opt_state = create_stage_b_optimizer_with_encoder(config, params)
    train_step_fn = make_stage_b_train_step(model, config, optimizer)
```

This already works! Just add the CLI flag and pass `--encoder-unfreeze-epoch 0`.

But wait — at epoch 0, the optimizer is created BEFORE the epoch loop. The unfreeze check is INSIDE the loop at the start of each epoch. Epoch 0's check: `epoch == 0` → True → rebuild optimizer. This happens before the first training step. Should work.

Actually, looking more carefully at the code flow:
```python
# Create optimizer (trains reasoning + dynamics, freezes encoder)
optimizer, opt_state = create_stage_b_optimizer(config, params)
train_step_fn = make_stage_b_train_step(model, config, optimizer)
...
for epoch in range(config.epochs):
    ...
    if (config.encoder_unfreeze_epoch is not None and
            epoch == config.encoder_unfreeze_epoch):
        optimizer, opt_state = create_stage_b_optimizer_with_encoder(config, params)
        train_step_fn = make_stage_b_train_step(model, config, optimizer)
```

At epoch=0, the check fires and rebuilds the optimizer to include encoder. This happens BEFORE the step loop. So the very first training step uses the unfrozen encoder. Good.

**Implementation**: Just add the CLI flag. No other code changes needed.

```bash
# Experiment-specific flags:
--encoder-unfreeze-epoch 0 --bc-weight-min 0.03
```

**What to watch**: If encoder_lr=3e-5 (default when unfreezing) is too high, gradient norms will spike and policy will degrade. If it works, the encoder learns trend-aware features and bull alpha improves.

**Risk**: Encoder destabilization. The encoder was trained in Phase 1 on all regimes. Unfreezing with directional pressure might break features that work for bear/crash.

---

## Experiment D: Return-Weighted BC Loss

**Hypothesis**: The oracle's BUY signals in bull markets have HIGH expected returns (the market is going up). The current BC loss weights all oracle signals equally — a BUY on a +5% asset counts the same as a BUY on a +0.01% asset. Weighting BC by the oracle's expected return magnitude emphasizes high-conviction directional signals.

**Changes**: Code change to `make_stage_b_train_step()`.

**File**: `train_phase2.py`, in the loss_fn inside `make_stage_b_train_step()`, around the BC loss computation (line ~790):

Current:
```python
# === Oracle BC regularizer (decaying, per-asset CE matching Phase 1) ===
policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
oracle_reshaped = oracle_targets.reshape(-1, n_assets, n_actions_per_asset)
bc_loss = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped).mean()
```

Change to return-weighted:
```python
# === Oracle BC regularizer — return-weighted (TASK_725D) ===
# Weight each sample's BC loss by the absolute oracle remaining PnL.
# High-return states (bull market opportunities) get amplified.
policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
oracle_reshaped = oracle_targets.reshape(-1, n_assets, n_actions_per_asset)
per_sample_bc = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped).mean(axis=-1)  # (B,)
# Weight by |oracle_remaining_pnl| — states with big future returns matter more
bc_weights = jnp.abs(oracle_values) / jnp.maximum(jnp.mean(jnp.abs(oracle_values)), 1e-8)  # Normalized
bc_loss = jnp.mean(per_sample_bc * bc_weights)
```

Note: `oracle_values` is already passed to the train step (it's the oracle remaining PnL). The mean-normalization ensures the total gradient magnitude stays similar.

This means: in bull periods where oracle_remaining_pnl is high (big future returns), the BC loss is amplified. The model gets stronger gradient signal from bull-market samples.

**No CLI flags needed** — this is a code change. Use a config flag:

Add to `phase2_config.py`:
```python
bc_return_weighted: bool = False  # TASK_725D: Weight BC by oracle return magnitude
```

Add to `parse_args()`:
```python
parser.add_argument('--bc-return-weighted', action='store_true',
                    help='Weight BC loss by oracle return magnitude (emphasizes bull)')
```

Guard the code change:
```python
if bc_return_weighted:
    # Return-weighted BC
    per_sample_bc = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped).mean(axis=-1)
    bc_weights = jnp.abs(oracle_values) / jnp.maximum(jnp.mean(jnp.abs(oracle_values)), 1e-8)
    bc_loss = jnp.mean(per_sample_bc * bc_weights)
else:
    bc_loss = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped).mean()
```

```bash
# Experiment-specific flags:
--bc-return-weighted --bc-weight-min 0.05
```

**What to watch**: If bull alpha improves without hurting bear/crash, the return-weighting successfully emphasizes directional signals. The BC loss value itself might increase (different scale due to weighting).

---

## Experiment E: Factored Temperature Sweep

**Hypothesis**: T=0.3 might be suboptimal. Lower temperature (T=0.1) makes factored targets very sharp (near-argmax), which might help the model commit to BUY/SELL decisions instead of spreading probability across HOLD. Higher temperature (T=0.5) might help by preserving more information from Q-value differences.

**Changes**: Config only — two sub-experiments.

```bash
# E1: Very sharp (T=0.1)
--factored-temperature 0.1 --bc-weight-min 0.03

# E2: Softer (T=0.5)
--factored-temperature 0.5 --bc-weight-min 0.03
```

Run both. Compare bull alpha. This is the cheapest experiment (no code changes, just config).

---

## Experiment F: Direction-Aware Observation Channel

**Hypothesis**: The encoder sees 64-step price windows but may not extract directional trend from them (uptrend vs downtrend). Adding an explicit trend indicator as an extra channel gives the model direct access to "are prices going up or down?"

**Changes**: Code change to the encoder/observation pipeline.

**File**: `mcts_trader/env/jax_encoder.py` — add a trend channel to the observation encoding.

The current observation has shape `(B, n_channels, window_size, n_assets)` where n_channels=27 (or 34 with state channels). Add one channel: per-asset trend sign.

**Implementation**:

In `jit_batch_encode_temporal()` (or wherever observations are constructed), after encoding price/volume channels, add:

```python
# Trend channel: sign of window-length return
# prices shape: (T, n_assets, 5) where channel 3 = close
# For window starting at t, trend = sign(close[t+ws-1] - close[t])
close_start = prices[window_starts, :, 3]  # (B, n_assets)
close_end = prices[window_starts + ws - 1, :, 3]  # (B, n_assets)
trend = jnp.sign(close_end - close_start)  # -1, 0, +1
# Tile across window dimension: (B, 1, ws, n_assets)
trend_channel = jnp.tile(trend[:, None, None, :], (1, 1, ws, 1))
# Concatenate: (B, n_channels+1, ws, n_assets)
obs = jnp.concatenate([obs, trend_channel], axis=1)
```

**Config change**: `n_channels: 28` (or 35 with state channels).

**WARNING**: This changes the model input shape. The encoder's first layer expects n_channels input features. Changing n_channels requires either:
1. Reinitializing the encoder's first layer (loses Phase 1 training)
2. Or extending it with a zero-initialized extra channel weight (preserves existing training)

Option 2 is better:
```python
# In model loading, after loading checkpoint params:
# Get existing encoder input kernel (n_channels_old, ...)
old_kernel = params['trm']['encoder'][...]['kernel']  # Shape depends on encoder version
# Extend with zeros for new channel
new_row = jnp.zeros((1, old_kernel.shape[1]))
params['trm']['encoder'][...]['kernel'] = jnp.concatenate([old_kernel, new_row], axis=0)
```

This is the most complex experiment — significant code changes across encoder, config, and checkpoint loading.

**What to watch**: If the model starts differentiating BUY from SELL after adding trend channel, the encoder was missing this feature. This is the most "correct" fix but highest implementation effort.

**Risk**: Architecture change → incompatible with existing checkpoints without kernel extension. The extended kernel starts at zero weight for the trend channel, so the model initially ignores it and gradually learns to use it.

---

## Priority Order

Based on the diagnostic (model can't distinguish BUY from SELL):

| Priority | Experiment | Code Changes | Expected Impact | Effort |
|----------|-----------|-------------|-----------------|--------|
| 1 | **A: High BC** | None | Medium — tests if oracle has the signal | 1 min |
| 2 | **B: Pure BC** | None | Medium — tests if factored search hurts | 1 min |
| 3 | **D: Return-weighted BC** | Small (loss fn) | High — emphasizes bull signals | 20 min |
| 4 | **E: Temperature sweep** | None | Low-Medium | 1 min |
| 5 | **C: Unfreeze encoder** | Small (CLI flag) | Medium — encoder learns direction | 10 min |
| 6 | **F: Trend channel** | Large (encoder + config + ckpt) | High — direct fix for root cause | 2 hours |

**Run A and B first** (zero code changes, fastest). Their results tell us:
- A works → oracle has the signal, need to amplify it (try D next)
- B works → factored search is harmful, remove it
- Neither works → model architecture can't learn direction (try C then F)

---

## Training Commands (Copy-Paste Ready)

### Experiment A: High BC
```bash
mkdir -p jax_ckpt_easy_phase2_725a && nohup python train_phase2.py --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl --stage B --epochs 200 --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 --factored --factored-temperature 0.3 --no-reset-value-head --value-target-mode nstep --td-discount 0.95 --value-weight 0.0 --bc-weight-min 0.15 --bc-decay-steps 100000 --output-dir jax_ckpt_easy_phase2_725a > jax_ckpt_easy_phase2_725a/stage_b_nohup.log 2>&1 &
```

### Experiment B: Pure BC (bc=1.0)
```bash
mkdir -p jax_ckpt_easy_phase2_725b && nohup python train_phase2.py --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl --stage B --epochs 200 --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 --factored --factored-temperature 0.3 --no-reset-value-head --value-target-mode nstep --td-discount 0.95 --value-weight 0.0 --bc-weight-min 1.0 --bc-decay-steps 100000 --output-dir jax_ckpt_easy_phase2_725b > jax_ckpt_easy_phase2_725b/stage_b_nohup.log 2>&1 &
```

### Experiment D: Return-Weighted BC (NEEDS CODE CHANGES FIRST)
```bash
mkdir -p jax_ckpt_easy_phase2_725d && nohup python train_phase2.py --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl --stage B --epochs 200 --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 --factored --factored-temperature 0.3 --no-reset-value-head --value-target-mode nstep --td-discount 0.95 --value-weight 0.0 --bc-weight-min 0.05 --bc-return-weighted --output-dir jax_ckpt_easy_phase2_725d > jax_ckpt_easy_phase2_725d/stage_b_nohup.log 2>&1 &
```

### Experiment E1: Sharp Temperature
```bash
mkdir -p jax_ckpt_easy_phase2_725e1 && nohup python train_phase2.py --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl --stage B --epochs 200 --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 --factored --factored-temperature 0.1 --no-reset-value-head --value-target-mode nstep --td-discount 0.95 --value-weight 0.0 --bc-weight-min 0.03 --output-dir jax_ckpt_easy_phase2_725e1 > jax_ckpt_easy_phase2_725e1/stage_b_nohup.log 2>&1 &
```

### Experiment C: Unfreeze Encoder (NEEDS CLI FLAG ADDED FIRST)
```bash
mkdir -p jax_ckpt_easy_phase2_725c && nohup python train_phase2.py --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl --stage B --epochs 200 --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 --factored --factored-temperature 0.3 --no-reset-value-head --value-target-mode nstep --td-discount 0.95 --value-weight 0.0 --bc-weight-min 0.05 --encoder-unfreeze-epoch 0 --output-dir jax_ckpt_easy_phase2_725c > jax_ckpt_easy_phase2_725c/stage_b_nohup.log 2>&1 &
```

---

## Benchmark After Each Experiment

```bash
# At E100:
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_easy.py --checkpoint jax_ckpt_easy_phase2_725X/phase2_stageB_e100.pkl

# At E200:
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_easy.py --checkpoint jax_ckpt_easy_phase2_725X/phase2_stageB_e200.pkl
```

**Key metrics to compare**: bull alpha (target: > -15%), avg alpha (target: > +5.73%), trade density in bull (target: > 5%).

---

## Implementation Checklist for Code-Change Experiments

### Experiment C (Unfreeze Encoder) — needs 2 changes:

1. **`train_phase2.py` parse_args()**: Add `--encoder-unfreeze-epoch` argument
2. **`train_phase2.py` overrides**: Add `if args.encoder_unfreeze_epoch is not None: config.encoder_unfreeze_epoch = args.encoder_unfreeze_epoch`

The `train_stage_b()` already has the unfreeze logic at the right place — just needs the CLI arg to trigger it.

### Experiment D (Return-Weighted BC) — needs 3 changes:

1. **`phase2_config.py`**: Add `bc_return_weighted: bool = False`
2. **`train_phase2.py` parse_args()**: Add `--bc-return-weighted` argument
3. **`train_phase2.py` make_stage_b_train_step()**: Modify BC loss computation (guarded by config flag)
4. **`train_phase2.py` overrides**: Add `if args.bc_return_weighted: config.bc_return_weighted = True`

### Experiment F (Trend Channel) — DEFER until A-E results are in

Largest change, most risk. Only pursue if simpler experiments fail.
