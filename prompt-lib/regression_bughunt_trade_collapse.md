# Regression Bug Hunt: Trade Action Learning Collapse

> **OUTPUT_DIR**: `{{OUTPUT_DIR}}`
> Replace `{{OUTPUT_DIR}}` with a unique folder path before use (e.g. `prompt-lib/out-reg01`).
> Each reviewer gets their own folder. Do NOT read other folders under `prompt-lib/out-*`.

You are investigating a **training performance regression**. Between two training runs on the same architecture, trade action learning collapsed. Your job is to find the specific code change(s) that killed it.

## The Symptom

Fresh training run, large preset (25.76M params), 80 epochs in. Model is a **turtle** — predicting hold everywhere. Trade actions are not being learned.

### CSV Evidence (Branch A, epochs 60–77)

```
epoch | agree_trade | agree_hold | oracle_agreement | pred_hold_ratio | oracle_hold_ratio | adaptive_trade_weight | oracle_mask_loss
60    | 0.058       | 0.953      | 0.506            | 0.951           | 0.961             | 24.78                 | 0.231
65    | 0.067       | 0.944      | 0.506            | 0.941           | 0.961             | 24.67                 | 0.226
70    | 0.064       | 0.945      | 0.504            | 0.942           | 0.961             | 24.72                 | 0.226
75    | 0.069       | 0.943      | 0.506            | 0.940           | 0.961             | 24.67                 | 0.226
77    | 0.076       | 0.934      | 0.505            | 0.931           | 0.961             | 24.63                 | 0.226
```

- `oracle_agreement` = `0.5 * agree_trade + 0.5 * agree_hold` (balanced accuracy)
- `agree_trade` = 6-7% → model almost never picks the correct non-hold action
- `pred_hold_ratio` ≈ 0.94 → model predicts hold ~94% of the time

### Previous Run (same architecture, earlier codebase)

- Same large preset, same Bella data
- agree_trade ~15% by epoch 80, climbing to ~20%+ by epoch 150
- oracle_agreement reached 55% (balanced) by epoch 165
- Loss curve was decreasing

### The Regression

| Metric | Previous Run @ E80 | Current Run @ E80 | Delta |
|--------|-------------------|-------------------|-------|
| agree_trade | ~15% | ~7% | **-53%** |
| oracle_agreement | ~52% | ~50.5% | -3% |
| Learning trend | Climbing | Flat | **Stalled** |

---

## The Suspect Commits

Exactly 5 commits separate the last known-good codebase from the current one:

```
77cb089 TASK_695: Auxiliary oracle mask prediction head (RL²F-inspired)
a86c206 BUG-2 completion: emit prev_conf in scan output for training alignment
92fc8a5 Avoid full-database GROUP BY scan in benchmark price loading
f6669f4 TASK_694: Fix all 11 adversarial bug hunt bugs + 1 latent issue
06c44ab TASK_692/693: Model-state rollout + fix 4 bugs in position/slicing logic
```

Commit `92fc8a5` only touches benchmark price loading (read-only path) — **skip it**.

The remaining 4 commits changed **train_easy.py** (+573 lines) and **easy_trainer.py** (+36 lines). These are your targets.

---

## Specific Changes to Investigate

### Change 1: Model-State Rollout (TASK_692/693) — `train_easy.py`

**What it does**: Added autoregressive rollout where the model's own actions generate state channels (instead of oracle's actions). This is a 240-line addition.

**Risk**: This is a massive change to what the model sees during training. If the rollout is wrong, every state channel is wrong.

**Key code**: `_build_model_state_trajectory()` (~line 173–300)
- Runs `jax.lax.scan` over episode timesteps
- At each step: builds state → runs model forward pass → takes argmax action → updates state
- Produces `(T, 6, n_assets)` state channels + `(T, n_assets)` confidence

**Check**:
- Is the scan init state correct? (5 zeros → 6 after BUG-2)
- Does the scan properly stop gradients? (`jax.lax.stop_gradient`)
- Are returns aligned correctly to window indices? (BUG-1 fix)
- Is `model_apply_fn` being called with correct signature?

### Change 2: Returns Alignment (BUG-1 fix) — `train_easy.py:570-576, 789-793`

**What it changed**:
```python
# BEFORE (working):
state_channels, confidence = self._model_state_rollout(
    observations,   # (n_valid, 27, ws, n_assets)
    log_returns,    # (T, n_assets)
    ...
)

# AFTER (broken?):
ws = self.config.window_size
returns_aligned = log_returns[ws - 1:]
state_channels, confidence = self._model_state_rollout(
    observations,       # (n_valid, 27, ws, n_assets)
    returns_aligned,    # (T - ws + 1, n_assets)
    ...
)
```

**Risk**: HIGH. This changes what returns the model-state rollout receives. If the alignment is wrong, PnL computation inside the scan is wrong → state channels are wrong → model gets corrupted gradient signal.

**Check**:
- `observations` has shape `(n_valid, 27, ws, n_assets)` where `n_valid = T - ws + 1`
- `returns_aligned` has shape `(T - ws + 1, n_assets)` — matches observations?
- Inside the scan, `returns[idx]` must be the return at the decision point of window `idx`
- The batch path (~line 789) has a different slicing: `log_returns[i, ws - 1:valid_lengths[i] - 1]` — is this consistent with the single-episode path?

### Change 3: Confidence Channel (BUG-2 fix) — `train_easy.py:237-240, 296-300`

**What it changed**:
```python
# BEFORE:
# conf_tiled = jnp.zeros((1, ws, n_assets))  # Zero placeholder
# return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow), (state_ch, model_confidence)

# AFTER:
conf_tiled = jnp.tile(prev_conf[None, None, :], (1, ws, 1))  # Feed prev confidence
return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, prev_conf)
```

**Risk**: MEDIUM. The carry now has 6 elements (was 5). The emitted confidence is `prev_conf` not `model_confidence`.

**Check**:
- Init state added `jnp.zeros(n_assets)` for `prev_confidence` — correct position in tuple?
- The carry unpacking: `pos, pnl, steps, h_fast, h_slow, prev_conf = carry` — matches init order?
- The observation now includes non-zero confidence from step i-1 → does this match benchmark inference?

### Change 4: Trade-Biased Sampling (BUG-7 fix) — `train_easy.py:1130-1138`

**What it changed**:
```python
# BEFORE:
if (len(trade_indices) > 0 and max_offset > 0 and self.rng.random() < self.config.trade_sample_prob):
    idx = self.rng.choice(trade_indices)
    offset = int(np.clip(idx, 0, max_offset))

# AFTER:
valid_trades = trade_indices[trade_indices <= max_offset]
if (len(valid_trades) > 0 and max_offset > 0 and self.rng.random() < self.config.trade_sample_prob):
    offset = int(self.rng.choice(valid_trades))
```

**Risk**: MEDIUM. If `valid_trades` is often empty (most trade indices exceed max_offset), trade-biased sampling effectively becomes disabled → the model sees fewer trade examples → agree_trade drops.

**Check**:
- What is typical `max_offset`? (episode_length - take_count)
- What is typical trade_indices distribution? (concentrated at end of episode?)
- If filtering removes most trade indices, the coin-flip path never fires

### Change 5: Buffer Param ID Fix (LATENT) — `train_easy.py:1070-1075`

**What it changed**:
```python
# BEFORE:
for idx in sorted(indices_to_refresh, reverse=True):
    if idx < len(self._buffer_param_ids):
        del self._buffer_param_ids[idx]

# AFTER:
for idx in indices_to_refresh:
    if idx < len(self._buffer_param_ids):
        self._buffer_param_ids[idx] = -1  # Sentinel
```

**Risk**: LOW-MEDIUM. Previous code could cause list length desync. New code uses sentinel values. If sentinel logic is wrong, old episodes never get refreshed → model trains on stale data.

### Change 6: Oracle Mask Head (TASK_695) — `easy_trainer.py:202-214`

```python
oracle_mask_loss = optax.sigmoid_binary_cross_entropy(oracle_mask_logits, oracle_mask_targets).mean()
# Added to total_loss with weight 0.1
```

**Risk**: LOW. Contributes ~0.023 to total_loss (~2.8). Negligible gradient mass. But verify:
- Does `oracle_mask_accuracy = 0.937` from epoch 0 indicate a trivially-solvable task? (If mask is constant, this head is useless overhead)
- Are the `oracle_mask_targets` correctly aligned to the same timesteps as policy targets?

### Change 7: Optimizer Warmup Guard — `easy_trainer.py:80-84`

```python
# BEFORE:
warmup_steps=config.warmup_steps,
decay_steps=config.epochs * config.steps_per_epoch,

# AFTER:
effective_warmup = min(config.warmup_steps, max(total_steps - 1, 0))
warmup_steps=effective_warmup,
decay_steps=max(total_steps, effective_warmup + 1),
```

**Risk**: LOW-MEDIUM for normal runs. But check: with `epochs=2500, steps_per_epoch=200`, `total_steps=500000`. Default `warmup_steps` is what? If warmup_steps was previously larger than total_steps (and silently clipped by optax), the new code might produce a different LR schedule.

### Change 8: Model Construction — `easy_trainer.py:51-58`

```python
# ADDED:
encoder_version=getattr(config, 'encoder_version', 2),
hidden_dim=getattr(config, 'trm_hidden_dim', None),
embed_dim=getattr(config, 'embed_dim', 32),
encoder_num_heads=getattr(config, 'encoder_num_heads', 4),
dtype=compute_dtype,
param_dtype=jnp.float32,
```

**Risk**: HIGH. These parameters were previously using MuZeroNetworks defaults. If the defaults differ from what the config specifies, the model architecture changes. Check:
- What is `config.encoder_version`? If it's 3 (V3 encoder) vs the old default of 2, the entire encoder is different.
- What is `config.embed_dim`? The default here is 32, but the model's built-in default might be different.
- What is `compute_dtype`? If bfloat16, training sees reduced precision vs the previous float32.

---

## Diagnostic Experiments (for the implementer, after your analysis)

1. **A/B test oracle mask**: Run with `use_oracle_mask_head=False` — if agree_trade goes to ~15%, the mask head is the culprit (unlikely given 0.023 loss contribution)
2. **Check encoder version**: Print `config.encoder_version` at startup — if it changed from 2→3 or vice versa, that's a different model
3. **Check compute_dtype**: Print it — if bfloat16 was not used before, precision change could affect trade learning
4. **Trade index stats**: Log `len(valid_trades)` vs `len(trade_indices)` in BUG-7 path — if valid_trades is usually empty, trade sampling is dead
5. **Returns shape check**: Add assertion in model-state rollout: `assert returns.shape[0] == market_obs.shape[0]` — if this fails, BUG-1 alignment is wrong

---

## Mandatory Process

### Phase 1: Read the Code

Read these files **in full** before analysis:

| File | What to look for |
|------|-----------------|
| `train_easy.py` | Model-state rollout, returns alignment, sampling, episode generation |
| `easy_trainer.py` | Loss computation, optimizer schedule, model construction |
| `easy_config.py` | All config defaults — especially encoder_version, compute_dtype, embed_dim |
| `tools/benchmark_easy.py` | Inference path — does it match training after these changes? |
| `mcts_trader/jax_agent/muzero_model.py` | Model architecture — encoder version dispatch |

### Phase 2: Rank Hypotheses

For each of the 8 changes above, determine:
1. Could this change reduce agree_trade from 15% to 7%?
2. Would this change affect agree_hold? (It shouldn't — agree_hold is stable at 94%)
3. Is this change consistently applied in both training AND inference?

### Phase 3: Write Findings

Use the same format as `adversarial_bughunt_easy_mode.md`:

```markdown
### REGRESSION-N: [Title]

**Suspect Commit**: [hash + title]
**Change**: [which of the 8 changes above]
**Confidence**: CONFIRMED / LIKELY / POSSIBLE

**Why It Kills Trade Learning**:
[Specific mechanism — what gradient signal is lost or corrupted]

**Evidence**:
[Code quotes, shape traces, logic]

**Proposed Fix**:
[Concrete code change]
```

---

## Output

Save ALL output to `{{OUTPUT_DIR}}/`:

| File | Contents |
|------|----------|
| `hypothesis_ranking.md` | All 8 changes ranked by likelihood of causing regression |
| `findings.md` | REGRESSION-N entries for confirmed/likely causes |
| `summary.md` | Executive summary with recommended fix order |

---

## Anti-Patterns

1. **Do NOT investigate code that didn't change** — the regression is in these 5 commits
2. **Do NOT report style issues** — you are finding the trade learning killer
3. **Do NOT blame the oracle mask head without evidence** — its loss contribution is 0.023/2.8 = 0.8% of total
4. **Do NOT ignore Change 8 (model construction)** — silently changing encoder_version or dtype is exactly the kind of thing that breaks training without any obvious error

---

*Created: 2026-02-23*
*Purpose: Diagnose agree_trade collapse from ~15% to ~7% after commits 06c44ab..77cb089*
*Scope: 4 commits, 8 specific code changes, 5 training-critical files*
