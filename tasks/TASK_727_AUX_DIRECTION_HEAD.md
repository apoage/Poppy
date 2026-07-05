# TASK_727: Auxiliary Per-Asset Direction Head — Inject Dense Directional Gradient

**Status**: READY
**Priority**: CRITICAL — TASK_726 regression recovery
**Depends on**: TASK_726 (applied), encoder audit (`prompt-lib/out_encoder_audit/findings.md`)
**Blocks**: Bull alpha ≥ 0, balanced accuracy ≥ 75%

---

## Why TASK_726 Regressed — And What It Actually Proved

Class-balanced BC (TASK_726) gave the policy head a gradient that says *"trade more,"* but no gradient that says *"trade this way."* The result, after 200 epochs:

| Metric | Pre-726 | Post-726 |
|---|---|---|
| Model Trade% | ~6% | **54.3%** |
| Bal. Accuracy | ~37% | **35.9%** (random = 33%) |
| policy loss | 1.5–1.6 | **1.593** (ln 5 = 1.609 = uniform) |
| bc loss (weighted) | ~1.4 | **1.394** |
| Bull alpha | −24% | **−432%** |

The policy loss sitting at ~`ln 5` means the per-asset softmax is **near-uniform** over the 5 actions. Class balancing successfully killed HOLD dominance; it replaced it with *uniform random*, not *correct direction*.

This matches the encoder audit's prediction:
- The 128-dim global latent **does** carry bull-vs-bear signal (cosine 0.38 bull/bear).
- But **per-asset** within a single window, direction is compressed through a 64-asset → 128-dim bottleneck, then must be decompressed into 64×5 action logits by the policy head.
- Without a direction-specific gradient, the policy head can't extract that per-asset direction.

TASK_726 rebalanced *what to predict* (trade vs hold). TASK_727 must add *how to predict* (which direction per asset).

---

## Solution: Auxiliary Per-Asset Sign-of-Future-Return Head

Add a small head off the shared latent that predicts, **per asset**, the sign of the next K-step cumulative return. Supervise with BCE. Targets are already implicit in the batch (per-asset returns over the horizon), just need to be exposed.

This provides:
- **Dense gradient** — 64 labels per sample (not 4 like the oracle selection)
- **Directional specificity** — sigmoid(+) vs sigmoid(−) forces direction into the latent
- **Per-asset** — each asset has its own prediction, no cross-asset contamination
- **Free data** — no oracle replay, no recompute; uses `slot['returns']` already in the pool

Critical placement choice: predict from the **global latent** `latent_t` (128-dim), not the per-asset encoder embeddings. This is because the policy head also reads from `latent_t`, so pressure on the latent to carry per-asset direction directly helps the policy.

---

## Implementation

### Change 1 — new head module

**File**: `mcts_trader/jax_agent/muzero_model.py` (alongside existing heads).

```python
class DirectionHead(nn.Module):
    """Predicts sign of next K-step cumulative return per asset.

    Dense auxiliary signal to push per-asset direction into the latent.
    """
    n_assets: int = 64
    hidden_dim: int = 256
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, latent: jnp.ndarray) -> jnp.ndarray:
        # latent: (B, state_dim) → per-asset direction logits (B, n_assets)
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype,
                     param_dtype=self.param_dtype, name='dir_1')(latent)
        x = nn.gelu(x)
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype,
                         param_dtype=self.param_dtype, name='dir_ln')(x)
        # One scalar logit per asset
        logits = nn.Dense(features=self.n_assets, dtype=self.dtype,
                          param_dtype=self.param_dtype, name='dir_out')(x)
        return logits.astype(jnp.float32)  # (B, n_assets)
```

### Change 2 — wire head into MuZeroNetworks

Add a flag `use_direction_head: bool = False` and a module attribute:
```python
def setup(self):
    # ... existing ...
    if self.use_direction_head:
        self.direction_head = DirectionHead(
            n_assets=self.n_assets,
            hidden_dim=self.direction_hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
```

Emit the logits in `initial_inference`:
```python
out = {'policy_logits': ..., 'value': ..., 'latent_state': latent, ...}
if self.use_direction_head:
    out['direction_logits'] = self.direction_head(latent)  # (B, n_assets)
return out
```

### Change 3 — expose per-asset future returns in the batch

**File**: `train_phase2.py`, `AsyncBatchPipeline._prepare_numpy` (around line 468).

After computing `step_returns = slot['returns'][decision_indices]`:
```python
# TASK_727: Per-asset future-horizon return target for direction head
# Sign of cumulative return over [decision_idx, decision_idx + K)
horizon = self.config.direction_horizon  # e.g. 20 (same as oracle lookahead)
future_idx = decision_indices[:, None] + np.arange(1, horizon + 1)
future_idx = np.minimum(future_idx, len(slot['returns']) - 1)
future_rets = slot['returns'][future_idx]  # (n, horizon, n_assets)
future_cum = future_rets.sum(axis=1)       # (n, n_assets)
all_future_cum.append(future_cum.astype(np.float32))
```

Then in the returned dict:
```python
'future_cum_returns': np.concatenate(all_future_cum)[:batch_size],
```

### Change 4 — add to `_encode_jax` output dict

```python
'future_cum_returns': jnp.array(np_batch['future_cum_returns']),
```

### Change 5 — direction loss in `make_stage_b_train_step`

Capture in closure:
```python
use_direction = config.use_direction_head
direction_w = config.direction_loss_weight
```

Add new arg to the train step signature: `future_cum_returns`.

In `loss_fn`:
```python
# === Direction head: per-asset sign-of-future-return (TASK_727) ===
if use_direction and 'direction_logits' in root_out:
    dir_logits = root_out['direction_logits']           # (B, n_assets)
    # Binary target: +1 if cum return > 0, 0 otherwise
    dir_target = (future_cum_returns > 0).astype(jnp.float32)  # (B, n_assets)
    # Dead-zone: don't supervise near-zero returns (noise)
    dead_zone = jnp.abs(future_cum_returns) > 1e-4
    dir_ce = optax.sigmoid_binary_cross_entropy(dir_logits, dir_target)
    dir_loss = jnp.sum(dir_ce * dead_zone) / jnp.maximum(jnp.sum(dead_zone), 1.0)
    dir_acc = jnp.mean(
        ((jax.nn.sigmoid(dir_logits) > 0.5) == (dir_target > 0.5)) * dead_zone
    ) / jnp.maximum(jnp.mean(dead_zone), 1e-6)
else:
    dir_loss = jnp.array(0.0)
    dir_acc = jnp.array(0.5)

# === Total ===
total_loss = (policy_w * policy_loss +
              value_w * value_loss +
              consistency_w * (consistency_loss + reward_loss) +
              bc_weight * bc_loss +
              direction_w * dir_loss)
```

Emit `dir_loss` and `dir_acc` in metrics so you can monitor them.

### Change 6 — config fields

**File**: `phase2_config.py`:
```python
# === Auxiliary Direction Head (TASK_727) ===
use_direction_head: bool = True         # Per-asset sign-of-future-return probe
direction_horizon: int = 20             # Steps ahead (match oracle_lookahead)
direction_hidden_dim: int = 256         # Head capacity
direction_loss_weight: float = 0.2      # Loss weight; dense signal so start modest
```

### Change 7 — CLI flags

```python
parser.add_argument('--no-direction-head', action='store_true')
parser.add_argument('--direction-weight', type=float, default=None)
parser.add_argument('--direction-horizon', type=int, default=None)
```

Override:
```python
if args.no_direction_head: config.use_direction_head = False
if args.direction_weight is not None: config.direction_loss_weight = args.direction_weight
if args.direction_horizon is not None: config.direction_horizon = args.direction_horizon
```

---

## Why This Should Work When TASK_726 Didn't

TASK_726 rebalanced the **oracle policy gradient** so trade slots matter. But the oracle policy target for a 64-asset window names only 4 BUY/SELL assets and 60 HOLD — it's a 4/64 = 6% signal. Class-balancing scales the gradient on those 4 up by 16×, but there are still only 4 labels carrying direction.

TASK_727 provides **64 labels per sample** — one sign prediction per asset — computed from already-available future returns. This is **16× denser** than oracle selection, and crucially, every asset's direction is individually supervised. The latent has no choice but to preserve per-asset direction.

If direction is preserved in the latent, the policy head can learn to use it. If not, direction head accuracy will stay at 50% and we'll know the problem is the encoder itself (→ unfreeze).

---

## Training Command

Warmstart from a **pre-TASK_726 checkpoint** to avoid the uniform-policy collapse. TASK_726 can be re-layered on top once direction is established.

```bash
mkdir -p jax_ckpt_easy_phase2_727 && nohup python train_phase2.py --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl --stage B --epochs 300 --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 --factored --factored-temperature 0.3 --no-reset-value-head --value-target-mode nstep --td-discount 0.95 --value-weight 0.0 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 --direction-weight 0.2 --output-dir jax_ckpt_easy_phase2_727 > jax_ckpt_easy_phase2_727/stage_b_nohup.log 2>&1 &
```

Notes:
- `--bc-max-trade-weight 4.0` reduces TASK_726 pressure (was 64). Pair with the direction head so BC doesn't alone drive the policy to uniform.
- `--direction-weight 0.2` — will need tuning; if direction accuracy stays low, raise to 0.5.

---

## Diagnostic Gates

### After 50 epochs
- `dir_acc` in the metrics log should rise from 0.50 → **>0.55**.
  If flat at 0.50: latent lacks per-asset direction → unfreeze encoder (set `encoder_lr: 3e-5`).

### After 200 epochs
- `dir_acc` **>0.60** (ceiling is probably ~0.70 — noisy returns).
- Benchmark:
  - Bal. Acc **>50%** (was 35.9% post-726, ~37% pre-726).
  - Trade% **15–25%** (not 54% — direction head should sharpen, not flatten, the policy).
  - Per-asset match rate when oracle trades **>15%** (was 2.9% pre-726, ~20% random post-726).

### Failure modes and fallbacks
| Symptom | Meaning | Fallback |
|---|---|---|
| `dir_acc` stuck at 0.50 | Latent can't carry direction | Unfreeze encoder with `encoder_lr=3e-5` |
| `dir_acc` >0.60 but policy still uniform | Policy head not using direction | Increase `direction_loss_weight` OR add direction-logit concatenation into policy head input |
| Total loss explodes | Imbalanced grad | Lower `direction_loss_weight` to 0.05 |

---

## Why This Is Not Already Done By Oracle-Mask Head (TASK_695)

The codebase already has an optional `oracle_mask` head (TASK_695) that predicts whether the oracle selects each asset. That's a *selection* signal — "will this asset be in the k-selected set" — not a *direction* signal. Two assets can both be selected (one BUY, one SELL) and the oracle-mask head gives the same answer for both. The direction head answers the orthogonal question.

Run both if you want — they supervise different aspects of the latent.

---

## Files Modified

| File | Changes |
|------|---------|
| `mcts_trader/jax_agent/muzero_model.py` | Add `DirectionHead` module; wire into `MuZeroNetworks.setup()` and `initial_inference()` |
| `phase2_config.py` | Add `use_direction_head`, `direction_horizon`, `direction_hidden_dim`, `direction_loss_weight` |
| `train_phase2.py` | Expose `future_cum_returns` in `_prepare_numpy` and `_encode_jax`; add `dir_loss` to total; pass through signature of `stage_b_train_step` |
| `tools/benchmark_easy.py` | (Optional) expose `dir_acc` in benchmark output to monitor |

No checkpoint incompatibility for the base model (head is additive; existing params load fine; new `direction_head` subtree initializes fresh). No encoder architecture changes.
