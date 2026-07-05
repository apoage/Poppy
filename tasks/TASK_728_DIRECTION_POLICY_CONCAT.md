# TASK_728: Direction Logits → Policy Head (Structural) + Encoder Unfreeze

**Status**: READY
**Priority**: CRITICAL — TASK_727 stalled at dir_acc=0.554, policy uniform
**Depends on**: TASK_727 (direction head exists, latent weakly encodes direction)
**Blocks**: Bal.Acc > 40%, bull alpha > −20%

---

## Why TASK_727 Stalled

After 300 epochs with `direction_loss_weight=0.2`:
- `dir_acc` plateaued at 0.553 — latent ceiling, not head ceiling
- `policy_loss` stuck at ~1.59 (≈ln 5) — policy uniform per asset
- Benchmark: Bal.Acc 32.9% vs pre-726 37% → **regression**
- Per-action: "when oracle BUYs, model slightly more likely to SELL than BUY"

TASK_727 placed the aux loss where it could shape the *latent*. But:
1. The auxiliary loss has no structural path into the policy-head output.
2. The encoder is frozen, so the latent can't actually grow the direction signal past ~0.55.

## The Two Coupled Fixes

### Fix A — Direction logits injected into policy logits (structural)

Add a residual pathway: the 64-d `direction_logits` is projected to 320-d policy-space and added to the policy logits. Weights initialize to zero so initial behavior is unchanged; gradient from `policy_loss` pulls them up once direction signal is informative.

```python
# In MuZeroNetworks, after prediction() and direction_head():
class PolicyDirectionRefiner(nn.Module):
    n_assets: int = 64
    n_actions_per_asset: int = 5

    @nn.compact
    def __call__(self, direction_logits):
        # direction_logits: (B, n_assets)
        # Project each per-asset direction into a 5-logit correction
        per_asset_bias = nn.Dense(
            features=self.n_assets * self.n_actions_per_asset,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='dir_to_policy',
        )(direction_logits)
        return per_asset_bias  # (B, 320)
```

Apply as: `policy_logits = policy_logits + refiner(direction_logits)`.

Zero init guarantees checkpoint compatibility — pre-728 model unchanged at step 0. Training pulls the weights.

### Fix B — Unfreeze encoder at low LR (3e-5)

The `dir_acc=0.553` plateau is the latent-capacity signal. Encoder frozen from Phase 1 Foxglove cannot grow per-asset direction representation. Unfreeze with separate slow LR so Phase 1 structure is preserved:

- Main params (reasoning, dynamics, heads): `policy_lr=3e-4`
- Encoder: `encoder_lr=3e-5` (10× slower)

Implementation: extend the multi_transform optimizer to three groups — `train_main` / `train_encoder` / `freeze` — each with its own schedule.

---

## Implementation

### Change 1 — `muzero_model.py`: `PolicyDirectionRefiner`

Add the module after `DirectionHead`. Add flag `use_direction_policy_bias: bool = False` to `MuZeroNetworks`. In `setup()`:

```python
if self.use_direction_policy_bias:
    self.policy_direction_refiner = PolicyDirectionRefiner(
        n_assets=self.n_assets,
        n_actions_per_asset=5,
    )
```

In `initial_inference` and `prediction_only`, after computing both `policy_logits` and `direction_logits`:

```python
if self.use_direction_head and self.use_direction_policy_bias:
    bias = self.policy_direction_refiner(direction_logits)
    policy_logits = policy_logits + bias
    result['policy_logits'] = policy_logits
    result['direction_bias_norm'] = jnp.linalg.norm(bias, axis=-1)  # metric
```

### Change 2 — `phase2_config.py`

```python
# TASK_728: Direction-logits → policy-head structural bias
use_direction_policy_bias: bool = True

# Encoder unfreeze at slow LR (TASK_728 Fix B)
encoder_lr: float = 3e-5   # Non-zero unfreezes with separate slow schedule
```

(`encoder_lr=0.0` currently keeps encoder frozen; change default to 3e-5, or leave 0.0 and expose via CLI.)

### Change 3 — `train_phase2.py`: three-group optimizer

Replace `create_stage_b_optimizer` mask with three labels. Build a `train_encoder` optimizer that uses its own warmup-cosine schedule with peak=`encoder_lr`. `make_stage_b_mask` returns `'encoder'` for encoder keys (if encoder_lr > 0), `'train'` for other trainable, `'freeze'` for `oracle_mask`.

### Change 4 — CLI

```python
parser.add_argument('--encoder-lr', type=float, default=None,
                    help='Encoder learning rate (0=frozen, default 3e-5)')
parser.add_argument('--no-direction-policy-bias', action='store_true')
```

Override in main:
```python
if args.encoder_lr is not None: config.encoder_lr = args.encoder_lr
if args.no_direction_policy_bias: config.use_direction_policy_bias = False
```

### Change 5 — Warmstart

Use the original pre-726 checkpoint `jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl` — NOT the 727 final. The 727 policy head settled in a worse basin (symmetric-noise equilibrium). Pre-726's "mostly HOLD with scattered correct trades" is a better starting point.

---

## Training Command

```bash
mkdir -p jax_ckpt_easy_phase2_728 && nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl \
  --stage B --epochs 300 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.3 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --output-dir jax_ckpt_easy_phase2_728 \
  > jax_ckpt_easy_phase2_728/stage_b_nohup.log 2>&1 &
```

Note `--direction-weight 0.5` (up from 0.2). Once the structural path exists, stronger direction pressure should help not hurt.

---

## Diagnostic Gates

### E50
- `dir_acc` > 0.58 (was stuck at 0.55 when encoder frozen).
  If still 0.55: encoder unfreeze isn't helping — either encoder_lr too low or architecture-bound.
- `direction_bias_norm` > 0.1 (gradient flowing to refiner).

### E150
- `dir_acc` > 0.62.
- `policy_loss` < 1.55 (down from 1.59 ≈ ln 5).
- `direction_bias_norm` in 0.3–1.0 range (meaningful but not dominant).

### E300 / benchmark
- Bal.Acc > 40% (pre-726 baseline was 37%).
- Bull alpha > −15%.
- "When oracle BUYs → model BUY%" > "When oracle BUYs → model SELL%" by ≥ 3pp (correlation, not inversion).

### Failure modes

| Symptom | Meaning | Fallback |
|---|---|---|
| `dir_acc` still 0.55 | Encoder architecture can't grow direction | TASK_729 encoder width/temporal changes |
| `dir_acc` > 0.6, policy still uniform | Refiner not pulling / zero-init basin | Warm refiner with hand-set priors (SELL=−1, BUY=+1) |
| Encoder grads collapse other tasks | encoder_lr too high | Drop to 1e-5 |
| `direction_bias_norm` → very large | Runaway refiner | Add layer norm or clip before adding |

---

## Why This Is the Right Fix

TASK_726 said "trade more" → model traded uniformly random.
TASK_727 said "latent, know direction" → latent weakly learned it; policy ignored it.
TASK_728 says "policy, READ direction" and "encoder, grow the signal" — both at once. Structural + capacity.

If TASK_728 stalls, the problem is encoder architecture, not training loss.
