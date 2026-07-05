# Config Audit: easy_config.py vs Task Specs

**Date**: 2026-02-23
**Config file**: `/home/lukas/Trader/easy_config.py`
**Orchestrator**: `tools/orchestration/e-dojo_v1.py`

---

## Full Config Table

| Config Field | Default | Large Preset | e-dojo CLI Override | Task | Expected Per Spec | Match? |
|---|---|---|---|---|---|---|
| `state_dim` | 64 | 128 | -- | TASK_683 | 128 | YES |
| `n_assets` | 64 | -- | -- | Original | 64 | YES |
| `action_space_size` | 320 | -- | -- | Original | 320 | YES |
| `n_channels` | 34 | -- | -- | TASK_689 | 34 | YES |
| `H_cycles` | 4 | 6 | -- | Original | 6 | YES |
| `L_cycles` | 6 | 8 | -- | Original | 8 | YES |
| `hidden_dim` | 256 | 512 | -- | Original | **DEAD** -- never read by model code | N/A |
| `trm_hidden_dim` | None | **1024** | -- | TASK_684 | 1024 per spec, but spec is wrong (should be 2048) | **WRONG** |
| `num_heads` | 8 | 16 | -- | Original | 16 | YES |
| `embed_dim` | 32 | 64 | -- | TASK_683 | 64 | YES |
| `encoder_num_heads` | 4 | 8 | -- | TASK_683 | 8 | YES |
| `encoder_version` | 2 | -- | `--encoder-version 2` | TASK_676 | 2 (V2, no causal leak) | YES |
| `compute_dtype` | "float32" | -- | -- | TASK_686 | "float32" | YES |
| `dropout` | 0.0 | -- | -- | Original | 0.0 | YES |
| `epochs` | 100 | 200 | `--epochs 2500` | Original | 2500 (CLI wins) | YES |
| `batch_size` | 256 | -- | `--batch-size 1024` | Original | 1024 (CLI wins) | YES |
| `steps_per_epoch` | 200 | -- | `--steps-per-epoch 400` | TASK_675 | 400 (CLI wins) | YES |
| `learning_rate` | 3e-4 | -- | -- | Original | 3e-4 | YES |
| `weight_decay` | 1e-4 | -- | -- | Original | 1e-4 | YES |
| `warmup_steps` | 1000 | -- | -- | Original | 1000 | YES |
| `grad_clip` | 5.0 | -- | -- | TASK_685 | 5.0 (was 1.0) | YES |
| `policy_weight` | 1.0 | -- | -- | Original | 1.0 | YES |
| `value_weight` | 0.5 | -- | -- | Original | 0.5 | YES |
| `value_scale` | 2.0 | -- | -- | TASK_674 | 2.0 | YES |
| `trade_loss_weight` | 60.0 | -- | -- | TASK_149 (DEPRECATED) | Deprecated | YES |
| `weight_warmup_epochs` | 0 | -- | -- | TASK_677 | 0 (disabled to break turtle) | YES |
| `oracle_k` | 4 | -- | -- | TASK_681 | 4 | YES |
| `oracle_lookahead` | 20 | -- | -- | TASK_152 | 20 | YES |
| `transaction_cost` | 0.001 | -- | -- | Original | 0.001 | YES |
| `min_alpha_threshold` | 0.0005 | -- | -- | TASK_679 | 0.0005 | YES |
| `state_aware_oracle` | True | -- | -- | TASK_141 | True | YES |
| `oracle_multipass` | False | -- | NOT passed by e-dojo | TASK_673 | False default | YES |
| `oracle_n_passes` | 10 | -- | -- | TASK_673 | 10 | YES |
| `oracle_noise_scale` | 0.03 | -- | -- | TASK_673 | 0.03 (tuned from 0.1) | YES |
| `oracle_sharpen_temperature` | 0.1 | -- | -- | TASK_677 | 0.1 | YES |
| `oracle_entropy` | False | -- | -- | TASK_691/692 | False (superseded) | YES |
| `oracle_drop_rate` | 0.15 | -- | -- | TASK_691 | 0.15 | YES |
| `oracle_phantom_rate` | 0.05 | -- | -- | TASK_691 | 0.05 | YES |
| `entropy_warmup_epochs` | 50 | -- | -- | TASK_691 | 50 | YES |
| `use_oracle_mask_head` | False | -- | `--oracle-mask-head` (flag) | TASK_695 | False default | YES |
| `oracle_mask_hidden_dim` | 256 | -- | -- | TASK_695 | 256 | YES |
| `oracle_mask_weight` | 0.1 | -- | -- | TASK_695 | 0.1 | YES |
| `autoregressive_state` | True | -- | -- | TASK_692 | True | YES |
| `state_refresh_fraction` | 0.2 | -- | -- | TASK_692 | 0.2 | YES |
| `trade_sample_prob` | 0.30 | -- | -- | TASK_687 (was 0.15/TASK_680) | 0.30 | YES |
| `window_size` | 64 | -- | -- | TASK_152 | 64 | YES |
| `episode_length` | 400 | -- | -- | Original | 400 | YES |

---

## Critical Config Flags (Detailed)

### `trm_hidden_dim` = 1024 (large) -- **REGRESSION**

The large preset sets `trm_hidden_dim=1024`. Before TASK_684, `hidden_dim` defaulted to `state_dim * 16`. With old `state_dim=64`, that was 1024. With new `state_dim=128` (TASK_683), it should be 2048. By capping at 1024, the reasoning FFN capacity is halved relative to the wider representation pipe. This is the confirmed primary regression -- ~23M parameters lost.

### `hidden_dim` = 512 (large) -- **DEAD CONFIG**

`EasyConfig.hidden_dim` is set in all presets but **never read** by any model or trainer code. Only `trm_hidden_dim` is wired to MuZeroNetworks. This is actively misleading.

### `embed_dim` / `encoder_num_heads` -- **DEAD with V2 encoder**

With `encoder_version=2`, the V2 encoder (`AssetTransformerEncoder`) uses its own defaults (`embed_dim=32`, `num_heads=4`). TRM does NOT pass the config values to V2 (`trm_flax.py:248`). These config values only take effect with V3.

### `compute_dtype` = "float32" -- OK

Default is float32. bfloat16 available but not active. No CLI flag exists to change it.

### `oracle_multipass` = False -- **NOT passed by e-dojo**

Multipass oracle, temperature sharpening, and soft targets are all dormant in production.

### `trade_sample_prob` = 0.30 -- **Diluted implementation**

The coin-flip sampling operates once per episode visit (choosing the start of a contiguous block), not once per sample. With `take_count=20`, the effective per-sample trade bias is ~10-20x weaker than intended.

### `weight_warmup_epochs` = 0 -- **Intentional**

TASK_677 disabled warmup to break turtle trap. Full ~25x trade weight from step 1.

---

## Config Mismatches

| Config Field | Actual | Spec Intent | Issue |
|---|---|---|---|
| `trm_hidden_dim` | 1024 (large) | Should be 2048 or None for state_dim=128 | Spec error: carried over old value |
| `hidden_dim` | 512 (large) | N/A | Dead config field, never used |
| `embed_dim` | 64 (large, V2) | 64 per TASK_683 | Dead for V2 encoder (hardcoded at 32) |
| `encoder_num_heads` | 8 (large, V2) | 8 per TASK_683 | Dead for V2 encoder (hardcoded at 4) |
| `trade_sample_prob` | 0.30 | 30-35% trade windows per batch | Actual: ~3-8% due to contiguous block sampling |
