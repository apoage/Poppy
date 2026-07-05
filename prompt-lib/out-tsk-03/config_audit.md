# Config Audit: Spec vs Reality

| Config Field | Default Value | Which Task Set It | Expected Value Per Spec | Match? |
|-------------|---------------|-------------------|------------------------|--------|
| `encoder_version` | 2 | TASK_676 | 2 (V2) or 3 (V3) | ✓ |
| `n_channels` | 34 | TASK_689 | 34 | ✓ |
| `trm_hidden_dim` | `None` (but 1024 in Large) | TASK_684 | `None` (decoupled) | ✗ (Regressed in Large preset) |
| `compute_dtype` | "float32" | TASK_686 | "bfloat16" (implied by title) | ✗ (Default is float32) |
| `grad_clip` | 5.0 | TASK_685 | 5.0 | ✓ |
| `weight_warmup_epochs` | 0 | TASK_677 | 50 (TASK_151) | ✗ (Disabled by TASK_677) |
| `oracle_k` | 4 | TASK_681 | 4 | ✓ |
| `oracle_multipass` | `False` | TASK_673 | `True` (for soft targets) | ✗ (Feature exists but off) |
| `oracle_sharpen_temperature` | 0.1 | TASK_677 | 0.1 | ✓ |
| `trade_sample_prob` | 0.30 | TASK_687 | 0.30 | ✓ |
| `use_oracle_mask_head` | `False` | TASK_695 | `False` | ✓ |
| `autoregressive_state` | `True` | TASK_692 | `True` | ✓ |
| `value_scale` | 2.0 | TASK_670B | 2.0 | ✓ |

## Critical Mismatches

1. **`trm_hidden_dim`**: In `get_large_config()`, this is hardcoded to `1024`.
   - **Spec (TASK_684)**: "Decouple hidden dim... default to state_dim * 16".
   - **Reality**: Large config has `state_dim=128`. `128 * 16 = 2048`. The hardcoded `1024` reduces capacity by 50%.

2. **`weight_warmup_epochs`**: Default is `0`.
   - **Spec (TASK_151)**: "Signal Soft Start... Warmup ramp for trade weight".
   - **Reality**: Disabled (0). Logic in `train_easy.py` handles it, but config turns it off. This re-introduces "gradient shock" risks.

3. **`oracle_multipass`**: Default is `False`.
   - **Spec (TASK_673)**: "Multipass Oracle... Soft targets".
   - **Reality**: Implementation exists but is disabled by default. The model is training on hard targets (one-hot), missing the benefit of soft labels for ambiguous states.
