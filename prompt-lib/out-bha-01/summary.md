# Executive Summary: Trade Action Learning Collapse

## Verdict

The agree_trade collapse (15% → 7%) is caused by **two compounding regressions**, with a possible minor third contributor:

| # | Regression | Confidence | Impact | Fix Effort |
|---|-----------|------------|--------|------------|
| 1 | Hidden dim halved (2048 → 1024) | **CONFIRMED** | ~23M params lost, minority-class discrimination destroyed | 1 line |
| 2 | Model-state rollout bootstrap tax | **LIKELY** | Noisy state channels compound with reduced capacity | Config flag |
| 3 | Oracle mask head noise gradients | **POSSIBLE** | 0.8% loss mass, trivially solvable task | Config flag |

## Root Cause

Commit 06c44ab (TASK_692/693) added `hidden_dim=getattr(config, 'trm_hidden_dim', None)` to the MuZeroNetworks constructor. Before this commit, `hidden_dim` was not passed, defaulting to `state_dim * 16 = 2048`. The large config sets `trm_hidden_dim=1024`, silently halving the reasoning FFN across all 54 refiner cycles (48 latent + 6 output).

The model dropped from ~48M to ~25.76M parameters. Hold prediction (96% of decisions) is unaffected because it's the default. Trade detection (3.9% of decisions) requires fine-grained nonlinear discrimination that the halved FFN can no longer provide.

This was compounded by the same commit introducing model-state rollout (`autoregressive_state=True`), which replaced clean oracle state channels with noisy model-generated channels — a harder learning problem presented to a smaller model.

## Recommended Fix Order

### Step 1: Restore hidden_dim (5 minutes)

**Option A** — Update `easy_config.py` (preferred, explicit):
```python
def get_large_config() -> EasyConfig:
    return EasyConfig(
        ...
        trm_hidden_dim=2048,  # Restore full reasoning capacity (was 1024)
    )
```

**Option B** — Force None in `easy_trainer.py` (quick, backward-compat):
```python
hidden_dim=None,  # Use default state_dim*16 = 2048
```

### Step 2: A/B test autoregressive_state (1 run)

After restoring hidden_dim, run with `autoregressive_state=False` to confirm the model recovers to ~15% agree_trade at epoch 80. If it does, the bootstrap tax is real and needs a warmup schedule (oracle channels first N epochs, then transition to model channels).

### Step 3: Disable oracle mask head (optional)

Set `use_oracle_mask_head=False` (already the default; e-dojo doesn't pass `--oracle-mask-head`). If the head IS active in your run, disable it — the 93.7% accuracy = predict-all-zeros baseline means it's learning nothing useful.

## Changes Ruled Out

| Change | Why Ruled Out |
|--------|---------------|
| Returns alignment (BUG-1) | Shape trace correct: `returns_aligned[idx]` = return at decision point of window `idx` |
| Trade-biased sampling (BUG-7) | Only filters last 8 of ~424 timesteps — negligible |
| Confidence channel (BUG-2) | Carry init/unpack order correct, emits prev_conf (matches inference) |
| Buffer param ID (LATENT) | Sentinel -1 sorts FIFO correctly, no functional change |
| Optimizer warmup guard | `min(1000, 999999) = 1000` — no effect for large preset |
| V2 encoder params | `embed_dim`/`encoder_num_heads` NOT passed to V2 encoder by TRM — confirmed at `trm_flax.py:248` |

## Cross-Validation Summary

| Claim | Verified By |
|-------|-------------|
| hidden_dim not passed before | `git show 06c44ab~1:easy_trainer.py` — 7-param constructor |
| None defaults to state_dim*16 | `trm_flax.py:112` — explicit fallback |
| Large config sets 1024 | `easy_config.py:175` — `trm_hidden_dim=1024` |
| V2 encoder ignores config params | `trm_flax.py:248` — only `state_dim`, `dtype`, `param_dtype` passed |
| e-dojo uses --no-latent-cache | `e-dojo_v1.py:887` — always appended |
| e-dojo doesn't pass --oracle-mask-head | `e-dojo_v1.py:876-893` — not in args list |
| autoregressive_state defaults True | `easy_config.py:91` — `autoregressive_state: bool = True` |

---

*Analysis performed: 2026-02-23*
*Scope: Commits 06c44ab..77cb089 (4 commits, 8 changes, 5 training-critical files)*
*Cross-validated against: trm_flax.py, trm_encoder_v2.py, e-dojo_v1.py, git history*
