# TASK_735: Auxiliary Log-Price Prediction Head (α-v3)

**Status**: ACTIVE (as of 2026-04-23). Corrected prediction-head pathway test after TASK_734's math bug.
**Priority**: HIGHEST in the prediction-head pathway. If this ceilings too, pathway is exhausted.
**Depends on**: 32-ch Phase 1 checkpoint `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`.
**Rationale**: `lore/notes_2026-04-22.md §48`.

---

## Scope

Add a small MLP head that takes the global latent state and predicts log-prices at multiple future horizons: `t+1, t+5, t+10, t+20`. Supervised with plain multi-horizon MSE against actual future log-prices extracted from the training windows. No flow matching (§48 proved single-step flow matching collapses to MSE identically — skipping the illusion).

**What this tests that nothing else has:** magnitude-aware forward prediction. direction_head learns sign only (binary). rank_head learns relative ordering (cross-sectional). Neither captures *how much* the price moves. The aux price head explicitly does.

**Pass/fail is informative either way:**
- dir_acc breaks 0.60 → magnitude signal was the missing piece → escalate to β (Bella distillation with same head architecture, richer teacher).
- dir_acc stays at 0.57 → magnitude doesn't help either → all latent-side prediction heads tested; ceiling is information-side; pivot to §46 Tier A1 (dense oracle) or Tier D (more data) or abandon prediction-head pathway.

---

## Design

### New module: `AuxPriceHead`

File: add to `mcts_trader/jax_agent/muzero_model.py` (alongside DirectionHead, RankHead, PerAssetPolicyHead).

```python
class AuxPriceHead(nn.Module):
    """TASK_735: Multi-horizon log-price prediction head.

    Takes the global latent (B, state_dim) and predicts log-prices for all
    n_assets at K future horizons. Supervised with per-horizon MSE against
    actual log-prices from the training window.
    """
    n_assets: int = 64
    n_horizons: int = 4       # [t+1, t+5, t+10, t+20] by default
    hidden_dim: int = 256
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, latent: jnp.ndarray) -> jnp.ndarray:
        # latent: (B, state_dim) → (B, n_horizons, n_assets)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype,
                     param_dtype=self.param_dtype, name='aux_price_1')(
            latent.astype(self.dtype))
        x = nn.gelu(x)
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype,
                         param_dtype=self.param_dtype, name='aux_price_ln')(x)
        # Zero-init final: head outputs 0 at init, model learns the non-trivial signal.
        out = nn.Dense(self.n_horizons * self.n_assets,
                       kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros,
                       dtype=self.dtype,
                       param_dtype=self.param_dtype, name='aux_price_out')(x)
        batch = out.shape[0]
        return out.reshape(batch, self.n_horizons, self.n_assets).astype(jnp.float32)
```

Wiring same pattern as RankHead (TASK_730):
- Add `use_aux_price_head`, `aux_price_hidden_dim`, `aux_price_horizons` (list) fields to `MuZeroNetworks`.
- Instantiate in `setup()` under `if self.use_aux_price_head`.
- Call in `initial_inference()` and include `result['aux_prices'] = ...`.
- Call in `initialize_parameters` branch so `initialize_missing_aux_heads` registers params fresh.

### Target extraction

The Phase 2 pipeline already has `windows` in `_prepare_numpy` (shape `(B, ws + nstep, n_assets, 5)` = `(B, 80, 64, 5)`). For each sample, the decision point is at index `ws - 1` (end of observation window). Future log-prices at offsets `[1, 5, 10, 20]` relative to decision are at indices `[ws, ws+4, ws+9, ws+19]`.

Close prices live at channel 3 of OHLCV (or 0 depending on the format — confirm in _prepare_numpy). Extract, take `log(price)`, pass through as `future_log_prices` in the JAX batch dict alongside `future_cum_returns`.

### Loss

In `make_stage_b_train_step`:

```python
if use_aux_price and 'aux_prices' in root_out:
    pred_log_prices   = root_out['aux_prices']                    # (B, K, n_assets)
    target_log_prices = future_log_prices                         # (B, K, n_assets)
    # Per-horizon MSE, weighted by 1/horizon so shorter horizons count more.
    # Longer horizons are noisier — lower weight is regularization.
    horizon_weights = jnp.array([1.0, 0.5, 0.33, 0.2])            # matches [1,5,10,20]
    per_horizon_mse = jnp.mean((pred_log_prices - target_log_prices) ** 2, axis=(0, 2))  # (K,)
    aux_price_loss  = jnp.sum(horizon_weights * per_horizon_mse) / jnp.sum(horizon_weights)
else:
    aux_price_loss  = jnp.array(0.0)

# Add to total
total_loss = ... + aux_price_w * aux_price_loss
```

Default `aux_price_w = 0.1` — modest weight so the new head doesn't dominate (similar to `direction_loss_weight = 0.2`).

Metrics surfaced: `aux_price_loss` (aggregate), plus per-horizon `aux_price_mse_h1`, `aux_price_mse_h5`, etc. for diagnostic visibility.

---

## Files Modified

| File | Change |
|---|---|
| `mcts_trader/jax_agent/muzero_model.py` | New `AuxPriceHead` module. Add fields + setup + wire into `initial_inference`/`initialize_parameters`. |
| `phase2_config.py` | Add `use_aux_price_head: bool = True`, `aux_price_hidden_dim: int = 256`, `aux_price_horizons: tuple = (1, 5, 10, 20)`, `aux_price_loss_weight: float = 0.1`. |
| `train_phase2.py` | Extract `future_log_prices` in `_prepare_numpy` and thread through `_encode_jax` batch dict. Add loss block + metrics + CSV schema + log line. Add `initialize_missing_aux_heads` trigger. Add CLI: `--no-aux-price-head`, `--aux-price-weight`, `--reset-aux-price-head`. Config overrides. Pass `use_aux_price_head` + `aux_price_hidden_dim` + `aux_price_horizons` through `create_model`. |

No encoder changes. No change to existing heads. Additive only.

---

## Smoke Test

```bash
source trader_env/bin/activate && \
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --test-init
```

Expectation: `Initialized fresh aux head params` log line now includes `aux_price_head` alongside the 729a/730 heads. Dynamics init + recurrent_inference pass as before.

Additionally: direct Python one-liner that calls `initial_inference` and verifies `result['aux_prices'].shape == (B, 4, 64)` and zero-init output (mean/std = 0 at step 0).

---

## Launch

```bash
mkdir -p jax_ckpt_easy_phase2_735 && \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1 \
  --output-dir jax_ckpt_easy_phase2_735 \
  > jax_ckpt_easy_phase2_735/stage_b_nohup.log 2>&1 &
```

[redacted: host service detail]

Tail (narrow — ceiling break or anomaly):
```bash
tail -F jax_ckpt_easy_phase2_735/stage_b_nohup.log | \
  grep --line-buffered -E "acc=0\.6[0-9]|acc=0\.[789]|sp=0\.[4-9]|aux_price_mse_h1=0\.0[01]|Traceback|ERROR|NaN|Killed|OOM|aborted"
```

---

## Gates

### First 20 epochs — sanity
- `aux_price_loss` should drop monotonically (head learning something about future prices).
- `aux_price_mse_h1` (nearest horizon) should drop faster than `aux_price_mse_h20` — closer horizons are more predictable.
- If all horizon losses stay flat → wiring bug or target extraction bug. Debug before judging.

### E50 — decision gate (same as 730/732/734)
- `dir_acc > 0.60` → magnitude prediction was the missing signal. Escalate to β (Bella-distilled version of this exact head).
- `dir_acc 0.58–0.60` → marginal. Full 200 epochs then decide.
- `dir_acc < 0.58` → prediction-head pathway fully exhausted (direction + rank + magnitude all ceiling identically). Pivot to §46 A1 (dense sign-of-return oracle) or §46 Tier D (more data) or accept ceiling.

### Benchmark (only if E50 passes)
- `tools/benchmark_easy.py --checkpoint jax_ckpt_easy_phase2_735/phase2_stageB_best.pkl`
- Bal.Acc > 45% → ship-candidate. Stack β next.
- Bal.Acc 38–45% → marginal improvement. β may amplify.
- Bal.Acc < 38% → head didn't help actual trading; roll back pathway.

---

## Why This Over α-v1b / α-v1c / α-v2 / β

**α-v1b (velocity head on dynamics)**: we proved in §48 that single-step velocity supervision collapses to MSE unless the velocity head is fully independent parameters. Fragile design, ~15% probability.

**α-v1c (multi-step autoregressive unroll)**: needs K extra encoder passes per batch + K-step action sequences (which the batch pipeline doesn't currently produce). ~2-3x compute overhead for the same ~20% probability. Engineering cost not justified before we've tested the cheaper α-v3.

**α-v2 (NeuralMarketODE)**: touches untested physics+neural code paths, has known collapse issues per user 2026-04-23 commentary. Complex inputs (noise, features, moving_avg, log_returns) that aren't currently in the batch. ~200 LoC. Higher risk than α-v3 for similar probability.

**β (Bella distillation)**: ~40% probability but 5-day engineering commitment (offline Bella inference pipeline, storage, new training mode). Should happen *after* α-v3 clears out the latent-side hypothesis. If α-v3 helps, we know adding magnitude prediction works → Bella is the richer teacher. If α-v3 fails, we know latent-side is exhausted → commit to β with cleaner justification.

α-v3 is ~60 LoC, 2hr train, ≈25% probability. Best cost/signal ratio of any prediction-head option.

---

## Known Risks

- **Target scaling**: log-prices span wide ranges across periods (BTC 30k vs DOGE 0.5). Without per-asset normalization, MSE will be dominated by high-magnitude assets. Either normalize log_prices within the 60-step window (same scheme as channels 0–4) or supervise log-RETURNS instead of log-prices (`log(price_{t+k}) - log(price_t)` which is centered near zero). Log-returns is cleaner; plan to use that.
- **Horizon weights**: `[1.0, 0.5, 0.33, 0.2]` is ad-hoc. If shorter horizons dominate gradient, model may not learn longer-horizon structure. Tune post-smoke if shorter horizons converge fast while longer stay high.
- **Compatibility**: TASK_734's `--flow-matching` flag stays in CLI but is a no-op (documented). This task is orthogonal; no conflict.
- **Rolling back**: `use_aux_price_head=False` in config reverts behavior. All new code is opt-in.
