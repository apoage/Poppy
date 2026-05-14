# TASK_741: Per-Asset Scalar Injection — Mint

**Status**: PROPOSED (2026-04-25). Architectural fix for the bottleneck TASK_733 exposed.
**Priority**: HIGH. Unlocks the 13pp ceiling between Val Agree 87% (TASK_733 cheat) and the 100% bound. Also a prerequisite for any future per-asset scalar feature (Bella tag, regime tag, position state, cross-sectional ranks).
**Rationale**: `lore/notes_2026-04-24.md §65` — TASK_733 cheat-channel diagnostic peaked at 87.0% Val Agree even though channel 32 carried the ground-truth answer. Hypothesis: the V3 encoder broadcasts the cheat scalar across the 60-step temporal window, then minGRU temporal pooling partially averages the constant DC offset away before per-asset features reach the heads.
**Blocks**: nothing strictly, but **TASK_740 (Bella distillation) gain is capped at the 87% architectural ceiling** until this lands.
**Naming**: **Mint** — clarity at the per-asset level. Scalars taste of their own asset, not a smeared average.

---

## The Problem

TASK_733 (`tasks/TASK_733_CEILING_DIAGNOSTIC.md`) trained a V3 encoder with channel 32 = `sign(log(p_{t+20} / p_t))` ∈ {−1, 0, +1} per asset. Even with the ground-truth answer present in the input, **peak Val Agree was 87.0% at E133** (notes §65). Smoke-tested clean: channel 32 confirmed in {−1, 0, +1}, channels 0–31 byte-identical with/without the cheat flag, no normalization pollution.

The V3 encoder's structural pipeline (`mcts_trader/jax_agent/trm_encoder_v3.py:50-87`):

1. Input `(B, 33, 60, 64)` — channel 32 is constant across the 60-step axis.
2. `temporal_proj` Dense(33→embed_dim) at every timestep — the cheat scalar enters as a constant additive bias `W_proj[:, 32] · cheat_sign[a]` at every `t`.
3. minGRU `_mingru_scan` (lines 156–188) computes `h_t = (1−z_t) · h_{t−1} + z_t · h̃_t`. With per-step sigmoid/tanh on a stationary input, the gates are also nearly stationary; the 60-step recurrent average attenuates the per-asset cheat contribution and entangles it with whatever drift the legitimate channels 0–4 exhibit. **Result**: a perfect per-asset signal placed at the input is delivered to the heads as `~0.87 · cheat + 0.13 · noise`.

Cross-sectional channels 27–31 (TASK_732) suffer the same fate. They are also computed once per window and broadcast across time — minGRU dilutes them by the same mechanism. The TASK_737 `legacy_27ch` ablation (notes §60) measured the size of this signal as Alpha −21pp when it is removed, which is the strength _after_ dilution; the undiluted signal is presumably larger.

This task moves per-asset scalar features from the **pre-temporal** input channel to the **post-temporal-pool** per-asset feature surface, where they cannot be averaged.

---

## The Change

### Injection point

Per `trm_encoder_v3.py::_post_recurrence` (line 124–153): minGRU produces `h: (batch, n_assets, embed_dim)` (line 81). Asset identity embed and cross-asset attention run on this; LayerNorm produces:

```python
per_asset_feats = self.pre_proj_norm(h)   # (batch, n_assets, embed_dim)   line 149
```

This is the first per-asset tensor where each asset's vector is exclusively a function of that asset's data plus cross-asset attention. **All downstream per-asset heads consume this exact tensor**: `DirectionHead`, `RankHead`, `PerAssetPolicyHead` all take `(B, n_assets, D)` from `representation_with_features` (`muzero_model.py:519, 552, 563, 570`). Inject scalars **here**, immediately before `pre_proj_norm` would have been the alternative; we choose **post-LN, pre-projection** so the LayerNorm cannot wash the new signal back out.

### Recommended scheme: FiLM (feature-wise linear modulation)

```
scalars: (B, n_assets, n_scalars)        # cheat + xs_27..31 + future per-asset scalars
γ, β = FiLMProjector(scalars)             # each shape (B, n_assets, embed_dim)
per_asset_feats = (1 + γ) * per_asset_feats + β
```

`FiLMProjector` is a 2-layer MLP shared across assets:

```python
class FiLMProjector(nn.Module):
    embed_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, scalars):           # (B, A, n_scalars)
        h = nn.gelu(nn.Dense(self.hidden_dim, name='film_1')(scalars))
        out = nn.Dense(2 * self.embed_dim,
                       kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros,
                       name='film_out')(h)
        gamma, beta = jnp.split(out, 2, axis=-1)
        return gamma, beta                  # both (B, A, embed_dim)
```

**Zero-init on `film_out`** → at step 0 the layer emits `γ = 0`, `β = 0`, so `per_asset_feats` is unchanged (byte-identical to today). Gradient pulls γ/β as the scalars become useful. The `(1 + γ)` form matches FiLM (Perez 2017) and keeps the multiplicative path stable through init.

#### Why FiLM and not concat+linear

| Scheme | Pro | Con | Verdict |
|---|---|---|---|
| **A. Concat + Dense** | Simplest. Adds Dense((D+n_s) → D). | Mixes scalar and pooled features in one linear map; gradient must learn to attend without help. ResNet-style additive coupling needs much more data to discover the multiplicative interaction. | Reject — likely under-uses the cheat for the same reason input-channel injection did. |
| **B. FiLM** | Multiplicative + additive in one step. Standard in conditional generation (StyleGAN, BigGAN). Zero-init gives byte-identical default. Cheap (~A·D·hidden params). | Slightly more parameters than A. | **Recommended.** |
| C. Gated injection (`out = feats + sigmoid(W·s) · Dense(s)`) | Per-feature gate on top of B. | Extra sigmoid on a tiny scalar count is overkill; FiLM already handles per-feature scaling via γ. | Reject — over-parameterized for n_scalars ≪ embed_dim. |
| D. Cross-attn from scalars to feats | Most expressive. | Heavy. n_assets queries × n_scalars keys is wasted compute when scalars are per-asset. Conflicts with the no-attention-in-reasoning rule (`mcts_trader/CLAUDE.md`). | Reject. |

**DO NOT pick concat+Dense.** It re-creates the same problem the input-channel cheat suffered: a small dimension's signal gets averaged into a much larger feature vector under linear projection. FiLM forces the scalar to scale every feature dimension explicitly.

### Scalar source plumbing

Two inputs to the new layer:

1. **Cheat scalar** (1 dim per asset) — already computed in the encoder data path (`env/jax_encoder.py:551-561`). When `--cheat-channel` is on, expose it as a separate tensor `cheat_scalars: (B, n_assets)` instead of (or in addition to) writing it to channel 32.
2. **Cross-sectional scalars (xs_5)** (5 dims per asset) — `rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom`, currently broadcast to channels 27–31 (`env/jax_encoder.py:531-535`). Optionally promote to `xs_scalars: (B, n_assets, 5)`.

Concat at the head: `scalars = jnp.concatenate([xs_scalars, cheat_scalars[..., None]], axis=-1)` → `(B, n_assets, 6)` when both flags on, `(B, n_assets, 5)` when xs-only, `(B, n_assets, 1)` when cheat-only.

**Backward-compat**: if `n_scalars == 0` (no flag set), skip the FiLM call entirely → identity. If `n_scalars > 0` and `film_out` weights are missing from the checkpoint, `initialize_missing_aux_heads` (pattern at `train_phase2.py:165-186`) inits them to zero, preserving exact prior behavior on first forward.

---

## Files to Modify

| File | Change | LoC |
|---|---|---|
| `mcts_trader/jax_agent/trm_encoder_v3.py` | New `FiLMProjector` Module. `StreamingAssetEncoder.__call__` and `step` accept optional `per_asset_scalars: (B, A, n_s)` kwarg, applied inside `_post_recurrence` after `pre_proj_norm`. | ~40 |
| `mcts_trader/jax_agent/trm_flax.py` | `TinyRecursiveTradingModel.encode_with_features` and `encode` forward `per_asset_scalars` kwarg through. Add `per_asset_scalar_dims: int = 0` field for shape introspection. | ~20 |
| `mcts_trader/jax_agent/muzero_model.py` | `representation` + `representation_with_features` accept and forward `per_asset_scalars`. `initial_inference` accepts `per_asset_scalars` kwarg, passes to representation. New `MuZeroNetworks.use_per_asset_injection: bool` and `per_asset_scalar_dims: int` fields. | ~25 |
| `mcts_trader/env/jax_encoder.py` | When `config.use_per_asset_injection=True`, _also_ return per-asset scalars as a separate tensor alongside the channel grid (or in lieu of writing channels 32/27–31 — see "two modes" below). New helper `extract_per_asset_scalars(prices, ws_start, config) -> (n_assets, n_scalars)` and batched / vmap variants. | ~50 |
| `easy_config.py` | New fields `use_per_asset_injection: bool = False`, `per_asset_inject_cheat: bool = False`, `per_asset_inject_xs: bool = False`. | ~6 |
| `phase2_config.py` | Mirror passthrough fields (same as TASK_733 pattern at `phase2_config.py:164-169`). | ~6 |
| `train_easy.py` | New CLI `--per-asset-injection {none,cheat,xs,both}`. Plumb scalar tensor through SegmentPool slot construction (mirrors current `_cheat_lookahead` extraction at `train_easy.py:1395-1400, 1590-1593, 2183-2190`) and into model.apply call. | ~70 |
| `easy_trainer.py` | Plumb `per_asset_scalars` from batch dict through `_train_step_impl` to `model.apply(...)`. | ~10 |
| `train_phase2.py` | Same as `easy_trainer.py` plus extend `initialize_missing_aux_heads` to cover `FiLMProjector` subtree. | ~15 |
| `tools/benchmark_easy.py` | If `config.use_per_asset_injection=True` and `per_asset_inject_xs=True` (legitimate, no future leakage), allow benchmark to proceed and compute xs_scalars at inference time. If `per_asset_inject_cheat=True`, refuse (extension of existing cheat safety rail at line 842). | ~10 |
| `tests/test_trm_encoder_v3.py` (new) | Smoke test: zero-init FiLM byte-identical to baseline; non-zero γ/β shifts output; shape correctness; vmap-clean. | ~50 |

**Total**: ~300 LoC. Target ~250 + 50 test.

---

## CLI Flags

```
--per-asset-injection {none, cheat, xs, both}    # default: none → byte-identical to today
```

- `none` — no FiLM layer instantiated; encoder behavior identical to current.
- `cheat` — only the TASK_733 cheat scalar fed through FiLM. Channel 32 of the input grid still emitted (for direct A/B), or omitted if `--per-asset-injection-replace-channels` (stretch flag).
- `xs` — only cross-sectional scalars (xs_5). **Production-safe** — no future leakage. This is the keeper after the diagnostic.
- `both` — cheat + xs together. Diagnostic only.

Existing `--cheat-channel` remains the way to enable cheat input (with or without injection). The two flags are orthogonal:
- `--cheat-channel` alone → channel-32 broadcast input (today's behavior; ceiling 87%).
- `--cheat-channel --per-asset-injection cheat` → channel-32 input AND post-pool FiLM. **The TASK_741 validation experiment.**

---

## Backward Compatibility & Checkpoint Loading

1. **Default path** (`use_per_asset_injection=False`) — `FiLMProjector` is never instantiated; no params added to the tree; existing TASK_736 / TASK_737 / TASK_733 checkpoints load unchanged.
2. **Opting in on a fresh run** — `model.init` produces the FiLM subtree fresh; zero-init on `film_out` means epoch 0 forward pass is byte-identical to a non-injection model with the same seed.
3. **Loading an old checkpoint into an injection-enabled model** — `initialize_missing_aux_heads` (pattern at `train_phase2.py:165-186`) detects the missing top-level key and initializes it. Because of the zero-init on `film_out`, the warmstarted model continues from the old checkpoint's behavior at step 0 and only diverges as gradient pulls γ/β.
4. **JSON config round-trip** — both `EasyConfig.use_per_asset_injection` and the `Phase2Config` mirror are present so a Phase-1 checkpoint loaded by Phase-2 doesn't drop the flag (this was the bug §61 line 89 caught for `use_cross_sectional`; mirror that fix preemptively).
5. **Benchmark safety** — `tools/benchmark_easy.py:842` cheat-channel safety rail extended: refuse if `(use_per_asset_injection=True AND per_asset_inject_cheat=True)`. The xs path is production-legal — no leakage.

---

## Success Criteria

The validation matrix is small (4 runs, ~10 hr GPU total).

### Gate 1 — Cheat reruns (the hypothesis test)

Re-run TASK_733 with `--cheat-channel --per-asset-injection cheat`, otherwise unchanged.

| Outcome | Interpretation | Next |
|---|---|---|
| Val Agree ≥ 95% by E60 | **Hypothesis confirmed.** Architecture can fully exploit per-asset scalars when delivered post-pool. The 13pp gap was purely the temporal-averaging wash. | Proceed to Gate 2. |
| Val Agree 90–95% | Partial confirmation. FiLM helps, but other architectural effects remain. | Investigate `pre_proj_norm` interaction; consider injecting before LN as a secondary experiment. |
| Val Agree ≤ 87% (no improvement) | **Hypothesis falsified.** Mechanism is something else (asset-identity embed shortcut? minGRU intrinsic capacity? cross-attn averaging across assets?). | Halt TASK_741 implementation; pivot to direct-investigation diagnostics on the V3 attention surface. |

### Gate 2 — Dense sign rerun (production validation)

Re-run TASK_739 v2 (calibrated ε=7.5e-3) with `--per-asset-injection xs`, otherwise unchanged baseline.

| Outcome | Interpretation | Next |
|---|---|---|
| dir_acc ≥ 0.62 | Cross-sectional scalars are now properly delivered. Real ceiling-break. Ship the xs-injection variant as the new baseline. | Promote xs-injection to the default config; deprecate channel 27–31 broadcast. |
| dir_acc 0.58–0.62 | Marginal improvement. Stack with TASK_740 (Bella distill) for additive gain. | Run TASK_740 with `--per-asset-injection xs` immediately. |
| dir_acc < 0.58 | xs-injection helps less than expected. Cheat result still informs whether to keep the layer for future scalars. | Decide based on Gate 1 outcome; FiLM may still be the right surface for Bella tag injection (TASK_740 follow-on). |

### Gate 3 — Benchmark on Gate-2 winner

Standard 6-period benchmark per `tools/benchmark_easy.py`. Spec: Alpha > +5%, beats 6/6.

---

## Why Now

- TASK_733 (notes §65) closed 4 weeks of model-side experiments with a single, clean diagnostic that every prior result was bottlenecked **upstream of any head config**. The exact place the bottleneck lives is now a known location in code (`trm_encoder_v3.py:68` projection + lines 71–78 minGRU scan).
- TASK_740 (Bella distillation, ~5–7 day infra build) gain is **capped at the same 87% architectural ceiling** until this fix lands. Sequencing TASK_741 first prevents wasting the Bella infra on a ceiling that defeats it.
- TASK_739 v2 dir_acc = 0.570 (notes §62) — every reformulation on the label side has converged to the same number. Architecture is the only remaining lever that hasn't been pulled in this dimension.
- ~300 LoC, ~10 hr GPU. Cheap relative to the leverage.

---

## Known Risks

| Risk | Mitigation |
|---|---|
| **Hypothesis is wrong**: averaging is not the mechanism, e.g. `asset_identity_embed` (TASK_731) is the actual shortcut and FiLM doesn't bypass it. | Gate 1's binary outcome catches this. ≤87% Val Agree on cheat-with-FiLM falsifies temporal-averaging definitively. Cost: 2 hr GPU. |
| **FiLM destabilizes existing-channel learning**: γ near 1.0 amplifies feature variance. | `(1 + γ)` form with γ zero-init means amplification is learned, not initial. Add `jnp.clip(γ, -2, 2)` if late-training values blow up — observe loss spikes per `notes §65` E77/E157 spikes already documented. |
| **PopArt / value head interaction**: Mint changes the per-asset feature surface; `global_proj` reads `per_asset_feats.reshape(...)` (line 152) and the value head consumes `latent` from TRM reasoning. New scaling could shift the value-target distribution outside PopArt's adaptive range. | PopArt re-normalizes online; should self-correct within ~100 steps. Monitor `value_loss` first 10 epochs; revert γ scaling if value_loss > 2× baseline. |
| **MCTS targets**: tree priors shift if policy logits move. Standard for any encoder change. | Same as TASK_729a/729b/729c — covered by ExIt loop convergence. No special handling. |
| **Cross-attention interaction**: cross-asset self-attention (line 146) runs _before_ FiLM. Attention may already be smoothing what FiLM is meant to sharpen. | Consider injecting **before** cross-attention as a follow-up — but the post-attn LN surface is what every head consumes today, so first run keeps the canonical ordering. |
| **Checkpoint round-trip silently drops the flag** (the §61 line 89 `use_cross_sectional` bug recurring). | Mirror `use_per_asset_injection` on `Phase2Config` from day one. Add round-trip test in `tests/`. |

---

## Out of Scope (Explicit Non-Goals)

- **Inject before cross-attn** — discussed above as a follow-up; out of scope for the first cycle.
- **Replace channels 27–31 / 32 entirely** — first cycle keeps them as inputs and adds the FiLM surface in parallel for an additive A/B. Channel-removal is a clean-up after the design is proven.
- **Apply FiLM in V2 encoder** — V2 is legacy. If the design lands and someone re-baselines on V2, port then.
- **Apply FiLM in `step()` streaming path** — required for live trading, but not for this validation cycle (Foxglove training uses `__call__`). Add when wiring up live inference.
- **Bella forecast as injected scalar** — `TASK_740` task; TASK_741 builds the surface, TASK_740 fills it.
- **Position state / regime tag as injected scalar** — future tasks.
- **Tier B (drop BC) experiment** — `--bc-weight-min 0.0` from notes §65 next-action — separate cheap test, can run before or after TASK_741.

---

## Dependencies

- TASK_733 result (Val Agree 87.0%, notes §65) — the empirical motivation. Already complete.
- TASK_736 baseline (`jax_ckpt_easy_phase2_736/phase2_stageB_best.pkl`) — reference for Gate 3 benchmark. Already in place.
- TASK_739 v2 baseline (`jax_ckpt_easy_phase2_739_dense_sign/phase2_stageB_best.pkl`) — reference for Gate 2. Already in place.
- 32-ch Phase-1 checkpoint (`jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`) for warmstart.
- No new infra (no Bella, no Iris-DEX, no extra data).

---

## Estimated Total Wall-Clock

| Step | Time |
|---|---|
| Implementation + smoke tests | 2 days (one engineer-day, one Claude-day) |
| Gate 1: rerun TASK_733 with `--cheat-channel --per-asset-injection cheat` (200 epochs ≈ 2.5 hr) | 3 hr |
| Gate 2: rerun TASK_739 v2 with `--per-asset-injection xs` (200 epochs ≈ 2.5 hr training + 7 min bench) | 3 hr |
| Gate 3: Stage-B benchmark on Gate-2 winner | 30 min |
| Analysis + decision + notes update | 1 hr |
| **Total from green-light to first answer** | **~3 working days** (1 day code + 1 overnight GPU window for the gates + 0.5 day analysis) |

---

## Implementation Notes (for the executing agent)

- **Tensor name to grep for**: `per_asset_feats` — the canonical name in `muzero_model.py` and `trm_encoder_v3.py:149`. Inject FiLM here, before line 149 returns it.
- **Apothecary theme name**: **Mint**. Suggested log tag: `[MINT]`.
- **JAX rules**: FiLM is pure Dense + reshape, vmap-clean. No `lax.cond`. Use `jnp.where(use_injection, ...)` if a runtime branch becomes necessary, but the cleanest is a Python-level `if self.per_asset_scalar_dims > 0` at module-construction time so the layer either exists or doesn't.
- **Compute dtype**: match `self.dtype` on FiLM Dense layers; cast inputs to `self.dtype` and outputs to `jnp.float32` per the established pattern (`muzero_model.py:35, 69, 107`).
- **Test harness**: TASK_733's tests file (`tests/test_cheat_channel.py` if extant — mirror) is the right pattern. Byte-identical assertion with γ/β = 0 is the load-bearing test.
