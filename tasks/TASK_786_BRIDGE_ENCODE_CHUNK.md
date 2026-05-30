# TASK_786 — Bridge trainer: memory-bounded encoder forward (780 OOM structural fix)

**Status**: IN PROGRESS (option B of the §171/§STATUS_2026-05-30 OOM triage).
**Created**: 2026-05-30.
**Owner**: workstation main agent.
**Depends on**: TASK_780 (bridge_trainer.py, oracle-as-MCTS distillation).
**Blocks**: 780 Phase-A at full config + `--bella` restored.

## Problem

TASK_780 Phase-A OOMs at E0 on all 3 seeds:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 12457304672 bytes  (11.6 GiB)
  bridge_trainer.py:517  train_step
```

The server-side unblock (`runs/STATUS_2026-05-30.md`) was to **drop `--bella`**, which
freed the market-generator's resident VRAM so the 11.6 GiB train-step peak fit in 24 GB.
That is a luck-based workaround, not a fix: the bridge train-step peak is unchanged, the
margin is thin, and 780 cannot run with the cross-sectional channels Bella feeds (which
the rest of the stack treats as load-bearing, cf. TASK_732 / `project_encoder_bella_class_closed.md`).

### Root cause (not the multipass tensor)

The OOM is **not** in `oracle_distillation.py`. The un-collapsed `(N, n_passes, T, 320)`
multipass tensor is ~0.5 GiB. The 11.6 GiB allocation is the **single batched encoder
forward** in `bridge_trainer._train_step_impl.loss_fn`:

```python
obs_flat = observations.reshape(B * T, C, W, N)   # (256*6, 27, 64, 64) = (1536, 27, 64, 64)
init_out_flat = self.model.apply({'params': params}, observation=obs_flat,
                                 mode='initial_inference')
```

With `batch_size=256`, `unroll_steps=5` (T=K+1=6), `window_size=64`, 27 channels, 64 assets,
the cross-asset-attention TRM encoder processes **1536 windows in one shot**, materialising a
single ~11.6 GiB activation. The same pattern repeats in `_evaluate_batch_impl` (`:550`).

## Fix — chunk the B·T encoder forward + remat

Bound the peak to one chunk's worth of activation, math-identical:

1. **Chunk** `obs_flat` over its leading `B·T` axis into `n_chunks` of `encode_chunk` rows,
   `jax.lax.map` over chunks (the proven `path_solver.py:319` / TASK_675 "lax.map caps VRAM
   at 1-pass peak" pattern). Rows are independent through `initial_inference` (encoder
   attention is cross-**asset** within a window, never cross-batch), so chunking is exact.
2. **Remat** the per-chunk apply with `jax.checkpoint` so the backward pass recomputes the
   chunk forward instead of retaining it — cuts the retained-activation tail across the
   encoder's layers.

Peak encoder activation drops from `B·T` to `encode_chunk` rows: e.g. `encode_chunk=384`
→ 4 chunks → ~2.9 GiB peak (vs 11.6 GiB). 780 then runs at full `batch_size=256` **with
`--bella` restored**, no luck required.

### Config / CLI

- New `EasyConfig.oracle_distill_encode_chunk: int = 384`.
  - Sentinel `0` (or `>= B·T`) → no chunking = legacy single-shot (escape hatch).
  - `384` divides `1536` evenly; padding handles any non-divisor (pad rows are sliced off
    before the loss, contribute zero gradient).
- CLI `--oracle-distill-encode-chunk N` on `train_easy.py`, mirroring the existing
  `--oracle-distill-passes` / `--oracle-distill-unroll` wiring.

### Scope (bit-identity)

- Canon EasyTrainer path is **untouched** — this only edits `bridge_trainer.py` (the
  `use_oracle_mcts_distill=True` path, default OFF) + 1 config field + 1 CLI flag.
- The change is numerically equivalent on the bridge path itself; the smoke asserts the
  chunked loss matches the legacy single-shot loss to float tolerance.

## Falsifier / acceptance

1. **Equivalence**: chunked vs `encode_chunk=0` loss agree to < 1e-4 relative on a synthetic
   bridge batch (B=8, K=2). 
2. **Memory**: a `batch_size=256, unroll=5` bridge train_step completes on a single 24 GB
   3090 **with `--bella`** without OOM (peak HBM < ~18 GB).
3. **Convergence**: 5-epoch synthetic smoke loss decreases monotonically (matches the
   §170 Phase-2 smoke: 11.70 → 4.78), unchanged from legacy.

## Re-entry anchors

- OOM site: `bridge_trainer.py` `_train_step_impl.loss_fn` (`obs_flat` encode) + `_evaluate_batch_impl`.
- Server forensics: `runs/STATUS_2026-05-30.md`, stashed `_OOM_780A-default_s{7,13,23}`.
- Pattern reference: `mcts_trader/jax_agent/path_solver.py:318-320` (lax.map over passes).
- Deploy: hold until current 780A trio finishes or operator authorizes — do not interrupt
  the in-flight G3 run.
