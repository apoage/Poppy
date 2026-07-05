# TASK_690: Dual Dojo — 2-GPU Parallel Training with Checkpoint Merging

## Status: IN PROGRESS

## Context

TASK_689 Phase A (agent state channels) broke the 62% plateau, reaching 72.7% on synthetic and 68.4% on Bella (still climbing at epoch 394). Phase B needs to bridge the train-to-inference gap via state corruption. Instead of post-hoc corruption (creates impossible states) or oracle entropy, we use **dual dojo merging** as natural corruption:

- **3090** (batch=512): smooth gradients → exploitation
- **90HX** (batch=128): noisy gradients → exploration
- Periodic checkpoint merge = natural corruption (merged model doesn't perfectly match either branch's oracle trajectory)

## GPU Mapping (verified via JAX)

| CUDA Index | GPU | VRAM | Role | Batch |
|-----------|-----|------|------|-------|
| `CUDA_VISIBLE_DEVICES=0` | RTX 3090 | 22.8 GB | Branch A (exploitation) | 512 |
| `CUDA_VISIBLE_DEVICES=1` | CMP 90HX | 9.3 GB | Branch B (exploration) | 128 |

**Note**: nvidia-smi shows reversed order (90HX=0, 3090=1), but CUDA ordering verified via JAX.

## VRAM Budget for 90HX (batch=128)

- Obs: 128 x 34 x 64 x 64 x 4 = 71 MB
- Model + optimizer (3x params): ~400 MB
- Gradients + activations: ~2-3 GB
- Bella generator: ~500 MB
- Buffer (8 episodes): ~1.4 GB
- **Total: ~5-6 GB → fits in 9.3 GB**

## Implementation

### New Files

1. `tools/orchestration/dual_dojo.py` (~400 lines) — Orchestrator
2. `tools/orchestration/reset_dual_dojo.sh` (~30 lines) — Reset script

### Modified Files

3. `tools/DSe/server.py` (~10 lines) — Add branch_A_3090/branch_B_90hx patterns

### Unchanged (reused as-is)

- `merge_utils.py`: fed_avg, fisher_merge, ties_merge, merge_optimizer_state
- `divergence_metrics.py`: cosine_similarity, compute_divergence_report
- `train_easy.py`: launched as subprocess, unchanged
- `easy_trainer.py`, `easy_config.py`: unchanged
- `e-dojo_v1.py`: untouched, stays available

## Design

### Orchestrator Architecture

```
DualDojoConfig     - GPU mapping, batch sizes, buffer sizes, merge schedule, directories
DualDojoLauncher   - subprocess.Popen with per-GPU CUDA_VISIBLE_DEVICES
DualDojo           - Poll loop: check epochs, compute divergence, merge, restart
main()             - CLI: --run --checkpoint --merge-interval --gpu-a --gpu-b etc.
```

### Key Differences from e-dojo

- Sets `CUDA_VISIBLE_DEVICES=""` + `JAX_PLATFORMS=cpu` at top → CPU-only orchestrator
- Each branch gets its own GPU via CUDA_VISIBLE_DEVICES in subprocess env
- No stagger needed (separate GPUs)
- Epoch-based merge interval (default 30 epochs of Branch A)
- Both branches use `--bella` (diversity from batch size asymmetry, not data source)

### Merge Flow (CPU, every ~30 epochs)

1. Read latest checkpoints from both branch dirs
2. Compute cosine similarity
3. If cosine < 0.85 → divergence brake (restart worse branch)
4. Else → select merge method (fedavg early, fisher late, ties moderate)
5. Merge params + optimizer state (reset mu, average nu, preserve counts)
6. Save to `jax_ckpt_dual_merged/merged_latest.pkl`
7. Kill both branches, restart from merged checkpoint
8. `gc.collect()` + `jax.clear_caches()`

## Checkpoint Directory Layout

```
/mnt/scratch/trader_checkpoints/
  jax_ckpt_dual_A/branch_A_3090/     # 3090 checkpoints
  jax_ckpt_dual_B/branch_B_90hx/     # 90HX checkpoints
  jax_ckpt_dual_merged/              # Merged checkpoints
```

## Usage

```bash
source trader_env/bin/activate

# Launch dual dojo (both GPUs, merge every 30 epochs)
python tools/orchestration/dual_dojo.py --run \
  --checkpoint /mnt/scratch/trader_checkpoints/jax_ckpt_easy/state_v1_bella/phase1_epoch394.pkl \
  --preset large --epochs 2500 --merge-interval 30

# Monitor
./tools/DSe/start_dse.sh  # Dashboard shows both branches
watch -n5 nvidia-smi      # Verify both GPUs loaded
sudo dmesg -w             # Watch for PCIe errors

# Reset
./tools/orchestration/reset_dual_dojo.sh
```

## Verification

1. VRAM fit test on 90HX (batch=128, buffer=8, n_envs=2)
2. CPU merge test (CUDA_VISIBLE_DEVICES="" on two test checkpoints)
3. Full smoke test (--epochs 50 --merge-interval 10)
4. Dashboard: DSe shows both branches

## Completion Criteria

- [ ] dual_dojo.py launches both branches on separate GPUs
- [ ] Merge fires at correct interval
- [ ] Divergence brake works (cosine < 0.85)
- [ ] Branches restart from merged checkpoint
- [ ] DSe dashboard shows both branches
- [ ] 90HX fits within 9.3GB VRAM
- [ ] No VRAM theft by orchestrator (CPU-only)
