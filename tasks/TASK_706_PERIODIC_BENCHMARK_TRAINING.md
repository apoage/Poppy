# TASK_706: Periodic Benchmark During Training — Stop When Passing

**Status**: DONE
**Priority**: 1 (training without eval feedback is flying blind)
**Depends on**: TASK_705 (benchmark pipeline audit — done)

---

## Problem

Training runs until agreement target (now 0.95) but agreement is a vanity metric — 90% agreement with 96% hold ratio just means both parties say HOLD. Model can hit 90% agreement and still lose to random on every benchmark period. Need real benchmark feedback during training and a finish condition based on actual benchmark performance.

---

## Design

### Run benchmark on CMP 90HX while training runs on RTX 3090

- Training: RTX 3090 (CUDA_VISIBLE_DEVICES=0)
- Benchmark: CMP 90HX (CUDA_VISIBLE_DEVICES=1) — 10GB VRAM, model is ~144MB + encoder/oracle well under 2GB
- Zero training interruption — benchmark runs in parallel on separate GPU

### Trigger: every 100 epochs

In `train_easy.py` epoch loop, after checkpoint save, launch benchmark as a subprocess:

```python
if (epoch + 1) % 100 == 0:
    # Launch benchmark on 90HX (non-blocking)
    ckpt_path = output_dir / f'phase1_epoch{epoch}.pkl'
    _launch_benchmark(ckpt_path, db_path)
```

### Finish condition: pass benchmark criteria

Current criteria from TASK_666:
- Positive alpha on >= 4/6 periods
- Beats random on 6/6 periods
- Trade density > 1% on 6/6 periods
- Balanced accuracy >= 75%
- Value MSE < 0.1

When benchmark passes, write a sentinel file. Training loop checks for sentinel and stops.

---

## Implementation

### Phase A: Benchmark subprocess launcher

Add to `train_easy.py`:

1. New CLI flag: `--benchmark-every N` (default 0 = disabled, 100 = every 100 epochs)
2. New CLI flag: `--benchmark-gpu 1` (CUDA device for benchmark, default 1 = 90HX)
3. Function `_launch_benchmark(ckpt_path, db_path, gpu_id)`:
   - Spawns `CUDA_VISIBLE_DEVICES={gpu_id} python tools/benchmark_easy.py --checkpoint {ckpt_path} --json {results_path}`
   - Non-blocking (subprocess.Popen)
   - Writes results to `logs/benchmark_epoch{N}.json`
4. Function `_check_benchmark_results(output_dir)`:
   - Reads latest benchmark JSON
   - Returns True if all TASK_666 criteria pass
   - Logs summary to training log

### Phase B: Integrate into epoch loop

In the epoch loop (after checkpoint save at line ~2020):

```python
# Periodic benchmark (runs on separate GPU, non-blocking)
if benchmark_every > 0 and (epoch + 1) % benchmark_every == 0:
    ckpt_path = output_dir / f'phase1_epoch{epoch}.pkl'
    if ckpt_path.exists():
        _launch_benchmark(ckpt_path, db_path, benchmark_gpu)

# Check if previous benchmark passed → stop training
if benchmark_every > 0 and _check_benchmark_results(output_dir):
    logger.info("=" * 60)
    logger.info("BENCHMARK PASSED — stopping training")
    logger.info("=" * 60)
    trainer.save_checkpoint(str(output_dir / 'benchmark_passed.pkl'), epoch, ...)
    break
```

### Phase C: benchmark_easy.py JSON output

`benchmark_easy.py` already has `--json` flag. Verify it:
1. Writes all PeriodResult fields
2. Includes a top-level `passed: bool` field based on TASK_666 criteria
3. Includes `checkpoint_path` and `epoch` in output

### Phase D: e-dojo integration

Update `e-dojo_v1.py` to pass `--benchmark-every 100 --benchmark-gpu 1` when launching training in single+historical mode.

---

## Files

| File | Change |
|------|--------|
| `train_easy.py` | Add `--benchmark-every`, `--benchmark-gpu`, launcher + checker functions, epoch loop integration |
| `tools/benchmark_easy.py` | Verify JSON output includes `passed` field |
| `tools/orchestration/e-dojo_v1.py` | Pass benchmark flags in single+historical mode |

---

## GPU Layout

```
RTX 3090 (24GB) — CUDA 0 — Training (full speed, uninterrupted)
CMP 90HX (10GB) — CUDA 1 — Benchmark (every 100 epochs, ~7 min each)
```

Benchmark needs: ~144MB model + ~500MB encoder/oracle working memory = <1GB VRAM. 90HX has 10GB — plenty.

---

## Acceptance Criteria

- [ ] Benchmark runs every 100 epochs on CMP 90HX without interrupting training on RTX 3090
- [ ] Benchmark results logged to `logs/benchmark_epoch{N}.json`
- [ ] Training stops automatically when benchmark criteria pass
- [ ] Summary printed to training log after each benchmark completes
- [ ] No VRAM contention between training and benchmark (separate GPUs)
- [ ] e-dojo passes correct flags in single+historical mode
