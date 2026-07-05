# TASK_742: Rank-head Python deadlock between epochs in `train_phase2.py`

**Status**: NEW (2026-05-12). Open regression; blocks TASK_737 `no_aux` variant and
any future `train_phase2.py` run that activates the rank head.
**Discovered by**: server-side agent during TASK_737 ablation suite run on gpu-server.
**Reproducibility**: deterministic on gpu-server — `no_aux` variant of TASK_737
hung at exactly E18→E19 boundary; other two variants (no rank head) ran clean.
**Reference**: `runs/737_ablations/{result.md, notes.md}`,
`lore/notes_2026-05-12.md` §84.

---

## Symptom

`train_phase2.py` with `--rank-loss mse --rank-weight 1.0` (and `--no-aux-price-head`,
i.e. the TASK_737 `no_aux` variant) deadlocks between epochs after the
best-checkpoint save at epoch 18. E19 never starts.

Process state on hang:

- State: `S` (sleeping)
- All 347 threads in `futex_wait_queue_me`
- GPU0: 0 % util, 21008 MiB VRAM still held
- VmRSS: 16.7 GB
- Open files: only `train.log`, `stage_b_training.log`, `stage_b_metrics.csv`,
  three `anon_inode:[eventfd]` FDs
- No Python traceback, no NaN, no OOM, no CUDA-side hang

This is a Python-level synchronization deadlock, not a CUDA issue.

## Implicated component

The two other TASK_737 variants (`no_rank`, `minimal`) share every code path
except the rank-head loss config and ran to completion (E199). The differentiator:

```
no_aux:  --rank-loss mse --rank-weight 1.0  --no-aux-price-head
no_rank: --no-rank-head                     --aux-price-weight 0.1
minimal: --no-rank-head                     --no-aux-price-head
```

→ The rank head's loss/grad path is the only feature exercised exclusively by
`no_aux`. Suspect a between-epoch code path that only runs when the rank head
is active — shared-state write, lock acquisition, queue put, or threading
primitive that's missing a release.

## Investigation plan (workstation-side, in order)

### Step 1 — diff between-epoch code paths active only when rank head is on

Grep `train_phase2.py` and `mcts_trader/training/` for:

```
grep -rn 'rank_loss\|rank_weight\|rank_head' train_phase2.py mcts_trader/training/
```

For each hit, ask: does this code path run **between epochs** (after one epoch's
optimizer step, before the next's data load), and does it involve any of:

- `threading.Lock`, `RLock`, `Condition`, `Event`, `Semaphore`
- `queue.Queue.put/get` with no timeout
- `multiprocessing` primitives
- `asyncio` await without cancellation
- `jax.block_until_ready` on a result that was never produced
- file locks
- shared-state writes that could collide with another thread

If the rank-head code path is small enough, eyeball it for obvious sync bugs
(e.g., acquire-without-release in an exception path).

### Step 2 — reproduce locally

Workstation has the same code at HEAD `c862430` (deployed) / current
working tree. Reproduce on smaller config:

```bash
python train_phase2.py \
  --stage B --epochs 25 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 2 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --no-aux-price-head \
  --output-dir /tmp/task_742_repro
```

`--max-runs-per-group 2` shrinks epochs ~5×. `--epochs 25` covers the E18→E19
deadlock window. Total wallclock target: ~5 min before hang.

### Step 3 — capture per-thread Python stacks

Once hung:

```bash
pip install py-spy        # if not already
py-spy dump --pid <PID>   # one-shot snapshot of all Python threads
```

The futex address + Python call stacks should pinpoint the synchronization
primitive holding things up.

### Step 4 — try gradient-magnitude finger-pointing

If Step 3 is ambiguous, drop rank weight:

```bash
... --rank-weight 0.001 ...
```

- Hangs same → rank-head's forward/backward or init logic is the cause (NOT
  the loss magnitude). Look at `RankHead` class, head init at checkpoint load,
  any custom autograd path.
- Hangs no longer → numerical issue (gradient magnitude, NaN-adjacent value)
  triggering some recovery path that deadlocks.

## Acceptance

- [ ] Root cause identified and documented in this task.
- [ ] Fix applied to `train_phase2.py` (or wherever the rank head's
      between-epoch code lives).
- [ ] Local reproduction (Step 2 config) completes 25 epochs cleanly.
- [ ] Re-run TASK_737 `no_aux` variant on server completes 200 epochs.
- [ ] All four TASK_737 variants now comparable (3 already done + no_aux fixed
      + legacy_27ch when 27-ch checkpoint is available).

## Notes

- The hang is **not** a CUDA hang. VRAM held but GPU at 0% util means JAX's
  device-side work finished; we're stuck in CPython.
- This is the first server-side run of `train_phase2.py`, but the deadlock
  shape (Python threading futex) is platform-agnostic — should reproduce on
  workstation.
- Don't fix on the server. Fix on workstation, redeploy, retry.
