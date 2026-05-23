# tools/orchestration — Training Dojos

Multiple dojo generations coexist. **Know which to touch before editing.**

## Active (edit freely)

| File | Model | Role |
|---|---|---|
| `e-dojo_v1.py` | Foxglove | Easy-mode dual-branch (Bella + Synthetic). Current research sandbox. |
| `dual_dojo.py` | Foxglove | 2-GPU parallel (90HX + 3090). Natural corruption via periodic merging. |
| `dojo_v4_full_curriculum.py` | `phoenix_v1` | Full 3-phase curriculum (L0–L14 → Speed → Historical). Rarely run. |
| `merge_utils.py` | shared | `fed_avg`, `fisher_merge`, `ties_merge`, `performance_weighted_merge` |
| `divergence_metrics.py` | shared | Cosine sim, agreement rate, loss barriers |

## Reference only (read, don't edit)

- `dojo_v3_historical.py` — v3 rainbow-table historical training (`poppy_v2`). Called by v4 Phase 3.

## Deprecated — DO NOT EDIT

- `dojo_v2.py`
- `dojo_manager.py`
- `dojo_dynamic_curriculum.py`
- `train_manager.py` (not used in current pipeline)

**Versioning is not linear.** e-dojo and dual_dojo are a separate lineage from v2/v3/v4. Model series: Foxglove (e-dojo, dual_dojo) vs phoenix_v1 (v4) vs poppy_v2 (v3).

## Entry commands

```bash
# Current Foxglove pipeline
python tools/orchestration/e-dojo_v1.py --run --preset large --epochs 2500 --dual-bella

# 2-GPU exploration
python tools/orchestration/dual_dojo.py --run --checkpoint jax_ckpt_easy_merged/model.pkl --preset large

# Full curriculum from scratch (rare)
python tools/orchestration/dojo_v4_full_curriculum.py --run

# Reset state (interactive — kills processes, deletes checkpoints/logs)
./tools/orchestration/reset_edojo.sh
./tools/orchestration/reset_dual_dojo.sh
```

## State / artifact layout

**Checkpoints:**
- e-dojo: `jax_ckpt_easy_A/`, `jax_ckpt_easy_B/`, `jax_ckpt_easy_merged/`
- dual_dojo: `jax_ckpt_dual_{A,B,merged}/` (typically on `/mnt/scratch/trader_checkpoints/`)
- dojo_v4: `phoenix_v1_*`

**State + logs:**
- `logs/e-dojo/dojo_state.json` — merge count, intervals, epoch tracker
- `logs/e-dojo/merge_history.json` — per-merge metadata
- `logs/dual-dojo/dual_dojo_state.json`
- `logs/*.csv` — training metrics
- `logs/benchmarks/*.{csv,json}` — benchmark results per level

**External configs (read-only):**
- `config/benchmark_periods.json`
- `training_schedule.json` (v3/v4 rainbow table)

## Gotchas

- **e-dojo does NOT forward arbitrary `train_easy.py` CLI flags from the
  orchestrator command line.** The dojos build child invocations from their
  own internal config + programmatic kwargs, not by passing through operator
  argv. Practical consequence: `--oracle-k 8` (or any other `train_easy.py`
  flag) supplied to `e-dojo_v1.py` is **silently dropped**, not forwarded to
  the child trainer. To change Phase-1 hyperparameters under e-dojo, edit the
  dojo's branch-config block (search for `train_easy.py` invocations in
  `e-dojo_v1.py` / `dual_dojo.py`) and rebuild the child argv there. The
  operator-facing CLI rejects unknown flags, so a typo'd flag fails loudly;
  the silent-drop hazard is specifically for **programmatic kwargs** that
  look right but never reach the trainer. (TASK_771/764/F2 doc note.)
- **Orchestrators run CPU-only.** `CUDA_VISIBLE_DEVICES=''` + `JAX_PLATFORMS=cpu` set before JAX import. **Do not remove** — prevents orchestrator from stealing VRAM from child trainers.
- **Subprocess file handles (TASK_141)** — `e-dojo_v1.py ~line 1407`: log file handles must stay open for subprocess lifetime. Don't wrap in `with`; keep `self.log_files` dict persistent.
- **Process management is pkill-based** — no PID files. `reset_*.sh` matches process name. Killing a dojo also kills its `train_easy.py` children.
- **Checkpoint path divergence** — e-dojo uses relative paths; dual_dojo hardcodes `/mnt/scratch/trader_checkpoints`. Check storage layout before running.
- **Merge scheduling** — e-dojo triggers on loss barriers + cosine divergence, interval starts 8000 steps. `merge_history.json` is JSON — inspect, don't hand-edit.
- **State corruption risk** — if edits fail mid-run, `dojo_state.json` desyncs from checkpoints. Always use reset scripts for clean slate.
- **Nested subprocess GPU env** — dojos launch `train_easy.py` children with branch-specific `CUDA_VISIBLE_DEVICES`. Parent CPU-only env must not leak into children.

## Related (outside this dir)

- Dashboards: `tools/DS24/` (train_jax_fast), `tools/DSe/` (e-dojo), `tools/DS56/`
- Data coverage: `tools/map_training_data.py`
