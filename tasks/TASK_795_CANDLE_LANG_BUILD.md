# TASK_795 — Candle-Language JAX build (the implementation of TASK_791/792)

**Status**: BUILD-READY — design + adversarial leak/OOM audit complete; 6 blocking corrections gate
first server launch.
**Created**: 2026-06-02 · **Parent**: TASK_789 · **Implements**: TASK_791 (tokenizer) + TASK_792 (gate)
**Plan**: `docs/candle_language/04_jax_build_plan.md` · **Audit**: `docs/candle_language/05_leak_oom_audit.md`
**Source**: workflow `wzx1vegea` (3 repo maps → build plan → adversarial OOM/leak audit).

## What this builds

A new package **`tools/candle_lang/`** (donor = the salvaged `tools/tokenizer_c0/`; lift verified
pieces, quarantine buggy ones). The **L4 FSQ-VQ** member runs on **one** server GPU; the symbolic
members **L1 BPE / L2 ABBA / L3 SAX-grammar / L5 Matrix-Profile** run CPU-side concurrently; all share
one base alphabet fitted on the pooled vol-unit corpus.

```
tools/candle_lang/
  db_stream.py     [BUILD] DB→bounded-RAM-ring→VRAM streamer (producer/consumer; the HARD-REQ#2 core)
  volunit.py       [BUILD] vol-unit substrate: causal EWMA σ (floor) → mv/gap/body/range/uw/lw
  fsq_core.py      [REUSE+FIX] FSQ from tokenizer_c0/fsq.py (bound() bug FIXED — see blocker #2)
  l4_model.py      [BUILD] 1D-conv VQ enc/dec, FSQ bottleneck, reconstruction-only, NO RevIN
  shard.py         [BUILD] flat (asset,window) → device batches (SINGLE-GPU default; see launch posture)
  train_l4.py      [BUILD] leak-proof GPU harness (jit step, donate xb only, block_until_ready, gc)
  eval_battery.py  [BUILD] A0/A1/A2/A8/A9 — online reductions (no host token-stream accumulation)
  symbolic/{l1_bpe,l2_abba,l3_sax_grammar,l5_matrix_profile,base_alphabet}.py  [BUILD] CPU members
  result_writer.py [REUSE] from run_gate.py writer
  run_789_panel.sh [BUILD] self-advancing tmux orchestrator (preconditions → alphabet → CPU L1/2/3/5 → GPU L4)
```

## HARD requirements (operator — load-bearing)

1. **Unbounded (asset,window) sample parallelism.** Single-asset tokens → flat sample axis, vmap/shard
   freely. **NOT tied to physical GPU count** (the audit's key correction — see launch posture).
2. **Stream DB → bounded RAM ring → VRAM.** `queue.Queue(maxsize=3)` backpressure; never full-load the
   64-asset minute history. Modeled on `train_phase2.py:513-535` SegmentBatchGenerator.
3. **OOM + leak safety paramount** (shared server; a leak/OOM is an incident). Full contract in plan §6.

## ⛔ BLOCKING corrections from the audit (apply DURING build; gate first launch)

1. **NO blind `--gpu 1,2,3`.** `MEM_FRACTION=0.7` is **per-visible-device** + PREALLOCATE default-on →
   a 3-GPU pmap grabs 0.7 of *each* at init and **collides with the standing 2-wide train + GPU-3 eval
   contract → OOM that can kill the co-tenant.** `safe_run` cgroup is **RAM-only, NO VRAM envelope.**
   → **Default single-GPU on a verified-idle GPU**; gate any multi-GPU on `nvidia-smi
   --query-gpu=memory.used` ≈ 0 on all targets. Parallelism is in the batch/vmap axis, not device count.
2. **`fsq.py` `bound()` is broken AND the plan's proposed fix is still wrong** (even L → only 7/8 codes
   reachable, code 0 unreachable). Use the **reference FSQ** formulation (pre-tanh half-level shift for
   even L) and add a **hard round-trip assertion gate**: assert all `prod(levels)` codes reachable on a
   uniform-z sweep before any run writes token IDs.
3. **Eval path is a full-load in disguise.** A1/A8 must **not** accumulate full token streams host-side
   (tens of millions of int64 at minute resolution). Use **online `np.bincount`** accumulators →
   host memory O(codebook), not O(windows).
4. **Tail-batch retrace + reshape crash.** Pad+mask the final partial batch to fixed `per_dev_batch`
   (mask the loss); assert `flat_batch % n_dev == 0`; `n_dev` static from `len(jax.local_devices())`,
   recomputed on the single-GPU fallback path.
5. **`donate_argnums`: donate only the batch buffer `xb`,** never `params`/`opt_state` (no repo
   precedent; donating params + reusing `state.params` in a separate jitted encode → leak/corruption).
6. **Verify the `history.py` citations** (354-373 / 375-389) before relying on them, or drop them; be
   precise that the RevIN **denorm at `fsq.py:102`** is the look-ahead vector to remove, not just 82-85.

Plus the MUST-FIX hygiene: parameterize `data.py:28` DB_PATH (CLI/env, server path), remove
`run_gate.py:119` `PREALLOCATE=false`, override the dead `--out` worktree default, add startup
`PRAGMA quick_check` (workstation DB copy was once page-corrupt).

## Leak/OOM safety contract (summary — full table in plan §6)

Static shapes (pad to fixed `max_len`; σ-EWMA `lax.scan` padded); `static_argnames` for configs;
`block_until_ready` before every D2H; **no device-array accumulation** (np.asarray per chunk, online
reduce); bounded host queue (maxsize=3, backpressure); periodic `gc.collect()`; **MEM_FRACTION=0.7 +
PREALLOCATE default** (safe_run owns it; harness uses `setdefault` only); no `cuda_malloc_async`;
cgroup RAM cap via `safe_run --mem 64G`; startup DB integrity check.

## Launch posture (CORRECTED by audit)

- **Single-GPU default**, on a GPU verified idle via `nvidia-smi` (convention: GPU 3 = eval GPU). This
  fully satisfies HARD-REQ#1 — the flat axis iterates batches; unbounded sample count ≠ 3 devices.
- **Multi-GPU is opportunistic only**, gated on the pool being confirmed idle (all of 1/2/3 ≈ 0 MB
  used). Never blind. Lower `--gpu-frac` (≈0.35) if co-tenanting.
- All under `tools/safe_run.sh --name <tag> --mem 64G --gpu N --gpu-frac 0.7`, in tmux, server only.
  **Never the workstation GPU.** CPU members: `safe_run --cpu --mem 64G`, concurrent with L4.

## First launch order (audit's must-do, ordered)

1. Single-GPU on a verified-empty GPU (highest OOM risk first). 2. Corrected `bound()` + round-trip
assertion. 3. Online bincount in eval. 4. Tail-batch pad+mask + divisibility assert. 5. Donate `xb`
only. 6. Verify/clean citations. → then L4 single-GPU smoke → CPU symbolic members → opportunistic
multi-GPU.

## Deploy note

git-push deploy is broken (TASK_785 / handoff #8) — panel code may need direct scp like A2/C0.
Sub-agents: **build + commit only, NEVER call `deploy.sh`** (main agent owns server writes).
