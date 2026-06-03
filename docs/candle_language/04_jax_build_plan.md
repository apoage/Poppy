I have everything I need. All references are verified against the actual code. Here is the build plan.

---

# BUILD PLAN — JAX-Native GPU-Parallel Candle-Language Tokenizer Panel (TASK_789/791/792, branch C)

**Scope:** the L4 (FSQ-VQ) member runs on the server GPU; L1/L2/L3/L5 symbolic members run CPU-side concurrently. All three hard requirements are load-bearing and honored explicitly below. Every file:line citation has been verified against the working tree.

---

## 0. Verified ground truth (corrections to the maps)

Three things I confirmed by reading the code that change the plan:

- **FSQ `bound()` bug is real** (`tools/tokenizer_c0/fsq.py:39-46`). It computes `offset`/`shift` then **discards both**, returning `jnp.tanh(z) * half`. For even levels (default `(8,8,8,5,5)` has three 8s) the grid is asymmetric and `codes_to_indices` (`fsq.py:54-59`) clips, biasing those dims. **Must fix before trusting L4 token IDs.**
- **`make_features` is `prev_close`-normalized, hourly** (`data.py:112-129`, `BAR_SECONDS=3600` at `data.py:40`). The vol-unit substrate (§2.1 of the design, just read) is a **different** feature set: causal EWMA σ with floor, `mv/gap/body/range` divided by σ, `uw/lw` as wick *fractions* (no vol norm). **Must build net-new.**
- **PREALLOCATE bug confirmed** (`run_gate.py:118-119`): `setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')` + `mem_fraction=0.5`. Violates the operator rule. **Remove the PREALLOCATE line; let safe_run own MEM_FRACTION=0.7** (`safe_run.sh:43` sets it, does not touch PREALLOCATE).

---

## 1. MODULE LAYOUT under `tools/`

New package `tools/candle_lang/` (the C0 salvage `tools/tokenizer_c0/` is the donor; we do not edit it in place — we lift verified pieces and quarantine the buggy parts).

```
tools/candle_lang/
  db_stream.py        # [BUILD] DB→bounded-RAM-ring streamer; per-asset cursor scan, vol-unit feats,
                      #         gap-respecting windows; yields fixed-shape numpy batches. Producer thread.
  volunit.py          # [BUILD] vol-unit substrate: causal EWMA σ (floor), mv/gap/body/range/uw/lw.
                      #         numpy builder (CPU worker) + a jax.vmap kernel mirror for on-device recompute.
  fsq_core.py         # [REUSE+FIX] FSQ class lifted from tokenizer_c0/fsq.py:29-64; bound() bug FIXED.
  l4_model.py         # [BUILD] reconstruction-only 1D-conv VQ encoder/decoder over minute bars,
                      #         FSQ bottleneck (fsq_core). NO RevIN (design uses vol-unit, not per-window norm).
  shard.py            # [BUILD] flat (asset,window) → (n_dev, per_dev_batch, ...) sharding across GPUs 1/2/3.
  train_l4.py         # [BUILD] leak-proof GPU harness: jit step (donate), block_until_ready, gc, static shapes.
                      #         Reuses run_gate.py:156-186 jit-step + chunked np.asarray encode idiom.
  eval_battery.py     # [BUILD] A0/A1/A2/A8/A9 GPU-tiled non-predictive metrics (RQ0 probe DELETED).
  symbolic/
    l1_bpe.py         # [BUILD] BPE-delta over vol-unit base symbols (CPU).
    l2_abba.py        # [BUILD] ABBA adaptive symbolic aggregation (CPU).
    l3_sax_grammar.py # [BUILD] SAX + Sequitur/RePair grammar (CPU).
    l5_matrix_profile.py # [BUILD] Matrix-Profile/VALMOD, SAMPLED+tiled (CPU; see §5).
    base_alphabet.py  # [BUILD] product-quantization bin table fitted ONCE on pooled corpus (design §2.2).
  result_writer.py    # [REUSE] RESULT.md/.json writer lifted from run_gate.py:300-399 pattern.
  run_789_panel.sh    # [BUILD] self-advancing tmux orchestrator: L1/L2/L3/L5 CPU then L4 GPU, exit-gated.
```

**Reuse map (verified file:line):**
| New file | Reuses | From |
|---|---|---|
| `fsq_core.py` | `FSQ`, `round_ste`, `codes_to_indices` | `tools/tokenizer_c0/fsq.py:24-64` (fix `bound` `:39-46`) |
| `db_stream.py` | read-only connect idiom, gap/validity discipline | `data.py:43-44`, `:108`, `:144-153` |
| `train_l4.py` | jit `tok_step`, chunked `np.asarray`-before-concat encode | `run_gate.py:156-162`, `:181-186` |
| `train_l4.py` | producer/consumer bounded-queue shape | `train_phase2.py:513-535` (`SegmentBatchGenerator`) |
| `train_l4.py` | `block_until_ready` fence, `float(v)` metric accum | `train_phase2.py:2142`, `:2151-2152` |
| `db_stream.py` | ranged-query + fixed-shape numpy fill (per-chunk only) | `history.py:354-373` |
| `result_writer.py` | RESULT.md/JSON writer | `run_gate.py` writer block |
| `symbolic/l5` / eval CPU members | numpy harness shape, no-CUDA path | `run_gate_numpy.py` |

**Quarantined (do NOT carry forward):** RevIN per-window norm (`fsq.py:82-85` — conflicts with vol-unit substrate, reintroduces in-window look-ahead); RQ0 next-token-sign probe (`run_gate.py` Probe/`make_seq`/`p_step` — TASK_792 throws it out); `build_dataset` full-concat (`run_gate.py:50-74` — the HARD-REQ #2 anti-pattern); `_preload_to_ram` dense matrix + forward-fill (`history.py:375-389` — manufactures fake bars).

---

## 2. STREAMING DATA PIPELINE — DB → bounded RAM ring → VRAM (HARD-REQ #2)

Three-stage, producer/consumer, modeled on `SegmentBatchGenerator` (`train_phase2.py:513-535`): **numpy only on the worker thread; `jnp.array`/`device_put` only on the consumer.**

### Stage A — per-asset cursor scan (CPU worker, `db_stream.py`)
- Read-only connect verbatim: `sqlite3.connect("file:{path}?mode=ro", uri=True)` (`data.py:43-44`).
- **Single per-asset ordered scan** using the `PRIMARY KEY(asset_id,timestamp)` covering index:
  `SELECT timestamp,open,high,low,close FROM ohlcv WHERE asset_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp`.
  Page by timestamp range (e.g. 256k-row pages) so no single `fetchall()` materializes a 5.4M-row asset. This replaces `data.py`'s **two full table scans** (`data.py:52`, `:81`) — at minute resolution with no aggregation the second scan is unnecessary.
- **Gap detection = `dt==60` runs.** A contiguous run is a maximal stretch where consecutive `timestamp` deltas equal 60 (gaps are *missing rows*, not flat bars — verified in maps). No forward-fill, ever (the `history.py:375-389` forward-fill is forbidden for branch C).
- **3-level validity discipline carried from `data.py`:** bar-valid (all OHLC present), feature-valid (bars t-1 and t valid + finite, `data.py:128`), window-valid (whole window ALL-valid, `data.py:144-148`).

### Stage B — bounded RAM ring (the cap)
- `queue.Queue(maxsize=prefetch)` with `prefetch=3` is the **RAM bound** (the `train_phase2.py:532` model). Worker computes vol-unit features (§3) + slices ALL-valid windows into a **fixed-shape** numpy array `(per_dev_batch * n_dev, W, F)` and `put()`s it; `put()` **blocks when full → backpressure**. Worker re-raises exceptions through the queue (`train_phase2.py` `_worker` pattern). `stop()` = `threading.Event` + join.
- **No window is ever fully materialized for all assets.** Per asset, we slice windows incrementally as the cursor advances and emit batch-sized numpy chunks; the asset's full window set never co-resides. This is exactly what the maps flag as missing — `build_dataset` (`run_gate.py:50-74`) and `_preload_to_ram` both full-load.
- Periodic `gc.collect()` every N batches in the worker (numpy arrays released after `put()`).

### Stage C — flat (asset,window) axis → shardable VRAM batch (HARD-REQ #1)
- Each window row carries `(A=asset_id, TS=end_timestamp)` columns — the single-asset-semantics tag — lifted from `run_gate.py:50,68-73`. **No cross-asset coupling enters:** the model sees `(W,F)` per row; `A`/`TS` are bookkeeping only.
- The producer **interleaves assets** (round-robin like `train_phase2.py:554`) so each emitted batch mixes assets → load-balanced shards, no asset-ordering bias.
- The flat axis is the leading vmap/batch axis (`jax_trading_env.py:49` `vmap(step, in_axes=(0,0,None))` model). **Assets are NOT a parallelism cap** — the batch is `n_dev × per_dev_batch` flat rows reshaped to `(n_dev, per_dev_batch, W, F)` and `jax.device_put_sharded` across `jax.local_devices()` = GPUs 1/2/3 (GPU0 reserved, hidden by `CUDA_VISIBLE_DEVICES=1,2,3`). Consumer does `jnp.array(...)`/`device_put` **only here**, right before the step.

```
DB (25GB, ro)
  └─ per-asset paged cursor (Stage A, numpy worker)
       └─ vol-unit feats + gap-respecting windows (§3, numpy)
            └─ queue.Queue(maxsize=3)   ← RAM CAP + backpressure (Stage B)
                 └─ consumer: reshape (n_dev, per_dev, W, F) → device_put_sharded (Stage C)
                      └─ pmap/shard_map FSQ step on GPUs 1,2,3
```

---

## 3. VOL-UNIT SUBSTRATE as a JAX vmap kernel (`volunit.py`, design §2.1)

Per-bar features (verified against the design doc I just read), with the σ-floor and `maximum(·,ε)` guards the design mandates and CLAUDE.md JAX-pitfalls require:

```python
# numpy builder (CPU worker, in db_stream) AND a jax.vmap mirror (on-device recompute/eval)
logret = jnp.log(jnp.maximum(c[1:], 1e-12) / jnp.maximum(c[:-1], 1e-12))
sigma  = ewma_causal(jnp.abs(logret), halflife=H)          # causal: bars ≤ t only
sigma  = jnp.maximum(sigma, SIGMA_FLOOR)                    # floor → flat regimes don't blow up
den    = jnp.maximum(sigma, 1e-10)                          # guard division (CLAUDE.md)
mv   = logret / den
gap  = jnp.log(jnp.maximum(o[1:],1e-12)/jnp.maximum(c[:-1],1e-12)) / den
body = jnp.log(jnp.maximum(c[1:],1e-12)/jnp.maximum(o[1:],1e-12)) / den
rng  = jnp.log(jnp.maximum(h[1:],1e-12)/jnp.maximum(l[1:],1e-12)) / den
hl   = jnp.maximum(h[1:]-l[1:], 1e-9)
uw   = (h[1:] - jnp.maximum(o[1:],c[1:])) / hl              # ∈[0,1], NO vol norm
lw   = (jnp.minimum(o[1:],c[1:]) - l[1:]) / hl
# F = [mv, gap, body, range, uw, lw]  (6 channels)
```

- **Causal EWMA** is a `lax.scan` over the (static, padded) window: `s_t = α·|r_t| + (1-α)·s_{t-1}`, `α=1-2^(-1/H)`. Padded to static `max_len` (§6) to avoid the dynamic-T retrace stall (`path_solver.py:74-75`, TASK_128).
- **Drift re-anchor every 1440 min (UTC-day)** per design §2.5 — reset/re-seed σ on the day boundary, done numpy-side in the worker on contiguous runs.
- `vmap` over the flat (asset,window) leading axis: `jax.vmap(volunit_kernel, in_axes=0)`. No `lax.cond` inside vmap; all guards are `jnp.maximum`/`jnp.where` (CLAUDE.md).
- **$225/$0.45 invariance** (design §2.1): pure rescale leaves log-ratios and σ unchanged → identical vol-units → identical token. This is what eval A0 (§5) checks.

---

## 4. L4 FSQ-VQ MODEL (reconstruction-only) + symbolic orchestration

### L4 model (`l4_model.py` + `fsq_core.py`) — GPU
- **FSQ core reused** from `fsq.py:24-64` (`round_ste` STE, `quantize`, `codes_to_indices`), collapse-proof (implicit codebook `prod(levels)`, no learned codebook → no collapse). **Bug fix in `bound()`:** make even levels symmetric — replace the dead-code body (`fsq.py:45-46`) with a half-integer offset for even L:
  ```python
  half = (self._levels - 1)/2.0
  return jnp.tanh(z) * half   # then quantize on a grid offset by 0.5 for even L in codes_to_indices
  ```
  (Validate token-id round-trip on a synthetic grid before trusting IDs.)
- **Encoder/decoder rebuilt:** replace the flatten-MLP + RevIN (`fsq.py:82-99`) with a **1D-conv encoder over minute bars on the 6-ch vol-unit substrate** → `Dense(d)` → FSQ → 1D-conv decoder. **No RevIN** (vol-unit substrate already normalizes; stacking RevIN reintroduces in-window look-ahead and breaks rescale-invariance reasoning — confirmed gotcha).
- **Reconstruction-only loss:** `mean((recon - x_volunit)**2)`. The next-token probe is **deleted** (not extended). This is the only term — no prediction framing anywhere (design §1).
- Reuse the jit step shape from `run_gate.py:156-162` (`value_and_grad` + `apply_gradients`), add `donate_argnums` (§6).

### Symbolic members (CPU-side, concurrent) — `symbolic/`
All five members share **one base alphabet** (`base_alphabet.py`): product-quantization bin table (move 8 / gap 3 / body·range 4 / wick 3, design §2.2) fitted **once on the pooled 64-asset vol-unit distribution** (quantile/Lloyd-Max, fat-tailed — not Gaussian). This shared table is what makes the same shape land on the same symbol across assets (transfer is structural, not learned).

| Member | What | Runs |
|---|---|---|
| **L1 BPE-delta** | frequency-greedy BPE/RePair merge over run-length-reduced base symbols (design §2.3) | CPU |
| **L2 ABBA** | adaptive Brownian-bridge symbolic aggregation | CPU |
| **L3 SAX+grammar** | SAX + Sequitur/RePair grammar induction | CPU |
| **L4 FSQ-VQ** | conv-VQ reconstruction (above) | **GPU 1/2/3** |
| **L5 Matrix-Profile** | VALMOD motif discovery, SAMPLED+tiled (§5) | CPU |

**Orchestration:** L1/L2/L3/L5 are counting/info-theoretic and run under `safe_run --cpu --mem 64G` **concurrently** with L4's GPU run (different resources, no contention). The panel orchestrator (`run_789_panel.sh`, §7) launches the four CPU members first (they can overlap), then L4 on GPU last, all process-exit gated in one tmux session (`run_overnight.sh` pattern, lore §172-173 convention). The base-alphabet fit is a shared **prerequisite cell** run once before any member.

---

## 5. GPU-TILED EVAL BATTERY (`eval_battery.py`) — non-predictive (TASK_792)

RQ0 sign-probe **deleted**. All metrics computed on TEST windows via the chunked `np.asarray`-before-concat encode (`run_gate.py:181-186`) so no device-array list grows.

| Eval | Definition | GPU-tiling |
|---|---|---|
| **A0 rescale-invariance** | encode `x` and `λ·x` (pure price rescale); token-ID agreement must be ~1.0 (vol-unit ⇒ identical tokens) | vmap over flat axis, two passes, compare IDs host-side |
| **A2 reconstruction** | recon MSE on vol-unit features + de-normalized price-path MSE within tolerance | reuse `run_gate.py:194-200` price-path reconstruction |
| **A1 corrected MDL** | code length of token stream **+ null bits coded at full base-symbol entropy** (design §2.4) + boundary bits, vs raw base-symbol baseline (like-for-like) | token histograms tiled per device, reduced host-side as `float()` |
| **A8 cross-asset transfer** | fit codebook on asset subset, measure recur/coverage on held-out assets; controls = **returns-shuffled** (`data.py:165` `phase_shuffle_returns` reused as the A8 null) **and** random-merge-vocab | shuffle control through same pipeline; per-asset chrono purge `run_gate.py:77-113` |
| **A9 length non-degeneracy** | token bar-length distribution not collapsed (not all len-1, not one giant token) | histogram, host-side |

**L5 Matrix-Profile O(n²) avoidance (the load-bearing tiling rule):**
- **Subsample anchors:** compute the matrix profile against a **sampled** set of query subsequences (e.g. K random anchors per asset), not all n. O(n²) → O(K·n).
- **Tile the distance matrix:** process query×reference in fixed `(tile, tile)` blocks (STOMP-style incremental), reducing each tile to its min host-side, never materializing the full n×n. Per-asset only (single-asset tokens) so n is one asset's length, not 64×.
- Runs CPU under `safe_run --cpu`; the tiling keeps RAM bounded under the 64G cap.

---

## 6. LEAK / OOM SAFETY CONTRACT (HARD-REQ #3)

Every rule, and exactly where it applies. (Note: `donate_argnums` has **zero precedent in this repo** — it is net-new and must be validated, not copied.)

| Rule | Where applied | Citation/precedent |
|---|---|---|
| **Static shapes — pad to fixed `max_len`** | window `W`, batch `per_dev_batch`, σ-EWMA `lax.scan` length, conv lengths all fixed at config time; variable-length token runs padded to `MAX_RUN` | `path_solver.py:38-84` (`static_argnames`, `jnp.zeros((max_len,))`); TASK_128 stall is real |
| **`static_argnames` for configs** | `W`, `fsq_levels`, `batch`, `n_dev` passed as static, never dynamic Python ints per call | `path_solver.py:38`, CLAUDE.md |
| **`donate_argnums` on jit step** | `train_l4.py` jit step donates `(params, opt_state)` (and batch buffer) | **net-new**; validate (no in-repo example) |
| **`block_until_ready` before D2H** | after every `tok_step` in the training loop AND before each consumer `device_put` of the next batch; explicit before each encode→`np.asarray` | `train_phase2.py:2142`; explicit, not relying on implicit `np.asarray` block |
| **No device-array accumulation in python lists** | encode loop converts `np.asarray(i)` per 4096-chunk before concat; metrics accumulated as `float(v)`; pop+discard raw arrays each step | `run_gate.py:181-186`; `train_phase2.py:2145,2151-2152` |
| **Bounded host buffer** | `queue.Queue(maxsize=3)` is the RAM cap; `put()` blocks (backpressure); numpy stays on worker, never `jnp` in worker | `train_phase2.py:532-535` |
| **Periodic `gc.collect()`** | every N batches in worker (after `put`) and every N steps in consumer loop (no precedent — add) | net-new |
| **MEM_FRACTION=0.7, PREALLOCATE default** | set by `safe_run.sh:43`; harness uses `os.environ.setdefault` only (so safe_run wins); **remove the `PREALLOCATE=false` line** that exists in `run_gate.py:119`; do NOT copy `train_phase2.py:39-40` (0.85) | `safe_run.sh:43`, MEMORY `feedback_jax_vram_preallocate` |
| **No `cuda_malloc_async`** | leave XLA_FLAGS default allocator (illegal-address under churn) | `train_phase2.py:34-35` |
| **bf16 carry / `jax.checkpoint(body)`** | ONLY if a temporal `lax.scan` with backward pass is added (σ-EWMA is forward-only in the worker → not needed unless a recurrent encoder is used) | `titans_memory.py:429-464`, TASK_749 |
| **cgroup hard RAM cap** | every run wrapped in `safe_run.sh` (`MemoryMax`, `MemorySwapMax=0`) → OOM reaps only the scope | `safe_run.sh:50-53` |
| **`RLock` not `Lock`** | follow queue.Queue + threading.Event model; no raw Lock | CLAUDE.md JAX-pitfalls |
| **DB integrity check at startup** | cheap `PRAGMA quick_check` or row-count sanity on the server DB copy (workstation copy was once page-corrupt; zstd→rsync doesn't detect source corruption) | MEMORY DB-integrity caveat |

---

## 7. safe_run LAUNCH COMMANDS (server, gpu-server)

DB path is **CLI/env**, not the hardcoded workstation path (`data.py:28` must not ship). GPU0 reserved → use 1/2/3.

**L4 single-GPU (start here — validate harness, bug fix, leak contract on one GPU):**
```bash
tools/safe_run.sh --name 789_L4 --mem 64G --gpu 1 --gpu-frac 0.7 -- \
  python -u tools/candle_lang/train_l4.py \
    --db /path/to/server/trading_data.db \
    --window 64 --fsq-levels 8,8,8,5,5 \
    --out runs/789_token_c0/L4_RESULT.md
```

**L4 multi-GPU (flat-axis sharding across 1/2/3, single scope, pmap sees 3 devices):**
`safe_run.sh:43` writes `CUDA_VISIBLE_DEVICES` as a single string, so a comma list passes through:
```bash
tools/safe_run.sh --name 789_L4x3 --mem 64G --gpu 1,2,3 --gpu-frac 0.7 -- \
  python -u tools/candle_lang/train_l4.py \
    --db /path/to/server/trading_data.db --shard pmap \
    --window 64 --per-dev-batch 1024 \
    --out runs/789_token_c0/L4x3_RESULT.md
```

**CPU symbolic members (concurrent, no GPU contention):**
```bash
tools/safe_run.sh --name 789_L1 --mem 64G --cpu -- \
  python -u tools/candle_lang/symbolic/l1_bpe.py --db /path/... --out runs/789_token_c0/L1_RESULT.md
# (L2, L3, L5 identical shape)
```

**Panel orchestrator (single self-advancing tmux session, exit-gated, survives workstation-off):**
```bash
tmux new-session -d -s 789_panel \
  'cd ~/Trader && source trader_env/bin/activate && bash tools/candle_lang/run_789_panel.sh 2>&1 | tee -a runs/789_token_c0/panel.log'
```
`run_789_panel.sh` (modeled on `run_overnight.sh` + `launch-experiment.sh:1-66`): preconditions block (DB integrity, devices visible) → base-alphabet fit cell → L1/L2/L3/L5 CPU cells → L4 GPU cell, each gated on the **prior process's exit** (no stale flag files; lore §172-173 convention `/tmp/789_*.flag`).

---

## 8. BUILD vs REUSE summary

**REUSE (verified):** FSQ core `fsq.py:24-64` (with `bound` fix); read-only connect + gap/validity discipline `data.py:43,108,144-153`; jit-step + chunked-encode `run_gate.py:156-186`; producer/consumer bounded-queue `train_phase2.py:513-535`; `block_until_ready` fence `train_phase2.py:2142`; `phase_shuffle_returns` `data.py:165` as A8 null; ranged-query fill `history.py:354-373`; `safe_run.sh` envelope as-is; `run_gate_numpy.py` for CPU members.

**BUILD net-new:** `db_stream.py` (streaming DB→ring→VRAM — does not exist anywhere); `volunit.py` (vol-unit substrate — `data.py` has the wrong `prev_close` hourly features); `l4_model.py` (conv-VQ encoder, no RevIN — replaces flatten-MLP); `shard.py` (multi-GPU flat-axis sharding — c0 is single-device); `eval_battery.py` (A0/A1/A2/A8/A9, RQ0 deleted); all five `symbolic/` members + `base_alphabet.py` (only L4-style FSQ exists in the salvage); `donate_argnums` wiring (zero repo precedent); `run_789_panel.sh`.

**MUST-FIX before trust:** `fsq.py:39-46` `bound()` dead-code (even-level bias); remove `run_gate.py:119` `PREALLOCATE=false`; parameterize `data.py:28` DB_PATH; override the dead `--out` worktree default (`run_gate.py:45`); add startup `PRAGMA quick_check`.

**Flagged risks:** git-push deploy is broken (handoff #8 / TASK_785) — panel code may need direct scp like A2/C0; sub-agent rule — **build + commit only, never call `deploy.sh`**.