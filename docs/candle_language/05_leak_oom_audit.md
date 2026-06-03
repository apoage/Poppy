I have all the ground truth I need. Now let me deliver the adversarial audit.

The plan's proposed `bound()` fix is itself subtly wrong (it still uses `tanh(z)*half` with no odd/even handling reaching `codes_to_indices`, and `codes_to_indices` already does the `+half` round/clip — so the "even-level bias" claim needs precise restatement). And the multi-GPU MEM_FRACTION interaction is the single biggest OOM risk. Here is the audit.

---

# ADVERSARIAL AUDIT — Candle-Language Tokenizer Build Plan

Verdict up front: the streaming architecture is sound in shape and HARD-REQ #2 is genuinely met. But the plan has **one plan-killing OOM hazard it is actively blind to (multi-GPU MEM_FRACTION + shared training pool contention)**, a **proposed `bound()` fix that is still wrong**, and **four leak/retrace gaps presented as "handled" that are not**. Citations are ~85% accurate; the ones that are wrong are wrong in load-bearing ways. Details, each with a concrete fix.

## THE ONE THAT MATTERS MOST — multi-GPU OOM on the shared training pool

The plan treats GPUs 1/2/3 as free for an exclusive `--gpu 1,2,3` pmap run at `--gpu-frac 0.7`. **They are not free.** Per MEMORY (`feedback_server_stays_under_load`, TASK_742): the standing server contract is *2-wide `train_phase2.py` on GPU 1/2 + evals on GPU 3*. Your `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` is **per-process per-visible-device** — a pmap process with 3 GPUs visible grabs 0.7 of *each* of GPU 1, 2, AND 3 at init. If anything else is training there (the default state of this server), you get **two processes each preallocating 0.7 → 1.4× VRAM demand → instant CUDA_ERROR_OUT_OF_MEMORY on whichever GPU collides**, and because PREALLOCATE is default-on, the OOM happens at *init*, not gradually — it can take down the co-tenant training job too. `safe_run`'s cgroup cap is **RAM-only** (`MemoryMax`, line 52); it does **nothing** for VRAM. There is no VRAM envelope here at all.

Concrete fixes (do all):
1. **Do not launch `--gpu 1,2,3` blind.** Gate the multi-GPU cell on `nvidia-smi --query-gpu=memory.used` being near-zero on all three target GPUs, in the orchestrator precondition block. If any is occupied, fall back to single-GPU on the one free GPU.
2. **Default to single-GPU** (`--gpu 3` — the eval GPU by convention) and treat 3-GPU as an opportunistic mode requiring an explicit "pool is idle" assertion. HARD-REQ #1 (unbounded sample parallelism) does **not** require 3 physical GPUs — it requires the flat axis to *vmap/shard freely*, which a single GPU satisfies by iterating batches. The plan conflates "embarrassingly parallel" with "must occupy the whole pool."
3. **Lower `--gpu-frac` for the co-tenant case** (e.g. 0.35) so two processes fit, OR keep 0.7 but only on a verified-empty GPU.
4. The plan says "GPU0 hidden by `CUDA_VISIBLE_DEVICES=1,2,3`." Fine — but verify GPU0 is *actually* the VirtualBox one on this host before trusting the index (MEMORY notes the index-mismatch class of bug existed on multi-GPU here).

## SECOND — the proposed `bound()` fix is still wrong

The diagnosis is correct (the existing `fsq.py:44-46` computes `offset`/`shift` then throws them away, returning `jnp.tanh(z) * half`). But the plan's *replacement* is:
```python
half = (self._levels - 1)/2.0
return jnp.tanh(z) * half   # "then quantize on a grid offset by 0.5 for even L in codes_to_indices"
```
This is **identical to the buggy line** and the "offset by 0.5 … in codes_to_indices" is hand-waved with no code. Look at the actual round-trip: `quantize` does `round_ste(tanh(z)*half)`; `codes_to_indices` (`fsq.py:56-58`) does `round(zhat + half)` then `clip(0, L-1)`. For **even L** (e.g. 8): `half = 3.5`, so `tanh(z)*3.5 ∈ (-3.5, 3.5)`, `round_ste` → integers `{-3..3}` (7 values, **never reaches 8**), then `+3.5` → `{0.5..6.5}` → round → `{1..7}` after clip → **only 7 of 8 codes ever used, and code 0 is unreachable**. The standard FSQ fix (Mentzer et al.) is to shift *before* tanh by a half-level for even L so the grid is symmetric in count: `z = tanh(z + atanh-style offset)`. Concretely, use the reference formulation:
```python
def bound(self, z):
    half = self._levels / 2.0                    # note: L/2, not (L-1)/2
    offset = jnp.where(self._levels % 2 == 0, 0.5, 0.0)
    shift = jnp.tan(offset / half)               # pre-tanh shift (reference FSQ)
    return jnp.tanh(z + shift) * half - offset
```
with `codes_to_indices` recomputed as `round(zhat + (self._levels-1)/2)` clipped to `[0, L-1]`. **And the plan's own validation step ("validate token-id round-trip on a synthetic grid") must be a hard gate, not a comment** — assert all `prod(levels)` codes are reachable on a uniform-z sweep before any training run writes token IDs.

## THIRD — leak/retrace gaps presented as solved but not

**(a) `donate_argnums` is net-new AND interacts badly with the encode pass.** The plan correctly flags zero repo precedent. But the real trap: you cannot donate `(params, opt_state)` on the train step **and** reuse `state.params` in a separately-jitted `encode` fn (the `run_gate.py:176-189` idiom you're copying) within the same loop — donation invalidates the donated buffer, and if the host still holds a reference JAX raises or silently corrupts. Fix: only donate the **batch input buffer** (`xb`), not params/opt_state, until you have a validated single-owner state machine. Donating params is the kind of "clever" that leaks or crashes under churn — and you have no in-repo example to copy, so keep it minimal.

**(b) `block_until_ready` placement is under-specified and will leak the queue.** Plan says "after every `tok_step` AND before each consumer `device_put`." The actual leak vector is different: in the `batched_encode` loop (`run_gate.py:181-186`), `np.asarray(i)` *does* force a sync, but you accumulate `idxs.append(np.asarray(i))` host-side — for A8 cross-asset transfer over the **full 64-asset test set**, that host list is O(total_windows) int64 token IDs. On minute data that's tens of millions of rows × the histogram pass. Fix: **never accumulate full token streams host-side**; reduce online — maintain a running `np.bincount` accumulator per device chunk and discard the chunk (`counts += np.bincount(np.asarray(i), minlength=cb)`), so host memory is O(codebook), not O(windows). The current `run_gate.py` pattern full-concats and only works because hourly aggregation made N tiny; at minute resolution it is the **same anti-pattern HARD-REQ #2 forbids, just moved to the eval path.**

**(c) Static-shape claim has a hole: the final ragged batch.** Plan pads `W`, `per_dev_batch`, EWMA scan length — good. But the producer slices "batch-sized numpy chunks" as the cursor advances; the **last chunk per asset/epoch is smaller than `per_dev_batch`**, and feeding it to a jit'd step with `static_argnames` batch size triggers a **retrace** (new shape) → recompile → VRAM spike + latency. This is exactly the dynamic-shape retrace the plan claims to prevent. Fix: **drop or pad the tail batch to fixed `per_dev_batch`** (pad with a validity mask, mask the loss). Never feed a short final batch.

**(d) `device_put_sharded` requires `n_dev` to evenly divide the flat batch — and the worker doesn't guarantee it.** Plan reshapes flat `(n_dev × per_dev_batch, W, F)` → `(n_dev, per_dev_batch, W, F)`. If the queue ever yields a partial batch (see (c)) or `n_dev` changes (single-GPU fallback from fix #1 above → `n_dev=1` but code wrote `n_dev=3`), the reshape throws or silently misshards. Fix: assert `flat_batch % n_dev == 0` at consumer, and make `n_dev` a config-time static derived from `len(jax.local_devices())`, recomputed on the fallback path.

## FOURTH — confirmed-correct findings in the plan (keep these)

- **PREALLOCATE removal is right.** `run_gate.py:119` does `setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')` — confirmed. Removing it and letting safe_run own only MEM_FRACTION (safe_run line 43 sets `XLA_PYTHON_CLIENT_MEM_FRACTION`, does NOT touch PREALLOCATE → default-on) is correct and matches `feedback_jax_vram_preallocate`.
- **`data.py` is hourly + `prev_close`-normalized.** Confirmed: `BAR_SECONDS=3600` (line 40), features are `/denom` where `denom=prev_c` (lines 119-125). Vol-unit substrate is genuinely net-new. Correct.
- **`data.py` does two full scans** (line 52 GROUP BY aggregation, line 81 `SELECT ... WHERE asset_id=?` full fetchall). Confirmed — and line 85 `for t,o,c in cur.fetchall()` materializes the whole asset in a Python dict. Replacing with paged ranged scan is correct and necessary.
- **`build_dataset` (`run_gate.py:50-74`) full-concats** all assets via `np.concatenate`. Confirmed HARD-REQ #2 anti-pattern. Correctly quarantined.
- **`phase_shuffle_returns` at `data.py:165`** exists and is a valid A8 null. Confirmed.
- **`--out` dead default** (`run_gate.py:45`) points at a stale worktree path. Confirmed; must override.
- **DB hardcoded** at `data.py:28`. Confirmed; must parameterize.

## CITATION ERRORS (fix before building — they erode trust in the rest)

- "`fsq.py:82-99` flatten-MLP + RevIN" — RevIN is `fsq.py:82-85`, MLP is `89-99`. Close enough but the **`recon` de-normalization at line 102 (`recon_n * std + mean`)** is the actual look-ahead vector you must drop, not just lines 82-85. Be precise about what you remove.
- "`history.py:354-373` ranged-query fill" and "`history.py:375-389` forward-fill" — **I could not verify these in this audit** (history.py not read). Treat as UNVERIFIED; the plan claims they're verified. Re-check before citing as reuse.
- "RESULT.md writer `run_gate.py:300-399`" — actual writer is `write_result_md` at **336-394**. Minor.
- "`block_until_ready` `train_phase2.py:2142`" — confirmed exact.
- "`SegmentBatchGenerator` `train_phase2.py:513-535`" — confirmed; queue at 532, Event at 533, `_worker` at 877 (plan cites the right pattern; the `_worker` exception-via-queue idiom you want is at **887-895**, not in 513-535).

## STRUCTURAL REQUIREMENT VERIFICATION

**(a) Unbounded (asset,window) parallelism — PARTIALLY MET, with a conceptual error.** The flat-axis design is correct: single-asset tokens → flat `(asset,window)` rows → leading vmap axis → shard. That genuinely removes the "64 assets = 64-way cap" failure. **But the plan ties unbounded parallelism to occupying 3 physical GPUs**, which is both false (one GPU iterating batches is already unbounded in sample count) and dangerous (the contention OOM above). Decouple: parallelism is in the *batch/vmap axis*, not the *device count*. Met once you stop equating them.

**(b) True DB→bounded-buffer→VRAM streaming — MET on the train path, NOT on the eval path.** Stage A/B/C with `queue.Queue(maxsize=3)` backpressure is a genuine bounded stream and correctly avoids `build_dataset`/`_preload_to_ram` full-load. **However** the eval battery (A1 histograms, A8 transfer) as specified inherits `run_gate.py`'s full-concat host accumulation (finding 3b) — that is a full-load in disguise at minute resolution. Not met until eval reductions are made online/streaming.

## MUST-DO BEFORE FIRST LAUNCH (ordered)
1. Single-GPU only, on a verified-empty GPU, until contention gating exists. (highest OOM risk)
2. Correct `bound()` to reference FSQ + hard round-trip assertion gate. (silent wrong token IDs)
3. Online `bincount` reduction in eval; no full token-stream host accumulation. (eval-path full-load)
4. Tail-batch pad+mask; assert `flat % n_dev == 0`. (retrace + reshape crash)
5. Donate only `xb`, not params/opt_state. (net-new leak risk)
6. Verify `history.py` citations or drop them. (unverified reuse)

Relevant files: `/home/lukas/Trader/tools/tokenizer_c0/fsq.py` (bound bug lines 39-46, recon denorm line 102), `/home/lukas/Trader/tools/tokenizer_c0/data.py` (DB hardcode line 28, two scans 52/81-85, phase_shuffle 165), `/home/lukas/Trader/tools/tokenizer_c0/run_gate.py` (PREALLOCATE 119, dead --out 45, full-concat 50-74, encode idiom 181-186, writer 336-394), `/home/lukas/Trader/tools/safe_run.sh` (RAM-only cgroup, no VRAM cap; CUDA_VISIBLE line 43), `/home/lukas/Trader/train_phase2.py` (producer/consumer 513-535, _worker 877-895, block_until_ready 2142).