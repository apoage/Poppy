# TASK_749 — Titans-MAG OOM Fix Review (Phase 1)

**Status**: REVIEW — no code changes yet. Phase 2 (implementation) blocked on commit of this review.
**Date**: 2026-05-18.
**Context**: Titans-MAG OOM'd twice (default batch → 42.66 GB request; `--batch-size 128` → 21.33 GB request). Both on 24 GB device. See `runs/749_titans_mag/result.md` for postmortems. Smoke at `--batch-size 64` (workstation, 2026-05-16) passed; production at b=128 OOM'd → smoke didn't exercise full-batch shape.

Target: fit Titans-MAG at `--batch-size 128 --n-slots 4 --titans-memory --titans-variant mag --titans-memory-depth 2 --titans-memory-hidden 32 --titans-persistent-tokens 4 --titans-gate-init -3 --titans-lr 3e-3 --reset-titans-memory` on a 24 GB RTX 3090, with ≥6 GB safety margin (i.e. peak VRAM ≤ 18 GB on workstation).

---

## 1. Root cause analysis

The OOM is the **`lax.scan` carry trajectory** of `_titans_scan_body_batched`. With `jax.checkpoint(body)` already in place (`titans_memory.py:430`), the body's per-step intermediates are remat'd — but `lax.scan` itself still stores the **carry-INPUT at every step** for reverse-mode autodiff. That carry tape is what hits the 18 GB budget overrun.

### Per-step carry shape (canonical Stage-B at b=128)

Config:
- `B = 128`, `A = 64` → `N = B*A = 8192` parallel streams (`titans_memory.py:375` — `N, T, D = x.shape` after reshape).
- `T_aug = T + persistent_tokens = 64 + 4 = 68` (`titans_memory.py:383-386`).
- `D = embed_dim = 64`, `H = memory_hidden = 32`, `memory_depth = 2`.
- MLP layer shapes: `(D=64, H=32)`, `(H=32, D=64)` (`titans_memory.py:349-353`).

Carry tensors at each scan step (both tuples held per step):

| Tensor | Shape | Bytes (fp32) |
|---|---|---|
| `M_params[0]` (`new_M_0`) | `(N=8192, 64, 32)` | 67.1 MB |
| `M_params[1]` (`new_M_1`) | `(N=8192, 32, 64)` | 67.1 MB |
| `S_params[0]` (`new_S_0`) | `(N=8192, 64, 32)` | 67.1 MB |
| `S_params[1]` (`new_S_1`) | `(N=8192, 32, 64)` | 67.1 MB |
| **Per-step total** | | **268 MB** |
| **× T_aug = 68 steps** | | **~18.2 GB** |

This matches the observed 21.33 GB OOM request size (the difference is bookkeeping/scratch buffers + transient cotangents).

### Why `jax.checkpoint(body)` doesn't help

`jax.checkpoint(body)` at `titans_memory.py:430` correctly remats the BODY intermediates (the `grad_M`, `pre_list`, `h_list` tensors inside `_batched_grad_M`). Those would be `(N, D)` or `(N, H)` and `(N, D, H)`-sized tensors, ~67-134 MB each per layer, all freeable.

BUT the `lax.scan` reverse-mode autodiff stores the **carry-input at every step** independently of `jax.checkpoint`. From the JAX reverse-mode-scan implementation, the backward scan iterates `t = T-1, ..., 0` and at each backward step needs the carry-input `c_t` to recompute the body forward (under remat) and accumulate cotangents. This carry tape is `O(T × |carry|)` regardless of any per-body checkpointing.

### Why `stop_gradient` on the carry doesn't help with memory

`titans_memory.py:221-223` applies `jax.lax.stop_gradient` to `M_params, S_params` (the carry-INPUT) inside the body:

```python
M_params, S_params = carry
M_params = jax.tree_util.tree_map(jax.lax.stop_gradient, M_params)
S_params = jax.tree_util.tree_map(jax.lax.stop_gradient, S_params)
```

This correctly cuts the **gradient flow** through the trajectory — the cotangent `d_carry` at the backward boundary is zero. But it does NOT cut the **forward dependence**: the body still reads `M_t, S_t` to compute `grad_M` and the retrieval `y_t`. So `lax.scan` still needs to store/recompute `M_t` at every step. The fact that this gradient is zero means we technically don't need the carry tape for backprop *through the carry chain*, but `lax.scan`'s default reverse-mode doesn't know that — the tape is allocated anyway.

### Cross-check against postmortem

`runs/749_titans_mag/result.md:101-108` already names this: "the truncated-BPTT scan over the temporal window retaining forward activations of the inner Titans MLP (even with `jax.lax.stop_gradient` on M_params/S_params, the per-step inner-loop activations are not freed)". The 18 GB carry tape is structural; postmortem correctly diagnosed the constraint surface.

---

## 2. Candidate fixes — ranked

| # | Fix | Expected VRAM after | LOC | Files | Risk |
|---|---|---|---|---|---|
| **A** | **bfloat16 carry storage (compute fp32)** | tape → 9 GB; total ~12 GB | ~15 | 1 (`titans_memory.py`) | Low — numerical risk only for `S_params` accumulation; clipping already in place |
| B | `memory_hidden = 32 → 16` | tape → 4.6 GB; total ~8 GB | ~5 | 1 (`titans_memory.py` default + CLI default) | Med — changes Titans capacity; needs full smoke re-validation |
| C | `memory_depth = 2 → 1` | tape → 9 GB (only one layer); total ~12 GB | ~3 | 1 (CLI default) | Med — strictly less expressive memory; needs smoke |
| D | `--titans-persistent-tokens 4 → 0` | tape → 17.1 GB (64/68); total ~20 GB — **insufficient** | 2 | 1 | Low impact, won't fix alone |
| E | `policy=jax.checkpoint_policies.nothing_saveable` on body | **NO CHANGE** — only affects body intermediates, not scan carry tape. | 2 | 1 | **Won't fix the OOM** |
| F | Slice scan window (last N steps) | tape → smaller but loses signal | ~20 | 1 | High — Titans semantics changed |
| G | Custom `vjp` that drops carry tape (recompute scan in backward) | tape → 268 MB; total ~3 GB | ~50 | 1 | High — out of fix scope; brittle |

### A. bfloat16 carry storage (compute fp32) — RECOMMENDED

Store `M_params, S_params` as bf16 in the carry. Inside the body, immediately cast back to fp32 for computation. Cast `new_M, new_S` back to bf16 before returning. Compute precision is unchanged; only the carry tape and the per-step carry-input copies halve.

Concrete change (3 sites in `titans_memory.py`):
1. **Init carry as bf16** (lines 403-410): `M_init_b` and `S_init_b` cast `.astype(jnp.bfloat16)` instead of `self.dtype`.
2. **Body upcast on read** (lines 220-223): after stop_gradient, cast carry to fp32 for compute.
3. **Body downcast on write** (lines 248-251): cast `new_M, new_S` to bf16 before returning.

Expected reduction: carry tape `18.2 GB → 9.1 GB`. Total Stage-B request `21.3 GB → ~12.2 GB`. Safety margin from 24 GB: 11+ GB. Well past the 18 GB target.

Smoke risk: the carry stores M (memory weights) and S (surprise momentum). Both have known dynamic ranges from the smoke runs:
- `mem_norm` saw `0.05 → 0.92` — comfortably in bf16 range (~3.4e38 max, ~1.2e-38 min normal, ~3.9e-3 minimum precision step).
- `S` is bounded by per-stream Frobenius clipping at `2.0` (`titans_memory.py:247`).
- `grad_M` is per-stream-norm-clipped at `1.0` (`titans_memory.py:239-240`).
- Inner-loop coefficients: `eta ≤ 1`, `theta ≤ 1`, `alpha ≤ 0.99`. No precision-killing tiny values.

bf16 has 7-bit mantissa (vs 23 for fp32). Worst-case relative precision loss is ~256× larger ULP. Since the clippings cap dynamic range and the per-step update is `(1-α)*M + S` with α near 0, the relative error per step is ~3.9e-3, dominated by the S contribution. Over 68 steps, accumulated drift in M is bounded by 68 × 3.9e-3 × ||S|| ≤ 68 × 3.9e-3 × 2 ≈ 0.53 in worst-case Frobenius norm. The mem_norm growth signal (0.05 → 0.92) and gate engagement (Δ = 3.4e-3) operate at scale ≥0.05, so 0.53 worst-case drift is large RELATIVE to the signal — concerning, but bounded.

Mitigations if smoke regresses on bf16 carry:
- **Fallback**: combine A + B (bf16 carry + hidden=16) for 4× reduction with smaller MLP.
- **Alternative**: float16 storage instead of bfloat16 — slightly more mantissa, smaller range. Probably worse for momentum S.

### B. `memory_hidden = 32 → 16`

Halves H in both MLP layers. Per-step carry: `8192 × (64×16 + 16×64) × 4 = 67 MB` → tape = 4.6 GB. Total Stage-B ~8 GB. Big margin.

Cost: Titans inner MLP capacity halves. Smoke was validated at H=32 only. The TASK_749_TITANS_PIVOT.md §5.3 explicitly mentions H=D as the default, with `memory_hidden=32` (H<D) as the "key VRAM savings for big B*A" path (`titans_memory.py:344`) — H=16 is going further in the same direction. Likely still works but no empirical validation exists.

### C. `memory_depth = 2 → 1`

Single-layer memory: shape `(D, D)`. Carry per step `2 × N × D × D × 4 = 8192 × 64 × 64 × 4 × 2 = 268 MB` (same!) — wait, same as before? No: single layer means M_params has 1 tensor `(N, D, D)`, S_params has 1 tensor `(N, D, D)` → `8192 × 64 × 64 × 4 × 2 = 268 MB`. **Same** as depth=2 because D×D = 64×64 = 4096 and depth=2 with H=32 gives 2×(64×32) = 4096. Same number of params per stream. **No memory reduction.**

(Actually slightly better because no GELU intermediate is needed in the body — but the body intermediates are remat'd already; carry is what matters.)

**C is not a memory fix.** Drop from the candidate list.

### D. `--titans-persistent-tokens 4 → 0`

T_aug = 64 instead of 68. Tape: `64 × 268 MB = 17.1 GB`. Still over budget. Worth combining with A but not alone.

### E. `policy=jax.checkpoint_policies.nothing_saveable`

Affects the per-body intermediates: forces ALL inside-body activations to be recomputed during backward. With our `_batched_grad_M` already analytical (single-level autodiff), `jax.checkpoint(body)` already saves nearly nothing — the policy change is a no-op on memory. **Won't fix this OOM.**

### F. Slice scan window

Reduces T_aug. Loses memory signal from earlier timesteps. Semantically changes Titans (no longer "long-term"). High risk for marginal benefit.

### G. Custom VJP that drops carry tape

In principle: write a `jax.custom_vjp` wrapper around the scan that:
- Forward: runs `lax.scan` normally, returns only `(y_t, ...)` (not the final carry).
- Backward: re-runs the entire forward scan with `jax.checkpoint`, computes gradients via standard backward.

This is the "ideal" structural fix (tape → 268 MB, recompute cost = 1 extra forward scan during backward). But ~50 LOC of custom_vjp code, sensitive to JAX version changes, and requires careful testing. **Out of fix scope per task constraints.**

---

## 3. Recommended fix

**Fix A — bfloat16 carry storage with fp32 compute.**

Why over alternatives:
- **B (hidden=16)**: changes Titans capacity, no smoke validation at H=16. A is precision-only, doesn't touch capacity.
- **C**: doesn't reduce memory (math above).
- **D**: insufficient alone.
- **E**: no-op.
- **F**: changes Titans semantics.
- **G**: outside fix scope.

Expected VRAM at b=128 production: **~12 GB** (down from 21.3 GB OOM-request). 12 GB ≤ 18 GB target ✓ (6 GB safety margin on 24 GB device — meets the constraint).

If A alone doesn't drop VRAM enough (e.g. due to scan-backward intermediate that I haven't accounted for): **STOP** and report, do NOT stack with B. The task constraints explicitly forbid Frankenstein patches.

---

## 4. Per-step plan

All smokes on workstation, GPU 0 (`CUDA_VISIBLE_DEVICES=0`, the RTX 3090 per `MEMORY.md`).

### Step 1 — Edit `titans_memory.py` carry-storage casts

Three sites in `titans_memory.py`:

(a) **Carry init** (currently lines 403-410): cast `M_init_b` and `S_init_b` to `jnp.bfloat16`.
(b) **Body read** (currently lines 220-223): after `stop_gradient`, immediately upcast to fp32.
(c) **Body write** (currently lines 248-251): downcast `new_M, new_S` to bf16 before tuple-return.

Add 1-line module docstring update at the top noting the bf16-carry pattern.

**Smoke after step 1** (≤ 60 s): import + `flax.linen.Module.init` only — no training. Verify the module compiles and the init params are returned with expected shapes.

```bash
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python -c "
import jax, jax.numpy as jnp
from mcts_trader.jax_agent.titans_memory import GatedTemporalMemory
m = GatedTemporalMemory(embed_dim=64, memory_depth=2, memory_hidden=32, persistent_tokens=4, gate_init=-3.0)
x = jnp.ones((4, 64, 64))
mingru = jnp.ones((4, 64))
params = m.init(jax.random.PRNGKey(0), x, mingru)
y, diag = m.apply(params, x, mingru)
print('OK', y.shape, y.dtype, 'gate', float(diag['gate_mean']), 'mem_norm', float(diag['mem_norm']))
"
```

Expected: `OK (4, 64) float32 gate 0.047... mem_norm ~0.0...`. No exceptions.

### Step 2 — Smoke verifies original validation criteria (gate engagement, mem_norm growth, finite grads)

Reproduce the previously-passing 2026-05-16 smoke pattern at small batch first, to verify the FIX itself didn't break Titans:

```bash
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 python train_phase2.py \
  --stage B --epochs 1 --steps-per-epoch 5 --batch-size 64 \
  --periods-file config/foxglove_periods.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --titans-memory --titans-variant mag --titans-memory-depth 2 \
  --titans-memory-hidden 32 --titans-persistent-tokens 4 \
  --titans-gate-init -3 --titans-lr 3e-3 --reset-titans-memory \
  --output-dir runs/000_smoke/titans_oomfix_b64 \
  --name 000_titans_oomfix_b64 2>&1 | tee runs/000_smoke/titans_oomfix_b64.log
```

Time budget: ≤ 5 min. Pass criteria (must all hold):
- `total_loss` finite at all 5 steps.
- `titans_mem_norm` recorded; grows non-trivially over 5 steps (matches 2026-05-16 smoke, growth from ~0.05 toward 0.2+ over 5 steps).
- `titans_gate_mean` measurable, sigmoid(-3)≈0.047 at init.
- `grad_norm` finite, in roughly the 1-100 range (matches 2026-05-16 smoke at lr3e3).

If gate engagement / mem_norm growth regress (e.g. mem_norm stays flat at ~0): bf16 underflow in S accumulation. STOP and report; do not proceed to step 3.

### Step 3 — Production-batch smoke (the OOM-reproducer)

This is the load-bearing smoke — exact production config that OOM'd:

```bash
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 python train_phase2.py \
  --stage B --epochs 1 --steps-per-epoch 5 --batch-size 128 \
  --periods-file config/foxglove_periods.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --titans-memory --titans-variant mag --titans-memory-depth 2 \
  --titans-memory-hidden 32 --titans-persistent-tokens 4 \
  --titans-gate-init -3 --titans-lr 3e-3 --reset-titans-memory \
  --output-dir runs/000_smoke/titans_oomfix_b128 \
  --name 000_titans_oomfix_b128 2>&1 | tee runs/000_smoke/titans_oomfix_b128.log
```

Time budget: ≤ 8 min total (XLA compile + 5 steps).

Pass criteria (must ALL hold):
- No OOM. `nvidia-smi --query-gpu=memory.used --format=csv` reading at step 3 shows ≤ 18 GB.
- All 5 steps complete with finite total_loss.
- `titans_mem_norm` grows over 5 steps (not flat).
- `titans_gate_mean` finite, in `[0.01, 0.5]` (init was sigmoid(-3)=0.047).
- `grad_norm` finite at all steps.

If OOM still occurs: STOP. Report (do NOT stack with fix B). The carry-tape model may be missing a bigger allocation that I haven't accounted for.

If smoke passes but Titans signal is regressed (mem_norm flat or grad explosion): STOP. Report; the fix is not viable as-is. (Fallback options to be discussed: stay with H=32 + bf16 + carry rescaling, OR pivot to H=16 fp32.)

### Step 4 — Update `runs/749_titans_mag/result.md`

Append the "## 749 OOM fix applied (2026-05-18)" section per the task instructions, with the measured VRAM at b=128, smoke validation outputs, and a one-line conclusion.

### Step 5 — Commit

Commit message: `TASK_749 OOM fix complete — Titans @ b=128 fits 24 GB`. Co-Authored-By trailer.

---

## 5. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | bf16 underflow in S accumulation kills mem_norm growth | Med | High (fix not viable) | Step 2 smoke catches this before step 3 |
| R2 | bf16 underflow in grad_M (clipped at 1.0, smallest grads ~1e-4) | Low | Med | Grad clipping at 1.0 caps the dynamic range; bf16 ULP at norm=1 is ~7.8e-3 — coarse but should still register |
| R3 | Stop_gradient on a typed-down (bf16) carry breaks reverse-mode JAX | Low | Med (recompile fail) | Step 1 smoke catches at init-time |
| R4 | Some OTHER allocation beyond the carry tape is the actual bottleneck | Med | High (fix doesn't fix OOM) | Step 3 smoke catches; if so STOP per task spec |
| R5 | The smoke passes but downstream evals regress on bf16 Titans | Low | Med | Out of fix scope — Titans verdict is already "deprioritized". The fix is to enable trainability, not to claim quality. |

The "smoke must mirror production batch shape" rule from `runs/749_titans_mag/result.md:58` is honored — step 3 uses `--batch-size 128 --n-slots 4` matching the OOM repro.

### Failure modes that map to STOP-and-report (per task constraints)

- **OOM still occurs at step 3** → the carry-tape model was incomplete. Don't try fix B on top; the constraint surface is unknown and the user needs to know.
- **Step 2 smoke shows mem_norm flat / grad inf** → bf16 underflow is killing the signal. Don't compensate with extra clipping or other patches; the precision-throughput tradeoff is the constraint.
- **Step 3 OOM is borderline (e.g. fits at one b=128 step but spikes at step 5)** → don't ship; report as NEEDS-MORE-WORK with measurement.

### Failure modes that are acceptable (continue)

- **Step 3 succeeds but wall-time per step is 30-40% higher than baseline at b=128** → expected, no extra forward (bf16 is fast on Ampere); but cast-back-and-forth has tiny overhead. Document.
- **`titans_gate_mean` Δ over 5 steps is smaller than the 2026-05-16 smoke (which ran 10 epochs)** → 5 steps is fewer iterations; the original smoke's gate Δ +3.4e-3 was over 10 epochs ≈ 200 steps. We expect proportionally smaller Δ at 5 steps. Pass criterion is "measurable change" not "matches absolute number".

---

## 6. Scope discipline

Per task constraints:
- No refactoring outside the bf16-carry edit.
- No renaming.
- No "while I'm here" cleanup.
- The fix touches `titans_memory.py` only.
- Commit each step with `TASK_749 OOM fix step N/M:` prefix.

If the recommended fix doesn't work or breaks the smoke: STOP, report, do NOT layer in additional candidate fixes (B, D, etc.) without re-checkpointing the constraint surface with the user.
