# TASK_749 — Titans Pivot (Neural Long-Term Memory for Poppy)

**Status**: DRAFTED — research-and-design pass complete 2026-05-15.
**Source**: Titans paper (Behrouz/Zhong/Mirrokni, Google, Jan 2025).
  - Local LaTeX: `/home/lukas/Trader/docs/arXiv-2501.00663v1-titans/`
  - arXiv: https://arxiv.org/abs/2501.00663
**Motivation**: 734_flow_matching is the first positive Tier-2 result on Poppy (mean Δα +3.38pp, all 6 periods positive). It is a *planning-side* mechanism. Titans is another planning-side family — a small neural memory module whose weights update at inference time. Worth scoping for Poppy.

**Recommended variant: MAG (Memory as a Gate)** acting on the **temporal axis inside the encoder**, gated against the existing minGRU/cross-attention path. Rationale in §3 below.

**Confidence**: medium-low. Titans assumes a transformer host (causal SWA / full attention along a token axis). Poppy has no per-token attention. The plug-in we recommend uses only the Titans *memory module* (Eq. 3-4 below), discarding the SWA branch that the paper's MAC/MAG/MAL diagrams pair it with. This is honest about the gap.

---

## Goal of this task

Investigate whether Titans-style neural long-term memory (MAC / MAG / MAL variants) can plug into Poppy's TRM+MuZero architecture, and produce a concrete implementation spec that we can deploy as an experiment SPEC.

---

## 1. Paper summary

**Titans (Behrouz et al., 2025)** add a *neural long-term memory module* $\mathcal{M}$ — a small MLP whose own weights are updated at inference time by gradient descent on an *associative-memory loss*. Given keys $\mathbf{k}_t = x_t W_K$ and values $\mathbf{v}_t = x_t W_V$ projected from each token $x_t$:

$$
\ell(\mathcal{M}_{t-1}; x_t) \;=\; \big\| \mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t \big\|_2^2 \tag{Eq. 1}
$$

the memory weights are updated via **surprise-momentum gradient descent with weight decay** (paper eq. 13):

$$
S_t = \eta_t\, S_{t-1} \;-\; \theta_t\, \nabla \ell(\mathcal{M}_{t-1}; x_t) \tag{Eq. 2}
$$

$$
\mathcal{M}_t = (1 - \alpha_t)\, \mathcal{M}_{t-1} \;+\; S_t \tag{Eq. 3}
$$

with three *data-dependent* scalars produced by linear projections of $x_t$:

- $\theta_t \in \mathbb{R}_+$ — momentary surprise step size (effective learning rate per token).
- $\eta_t \in [0,1]$ — surprise momentum decay (how much past surprise to retain).
- $\alpha_t \in [0,1]$ — memory weight-decay / forgetting gate.

Retrieval is a plain forward pass without update: $y_t = \mathcal{M}^*(\mathbf{q}_t)$ where $\mathbf{q}_t = x_t W_Q$.

The paper notes (App. MAL) that Eq. 3 with $\mathcal{M}_t = W_t$ linear and $\eta_t=0$ degenerates to **DeltaNet**; with $\eta_t \ne 0$ and weight decay it generalises **Gated DeltaNet**; with a *non-linear* (MLP) memory it strictly exceeds linear-attention / SSM expressivity and is closely related to **TTT layers** (Sun et al., 2024).

Three architectural variants tie the memory into a host transformer:

- **MAC — Memory as a Context**. Chunk sequence into segments $S^{(t)}$. Query memory with $\mathbf{q}_t = S^{(t)} W_Q$, get historical context $h_t = \mathcal{M}^*_{t-1}(\mathbf{q}_t)$, *prepend* $h_t$ + persistent tokens to $S^{(t)}$, run full causal attention, then update memory with the output. Strength: attention "decides" when memory matters; best on long-range NIAH (paper §3.1, Fig. MAC-attention).
- **MAG — Memory as a Gate**. No segmentation. Run SWA branch and memory branch in parallel on the same input; combine with a learnable scalar/vector sigmoid gate: $o = y \otimes \mathcal{M}(\tilde{x})$ (paper §3.2). Strength: simpler interface, doesn't need to interleave with attention masks; performs close to MAC on most tasks (paper Table ablation).
- **MAL — Memory as a Layer**. Stack memory before SWA, sequentially. Closest to existing hybrid SSM+attention literature (H3, Samba, etc.). The paper itself notes this is weaker than MAC/MAG because the layers can't compensate for each other (paper §3.3).

Surprise momentum (Eq. 2) and forgetting gate (Eq. 3) are jointly responsible for ~70% of the headroom over TTT in the ablation (paper Table tab-ablation: removing weight decay and removing momentum are the two largest drops). Persistent memory (learnable input-independent tokens, paper §2.3) is a smaller but consistent contributor.

## 2. Architecture mapping

Where does $\mathcal{M}$ plug into Poppy? Three candidates, ranked by promise:

**Candidate A — Inside the V3 encoder, on the temporal axis (chosen).** Poppy's encoder receives `(B, C=39, T=64, A=64)` and currently compresses the time axis via `_mingru_scan` in `mcts_trader/jax_agent/trm_encoder_v3.py:209-241`. minGRU is a *linear, time-invariant* recurrence (gate + candidate are input-only). Titans memory is the higher-expressivity replacement / sibling: data-dependent $\eta_t, \theta_t, \alpha_t$, MLP memory, momentum. Per-asset (B*A streams) temporal sequence of length T=64 — short enough that a depth-1 MLP memory fits trivially, long enough that surprise-momentum is meaningful. **Tokens here are per-asset per-timestep feature vectors of size embed_dim=64.** Cross-asset attention runs *after*, unchanged.

**Candidate B — Between encoder and MCTS, across the planning unroll.** Run $\mathcal{M}$ over the latent trajectory $(z_0, z_1, \dots, z_K)$ produced by `dynamics()` (`muzero_model.py:501-541`) during MCTS unroll. Conceptually attractive (memory of the planning path) but the unroll is at most ~5-10 steps and is *non-sequential* (it's a tree, not a chain). Each leaf has a different history. JAX-side this requires running the memory update inside `mctx_search.py`'s rollout, which doesn't currently materialise trajectories. **Defer** — too much surgery for a first probe.

**Candidate C — Inside the dynamics MLP, as a sequence of latents across consecutive training windows.** Treat the latent stream $(z_{t=0}, z_{t=1}, \dots)$ over the *full 400-step episode* as the Titans token stream. Memory persists across MCTS root steps inside an episode. This is the cleanest match to "long-term memory" but requires changing the training-step contract (currently the model never sees more than one window pair `(obs_t, obs_t1)` at a time). **Defer** — code surgery is larger than expected from a first probe.

**Chosen mapping (A)**:

| Titans variable | Poppy variable | Shape |
|---|---|---|
| $x_t$ — input token | Encoder per-timestep, per-asset projection `temporal_proj(x)` output | `(B*A, T=64, D=embed_dim=64)` |
| $\mathbf{k}_t, \mathbf{v}_t, \mathbf{q}_t$ | Three new Dense(D→D) projections inside `TitansMemoryBlock` | `(B*A, T, D)` |
| $\mathcal{M}$ — long-term memory | 2-layer MLP (`L_M = 2`, hidden=D, output=D), parameters carried in `state` of `lax.scan` | params shape `((D,D),(D,D))` per (B*A) stream |
| $\eta_t, \theta_t, \alpha_t$ | Three Dense(D→1) heads with sigmoid; same shared trunk | scalars per `(B*A, T)` |
| memory output $y_t = \mathcal{M}^*(\mathbf{q}_t)$ | feed to gate combiner with minGRU output | `(B*A, T, D)` |
| persistent memory tokens $P$ | learnable param `(N_p, D)` with `N_p=4` initially | `(4, D)` |

**Insertion point**: replace `_mingru_scan(z, h_tilde)` in `trm_encoder_v3.py:116` with a gated combiner that takes both `mingru_out` and `titans_out`, controlled by a learnable scalar gate. Zero-init the gate so initial behaviour is byte-identical to the V3 encoder (allowing existing Phase 1 ckpts to load).

## 3. Variant choice — MAG (Memory as a Gate)

We pick **MAG** for the first implementation.

**Why not MAC**: MAC's distinguishing feature is *prepending* the retrieved memory $h_t$ to the segment, then running *full causal attention* over the augmented sequence. Poppy has no per-token causal attention — the only attention is cross-*asset*, which by design ignores time. Implementing MAC honestly would require adding a temporal-attention module that doesn't currently exist; doing so would conflate the "did Titans help?" question with the "did we just add a transformer?" question, and we'd never know which moved the needle. The paper itself notes (Experiments §3) that MAC's win over MAG is concentrated in *long-range* needle-in-haystack tasks (16k+ tokens). Poppy's effective temporal context is 64 timesteps — nowhere near the regime where MAC's segment-then-attend pays off.

**Why not MAL**: MAL stacks memory before attention sequentially. Same transformer-host requirement as MAC. The paper's own §4.1 ablation finds MAL is the weakest variant *even when the rest of the stack is held fixed*; it underperforms MAC and MAG. No reason to pick the paper's worst variant for a stack that isn't even its native habitat.

**Why MAG**: MAG combines memory and "short-term" (here: minGRU) branches by a learnable gate, with no segmentation and no requirement that the short-term branch be attention. We substitute minGRU for SWA — the gate-merge interface is unchanged. This isolates the Titans-memory contribution and lets us answer the clean question: *does data-dependent surprise-momentum-decay memory help where a fixed minGRU does not?*

**Tier-2 follow-up if MAG works**: implement a stripped MAC variant where the "attention" surface is the *existing* cross-asset attention augmented with persistent memory tokens prepended along the asset axis — i.e., add $N_p$ extra "virtual assets" carrying retrieved-memory features. This keeps the chosen surface (cross-asset attention) and adds MAC-style historical conditioning. Document this as TASK_749b.

## 4. Concrete code-change plan

### New module

Create **`mcts_trader/jax_agent/titans_memory.py`** containing:

- `class TitansMemoryBlock(nn.Module)` — the data-dependent K/V/Q projections, persistent memory `P`, gate weights, and the lax.scan-based associative-memory update.
- `class GatedTemporalMemory(nn.Module)` — wraps `TitansMemoryBlock` and the existing minGRU output, computes `o = sigmoid(g) * mingru_out + (1 - sigmoid(g)) * memory_out` with zero-init on `g`.
- Standalone `_titans_scan_step(state, x_t)` helper for `lax.scan`.

Estimated size: ~250 LOC including docstrings.

### File-line edits

| File | Lines | Change |
|---|---|---|
| `mcts_trader/jax_agent/titans_memory.py` | NEW | Implement `TitansMemoryBlock`, `GatedTemporalMemory`, `_titans_scan_step` |
| `mcts_trader/jax_agent/trm_encoder_v3.py:51-59` | edit | Add fields `use_titans_memory: bool = False`, `titans_memory_depth: int = 2`, `titans_persistent_tokens: int = 4` |
| `mcts_trader/jax_agent/trm_encoder_v3.py:61-83` | edit | In `setup()`, instantiate `self.titans = GatedTemporalMemory(...)` only when `use_titans_memory` |
| `mcts_trader/jax_agent/trm_encoder_v3.py:108-117` | edit | After `temporal_proj` (line 106) and before `_mingru_scan` (line 116), call `self.titans` on `(B*A, T, D)` if enabled. Gate-combine outputs. Zero-init gate so MAG output equals minGRU output at step 0 (checkpoint-safe). |
| `mcts_trader/jax_agent/trm_encoder_v3.py:128-166` | edit | Mirror change in `step()` for streaming inference: maintain `M` state across calls via the carry argument. |
| `mcts_trader/jax_agent/trm_flax.py:240-265` | edit | Pipe four new fields through `TinyRecursiveTradingModel` to the V3 encoder constructor: `use_titans_memory`, `titans_memory_depth`, `titans_persistent_tokens`, `titans_surprise_momentum_init`. |
| `mcts_trader/jax_agent/muzero_model.py:240-285` | edit | Add four new fields to `MuZeroNetworks`; pass through to `TinyRecursiveTradingModel` constructor in `setup()` around line 297-309. |
| `phase2_config.py:175-176` | edit | Add Titans config fields (see CLI table below) at the end of the dataclass, before `phase2_checkpoint_dir`. |
| `train_phase2.py:362-400` | edit | In `create_model()`, pull Titans fields from config and pass to `MuZeroNetworks(...)` constructor. |
| `train_phase2.py:1768-1771` | edit | Add Titans CLI flags next to flow-matching flags (kept together as planning-side mechanisms). |
| `train_phase2.py:1934-1938` | edit | In the args→config translation block, wire Titans CLI overrides. |
| `train_phase2.py:1559-1568` | edit | Add `titans_info` string to E-line log when `use_titans_memory` is True (memory norm, mean surprise, mean gate value). |
| `train_phase2.py:1571-1601` | edit | Add five new fields to CSV writer: `titans_mem_norm`, `titans_surprise_mean`, `titans_alpha_mean`, `titans_eta_mean`, `titans_gate_mean`. |

### New CLI flags

| Flag | Default | Effect |
|---|---|---|
| `--titans-memory` | `False` (store_true) | Master toggle. Off → zero-init gate, encoder behaves byte-identical to V3. |
| `--titans-variant {mag}` | `mag` | Only `mag` implemented in this task. `mac`, `mal` reserved (NotImplementedError). |
| `--titans-memory-depth` | `2` (`L_M`) | MLP depth. Paper §2.2 Fig. effect-deep-memory: 2 ≫ 1 on long sequences, 3-4 plateaus. |
| `--titans-memory-hidden` | `None` → `embed_dim` (64) | Hidden width of memory MLP. |
| `--titans-persistent-tokens` | `4` (`N_p`) | Persistent memory tokens prepended pre-scan. Paper §2.3. |
| `--titans-surprise-momentum-init` | `0.9` | Init bias for $\eta_t$ sigmoid → softplus saturation toward retaining past surprise. |
| `--titans-decay-init` | `0.01` | Init bias for $\alpha_t$ sigmoid → near-zero forgetting at step 0 (memory persists). |
| `--titans-lr-init` | `0.01` | Init bias for $\theta_t$ sigmoid → small inner-loop step. |
| `--reset-titans-memory` | `False` | Drop `titans` subtree from checkpoint at load — for re-init or first runs on legacy ckpts. |

All scalars in the last three flags are *initial biases* on the data-dependent sigmoid heads, not fixed values. Data-dependence is preserved per paper §2.1.

### Loss-side changes

**None at first.** MAG plugs into the encoder's forward pass; the existing Stage-B loss stack (policy + value + consistency + reward + BC + dir + rank + aux_price) supervises end-to-end *through the outer loop*. The **inner-loop** memory update happens inside `lax.scan`; the autodiff graph propagates outer-loop gradients through every $\mathcal{M}_t$ (via Eq. 3 — $\mathcal{M}_t$ is a differentiable function of $\mathcal{M}_{t-1}$, $\nabla\ell$, and the data-dependent scalars). This is exactly the TTT-style training pattern — the inner loop is differentiable, so the outer-loop loss reaches the K, V, Q projections and the gate via standard backprop. No auxiliary memorisation loss is required *for training*. (We optionally log the *inner* loss $\ell(\mathcal{M}_{t-1}; x_t)$ as a diagnostic — see §5.)

**Caveat**: the gradient through Eq. 2-3 is second-order (it differentiates through a gradient). JAX's `jax.grad` supports this natively (it's just nested autodiff), but it doubles the backward FLOPs through the scan body. We estimate memory and time impact in §5.

### Initialization (checkpoint compatibility)

The `--titans-memory` flag adds three new parameter subtrees in the encoder:
- `titans.K`, `titans.V`, `titans.Q` (Dense layers)
- `titans.theta_head`, `titans.eta_head`, `titans.alpha_head` (Dense → scalar)
- `titans.persistent` (parameter `(N_p, D)`)
- `titans.M_init` (parameter — initial memory MLP weights, $L_M \cdot D^2$ params)
- `titans.gate` (single scalar param, **zero-init**)

The gate is zero-init → `sigmoid(0)=0.5` would mix half-and-half. We want **byte-identical** with V3 baseline at step 0, so the gate combiner uses `mix = sigmoid(g)` with $g$ **initialised to large negative** (e.g. -10), making `sigmoid(g) ≈ 0` and the memory branch contributes nothing until gradient pulls it in. This mirrors the zero-init pattern used in `FiLMProjector` (TASK_741) and `PolicyDirectionRefiner` (TASK_728).

Standard Phase 1 checkpoints lack the new subtrees → load with `--reset-titans-memory` once on the first run; subsequent ckpts (Phase 2 Titans runs) carry them forward.

## 5. JAX/Flax implementation notes

### The test-time-update problem (the core technical challenge)

Titans' "memory updates at inference time" is **not actually an inference-only operation** in our setup — the memory updates happen inside the *forward pass*, and we want gradients to flow through them. The right JAX pattern is:

**Recommended approach: `lax.scan` carrying $\mathcal{M}$ as state, autodiff through the body.**

```python
def titans_scan_body(carry, x_t):
    M_t_minus_1, S_t_minus_1 = carry
    # 1. Compute data-dependent scalars
    theta_t = jax.nn.sigmoid(theta_head(x_t) + theta_init_bias)   # (B*A,)
    eta_t   = jax.nn.sigmoid(eta_head(x_t)   + eta_init_bias)
    alpha_t = jax.nn.sigmoid(alpha_head(x_t) + alpha_init_bias)

    # 2. Compute keys/values
    k_t = x_t @ W_K
    v_t = x_t @ W_V

    # 3. Surprise = grad of Eq. 1 wrt M
    def inner_loss(M_params, k, v):
        return jnp.mean((apply_mlp(M_params, k) - v) ** 2)
    grad_M = jax.grad(inner_loss, argnums=0)(M_t_minus_1, k_t, v_t)

    # 4. Momentum + decay update (Eqs. 2-3)
    S_t = eta_t[..., None, None] * S_t_minus_1 - theta_t[..., None, None] * grad_M
    M_t = (1.0 - alpha_t)[..., None, None] * M_t_minus_1 + S_t

    # 5. Retrieve
    q_t = x_t @ W_Q
    y_t = apply_mlp(M_t, q_t)

    return (M_t, S_t), y_t

(M_final, _), y = jax.lax.scan(titans_scan_body, (M_init, S_init), x_seq)
```

**Why this works**:
- `lax.scan` is fully JIT-compilable and `vmap`-compatible (we'll vmap over the `(B*A)` axis).
- `jax.grad` inside the scan body computes the *inner-loop* gradient of $\ell$ wrt $\mathcal{M}$. This is **first-order** wrt the inner loop and **second-order** wrt the outer loop — handled natively by JAX's autodiff (`jax.grad(jax.grad(f))` works).
- The carry $(\mathcal{M}_t, S_t)$ is differentiable through `scan`; outer-loop loss reaches every parameter through the scan's gradient computation.
- No `custom_vjp` required, no manual VJP, no persistent state hacks.

**Pitfalls (named explicitly)**:

1. **JAX pitfall list from `CLAUDE.md`**:
   - *No `jax.lax.cond` inside `vmap`* — we use `jnp.where` for any branches inside the scan body (e.g. clipping $\alpha_t$ away from exact 1.0 to keep memory non-degenerate).
   - *`jnp.where` evaluates both branches* — fine here, no conditionally-expensive paths.
   - *Guard division* — when computing the gradient of MSE with respect to $\mathcal{M}$, we apply `jnp.maximum(grad_norm, 1e-8)` before any normalisation (only if we add the optional gradient-norm clipping).
   - *Use `static_argnames` for configs in JIT* — `L_M`, `D`, `N_p` are baked at construction time, not passed as scan kwargs.
   - *Dynamic shapes in `lax.scan` → pad to static `max_len`* — our `T=64` is static, so this is a non-issue.

2. **Second-order autodiff cost**: `jax.grad` inside `scan` doubles the backward FLOPs through the scan body (the outer backward differentiates through the inner forward gradient computation). We expect a ~1.4–1.6× slowdown on the encoder forward+backward vs the V3-minGRU baseline. The full Phase-2 step (which is dominated by MCTS sims + policy/value compute) should slow by ~10–20%. Concrete budget: if a baseline epoch is 45 s/epoch (from 734's logs), Titans-MAG should land around 55 s/epoch. 200 epochs ⇒ ~3 h vs baseline's ~2.5 h. Acceptable.

3. **`Module` capture inside `scan` body**: Flax `nn.Module` calls cannot live inside `lax.scan` directly without `nn.scan` or hoisting parameters. We use the latter: extract `W_K`, `W_V`, `W_Q`, and the inner-MLP params via `self.param(...)` calls in the outer `__call__`, then pass them as scan constants. This keeps the scan body a pure JAX function and is the same pattern already used by `_mingru_scan` in `trm_encoder_v3.py:209`.

4. **Inner-MLP "apply" function**: implement as a plain `jnp.dot`-and-`gelu` chain (no Flax module call inside scan). For `L_M=2` with width D=64: ~8200 params per (B*A) stream, but **the params are shared across all streams** (not per-stream) — only the *state* $\mathcal{M}_t$ is per-stream. Wait — re-read paper §2.1: $\mathcal{M}_t$ IS the weights themselves; the per-stream "state" is a per-stream copy of the entire MLP weight tensor. That's expensive (see §5.3 below). Mitigation: keep memory MLP small (D=64, `L_M=2`) → 8200 params/stream × (B*A) = 8200 × (256 × 64) = 134 M float32 = ~540 MB per copy. Multiplied by two (momentum buffer S) and held during backward → 1.5–2 GB peak. Tight but fits in the 24 GB budget alongside the existing ~6 GB Stage-B footprint.

5. **`vmap` axis ordering**: scan must be over the time axis (`T=64`). The encoder reshapes to `(B*A, T, D)` before the minGRU scan; the same reshape applies. Then we `vmap` the scan body over `B*A` (asset×batch streams are independent). Standard pattern.

6. **`bfloat16` interaction**: keep memory MLP params in `param_dtype=float32` even when compute is `bfloat16`. The inner-loop gradient is small in magnitude and prone to underflow in bf16. The existing convention in `muzero_model.py` already separates these.

7. **PopArt / value-target interaction**: none — Titans operates pre-bottleneck, far from the value head.

8. **MCTS interaction**: at MCTS time we call `representation()` → encoder forward → memory state advances along `T=64` once, then is *discarded*. The memory is **not persisted across MCTS calls** in this first iteration. This is the safe path: each call gets a fresh $\mathcal{M}_0$ (the learnable init). Persistent cross-call memory is Candidate C (deferred).

### Memory footprint estimate

Concrete numbers using actual Poppy dimensions:

- `B = 256` (Stage-B batch), `A = 64` (assets), `T = 64` (window), `D = 64` (embed_dim).
- `B*A = 16384` parallel streams in the scan.
- Memory MLP params per stream: `L_M=2` × `D × D` = `2 × 64 × 64` = **8 192 params**.
- Per-stream state tensor $\mathcal{M}_t$: 8 192 × 4 bytes = **32 KB**.
- Per-stream momentum buffer $S_t$: same shape = **32 KB**.
- Per-step carry: 64 KB × 16 384 streams = **1.05 GB**.
- Held *across all T scan steps* during backward (since we differentiate through every step): $T \times 1.05$ GB = **67 GB** ← too large with naive checkpointing!

**Mitigation**: use `jax.checkpoint` (`remat`) on the scan body. Re-materialise inner-loop computation during backward; only the carry at each step is retained. With `remat`, peak ≈ 1.5 × per-step carry ≈ **1.6 GB**. This is the standard pattern for differentiable scans and is documented in JAX's `lax.scan` examples.

If even 1.6 GB is too tight when combined with the ~18 GB Stage-B working set, the fallback is to keep `embed_dim=32` (pre-TASK_683 width) only inside the memory MLP — memory becomes 1/4 the size, total budget halves. Document this as the escape hatch in the SPEC.

### Smoke-test plan (mirrors 734)

Single workstation GPU, ~10 epochs, batch=64 (small):

```bash
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 10 --steps-per-epoch 50 --batch-size 64 \
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
  --titans-persistent-tokens 4 --reset-titans-memory \
  --output-dir runs/000_smoke/titans_smoke \
  --name 000_titans_smoke 2>&1 | tee runs/000_smoke/titans_smoke/train.log
```

**Pass criteria** (must all hold):
1. Compile completes in ≤ 8 minutes (60% slower than V3 baseline ≈ 5 min — generous because of second-order autodiff).
2. `total_loss` finite for every epoch (no NaN).
3. `titans_mem_norm` is recorded and stays in `[0.01, 10.0]`. Drift past either bound = unstable.
4. `titans_gate_mean` (sigmoid of the mix scalar) drifts measurably away from its initial 0.0 by E10. If it stays pinned at ~0, the memory branch is dead and we have an init/grad-path bug.
5. `consistency_loss` magnitude within 5× of the baseline at E10 (we're not training to convergence, just verifying no catastrophic regression).
6. Peak VRAM ≤ 18 GB on the workstation 3090 with batch=64.
7. Wall time ≤ 30 minutes.

If 1, 2, 5, or 6 fail → debug. If only 3 or 4 fail with everything else healthy → likely a hyperparameter issue (init biases), document and adjust before promoting to the SPEC run.

## 6. Smoke-test plan

See §5 ("Smoke-test plan") above for the full command. Wall ≤ 30 min on workstation, single 3090 (GPU index 1 = the actual RTX 3090 per `MEMORY.md`). Smoke runs land in `runs/000_smoke/titans_smoke/`.

## 7. Experiment SPEC sketch

Lives at `experiments/749_titans_mag/SPEC.md`. Paired 3-seed protocol following the standard Tier-2 pattern (seeds 7, 13, 23) against the **741_mint a1b1 baseline** at the same seeds. Promotion gate: mean(Δα) ≥ +3pp AND all 3 positive, mirroring 734.

**Conditional baseline switch**: if 734-x5 (currently running, seeds 31+42) promotes flow-matching to canonical, swap the baseline to `runs/734_flow_matching/flow_seed_{7,13,23}/` and re-pair. Note the conditional in the SPEC. If 734 promotes, the SPEC should also document a "Titans + flow-matching" stacked variant as a follow-up (no implementation needed in TASK_749 itself; that's Tier-D).

## 8. Open questions / risks

Listed in descending order of severity.

**R1 (MAJOR) — Titans is fundamentally a transformer companion; the cleanest plug-in slot in Poppy doesn't exist.** The paper's MAC/MAG/MAL all pair the memory with a *causal attention surface along the token axis*. Poppy has no such surface (cross-asset attention is along assets, not time). We are using only the *memory module* from the paper, dropping the SWA/full-attention branch and substituting minGRU as the "short-term memory". This is a deliberate adaptation, not a paper-faithful implementation. **Risk**: the surprise-momentum-decay machinery may need the attention partner to express its full headroom — the paper's ablations don't isolate "memory alone vs memory + something else" cleanly enough to predict. **Mitigation**: this is exactly why we run paired Δα — if MAG underperforms baseline at the 3-seed gate, the verdict is honest. We do NOT claim this is a true Titans implementation; the SPEC labels it "Titans-MAG-style memory".

**R2 (MAJOR) — Second-order autodiff fragility.** `jax.grad` inside `lax.scan` works in principle, but pathological cases (NaN in the inner gradient, numerical degeneracy when $\alpha_t \to 1$, second-order memory blow-up under bf16) are not rare. **Mitigation**: keep `compute_dtype='float32'` for the smoke run; only switch to bf16 after numerical stability is confirmed at full scale. Add `--titans-grad-clip-inner` as an undocumented escape-hatch flag if needed (zero overhead when off).

**R3 (MODERATE) — Per-stream memory state is non-trivial VRAM.** 1.6 GB after `remat` is fine on a 24 GB 3090 but leaves limited headroom. **Mitigation**: §5 documents the `embed_dim`-shrink fallback. Worst case: batch ≤ 128 for Titans runs and document the smaller batch in the SPEC.

**R4 (MODERATE) — Conceptual overlap with flow-matching.** 734's flow-matching is a *velocity prediction in dynamics* (predict $z_{t+1} - z_t$ instead of $z_{t+1}$). Titans is *online associative memory* (learn $\mathbf{k} \to \mathbf{v}$ mapping at inference). **They are different mechanisms** — flow-matching changes the dynamics-loss target; Titans changes the encoder representation. They should be combinable (stack as Tier-D). **However**: if flow-matching's win is really about *temporal smoothness regularisation* (which it might be, since velocity is naturally smoother than position), Titans-MAG may capture some of the same headroom by giving the temporal compression a richer mechanism (data-dependent forgetting + surprise). If 749 wins by +3pp and 734 wins by +3pp, the stacked combination might only deliver +3pp total — they may be competing for the same explainable variance. **Mitigation**: note this in the SPEC's "Notes" section; do not commit to a stacked Tier-D until 749 has a verdict.

**R5 (MINOR) — MIRAS-style fallback.** If §5's pure-functional scan turns out fragile beyond R2's mitigations, the next-best alternative is a *MIRAS-framework-inspired* memory module that just borrows the **surprise update rule (Eq. 2-3)** into the existing TRM dynamics MLP. Concretely: re-interpret the dynamics MLP's weights as $\mathcal{M}_t$, apply Eq. 3 along the MCTS unroll (1-5 steps) instead of along the encoder time axis, and use a single MCTS-step "online loss" $\| \text{dyn}(z_t, a_t) - z_{t+1} \|^2$ as $\ell$. Much smaller state (the dynamics MLP is ~50k params shared globally, not per-stream), no per-asset blowup, no encoder surgery. Documented here as **fallback path; not the primary recommendation**.

**R6 (MINOR) — Persistent memory tokens require a Phase-1 retrain to be useful at scale.** Paper §2.3 motivates $P$ as task-knowledge storage. We init it from scratch in Phase 2 with $N_p=4$. If the headroom is mostly in $P$, we won't see it in a 200-epoch Stage-B run. **Mitigation**: low-priority; even paper §4.1 ablation shows $P$ contributes ~10% of the total Titans win, not 90%.

## 9. Estimated effort

- **Code work**: **medium**. New module ~250 LOC. Encoder edit ~50 LOC. Config/CLI plumbing ~80 LOC. Total ~380 LOC across 7 files. The `lax.scan` body with second-order autodiff is the trickiest part; everything else is mechanical (the model already supports zero-init gates, optional subtree loading, multi-flag CLI plumbing — TASK_741 was a good template).
- **Smoke-test debugging**: **2–8 hours**. Most likely failure modes are (a) NaN in inner gradient on the first batch (fix: gradient clipping inside scan), (b) memory blowup from missing `remat` (fix: add `jax.checkpoint`), (c) the gate stays pinned at 0 (fix: gate init bias, or add a small auxiliary regulariser that encourages `gate ≠ 0` for the first 50 epochs). Each fix is ~30 min once diagnosed.
- **First experiment wall**: matching Tier-2 estimates — 3 paired seeds × ~3 h per run × 2-wide GPU packing = **~4.5 h wall** (slightly more than 734's 3.75 h because of the ~10–20% per-step slowdown).
- **End-to-end from now to verdict**: ~1 day of focused implementation + smoke, then queue the SPEC for an overnight 3-seed run + paired eval.

---

## Constraints (load-bearing for the agent)

- **JAX/Flax** required (current stack). No PyTorch port.
- Must fit a single RTX 3090 (24 GB) alongside existing model + MCTS rollouts.
- Don't break existing Phase-1 checkpoint loading (`jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`). Memory module must be optionally toggleable via flag, with gate large-negative init so old ckpts load and baseline behaviour reproduces exactly.
- Stay within the **planning-side mechanism class** — the memory module changes how the *encoder compresses time*, not how a "better classifier" is built. The paper's framing of Titans is unambiguously planning-side (memory for long-context reasoning), so this is honest.
- Respect the TASK_742 rule (≤2 concurrent train_phase2 per host) and the GPU-0-forbidden rule (server-side).

## Out of scope (not in this task)

- Full Bella retrain to use Titans memory inside the generator. That's Tier-D strategic work; the first experiment keeps Bella fixed.
- Multi-mechanism stacks (Titans + flow_matching + oracle). After-individual-results work.
- A full PyTorch reference port. JAX-native.
- MAC and MAL variants — reserved for TASK_749b/c.
- Candidate B (memory across MCTS rollout) and Candidate C (memory across episode-wide latents) — both deferred to follow-up tasks.

---

## Operator notes

- Paper LaTeX is at `/home/lukas/Trader/docs/arXiv-2501.00663v1-titans/main.tex` plus `MainText/`, `Appendix/`, `Figures/`, `Tables/`. Read `MainText/Methods.tex` (esp. §2.1 and §3.2 for MAG); skim `Appendix/MAL.tex` for the DeltaNet/TTT family relationships if you want to understand the lineage.
- Poppy entry: `train_jax_fast.py`, encoder under `mcts_trader/`. Phase 2 entry: `train_phase2.py`. Foxglove entry: `train_easy.py`.
- TRM (Tiny Recursive Model): minGRU temporal compression (V3) + cross-asset attention + MLP reasoning loops + 2-layer MLP dynamics. NOT a transformer — load-bearing distinction; see §3 for why this constrains the variant choice.
- Subtree CLAUDE.md at `mcts_trader/CLAUDE.md` documents Poppy architecture details. The "do not suggest transformer/attention changes to reasoning or dynamics" rule is respected here — Titans memory plugs into the encoder's *temporal* compression stage and the cross-asset attention is unchanged.
- SPEC sketch: `/home/lukas/Trader/experiments/749_titans_mag/SPEC.md`.
