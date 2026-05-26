# TASK_778: Bella-Conditioned MCTS Rollouts (Revival of TASK_722)

**Status**: DRAFT — design only, awaiting code-budget authorization. No implementation this session.
**Created**: 2026-05-25
**Refs**: TASK_722 (parked Feb-2026 design, fully spec'd); ship-path recommendations §4 (`lore/ship_path_recommendations_2026-05-25.md`); `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md`
**Class**: planning + closed-loop (touches dynamics + MCTS search; does NOT touch policy classifier — see anti-correlation check below).
**Multi-week code budget**: ~3-5 days build + 2-gate review + smoke + paired 3-seed eval. No commit until reviewer authorization.

---

## 1. Hypothesis

The dynamics MLP (`mcts_trader/jax_agent/muzero_model.py:380-388` definition / `:594-598` forward / `:701-728` `recurrent_inference`) is the **search bottleneck** on the current stack:

- 3.3M-param 2-layer MLP (`dynamics_dense1` → LayerNorm → GELU → `dynamics_dense2`) tries to compress full multi-asset latent evolution into a single one-step prediction.
- Consistency loss stuck at ~1.34 across 20k+ Stage-B epochs (per TASK_722 §Motivation; observed across the entire 738-769 campaign).
- Inaccurate dynamics ⇒ MCTS expansions "dream" of latents the value/policy heads were never trained on.

**Proposed swap**: replace the dynamics call in MCTS expansion with Bella flow-matching rollouts.

- Bella generates K plausible OHLCV futures conditioned on the current observed window.
- Encoder re-encodes each future to recover **real** observation-grade latents (the same encoder the value/policy heads were trained against).
- Dynamics MLP is repurposed as a **consistency filter** — its prediction at `(latent_t, action)` is compared against Bella's actual encoded `latent_{t+1}`; a low MSE means "the learned dynamics agrees that Bella's future is reachable," a high MSE means Bella hallucinated something off-manifold. MCTS branch priors are reweighted by consistency.

This is the original TASK_722 ansatz, never built. Per ship-path memo §4: "the *largest unexecuted designed mechanism* in the campaign."

Cite: ship_path_recommendations_2026-05-25.md §4; project_encoder_bella_class_closed.md "How to apply" §1.

---

## 2. Why this re-opens the (otherwise CLOSED) Bella mechanism class

Memory entry `project_encoder_bella_class_closed.md` declares the Bella class closed as of 2026-05-19 based on three failures:

| Attempt | Surface | Paired Δα |
|---|---|---:|
| 740 bella_distill | encoder per-asset MSE | −1.92pp |
| 757 bella_shape | encoder permutation-invariant shape | −5.80pp |
| 758 bella_augment | **training data slot substitution** | **−16.54pp** (s13 = −37.93pp catastrophe) |

All three loaded Bella into the **supervision signal** or the **training distribution** — i.e., the encoder/classifier saw Bella outputs as gradient targets, triggering the `[[classifier-alpha-anticorrelation]]` pathway and (for 758) shifting the train distribution off the deployment regime.

The memory file explicitly carves out an exception (verbatim):

> Don't propose new Bella variants without a fundamentally different ansatz (e.g., Bella as world-model rollouts inside MCTS, not as supervision or data).

TASK_778 is precisely that exception:

- **No new gradient flows into the encoder via Bella.** The encoder runs in inference mode when re-encoding Bella outputs during MCTS rollouts.
- **No Bella samples enter the BC / policy loss as training data.** The policy is still trained on historical-data MCTS targets; Bella only changes how MCTS *explores* during expansion.
- **The "too easy" 7× oracle-PnL-ratio finding** (`lore/notes_2026-05-18.md` §134) is mitigated by the dynamics-MLP consistency filter — if Bella's futures are unrealistically smooth/profitable, the dynamics MLP's MSE against the learned latent manifold should be high, downweighting them in the MCTS prior. This is the failure-mode-aware design.

If the consistency filter does NOT reject the easy Bella futures (i.e., the dynamics MLP itself has learned an over-smooth manifold), that is the informative null — it means the dynamics class IS the bottleneck but the consistency score is not separable enough to gate it.

---

## 3. Verified code state (citations against current tip `79430e3`)

All file:line citations below were verified live during this design pass (2026-05-25).

### 3.1 Bella generator (entry points)

- `diff_teacher/generation/generator.py:62-149` — `MarketGenerator` class + `from_checkpoint` classmethod (loads conv 768d/12L checkpoint per `lore/notes_2026-05-18.md` §134, NOT the 1536d/16L claimed in `diff_teacher/CLAUDE.md`).
- `diff_teacher/generation/generator.py:151-268` — `MarketGenerator.generate(...)` (CPU roundtrip; **broken at `seq_length=1440`** per §134 — `ValueError: jnp.pad unsupported pad_width with nd=4`; workaround = chained `generate_on_device` at native `seq_length=400`).
- `diff_teacher/generation/generator.py:271-321` — `MarketGenerator.generate_on_device(...)` (on-device JAX entry point; JIT-compiled via `_generate_jit`; this is the integration point for `BellaMCTSDynamics`).
- `diff_teacher/skeleton/extraction.py` — `extract_market_skeleton(...)` referenced by TASK_722 §"Skeleton Extraction from Observed Window" as the simplest conditioning path (no new params).

### 3.2 Bella checkpoint deployability

- Per `MEMORY.md` workstation↔server deploy contract: `tools/deploy/README.md` covers Bella ckpt rsync to server (TASK_758 already shipped this for the bella_augment experiment).
- Current ckpt verified working in §134: 768d/12L conv, 10 ODE steps.

### 3.3 Encoder reuse point (for re-encoding Bella futures)

- `mcts_trader/env/jax_encoder.py:1-77` — module docstring + `EncoderConfig` dataclass. 32-channel encoder post-TASK_732 (`n_channels=32`, `window_size=60`, `n_assets=64`).
- `mcts_trader/env/jax_encoder.py:480-540` — the 27 per-asset-independent channels (broadcast_scalar fill for multi-horizon returns/volume/momentum/volatility/percentiles). This block is the section that must be sliding-window-applied to Bella's `(K, H, n_assets, 5)` outputs to recover encoder-grade `(32, ws, n)` observations.
- The encoder is pure (no Python control flow, vmap-safe per `mcts_trader/CLAUDE.md`), so reuse via `jax.vmap` across `K × T` Bella futures is mechanical.

### 3.4 Dynamics MLP swap point (CORRECTED line citations vs ship-path memo)

The user's task spec cited "muzero_model.py:415-420 dynamics_dense1/2" — VERIFIED INCORRECT at current tip. Lines 415-420 hold the TASK_695 oracle_mask aux head (`oracle_mask_dense1/2`). The actual dynamics MLP is at:

- **Definition**: `mcts_trader/jax_agent/muzero_model.py:380-388` (`dynamics_dense1`, `dynamics_ln`, `dynamics_dense2`, `dynamics_state_norm`). `dynamics_hidden = self.state_dim * 8` (512 for 64-d state, 1024 for 128-d).
- **Forward `dynamics(latent_state, action)`**: `:583-610`. Returns `(next_state, reward)` after LayerNorm + GELU + linear + state-norm.
- **`recurrent_inference(latent_state, action)`**: `:701-728`. Wraps `self.dynamics(...)` and passes the result through the prediction head under `stop_gradient` (BUG-017 fix preserved). **This is the entry point the MCTS search invokes via `mode='recurrent_inference'`.**

### 3.5 MCTS expansion call sites (the swap targets)

- **Factored search** (production canon per `runs/762_reward_shaping/*/stage_b_config.json:147`): `mcts_trader/jax_agent/factored_search.py:80-114` (1-step lookahead per asset, calls `self.model_apply_fn(..., mode='recurrent_inference')` at `:86-91`, sub-batches over 320 actions). Factored is single-step by design — Bella integration here is **one rollout per `(asset, action)` Q evaluation**, not a tree.
- **Joint MCTS** (eval scorecard canon, `benchmark_easy.py --eval-mcts-sims 16 joint`): `mcts_trader/jax_agent/mctx_search.py:89-140` (`_recurrent_fn`, calls `self.model_apply_fn(..., mode='recurrent_inference')` at `:109-114`, runs `num_simulations × depth` per search). Joint is the multi-step tree — Bella integration here uses pre-generated K-future tensor indexed by (branch, step).
- **The "Bella-MCTS" integration replaces these `recurrent_inference` calls** with `BellaMCTSDynamics.expand_node(...)` (see §4.1 below). The call-site signature is preserved so both factored and joint paths can call into the same `BellaMCTSDynamics` object.

### 3.6 1440-pad bug status (known workaround, no new work)

Per `lore/notes_2026-05-18.md` §134: `generator.generate(seq_length=1440)` raises `ValueError: jnp.pad unsupported pad_width with nd=4` in the current ckpt. Workaround = chained `generate_on_device` at native `seq_length=400`. TASK_778's `BellaMCTSDynamics` will:

- Use `generate_on_device(seq_length=400)` directly (JIT-compiled, on-device JAX).
- Need only **H = 20-50 future steps** per rollout, so 400 is over-provisioned (we slice the first H from the generated 400). Pad bug is moot at our needed horizon.

### 3.7 The "7× too-easy" risk surface (mitigation embedded in design)

Per §134, Bella outputs measured at **~7× the historical per-step oracle PnL** during 758 smoke. The TASK_722 design pre-dates this finding. TASK_778 inherits the §138 lesson: "Synthetic data carries hidden 'too easy' risk." Mitigation surfaces in the design (§5 promotion gate + diagnostic).

---

## 4. Code-change scope (estimated ~500-600 LOC, multi-week)

This task **does not commit any code in this session**. The breakdown below is a code-budget proposal for the reviewer.

### 4.1 NEW module: `mcts_trader/jax_agent/bella_dynamics.py` (~500 LOC)

Modelled directly on TASK_722 §"Step 3 Bella-MCTS Integration":

```python
class BellaMCTSDynamics:
    """Bella generates futures, dynamics MLP scores their plausibility."""

    def __init__(self, bella_generator, trm_model_apply_fn,
                 n_rollouts=16, horizon=20, ode_steps=10,
                 consistency_temp=1.0): ...

    def generate_and_score(self, params, current_window, rng_key):
        # 1. Skeleton extraction from observed window (diff_teacher.skeleton.extraction)
        # 2. Bella.generate_on_device(n_samples=K, seq_length=400)[:, :horizon, ...]
        # 3. Encode each Bella future via sliding-window encoder (jax.vmap over K x T)
        # 4. Dynamics MLP predicts latent_{t+1}; compare to encoded latent_{t+1}
        # 5. Per-branch consistency score: exp(-MSE / H) in [0, 1]
        return futures, bella_latents, consistency_scores

    def expand_node(self, params, bella_latents, branch_k, step_t, action):
        # MCTS expansion: index into pre-encoded latents + run prediction head
        # for policy/value at that latent. NO dynamics MLP forward — Bella supplies the latent.
        latent_t1 = bella_latents[branch_k, step_t]
        # Prediction head on real encoded latent
        return policy_logits, value, reward
```

Internal sub-components:
- `_extract_skeleton_from_window(prices_window)` — calls `diff_teacher.skeleton.extraction.extract_market_skeleton` per TASK_722 §Skeleton.
- `_encode_trajectories(params, current_window, futures)` — sliding-window vmap of `mcts_trader/env/jax_encoder.py:encode_observation` across `(K × (H − ws + 1))`. **~50 LOC additive to encoder usage**, NOT a modification of `jax_encoder.py` itself (the encoder is reused as-is).
- `_score_consistency(params, bella_latents)` — per-step dynamics MLP MSE accumulation per TASK_722 §3 pseudocode lines 209-222. **~30 LOC** total. Uses the existing `MuZeroNetworks.dynamics(...)` method (`muzero_model.py:583-610`) as a callable — no new model layers.

### 4.2 `train_phase2.py` integration (~30 LOC)

- New CLI flag `--use-bella-rollouts` (default OFF; canon path unchanged).
- New CLI flags `--bella-rollout-k INT` (default 16), `--bella-ode-steps INT` (default 10), `--bella-horizon INT` (default 20), `--bella-conditioning {skeleton,context}` (default `skeleton`).
- Conditional instantiation of `BellaMCTSDynamics` in the Stage-B setup block, passed into `FactoredPerAssetSearch` or `MuZeroMCTXSearch` via constructor.
- Search-class injection point: the `model_apply_fn` argument these searches receive becomes a thin wrapper that calls `bella_dynamics.expand_node(...)` instead of the raw `params, latent, action -> recurrent_inference` path. **Wrapper is the only place the swap is materialized.** Canon path (flag OFF) preserved bit-identical.

### 4.3 Encoder modification: sliding-window encode of Bella outputs (~50 LOC, ADDITIVE)

**No modification to `jax_encoder.py` itself** — the encoder is pure and vmap-safe. The sliding-window logic lives in `bella_dynamics.py`:

- Given Bella output `(K, H, n_assets, 5)`, slide a `ws=60` window forward over the H axis (yielding `H − ws + 1` observations per branch).
- For each window position, build `(prices, volumes, highs, lows, window_start)` tuple and call existing `encode_observation` (the function at `jax_encoder.py:480-540` block plus the cross-sectional block at `:540+`).
- Output: `(K, H − ws + 1, 32, ws, n)` encoded latents — fed straight into the dynamics MLP for consistency scoring AND the prediction head for value/policy at each branch step.

### 4.4 Consistency-score head (~30 LOC)

**No new params.** The "head" is the existing `dynamics_dense1/2` MLP repurposed: per branch, accumulate MSE between `dynamics(latent_t, oracle_action_t)` and the Bella-encoded `latent_{t+1}` across steps. Average, exponentiate to `[0, 1]`, multiply into MCTS branch prior. The dynamics MLP keeps its consistency-loss gradient signal in the standard Stage-B loss (`train_phase2.py` consistency-loss term untouched) so it improves as a filter via Bella-trajectory consistency targets — exactly the virtuous-cycle pattern in TASK_722 §4.

### 4.5 `Phase2Config` additions (~10 LOC)

```python
use_bella_rollouts: bool = False         # MUST default False (canon bit-identity)
bella_rollout_k: int = 16                # branches per MCTS root
bella_ode_steps: int = 10                # Bella's native config
bella_horizon: int = 20                  # forward steps generated
bella_conditioning: str = "skeleton"     # "skeleton" | "context"
bella_consistency_temp: float = 1.0      # temperature on exp(-MSE/H)
bella_ckpt_path: str = ""                # path to deployed Bella ckpt
```

### 4.6 Total NEW LOC estimate

| Component | LOC |
|---|---:|
| `bella_dynamics.py` new module | ~500 |
| `train_phase2.py` integration | ~30 |
| Sliding-window encoder usage (in bella_dynamics) | ~50 |
| Consistency-score scaffolding | ~30 |
| `Phase2Config` fields + CLI plumbing | ~20 |
| **Total** | **~630 LOC** |

Estimated build time: 3-5 days code on workstation, including unit smoke + bit-identity smoke (flag OFF).

---

## 5. Two-gate review protocol (per `lore/notes_2026-05-18.md` §138 lesson #1)

The §138 lesson: "Two-gate review protocol works." Both 753 and 754 implementations were caught with 7-9 spec gaps per task by review agents, then implemented cleanly. TASK_778 inherits the protocol:

### Gate 1 — Static code review (PRE-MERGE)

- Run `qa-hunt` agent over `bella_dynamics.py` + the `train_phase2.py` diff with the TASK_778 spec as reference. Expected: 5-10 findings re: edge cases, JAX rng handling (`jax.random.fold_in` not `split` per `diff_teacher/CLAUDE.md` HIGH-009), vmap shape contracts, sub-batch sizing, ckpt-loading nested-params handling.
- Run `qa-crossval` agent for cross-module consistency (does `bella_dynamics.py`'s consistency-MSE align with the existing Stage-B consistency-loss accounting in `train_phase2.py`? do the prior weights play correctly with TASK_771/F1+F2 masking when `--disable-limit-orders` is on?).
- Reviewer (operator) approves with mitigations; agent re-implements; second smoke pass.

### Gate 2 — Bit-identical smoke (PRE-PROD)

- Smoke 1: full `stage_b_train_step` at production batch (per §138 lesson #2, the 749 OOM smoke-test rule). Flag `--use-bella-rollouts OFF`. **Must be bit-identical to current canon a1b1 trajectory** (same loss curves, same trade counts, same checkpoints at epoch 5). Any drift = bug in the flag-OFF code path.
- Smoke 2: flag ON, 1 epoch only. Verify (a) no OOM at production batch with Bella loaded (~1.7 GB additional VRAM per TASK_722 §VRAM Budget), (b) consistency score distribution is NOT degenerate (not all 0, not all 1), (c) per-batch wall ≤ 500 ms (TASK_722 estimate).

### Pre-prod operator decision

If both gates clear, authorize the paired 3-seed × 2-wide Stage-B run on server GPU 1/2/3. Until then, do **not** deploy.

---

## 6. Variants (single cell, paired against 760 a1b1 canon)

| cell | flag | k | ode | horizon | conditioning | seeds |
|---|---|---:|---:|---:|---|---|
| 778-bella-k16 | `--use-bella-rollouts --bella-rollout-k 16 --bella-ode-steps 10 --bella-horizon 20 --bella-conditioning skeleton` | 16 | 10 | 20 | skeleton | {7, 13, 23} |

Comparison baseline: 760 a1b1 sims=32 paired (eval-MCTS sims=16 joint per canon scorecard). Use the existing 760 paired baselines — no fresh canon baseline needed.

**Wave 2 (conditional on Wave 1 mechanism gate firing)**:
- Sweep `K ∈ {8, 16, 32}` to find branching sweet spot.
- Sweep `bella_horizon ∈ {10, 20, 40}` to find depth sweet spot.
- Total: 5-6 additional cells × 3 seeds = ~45 h wall at 2-wide.

---

## 7. Promotion gate + mechanism diagnostic

### Standard promotion gate
- Paired Δα ≥ +3pp vs 760 a1b1 under eval-MCTS sims=16 joint.
- No seed < −5pp.
- If both hold → 5-seed extension. Else FAIL.

### Mechanism diagnostic (REQUIRED, even if standard gate fails)
- **Consistency score distribution must concentrate.** Per-batch histogram of `consistency_scores` over the K branches should be non-uniform — Bella futures the dynamics MLP "agrees with" should be a non-trivial subset. If `consistency_scores` are uniformly ~1.0 (filter accepts everything) OR uniformly ~0 (filter rejects everything), the mechanism is degenerate and the +3pp result (if positive) is from a different surface (e.g., better MCTS coverage by accident).
- **Log per-window oracle-PnL ratio on Bella branches.** Per `lore/notes_2026-05-18.md` §134, the "7× ratio" is a known risk: tag the `result.md` if per-window oracle-PnL ratio on Bella branches > **2×** historical (same protocol as 758). If > 2× AND Alpha is positive, the result is suspect — the learned policy may be Bella-shaped not real-market-shaped (§134 line 100-102).
- **Anti-correlation early-exit**: if Δ-Bal.Acc > +1pp AND Δα < 0 at epoch 50 of seed 7, abort cell (same protocol as TASK_763 / TASK_725-D-stack).

### Informative null
If `consistency_scores` collapse to uniform AND Δα is null, the result is interpretable: the dynamics MLP has not learned a manifold separable enough to distinguish realistic from off-manifold Bella futures. That is the closure of "Bella class can be re-opened via rollouts." See §9 follow-on.

---

## 8. Cost estimate (post-authorization)

| Phase | Wall |
|---|---|
| Code (3-5 days workstation) | ~3-5 days |
| Gate 1 (qa-hunt + qa-crossval review + mitigations) | ~0.5-1 day |
| Gate 2 (both smokes) | ~0.5 day |
| Paired 3-seed × 2-wide Stage-B on server (~2-3× canon step → 5-7 h/seed; 3 seeds at 2-wide) | ~12-15 h |
| Eval-MCTS sims=16 joint per ckpt + result.md authoring | ~3-4 h |
| **Total post-authorization wall** | **~5-6 days first measurable result** |

VRAM budget per TASK_722 §VRAM Budget: ~2.5 GB combined (model + Bella + buffers) — easy fit on RTX 3090's 24 GB. No OOM risk at production batch.

---

## 9. Risks + anti-correlation check

### Direct risks

1. **Dynamics consistency score rejects all Bella futures.** Means Bella's flow-matching distribution is off-manifold from the encoder/dynamics joint learned manifold. **Informative null** — Bella class stays closed; only the "Bella-as-rollouts" exception fails, not the broader idea of world-model-in-MCTS. Pivot: NeuralOGCM physics dynamics (`lore/NEURAL_MARKET_PHYSICS_DESIGN.md`) as alternative imagination engine.
2. **Dynamics consistency score accepts ALL Bella futures (degenerate filter).** Means dynamics MLP has learned an over-smooth manifold; Bella's "7× too easy" outputs slip through. Alpha may be positive but the mechanism diagnostic fails. Tag result.md per §7; do not promote.
3. **Per-batch wall > 500 ms.** TASK_722 §"Performance Budget" estimates 200-500 ms (vs 105 ms canon). If actual ≥ 1000 ms, Stage-B becomes ≥10× slower, budget collapses. Mitigation per TASK_722: generate futures once per epoch (not per step), amortize Bella cost over 400 steps. Adds ~30 LOC to `bella_dynamics.py` (cache + invalidation).
4. **K=16 × H=20 × n_assets=64 × 5 = ~100K float32 per sample × batch 256 ≈ 100 MB working memory.** Within budget but worth a sanity check at smoke time.

### Anti-correlation check (per `project_classifier_alpha_anti_correlation.md`)

- **DIRECT risk: LOW.** TASK_778 touches the dynamics MLP and the MCTS expansion path. It does NOT add any gradient flow into the encoder, does NOT modify the BC/classifier loss, does NOT change per-asset 5-class softmax output structure. The encoder runs in inference mode when re-encoding Bella futures. The anti-correlation finding (738/740/741/757) implicates encoder-side sharpening gradients; TASK_778 has none.
- **INDIRECT risk: MEDIUM.** Bella futures could be systematically *easier* (the 7× ratio finding). The MCTS policy targets distilled from Bella-anchored searches could then over-anchor the policy toward easy-regime trades, generalizing poorly to hard regimes (the `[[bull-blindness]]` pattern). Mitigation: per-window oracle-PnL ratio tagging (§7); 5-seed extension required before promotion to validate hard-regime generalization (`runs/746_value_oracle/result.md`'s per-period breakdown is the template).

### Indirect novelty: this is the FIRST mechanism that touches the dynamics class

- Single-mechanism class taxonomy per `lore/notes_2026-05-19.md` §149 and `project_single_mechanism_exhausted.md`: every prior mechanism touched encoder OR data OR planning-target OR value-target OR memory OR search-depth. **Dynamics class is genuinely untouched.** TASK_778 is the first probe.
- A null result here is itself a class-closure data point. A positive result is a new mechanism class.

---

## 10. References (load-bearing)

- `lore/ship_path_recommendations_2026-05-25.md` §4 — TASK_722 candidate framing.
- `tasks/TASK_722_BELLA_DYNAMICS_ROLLOUTS.md` — the original (parked) design; this task is a revival audit.
- `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md` — explicit "Bella class CLOSED" with the world-model carve-out.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — the anti-correlation finding that constrains mechanism design.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — single-mechanism class closure; structural pivot roadmap.
- `lore/notes_2026-05-18.md` §134 — Bella state (768d/12L conv, 10 ODE steps, 1440-pad bug workaround, 7× too-easy ratio).
- `lore/notes_2026-05-18.md` §138 — two-gate review protocol + smoke-at-production-batch rule.
- `mcts_trader/jax_agent/muzero_model.py:380-388` — dynamics_dense1/dynamics_ln/dynamics_dense2 (definition).
- `mcts_trader/jax_agent/muzero_model.py:583-610` — `dynamics(latent, action)` forward.
- `mcts_trader/jax_agent/muzero_model.py:701-728` — `recurrent_inference(...)` (MCTS entry point).
- `mcts_trader/jax_agent/factored_search.py:80-114` — factored search expansion call (`mode='recurrent_inference'`).
- `mcts_trader/jax_agent/mctx_search.py:89-140` — joint MCTS `_recurrent_fn` (`mode='recurrent_inference'`).
- `mcts_trader/env/jax_encoder.py:480-540` — encoder body (sliding-window reuse target).
- `diff_teacher/generation/generator.py:62-321` — `MarketGenerator` + `from_checkpoint` + `generate_on_device`.
- `diff_teacher/skeleton/extraction.py` — `extract_market_skeleton` (skeleton conditioning path).
- `runs/762_reward_shaping/*/stage_b_config.json:147` — production uses `use_factored_search: true` (factored is the canon expansion path; joint is eval-only at sims=16).
- `experiments/758_bella_augment/SPEC.md` — sibling Bella experiment with `too_easy` tagging protocol (template).
- `tools/deploy/README.md` — Bella ckpt deploy contract.

---

## 11. Status / next step

- **DRAFT**, design only. No code in this session. No commits, no push, no deploy.
- Next operator action: code-budget authorization (multi-week build = serious commitment). Recommended sequencing per ship-path memo §"Recommended execution order": run candidates #1 (TASK_775), #2 (TASK_776), #3 (TASK_777), #5 (TASK_779) first as the cheap small-mechanism last probes. If all close null, TASK_778 is the largest-already-designed unrun rewrite under 2 weeks.
- Companion SPEC: `experiments/778_bella_rollouts/SPEC.md` (placeholder, `STATUS: BLOCKED on code-budget authorization`).
