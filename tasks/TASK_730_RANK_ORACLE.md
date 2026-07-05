# TASK_730: Cross-Sectional Rank Oracle

**Status**: ACTIVE (as of 2026-04-22) — V3 (729c) failed E50 gate at dir_acc=0.570. This is the executing pivot.
**Priority**: HIGHEST. Cheapest label-side fix with strongest literature support.
**Rationale**: `research/729c_ceiling_alternatives.md` §1; arXiv:2012.07149.

---

## ⚠️ STALENESS NOTE — READ FIRST

This doc was written as a contingency plan before V3 ran. Sections below marked *stale* have been superseded. **Authoritative current plan: `lore/notes_2026-04-22.md` §42 "Post-Compact Continuation Instructions"** — that has the correct warmstart path, V3 encoder flags, VRAM-safety env vars, 11-step implementation plan, and updated gate thresholds.

Specific corrections:
- **Warmstart checkpoint**: use `jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl` (V3 Phase 1 best, 76.7% agreement). **NOT** `jax_ckpt_easy_phase2_728c/best.pkl` (wrong filename AND wrong encoder family).
- **Encoder flags required in launch**: `--encoder-version 3 --embed-dim 64 --encoder-heads 8`.
[redacted: host service detail]
- **No new oracle function needed**: `future_cum_returns[B, n_assets]` is ALREADY in the batch (added by TASK_727 pipeline). Rank target is computed INLINE in `loss_fn`: `ranks = jnp.argsort(jnp.argsort(fut_ret, axis=-1), axis=-1) / (n_assets - 1)`. Do NOT add `cross_sectional_rank_oracle` as a new function unless there's a specific reason.
- **Loss first pass**: MSE on normalized ranks (simplest, unambiguous gradient, no temperature hyperparam). Listwise/pairwise held as stretch.
- **RankHead**: mirror `DirectionHead` pattern in `muzero_model.py`. Shared MLP over `per_asset_feats` (B, n_assets, embed_dim) → (B, n_assets) scalar scores. Zero-init final Dense.
- **MI prerequisite**: OPTIONAL (user decision 2026-04-22). Script at `research/mi_diagnostic_728c.py` (not `mi_notebook_728c.py`). Can run post-hoc on 728c data anytime.
- **Implementation approach**: main thread, NOT subagent chain (see notes §41 orchestration retrospective).

The "Loss option A/B", "MI-Diagnostic Prerequisite", "Phase 2 Retrain" launch-command, and "Gates" sections below are historical design rationale — reference only; use notes §42 for action plan.

---

## The Problem (Restated)

Current oracle `greedy_lookahead_path(k=4, lookahead=20)` produces:
- 93.75% HOLD labels by construction (60/64 assets get HOLD every step)
- Only 4 assets/step receive trade labels
- BC gradient fits the HOLD marginal; per-asset BUY/SELL conditional never learned
- Mode-collapse textbook case (arXiv:2510.20817): KL-regularized RL with sparse labels → majority-class collapse

All of 726-729c attempted to fix this from the model side (class weights, structural bias, wider encoder, temporal encoder). **The fix is on the label side.**

---

## The Change

Replace/augment the 4-of-64 selection mask with a **per-asset rank target**: every asset gets a label every step. Instead of "which 4 assets to trade", the oracle emits "what is each asset's expected-return rank among the 64 at lookahead=20".

### Loss option A (listwise, recommended)
```python
# Oracle produces ranks r[B, T, 64] ∈ [0, 63]
# Model head predicts per-asset scores s[B, T, 64]
# Listwise loss (Plackett-Luce / ListNet):
rank_loss = -jnp.sum(nn.log_softmax(s) * soft_target_from_rank(r))
```

### Loss option B (pairwise, faster)
```python
# Sample pairs (i, j); push model to rank them correctly
# RankNet / LambdaRank style
pair_loss = -jnp.mean(jnp.log(sigmoid(s_i - s_j)) * (r_i < r_j))
```

Start with listwise (denser gradient). Switch to pairwise if memory tight.

### Blend with existing BC
```python
total_loss = bc_weight * bc_loss + rank_weight * rank_loss + direction_weight * dir_loss + ...
```
Target blend: `rank_weight=1.0`, `bc_weight` ramps from 0.5→0.1 over training.

---

## Files Modified

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/path_solver.py` | Add `cross_sectional_rank_oracle(prices, lookahead)` returning `(B, T, 64)` rank tensor |
| `mcts_trader/jax_agent/muzero_model.py` | Add `RankHead`: per-asset scalar score. Reuse per-asset features from `encode_with_features`. Zero-init final layer for stability. |
| `train_phase2.py` | Compute rank oracle in batch pipeline alongside existing oracle. Add `rank_loss` to total loss. CLI: `--rank-weight`, `--rank-loss` (listwise/pairwise). |
| `phase2_config.py` | `use_rank_head: bool = True`, `rank_loss_weight: float = 1.0`, `rank_loss_type: str = "listwise"`. |

---

## MI-Diagnostic Prerequisite *(STALE — now optional per 2026-04-22 user decision)*

Script actually at `research/mi_diagnostic_728c.py`. Before implementing, run it:
- Compute `MI(latent_728c, per_asset_sign)` and `MI(latent_728c, per_asset_rank)`.
- If `MI(rank) > 2×MI(sign)` → proceed with full rank-oracle implementation.
- If `MI(rank) ≈ MI(sign)` → rank target won't beat sign target; re-evaluate.
- If `MI(rank) < MI(sign)` → counterintuitive, debug MI notebook before proceeding.

This is a 1-hour check that prevents a multi-day implementation from being ungrounded.

---

## Phase 2 Retrain *(STALE — use the authoritative command in `lore/notes_2026-04-22.md` §42)*

Shape-compatible only with the V3 Phase 1 checkpoint (V2 warmstarts would fail because encoder shape differs). Warmstart from `jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl`.

Authoritative launch command (copy from notes §42):
```bash
[redacted: host service detail]
mkdir -p jax_ckpt_easy_phase2_730 && \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --rank-loss mse --rank-weight 1.0 \
  --output-dir jax_ckpt_easy_phase2_730 \
  > jax_ckpt_easy_phase2_730/stage_b_nohup.log 2>&1 &
```

---

## Gates *(use notes §42 as source of truth; kept here for historical context)*

### E20
- `dir_acc > 0.58` AND `rank_spearman > 0.25` (spearman between predicted rank_scores and oracle ranks, averaged over batch).

### E50 (THE GATE)
- `dir_acc > 0.65` AND `rank_spearman > 0.40` → strong positive, continue to 200.
- `dir_acc 0.60–0.65` → continue; benchmark at E100.
- `dir_acc < 0.60` → rank didn't help. Pivot to TASK_731 (identity-embed ablation).
- Mixed (dir improves, rank doesn't OR vice versa) → diagnostic: head architecture or target, not encoder.

### Benchmark
- Bal.Acc > 45% AND positive PnL → ship.
- Bal.Acc 38–45% → incremental improvement; stack with identity-embed ablation.
- Bal.Acc < 38% → rank alone insufficient. Full stack (730 + 731 + 732).

---

## Why This First

- Does not require encoder retrain (head-only change, warmstart OK).
- Lit-backed (2012.07149 shows cross-sectional ranking outperforms top-k on multi-asset).
- Diagnostic-gated (MI notebook prevents blind commitment).
- Orthogonal to all other alternatives — stacking with 731/732 later is clean.

---

## Known Risks

- **Rank explosion with ties**: assets with near-equal returns fight over rank. Use soft-rank (argsort of softmax) or break ties with random tiebreak (consistent across batch).
- **Rank supervision vs MCTS target incompatibility**: MCTS still produces action visits, not ranks. Two parallel training signals — similar to how direction head already coexists with BC.
- **Batch size effects**: listwise loss scales O(n²). For n=64 assets it's fine, but watch VRAM if batch grows.
