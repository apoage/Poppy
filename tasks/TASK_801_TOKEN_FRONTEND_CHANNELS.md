# TASK_801 — Token front-end: per-timestep token embedding into the encoder (control arm)

**Status**: 📋 SPEC (2026-06-09, amended 2026-06-10 with the [[TASK_800]] serving-lag verdict).
**UNBLOCKED** — TASK_800 signed off: L8 SERVABLE, L1 SERVABLE-WITH-LAG-1 (20 effective assets ×
1000 cuts, 968 merge-stress, lag_hist {0:987, 1:13}, A0 byte-identical).
Concretizes the [[TASK_794]] front-end fork. Feeds [[TASK_804]] arms 2-3.

## Serving-lag convention (from the TASK_800 verdict — NOT optional)

Churn at k=1 means: at decision time *t*, the annotation of bar *t−1* is still provisional (13/1000
cuts). The final-only horizon at time *t* is bars ≤ *t − (max_lag + 1)*. Therefore:

- **L1 inputs**: full-stream annotation **delayed by 2 bars** (`token_input[t] = token_id[t−2]`) —
  final-only, identical to what live serving sees, zero train/serve skew. A 2-bar information delay
  on a 64-bar window is negligible.
- **L8 inputs**: **unshifted** (LAG-0 exact; churn ≡ 0 over 1000 cuts).
- The shift is applied at the timeline-slicing layer (one place), NOT inside the model — the model
  never knows about lag. The first 2 bars of an L1 stream map to the reserved invalid id.
- "Use tokens up to t−1" is WRONG for L1 (off-by-one: t−1 is provisional at time t). Do not
  implement that.

## Honest expectation — read this first

This arm is **predicted null on Alpha** by the project's own scar tissue: tokens-into-encoder is
still an encoder-side improvement, and 737/741/738/757/758 all closed that class
([[project_classifier_alpha_anti_correlation]], `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`).
It is built anyway because it is the **control arm** the real-vs-random gate requires, and because
the per-timestep entry is mechanically different from Mint's diluted broadcast scalars. Do not
sell it as the bet; [[TASK_802]] is the bet.

## Design

Tokens enter as a **separate int tensor**, embedded inside the model — NOT as float channels in the
obs grid (no float-cast ids, no V-wide one-hots).

1. **New optional kwarg** `token_ids: (B, T, A) int32` threaded exactly like the Mint
   `per_asset_scalars` chain: `StreamingAssetEncoder.__call__`
   (`mcts_trader/jax_agent/trm_encoder_v3.py:119-122`) ← `TinyRecursiveTradingModel.encode`
   (`trm_flax.py:288-310`) ← `MuZeroNetworks.representation*` (`muzero_model.py:514-532`) ←
   trainer batch dict.
2. **Inside the encoder**, before the minGRU reshape:
   - `tok_emb = nn.Embed(V, E)(token_ids)` → `(B, T, A, E)`; follow the in-repo embed precedent —
     **params float32, cast to compute dtype after lookup** (`muzero_model.py:372-378`).
   - `x = temporal_proj(x) + token_proj(tok_emb)` at `trm_encoder_v3.py:141`, where `token_proj` is
     `Dense(E → embed_dim)` with **zero-init kernel AND bias** → **byte-identical no-op at step 0**
     (the FiLM trick, `trm_encoder_v3.py:9-29`). This satisfies the channel-order invariant
     (`jax_encoder.py:382-387`) by never touching the obs channel layout at all.
   - `E = 16` default, `V` from the frozen vocab (static module attribute → JIT-static).
   - Invalid bars (`valid=False`, including `token_id == -1` pre-first-token bars per the
     [[TASK_800]] last-completed convention) map to a reserved id whose embedding row is
     zero-masked.
3. **Streaming parity**: the same lookup must work in `step()` (`trm_encoder_v3.py:175-220`,
   T=1 + carried state) or live trading breaks. Add it in the same PR.
4. **Batch plumbing**: the [[TASK_800]] RAM timeline is sliced by the same window indexing as raw
   OHLCV in `_prepare_numpy` / `AsyncBatchPipeline` (`train_phase2.py:579-615, 820-875`) and in the
   sequential eval loop (`tools/benchmark_easy.py:601-688`). Same slice op, same `window_start`,
   nstep tail included (win_len = ws + nstep = 84).
5. **Checkpoint warm-start**: backfill `trm/encoder/token_embed` + `trm/encoder/token_proj` via the
   `initialize_missing_aux_heads` deep-merge pattern (`train_phase2.py:167-211`).
6. **Optimizer labels**: Stage-B freezes the encoder — the new token params must be labeled
   **'train'**, not 'encoder', in `make_stage_b_labels` (`train_phase2.py:921-948`), or the whole
   arm silently trains nothing.
7. **Flag**: `use_token_frontend` default **OFF** in `phase2_config.py` (+ `easy_config.py` if
   Phase-1 is ever run with it). Flag OFF ⇒ canon bit-identical.

## Acceptance (before any training run)

- **E1**: flag OFF → checkpoint-identical forward outputs on a fixed batch (canon untouched).
- **E2**: flag ON, fresh zero-init params → **byte-identical** outputs to flag OFF (no-op proof,
  the load-bearing TASK_741 check).
- **E3**: gradients reach `token_embed`/`token_proj` under the Stage-B optimizer (nonzero after
  one step on a synthetic batch).
- **E4**: `step()` streaming path produces the same per-asset feats as the windowed path on a
  replayed sequence (existing parity-test pattern).
- **E5**: random-codebook arm wiring — swapping the vocab for a frequency-matched permutation
  ([[TASK_804]]) touches **only the id mapping**, zero code branches.

## VRAM note (do not OOM)

Additions are an embed table (≤1536 × 16 f32 ≈ 100 KB) and a `(256, 84, 64) int32` batch tensor
(~5.5 MB). Nothing. Keep PREALLOCATE default + `MEM_FRACTION=0.7`
([[feedback_jax_vram_preallocate]]). Keep `token_ids` OUT of the factored-search latent expansion
(`factored_search.py:72-91` multiplies by 320) — tokens exist at the root only.

## What NOT to do

- No post-pool FiLM injection of token info (Mint surface, −6.98pp precedent, [[project_task741_result]]).
- No multilingual embedding yet ([[TASK_793]] deferred — single language per arm keeps the
  real-vs-random control clean).
- No new attention anywhere (`mcts_trader/CLAUDE.md` architecture rule).
- Commit + report; **never `deploy.sh`** ([[feedback_sub_agent_no_deploy]]).
