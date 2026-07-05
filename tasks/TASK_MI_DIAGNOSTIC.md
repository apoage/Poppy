# TASK_MI_DIAGNOSTIC: Latent Mutual Information — Rank vs Sign

**Status**: PREREQUISITE for TASK_730. CPU-only, <1 hr wall clock.
**Gate referenced by**: `tasks/TASK_730_RANK_ORACLE.md` §"MI-Diagnostic Prerequisite".
**Checkpoint under test**: `jax_ckpt_easy_phase2_728c/phase2_stageB_best.pkl`.

---

## Rationale (3 sentences)

728c plateaus at `dir_acc ≈ 0.57` despite all model-side interventions (726–729c); rank-oracle (TASK_730) pivots supervision to dense cross-sectional ordering, but costs days to wire in. Before committing, measure whether the current 728c encoder latent already carries more information about per-asset **rank** than per-asset **sign** — if not, the rank target cannot outperform the sign/direction target the model already fails to fit. Decision: proceed with TASK_730 only if `MI(latent, rank) > 2 × MI(latent, sign)` under identical aggregation.

---

## Data collection plan

- **Checkpoint shape (verified from `stage_b_config.json`)**: `state_dim=128`, `n_channels=34`, `window_size=64`, `n_assets=64`, `encoder_version=2`, `embed_dim=64`.
- **Source**: reuse `AsyncBatchPipeline` from `train_phase2.py:420`. It handles the 34-channel path (`use_state_channels = n_channels > 27`) by running `jit_batch_encode_temporal` (27 market channels) then appending 6 state channels + 1 confidence channel. Its `get_batch()` returns `obs_t` of shape `(B, 34, 64, 64)` plus `future_cum_returns` of shape `(B, 64)` — exactly what we need for both input and target.
- **Sample size**: N = 8000. Rationale: 128-dim latent, sklearn per-feature MI with `n_neighbors=3` has ~2% bias per marginal; 3-seed stability check (±20%) catches residual noise.
- **Periods**: write `research/mi_periods_tmp.json` with 3 entries matching `foxglove_periods.json` schema (`start_date`, `end_date`, `assets` — `assets` is REQUIRED; copy asset lists from the nearest matching entries of `config/foxglove_periods.json`):
  1. bull: `2020-10-01 → 2021-04-01`
  2. bear: `2022-05-01 → 2022-11-01`
  3. chop: `2023-06-01 → 2023-11-01`
- **Tensors cached to `research/mi_diagnostic_data.pkl`** (pickle dict):
  - `latents: np.float32 (N, 128)` — `encode_with_features(...)[0]` (global state).
  - `per_asset_future_sign: np.int8 (N, 64)` — `sign(future_cum_returns)`.
  - `per_asset_future_rank: np.int8 (N, 64)` — `argsort(argsort(future_cum_returns))`.
  Lookahead = `direction_horizon=20`, already baked into `future_cum_returns`.

---

## MI estimator choice

**k-NN Kraskov** via `sklearn.feature_selection.mutual_info_regression` (rank target — ordinal-continuous) and `mutual_info_classif` (sign target — discrete). Per asset `i ∈ [0, 64)`, per latent dim `d ∈ [0, 128)`:
- `mutual_info_*(X=latents, y=target[:, i])` returns a length-128 array of per-latent-dim MIs.
- Aggregate across latent dims: **mean** (reported as "average per-latent-dim MI"). Average again across 64 assets.
- The reported number is therefore **mean over (asset, latent-dim) pairs**, NOT joint MI. Both sign and rank use the identical aggregator, so the ratio is the decision criterion.

`n_neighbors=3`, `random_state=42`. Bootstrap CI over the 64-asset axis (200 resamples).

Rejected: MINE (overkill at N=8k); binned histogram (bias > signal at 128-D); true joint MI on the 64-dim vector target (unstable at this N).

---

## Pseudocode — full notebook (`research/mi_diagnostic_728c.ipynb`)

```python
# === Cell 1: Imports + force CPU (MUST precede jax import) ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, pickle, json, jax, jax.numpy as jnp
assert jax.devices()[0].platform == "cpu"

# === Cell 2: Load 728c checkpoint ===
from train_phase2 import load_phase1_checkpoint, Phase2Config
from mcts_trader.jax_agent.trm_flax import TinyRecursiveTradingModel
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE

CKPT = "jax_ckpt_easy_phase2_728c/phase2_stageB_best.pkl"
CFG_JSON = "jax_ckpt_easy_phase2_728c/stage_b_config.json"
params, ckpt_cfg, epoch, metrics = load_phase1_checkpoint(CKPT)
p2cfg_dict = json.load(open(CFG_JSON))  # state_dim=128, n_channels=34, window_size=64

model = TinyRecursiveTradingModel(
    state_dim=p2cfg_dict["state_dim"], n_actions=p2cfg_dict["action_space_size"],
    value_support_size=VALUE_SUPPORT_SIZE,
    hidden_dim=p2cfg_dict["hidden_dim"],
    encoder_version=p2cfg_dict["encoder_version"],
    embed_dim=p2cfg_dict["embed_dim"],
    encoder_heads=p2cfg_dict["encoder_num_heads"],
)
# True ckpt input shape: (B, 34, 64, 64)
dummy = jnp.zeros((1, 34, 64, 64), jnp.float32)
_ = model.init(jax.random.PRNGKey(0), dummy, mode='initialize_parameters')  # shape-check only
encode_fn = jax.jit(lambda p, x: model.apply({'params': p}, x, method='encode_with_features'))

# === Cell 3: Build periods + SegmentPool + AsyncBatchPipeline ===
from train_phase2 import load_segment_pool

# Write a trimmed periods file (3 entries, copy `assets` from foxglove_periods.json nearest matches)
trimmed = {"periods": [
    {"start_date":"2020-10-01","end_date":"2021-04-01","assets":[...]},  # bull
    {"start_date":"2022-05-01","end_date":"2022-11-01","assets":[...]},  # bear
    {"start_date":"2023-06-01","end_date":"2023-11-01","assets":[...]},  # chop
]}
json.dump(trimmed, open("research/mi_periods_tmp.json","w"))

p2cfg = Phase2Config(**{k:v for k,v in p2cfg_dict.items()
                        if k in Phase2Config.__dataclass_fields__})
pool = load_segment_pool(p2cfg, "research/mi_periods_tmp.json",
                         db_path="data/kraken.db", n_slots=3, max_runs_per_group=3)

# AsyncBatchPipeline gives us the FULL (B, 34, 64, 64) obs w/ portfolio channels.
# Portfolio/state channels come from slot['state_channels']/'confidence' → reflects
# oracle rollouts, not a live agent's positions. Documented below in §Assumptions.
from train_phase2 import AsyncBatchPipeline
pipe = AsyncBatchPipeline(pool, p2cfg, batch_size=64, prefetch=2)

# === Cell 4: Collect (latent, sign, rank) via get_batch() ===
N_TARGET = 8000
BATCH = 64
latents, signs, ranks = [], [], []
total = 0
while total < N_TARGET:
    b = pipe.get_batch()   # keys: obs_t, obs_t1, obs_tN, actions, future_cum_returns, ...
    obs = b['obs_t']       # (64, 34, 64, 64)
    fut = np.asarray(b['future_cum_returns'])  # (64, 64) [B, n_assets]
    global_state, _per_asset = encode_fn(params, obs)   # (B, 128)
    latents.append(np.asarray(global_state))
    signs.append(np.sign(fut).astype(np.int8))
    ranks.append(np.argsort(np.argsort(fut, axis=-1), axis=-1).astype(np.int8))
    total += obs.shape[0]
pipe.stop()

latents = np.concatenate(latents)[:N_TARGET]
signs   = np.concatenate(signs)[:N_TARGET]
ranks   = np.concatenate(ranks)[:N_TARGET]
pickle.dump({"latents":latents,"signs":signs,"ranks":ranks},
            open("research/mi_diagnostic_data.pkl","wb"))

# Sanity: assert encoder not collapsed
assert np.std(latents, 0).mean() > 1e-3, "Latent appears dead — abort before MI."

# === Cell 5: MI estimation — per-asset, per-latent-dim, then average ===
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def per_asset_mi(X, Y, estimator):
    # Y: (N, n_assets). Returns (n_assets,) array: mean per-latent-dim MI.
    out = np.empty(Y.shape[1], dtype=np.float64)
    for i in range(Y.shape[1]):
        mi_vec = estimator(X, Y[:, i], n_neighbors=3, random_state=42)
        out[i] = mi_vec.mean()   # avg across 128 latent dims (NOT sum — avoids double count)
    return out

mi_sign = per_asset_mi(latents, signs, mutual_info_classif)     # (64,)
mi_rank = per_asset_mi(latents, ranks.astype(np.float32), mutual_info_regression)  # (64,)

rng = np.random.default_rng(0)
def boot(x, B=200):
    idx = rng.integers(0, len(x), (B, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 50, 97.5])

# === Cell 6: Report ===
ratio = mi_rank.mean() / max(mi_sign.mean(), 1e-9)
print(f"MI(latent→sign)  avg-per-dim mean={mi_sign.mean():.4f}  CI={boot(mi_sign)}")
print(f"MI(latent→rank)  avg-per-dim mean={mi_rank.mean():.4f}  CI={boot(mi_rank)}")
print(f"ratio rank/sign = {ratio:.2f}")
decision = ("COMMIT TO TASK_730" if ratio > 2.0 else
            "ABORT TASK_730"     if ratio < 1.2 else
            "AMBIGUOUS — re-run N=10k")
print("DECISION:", decision)
# Append full numbers + decision to research/mi_diagnostic_results.md
```

---

## Assumptions & Limitations

1. **Portfolio/state channels**: `AsyncBatchPipeline` populates channels 28–34 from `slot['state_channels']` + `slot['confidence']`, reflecting **oracle** positions, not a live agent's state. For an encoder-only MI probe this is an acceptable baseline — the encoder sees a realistic (ws, 34, n_assets) distribution consistent with Phase 2 training. Swapping to a zeros-baseline ("neutral / no open position") is a valid alternative probing what the market-channel component alone encodes, but we use the native pipeline path to match training distribution.
2. **MI aggregation**: the reported number is **mean per-latent-dim MI, averaged over 64 assets**. This is NOT the joint `I(latent; rank_vector)`. Under identical aggregation on both sides the ratio remains a well-defined comparative criterion; it is NOT an unbiased estimator of the joint-MI ratio.
3. **Decision ratio `>2.0`**: heuristic, not literature-anchored. Chosen conservatively against false positives. 20–50% MI lifts are sometimes material in RL feature-quality work; the ambiguous band (1.2–2.0) triggers a 1-day head-only ablation before full TASK_730 commit.
4. **k-NN bias**: sklearn Kraskov on N=8k with k=3 has per-feature bias ~2%; multi-seed check (±20%) guards against over-interpretation.

---

## Files referenced / modified

Read-only imports (no edits):
- `train_phase2.py` — `load_phase1_checkpoint` (L94), `load_segment_pool` (L378), `Phase2Config`, `AsyncBatchPipeline` (L420).
- `mcts_trader/jax_agent/trm_flax.py` — `TinyRecursiveTradingModel.encode_with_features` (L271).
- `mcts_trader/env/jax_encoder.py` — `EncoderConfig`, `jit_batch_encode_temporal` (used internally by pipeline).
- `mcts_trader/jax_agent/__init__.py` — `VALUE_SUPPORT_SIZE`.
- `jax_ckpt_easy_phase2_728c/{phase2_stageB_best.pkl, stage_b_config.json}`.
- `config/foxglove_periods.json` (source for `assets` lists).

Written:
- `research/mi_diagnostic_728c.ipynb`
- `research/mi_diagnostic_data.pkl`
- `research/mi_diagnostic_results.md`
- `research/mi_periods_tmp.json`

NO edits under `mcts_trader/` or training code.

---

## Decision gates

| `MI(rank) / MI(sign)` | Action |
|---|---|
| > 2.0 | COMMIT to TASK_730 (rank oracle). |
| 1.2 – 2.0 | AMBIGUOUS. Re-run N=10k + different seed. If still here, do 1-day head-only ablation before full commit. |
| 0.8 – 1.2 | No advantage. Abort TASK_730; pivot per `research/729c_ceiling_alternatives.md`. |
| < 0.8 | Counterintuitive — debug rank orientation, verify `std(latents)` non-degenerate, check `n_neighbors`. |

**Stability check**: Cells 5–6 run 3 times with seeds `{0,1,2}`. If ratio varies >20%, raise N→10k, `n_neighbors`→5.

---

## Constraints

- CPU-only (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`). Must not contend with V3 training.
- Wall-clock budget: 60 min. Distribution: 5 min ckpt+pool+pipe warmup, 15 min batch collection (125 batches × ~5s CPU encode), 35 min sklearn MI (64 assets × 2 estimators, `n_jobs=-1` where supported).
- If MI overshoots 45 min, halve N to 4000 and record N alongside the ratio.
