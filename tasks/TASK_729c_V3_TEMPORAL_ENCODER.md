# TASK_729c: V3 Temporal Encoder — Preserve Time-Ordering, Last Architectural Lever

**Status**: STANDBY (contingent on TASK_729b outcome)
**Priority**: Activate ONLY IF 729b E50 gate fails (dir_acc < 0.60 even with wider encoder)
**Depends on**: TASK_729b negative result
**Scope**: Switch Phase 2 from V2 (flatten-and-Dense) encoder to V3 (minGRU temporal) encoder
**Budget**: Full Phase 1 retrain required. Large commitment (24–48 hrs electricity + Phase 2).

---

## Why V3, And Why Only If 729b Fails

The V2 `AssetTransformerEncoder` architecture:
```
(B, 27, 60, 64)  →  transpose  →  (B, 64, 60, 27)
                 →  flatten per asset  →  (B, 64, 60*27=1620)
                 →  Dense(1620 → embed_dim)  →  (B, 64, embed_dim)
                 →  + positional embed  →  cross-asset attention  →  ...
```

The `flatten(60 time steps × 27 channels) → Dense(1620 → embed_dim)` is a single-layer compression that destroys temporal ordering. A "price went up then down" window produces the same first-layer embedding as "price went down then up" *if* the positional bias of the Dense happens to be similar. Wider embed_dim helps (729b), but temporal ordering is *structurally* lost.

The V3 `StreamingAssetEncoder` uses minGRU sequential processing per asset: each time step feeds into the next via a recurrent state. Temporal order is explicit. In principle it can extract richer direction features because "current price above moving average" and "current price below moving average + rising" are encoded differently — per-asset, temporally aware.

If even a wider V2 (TASK_729b) cannot raise dir_acc past 0.60, the temporal-blindness of V2 is the likely reason, not width.

---

## Cost Reality

V3 is a different encoder class. Any existing checkpoint (728c, 729a, 729b) is shape-incompatible. You must retrain Phase 1 from scratch with `encoder_version=3`. This is 12–24 hrs of Foxglove training alone, before Phase 2 even starts.

Phase 2 then needs at minimum 100–200 epochs (4–8 more hrs) to reach a verdict.

**Total commitment: 24–48 hrs of GPU electricity.** Do not start this until you've seen 729b's verdict AND are prepared to accept the multi-day runtime.

---

## Implementation — Minimal

### Change 1 — Config

`phase2_config.py` (and `easy_config.py` if this lives there):
```python
encoder_version: int = 3   # was 2 — switch to StreamingAssetEncoder (minGRU temporal)
```

### Change 2 — V3 per-asset feature exposure (TASK_729a compatibility)

Check `trm_encoder_v3.py` — currently does NOT support the `return_per_asset=True` path that 729a added to V2. You need to make V3 expose its per-asset pre-pool features the same way.

In `trm_flax.py::encode_with_features`:
```python
def encode_with_features(self, x):
    if self.encoder_version == 2:
        return self.encoder(x, return_per_asset=True)
    elif self.encoder_version == 3:
        return self.encoder(x, return_per_asset=True)   # requires V3 support
    # fallback
    state = self.encoder(x)
    return state, None
```

Then in `trm_encoder_v3.py::StreamingAssetEncoder.__call__`, before the final global pool, expose the per-asset GRU hidden states as `(B, n_assets, hidden_dim)`. Return tuple when `return_per_asset=True`. Minor surgery; mirrors the V2 change.

If V3's architecture doesn't have a clean "pre-pool per-asset" point, fall back to exposing the raw per-asset embeddings just before pooling. Document the choice.

### Change 3 — Phase 1 Retrain with V3

```bash
python train_easy.py \
  --periods-file config/foxglove_periods.json \
  --epochs 200 \
  --encoder-version 3 \
  --embed-dim 64 \
  --encoder-heads 8 \
  --output-dir jax_ckpt_easy_phase1_v3 \
  > jax_ckpt_easy_phase1_v3/phase1_nohup.log 2>&1 &
```

(Add `--encoder-version` CLI flag to train_easy.py if missing.)

Expected baseline trade-agreement on real data: should hit ≥ 75% given enough epochs. If V3 can't reach this, V3 itself is broken for this task — stop.

### Change 4 — Phase 2 from V3 Phase 1 Checkpoint

```bash
mkdir -p jax_ckpt_easy_phase2_729c && nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_v3/easy_final.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --output-dir jax_ckpt_easy_phase2_729c \
  > jax_ckpt_easy_phase2_729c/stage_b_nohup.log 2>&1 &
```

All TASK_727/728/729a wiring still applies — per-asset direction head, policy refiner, per-asset policy head all read from V3's exposed per-asset features.

---

## Diagnostic Gates

### Phase 1 (V3) 
- Trade-agreement ≥ 70% by E50 of Foxglove. If still < 60% at E100, V3 is failing to imprint. Abort.

### Phase 2 E50 (THE GATE)
- `dir_acc` > 0.65 → V3 temporal encoding is the fix. Continue to E200, benchmark.
- `dir_acc` 0.60–0.64 → modest improvement only. Benchmark anyway to see Bal.Acc impact.
- `dir_acc` < 0.60 → **definitive terminal result**. The limit is not V2 vs V3. It's either: (a) the 27 input channels themselves don't carry enough direction info per asset, or (b) the oracle formulation is fundamentally mismatched with what's predictable from this data. Stop. Ship 728c or accept the project is at its baseline.

### Phase 2 Benchmark
- Bal.Acc > 45% → productize this encoder. Re-run full Phase 1 (Strategy B of 729b) for production quality.
- Bal.Acc 38–45% → meaningful improvement but below the TASK_666 75% target. Decide: deploy incrementally, or park the project.
- Bal.Acc ≤ 38% → even V3 doesn't help. Terminal for all encoder-architecture hypotheses.

---

## Failure Modes

| Symptom | Meaning | Action |
|---|---|---|
| V3 minGRU numerically unstable (NaN early) | Known issue at large batch × long seq | Reduce batch, check `jax.block_until_ready` fences |
| Phase 1 V3 plateaus at low trade-agreement | V3 architecture mismatched to task | Abort; don't push to Phase 2 |
| V3 per-asset feats not exposable cleanly | Encoder ends with flatten/pool that destroys asset dim | Implement a parallel "feature branch" that keeps per-asset dim explicit |
| Phase 2 E50 gate fails | Neither V2 nor V3 breaks 0.60 ceiling | Project has hit architectural ceiling for this input/oracle |

---

## Why This Is The Last Architectural Lever

After 729b (wider V2) and 729c (switch to V3), you've tested:
- Loss weighting (726)
- Latent direction signal (727)
- Structural routing (728)
- Refiner cold-start (728b)
- Target sharpness (728c)
- Per-asset routing (729a)
- Encoder width (729b)
- Temporal encoding class (729c)

If all fail to raise Bal.Acc past ~40%, the ceiling is NOT the model side. It's either:
1. The input feature set (27 channels may not carry enough per-asset direction signal — re-examine channel definitions in `env/jax_encoder.py::_compute_all_channels_temporal`).
2. The oracle formulation (greedy k=4, lookahead=20 may be producing targets that no model can consistently match from current observations).

Both are project-scope changes. Stop iterating on Poppy training at that point and either ship 728c (realistic baseline) or rethink the data/oracle pipeline.

---

## Files Modified

| File | Change |
|------|--------|
| `phase2_config.py` / `easy_config.py` | `encoder_version=3` default or CLI override |
| `mcts_trader/jax_agent/trm_encoder_v3.py` | Add `return_per_asset` path |
| `mcts_trader/jax_agent/trm_flax.py` | `encode_with_features` routes V3 same as V2 |
| `train_phase2.py` | `--encoder-version` CLI flag (if not already) |
| `train_easy.py` | `--encoder-version` CLI flag (if not already) |

---

## Decision Tree At Compact Time

1. **Start with 729b** — wider V2 encoder, Strategy A (Phase-1-Lite → Phase-2).
2. At Phase 2 E50:
   - dir_acc > 0.63 → complete 729b. Benchmark. If Bal.Acc > 40%, ship.
   - dir_acc < 0.60 → activate 729c. Accept 24–48 hr budget.
   - ambiguous (0.60–0.62) → benchmark current 729b, decide on 729c based on whether Bal.Acc moved at all.
3. At 729c Phase 2 E50:
   - dir_acc > 0.65 → complete 729c, benchmark.
   - dir_acc < 0.60 → project ceiling confirmed, ship 728c.

This is the full decision tree. Safe to compact; both tasks are self-contained with commands and gates.
