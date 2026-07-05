# TASK_733: Oracle-Knowing Upper-Bound Diagnostic

**Status**: PROPOSED (as of 2026-04-23). Gates all further TASK_73x/74x work.
**Priority**: HIGHEST. Answers "is this task even learnable at this data granularity?" before any more multi-day training commitments.
**Rationale**: `lore/notes_2026-04-22.md §46`.

---

## Problem

After 726 → 732, the direction-accuracy ceiling in Stage B is stuck at ~0.57 regardless of:
- model side (class-balanced BC, direction head, per-asset policy, wider encoder, V3 temporal encoder)
- label side (rank oracle / listwise target)
- input side (cross-sectional channels, rank-of-return, beta, idio vol)

Two mutually exclusive explanations for the ceiling:

1. **Information ceiling** — 60-step OHLCV windows don't contain enough signal to predict 20-step forward sign at crypto-minute scale. 0.57 is near the theoretical limit of this feature set. Literature shows 0.53–0.58 SOTA for minute-bar crypto direction prediction, consistent with this hypothesis.
2. **Architecture/training ceiling** — the signal is there, but our model/loss/optimizer can't extract it. Under this hypothesis, a stronger model or different training signal could break 0.57.

These two imply very different next moves. **C1 answers which it is.**

---

## The Diagnostic

Train a model whose input *includes future prices at lookahead=20*. That model has perfect information about the target. Measure its dir_acc ceiling.

- If "cheat model" caps at 0.60–0.65 → **information ceiling confirmed**. Signal in 60-step OHLCV is exhausted. Pivot to Tier D (richer inputs) or Tier E (change game).
- If "cheat model" hits 0.90+ → **architecture ceiling**. The signal is there, our training isn't extracting it. Pivot to Tier B (drop BC, pure RL) and/or further encoder/loss work.
- Intermediate (0.70–0.85) → **partial signal**. Some combination of richer inputs and better training. Tier A (sign-oracle) + Tier B (drop BC) is the next cheap pair.

The diagnostic is non-committal and cheap: ~1 day of work, one 2-hour training run. It can be wrong in only one direction (missing a hidden architectural issue that even the cheat model can't exploit), which is acceptable.

---

## Implementation

### Design — "cheat channel" as 33rd channel

Add one more channel to the encoder output: a per-asset scalar = sign of 20-step forward log-return, scaled to {−1, 0, +1} (with ε dead-zone for near-zero returns).

- Computed from `prices[t+1 : t+21]` vs `prices[t]` — i.e., uses data strictly after the observation window.
- Broadcast across time dim via existing `broadcast_scalar` helper.
- Only enabled via explicit CLI flag `--cheat-channel` on `train_easy.py`. Disabled by default — never in normal training.

This is the minimal invasive change. No architectural changes, no loss changes, no oracle changes. One extra input channel that is literally "the answer."

### Files to modify

| File | Change |
|---|---|
| `mcts_trader/env/jax_encoder.py` | Add `use_cheat_channel: bool = False` to `EncoderConfig`. Bump `n_channels: 32 → 33` when enabled. Add channel 32 computation in `_compute_all_channels_temporal` — takes `future_prices` argument, emits sign of forward log-return. |
| `mcts_trader/env/jax_encoder.py` | Plumb `future_prices` through `encode_observation_temporal` / `jit_encode_temporal` / `jit_batch_encode_temporal` / `jit_vmap_encode_temporal`. |
| `train_easy.py` | Add CLI flag `--cheat-channel`. When set: `encoder_config.use_cheat_channel=True`, bump `EasyConfig.n_channels` by 1 (→ 40 if state_channels on), extract future window from the same price tensor already in RAM, pass to encoder. |
| `easy_config.py` | No changes (n_channels flows from train_easy.py). |

**Key constraint**: future prices must come from the SAME data array the oracle uses. No leakage concerns — it's intentional leakage, controlled by the flag.

### Smoke test

- Without `--cheat-channel`: encoder output shape `(B, 32, 60, 64)` — unchanged from 732 default.
- With `--cheat-channel`: encoder output `(B, 33, 60, 64)`. Channel 32 values ∈ {−1, 0, +1} (approximately, after dead-zone).
- Channel 32 should perfectly predict the direction head's target (sigmoid → ≥0.5 when sign=+1).

### Launch

```bash
mkdir -p jax_ckpt_easy_phase1_cheat && \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 nohup python train_easy.py \
  --periods-file config/foxglove_periods.json --epochs 200 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --cheat-channel \
  --output-dir jax_ckpt_easy_phase1_cheat \
  > jax_ckpt_easy_phase1_cheat/phase1_nohup.log 2>&1 &
```

[redacted: host service detail]

### Tail (monitor only for gate / error)

```bash
tail -F jax_ckpt_easy_phase1_cheat/phase1_nohup.log | \
  grep --line-buffered -E "Val Agree: (9[0-9]|8[5-9])\.|Traceback|ERROR|NaN|Killed|OOM|aborted"
```

Fires only on Val Agree ≥ 85% (which would be a clear signal the cheat input is doing its job) or errors.

### Stage B with cheat checkpoint (optional, only if Phase 1 hits 0.90+)

If Phase 1 with cheat input reaches trade-agreement ≥ 90%, run a matching Stage B (cheat input still enabled) and measure its dir_acc. This tells us the *Stage B* ceiling with perfect information.

- If cheat Stage B dir_acc ≥ 0.85: confirms Stage B is OK when the signal is present; real problem is the information ceiling.
- If cheat Stage B dir_acc caps < 0.80 despite cheat input: Stage B has its own structural problem on top of the information ceiling. Investigate Tier B (burn BC) regardless.

---

## Gates

### Phase 1 (cheat)
- Val Agree ≥ 90% by E50 → signal is present; architecture can use it.
- Val Agree 80–90% → signal partially present; input granularity is borderline.
- Val Agree < 80% with cheat input → something is broken in how we extract signal; or data itself (preprocessing, normalization) is destroying the future-price signal. Debug before concluding anything.

### Benchmark on cheat P1 checkpoint
Expect very high Bal.Acc because direction is known. This is the upper bound for the architecture:
- Bal.Acc ≥ 80%: architecture can trade well given perfect info. Real ceiling is information.
- Bal.Acc 50–80%: even with perfect info, something else limits trading (transaction costs, HOLD bias, value head).
- Bal.Acc < 50% with cheat: *benchmark pipeline itself* has an issue. Investigate before going further.

---

## Interpretation Matrix

| Phase 1 cheat Val Agree | Cheat Bal.Acc | Conclusion | Next |
|---|---|---|---|
| ≥ 90% | ≥ 80% | Information ceiling. Signal not in 60-step OHLCV. | Tier D (richer inputs) or Tier E (change game) |
| ≥ 90% | 50–80% | Architecture can see signal but not trade on it. | Tier B (drop BC / pure RL) |
| 80–90% | any | Partial signal extraction. | Tier A1 (sign-oracle) + Tier B1 (no-BC) stacked |
| < 80% | any | Bug in cheat input or encoder. | Debug before any pivot |

---

## Files Modified (summary)

| File | Change | Breaking? |
|---|---|---|
| `mcts_trader/env/jax_encoder.py` | +1 channel when flag set, +`use_cheat_channel` config field, +`future_prices` arg through encoder stack | No — opt-in flag |
| `train_easy.py` | +`--cheat-channel` CLI, +future-window extraction | No |

All changes are opt-in via CLI flag. Default training path unaffected. Delete the flag + 30 lines of code after diagnostic is done, or keep as permanent audit tool for future arch changes.

---

## Known Risks

- **Leakage through training data structure** — if the future window crosses episode boundaries, the "future price" might be from a different regime. Add assertion that future window is within the same regime's contiguous data.
- **Normalization interaction** — channels 0–4 are normalized within the 60-step window. Channel 32 (cheat) uses future prices; if naively normalized together with channels 0–4, the window min/max changes and pollutes the other channels. Keep cheat channel's normalization isolated.
- **Dead-zone ε calibration** — ε too low → all signs resolve to +1/−1 → cheat too powerful → model ignores other channels entirely → Val Agree 100% trivially. ε too high → too many HOLDs → cheat signal diluted. Start with ε = 1e-4 (median absolute return scale) and tune from smoke test.
- **Don't accidentally ship** — a model with cheat input must NEVER be benchmarked as a production candidate. Benchmark path should assert `use_cheat_channel=False` before running live data.

---

## Why This First

- Cheapest possible diagnostic: ~1 day wall-clock, 2 hr training, no multi-run commitment.
- Resolves the single biggest open question (information vs architecture ceiling) with a clean binary answer.
- All other Tier A/B/D/E decisions depend on this result.
- Prevents wasting another 700 tasks grinding on a problem that may not be solvable at 60-step-OHLCV granularity.
- Low reversibility cost: changes are opt-in CLI flags; rollback is trivial.
