# Deep Analysis: Foxglove Stage B — MCTS ExIt Trading Model

You are an expert ML engineer analyzing a trading model that is *close* to working but has structural issues. Your job is to find pitfalls, mistakes, missed opportunities, and propose solutions. Be thorough, be critical, question assumptions. Read the actual code — don't trust summaries.

---

## System Overview

**Foxglove** is a TRM (Tiny Recursive Model) + MuZero-style MCTS trader trained on historical crypto market data via Expert Iteration (ExIt).

**Training pipeline (Phase 2, Stage B)**:
1. Sample observation windows from historical market data (SegmentPool)
2. Run MCTS (32 sims, Gumbel MuZero) on each observation → visit count distribution
3. Train policy to match MCTS visit counts (distillation)
4. Train value head on n-step TD returns (20-step, γ=0.95, oracle rewards)
5. Train dynamics MLP on state consistency + reward prediction
6. BC regularizer (oracle policy, weight=0.005, decayed from 0.05)

**Architecture** (see `CLAUDE.md` for full details):
- TRM: encoder (cross-asset attention) → reasoning (MLP loops, H=6, L=8) → policy/value heads
- Dynamics: 2-layer MLP (latent + action → next_latent + reward)
- Value: categorical over 401 bins in [-10, 10], PopArt normalized
- 49.59M parameters, 320 actions (5 × 64 assets)

**Key files to read**:
- `train_phase2.py` — training loop, `AsyncBatchPipeline`, `make_stage_b_train_step`
- `phase2_config.py` — all config fields
- `mcts_trader/jax_agent/mctx_search.py` — MCTS search implementation
- `mcts_trader/jax_agent/muzero_model.py` — TRM model architecture
- `mcts_trader/jax_agent/popart.py` — PopArt normalization
- `tools/benchmark_easy.py` — benchmark evaluation
- `lore/notes_2026-04-05.md` through `lore/notes_2026-04-13.md` — session history
- `tasks/TASK_718_POPART_VALUE_FIX.md`, `tasks/TASK_719_VALUE_TARGET_ROADMAP.md`, `tasks/TASK_721_PIPELINE_SPEEDUP.md`

---

## Current State

### Latest Benchmark (E982, n-step + async pipeline + random sampling)

```
Period         Bal.Acc  Trade%  OraclePnL   ModelPnL  RandomPnL    Alpha  
bull            33.0%    3.1%   +265.20%     -3.80%    -66.39%   -21.95%  ← ALWAYS bad
bear            33.2%    3.1%   +179.69%     +1.53%    -66.73%   +27.57%
sideways        34.4%    3.1%    +90.09%     -2.63%    -66.86%    -0.34%  ← barely negative
ftx_crash       33.2%    3.1%   +257.65%     -4.03%    -66.50%   +24.26%
etf_approval    32.8%    3.1%   +165.10%     +0.84%    -65.67%    -8.39%  ← ALWAYS bad
boring_week     33.0%    3.1%   +231.00%     -4.13%    -66.87%    +6.81%
AVERAGE         33.3%    3.1%   +198.12%     -2.04%    -66.50%    +4.66%

Value MSE: 302.67
Gate: 3/6 positive alpha (need 4/6), beats random 6/6 ✓
```

### Phase 1 Baseline (for comparison)
```
Phase 1 (oracle imitation, no MCTS): 0/6 positive alpha, 0/6 beats random, -154% avg alpha
```

So MCTS ExIt dramatically improved things (0/6 → 3/6, -154% → +4.66%), but two specific regimes (bull, etf_approval) are structurally failing.

### Training Metrics (E999)
```
policy=1.93  value=1.09  consistency=1.32  bc=1.44  total=4.41  grad=8.67
td_mean=2.47  td_std=2.39  pa_mu=2.47  pa_sig=2.39
gen=4.4s  mcts=38.2s  train=7.9s  (50.8s/epoch total)
```

### Value Loss History
```
E0:    value=6.67 (cold start, reset head)
E100:  value=2.39
E500:  value=1.50
E982:  value=1.10 (still declining at end)
```

### Policy Loss History  
```
E0:    policy=1.82
E93:   policy=1.71  ← best ever
E500:  policy=1.90
E999:  policy=1.93  ← plateaued/slightly worse
```

---

## Known Issues & History

### 1. Bull Market Blindness (PERSISTENT)
Model consistently loses -20% to -27% alpha in bull and etf_approval periods across ALL training runs. This pattern held through:
- Oracle value targets (719a)
- N-step TD returns (719b)
- Diverse data (10 runs/group)
- Async pipeline with random sampling (721)

This is also observed in Poppy (the production MuZero model). Hypothesis: model defaults to conservative HOLD in uptrends instead of BUY-and-hold.

### 2. No Slot Rotation with Random Sampling
`AsyncBatchPipeline` uses random cursor positions (not sequential). This means `_rotate_slot()` is never triggered — the same 17 initial segments stay in RAM for all 1000 epochs. `max_runs_per_group=10` means 140 segments are available but only 17 are used. The old sequential pipeline naturally rotated when cursors exhausted.

### 3. Policy Plateau
Policy improved to 1.71 early (E93) then regressed to 1.90+ and stayed there. Value continued improving (2.4 → 1.1). The policy is not benefiting from the improved value function. Possible causes:
- MCTS visit distributions not changing despite better value (search too shallow?)
- Learning rate too high/low for policy after initial convergence
- BC weight floor (0.005) pulling policy away from MCTS targets
- Random sampling destroying temporal structure that helped early training

### 4. Value MSE = 302 at Benchmark
During training, value_loss=1.09 (categorical CE, PopArt normalized). At benchmark, raw value MSE=302. The PopArt normalization masks the scale mismatch. MCTS tree search uses `value_scale` and `value_shift` from PopArt to denormalize, but this may not work correctly at inference time with different market regimes.

### 5. Trade Density = 3.1%
Model trades very rarely. This helps avoid losses but means it's not capturing opportunities. Oracle trades ~40-50% of the time. Random trades ~50%. The model at 3.1% is extremely conservative — likely defaulting to HOLD for almost everything.

---

## What to Investigate

### A. The Training Loop & Loss Function
Read `make_stage_b_train_step()` in `train_phase2.py`. Look for:
- Is the value loss actually teaching the right thing? (n-step TD with oracle rewards, PopArt normalized, categorical CE)
- Is the policy loss well-formed? (CE between policy logits and MCTS visit counts)
- Is the BC regularizer helping or hurting at 0.005 weight?
- Are gradients flowing correctly to all heads? (check freeze masks, optimizer masks)
- Is the dynamics consistency loss actually calibrating the transition model?

### B. MCTS Search Quality
Read `mctx_search.py`. Look for:
- Is Gumbel MuZero configured correctly? (32 sims, 16 max considered actions)
- How does value denormalization work? (`value_scale`, `value_shift` from PopArt)
- Is the dynamics model good enough for multi-step tree search? (2-layer MLP, compounding errors)
- What does the visit distribution actually look like? (entropy, concentration)
- With 320 actions and only 16 max_considered, is the model exploring enough?

### C. Data Pipeline & Sampling
Read `AsyncBatchPipeline` in `train_phase2.py`. Look for:
- Is random sampling actually producing diverse batches? Or are the same 17 segments too repetitive?
- Are the n-step rewards computed correctly? (vectorized numpy)
- Is the oracle remaining PnL horizon (400 steps) appropriate?
- Are the rewards scaled correctly? (REWARD_SCALE=100, per-step PnL ~0.001)

### D. Bull Market Specific Analysis
The model is profitable in bear/crash/boring but loses in bull/etf_approval. Investigate:
- What does the oracle do differently in bull vs bear periods? (action distributions, hold durations)
- What does the MODEL do in bull periods? (action distributions from benchmark JSON)
- Is the training data balanced across bull/bear regimes?
- Could the oracle's formulation (absolute returns, k-of-N selection) bias against trend-following?

### E. Architecture Concerns
Read `muzero_model.py`. Look for:
- Is the TRM architecture suitable for this task? (MLP reasoning, no temporal memory)
- Is the value head capacity sufficient? (3 layers: 128→512→512→401)
- Is the dynamics MLP too simple for multi-step planning? (2 layers: 128+320→128)
- Are there any numeric issues? (support range [-10,10], bin width 0.05, PopArt interaction)

### F. Benchmark vs Training Gap
Read `tools/benchmark_easy.py`. Look for:
- How does inference differ from training? (sequential state tracking, different observation construction)
- Is PopArt denormalization applied correctly at inference? 
- Does the benchmark use MCTS or just the policy? 
- Are there distribution shift issues between training periods and benchmark periods?

---

## What I Want From You

1. **Read the actual code** — don't speculate. Look at `train_phase2.py`, `mctx_search.py`, `muzero_model.py`, `benchmark_easy.py`, `popart.py`.
2. **Find bugs or logical errors** — wrong signs, wrong scales, missing operations, incorrect gradient flow.
3. **Find design mistakes** — things that are "correct" but counterproductive (e.g., hyperparameters that fight each other, loss terms that cancel out).
4. **Explain the bull market failure** — what specifically about uptrends would cause this model to fail?
5. **Explain the policy plateau** — why did policy stop improving despite value improving 2x?
6. **Propose concrete fixes** — ranked by expected impact and implementation effort. Include specific code changes or parameter adjustments.
7. **Challenge assumptions** — is MCTS actually helping? Is the categorical value head the right choice? Is 320 actions too many? Should we be doing something fundamentally different?

Be systematic. Start with the data flow and trace it end-to-end. Don't skip anything because it "looks fine."
