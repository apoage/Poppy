# Phase 2 Audit Prompts

Use these prompts after completing each subtask gate. Feed to a separate agent for cross-validation before proceeding.

---

## 716a Audit — Infrastructure

```
Audit train_phase2.py and phase2_config.py for TASK_716a.

Read first:
- tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md (subtask 716a section)
- easy_config.py (base config being extended)
- easy_trainer.py:37-80 (Phase 1 model creation pattern)

Check:
1. Does Phase2Config inherit all EasyConfig fields correctly?
2. Does the model creation match Phase 1 exactly? (state_dim=128, action_space=320, H=6, L=8, etc.)
3. Does checkpoint loading verify dynamics=zero AND encoder=trained?
4. Does --dry-run produce the expected output format?
5. Are there any import cycles with train_easy.py (SegmentPool import)?
6. Is JAX preamble (XLA cache, GPU allocator) present before JAX imports?
```

---

## 716b Audit — Dynamics Initialization

```
Audit the dynamics initialization code for TASK_716b.

Read first:
- tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md (subtask 716b section)
- mcts_trader/jax_agent/muzero_model.py:169-209 (dynamics network)
- mcts_trader/jax_agent/muzero_model.py:256-283 (recurrent_inference)

Check:
1. Are ALL dynamics keys initialized? (action_embedding, dynamics_dense1/2, dynamics_ln, dynamics_state_norm)
2. Is He normal used for Dense kernels? Normal(0, 0.02) for embeddings?
3. Are LayerNorm scale/bias initialized to 1.0/0.0 (not left at zero)?
4. Does the test use config.state_dim (not hardcoded 128 or 64)?
5. Does recurrent_inference produce valid shapes after init? (latent: batch×state_dim, reward: batch×401)
6. No NaN in output? Check with multiple random seeds.
7. Are Phase 1 trained params (trm/) preserved exactly after init? (diff before/after)
```

---

## 716c Audit — Dynamics Loss

```
Audit the dynamics training step for TASK_716c.

Read first:
- tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md (subtask 716c + Stage A section)
- mcts_trader/jax_agent/muzero_model.py:256-283 (recurrent_inference with stop_gradient)
- easy_trainer.py:128-302 (Phase 1 train step pattern)

Check:
1. Is encoder ACTUALLY frozen? Verify gradients are zero for trm/ keys after a training step.
2. Is consistency loss correct? MSE(pred_latent, stop_gradient(true_latent_from_encoder))
3. Is reward scaled by REWARD_SCALE (100x) before categorical conversion?
4. Does scalar_to_categorical match the VALUE_SUPPORT range [-10, 10] with 401 bins?
5. Are consecutive window pairs (obs_t, obs_t1) correctly aligned? (cursor c and c+1)
6. What action is used for dynamics? Oracle action at decision_idx = c + ws - 1?
7. Does loss decrease over 10 epochs? (gate criterion)
8. Is the optimizer only updating dynamics params? (frozen params should have zero updates)
9. Step reward computation: is it sum(position * returns) / n_assets - txn_costs?
```

---

## 716d Audit — Stage A Full Run

```
Audit Stage A convergence for TASK_716d.

Read first:
- tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md (subtask 716d section)
- The training logs from Stage A run

Check:
1. Did consistency loss converge below 0.05? Plot the loss curve.
2. Did reward loss converge below 0.5?
3. Run 5-step unroll test: encode 5 consecutive observations, predict latent chain via dynamics, compare final predicted latent to actual encoded latent. Drift should be < 0.1.
4. Are dynamics weights no longer zero? Check sparsity (should be 0-5%).
5. Are encoder weights UNCHANGED from Phase 1 checkpoint? (byte-level comparison of trm/ keys)
6. Is the dynamics network producing diverse outputs? (check std of pred_latent across batch — should not be collapsed to a point)
7. Checkpoint saved correctly? Can it be loaded and resume training?
```

---

## 716e Audit — MCTS Smoke Test

```
Audit MCTS integration for TASK_716e.

Read first:
- tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md (subtask 716e section)
- mcts_trader/jax_agent/mctx_search.py (search wrapper)

Check:
1. Does search use the Stage A trained dynamics? (not zero-init dynamics)
2. Are visit counts normalized and non-negative?
3. Visit entropy in range 0.5-2.0? (not collapsed to single action, not uniform)
4. Root value is finite and reasonable? (not 0.0, not ±inf)
5. Are actions within valid range [0, 319]?
6. Does max_num_considered_actions=16 limit the search correctly?
7. Does Gumbel MuZero mode work? (use_gumbel=True)
8. Timing: how long does 1 batch of 16 take? Extrapolate to batch=256.
9. Does the search use the stop_gradient on prediction within recurrent_inference?
```

---

## 716e2 Audit — JAX-Native Pipeline (if implemented)

```
Audit the JAX-native data pipeline for TASK_716e2.

Read first:
- tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md (subtask 716e2 section)

Check:
1. Are prices resident on GPU? No CPU→GPU transfers during training loop?
2. Does lax.scan produce identical results to the Python pipeline? (compare losses on same data)
3. VRAM usage: how much GPU memory does the resident data consume?
4. Slot rotation handling: does it correctly break out of scan for I/O?
5. Speedup measured: >3x over Python pipeline?
6. No JIT recompilation during steady-state training? (check XLA compilation logs)
7. Cursor state correctly carried through scan iterations?
```

---

## 716f Audit — ExIt Training Loop

```
Audit the Stage B ExIt loop for TASK_716f.

Read first:
- tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md (subtask 716f + Stage B section)

Check:
1. Are MCTS visit counts used as policy targets? (not oracle actions)
2. Is oracle BC present as regularizer with decaying weight (0.05→0.005)?
3. Is the decay schedule correct? (over 5000 steps, not epochs)
4. Policy loss: cross_entropy(policy_logits, visit_counts)?
5. Value loss: MSE or CE against MCTS root value?
6. Dynamics consistency loss still active? (ongoing calibration)
7. Which params are unfrozen? (policy + value + dynamics, encoder frozen or low LR)
8. Benchmark comparison:
   - model_trade_rate vs Phase 1 baseline (~18-22%)
   - agree_trade vs Phase 1 baseline (~42.7%)
   - Document: Phase 1 oracle rate ~2-3%, random ~0.3%
9. No loss divergence over 100 epochs?
10. Are MCTS targets generated on-policy? (using current params, not stale)
```

---

## 716g Audit — Alpha Hunt

```
Audit Stage B extended run for TASK_716g.

Read first:
- tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md (subtask 716g section)
- Recent benchmark JSON files

Check:
1. Has any benchmark period achieved positive alpha?
2. Model trade rate trend: is it decreasing from 18%? Target: 5-10%.
3. agree_trade trend: is it increasing from 42.7%?
4. beats_random: trending above 3/6?
5. Is the model becoming more selective? (fewer trades, better placed)
6. Loss curves stable? (no divergence, no mode collapse)
7. BC weight decayed appropriately? (should be near 0.005 by now)
8. Dynamics consistency still low? (< 0.05, not drifting)
9. Compare best Phase 2 checkpoint to best Phase 1 checkpoint on ALL benchmark metrics.
```
