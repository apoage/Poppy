# TASK_127: MASTER BUG CATALOG

**Created**: 2026-01-31
**Status**: Cataloged - Pending Validation
**Total Unique Bugs**: 32

---

## BUG VALIDATION LEGEND

| Status | Meaning |
|--------|---------|
| 🔴 CONFIRMED | Bug verified in code, needs fix |
| 🟡 LIKELY | High probability bug, needs validation |
| 🟢 FALSE POSITIVE | Not a bug or by design |
| ⚪ UNVALIDATED | Not yet checked |

---

## CRITICAL BUGS (Training Breaking)

### BUG-001: BC Confidence Masking Dilutes Gradients
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127_training_audit, TASK_127d_REPLAY_BUFFER_SAMPLING_AUDIT
**Location**: `trainer.py:1420-1422`
**Issue**: BC loss multiplied by confidence then averaged with `jnp.mean()`, samples with confidence=0 included in denominator
```python
margin_loss = margin_loss * expert_confidence  # Zeros kill gradients
bc_loss_total = bc_weight * jnp.mean(margin_loss)  # Diluted by zero samples
```
**Fix**: Use masked mean: `jnp.sum(loss * mask) / jnp.sum(mask)`

---

### BUG-002: PER Importance Weights Computed But Never Applied
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127c_replay_buffer_audit, TASK_127d_REPLAY_BUFFER_SAMPLING_AUDIT, TASK_127k2_Replay_Buffer_Audit
**Location**: `expert_buffer.py:358-381` → `trainer.py`
**Issue**: Weights computed correctly but never used in loss computation
**Fix**: Apply `loss * batch['weights']` in trainer

---

### BUG-003: PopArt Weight Correction Outside JIT
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127e_POPART_NORMALIZATION_AUDIT, TASK_127f_popart_audit, TASK_127k3_PopArt_Audit
**Location**: `trainer.py:1597-1623`, `popart.py:130-157`
**Issue**: Weight correction happens in Python loop (every 10 iterations), outside JIT. Gradients computed on corrected params don't inform the correction.
**Fix**: Move inside JIT or disable weight correction (incompatible with categorical heads anyway - see BUG-004)

---

### BUG-004: PopArt Incompatible with Categorical Value Head
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k3_PopArt_Audit
**Location**: `popart.py:160-250`, `trainer.py`
**Issue**: PopArt weight correction assumes scalar outputs, but value head outputs categorical logits. Scaling logits doesn't preserve expected value.
**Fix**: Disable weight correction or switch to scalar value head

---

### BUG-005: MCTS Visit Counts Not Normalized Before Cross-Entropy
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127j_mcts_audit, TASK_127k_MCTS_SEARCH_INTEGRATION_AUDIT, TASK_127k5_MCTS_Search_Audit
**Location**: `self_play.py:219`, `trainer.py:1099-1103`
**Issue**: Raw visit counts passed to softmax_cross_entropy which expects probability distributions
```python
policy_probs.append(np.array(policy_jax[0]))  # Raw counts!
```
**Fix**: Normalize: `policy / (jnp.sum(policy) + 1e-8)`

---

### BUG-006: Missing MuZero Consistency Loss
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127a_muzero_audit, TASK_127b_MUZERO_DYNAMICS_AUDIT, TASK_127k1_MuZero_Dynamics_Audit
**Location**: `trainer.py:1079-1173`
**Issue**: Standard MuZero requires `||h(s') - dynamics(h(s), a)||²` to keep dynamics aligned with representation. Completely missing.
**Fix**: Add consistency loss in unrolling loop

---

### BUG-033: Alpha-Relative Scale Mismatch
**Status**: 🔴 CONFIRMED
**Sources**: BUG_MAYHEM_MASTER_REPORT
**Location**: `trainer.py`
**Issue**: 100x inflation/deflation of reward signals due to alpha-relative scaling mismatch
**Fix**: Align max_alpha scaling in trainer

---

### BUG-034: Phantom Rewards (Summing Past Termination)
**Status**: 🔴 CONFIRMED
**Sources**: BUG_MAYHEM_MASTER_REPORT
**Location**: `self_play.py`
**Issue**: Summing rewards past episode termination (penalizing dead agents)
**Fix**: Apply mask before summing episode score

---

## HIGH SEVERITY BUGS (Training Degraded)

### BUG-007: Demo vs Agent Priority Scale Mismatch
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127c_replay_buffer_audit, TASK_127d_REPLAY_BUFFER_SAMPLING_AUDIT, TASK_127k2_Replay_Buffer_Audit
**Location**: `expert_buffer.py:160-162, 199-201`
**Issue**: Demos use `confidence + 0.001` (~0.9), agents use `|td_error| + 1e-6` (~1-10). Agents have 10-100x more priority.
**Fix**: Use multiplicative boost or normalize scales

---

### BUG-008: Double Alpha Exponentiation in PER Weights
**Status**: 🟢 FALSE POSITIVE
**Sources**: TASK_127k2_Replay_Buffer_Audit
**Location**: `expert_buffer.py:231-232, 359`
**Issue**: Initially thought priorities stored with alpha applied, but verified:
```python
# Line 231-232: Raw priority stored, alpha computed but NOT stored
priority_alpha = priority ** self.per_alpha  # Only for tree update
self.indices[idx]['priority'] = priority      # RAW priority stored!
```
**Assessment**: Alpha applied CORRECTLY once during sampling (line 359). Not a bug.

---

### BUG-009: Demo Zone Uses Different Sampling Than Agent Zone
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k2_Replay_Buffer_Audit
**Location**: `expert_buffer.py:484-508`
**Issue**: Agent zone uses proper PER via `_sample_proportional`, demo zone uses `_sample_from_range` with `np.random.choice`
**Fix**: Use unified tree-based sampling

---

### BUG-010: Main Replay Buffer Only Samples Recent Data
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k2_Replay_Buffer_Audit
**Location**: `replay_buffer.py:255-267`
**Issue**: Heuristic window samples only from most recent ~0.2% of trajectories (1000 of 500k)
**Fix**: Mix recent (50%) and uniform (50%) sampling

---

### BUG-011: Weight Decay Disabled (Not Configured)
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127l_optimizer_audit, TASK_127m_LEARNING_RATE_OPTIMIZER_AUDIT, TASK_127k6_Optimizer_LR_Audit
**Location**: `config.py`, `trainer.py:553-556`
**Issue**: `weight_decay` parameter exists in trainer but never passed from config. Defaults to 0.
**Fix**: Add to config and pass to trainer

---

### BUG-012: Learning Rate Default Inconsistent
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k6_Optimizer_LR_Audit
**Location**: `trainer.py:367` vs `config.py:147`
**Issue**: Trainer default 2e-4, config default 1e-4 (2x difference)
**Fix**: Sync defaults

---

### BUG-013: Global Gradient Clip Too Permissive
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127l_optimizer_audit, TASK_127k6_Optimizer_LR_Audit
**Location**: `trainer.py:554`
**Issue**: Verified `optax.clip_by_global_norm(10.0)` - very permissive (typical: 1.0-5.0)
**Fix**: Reduce to 5.0 or enable per-head clipping

---

### BUG-014: Per-Head Gradient Clipping Disabled by Default
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k6_Optimizer_LR_Audit
**Location**: `trainer.py:399-404`
**Issue**: `per_head_clipping=False` by default, feature exists but unused
**Fix**: Enable by default

---

### BUG-015: Target Network Exists But Never Used
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127n_TARGET_NETWORK_AUDIT, TASK_127o_target_network_audit, TASK_127k7_Target_Network_Audit
**Location**: `trainer.py:251, 591, 1508-1519`
**Issue**: `target_params` defined, initialized, updated every 100 steps, but NEVER used in any computation
**Fix**: Either use for TD targets or remove (dead code/memory waste)

---

### BUG-016: Reward Scale May Exceed Value Support
**Status**: 🟢 FALSE POSITIVE
**Sources**: TASK_127k1_MuZero_Dynamics_Audit
**Location**: `trainer.py:28, 1154`, `__init__.py:51-53`
**Issue**: Initial concern about REWARD_SCALE=100 exceeding [-10, 10] support
**Assessment**: By design - comment in code explains:
- Per-step symlog rewards are tiny (~0.001 to 0.05)
- Scaled by 100 → range 0.1 to 5.0
- Fits well within [-10, +10] support
- Would only fail with extreme market events (not normal operation)

---

### BUG-017: No Gradient Blocking Between Dynamics and Representation
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k1_MuZero_Dynamics_Audit
**Location**: `muzero_model.py`
**Issue**: Verified - no `stop_gradient` found in muzero_model.py. Standard MuZero blocks gradients from dynamics to representation in unroll.
**Fix**: Add `jax.lax.stop_gradient()` on `next_latent_state`

---

### BUG-035: Burn Sign Error in Bearish Markets
**Status**: 🔴 CONFIRMED
**Sources**: BUG_MAYHEM_MASTER_REPORT, TASK_127_training_audit
**Location**: `jax_trading_env.py`
**Issue**: Being flat in bear market is rewarded as "burn" due to symlog logic. Agent learns wrong signal.
**Fix**: Use linear penalty or ensure symlog handles negative correctly

---

### BUG-036: Inconsistent Return Metric (Burn Not Reflected in Agent State)
**Status**: 🟢 FALSE POSITIVE
**Sources**: BUG_MAYHEM_MASTER_REPORT
**Location**: `jax_trading_env.py:1251-1271`
**Issue**: Initial concern that agent state doesn't reflect burn losses
**Assessment**: Verified - burns ARE correctly reflected:
- Cash reduced (line 1251-1252)
- Portfolio value updated (line 1268)
- Cumulative return recalculated (line 1271)

---

### BUG-037: Path Solver Transaction Costs Kill Oracle Performance
**Status**: 🔴 CONFIRMED
**Sources**: BUG_MAYHEM_MASTER_REPORT
**Location**: `path_solver.py`
**Issue**: Default 10bps transaction cost kills greedy oracle performance on minute data
**Fix**: Use 0 cost for oracle comparison or tune for data frequency

---

### BUG-038: Stratified IS Bias (Fixed 12.5% Demo Ratio Ignored)
**Status**: 🔴 CONFIRMED
**Sources**: BUG_MAYHEM_MASTER_REPORT
**Location**: `expert_buffer.py:302, 331, 370-371`
**Issue**: Verified - `demo_ratio = 0.125` hardcoded. IS weights use `N = total_available` without accounting for stratified sampling.
- If demos are 5% of buffer but sampled at 12.5%, they're over-sampled and IS weights are wrong
- Should adjust: `demo_prob = demo_ratio`, `agent_prob = 1 - demo_ratio`
**Fix**: Account for stratified sampling in weight computation

---

## MEDIUM SEVERITY BUGS (Suboptimal)

### BUG-018: jnp.where Evaluates Both Branches in Huber Loss
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127_training_audit, TASK_127a_muzero_audit
**Location**: `trainer.py:704, 927`
**Issue**: JAX evaluates both branches causing conflicting gradient signals
**Fix**: Use smooth Huber approximation

---

### BUG-019: Adaptive Entropy Bonus Discontinuity
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127g_ENTROPY_POLICY_COLLAPSE_AUDIT, TASK_127h_entropy_audit, TASK_127k4_Entropy_Policy_Collapse_Audit
**Location**: `trainer.py:1177-1181`
**Issue**: Entropy bonus jumps from 0 to 5x at threshold, gradient discontinuity
**Fix**: Use smooth function like `tanh(k * (min_entropy - entropy))`

---

### BUG-020: Entropy Coefficient Too Small
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k4_Entropy_Policy_Collapse_Audit, TASK_127k8_Loss_Function_Audit, TASK_127p_LOSS_WEIGHTING_AUDIT
**Location**: `trainer.py:1188`, `config.py:152`
**Issue**: Verified `entropy_coef=0.01` default - contributes only ~0.5% of total loss
**Fix**: Consider increasing to 0.05-0.1

---

### BUG-021: Learning Rate Schedule Constant by Default
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k6_Optimizer_LR_Audit, TASK_127l_optimizer_audit
**Location**: `config.py:148`
**Issue**: `lr_schedule="constant"` by default, no annealing
**Fix**: Change default to "cosine"

---

### BUG-022: Value Loss Weight Hardcoded to 2.0 in setup.py
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127p_LOSS_WEIGHTING_AUDIT
**Location**: `setup.py:510-511`
**Issue**: Overrides trainer default of 1.0, not configurable via CLI
**Fix**: Add CLI flag or document intentional 2x weighting

---

### BUG-023: Importance Sampling Weight Clipping Too Aggressive
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k2_Replay_Buffer_Audit
**Location**: `expert_buffer.py:381`
**Issue**: Verified `weights = np.clip(weights, 0.1, 10.0)` - aggressive clipping defeats purpose of bias correction
**Fix**: Widen range to [0.01, 100.0] or remove

---

### BUG-024: Loss Scale Imbalance (Policy/Reward 5-6x Value)
**Status**: 🟢 FALSE POSITIVE
**Sources**: TASK_127k8_Loss_Function_Audit
**Location**: `trainer.py:1183-1190`
**Issue**: Different loss types have different natural scales
**Assessment**: By design for MuZero - cross-entropy naturally larger than MSE

---

### BUG-025: Demo Priority Boost Applied Inconsistently
**Status**: 🔴 CONFIRMED
**Sources**: TASK_127k2_Replay_Buffer_Audit
**Location**: `expert_buffer.py:160-162` vs `518-519`
**Issue**: Verified different formulas - At add (line 161): `priority = confidence + boost`, at update (line 519): `priority = |td_error| + boost`. Different base values.
**Fix**: Use consistent priority formula

---

### BUG-026: No Temperature Annealing Schedule
**Status**: 🟢 FALSE POSITIVE
**Sources**: TASK_127k5_MCTS_Search_Audit, TASK_127k4_Entropy_Policy_Collapse_Audit
**Location**: `mctx_search.py:51`
**Issue**: Temperature parameter not used - verified codebase uses Gumbel MuZero which uses `gumbel_scale` parameter instead
**Assessment**: Temperature is for standard MuZero, this uses Gumbel variant with configurable `gumbel_scale`

---

### BUG-027: Confidence Threshold Creates Synthetic HOLD Policies
**Status**: 🟢 FALSE POSITIVE
**Sources**: TASK_127k5_MCTS_Search_Audit
**Location**: `self_play.py:167-182`
**Issue**: Low confidence forces HOLD with synthetic 90% HOLD policy
**Assessment**: Intentional - prevents random actions when uncertain

---

### BUG-039: Div-by-Zero Risk When unroll_steps=1
**Status**: 🔴 CONFIRMED
**Sources**: BUG_MAYHEM_MASTER_REPORT
**Location**: `trainer.py:1169`
**Issue**: Verified `reward_loss /= (unroll_steps - 1)` - crash if unroll_steps=1
**Fix**: Guard division: `reward_loss /= max(unroll_steps - 1, 1)` or enforce unroll_steps >= 2

---

### BUG-040: Path Solver Greedy vs Soft Normalization Mismatch
**Status**: 🟢 FALSE POSITIVE
**Sources**: BUG_MAYHEM_MASTER_REPORT
**Location**: `path_solver.py:117, 219`
**Issue**: Initial concern about normalization mismatch
**Assessment**: Verified both methods use consistent approach:
- `greedy_lookahead_path`: `total_alpha = cumulative_pnl[-1]` (line 117)
- `soft_topk_path`: `total_alpha=cumulative_pnl[-1]` (line 219)
Both use cumulative sum, no mismatch.

---

## LOW SEVERITY BUGS (Minor Issues)

### BUG-028: Dynamics State Normalization May Mask Problems
**Status**: ⚪ UNVALIDATED
**Sources**: TASK_127k1_MuZero_Dynamics_Audit
**Location**: `muzero_model.py:171-172`
**Issue**: LayerNorm on dynamics output may hide poor dynamics quality
**Assessment**: May be fine if consistency loss is added

---

### BUG-029: Documentation Shape Mismatch in Trainer
**Status**: 🟢 FALSE POSITIVE
**Sources**: Earlier observation audit
**Location**: `trainer.py:1382`
**Issue**: Comment says "(batch, 27, 60, 64)" but actual shape is "(batch, 27, 64, 64)"
**Assessment**: Comment typo, code is correct

---

### BUG-030: MCTS Simulations Count May Be Low
**Status**: ⚪ UNVALIDATED
**Sources**: TASK_127k5_MCTS_Search_Audit
**Location**: `mctx_search.py`
**Issue**: 32 simulations for 192 actions (~2 visits per considered action)
**Assessment**: Acceptable for trading, consider 64 if compute allows

---

### BUG-031: Training Hang at Epoch 90
**Status**: 🟢 FALSE POSITIVE
**Sources**: TASK_127r_hang_at_epoch_90
**Location**: JAX/XLA level
**Issue**: Training hung at epoch 91 for >20 minutes
**Assessment**: JAX/XLA deadlock, not logic bug. Fixed by restart.

---

### BUG-032: PopArt Stats Update Every 10 Iterations (Performance Opt)
**Status**: 🟢 FALSE POSITIVE
**Sources**: TASK_127k3_PopArt_Audit
**Location**: `trainer.py:1598`
**Issue**: PopArt stats updated every 10 iterations, not every iteration
**Assessment**: Intentional performance optimization, acceptable

---

## SUMMARY BY SEVERITY

| Severity | Count | Confirmed | Likely | False Positive | Unvalidated |
|----------|-------|-----------|--------|----------------|-------------|
| CRITICAL | 8 | 8 | 0 | 0 | 0 |
| HIGH | 15 | 13 | 0 | 2 | 0 |
| MEDIUM | 12 | 8 | 0 | 4 | 0 |
| LOW | 5 | 0 | 0 | 3 | 2 |
| **TOTAL** | **40** | **29** | **0** | **9** | **2** |

### Validation Session (2026-01-31)
**LIKELY → CONFIRMED:**
- BUG-013: Global gradient clip (10.0)
- BUG-017: No gradient blocking (stop_gradient missing)
- BUG-020: Entropy coefficient (0.01)
- BUG-023: IS weight clipping ([0.1, 10.0])
- BUG-025: Demo priority inconsistent
- BUG-038: Stratified IS bias
- BUG-039: Div-by-zero risk

**LIKELY → FALSE POSITIVE:**
- BUG-008: Double alpha (raw priority stored correctly)
- BUG-016: Reward scale (by design, symlog rewards tiny)
- BUG-026: Temperature annealing (uses Gumbel variant)
- BUG-036: Burn state (correctly reflected)
- BUG-040: Path solver normalization (consistent)

---

## PRIORITY FIX ORDER

### Phase 1: Critical (Stop the Bleeding) - 8 bugs
1. BUG-001: BC confidence masking
2. BUG-005: MCTS visit count normalization
3. BUG-002: Apply PER importance weights
4. BUG-003/004: Disable PopArt weight correction
5. BUG-033: Alpha-relative scale mismatch
6. BUG-034: Phantom rewards (mask past termination)
7. BUG-006: Add MuZero consistency loss (critical for world model)

### Phase 2: High (Core Fixes) - 15 bugs
8. BUG-007: Fix priority scale mismatch (demo/agent)
9. BUG-008: Remove double alpha exponentiation
10. BUG-009: Fix demo zone sampling (use segment tree)
11. BUG-010: Fix main replay buffer sampling
12. BUG-011: Enable weight decay
13. BUG-015: Remove or use target network
14. BUG-035: Burn sign error
15. BUG-037: Path solver transaction costs

### Phase 3: Medium (Polish) - 12 bugs
16. BUG-012: Sync LR defaults
17. BUG-013: Reduce global gradient clip
18. BUG-014: Enable per-head clipping
19. BUG-018: Smooth Huber loss
20. BUG-019: Smooth entropy bonus
21. BUG-020: Increase entropy coefficient
22. BUG-021: Change LR schedule default to cosine
23. BUG-022: Make value_loss_weight configurable

---

## NEXT STEPS

1. **Validate 🟡 LIKELY bugs** - Check code to confirm
2. **Prioritize Phase 1 fixes** - These compound and explain flat training
3. **Create fix PRs** - One per bug or grouped by file
4. **Retrain with fixes** - Compare to baseline

---

## FILES TO MODIFY

| File | Bug Count | Bugs |
|------|-----------|------|
| `trainer.py` | 14 | 001,002,003,014,015,018,019,020,021,022,033,039 |
| `expert_buffer.py` | 8 | 002,007,008,009,023,025,038 |
| `popart.py` | 2 | 003,004 |
| `self_play.py` | 3 | 005,034 |
| `muzero_model.py` | 2 | 016,017 |
| `replay_buffer.py` | 1 | 010 |
| `config.py` | 3 | 011,012,021 |
| `setup.py` | 1 | 022 |
| `mctx_search.py` | 1 | 026 |
| `jax_trading_env.py` | 2 | 035,036 |
| `path_solver.py` | 2 | 037,040 |

---

## CROSS-REFERENCE WITH ORIGINAL REPORTS

| Original Bug | Master Bug # | Status |
|--------------|--------------|--------|
| Alpha-Relative Scale Mismatch | BUG-033 | 🔴 CONFIRMED |
| BC Confidence Masking | BUG-001 | 🔴 CONFIRMED |
| Demo Sampling Ignores Priority | BUG-009 | 🔴 CONFIRMED |
| Double Alpha Exponentiation | BUG-008 | 🟢 FALSE POSITIVE |
| PER Weights Unused | BUG-002 | 🔴 CONFIRMED |
| PopArt Desync | BUG-003 | 🔴 CONFIRMED |
| PopArt Cat. Head Incompatibility | BUG-004 | 🔴 CONFIRMED |
| Raw Visit Counts | BUG-005 | 🔴 CONFIRMED |
| Phantom Rewards | BUG-034 | 🔴 CONFIRMED |
| Missing Consistency Loss | BUG-006 | 🔴 CONFIRMED |
| Burn Sign Error | BUG-035 | 🔴 CONFIRMED |
| Inconsistent Return Metric | BUG-036 | 🟢 FALSE POSITIVE |
| Path Solver Transaction Costs | BUG-037 | 🔴 CONFIRMED |
| Return Type Mismatch | (part of BUG-033) | 🔴 CONFIRMED |
| Stratified IS Bias | BUG-038 | 🔴 CONFIRMED |
| Weight Decay Missing | BUG-011 | 🔴 CONFIRMED |
| Priority Scale Mismatch | BUG-007 | 🔴 CONFIRMED |
| Unused Target Network | BUG-015 | 🔴 CONFIRMED |
| Div-by-Zero Risk | BUG-039 | 🔴 CONFIRMED |
| jnp.where Branching | BUG-018 | 🔴 CONFIRMED |
| Entropy Discontinuity | BUG-019 | 🔴 CONFIRMED |
| Path Solver Normalization | BUG-040 | 🟢 FALSE POSITIVE |
| JAX/XLA Deadlock | BUG-031 | 🟢 FALSE POSITIVE |
