# TASK_681: Oracle Trade Density Ceiling — Dense Signal Oracle

**Status**: Research complete — awaiting design decision
**Priority**: CRITICAL (blocks all agreement progress)
**Depends on**: None (root cause identified)
**Blocks**: TASK_676, TASK_678, TASK_680 (all ineffective until this is resolved)

---

## Problem: The Constraint Conflict

Training has plateaued at **52.5% balanced agreement** across two full runs (620 + 210 epochs) despite:
- TASK_674: 6 plateau bug fixes
- TASK_675: 15 tech debt fixes
- TASK_677: Warmup disabled
- TASK_679: Threshold lowered (0.001 → 0.0005)
- TASK_680: Trade-biased sampling (15% coin-flip)

**None moved `oracle_hold_ratio` from 98.9%.** agree_trade = 8.9% (below random).

### Root Cause: Physics vs Portfolio Conflation

Current training conflates **Market Physics** (is an asset going up?) with **Portfolio Management** (is it the *best* asset out of 64?).

By enforcing `k=1` at the label level:
1. **Signal Sparsity**: Only 1/64 (1.56%) of assets get trade labels. 98.4% are labeled HOLD.
2. **Conflicting Guidance**: 63 assets might have identical bullish structures, but the oracle picks only one (due to the k constraint). The model sees 63 bullish patterns labeled "HOLD" — forcing the encoder to ignore valid market physics.
3. **Turtle Collapse**: The model learns all-HOLD because it correctly classifies 98% of data by doing nothing.

```
Maximum trade density = k / n_assets = 1/64 = 1.56%
Observed trade density = 1 - 0.989 = 1.1% (70% of ceiling — already near max)
Per-asset per-class signal: 1024 × 0.011 / 4 classes ≈ 2-3 examples per batch
```

Note: "2-3 examples per class per asset per batch" is very sparse, though not
strictly unlearnable given curriculum volatility variation, asset clustering, and
proper regularization. The signal is sparse enough to explain the plateau.

### Model Architecture Context

The model is NOT independent per-asset internally:
- **Encoder** (`trm_encoder_v2.py:67-72`): Cross-asset self-attention, assets attend to each other
- **TRM Reasoning** (`trm_flax.py:138`): 64-dim global state → Dense → 320 logits, fully joint
- **Output only** (`path_solver.py:772`): Per-asset 5-way softmax (normalization, not prediction)

The model can learn ranking and relative comparisons. Independent labels may underutilize this.

### Value Head Coupling

Value head trains on `remaining_pnl` (`train_easy.py:283`):
```python
remaining_pnl = cumulative_pnl[-1] - cumulative_pnl  # PnL from oracle's k=1 path
```
Any fix must account for value target consistency with MCTS downstream.

---

## Fix Options

### Option B: Increase oracle_k (k=4)

**Concept**: Let oracle select more assets per step. Trade density = k/64.

```
k=4:  ceiling = 4/64 = 6.25% → per-asset ~16 trade windows/batch → ~4-8 per class
k=8:  ceiling = 8/64 = 12.5%
```

**Implementation**: Single config change — `oracle_k: int = 4`.

**Pros**:
- Trivial (1 line), zero risk of breaking anything
- Clean gradient flow (all 64 assets supervised, no masking)
- MCTS policy/value stay coupled
- Multipass oracle automatically benefits (more selection diversity)
- 4× more trade diversity in actual data (different assets selected)
- Works synergistically with TASK_680 trade-biased sampling (15%)

**Cons**:
- 6.25% may not be enough (but combined with TASK_680 → viable)
- Value head trains on k=4 PnL, inference uses k=1 (see CRIT-8)
- Model may learn to spread bets rather than concentrate

**Mitigation for value mismatch**: Consider scaling value targets by k at training
time, or accepting the mild overestimation (value predicts "market opportunity"
rather than "single-trade PnL").

**Risk**: LOW

---

### Option D: Hybrid — Unconstrained Physics, Constrained Reality

**Concept**: Decouple policy oracle from value oracle.

1. **Unconstrained Policy Oracle**: Each asset evaluated independently. If forward return covers transaction costs + threshold → labeled BUY/SELL. Trade density ~10-15%.
2. **Constrained Value Oracle** (k=1): Existing `greedy_lookahead_path`. Value head learns achievable PnL from realistic k=1 strategy.
3. **Inference Recovery**: Policy head scores all assets; argmax naturally picks highest "physics confidence." k=1 execution recovered automatically.

```python
def generate_episode_data(returns, config):
    # Policy: unconstrained per-asset (market physics)
    independent_selections = unconstrained_oracle_policy(
        returns, lookahead=config.oracle_lookahead,
        min_alpha_threshold=config.policy_alpha_threshold,
        transaction_cost=config.transaction_cost,
    )
    policy_targets = optimal_path_to_policy(independent_selections, returns, ...)

    # Value: k=1 constrained (portfolio reality)
    k1_path = greedy_lookahead_path(returns, k=1, ...)
    remaining_pnl = k1_path.cumulative_pnl[-1] - k1_path.cumulative_pnl

    return policy_targets, remaining_pnl
```

**Pros**:
- 10× more trade examples, model learns actual market structure
- Model architecture unchanged
- Value head stays grounded in k=1 reality

**Cons**:
- Two oracle calls per episode (~2× oracle compute)
- New hyperparameter (policy_alpha_threshold) needs calibration
- See CRIT-2 and CRIT-4 in adversarial review

**Risk**: MEDIUM-HIGH

---

### Option F: Negative Sampling Loss Mask

**Concept**: Keep k=1 oracle exactly as-is. Instead of computing policy loss on all 64 assets, mask to only the selected asset (trade) plus N randomly sampled non-selected assets (holds).

```python
# In loss_fn (easy_trainer.py):
trade_mask = oracle_action_type != 1  # (batch, 64) — the ~1 trade per step
# Sample N holds per batch row as negative examples
neg_mask = sample_n_holds(hold_mask, n=N_NEGATIVES, rng=rng)
loss_mask = trade_mask | neg_mask  # (batch, 64)

per_asset_loss = softmax_cross_entropy(logits_5, targets_5)  # (batch, 64)
policy_loss = (per_asset_loss * loss_mask).sum() / loss_mask.sum()
```

With `N_NEGATIVES=4`: 1 trade + 4 holds = **~20% trade density in loss** (when
oracle produces exactly 1 trade; varies 0-33% depending on per-step trade count).

**Pros**:
- Zero oracle changes, zero value changes
- No threshold calibration
- ~15 lines in `easy_trainer.py`
- MCTS policy/value consistency preserved (though asymmetric — see CRIT-10)

**Cons**:
- Only ~1 real trade example per step (no new diversity, just less noise)
- Non-IID batch statistics from random hold selection (CRIT-7)
- Gradient starvation: 59/64 assets get zero gradient per step (CRIT-10)
- 20% ratio not guaranteed: 0 trades → 0%, 2 trades → 33% (CRIT-11)
- Asymmetric learning: policy masked, value sees all 64 assets

**Risk**: MEDIUM (gradient concerns)

---

### Other Options (lower priority)

**Option A: Per-Asset Independent Oracle** — Like Option D but uses independent PnL for value head too. Value becomes unrealizable (can't hold 10 assets with capital for 1). MCTS-incompatible.

**Option C: Reduce Training Assets** (n=8, k=1) — 12.5% density, k=1 preserved. But loses cross-asset correlation learning (the whole point of 64 assets). High transfer risk.

**Option E: Action Space Restructure** — Binary gate (trade/hold) + direction head. Cleanest formulation but requires architecture change. Better as a v2 approach.

---

## Adversarial Review — 15 Critical Issues

### Round 1 (2026-02-18, internal)

#### CRIT-1: Independent Labels May Waste Cross-Asset Attention

The model uses cross-asset self-attention in the encoder. Independent per-asset
labels (Options A/D) teach **scoring** (absolute tradability per asset), but the
model's strength is **cross-asset reasoning** (which asset is RELATIVELY best).

Independent labels may produce a model that says "10 assets should trade" with no
meaningful differentiation between them. At inference, top-1 from a flat
distribution is random noise.

**Severity**: HIGH for Options A/D. Does not affect B or F.

#### CRIT-2: Policy/Value Decoupling in Option D Breaks MCTS

Option D trains policy on ~10% density but value on k=1 PnL. In MCTS:
- Policy says: "Consider trading asset 5, 12, 37 (all high-confidence)"
- Value says: "Expected PnL = 0.03" (from single best trade, always same)

The value doesn't change based on which asset the policy picks — it reflects
the same k=1 path regardless. This severs the policy-value feedback loop.

**Severity**: HIGH for Option D. Does not affect B or F.

#### CRIT-3: Per-Class Per-Asset Density Is the Real Metric

1.1% trades across 64 assets × 4 trade classes = 2-3 examples per class per
asset per batch. k=4 gives ~4-8 per class (viable with curriculum variation
and clustering). **Option B alone may be sufficient.**

**Severity**: MEDIUM (affects prioritization).

#### CRIT-4: Threshold Calibration Across 15 Curriculum Levels (Options A/D)

| Level | Volatility | Independent oracle density |
|---|---|---|
| L0 (basic) | σ ≈ 0.005 | ~2% (sparse) |
| L8 (GARCH) | σ ≈ 0.02-0.05 | ~30% (flood) |
| L14 (micro) | σ ≈ 0.03+ | ~40%+ (everything trades) |

Single threshold produces wildly inconsistent density. Needs percentile-based
adaptive threshold — adds complexity not accounted for in Option D estimates.

**Severity**: MEDIUM for Options A/D. Does not affect B or F.

#### CRIT-5: Option B Is Undersold

B is trivial (1 line), has clean gradient flow, MCTS-safe, and 6.25% with
TASK_680's 15% biased sampling gives meaningful signal. Should be primary,
not validation.

**Severity**: HIGH (prioritization).

#### CRIT-6: 9.4% agree_trade Nuance

Trade classes are skewed (mostly Buy/Sell, few Limit*). Primary failure: model
predicts HOLD where it should trade (class confusion, not wrong direction).
Confirms density argument.

**Severity**: LOW (cosmetic).

---

### Round 2 (2026-02-18, external multi-model review)

#### CRIT-7: Option F Creates Non-IID Batch Statistics

Negative sampling randomly selects different hold positions per batch row:
- Non-stationary loss landscape: each batch sees a different asset subset
- Optimization instability: gradients vary wildly between batches
- No guarantee of fair asset coverage over finite training

**Mitigation**: Stratified sampling with per-asset counters ensuring full
coverage every N batches. Adds complexity to what was supposed to be "simple."

**Severity**: MEDIUM for Option F.

#### CRIT-8: Value Head Target Mismatch in Option B

With k=4 oracle, `remaining_pnl` comes from the optimal k=4 path. At inference
with k=1:
- Value overestimation: model expects PnL from 4 concurrent trades, executes 1
- Policy-value gap: policy learns 4-asset selection, value expects 4-way capital split

More severe than "mild mismatch" — especially for capital-constrained trading.

**Mitigation**: Scale value targets by 1/k, or accept overestimation and treat
value as "market opportunity estimate" rather than "realizable PnL."

**Severity**: MEDIUM for Option B.

#### CRIT-9: Cross-Asset Attention Impact Underanalyzed

| Option | Effect on Cross-Asset Attention |
|---|---|
| A/D (Independent) | Degrades to per-asset scoring, attention underutilized |
| B (k=4) | Model learns top-4 ranking, cross-asset reasoning preserved |
| F (Neg Sampling) | Model sees 1+4 assets/step, which 4 varies — partial cross-asset signal |

Option F's random masking gives the attention mechanism inconsistent peer
groups. The model may learn different cross-asset patterns per batch depending
on which holds are visible. Option B gives consistent full-asset visibility.

**Severity**: LOW (architectural depth).

#### CRIT-10: Gradient Starvation from Loss Masking (Option F)

Option F: 59/64 assets get zero gradient per batch step.
- Delayed learning: each asset only gets gradient ~1/5 of steps
- Asymmetric learning: policy loss masked (5 assets), value loss unmasked (64 assets)
- Risk of overfitting to the negative sampling pattern itself

The BERT analogy ("train on 15% of tokens, generalize") holds for language but may
not transfer to structured trading where each "token" (asset) has distinct learned
position embeddings and persistent market regimes.

**Severity**: MEDIUM for Option F.

#### CRIT-11: Curriculum Level Sensitivity Affects ALL Options

CRIT-4 flags threshold issues for A/D, but volatility-driven density variation
affects ALL options including B:
- Low volatility levels (L0-L2): Even k=4 may yield <3% trades
- High volatility levels (L8-L14): k=4 may yield >20% trades

The phased approach doesn't address per-level calibration. However, B is most
robust here since the oracle's greedy selection naturally adapts — more volatile
markets produce more above-threshold assets, and k=4 caps selection at 4 regardless.

**Severity**: MEDIUM (but B handles it best organically).

---

### Challenged Assumptions

**ASSUMPTION-1**: "2-3 examples per class per asset per batch is unlearnable."

Overstated. Curriculum varies volatility, assets have different profiles, and
correlation between assets means trades cluster. Small effective batch size can
work with proper regularization and curriculum pacing. But it IS sparse enough
to explain the 52.5% plateau.

**ASSUMPTION-2**: "Option F guarantees 20% trade ratio regardless of volatility."

Only holds when oracle produces exactly 1 trade per step. With k=1:
- 0 trades → ratio = 0/(0+4) = 0%
- 1 trade → ratio = 1/(1+4) = 20%
- 2 trades (rare, high vol) → ratio = 2/(2+4) = 33%

Not a guarantee — varies with oracle output.

**ASSUMPTION-3**: "MCTS policy/value consistency fully preserved for Option F."

Policy loss is masked (5 assets), value loss is unmasked (64 assets). This
creates asymmetric learning: the value head sees opportunities that the policy
head never gets gradient for. Minor but worth noting.

---

### Missing Considerations

**MISSING-1: Class Imbalance Within Trades.** Buy/Sell dominate; LimitBuy/LimitSell
are extremely rare. With k=4, all 4 selections could be Buy/Sell, leaving Limit
classes with 0 examples. The density fix helps trades overall but doesn't address
intra-trade class balance. (Acceptable for now — Limit orders are rare in practice.)

**MISSING-2: Concrete Success Criteria.** See Rollback Protocol below.

**MISSING-3: Rollback/Recovery Plan.** See Rollback Protocol below.

---

### Round 3 (2026-02-18, TASK_666 context review + historical audit)

#### CRIT-12: TASK_666 Original Design Assumed k=4

`TASK_666_IMPLEMENTATION_PLAN.md:179` defines `oracle_k: int = 4`. The entire
TASK_666 Easy Mode design was built around k=4 density. Implementation diverged
to k=1 after TASK_143/TASK_145 reverted a failed k=10 experiment. k=4 was never
actually tested — the revert went straight from k=10 to k=1, skipping the
moderate option that TASK_666 actually planned.

**Impact**: Option B (k=4) is not "a new idea" — it's the TASK_666 original spec.
We're debugging a deviation from the approved design.

#### CRIT-13: TASK_666 Data Pipeline Plans vs Reality

TASK_666 originally planned pre-generated data (`generate_oracle_data.py` →
`oracle_data.npz`). This was NEVER implemented — current code uses
`OnTheFlyGenerator` (on-the-fly VRAM generation per CLAUDE.md efficiency rules).

This means:
- The on-the-fly pipeline generates new oracle paths every epoch
- Changing k requires zero data pipeline changes (just a config value)
- Option F IS technically viable with current architecture (no pre-generated data to reprocess)

**Impact**: Lower barrier to all options. Option B especially trivial.

#### CRIT-14: Historical k=10 Failure (TASK_143/TASK_145) Informs k=4

TASK_143 identified that k=10 caused value overestimation: oracle picks 10
assets but model executes 1 trade/step. TASK_145 confirmed and reverted to k=1.

Key difference between k=10 (failed) and k=4 (proposed):
- k=10: 15.6% density, value trained on 10-asset portfolio PnL — massive overestimation
- k=4: 6.25% density, value trained on 4-asset portfolio PnL — moderate overestimation
- k=4 PnL ≈ 2-4× realistic k=1 PnL (vs k=10 ≈ 5-10× overestimation)
- CRIT-8 mitigation (scale value by 1/k) is tractable at k=4

**Impact**: k=4 is the sweet spot — enough density to learn, small enough value gap to manage.

#### CRIT-15: Option B Is the Only TASK_666-Compatible Path

TASK_666's dependency map (`TASK_666_DEPENDENCY_MAP.md`) references
`greedy_lookahead_path(returns, k=4)` throughout. Every downstream component
(curriculum, benchmarks, MCTS integration) was designed assuming k=4 density.

Options D and F introduce novel oracle mechanics not in any TASK_666 spec.
Option B restores the approved design. This is a restoration, not a new feature.

**Impact**: Conclusive — Option B is not just "lowest risk," it's the only option
that doesn't require replanning downstream tasks.

---

## Revised Comparison Matrix

| | Trade Density | Value OK? | MCTS Safe? | Gradient Stable? | Code | Risk |
|---|---|---|---|---|---|---|
| **B: k=4** | 6.25% (data) | Overestimated (CRIT-8) | YES | **YES** | 1 line | **Low** |
| **F: Neg Sampling** | 0-33% (varies) | YES | Asymmetric (A-3) | NO (CRIT-7,10) | ~15 lines | Medium |
| **B+F Combined** | 6.25% + masked | Overestimated | Asymmetric | NO | ~16 lines | Medium |
| **D: Hybrid** | ~10-15% | Decoupled (CRIT-2) | NO (CRIT-2) | YES | ~100 lines | High |

---

## Recommendation: B first (TASK_666 restoration), then B+F, then D (escalating)

### Phase 0: Option B alone — k=4 (5 min, 1 line) — TASK_666 design restoration
1. Set `oracle_k: int = 4` in easy_config.py
2. Save pre-phase checkpoint
3. Run 100 epochs with TASK_680 biased sampling still active
4. k=4 + 15% biased sampling gives clean signal with stable gradients
5. Historical note: k=10 failed (TASK_143/145), k=4 was TASK_666's designed value
5. **Success criteria**:
   - agree_trade > 20% (currently 8.9%)
   - balanced_agreement > 55% (currently 52.5%)
   - variance(agree_trade) over last 20 epochs < 5pp (stability)
   - agree_hold > 85% (no overtrade collapse)
6. **If success** → DONE, monitor longer
7. **If fail** → Phase 1

### Phase 1: Add Option F — Negative Sampling (30 min, ~15 lines)
1. Keep k=4 from Phase 0
2. Add loss masking with N_NEGATIVES=4 and stratified sampling (CRIT-7 mitigation)
3. Save pre-phase checkpoint
4. Run 100 epochs
5. **Success criteria**: same as Phase 0 but higher bar:
   - agree_trade > 30%
   - balanced_agreement > 58%
6. **If success** → DONE
7. **If fail AND training unstable (CRIT-7)** → revert F, keep B only
8. **If fail AND stable but plateau** → density is NOT the bottleneck → TASK_676 (encoder)

### Phase 2: Option D — Only if B+F plateau
1. Fix CRIT-2 (policy/value decoupling) first
2. Fix CRIT-4 (adaptive percentile threshold per level)
3. Implementation:
   a. Add `unconstrained_oracle_policy()` to `path_solver.py`
   b. Add `oracle_unconstrained: bool`, `oracle_unconstrained_threshold: float` to EasyConfig
   c. Update `_generate_episode_data` and `_generate_batch_episodes` in `train_easy.py`
   d. Call `greedy_lookahead_path(k=1)` for value target
   e. Call `unconstrained_oracle_policy()` for policy target
4. Verify `oracle_hold_ratio` drops to ~85-90%

---

## Rollback Protocol

1. **Checkpoint before each phase**: Save current best model before any changes
2. **Run 50 epochs minimum**: Shorter runs may not show convergence trends
3. **Compare rolling 20-epoch average** of agree_trade and balanced_agreement
4. **Revert criteria**:
   - agree_trade drops > 5pp below pre-phase baseline → revert
   - Training loss becomes unstable (>2× variance of pre-phase) → revert
   - agree_hold drops below 80% (overtrade collapse) → revert
5. **Document**: Record pre/post metrics for each phase in CSV log

---

## Testing Requirements

### Test 1: k=4 trade density
```python
path = greedy_lookahead_path(returns, k=4, ...)
policy = optimal_path_to_policy(path.selections, returns, ...)
policy_5 = policy.reshape(-1, 64, 5)
hold_ratio = float(jnp.mean(jnp.argmax(policy_5, axis=-1) == 1))
assert hold_ratio < 0.96, f"Hold ratio {hold_ratio:.3f} should be < 96% with k=4"
print(f"Trade density: {1 - hold_ratio:.1%} (target: ~6%)")
```

### Test 2: Negative sampling loss balance (Phase 1)
```python
loss_mask = compute_loss_mask(oracle_action_type, n_negatives=4, rng=rng)
trade_in_mask = jnp.sum(loss_mask & (oracle_action_type != 1))
total_in_mask = jnp.sum(loss_mask)
trade_ratio = float(trade_in_mask / jnp.maximum(total_in_mask, 1))
print(f"Trade ratio in loss: {trade_ratio:.1%} (target: ~15-25%)")
assert 0.10 < trade_ratio < 0.40
```

### Test 3: Stratified coverage (Phase 1)
```python
# Over 100 batches, each asset should appear in mask at least 10 times
asset_counts = jnp.zeros(64)
for _ in range(100):
    mask = compute_loss_mask(oracle_action_type, n_negatives=4, rng=rng)
    asset_counts += jnp.sum(mask, axis=0)
min_coverage = float(jnp.min(asset_counts))
assert min_coverage > 10, f"Asset coverage {min_coverage} too low — stratify sampling"
```

### Test 4: Value head consistency (Phase 2 only)
```python
k1_path = greedy_lookahead_path(returns, k=1, ...)
old_remaining = k1_path.cumulative_pnl[-1] - k1_path.cumulative_pnl
hybrid_remaining = compute_hybrid_value_targets(returns, config)
assert jnp.allclose(old_remaining, hybrid_remaining), "Value targets must match k=1 path"
```

### Test 5: Agree_trade improvement (post-training, all phases)
```python
# After 100 epochs with any fix:
assert metrics['agree_trade'] > 0.20, f"agree_trade should break 20%"
assert metrics['agree_hold'] > 0.80, f"agree_hold must stay above 80%"
assert metrics['oracle_agreement'] > 0.55, f"balanced should break 55%"
```

---

## Status

- [x] Root cause identified (density ceiling: k=1 × 64 = 1.56%)
- [x] Fix options researched (A-F)
- [x] Adversarial review round 1 (6 CRITs)
- [x] Adversarial review round 2 (5 more CRITs, 3 challenged assumptions, 3 missing items)
- [x] Adversarial review round 3 (4 CRITs from TASK_666 context + TASK_143/145 history)
- [ ] Phase 0: Set oracle_k=4, run 100 epochs
- [ ] Phase 1: Add negative sampling if needed
- [ ] Phase 2: Option D (hybrid) if needed
