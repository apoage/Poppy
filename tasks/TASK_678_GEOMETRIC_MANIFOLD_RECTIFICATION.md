# TASK_678: Geometric Manifold Rectification for Imbalanced Oracle Training

**Status**: Research
**Priority**: Medium (promising, post-streaming-encoder)
**Paper**: `docs/arXiv-2602.13045v1` — Wang, Li, Jia (Feb 2026)

---

## Motivation

Poppy's training has a **63:1 HOLD-to-TRADE imbalance**. We've attacked this through:

| Fix | Layer | What it does |
|-----|-------|-------------|
| TASK_674 | Data | Buffer refresh 17× more often |
| TASK_677 | Gradient | Continuous trade weighting, MAX=64, warmup=0 |
| Multipass oracle | Labels | Soft targets for ambiguous decisions |

**What's missing**: All current weighting is based on the **oracle label** (trade_strength).
Not all samples with the same label are equally informative:

- HOLD during flat market = boring noise (should downweight)
- HOLD during a near-miss trade signal = critical boundary sample (should upweight or flip)
- TRADE in a clear trend = easy, high-confidence (diminishing returns)
- TRADE in an ambiguous zone = hard, low-confidence (most informative)

GMR provides the **geometric** signal we're missing — where a sample sits relative to
its class manifold, not just what label the oracle gave it.

---

## Paper Summary (GMR)

**Core idea**: Majority class (HOLD) topologically intrudes into minority manifold (TRADE),
obscuring the decision boundary. Fix by measuring geometric confidence per sample and
cleaning asymmetrically.

### Algorithm

1. **Geometric confidence** via inverse-distance weighted kNN:
   ```
   w_j = 1 / (d(x_i, x_j) + eps)      # Closer neighbors count more
   w_j = w_j / sum(w)                    # Normalize
   conf(x_i) = sum(w_j * I(y_j == y_i)) # Weighted vote for own class
   ```

2. **Asymmetric cleaning**:
   - Majority (HOLD): Remove if misclassified by kNN OR conf < α (aggressive, α=0.3)
   - Minority (TRADE): Remove only if misclassified AND deeply embedded in majority (β=0.7)
   - Safe-guard cap: Never remove more than γ=10% of minority samples

3. **Adaptive metric**: Euclidean for d≤100, cosine for d>100 (distance concentration)

### Results
- Rank 4.22/20 across 27 datasets × 7 classifiers
- Best on Decision Tree (1.00), kNN (2.19), Logistic Regression (2.89)
- Classifier-agnostic — single preprocessing step benefits diverse learners

---

## Adversarial Review (2026-02-17)

Four critical flaws identified in the initial approach. All addressed below.

### CRIT-1: Latent Collapse Feedback Loop

**Problem**: Online GMR in latent space creates a death spiral early in training.
- Turtle encoder clusters everything as HOLD in latent space
- GMR sees TRADEs as "misclassified" (low geometric confidence)
- Loss on TRADEs gets downweighted → encoder learns even less about TRADEs
- Latent space collapses further into HOLD-only manifold

**Resolution**: Do NOT use model latents for GMR. Use **raw log-returns** (price
structure) as the distance metric. These are model-independent, stable from epoch 0,
and capture the actual market geometry we care about.

### CRIT-2: Within-Batch kNN Sparsity

**Problem**: With 63:1 imbalance, a single batch has ~4 TRADE samples per asset.
For a specific asset like BTC, you might have **zero** TRADE neighbors. kNN is
forced to pick HOLD neighbors → confidence = 0.0 for every trade → all trades
get cleaned or downweighted.

**Resolution**: Do NOT compute kNN within a training batch. Compute it **per-episode
in the buffer** (T=60 timesteps × structure similarity across assets). An episode
has enough structure for k=7-15 meaningful neighbors when comparing log-return
curves across timesteps.

### CRIT-3: Soft Target Contradiction

**Problem**: GMR assumes discrete labels (y ∈ {0, 1}). Multipass oracle produces
soft targets (e.g., 40% BUY, 60% HOLD). "Misclassified" and "same class" are
undefined for continuous distributions.

**Resolution**: Replace discrete voting with **KL-divergence consistency**:
```python
# For each (timestep, asset), compare its soft target to neighbors' soft targets
# High KL-divergence to neighbors = geometric outlier (conflicting labels on similar structure)
kl_div = sum(p_i * log(p_i / p_neighbor))  # Per neighbor, then weighted average
geo_outlier_score = weighted_mean_kl        # High = inconsistent with neighbors
```

This is a "Softer GMR" — instead of asking "is this sample misclassified?", ask
"does this sample's label distribution conflict with its structural neighbors?"

### CRIT-4: Computational Overhead in JIT Loop

**Problem**: Full distance matrix for (B × A) = 16,384 points is 268M entries (1 GB
float32). Computing this inside every training step drops throughput from 3,000 to
<100 steps/sec.

**Resolution**: GMR runs **offline in the OnTheFlyGenerator background thread**, NOT
in the JIT training loop. Cost is amortized over buffer lifetime (~17 batches per
episode). The generator already runs on CPU in parallel with GPU training.

---

## Revised Application: Buffer-Level Structure-GMR

The only viable approach is **offline buffer cleaning using raw price structure**.

### Architecture

```
OnTheFlyGenerator._refill_buffer():
  1. Generate episode (prices, oracle labels)      # Existing
  2. Extract log-return curves per asset            # (T, A) matrix
  3. Compute structure similarity (cosine on returns)
  4. kNN per (timestep, asset) using return structure
  5. KL-divergence consistency of soft targets vs neighbors
  6. Asymmetric rectification:
     - High-KL HOLDs near TRADE structure → soften toward TRADE
     - High-KL TRADEs deeply in HOLD structure → flag but protect (γ cap)
     - Low-KL everything → keep as-is (consistent)
  7. Store rectified labels in buffer
```

### Concrete Implementation

```python
def rectify_episode_gmr(episode, k=7, alpha=0.3, beta=0.7, gamma=0.1):
    """
    Offline GMR on a single episode using raw price structure.
    Runs on CPU in background thread. Cost: ~1ms per episode.
    """
    returns = episode['log_returns']   # (T, A) - raw log returns
    soft_targets = episode['actions']  # (T, A, 5) - multipass soft targets

    T, A = returns.shape

    # Per-asset: compare timesteps by return structure
    # Feature vector per (t, a): local return window [t-3:t+3] → (7,) or similar
    for a in range(A):
        features = returns[:, a]  # (T,) — could expand to local window

        # kNN over timesteps for this asset
        # Cosine distance on return structure
        dists = cdist(features.reshape(T, -1), metric='cosine')  # (T, T)
        neighbors = np.argsort(dists, axis=1)[:, 1:k+1]          # (T, k)
        neighbor_dists = np.take_along_axis(dists, neighbors, axis=1)

        # Inverse-distance weights
        weights = 1.0 / (neighbor_dists + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)    # (T, k)

        # KL-divergence consistency
        for t in range(T):
            p = soft_targets[t, a]          # (5,) this sample's soft target
            neighbor_ps = soft_targets[neighbors[t], a]  # (k, 5)

            # Weighted mean KL-divergence to neighbors
            kl_divs = np.sum(p * np.log(np.maximum(p, 1e-8) /
                             np.maximum(neighbor_ps, 1e-8)), axis=1)  # (k,)
            geo_outlier = np.sum(weights[t] * kl_divs)

            hold_prob = soft_targets[t, a, 1]  # HOLD probability

            # Asymmetric rectification:
            if hold_prob > 0.5 and geo_outlier > alpha:
                # High-KL HOLD near TRADE structure → soften toward neighbor consensus
                neighbor_consensus = np.sum(
                    weights[t, :, None] * neighbor_ps, axis=0)  # (5,)
                # Blend: move partially toward neighbors
                blend = min(geo_outlier, 0.5)  # Cap blend at 50%
                soft_targets[t, a] = (1 - blend) * p + blend * neighbor_consensus

            elif hold_prob < 0.5 and geo_outlier > beta:
                # High-KL TRADE deeply in HOLD structure → flag but protect
                n_flagged_trades += 1
                if n_flagged_trades <= gamma * n_total_trades:
                    # Only clean up to γ cap
                    neighbor_consensus = np.sum(
                        weights[t, :, None] * neighbor_ps, axis=0)
                    blend = min(geo_outlier - beta, 0.3)
                    soft_targets[t, a] = (1 - blend) * p + blend * neighbor_consensus

    episode['actions'] = soft_targets
    return episode
```

### What This Achieves

Instead of flipping or removing labels (hard GMR), we **soften toward neighbor consensus**:

- Two episodes with identical price structure but conflicting oracle labels → both get
  softened toward the average (reduces noise)
- HOLD in a clear flat market with all-HOLD neighbors → untouched (consistent, low KL)
- HOLD right before a trade signal with mixed neighbors → softened toward trade
  (the oracle was probably wrong to HOLD here)
- TRADE in an obvious trend with all-TRADE neighbors → untouched (clear signal)

### Scope: Level 5-14 Only

GMR on Level 0 (pure noise) is meaningless — there's no manifold structure to rectify.
GMR is most powerful on levels with real market patterns (trends, mean-reversion,
momentum) where the oracle's boundary decisions are genuinely ambiguous.

```python
if episode['curriculum_level'] < 5:
    return episode  # Skip GMR — no structure to rectify
```

---

## Hyperparameters (Revised)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| k (neighbors) | 7 | T=60 per asset, need k < T/4 for stability |
| α (HOLD outlier threshold) | 0.3 | KL > 0.3 nats = significant label conflict |
| β (TRADE outlier threshold) | 0.7 | Only clean TRADEs when deeply inconsistent |
| γ (TRADE protection cap) | 0.1 | Never modify more than 10% of TRADE labels |
| blend cap | 0.5 (HOLD), 0.3 (TRADE) | Asymmetric — more aggressive on HOLD, conservative on TRADE |
| feature window | 7 (t-3:t+3) | Local return context for structural similarity |
| metric | cosine | Natural for return direction similarity |
| min curriculum level | 5 | No structure to rectify below L5 |

---

## Implementation Plan (Revised)

### Phase 1: Structure Similarity Analysis (Research)

1. For 100 episodes, compute pairwise cosine distance on log-returns per asset
2. Visualize: do structurally similar timesteps get similar oracle labels?
3. Measure baseline KL-divergence distribution — what's "normal" vs "outlier"?
4. Set α, β thresholds empirically from the distribution

### Phase 2: Buffer Cleaning Integration

5. Implement `rectify_episode_gmr()` in `OnTheFlyGenerator`
6. Call during `_refill_buffer()`, runs on CPU background thread
7. Log metrics: `n_hold_rectified`, `n_trade_flagged`, `avg_kl_divergence`
8. A/B test: GMR ON vs OFF, compare oracle_agreement and convergence speed

### Phase 3: Curriculum-Aware GMR

9. Different α/β per curriculum level (L5 stricter, L14 more conservative)
10. Geometric confidence as curriculum difficulty score
11. Feed into e-dojo level selection

---

## Key Equations (Reference)

**KL-divergence consistency** (replaces discrete voting for soft targets):
$$\text{Outlier}(x_i) = \sum_{j \in N_k(x_i)} w_j \cdot D_{KL}(p_i \| p_j), \quad w_j = \frac{1/(d_{ij} + \epsilon)}{\sum_{l \in N_k} 1/(d_{il} + \epsilon)}$$

**Asymmetric soft rectification** (replaces hard removal):
$$\tilde{p}_i = (1 - \lambda) \cdot p_i + \lambda \cdot \bar{p}_{N_k}, \quad \lambda = \min(\text{Outlier}(x_i), \lambda_{\max})$$

Where $\bar{p}_{N_k} = \sum_j w_j \cdot p_j$ is the weighted neighbor consensus and
$\lambda_{\max}$ is 0.5 for HOLD, 0.3 for TRADE (asymmetric blending).

**Asymmetric risk**:
$$R(f) = C_{01} \cdot P(Y{=}1) \cdot P(\hat{Y}{=}0|Y{=}1) + C_{10} \cdot P(Y{=}0) \cdot P(\hat{Y}{=}1|Y{=}0)$$

Where $C_{01} \gg C_{10}$ — missing a TRADE is worse than false-trading a HOLD.

---

## Connection to Existing Work

| Component | How GMR relates |
|-----------|----------------|
| TASK_677 trade weighting | GMR is orthogonal — cleans labels, TASK_677 rebalances gradients |
| Multipass oracle soft targets | GMR validates soft targets against structural neighbors via KL-div |
| TASK_676 streaming encoder | Better temporal features → better structure distance → better GMR |
| L0-L14 curriculum | GMR only applies L5+ where manifold structure exists |
| Value head | Consistent labels → less noisy value targets → better value head |

---

## Rejected Approaches

### ~~Approach 1: Online Latent-Space GMR~~ — REJECTED

**Why**: Latent collapse feedback loop (CRIT-1). Early training has random/turtle
latent space. GMR would downweight the very TRADE samples the model needs to learn.
Also: 16K×16K distance matrix in JIT loop is 1GB, kills throughput (CRIT-4).

### ~~Within-Batch kNN~~ — REJECTED

**Why**: 63:1 imbalance means most assets have zero TRADE neighbors in a batch (CRIT-2).
kNN forced to pick HOLD neighbors → geometric confidence meaningless.

### ~~Discrete Label Voting~~ — REJECTED

**Why**: Multipass oracle produces soft targets. Discrete "same class" voting is
undefined for continuous distributions (CRIT-3). KL-divergence is the natural
generalization.

---

## Testing Requirements

**Test every function and feature immediately after implementation.**

### Test 1: KL-divergence computation
```python
# Verify KL-div is 0 for identical distributions, positive for different
p = np.array([0.0, 0.6, 0.0, 0.2, 0.2])  # Soft target
q_same = p.copy()
q_diff = np.array([0.0, 0.1, 0.7, 0.1, 0.1])

kl_same = np.sum(p * np.log(np.maximum(p, 1e-8) / np.maximum(q_same, 1e-8)))
kl_diff = np.sum(p * np.log(np.maximum(p, 1e-8) / np.maximum(q_diff, 1e-8)))

assert abs(kl_same) < 1e-6, f"KL(p||p) should be 0, got {kl_same}"
assert kl_diff > 0.1, f"KL(p||q) should be positive, got {kl_diff}"
```

### Test 2: Cosine distance on returns
```python
# Identical return curves → distance 0. Opposite → distance ~2.0
returns_a = np.array([0.01, 0.02, -0.01, 0.03])
returns_b = returns_a.copy()
returns_c = -returns_a

from scipy.spatial.distance import cosine
assert cosine(returns_a, returns_b) < 1e-6, "Same returns → 0 distance"
assert cosine(returns_a, returns_c) > 1.5, "Opposite returns → high distance"
```

### Test 3: Asymmetric rectification preserves trade labels
```python
# With γ=0.1 cap, at most 10% of TRADEs should be modified
episode = generate_test_episode(T=60, A=64, trade_ratio=0.03)
original_trades = count_trades(episode['actions'])
rectified = rectify_episode_gmr(episode, gamma=0.1)
modified_trades = count_modified_trades(episode['actions'], rectified['actions'])

assert modified_trades <= 0.1 * original_trades + 1, \
    f"Modified {modified_trades} trades but cap is {0.1 * original_trades}"
```

### Test 4: Low-level curriculum bypass
```python
# Episodes below L5 should pass through unchanged
episode = generate_test_episode(curriculum_level=2)
original = episode['actions'].copy()
rectified = rectify_episode_gmr(episode)
assert np.array_equal(original, rectified['actions']), "L2 should be untouched"
```

### Test 5: Rectification impact on label distribution
```python
# Run GMR on 20 episodes, verify:
# - Mean hold ratio stays within 5% of original (not destroying balance)
# - Some HOLDs were softened (non-zero rectification)
# - No NaN/Inf in rectified labels
# - All probability distributions still sum to 1.0
for ep in episodes:
    rectified = rectify_episode_gmr(ep)
    actions = rectified['actions'].reshape(-1, 64, 5)
    assert np.allclose(actions.sum(axis=-1), 1.0, atol=1e-5), "Probs must sum to 1"
    assert not np.any(np.isnan(actions)), "No NaNs"
```

### Test 6: A/B training comparison (10-epoch mini-run)
```python
# Same config, same seeds. GMR ON vs OFF. Compare:
# - oracle_agreement (should improve or be neutral)
# - agree_trade (primary metric — should improve)
# - convergence speed (epochs to 55% agreement)
```

---

## Status

- [x] Paper review complete
- [x] Adversarial review — 4 critical flaws identified and addressed
- [ ] Phase 1: Structure similarity analysis (research prototype)
- [ ] Phase 1: Tests 1-2 pass
- [ ] Phase 2: Buffer cleaning integration
- [ ] Phase 2: Tests 3-5 pass
- [ ] Phase 2: Test 6 (A/B comparison)
- [ ] Phase 3: Curriculum-aware GMR
