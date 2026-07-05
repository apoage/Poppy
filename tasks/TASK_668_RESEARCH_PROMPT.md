# Deep Research Prompt: Dynamic Model Merging for Federated RL

## Context

I'm training a trading agent using **Expert Iteration** (behavioral cloning from oracle demonstrations). I run **two parallel training branches** on the same GPU with staggered timing:

- **Branch A (Bella)**: Trains on diffusion-generated synthetic market data (stochastic shapes, distributional robustness)
- **Branch B (Synthetic)**: Trains on rule-based curriculum data (mathematical patterns L0-L14, specific trading mechanics)

Both branches use the same architecture (MuZero-style transformer encoder + policy/value heads, ~15M params). I periodically merge their weights using Federated Averaging and continue training from the merged checkpoint.

**Goal**: Achieve >80% oracle agreement on BOTH data sources with a single unified model.

---

## Research Questions

### 1. Weight Merging Strategies

Compare these approaches for merging two neural network checkpoints trained on different data distributions:

1. **FedAvg**: Simple averaging `W = 0.5*A + 0.5*B`
2. **Performance-Weighted**: `W = (acc_A/(acc_A+acc_B))*A + (acc_B/(acc_A+acc_B))*B`
3. **FedProx**: FedAvg + regularization term toward previous global model
4. **Fisher Merging**: Weight by diagonal Fisher Information Matrix (importance-weighted)
5. **TIES-Merging**: Trim, Elect Sign, Disjoint Merge (handles sign conflicts)
6. **Model Soups**: Greedy selection from multiple checkpoints
7. **Git Re-Basin**: Permutation alignment before averaging (neural network symmetries)
8. **Layer-wise Merging**: Different merge ratios per layer (e.g., freeze encoder, blend heads)

For each, provide:
- Mathematical formulation
- Computational cost (memory, FLOPs)
- When it works best / failure modes
- JAX/Flax implementation sketch

### 2. Compatibility & Divergence Metrics

How do I measure if two checkpoints are "compatible" for merging?

Research these metrics:
1. **Weight Space**: Cosine similarity of flattened parameters
2. **Gradient Space**: Alignment of gradients on shared validation set
3. **Loss Landscape**: Linear interpolation loss (mode connectivity)
4. **Behavioral**: Agreement rate on held-out data (both predict same action)
5. **Representation**: CKA (Centered Kernel Alignment) on hidden activations

For each metric:
- How to compute efficiently in JAX
- Threshold values indicating "safe to merge" vs "too diverged"
- Correlation with actual merge success

### 3. Progressive Alignment Schedule

I want branches to explore independently early, then converge later:

| Phase | Training Progress | Required Alignment |
|-------|------------------|-------------------|
| Early | 0-30% | 0% (free exploration) |
| Mid | 30-70% | 20-60% (gradual) |
| Late | 70-95% | 60-80% (consistent) |
| Final | 95-100% | 85%+ (unified) |

Questions:
- Is this schedule optimal? What does literature suggest?
- How to enforce alignment (reject merges that decrease it? adjust LR?)
- Should alignment be measured in weight space or behavior space?

### 4. Merge Triggering Strategies

When should I merge? Options:
1. **Fixed schedule**: Every N epochs
2. **Performance-triggered**: When either branch improves by X%
3. **Divergence-triggered**: When branches drift apart (alignment drops below threshold)
4. **Plateau-triggered**: When neither improves for N epochs
5. **Adaptive**: Combination based on training phase

Which works best for my setup (two data sources, same architecture)?

### 5. Catastrophic Forgetting Prevention

After merging, branches continue training. Risk: merged model forgets what one branch learned.

Research:
- **EWC (Elastic Weight Consolidation)**: Regularize important weights
- **SI (Synaptic Intelligence)**: Online importance estimation
- **PackNet / Progressive Nets**: Freeze subnetworks
- **Experience Replay**: Mix old data during fine-tuning

Which is most practical for my federated setup?

### 6. Theoretical Foundations

- **Mode Connectivity**: Are the two minima connected by low-loss path?
- **Loss Landscape Geometry**: Sharpness, flatness, basin structure
- **Neural Network Symmetries**: Permutation invariance implications
- **Lottery Ticket Hypothesis**: Do branches find same subnetwork?

How do these affect merge success?

---

## Specific Implementation Questions

1. **Fisher Information in JAX**: How to efficiently compute diagonal Fisher for 15M param model?

2. **Git Re-Basin**: Is permutation alignment worth the cost for same-init branches?

3. **Layer-wise strategy**: Should I merge encoder less aggressively than policy head? (Intuition: encoder learns shared representations, heads specialize)

4. **Merge frequency vs magnitude**: Better to merge often with small changes, or rarely with big changes?

5. **Optimizer state**: Should I merge optimizer momentum/Adam state, or reset after merge?

---

## Desired Output Format

For each merging strategy, provide:

```
## [Strategy Name]

### Theory
[1-2 paragraph explanation]

### Formula
[Mathematical formulation]

### JAX Implementation
[Code sketch using JAX/Flax/Optax]

### Pros/Cons
- Pro: ...
- Con: ...

### Best For
[When to use this strategy]

### References
[Key papers]
```

For metrics, provide threshold recommendations and JAX implementation.

---

## Literature Starting Points

- McMahan et al., 2017 - "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- Li et al., 2020 - "Federated Optimization in Heterogeneous Networks" (FedProx)
- Wortsman et al., 2022 - "Model Soups: Averaging Weights of Multiple Fine-tuned Models"
- Ainsworth et al., 2022 - "Git Re-Basin: Merging Models modulo Permutation Symmetries"
- Yadav et al., 2023 - "TIES-Merging: Resolving Interference When Merging Models"
- Matena & Raffel, 2022 - "Merging Models with Fisher-Weighted Averaging"
- Kirkpatrick et al., 2017 - "Overcoming catastrophic forgetting in neural networks" (EWC)
- Frankle & Carlin, 2020 - "Linear Mode Connectivity and the Lottery Ticket Hypothesis"

---

## My Setup Summary

- **Framework**: JAX + Flax + Optax
- **Model**: MuZero-style (TRM encoder + policy/value heads), ~15M params
- **Task**: Trading agent, behavioral cloning from oracle
- **Branches**: 2 (Bella=diffusion shapes, Synthetic=math curriculum)
- **Hardware**: Single RTX 3090 (24GB), branches run interleaved
- **Current merge**: Simple FedAvg every 10 epochs
- **Target**: 80%+ oracle agreement on both data sources
