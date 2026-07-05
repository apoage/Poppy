# Dynamic model merging for two-branch federated reinforcement learning

**Fisher-weighted merging with adaptive divergence-triggered scheduling is the theoretically optimal strategy for your setup, but simple averaging works surprisingly well when branches share initialization.** The strongest theoretical result in your favor: Frankle et al. (2020) proved that networks branching from a shared checkpoint remain linearly mode connected — meaning weight averaging produces a low-loss merged model without any permutation alignment. This holds because same-init branches explore the same loss basin, and a ~15M parameter transformer with residual connections has particularly favorable geometry. The practical path to >80% oracle agreement on both distributions combines periodic Fisher-weighted merges, SCAFFOLD-style drift correction, and experience replay across branches.

---

## The eight merging strategies, ranked for your setup

Your branches share initialization, architecture, and scale — a configuration that dramatically narrows the field. Here is each strategy's formulation, cost, and applicability.

**FedAvg** averages weights directly: θ_merged = 0.5·θ_A + 0.5·θ_B. Cost is O(|θ|) — **~15M multiply-adds, negligible compared to a single forward pass**. McMahan et al. (2017) proved O(1/T) convergence for strongly convex objectives, but Li et al. (2019) showed FedAvg with fixed learning rate converges to a solution Ω(η(E−1)) away from optimal on non-IID data. For your two-branch setup, FedAvg is the correct baseline. It works when branches haven't diverged far; it fails when local training runs for many epochs on heterogeneous data, causing "client drift" that destroys learned features through destructive weight-space interference.

**Fisher Merging** (Matena & Raffel, 2022) replaces uniform averaging with importance-weighted combination: θ\*(j) = [F_A(j)·θ_A(j) + F_B(j)·θ_B(j)] / [F_A(j) + F_B(j)], where F is the diagonal Fisher Information Matrix. This is derived from a Bayesian perspective using the Laplace approximation — each branch's posterior is N(θ_i, F_i⁻¹), and the merged model maximizes the joint posterior. Fisher merging **significantly outperforms isotropic averaging** because parameters important for Branch A's distribution get weighted toward A's values, and vice versa. Cost: **5× model memory** (~300MB for 15M params at float32) plus ~200 forward-backward passes to estimate Fisher. A critical shortcut: Li et al. (ICML 2025, "Squisher") showed Adam's second-moment accumulator v approximates the diagonal Fisher, making Fisher estimation **essentially free** if you already use Adam.

**FedProx** (Li et al., 2020) modifies the training objective, not the merge: each branch minimizes F_k(w) + (μ/2)·‖w − w_global‖². The proximal term constrains local models to stay near the global checkpoint, reducing drift before merge time. For highly heterogeneous data, FedProx improved accuracy by **up to 22% over FedAvg**. Optimal μ ranges from 0.001 to 1.0 — start with μ=0.01 for moderate heterogeneity. The critical caveat: FedProx is a training-time strategy. It cannot improve a post-hoc merge of already-trained checkpoints.

**TIES-Merging** (Yadav et al., NeurIPS 2023) operates on task vectors τ = θ − θ_init through three steps: Trim (keep top-20% parameters by magnitude, zero the rest), Elect Sign (majority vote resolves sign conflicts between branches), and Disjoint Merge (average only values aligned with the elected sign). Final model: θ_merged = θ_init + λ·τ_merged. The paper found **35–65% of parameters have sign conflicts** when merging two models — a problem simple averaging ignores entirely. TIES consistently outperforms simple averaging and task arithmetic in reproduction studies by ~2–5%. Cost is trivial: O(n log n) for sorting, under 1 second for 15M params.

**Task Arithmetic** (Ilharco et al., ICLR 2023) provides the cleanest framework for your same-init setup: θ_merged = θ_init + λ_A·τ_A + λ_B·τ_B, where τ = θ_finetuned − θ_init. When λ_A = λ_B = 0.5, this is algebraically identical to FedAvg. The power comes from tuning λ independently — setting λ_A = λ_B = 1.0 works when task vectors are approximately orthogonal (Ortiz-Jimenez et al., NeurIPS 2023 proved this in the NTK regime). Grid search λ ∈ {0.3, 0.5, 0.7, 1.0} on a validation set covering both distributions.

**Model Soups** (Wortsman et al., ICML 2022) uses greedy checkpoint selection: sort all checkpoints from both branches by validation performance, greedily add each to the running average if it improves validation score. This naturally selects the most compatible checkpoints. For CLIP ViT-L/14, greedy soup improved by **+0.5% on ImageNet and +1.8% on distribution shifts** over the best single model. Adapt this by collecting checkpoints over your training trajectory from both branches.

**Layer-wise Merging** uses per-layer coefficients: θ_merged(ℓ) = α_ℓ·θ_A(ℓ) + (1−α_ℓ)·θ_B(ℓ). AdaMerging (Yang et al., ICLR 2024) learns these coefficients through entropy minimization, achieving **11% improvement over uniform task arithmetic**. For your MuZero-style architecture: merge the encoder/backbone aggressively (α=0.5) since both branches share general state representations, but merge policy and value heads more conservatively using Fisher-weighted or learned coefficients.

**Git Re-Basin** (Ainsworth et al., ICLR 2023) finds permutation matrices that align neuron orderings before averaging. For same-init branches, the literature consensus is clear: **permutation alignment is unnecessary**. Frankle et al. directly showed same-init models are linearly mode connected without any permutation correction. Weight matching costs seconds for 15M params, but the expected benefit is near-zero. Only consider it if you observe a significant loss barrier (>5%) at the interpolation midpoint — which would be surprising given shared initialization.

**SLERP** (Spherical Linear Interpolation) interpolates along the hypersphere surface rather than the chord: SLERP(v₁, v₂; t) = sin((1−t)Ω)/sin(Ω)·v₁ + sin(tΩ)/sin(Ω)·v₂. It preserves weight norms that linear interpolation shrinks in high dimensions. Cost is nearly identical to linear interpolation. Git Re-Basin noted SLERP "performed slightly better than linear interpolation in some cases." Apply per-tensor. It's free to try.

---

## Measuring when two checkpoints can safely merge

Five divergence metrics span from cheap weight-space checks to expensive behavioral diagnostics. **The loss barrier at the interpolation midpoint is the single most predictive and cost-effective metric.**

**Weight-space cosine similarity** — cos(flatten(θ_A), flatten(θ_B)) — provides a fast global health check. Compute both globally and per-layer, since early layers stay aligned while later layers diverge faster (L-DAWA, ICCV 2023). For same-init branches, cosine similarity starts at 1.0 and decays with training. Values **above 0.95 indicate safe merging**; below 0.90 signals danger. However, high cosine similarity is necessary but not sufficient — nonlinear neural network behavior means two high-cosine models can still produce a high loss barrier.

**Loss barrier via linear interpolation** is the gold standard. Evaluate L(α·θ_A + (1−α)·θ_B) for α ∈ {0, 0.25, 0.5, 0.75, 1.0}. The barrier = max loss along the path minus the average endpoint loss. Frankle et al. (2020) found that same-init networks become linearly mode connected (barrier ≈ 0) after just **3% of training for ResNet-20/CIFAR-10 and 20% for ResNet-50/ImageNet**. A barrier below **1–2% of endpoint performance** means safe averaging; above 5% requires alignment techniques or more frequent merging. Just checking the midpoint (α=0.5) suffices for screening — the barrier is generally unimodal.

**Gradient alignment** — cos(∇L(θ_A; D_val), ∇L(θ_B; D_val)) — reveals whether both models are being pushed in the same direction by shared validation data. This captures information weight similarity misses: two nearby models might have opposing gradients on the validation set. Positive alignment ≥ **0.1–0.3** on shared evaluation data indicates beneficial merging. Compute on a small batch (256–512 samples) for efficiency.

**CKA (Centered Kernel Alignment)** from Kornblith et al. (2019) measures representation alignment: CKA(X, Y) = ‖Y^TX‖²_F / (‖X^TX‖_F · ‖Y^TY‖_F). Use linear CKA — Kornblith found it performs as well as RBF kernel CKA. For same-init branches, early layers show CKA **0.85–0.99** while later layers may drop to 0.6–0.8. CKA > **0.8** across all layers suggests compatible representations. Compute with minibatch CKA for streaming efficiency — seconds on GPU for 1000 examples across all layers.

**Behavioral agreement** — the fraction of held-out inputs where argmax(f_A(x)) == argmax(f_B(x)) — directly measures functional similarity. More refined: Jensen-Shannon divergence of output distributions. Behavioral agreement often exceeds 90% even when weight-space metrics look worrying, because models agree on easy examples. Lubana et al. (ICML 2023) showed that models lacking linear connectivity must be mechanistically dissimilar, connecting behavioral metrics to loss geometry.

---

## The optimal three-phase alignment schedule

The literature strongly supports a phased approach: free exploration → gradual alignment → unified convergence. This mirrors the theoretical finding that networks become "stable to SGD noise" early in training (Frankle et al., 2020), after which the optimization outcome is determined to a linearly connected region.

**Phase 1 (0–10% of training): Free exploration.** Merge every 50–100 steps with simple FedAvg. No alignment penalty. Both branches explore the loss landscape freely, building diverse representations. This parallels the "instability" period Frankle identified, where the network hasn't yet committed to a basin.

**Phase 2 (10–70% of training): Gradual alignment.** Increase merge interval to 200–500 steps. Add FedProx regularization (μ=0.01) to limit drift. Switch from FedAvg to Fisher-weighted merging for better knowledge preservation. Monitor the loss barrier — if it exceeds 2%, tighten the schedule. SCAFFOLD (Karimireddy et al., ICML 2020) is ideal here: it maintains control variates c_A, c_B that correct local gradient updates, eliminating drift bias **without constraining the solution**. SCAFFOLD achieves convergence rates independent of data heterogeneity and is trivial for two branches.

**Phase 3 (70–100% of training): Convergence.** Merge infrequently (500–1000 steps) or transition to a single unified model. Apply knowledge distillation at merge time if the loss barrier exceeds a soft threshold. Use SWA (Stochastic Weight Averaging) on the merged model — Izmailov et al. (2018) showed SWA finds the **center of flat loss regions**, producing better generalization than SGD which converges to the boundary. The merged model should increasingly train on mixed data from both distributions.

The literature favors **weight-space averaging as primary alignment** (cheap, effective for same-init) with **behavior-space corrections via distillation** as a secondary mechanism when divergence metrics indicate trouble. FedDF (Lin et al., NeurIPS 2020) demonstrated that distillation-based aggregation outperforms weight averaging under high non-IID conditions.

---

## When to trigger each merge: adaptive divergence monitoring

Fixed schedules waste resources when models converge slowly and risk disaster when they diverge quickly. **Divergence-triggered merging with adaptive intervals** is optimal for two same-init branches.

The recommended adaptive algorithm monitors three cheap diagnostics at regular intervals:

When weight cosine similarity exceeds 0.99 and the loss barrier is below 1%, models have barely diverged — increase the merge interval by 1.5× and apply simple averaging. When cosine similarity is above 0.95 and the barrier below 2%, normal divergence — merge by averaging and maintain the current interval. When cosine similarity drops below 0.95 or the barrier exceeds 2%, apply Fisher-weighted averaging plus a short distillation phase (10–50 steps), and decrease the interval by 25%. For severe divergence (cosine below 0.90 or barrier above 5%), use FedNova-normalized averaging (Wang et al., NeurIPS 2020) to prevent the more-diverged branch from dominating, reset the interval to minimum, and increase FedProx μ.

Start with K=200 steps between merge checks. For your single-GPU setup, communication cost is effectively zero — the merge operation takes milliseconds. **This means you can afford much more frequent merging** than standard federated learning scenarios, which is a significant advantage. Higher learning rates require more frequent merging (faster divergence), so adapt K inversely with the learning rate schedule.

---

## Preventing catastrophic forgetting after each merge

When branches resume training on their respective data after a merge, the merged model risks forgetting the other branch's knowledge. Four methods form a practical defense stack.

**Experience Replay is the primary defense** and is nearly free in your setup. MuZero already maintains game history buffers — after each merge, share or blend replay buffers so each branch trains on mixed data. Literature suggests **5–10% of training data** in the replay buffer suffices, with 25–50% replay in each mini-batch. CLEAR (Rolnick et al., 2019) showed that even modest replay prevents catastrophic forgetting in RL.

**EWC via Adam's v accumulator costs nothing additional.** The "Squisher" result (Li et al., ICML 2025) proved that Adam's second-moment estimate v serves as a functional Fisher approximation. After merging, the regularization loss becomes L_total = L_new(θ) + (λ/2)·Σ_i v_i·(θ_i − θ*_merged_i)². This penalizes changing parameters that were important for the merged model's performance, proportional to their historical gradient magnitude. You already store v as part of Adam's state — **no additional computation or memory required**.

**Knowledge distillation preserves soft policy/value knowledge.** Keep pre-merge snapshots of both branches as teachers. During continued training: L = L_task + α·KL(p_teacher_A ‖ p_student) + β·KL(p_teacher_B ‖ p_student). For the branch training on distribution A's data, weight Branch B's distillation more heavily (that's the knowledge at risk). Apply temperature scaling τ=2–4 for softer targets. Distill both policy logits (KL divergence) and value predictions (MSE) separately.

**Synaptic Intelligence (SI)** tracks per-parameter importance online during training: Ω_i = Σ_τ [g_i(τ)·Δθ_i(τ)] / [(Δθ_i)² + ξ]. Unlike EWC, SI requires no separate computation phase — it piggybacks on the training loop. SI is **recommended as the first-choice regularizer for continuous RL** (over EWC) because it doesn't require explicit task boundaries and adds negligible overhead.

PackNet and Progressive Networks are designed for sequential task learning, not parallel federated branches, and are **not recommended** for this paradigm.

---

## Theoretical foundations: why same-init branches are easy to merge

The strongest theoretical result underpinning your setup comes from **mode connectivity research**. Garipov et al. (2018) discovered that loss function optima are connected by simple curves with nearly constant accuracy. Draxler et al. (2018) strengthened this: using nudged elastic band algorithms, they found paths between minima that are "essentially flat." But these curved paths don't help merging — averaging follows a linear path.

Frankle et al. (2020) provided the critical result: networks sharing initialization are **linearly** mode connected. They showed that two copies of a network, branched from a shared checkpoint and trained with different SGD noise, converge to solutions connected by a low-loss linear path. This holds after just 3% of training for small networks and 20% for large ones. The barrier along the linear interpolation θ_t = t·θ_A + (1−t)·θ_B approaches zero. This directly predicts that your branches — same init, same architecture — should merge well via simple averaging, provided periodic synchronization prevents excessive drift.

The **permutation symmetry** question is settled for your case. Neural networks have exact permutation symmetries — swapping neurons in a hidden layer with all connected weights produces an identical function. The number of such symmetries is astronomical (a 3-layer MLP with 512 hidden units has ~10^1,200 permutations). Entezari et al. (2022) conjectured that modulo these symmetries, all SGD solutions lie in one basin. Ainsworth et al. (2023) confirmed this empirically. But for same-init branches, this is irrelevant: both branches start with identical neuron ordering, and SGD preserves approximate alignment. **Transformers have even more constrained permutation structure** due to residual connections (Verma et al., 2024), making alignment breaks less likely.

**SAM (Sharpness-Aware Minimization) improves mergeability** by pushing branches toward flatter minima with wider basins. FedSAM (Qu et al., 2022; Caldarola et al., 2022) combines SAM with federated averaging and substantially improves generalization of the merged model. One caveat from FedGloSS (CVPR 2025): local flatness doesn't guarantee global flatness. But for two branches with same init, local SAM is likely sufficient since both branches stay in the same basin.

The **Lottery Ticket Hypothesis** predicts that same-init branches will find substantially overlapping winning subnetworks. Morcos et al. (2019) showed winning ticket initializations generalize across datasets. For your MuZero transformer: the representation network likely activates largely the same subnetwork across both branches (shared state-processing features), while prediction heads may find partially divergent subnetworks. High subnetwork overlap favors simple averaging; partial divergence in heads motivates Fisher-weighted or layer-wise merging for those components.

---

## JAX/Flax implementation specifics

**Computing diagonal Fisher** is the most implementation-critical detail. The theoretically correct approach accumulates squared per-sample gradients: F_ii = E[(∂log p(y|x;θ)/∂θ_i)²]. In JAX, use `jax.vmap` over mini-batches of 16–32 samples (each gradient tree is ~60MB for 15M params), accumulate running mean over 200–500 samples total. But the **Squisher shortcut eliminates this entirely**: extract Fisher from Adam's v accumulator directly. In Optax, this is accessible via the optimizer state's nu field. This approximation was shown to match explicit Fisher computation across five applications.

**Optimizer state after merge** requires careful handling. Average the second-moment estimates v across branches — this serves dual purpose as a merged Fisher estimate and stabilizes resumed training (FedAdamW, Liu et al., 2025). Reset the first-moment m to zero — momentum vectors point in branch-specific directions, and averaging them produces a compromise that's poor for both. **Apply linear learning rate warmup for 200–500 steps** after each merge, since resetting momentum is equivalent to restarting optimization.

**Layer-wise merge strategy for the MuZero transformer**: merge embedding and early transformer blocks at α=0.5 (equal weighting) — these encode general, transferable features. For middle and late blocks, slight bias toward the stronger branch (α=0.5–0.6). For policy and value heads, use Fisher-weighted merging rather than uniform averaging, since these layers are most task-specific. After merge, apply layer-wise learning rate decay: higher rates for heads (readaptation) and lower for backbone (preservation), with a standard multiplier of 0.9^ℓ from top.

**Merge frequency for two branches on single GPU**: since communication cost is zero (just parameter copying in memory), merge often. Start at every **100–500 steps** and adapt based on divergence monitoring. Higher learning rates and more heterogeneous data → merge more frequently. The merge operation itself is O(|θ|) multiply-adds — milliseconds for 15M params. Fisher computation (if not using the Squisher shortcut) adds ~200 forward-backward passes per model, taking a few minutes.

---

## Conclusion

The theoretical landscape for your setup is unusually favorable. Same initialization guarantees linear mode connectivity (Frankle et al., 2020), eliminating the need for expensive permutation alignment (Git Re-Basin) or complex nonlinear merging paths. The **recommended strategy stack** is:

- **Training time**: FedProx (μ=0.01–0.1) + SAM for flatter minima + SCAFFOLD control variates for drift correction
- **Merge method**: Fisher-weighted averaging using Adam's v as the Fisher estimate (zero additional cost), with TIES-Merging as a secondary option for handling sign conflicts
- **Merge schedule**: Adaptive divergence-triggered, starting at K=200 steps, monitoring the midpoint loss barrier as the primary trigger
- **Anti-forgetting**: Experience replay from shared buffers (primary) + EWC via Adam's v (free) + distillation from pre-merge snapshots for policy/value heads
- **Layer strategy**: Equal-weight backbone, Fisher-weighted heads, learning rate warmup after each merge

The path to >80% oracle agreement on both distributions is achievable: the shared initialization keeps branches in the same loss basin, Fisher-weighted merging preserves distribution-specific knowledge in the parameters that matter most, and experience replay prevents post-merge forgetting. The key risk factor is allowing branches to diverge too far between merges — monitor the loss barrier and tighten the schedule if it exceeds 2%.