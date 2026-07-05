# GLM-4.7 Ceiling Analysis (Run 004)

Model: GLM-4.7 via Z.AI
Timestamp: 2026-02-20 21:08:30
Prompt: ceiling_research_prompt.md + ambiguity_results.txt

---

Based on the system description, plateau evidence, and oracle ambiguity analysis, here is the detailed actionable analysis to break the 61% agreement ceiling.

### Executive Summary
The bottleneck appears to be **information compression** in the encoder feeding the reasoning network. The ambiguity analysis confirms the oracle is providing high-quality targets (94.5% confidence on trades), so the "ceiling" is not Oracle noise. The model is failing to extract sufficient signal from the noisy time-series input to feed the TRM. The 27-channel input is aggressively compressed into a 64-dim state, limiting the TRM's ability to distinguish subtle market regimes required for precise trade timing.

---

### Strategy 1: Architectural Overhaul (The "Fat Encoder")

**Hypothesis**
The V3 encoder (0.14M params) is the primary bottleneck. It compresses 27x64x64 inputs (110k values) into a 64-dim vector (~1700:1 compression) before the TRM sees it. The TRM is reasoning on "starved" data. Increasing the encoder's capacity (minGRU dim 32→64 or 128) and allowing it to output a higher-dimensional latent state (e.g., 128 or 256) will preserve more gradient information and temporal structure, allowing the TRM to perform effective logic.

**Expected impact**
**High**. This directly addresses the weak signal-to-noise ratio in trading data. A "Fat Encoder" acts as a high-quality feature extractor, which is the hardest part of time-series modeling.

**Implementation effort**
**Medium**. Requires modifying `trm_encoder_v3.py` (increase `hidden_dim`) and `muzero_model.py` (accept larger `state_dim`). No logic changes in the trainer.

**Risk**
- VRAM usage will spike. The current model fits 22.7GB/24GB. If encoder grows 4x, you might OOM at Batch=1024.
- Training time may increase per step.

**Priority**
**1 (Try First)**. This is the most likely source of the performance cap.

---

### Strategy 2: Hard Negative Mining (Focus on the "Margin")

**Hypothesis**
The model is "blind" to the specific conditions that trigger a trade because 96% of the data reinforces doing nothing. The 24× class weighting helps, but the model still optimizes for the global loss, which is dominated by holds. By specifically filtering the batch to include only "hard" examples—steps where the Oracle executed a trade OR steps where the Model predicted a trade but the Oracle held—you force the model to learn the boundary conditions between noise and signal.

**Expected impact**
**Medium**. Directly targets the 43% trade accuracy. If the model learns the "trade surface" accurately, balanced accuracy must rise.

**Implementation effort**
**Low**. Logic lives in `train_easy.py` / `easy_trainer.py`. You can implement a `BatchSampler` or a buffer that prioritizes high-loss samples or Oracle-trade timesteps.

**Risk**
- If the sampler is too aggressive, the model may forget the "Hold" baseline ( catastrophic forgetting).
- Validation metrics will be noisy if the val distribution doesn't match the train sampling.

**Priority**
**2**. Low effort, targets the specific symptom (low trade accuracy).

---

### Strategy 3: Cascaded Architecture (The "Switch")

**Hypothesis**
The current single-head network is trying to do two mutually exclusive things: detect *if* a trade should happen (classification) and *what* trade it is (action selection). These require different features. By splitting the network:
1.  **Head A (Binary)**: Classifies Trade vs. Hold (Sigomid).
2.  **Head B (Action)**: Classifies Action (5 classes) *only if* Head A > threshold.
This reduces the "Hold" signal from swamping the action logits. The 320-dimensional softmax is mathematically forced to split probability mass across 320 classes; a binary head concentrates the learning signal on the trigger condition.

**Expected impact**
**Medium**. This resolves the class imbalance structurally rather than via loss weighting.

**Implementation effort**
**Medium**. Requires architectural changes in `muzero_model.py` to add a dense "switch" head. The loss function in `easy_trainer.py` needs to handle the conditional logic (e.g., mask action loss if switch predicts hold).

**Risk**
- Complexity: Two heads must be balanced.
- If the "Switch" is wrong, the Action head doesn't learn.

**Priority**
**3**. Good alternative if simply scaling the encoder (Strategy 1) fails.

---

### Strategy 4: Real-Data Fine-Tuning (Closing the Sim2Real Gap)

**Hypothesis**
The model is hitting 61% on synthetic but only 48% on real data. The synthetic generator (Bella) might be missing "impossible" regimes or specific noise structures found in real markets. Fine-tuning the pre-trained synthetic model on a curated set of real historical data (even if sparse) will anchor the model to real market dynamics and prevent it from overfitting to synthetic artifacts. The fact that Val accuracy matches Train suggests the model has fully mastered the synthetic distribution—there is nothing left to learn there.

**Expected impact**
**Medium**. Critical for actual deployment, but might lower synthetic accuracy slightly initially (distribution shift).

**Implementation effort**
**Medium**. Requires a data pipeline for real data (from DB) and a secondary training phase in `easy_trainer.py`.

**Risk**
- Real data is unlabeled (no Oracle). You can only use Self-Supervised (predict next step) or weak supervision (Sharpe ratio on validation), which is noisier than the greedy oracle.

**Priority**
**4**. Essential for production, but perhaps not the immediate fix for the synthetic ceiling unless the synthetic data is confirmed to be "too easy".

---

### Strategy 5: Model Scaling (The "Gigantic" Config)

**Hypothesis**
The reasoning layer (TRM) needs more depth (H=6→8, L=8→10) to perform the complex lookahead simulation implied by the oracle's behavior. 14M parameters might be insufficient to memorize/imitate the complex state-value function of the 20-step lookahead oracle.

**Expected impact**
**Low**. The gradient norms are healthy (13-16), and train/val gap is tiny. The model is showing signs of *underfitting* due to capacity constraints OR *representation* limits, not lack of depth. Increasing depth of reasoning on a blurry (64-dim) state vector is inefficient. Better to widen the state (Strategy 1) than deepen the reasoner.

**Implementation effort**
**Low** (Config change in `easy_config.py`).
**Risk**
**High**. OOM on 24GB VRAM is almost guaranteed at Batch=1024 for H=10/L=12.

**Priority**
**5**. Only do this after VRAM optimization (mixed precision) and Strategy 1.

---

### Detailed Implementation Recommendation

**Immediate Path (To break 62%):**
1.  **Modify Encoder**: In `trm_encoder_v3.py`, increase the `minGRU` hidden dimension from 32 to **64**.
2.  **Widen State**: Increase output embedding to **128 dimensions**.
3.  **Adjust TRM**: Update the TRM config to accept `state_dim=128`.
    *   *Constraint Check*: 128-dim state with H=6, L=8.
    *   *VRAM Check*: Use `bfloat16` (mixed precision) in the trainer. This will roughly halve VRAM usage, allowing the larger encoder and state to fit in 24GB with Batch=1024.
4.  **Loss Tweaks**: Keep the 24× trade weighting but add a small `label_smoothing=0.1` to the trade targets to prevent the model from becoming overconfident on the 43% of trades it gets wrong.

**Secondary Path (If accuracy plateaus at 65-68%):**
Implement **Strategy 2 (Hard Negative Mining)**. Create a dataset sampler that ensures 30% of every batch consists of timesteps where the Oracle executed a trade (even if this means repeating samples).

**Analysis of Ambiguity Results:**
The oracle is highly confident (94.5% on trades). The theoretical ceiling of 97.1% balanced accuracy is **real and reachable**. The current gap of 36 percentage points is massive. This confirms the model is failing to *see* what the oracle sees. The oracle is not "guessing," it is seeing specific geometric patterns in the price action. The V3 encoder is likely blurring these patterns. The "Fat Encoder" is the right tool to sharpen them.