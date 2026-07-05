# TASK_127c: Replay Buffer & Sampling Audit

## Findings

### 1. `UnifiedIndexBuffer` Architecture (`expert_buffer.py`)
-   **Structure:** Dual-zone design (Protected Demo + FIFO Agent) is implemented correctly and is highly memory-efficient.
-   **Stratified Sampling:** The buffer explicitly enforces a `demo_ratio` (default 12.5%) by sampling separately from the demo and agent zones. This effectively prevents the "agent flood" from drowning out expert demonstrations, which is a critical feature for effective imitation learning.

### 2. Importance Sampling (IS) Weight Mismatch (Theoretical Bug)
-   **Issue:** The Importance Sampling weights are calculated assuming **global** sampling probabilities:
    ```python
    probs = priorities / total_global_priority
    weights = (N * probs) ** (-beta)
    ```
-   **Reality:** The actual sampling is **stratified**.
    -   A demo sample $i$ is chosen with probability $P_{actual}(i) = \text{demo\_ratio} \times \frac{p_i}{\sum_{demo} p}$.
    -   The formula uses $P_{assumed}(i) = \frac{p_i}{\sum_{global} p}$.
-   **Impact:**
    -   If the demo zone has high average priority (likely, due to boost), $P_{assumed}$ might be close to $P_{actual}$.
    -   If the demo zone is small or low priority relative to the massive agent buffer, $P_{assumed}$ underestimates the sampling probability.
    -   Since $w \propto (1/P)^\beta$, underestimating $P$ leads to **overestimating** the weight.
    -   **Result:** Demo samples might have inflated gradients relative to their true "rarity" in the batch.
-   **Mitigation:** This arguably acts as a "hyper-prioritization" of demos, which might actually help learning in the early stages. However, strictly speaking, it violates the mathematical derivation of unbiased importance sampling.

### 3. Priority Calculation
-   **Demo Boost:** Demos get a static `demo_priority_boost` (default 0.001) added to their confidence score. This ensures they are never zero-priority.
-   **Agent Priority:** Based on `abs(td_error)`. Correct standard practice.

### 4. Batch Construction (`trainer.py`)
-   **Flow:** The trainer correctly requests a batch, which triggers the stratified sampling and on-demand observation reconstruction from the `PriceCache`.
-   **Starvation:** The FIFO eviction for the agent zone prevents starvation of new experiences. The protected demo zone ensures expert knowledge is never starved.

## Recommendations

1.  **Correct IS Weights (Optional):**
    -   To be mathematically rigorous, calculate weights based on the stratified probabilities:
        -   For demos: `prob = demo_ratio * (p_i / sum_demo_p)`
        -   For agent: `prob = (1 - demo_ratio) * (p_i / sum_agent_p)`
    -   This would normalize the gradient magnitudes between the two populations.

2.  **Monitor Demo Influence:**
    -   Add a DS24 metric for `avg_demo_weight` vs `avg_agent_weight` to see if the theoretical mismatch is causing massive scale differences in the loss.

## Conclusion
The replay buffer is robust and well-designed for the specific needs of "Causal Expert Imprinting" (TASK_125). The IS weight mismatch is a known side effect of combining stratified sampling with PER, but given the goal is to force-feed expert knowledge, the current implementation is likely practically effective even if theoretically slightly biased.
