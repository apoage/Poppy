# Per-Step "Mark-to-Market" Rewards for Trading Agents

## Concept
Standard RL for trading often uses sparse rewards (e.g., realized PnL at trade exit or total return at episode end). This can make credit assignment difficult: the agent doesn't know *which* action in a 500-step sequence caused the final loss.

**Per-Step Mark-to-Market (MtM) Reward** addresses this by providing immediate feedback at every timestep based on the change in the **Total Portfolio Value** (Cash + Unrealized Value of Assets).

**Formula:**
$$ R_t = \frac{V_t - V_{t-1}}{V_{t-1}} \times 100 $$
Where $V_t$ is the total portfolio value at step $t$.

## Pros (Why consider it for final tuning?)

1.  **Dense Reward Signal:** The agent receives feedback for *every* action. If it buys an asset and the price goes up in the next minute, it gets a positive reward immediately. This eliminates the "sparse reward" problem.
2.  **Encourages Activity:** Since holding cash yields $R_t = 0$ (assuming no interest), the agent is naturally incentivized to find assets with positive expected returns to beat the baseline.
3.  **Smoother Value Function:** The value function $V(s)$ becomes a smoother approximation of the cumulative future returns, as the variance of the target is distributed across many small steps rather than lumped into huge terminal jumps.
4.  **Truthful Valuation:** It reflects the accounting reality that unrealized gains *are* real economic value (until prices change).

## Cons (Why it's risky for initial training)

1.  **Myopia (Short-termism):** The agent might learn to chase instantaneous noise or volatility. It could buy an asset that ticks up for one second, getting a +0.01% reward, even if the long-term trend is negative. It might fail to learn complex setups (e.g., buying a dip).
2.  **Noise Overload:** Financial data is incredibly noisy. A per-step reward signal is dominated by market noise rather than the signal of the agent's decision quality. The Signal-to-Noise Ratio (SNR) of $R_t$ is extremely low compared to the SNR of a trade-level PnL.
3.  **High Variance:** The variance of the gradient updates can explode, making training unstable.
4.  **Transaction Cost Blindness:** If not carefully tuned, the agent might churn (over-trade) to capture small fluctuations, ignoring transaction fees that are only fully felt when the position is closed (or if fees are amortized incorrectly).

## Implementation Strategy for Final Training

If used for a final "polishing" or "fine-tuning" phase:

1.  **Hybrid Approach:** Use a weighted sum of the standard episodic return and the per-step return.
    $$ R_{total} = \alpha \cdot R_{episodic} + (1-\alpha) \cdot \sum R_{step} $$
2.  **Fee Integration:** Crucial to deduct estimated fees/slippage *at every step* based on the turnover, even for unrealized positions, to prevent churning.
    $$ V_t = \text{Cash} + \text{Assets} - \text{EstimatedExitFees} $$
3.  **Curriculum:** Start with sparse rewards (to learn long-term structure) and gradually blend in MtM rewards to sharpen the entry/exit timing.

## Conclusion
Per-step rewards are a powerful tool for **execution optimization** (getting the best price, minimizing slippage) but can be detrimental for **strategy learning** (finding trends, regime changes). They are best reserved for a specialized fine-tuning stage or a separate "Execution Agent" in a hierarchical system.
