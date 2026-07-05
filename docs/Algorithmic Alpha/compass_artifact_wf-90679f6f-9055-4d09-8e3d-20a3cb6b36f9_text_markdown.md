# Optimal transport for portfolio rebalancing: elegant theory meets practical limits

**Optimal transport theory provides a mathematically rigorous framework for portfolio transitions, mapping the rebalancing problem to minimizing "transport cost" between probability distributions.** However, while OT delivers provably optimal solutions *for its specific formulation*, that formulation makes simplifying assumptions—fixed costs, continuous mass, static optimization—that diverge significantly from real trading conditions. The practical benefits over existing methods remain limited: OT excels as a theoretical lens, distance metric, and calibration tool, but traditional quadratic programming with Almgren-Chriss execution algorithms remains the operational standard for actual portfolio transitions.

## The mathematical mapping is precise and elegant

The OT framework treats portfolio weights as probability distributions, with rebalancing as transporting mass from source (current) to destination (target). This isn't merely an analogy—it's a rigorous mathematical correspondence:

| Financial Concept | OT Equivalent |
|-------------------|---------------|
| Current portfolio weights | Source measure μ |
| Target portfolio weights | Target measure ν |
| Trade plan (what to sell/buy) | Transport plan γ |
| Transaction costs | Cost function c(x,y) |
| Budget constraint | Source marginal constraint |
| Target allocation | Target marginal constraint |

The **Kantorovich formulation** transforms portfolio rebalancing into a linear program: minimize Σᵢⱼ γᵢⱼc(i,j) subject to marginal constraints ensuring mass conservation. The transport plan γᵢⱼ represents the fraction of wealth moving from asset i to asset j—a direct encoding of trades. When i ≠ j, γᵢⱼ represents selling i and buying j; diagonal entries γᵢᵢ represent untouched positions.

This formulation has strong theoretical properties. **Existence** of optimal transport plans is guaranteed for any lower semi-continuous cost function on Polish spaces. The problem is convex, enabling efficient computation. The **dual formulation** yields Kantorovich potentials (φᵢ, ψⱼ) interpretable as shadow prices—what a market maker would charge to handle all trades competitively.

The **Wasserstein distance** provides a natural metric for portfolio similarity. W₁ (Earth Mover's Distance) measures total effort to transform one portfolio into another; W₂ penalizes large individual trades more heavily (relevant when market impact is convex in trade size). The choice of ground metric on asset space—correlation distance, sector hierarchy, factor loadings, or liquidity scores—determines what "closeness" means financially.

## Algorithmic advances make OT computationally tractable

The **Sinkhorn algorithm** with entropic regularization has revolutionized computational OT. Adding an entropy term −εH(γ) to the objective transforms the problem into iterative matrix-vector multiplications that converge in O(n²/ε²) time—dramatically faster than the O(n³ log n) complexity of exact network simplex methods. Each iteration is trivially parallelizable on GPUs, enabling **100× speedups** for problems with tens of thousands of assets.

The regularization parameter ε controls a fundamental tradeoff. Small ε produces sparse transport plans (fewer trades) but slower convergence and numerical instability. Large ε yields dense plans with many small trades but faster, more stable computation. For financial applications where **concentrated rebalancing** is preferred, practitioners should use small ε with log-domain stabilization. A rule of thumb: ε ≈ 0.01 × median(cost matrix).

Entropic regularization also makes OT **differentiable**, enabling integration into end-to-end learning pipelines. Gradients with respect to portfolio weights can be computed via implicit differentiation of KKT conditions, supporting applications like distributionally robust optimization where one minimizes worst-case risk over a Wasserstein ball around the empirical return distribution.

**Unbalanced optimal transport** extends the framework to portfolios with different total masses—handling deposits or withdrawals. UOT adds divergence penalties (typically KL) for mass creation/destruction rather than strict marginal constraints. Recent work by Kováčová, Patel, and Biagini (2025) develops "non-conservative OT" specifically for portfolio rebalancing, modeling trading costs as mass loss during transport with bid-ask spreads incorporated directly into prices.

## Academic literature spans theory and applications

The intersection of OT and quantitative finance has produced substantial literature across three main threads:

**Distributionally robust portfolio optimization** uses Wasserstein balls to define uncertainty sets around empirical return distributions. Blanchet, Chen, and Zhou's 2022 *Management Science* paper shows this approach naturally induces regularization similar to Lasso, with extensive S&P 500 backtesting demonstrating competitive performance against Fama-French and Black-Litterman benchmarks. Nguyen et al. (2024, *Operations Research*) extend this to conditional portfolio selection incorporating side information.

**Martingale optimal transport** connects OT to model-free option pricing, providing arbitrage-free price bounds without assuming parametric models. Beiglböck, Henry-Labordère, and Penkner's 2013 *Finance and Stochastics* paper established the foundation; subsequent work addresses robust hedging with pricing-hedging duality.

**Stochastic portfolio theory** links functionally generated portfolios to OT via exponentially concave functions. Wong (2017) shows that portfolio maps can be characterized through optimal transport couplings between probability distributions on the simplex—a theoretical connection rather than a practical algorithm.

Bernard et al. (2023, *SIAM Journal on Financial Mathematics*) examine portfolios constrained to lie within a Wasserstein ball around a benchmark, characterizing optimal strategies through isotonic projections. Backhoff-Veraguas et al. (2020, *Finance and Stochastics*) prove that standard Wasserstein distance is insufficient for hedging—adapted Wasserstein distances accounting for temporal structure are necessary for dynamic strategies.

## Implementation tools are mature and accessible

Three Python libraries dominate practical OT computation:

**POT (Python Optimal Transport)** offers the most comprehensive feature set with excellent documentation. It supports exact OT via network simplex, multiple Sinkhorn variants, and backends including NumPy, PyTorch, JAX, and CuPy. Basic usage for portfolio rebalancing:

```python
import ot
# Current and target weights (normalized)
a = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
b = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
# Transaction cost matrix
C = build_transaction_cost_matrix(n_assets=5)
# Solve with entropic regularization
T = ot.sinkhorn(a, b, C, reg=0.01)
total_cost = np.sum(T * C)
```

**OTT-JAX** provides native JAX implementation with JIT compilation and automatic differentiation—ideal for gradient-based optimization loops. **GeomLoss** integrates with PyTorch via KeOps for memory-efficient computation on 100k+ samples.

For portfolio rebalancing, the cost matrix should encode realistic trading frictions: bid-ask spreads, market impact estimates, and potentially asymmetric costs (selling cheaper than buying or vice versa). The diagonal should be zero (no cost for holding). Lower regularization parameters produce sparser transport plans with fewer trades.

## Critical limitations constrain practical applicability

Despite theoretical elegance, OT faces fundamental mismatches with real portfolio transitions:

**Market impact is size-dependent, not fixed.** Standard OT assumes cost c(x,y) independent of trade quantity. Real market impact follows approximately square-root scaling: moving $10 million costs more than 10× moving $1 million. The Almgren-Chriss model captures this with temporary impact h(v) = ε + η·|v|^{1/2} and permanent impact g(v) = γ·v. OT cannot naturally distinguish temporary impact (dissipates) from permanent impact (persists).

**Portfolios have discrete shares, not continuous mass.** OT works with probability measures allowing arbitrary mass splitting. Real portfolios have integer share counts, minimum lot sizes, and rounding constraints. The Kantorovich relaxation—necessary for tractability—inherently assumes divisibility. Post-hoc rounding can violate optimality; mixed-integer formulations lose computational efficiency.

**Static optimization misses dynamic benefits.** OT provides a snapshot solution: given current and target, find the optimal transport. Real transitions occur over time with evolving market conditions, information updates, and urgency constraints. Multi-period dynamic programming captures "no-trade regions" that expand/contract with the horizon; Almgren-Chriss explicitly parameterizes urgency via risk aversion λ. Benamou-Brenier dynamic OT exists theoretically but remains computationally expensive.

**Complex constraints break the marginal structure.** OT handles marginal constraints elegantly, but real portfolios face sector limits, position caps, leverage restrictions, and regulatory requirements that don't fit this framework. Adding such constraints destroys the special structure enabling efficient algorithms.

## The verdict: theory yes, operations no

**Can OT genuinely provide optimal rebalancing paths?** Yes—for the OT formulation. The Kantorovich problem has proven existence and uniqueness results; algorithms compute global optima efficiently. But "optimal for the formulation" differs from "optimal for real trading." The fixed-cost, continuous-mass, static, marginal-constraint assumptions diverge substantially from market reality.

**Practical benefits over existing methods** are limited for operational transitions. Traditional approaches remain superior:

- **Quadratic programming** with transaction cost penalties solves the target portfolio problem with decades of production experience and fast solvers (CPLEX, Gurobi)
- **Almgren-Chriss implementation shortfall** explicitly optimizes the market impact vs. timing risk tradeoff with size-dependent costs
- **VWAP/TWAP** provides benchmark-tracking execution with understood properties
- **Multi-period dynamic programming** captures evolving constraints and information

OT contributes most as:
- A **distance metric** (Wasserstein) for comparing portfolio distributions
- A **calibration tool** for derivatives models in non-parametric settings
- A **theoretical framework** for model-free pricing bounds via martingale OT
- A **robustification mechanism** through distributionally robust optimization with Wasserstein balls

## Conclusion

Optimal transport provides a mathematically beautiful formulation of portfolio rebalancing with rigorous optimality guarantees, efficient algorithms, and mature software implementations. The Kantorovich dual interpretation connects to pricing theory; Wasserstein distances offer principled portfolio comparison metrics; unbalanced OT handles deposits and withdrawals.

Yet the framework's assumptions—**fixed costs, continuous mass, static optimization, marginal constraints only**—create a gap between theoretical optimality and practical usefulness. Real portfolio transitions face size-dependent market impact, discrete lot sizes, dynamic urgency tradeoffs, and complex regulatory constraints that OT handles poorly or not at all.

The most promising near-term applications lie not in replacing traditional transition management but in **complementing** it: using Wasserstein balls for robust portfolio optimization, OT-based metrics for portfolio similarity analysis, and martingale OT for model-free derivatives pricing. For operational rebalancing, the established QP + Almgren-Chriss + dynamic programming toolkit remains the practical choice—at least until OT methods better incorporate size-dependent costs and dynamic adaptation.