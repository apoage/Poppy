# Self-Play Dojo Vision

**Date**: 2026-01-11
**Status**: Concept / Future Module
**Inspiration**: AlphaZero self-play, DeepMind AlphaStar League Training

## Core Insight

Current curriculum (synthetic levels 0-14) is limited by human imagination. We design the patterns, agent learns to exploit them. But what patterns are we missing?

**AlphaZero's breakthrough**: Agents discover strategies no human conceived by playing against themselves. The curriculum emerges from competition, not design.

Apply this to trading: instead of scripted market patterns, let agents compete and discover emergent strategies.

## Architecture Evolution

```
Phase 1 (Current):
  Agent vs Synthetic Market
  └── Human-designed curriculum (15 levels)
  └── Predictable, limited complexity
  └── Good for bootstrapping

Phase 2 (Self-Play Dojo):
  Agent vs Agent Population
  └── Emergent curriculum
  └── Unlimited complexity
  └── Discovers unknown strategies

Phase 3 (Hybrid):
  Agent vs (Synthetic + Agent Pool + Real Market)
  └── Best of all worlds
  └── Robust to any opponent
```

## Framework Compatibility

The modular framework makes this a **module swap, not a rewrite**:

```
┌─────────────────────────────────────────────┐
│           CORE FRAMEWORK (unchanged)        │
├─────────────────────────────────────────────┤
│  Environment │ Training │ MCTS │ Checkpoint │
└──────────────┴────┬─────┴──────┴────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
   ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
   │Synthetic│ │Self-Play│ │ Hybrid  │
   │  Dojo   │ │  Dojo   │ │  Dojo   │
   │  (v1)   │ │  (v2)   │ │  (v3)   │
   └─────────┘ └─────────┘ └─────────┘
```

## Self-Play Dojo Design

### Population Structure

```
Population (N=8 to 32 agents)
├── Champions (top performers, protected)
├── Challengers (training against champions)
├── Exploiters (specifically trained to beat champions)
└── Explorers (high entropy, random mutations)
```

### Competition Modes

| Mode | Description | Purpose |
|------|-------------|---------|
| **Round Robin** | Everyone vs everyone | Fair ranking |
| **Tournament** | Elimination brackets | Find single best |
| **League** | Tiered divisions | Stable training |
| **Predator/Prey** | Asymmetric objectives | Adversarial robustness |

### Knowledge Transfer

```
Winner-Loser Distillation:
  1. Winner plays, generates trajectories
  2. Loser trains on winner's (state, action) pairs
  3. Loser learns winner's policy without full training cost

Cross-Pollination:
  1. Take top K agents
  2. Average their weights (polyak averaging)
  3. Create "ensemble agent" with combined knowledge

Specialist Breeding:
  1. Identify agents good at specific regimes
  2. Trend-follower + Mean-reverter + Scalper
  3. Meta-agent learns when to use which style
```

## Implementation Sketch

### Minimal Self-Play Loop

```python
class SelfPlayDojo:
    def __init__(self, n_agents=8, framework=None):
        self.agents = [
            framework.create_agent(seed=i)
            for i in range(n_agents)
        ]
        self.elo_ratings = {i: 1500 for i in range(n_agents)}
        self.champion = None

    def run_generation(self):
        # 1. Competition phase
        matchups = self.create_matchups()
        results = []

        for agent_a, agent_b in matchups:
            # Shared market environment
            env = TradingEnv(seed=random())

            # Both agents trade same market
            score_a = self.evaluate(agent_a, env)
            score_b = self.evaluate(agent_b, env)

            results.append((agent_a, agent_b, score_a, score_b))

        # 2. Update ratings
        self.update_elo(results)

        # 3. Selection
        ranked = sorted(self.agents, key=lambda a: self.elo_ratings[a.id])
        winners = ranked[-4:]  # Top half
        losers = ranked[:4]    # Bottom half

        # 4. Knowledge transfer
        for winner, loser in zip(winners, losers):
            self.distill(teacher=winner, student=loser)

        # 5. Mutation for diversity
        self.mutate_explorers()

        # 6. Update champion
        self.champion = ranked[-1]

    def distill(self, teacher, student, n_trajectories=1000):
        """Student learns from teacher's behavior"""
        trajectories = teacher.generate_trajectories(n_trajectories)
        student.train_supervised(trajectories)

    def mutate_explorers(self, mutation_rate=0.1):
        """Add noise to explorer weights for diversity"""
        for agent in self.explorers:
            agent.params = add_noise(agent.params, scale=mutation_rate)
```

### Multi-Agent Market Environment

```python
class MultiAgentMarketEnv:
    """
    Multiple agents trading same market.
    Their actions affect each other (market impact).
    """
    def __init__(self, base_market, agents):
        self.market = base_market
        self.agents = agents
        self.positions = {a.id: 0 for a in agents}

    def step(self, actions: Dict[AgentID, Action]):
        # Aggregate order flow
        net_flow = sum(action_to_flow(a) for a in actions.values())

        # Market impact (big orders move price)
        impact = self.compute_impact(net_flow)

        # Update market with impact
        self.market.step(external_impact=impact)

        # Execute each agent's orders (with slippage from others)
        rewards = {}
        for agent_id, action in actions.items():
            fill_price = self.get_fill_price(action, net_flow)
            rewards[agent_id] = self.execute(agent_id, action, fill_price)

        return self.get_states(), rewards
```

## Why Self-Play Beats Scripted Curriculum

| Aspect | Scripted Curriculum | Self-Play |
|--------|---------------------|-----------|
| **Complexity ceiling** | Limited by designer | Unlimited emergence |
| **Adversarial robustness** | Only vs known patterns | Vs adaptive opponents |
| **Strategy discovery** | Human-conceived only | Novel, alien strategies |
| **Curriculum progression** | Manual level design | Automatic difficulty scaling |
| **Overfitting risk** | High (memorize levels) | Low (opponents adapt) |

### Emergent Behaviors to Expect

From game AI research, self-play discovers:
- **Feints**: Fake one direction, go another
- **Timing attacks**: Exploit opponent's update frequency
- **Resource denial**: Act to limit opponent's opportunities
- **Meta-gaming**: Adapt strategy based on opponent's style

In trading context:
- **Front-running patterns**: Detect and exploit other agents' signals
- **Regime switching**: Change strategy when opponent adapts
- **Liquidity games**: Provide/take liquidity strategically
- **Information hiding**: Avoid telegraphing intentions

## Integration with Current System

### Phase 1: Shadow Mode
- Run self-play dojo alongside current training
- Compare strategies discovered vs current curriculum
- No changes to main training loop

### Phase 2: Curriculum Augmentation
- Add "self-play levels" to existing curriculum (Level 15+)
- Agent faces frozen snapshots of evolved opponents
- Gradual introduction

### Phase 3: Full Self-Play
- Replace synthetic curriculum with population training
- Keep synthetic levels for bootstrapping only
- Main training is agent vs agent

## Resource Requirements

| Component | Current | Self-Play |
|-----------|---------|-----------|
| **GPU memory** | 1 agent | N agents (8-32x) |
| **Training time** | 1x | 2-4x (more games) |
| **Complexity** | Low | Medium |
| **Debugging** | Easy | Harder (emergent behavior) |

### Mitigation
- Share weights across population (only heads differ)
- Asynchronous evaluation (not all agents need GPU simultaneously)
- Periodic snapshots (don't need all agents live)

## Success Metrics

| Metric | Target |
|--------|--------|
| **Strategy diversity** | Population uses 5+ distinct strategies |
| **Champion improvement** | Beats previous champion 60%+ |
| **Generalization** | Self-play agent beats curriculum agent on held-out data |
| **Novel discovery** | At least 1 strategy not in original curriculum |

## Open Questions

1. **How many agents?** 8 might be too few for diversity, 32 might be too slow
2. **Market impact modeling?** How much should agents affect each other?
3. **Reward shaping?** Pure PnL or relative performance?
4. **When to distill?** Every generation or only on plateaus?
5. **Specialist vs generalist?** One champion or ensemble?

## References

- **AlphaZero**: Silver et al., "Mastering Chess and Shogi by Self-Play"
- **AlphaStar**: Vinyals et al., "Grandmaster level in StarCraft II using multi-agent RL"
- **OpenAI Five**: Dota 2 self-play at scale
- **Population Based Training**: Jaderberg et al., "Population Based Training of Neural Networks"

## Next Steps

1. [ ] Prototype `SelfPlayDojo` class
2. [ ] Implement ELO rating system
3. [ ] Create `MultiAgentMarketEnv` wrapper
4. [ ] Test with 4 agents on Level 0
5. [ ] Compare discovered strategies vs curriculum
6. [ ] Write TASK file if promising

---

*"The best teacher is a worthy opponent."*
