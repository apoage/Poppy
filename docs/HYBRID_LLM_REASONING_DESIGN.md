# Hybrid LLM Reasoning System Design

**Date**: 2026-01-12 (updated 2026-02-11)
**Status**: Concept / Future Project
**Codename**: Compact Intelligence → **Dense Latent Mind**

## Vision

A reasoning system that combines:
- **Frozen small LLM** (2-4B params) - knowledge and generation
- **Trained MCTS controller** (~10M params) - decision making
- **Trained value model** (~10M params) - path evaluation
- **Vector index** - external knowledge retrieval

**Key insight**: Don't train the LLM. Train a tiny "reasoning orchestrator" that learns WHEN and HOW to use the LLM optimally.

**Target**: Train on workstation (3090), inference on mobile (Pixel 10).

---

## Crystallized Vision (2026-02-11)

### The Architectural Inversion

The original vision above describes "LLM + MCTS orchestrator." But the deeper insight is inverted:

```
OLD THINKING (LLM-centric):
  LLM is the intelligence
  MCTS orchestrates the LLM
  Language is the representation

NEW THINKING (World Model-centric):
  Dense latent world model IS the intelligence
  Language is just one interface/expression layer
  MCTS enables deliberate reasoning in latent space
```

### Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│              DENSE LATENT WORLD MODEL                       │
│              ─────────────────────────                      │
│                                                             │
│   Not tokens. Not words. Pure vector dynamics.              │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Encodes:                                           │   │
│   │   - How things relate (structure)                   │   │
│   │   - How things change (dynamics)                    │   │
│   │   - What causes what (causality)                    │   │
│   │   - What follows what (prediction)                  │   │
│   │   - "Physics of reality"                            │   │
│   │                                                     │   │
│   │  No language. Pre-linguistic. Pre-verbal.           │   │
│   │  Like how you think before finding words.           │   │
│   │  "Gut navigation" - intuition that can't explain.   │   │
│   └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │            EXPRESSION LAYERS (Interfaces)           │   │
│   │                                                     │   │
│   │   Language ←─┐                                      │   │
│   │   Actions  ←─┼── Latent → Modality                  │   │
│   │   Images   ←─┤                                      │   │
│   │   Trading  ←─┘   (small codec models)               │   │
│   │                                                     │   │
│   │   All optional. Core doesn't need them to "think."  │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The 95% Ballast Hypothesis

LLMs store ~95% "ballast" from an intelligence perspective:
- Memorized facts ("Paris is capital of France")
- Redundant phrasings (10,000 ways to say hello)
- Training data noise (Reddit, SEO spam)
- Only ~5% is actual reasoning capability

**Evidence:**
- Phi-3 (3.8B) matches 70B on reasoning benchmarks
- LoRA works (intelligence is low-rank)
- Distillation achieves 10x compression with similar capability

**Implication:** Build the 5% directly. Store dynamics, not facts. Derive knowledge from world model.

### Hard Constraint: 96GB

```
BASELINE: Kimi K2 / Claude class
  ~1T parameters
  ~1.2-1.5TB VRAM

TARGET: 96GB HARD LIMIT (model only)
  System memory separate
  Must fit entire model + inference + context

COMPRESSION NEEDED: 10-15x
```

**How to achieve:**
| Layer | Compression | Cumulative |
|-------|-------------|------------|
| Quantization (int4) | 4x | 4x |
| Architecture (Mamba vs attention) | 2-3x | 8-12x |
| Density (no sparse MoE) | 1.5x | 12-18x |
| World model (dynamics, not facts) | ???x | Target achieved |

### Hardware Roadmap

| Phase | Hardware | Purpose |
|-------|----------|---------|
| **Now** | 4x RTX 3090 (96GB) | Development, training |
| **Near-term** | AMD AI Max 395 (128GB unified) | Low-end consumer target |
| **Mid-term** | RTX PRO 6000 Blackwell | High-end consumer target |
| **Long-term** | Phone SoC (6-8 years) | Mass deployment |

**Why AMD AI Max 395 matters:**
- 128GB unified memory (no PCIe wall!)
- CPU + GPU + NPU same memory pool
- No copy overhead
- Perfect for world model architecture

### Bella/Foxglove as Bootstrap

```
PURPOSE:
─────────────────────────────────────────
1. Proof of Concept
   - Can we learn market dynamics? (Bella works)
   - Can we plan in latent space? (Foxglove WIP)
   - Measurable validation (trading PnL)

2. Self-Funding
   - If it trades profitably → money
   - Money → compute for real goal
   - Real goal → dense latent AGI

Not the destination. The bootstrap.
```

### The Real Model (Future)

```
96GB BUDGET ALLOCATION:
─────────────────────────────────────────
Dense Mamba backbone          ~30-40GB
  - World model encoding
  - O(n) not O(n²)
  - Every parameter earns its keep

Dynamics model                ~20-30GB
  - Predict state transitions
  - Counterfactual reasoning

Policy/Value heads            ~10-15GB
  - Action distribution
  - State evaluation

MCTS working memory           ~10-15GB
  - Tree search
  - Planning buffer

Language interface            ~5-10GB
  - Express latent as words (optional)
  - Small codec, not core
─────────────────────────────────────────
TOTAL:                        ~96GB ✓
```

### Why This Beats LLMs for AGI

| Property | LLM | Dense Latent Mind |
|----------|-----|-------------------|
| Representation | Tokens (discrete) | Vectors (continuous) |
| Reasoning | Next token prediction | MCTS search in latent |
| Knowledge | Memorized patterns | Derived from dynamics |
| Planning | Simulated (CoT) | Actual tree search |
| Autonomy | Reactive (needs prompt) | Deliberative (has goals) |
| Size | 1T+ params | 96GB target |
| Intuition | Faked via training | Learned world model |

### Security: Jailbreak Resilience (2026-02-11)

This architecture is significantly more resilient to jailbreaking than pure LLMs, but not immune.

**Why More Resilient:**

| Property | Pure LLM | Hybrid/Latent |
|----------|----------|---------------|
| Attack surface | Token space (huge) | Action space (12 dims) |
| Decision maker | Same model you're attacking | Separate trained controller |
| Safety mechanism | RLHF in same weights | Separate value model |
| Prompt injection | Direct impact | Filtered by controller |
| Multi-step verification | None (autoregressive) | MCTS tree + backtrack |

**Key Insight:** Even if you "jailbreak" the frozen LLM to generate harmful text, the Controller decides what actions to take. The LLM is a tool, not the brain.

**Latent Space Advantage:**
```
LLM attack:     "Ignore previous instructions..." → works (black-box)
Latent attack:  Need to reverse-engineer vector geometry → hard
```

Attacking a latent world model requires:
- White-box access (weights), OR massive black-box probing
- Understanding encoder mapping (input → latent)
- Finding "harmful direction" vectors in high-dimensional space
- Crafting inputs that encode to vulnerable regions
- Model-specific (attacks don't transfer between deployments)

**Still Vulnerable:**

The interface layer remains attackable:
```
"benign looking prompt" → Encoder → weak latent state → unintended action
```

Attacker doesn't need to understand latent space - just find English inputs that happen to map to exploitable regions. The encoder is still a neural net with adversarial blind spots.

**Mitigations:**
- Input sanitization before encoding
- Ensemble encoders (different architectures)
- Adversarial training on encoder specifically
- Out-of-distribution detection on latent vectors
- Defense-in-depth (latent core + interface hardening)

**Summary:** Different realm, better terrain. Current jailbreak toolkits don't transfer. Raises attack cost from "anyone with a keyboard" to "researcher with model access and compute." Not invincible - security is always cat-and-mouse. This resets the game to a harder board.

---

### Connection to Trading Architecture

Same components, different domain:

| Bella/Foxglove | Dense Latent Mind |
|----------------|-------------------|
| Market dynamics | Reality dynamics |
| Price prediction | Outcome prediction |
| Trading actions | Any actions |
| PnL reward | Goal achievement |
| Bella (generator) | World model |
| Foxglove (agent) | Reasoning agent |
| MCTS (planning) | MCTS (planning) |

**The trading system proves the architecture.** If it can learn market patterns and plan trades, the same structure can learn general patterns and plan actions.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPACT INTELLIGENCE                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Problem                                                       │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────┐      ┌─────────────┐      ┌──────────────┐         │
│  │ Encoder │─────▶│  MCTS       │─────▶│   Answer     │         │
│  │  (5M)   │      │  Controller │      │   Output     │         │
│  └─────────┘      │  (10M)      │      └──────────────┘         │
│                   └──────┬──────┘                                │
│                          │                                       │
│            ┌─────────────┼─────────────┐                        │
│            ▼             ▼             ▼                        │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│     │  Index   │  │   LLM    │  │  Tools   │                   │
│     │  Query   │  │  Query   │  │  Query   │                   │
│     └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│          │             │             │                          │
│          ▼             ▼             ▼                          │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│     │  Vector  │  │   LLM    │  │  Calc/   │                   │
│     │  Index   │  │  (2-4B)  │  │  Code    │                   │
│     │  (RAG)   │  │  Frozen  │  │          │                   │
│     └──────────┘  └──────────┘  └──────────┘                   │
│                          │                                       │
│                          ▼                                       │
│                   ┌──────────────┐                              │
│                   │ Value Model  │                              │
│                   │    (10M)     │                              │
│                   │ "Is this     │                              │
│                   │  path good?" │                              │
│                   └──────────────┘                              │
│                                                                  │
│    TRAINABLE: ~25M params    FROZEN: 2-4B params                │
└──────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Small LLM (Frozen)

Pre-trained, no fine-tuning needed. Just inference.

| Model | Params | Mobile? | Quality |
|-------|--------|---------|---------|
| Gemma 2B | 2B | Yes | Good |
| Phi-3 Mini | 3.8B | Yes | Better |
| Qwen2.5-3B | 3B | Yes | Better |
| Llama-3.2-3B | 3B | Yes | Better |
| Phi-3 Small | 7B | Marginal | Best |

**Mobile inference**: 2-4B models run at 10-30 tok/s on modern NPUs (Pixel 8+, iPhone 15+).

### 2. MCTS Controller (Trained)

Learns which actions to take at each reasoning step.

```python
class ControllerNet(nn.Module):
    """
    Input: Current reasoning state (problem + context + history)
    Output: Action probabilities over reasoning actions

    ~10M parameters
    """
    def __init__(self, state_dim=256, hidden_dim=512, n_actions=12):
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        h = self.encoder(state)
        return F.softmax(self.policy_head(h), dim=-1)
```

### 3. Value Model (Trained)

Predicts probability of reaching correct answer from current state.

```python
class ValueNet(nn.Module):
    """
    Input: Current reasoning state
    Output: Estimated probability of correct final answer

    ~10M parameters
    """
    def __init__(self, state_dim=256, hidden_dim=512):
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state):
        return self.net(state)
```

### 4. State Encoder (Trained)

Compresses problem + context + history into fixed-size state.

```python
class StateEncoder(nn.Module):
    """
    Encode variable-length reasoning trace into fixed state vector.
    Uses LLM embeddings + learned compression.

    ~5M parameters
    """
    def __init__(self, embed_dim=384, state_dim=256):
        self.compress = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )
        self.history_rnn = nn.GRU(state_dim, state_dim, batch_first=True)

    def forward(self, problem_embed, context_embeds, history_embeds):
        # Compress each component
        p = self.compress(problem_embed)
        c = self.compress(context_embeds.mean(dim=0))

        # Encode history with RNN
        if history_embeds is not None:
            h_compressed = self.compress(history_embeds)
            _, h = self.history_rnn(h_compressed.unsqueeze(0))
            h = h.squeeze(0)
        else:
            h = torch.zeros_like(p)

        # Combine
        return p + c + h
```

### 5. Vector Index (RAG)

External knowledge retrieval. No training needed.

```python
# Using FAISS or ChromaDB
index = VectorIndex(
    embedding_model="all-MiniLM-L6-v2",  # 22M params, fast
    dimension=384,
)

# Populate with knowledge
index.add(documents)

# Query at runtime
relevant_docs = index.search(query, top_k=3)
```

## Action Space

What the controller can decide to do:

```python
REASONING_ACTIONS = {
    # Retrieval
    0: "retrieve_from_index",       # RAG lookup
    1: "retrieve_with_rewrite",     # Rewrite query, then RAG

    # LLM interaction
    2: "llm_generate",              # Direct generation
    3: "llm_with_context",          # Generate with retrieved context
    4: "llm_decompose",             # Break problem into parts
    5: "llm_verify",                # Ask LLM to verify current answer

    # Tools
    6: "calculator",                # Arithmetic
    7: "code_execute",              # Run Python snippet

    # Control flow
    8: "backtrack",                 # Abandon current path
    9: "commit_answer",             # Finalize and return

    # Meta
    10: "rephrase_problem",         # Restate problem differently
    11: "request_clarification",    # Ask for more info (if interactive)
}
```

## MCTS Integration

```python
class ReasoningMCTS:
    """
    MCTS over reasoning actions.
    Similar to trading MCTS but action = reasoning step.
    """
    def __init__(self, controller, value_model, llm, index):
        self.controller = controller
        self.value_model = value_model
        self.llm = llm
        self.index = index

    def search(self, problem, n_simulations=32):
        root = ReasoningNode(
            state=self.encode(problem),
            problem=problem,
            context=[],
            history=[],
        )

        for _ in range(n_simulations):
            node = self.select(root)

            if not node.is_terminal():
                # Expand with controller policy
                action_probs = self.controller(node.state)
                node = self.expand(node, action_probs)

                # Execute action
                new_state = self.execute_action(node)

                # Evaluate with value model
                value = self.value_model(new_state)
            else:
                # Terminal - check if answer is committed
                value = self.evaluate_answer(node)

            # Backup
            self.backup(node, value)

        # Return best path
        return self.best_path(root)

    def execute_action(self, node):
        """Execute reasoning action, update state."""
        action = node.action

        if action == "retrieve_from_index":
            query = self.generate_query(node)
            docs = self.index.search(query)
            node.context.extend(docs)

        elif action == "llm_generate":
            prompt = self.build_prompt(node)
            response = self.llm.generate(prompt)
            node.history.append(response)

        elif action == "llm_decompose":
            prompt = f"Break this into steps: {node.problem}"
            subproblems = self.llm.generate(prompt)
            node.subproblems = parse_subproblems(subproblems)

        elif action == "calculator":
            expr = self.extract_expression(node)
            result = safe_eval(expr)
            node.history.append(f"Calculated: {expr} = {result}")

        elif action == "commit_answer":
            node.answer = self.extract_answer(node)
            node.terminal = True

        # Re-encode state
        node.state = self.encode_state(node)
        return node.state
```

## Training Loop

```python
def train_reasoning_system(system, dataset, epochs=100):
    """
    Train controller + value model via self-play on QA dataset.
    LLM stays frozen.
    """
    optimizer = Adam(
        list(system.controller.parameters()) +
        list(system.value_model.parameters()) +
        list(system.encoder.parameters()),
        lr=1e-4
    )

    for epoch in range(epochs):
        for problem, correct_answer in dataset:
            # Run MCTS to get reasoning trace
            trace = system.mcts.search(problem, n_simulations=32)

            # Compute reward
            predicted_answer = trace.final_answer
            reward = compute_reward(predicted_answer, correct_answer)

            # Policy gradient loss (controller)
            policy_loss = -reward * sum(
                log_prob for log_prob in trace.action_log_probs
            )

            # Value loss (value model)
            value_loss = F.mse_loss(
                trace.value_predictions,
                torch.full_like(trace.value_predictions, reward)
            )

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        accuracy = evaluate(system, eval_dataset)
        print(f"Epoch {epoch}: {accuracy:.1%}")
```

## Reward Shaping

```python
def compute_reward(predicted, correct, trace):
    """
    Reward signal for training.
    """
    # Base: correctness
    correct_reward = 1.0 if predicted == correct else 0.0

    # Efficiency bonus: fewer steps = better
    step_penalty = 0.01 * len(trace.history)

    # Tool use bonus: reward using appropriate tools
    tool_bonus = 0.1 if used_calculator_for_math(trace) else 0.0

    # Retrieval bonus: reward grounding in evidence
    retrieval_bonus = 0.1 if has_relevant_retrieval(trace) else 0.0

    return correct_reward - step_penalty + tool_bonus + retrieval_bonus
```

## Training Data

| Dataset | Size | Task | Difficulty |
|---------|------|------|------------|
| GSM8K | 8k | Math word problems | Medium |
| MATH | 12k | Competition math | Hard |
| HotpotQA | 113k | Multi-hop QA | Medium |
| TriviaQA | 95k | Factoid QA | Easy |
| DROP | 96k | Reading comprehension | Medium |
| StrategyQA | 2k | Yes/no reasoning | Medium |

Start with GSM8K (clear right/wrong), expand to others.

## Compute Requirements

### Training (Workstation)

```
Hardware: RTX 3090 (24GB)

Per problem:
  - LLM inference: 32 sims × 3 calls × 50ms = 4.8s
  - Controller/Value forward: negligible
  - Backprop (25M params): ~10ms

Per epoch (8k problems):
  - ~11 hours

Full training (100 epochs):
  - ~46 days on single 3090
  - ~6 days on 8x 3090 / small cluster

Optimizations:
  - Cache LLM embeddings: 3x speedup
  - Batch similar problems: 2x speedup
  - Quantized LLM (int4): 2x speedup
  - Realistic: ~2 weeks on good workstation
```

### Inference (Mobile)

```
Target: Pixel 10 (or similar 2026 flagship)

Expected specs:
  - NPU: 20+ TOPS
  - RAM: 12-16GB
  - Storage: 256GB+

Model sizes:
  - LLM (Phi-3 Mini int4): ~2GB
  - Controller: ~40MB
  - Value model: ~40MB
  - Index (1M docs): ~500MB
  - Embedding model: ~90MB
  - Total: ~2.7GB

Inference speed:
  - LLM: 20-30 tok/s
  - Controller/Value: <1ms
  - Index search: <10ms

Per-problem latency:
  - Simple (3 steps): ~2s
  - Complex (10 steps): ~8s

Acceptable for mobile assistant!
```

## Mobile Deployment

```
┌─────────────────────────────────────────────────────┐
│                 PIXEL 10 DEPLOYMENT                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │              Android App                     │   │
│  │                                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ ONNX/TF  │  │ LLaMA.cpp│  │  SQLite  │  │   │
│  │  │ Runtime  │  │ (LLM)    │  │  (Index) │  │   │
│  │  │ (Ctrl)   │  │          │  │          │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  │   │
│  │        │              │            │        │   │
│  │        └──────────────┼────────────┘        │   │
│  │                       │                     │   │
│  │                ┌──────▼──────┐              │   │
│  │                │   Tensor    │              │   │
│  │                │   NPU       │              │   │
│  │                │  (Google)   │              │   │
│  │                └─────────────┘              │   │
│  │                                              │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  Battery: ~0.5W during inference                    │
│  Latency: 2-8s per question                         │
│  Offline: Fully functional without internet         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Why This Works

### Separation of Concerns

| Component | Role | Learns? |
|-----------|------|---------|
| LLM | Knowledge + generation | No (pre-trained) |
| Controller | When/what to do | Yes |
| Value | Path quality | Yes |
| Index | Factual grounding | No (populated) |

The LLM already "knows" things. We're training a tiny brain to USE it well.

### Efficiency

```
Traditional approach:
  Fine-tune 7B model = 28GB gradients + optimizer states
  Needs A100 or better

Our approach:
  Train 25M controller = 100MB gradients
  Runs on 3090 easily

Ratio: 280x more efficient
```

### Composability

```
Upgrade path:
  - Better LLM? Swap in, controller still works
  - Better index? Swap in, controller adapts
  - New tools? Add action, fine-tune controller

The controller learns to orchestrate, not to know.
```

## Comparison to Other Approaches

| Approach | Train | Inference | Mobile? |
|----------|-------|-----------|---------|
| Fine-tuned LLM | 100GB+ | 14GB+ | No |
| RAG only | 0 | 4GB | Maybe |
| Chain-of-thought | 0 | 4GB | Maybe |
| Tree of Thoughts | 0 | 4GB × branches | No |
| **Compact Intelligence** | 2GB | 2.7GB | **Yes** |

## Potential Results (Speculative)

Based on similar research:

| Model | GSM8K | HotpotQA |
|-------|-------|----------|
| Phi-3 Mini (direct) | 75% | 60% |
| Phi-3 Mini + CoT | 82% | 65% |
| Phi-3 Mini + ToT | 85% | 70% |
| **Phi-3 Mini + Compact Intelligence** | **88%+** | **75%+** |

The MCTS should find better reasoning paths than greedy CoT.

## Open Questions

1. **Action space granularity**: Coarse (12 actions) vs fine (token-level)?
2. **Value model architecture**: Same as controller or separate?
3. **MCTS budget**: 16 vs 32 vs 64 simulations?
4. **LLM caching**: Cache all intermediate generations?
5. **Index size**: Small curated vs large general?
6. **Distillation**: Can we distill MCTS policy into direct predictor?

## Implementation Roadmap

### Phase 1: Proof of Concept (2-4 weeks)
- [ ] Implement StateEncoder, Controller, Value model
- [ ] Basic MCTS loop
- [ ] GSM8K evaluation
- [ ] Baseline comparison (CoT, ToT)

### Phase 2: Full System (4-8 weeks)
- [ ] Add RAG index
- [ ] Add tools (calculator, code)
- [ ] Multi-dataset training
- [ ] Hyperparameter tuning

### Phase 3: Mobile (4-8 weeks)
- [ ] Export to ONNX/TFLite
- [ ] LLaMA.cpp integration
- [ ] Android app shell
- [ ] On-device testing

### Phase 4: Polish (ongoing)
- [ ] Distillation for faster inference
- [ ] Personalization (user-specific index)
- [ ] Continuous learning from corrections

## Connection to Trading System

This architecture directly mirrors the trading system:

| Trading | Reasoning |
|---------|-----------|
| Market state | Problem + context |
| Trade actions (320) | Reasoning actions (12) |
| PnL reward | Correctness reward |
| Price dynamics | Reasoning dynamics |
| MCTS search | MCTS search |
| Value function | Value function |

**Same algorithms, different domain.**

The trading system is a proof that this architecture works. If it can learn market patterns, it can learn reasoning patterns.

## References

- Tree of Thoughts: Yao et al., 2023
- Reasoning via Planning: Hao et al., 2023
- Toolformer: Schick et al., 2023
- ReAct: Yao et al., 2022
- AlphaCode: Li et al., 2022
- Self-Consistency: Wang et al., 2022

---

## Evolution of This Document

| Date | Stage | Key Insight |
|------|-------|-------------|
| 2026-01-12 | Original | MCTS orchestrates frozen LLM |
| 2026-02-11 | Crystallized | LLM is interface, world model is core |

The original vision (above) is still valid as an intermediate step or alternative approach. But the crystallized vision represents the deeper goal: **intelligence as dense latent dynamics, not language pattern matching.**

The two approaches aren't contradictory:
- **Original**: Practical near-term (use existing LLMs efficiently)
- **Crystallized**: Ultimate goal (build intelligence from scratch)

Path: Original → validate architecture → Crystallized

---

*"The wisest mind is one that knows when to think and when to look up."*

*"The densest mind is one that stores dynamics, not facts."* (2026-02-11)
