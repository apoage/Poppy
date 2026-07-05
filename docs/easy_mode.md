# Easy Mode: System 2 Trading on a Single GPU (RTX 3090)

**Philosophy:** Stop trying to learn everything at once.

End-to-end MuZero (learning representation, dynamics, policy, and value simultaneously from a cold start) is "Hard Mode." It requires massive batch sizes to stabilize gradients and often leads to non-stationary "death spirals" (bad model → bad search → bad data).

**"Easy Mode"** decouples these problems into three stable, solvable steps using **Expert Iteration (ExIt)**. This fits perfectly within the memory constraints of a single RTX 3090 (24GB VRAM) and provides clear debugging checkpoints.

---

## The Architecture

Instead of one giant RL loop, we build the agent in layers.

| Phase | Component | Goal | Method | Hardware Load |
|-------|-----------|------|--------|---------------|
| **1** | **The Cheat** | Perfect "System 1" Intuition | Supervised Learning (BC) on Oracle Data | High Compute / Low VRAM |
| **2** | **The Dream** | Stable World Simulator | Self-Supervised Learning on Market Data | Med Compute / Med VRAM |
| **3** | **The Thought** | "System 2" Reasoning | MCTS at Inference Time (Frozen Model) | Low Compute / High VRAM (Buffer) |

---

## Phase 1: The Cheat (Supervised Imprinting)

We already have a **Hindsight Oracle** (implemented in `TASK_125`/`path_solver.py`) that knows the optimal path through historical data.

**The Goal:** Train the Policy (`pi`) and Value (`v`) networks to mimic this Oracle perfectly.

1.  **Generate Data:** Run the Oracle on your historical/synthetic dataset.
    *   Input: `Observations(t)`
    *   Label: `Optimal_Action(t)` and `Realized_Return(t)`
2.  **Train:** Standard Supervised Learning.
    *   `Loss_Policy = CrossEntropy(Predicted_Action, Oracle_Action)`
    *   `Loss_Value = MSE(Predicted_Value, Oracle_Return)`
3.  **Why this works:**
    *   Stable gradients (ground truth labels, not shifting bootstrapping).
    *   Massive batch sizes possible (no MCTS overhead during training).
    *   Result: An agent that is 80-90% optimal purely on "intuition."

**Codebase Adaptation:**
*   Use `train_jax_fast.py` with a new flag `--imprint-only`.
*   Disable MCTS (`sims=0`) and Replay Buffer.
*   Feed `expert_buffer` batches directly to the trainer.

## Phase 2: The Dream (Fixed World Model)

MCTS requires a stable simulator to "dream" about the future. Learning this online causes instability.

**The Goal:** Learn `State(t) -> State(t+1)` and `State(t) -> Reward` perfectly.

1.  **Pre-train Dynamics:**
    *   Freeze the Encoder (from Phase 1).
    *   Train the Dynamics network (2-layer MLP: 256→1024→529) to predict the next latent state and immediate reward.
    *   Loss: `Similarity(Predicted_Next_Latent, Real_Next_Latent) + MSE(Predicted_Reward, Real_Reward)`.
2.  **Freeze It:** Once trained, **lock the weights**.
3.  **Why this works:**
    *   Prevents "representation collapse" where the model creates shortcut solutions.
    *   MCTS search becomes consistent because the "physics" of the dream world don't change between steps.

**Codebase Adaptation:**
*   Use `diff_teacher` (Bella) or the existing `dynamics` network.
*   Train on a large corpus of market data (L0-L14).
*   Save checkpoint as `world_model_frozen.ckpt`.

## Phase 3: The Thought (Inference-Only MCTS + Autodidact Self-Play)

Now we turn on the "System 2" brain, but **only at runtime**.

**The Goal:** Improve upon the Phase 1 Intuition using the Phase 2 Simulator — then surpass the oracle entirely through self-play.

1.  **Runtime Search:**
    *   Agent sees state `s`.
    *   Uses Phase 1 Policy to propose candidate actions (priors).
    *   Uses Phase 2 Model to simulate outcomes of those actions.
    *   MCTS aggregates values to find the action with the highest expected utility.
2.  **No Backward Pass:**
    *   We do **NOT** backpropagate through the MCTS search tree (unlike MuZero).
    *   We trust the search result and execute it.
3.  **Why this works:**
    *   Saves massive VRAM (no need to store gradients for the tree).
    *   Allows deeper searches (50-100 sims) on a 3090 because inference is cheap.
    *   If the Oracle (Phase 1) missed a complex trap, MCTS (Phase 3) will spot it in the simulation.

### The Autodidact Mechanism

**Research basis:** *"Improving Interactive In-Context Learning from Natural Language
Feedback"* (Klissarov et al., Google DeepMind, 2025; `docs/arXiv-2602.16066v1/`).

Key finding (RL²F): When a student model is trained to not only follow a privileged
teacher's guidance but also to **predict what the teacher would say**, the student
internalizes the teacher's reasoning. At inference, the student can self-correct
without the teacher present — becoming an "autodidact." Remarkably, the autodidact
**outperforms** the original teacher-student setup.

**How this maps to Foxglove:**

Our oracle (path_solver, 20-step greedy lookahead) is the "privileged teacher" — it
sees future returns that the model cannot. In Phase 1, we train the model to imitate
the oracle's actions (policy targets) and predict remaining value (value head). But we
also train an auxiliary **oracle mask prediction head** (TASK_695) that predicts
*which assets the oracle would select* given only the current observation.

This is the "predict the teacher's feedback" objective from RL²F. During Phase 1,
the head learns an internal model of oracle reasoning. During Phase 3 self-play:

```
Phase 1 (teacher-student):
    Oracle → policy targets → model imitates
    Oracle → selection mask → oracle_head learns to predict

Phase 3 (autodidact):
    Model proposes actions via policy head
    Oracle head evaluates: "would the oracle agree with this?"
    Value head evaluates: "what's the expected remaining PnL?"
    MCTS combines all three signals to select actions
    → No external oracle needed
```

The oracle head acts as an internalized critic during tree search. The model
effectively asks itself "what would the oracle do here?" and uses the answer to
evaluate candidate action sequences — replacing the external oracle with an
internal approximation trained on thousands of episodes of privileged guidance.

This creates a self-improvement loop: MCTS search with the internalized oracle
produces actions that are better than the Phase 1 policy alone, which can then
be used as new training targets for further refinement (ExIt cycle).

## Implementation Roadmap

### 1. New Training Mode (`--easy-mode`)

Modify `train_jax_fast.py` to support a tiered workflow:

```bash
# Step 1: Clone the Oracle
python train_jax_fast.py --mode supervised --imprint-only --epochs 50 --save-dir ckpt/phase1

# Step 2: Learn Physics (optional, if not using simple market model)
python train_jax_fast.py --mode dynamics --load-checkpoint ckpt/phase1 --freeze-policy --epochs 20 --save-dir ckpt/phase2

# Step 3: Deployment / Evaluation
python real_trade_jax.py --load-checkpoint ckpt/phase2 --sims 50 --use-mcts
```

### 2. Tuning for 3090

*   **Batch Size:** Phase 1 (Supervised) can handle batch sizes of 1024-2048.
*   **Search Sims:** At inference, you can afford ~50 simulations per step.
*   **Precision:** Use `bfloat16` for everything.

## Debugging "Easy Mode"

If it fails, you know exactly where:

1.  **Agent trades poorly?**
    *   Check Phase 1 Loss. Does it match the Oracle?
    *   If no -> Net architecture too small or Oracle data buggy.
2.  **Agent hallucinates rewards?**
    *   Check Phase 2 Dynamics Loss.
    *   If high -> Dynamics model too weak to capture market moves.
3.  **Search doesn't improve performance?**
    *   Check MCTS parameters (`dirichlet_noise`, `exploration_fraction`).
    *   The "Intuition" might be so good the search is adding noise.

## Summary

*   **MuZero (Hard Mode):** Learn to walk, talk, and predict physics while running a marathon.
*   **Easy Mode:**
    1.  Read a textbook on running (Supervised).
    2.  Study a physics book (Dynamics).
    3.  Run the marathon (Inference).
