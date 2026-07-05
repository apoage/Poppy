# Interaction Analysis

## 1. Oracle Targets vs. Student State (TASK_673 vs TASK_692)

**Conflict**:
- **TASK_673** (and base logic) generates targets based on the Oracle's optimal path. The Oracle always takes the optimal action, so its position sequence is perfect.
- **TASK_692** generates the state trajectory based on the *Student's* model actions. The Student makes mistakes.
- **Result**: Divergence.
  - Step T: Oracle is Flat. Student is Long.
  - Oracle sees opportunity, actions "Buy".
  - Student (being Long) *cannot* Buy (or shouldn't).
  - Training Pair: Input=(Student Long), Label=(Buy).
  - Model learns: "When Long, Buy". This is noise/garbage.

**Resolution**: `policy_targets` must be re-computed using the Student's position history (`prev_selections`) as the constraint, NOT the Oracle's position history.

## 2. Config vs. Spec (TASK_151 vs TASK_677)

**Conflict**:
- **TASK_151** introduced "Signal Soft Start" (warmup=50) to prevent gradient shock.
- **TASK_677** ("Turtle Breaker") apparently set `weight_warmup_epochs=0` to force early trades.
- **Result**: Gradient shock is back. With the high trade weight from TASK_149 (~63x), early training steps (where model is random) produce massive gradients, potentially destabilizing the weights before the model learns basic features.

## 3. Capacity Regression (TASK_684)

**Conflict**:
- **TASK_684** allowed decoupling `trm_hidden_dim`.
- **Large Preset** utilized this to *reduce* the dimension from implicit 2048 to explicit 1024.
- **Result**: Large model has less reasoning capacity than it should.
