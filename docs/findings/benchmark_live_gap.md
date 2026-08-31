# benchmark live gap

> Benchmark uses argmax(policy_logits); live trader uses MCTS. Entire 14-evaluation campaign scoreboard is argmax-based, not deploy-truthful. TASK_759 P1' tests whether the gap is material.

*Migrated from local agent memory 2026-08-31 (`project_benchmark_live_gap.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
Discovered during TASK_759 P1 Phase-1 review (lore §150, 2026-05-19):

- `tools/benchmark_easy.py:493, 561` uses `jnp.argmax(policy_logits)` for action selection — **no MCTS at eval**.
- `real_trade_jax.py:131, 326` constructs `MuZeroMCTXSearch` and calls it for action selection at every step — **live trading uses MCTS**.

**Implication**: every campaign verdict to date (14 evaluations, 0 promoted under paired Δα) was measured with argmax. If MCTS at eval materially shifts Δα, the entire scoreboard could look different under the deploy-truthful operator.

**Test in flight**: TASK_759 P1' at `experiments/759_mcts_eval/SPEC.md`. Pure benchmark sweep on existing 741_mint checkpoints — no retraining. Variants: sims ∈ {0, 16, 32, 64} × decode ∈ {joint, per_asset}.

**Implementation**: commit `bcc5be1`. `--eval-mcts-sims N` and `--eval-mcts-decode {joint, per_asset}` flags in `benchmark_easy.py`. Workstation smoke passed (argmax regression diff 3.34e-05, MCTS@16 bit-identical determinism across two runs).

**Why this is load-bearing**:
- If eval-MCTS lifts Δα significantly → adopt as new benchmark default, recalibrate historical campaign verdicts, then resume mechanism probes under corrected baseline. Possibly some "not promoted" mechanisms re-promote.
- If eval-MCTS leaves Δα flat → argmax was a fine proxy. Move on to TASK_759 P2 (value-target MC mixture).

Either outcome is publishable for Bloom — "we were measuring the wrong operator for an entire campaign" is a strong lore arc.

**How to apply**:
- Until P1' returns, treat all paired-Δα numbers in the campaign scorecard as provisional pending recalibration.
- Any new mechanism SPEC should reference this — once eval-MCTS becomes default, baselines may need re-running under the new operator.

**Related**: [encoder_bella_class_closed](encoder_bella_class_closed.md) (the closed Bella verdicts were also argmax-based; may or may not survive recalibration), [classifier_alpha_anti_correlation](classifier_alpha_anti_correlation.md) (the pattern was observed under argmax — verify under eval-MCTS).
