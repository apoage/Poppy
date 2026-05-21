# TASK 770 — Gemini Review of TASK_762/763 (Commit `b4e0158`)

### Finding F1 — P0: Fee penalty omitted from N-step value targets
**Severity**: P0
**Location**: `train_phase2.py:646-647` (inside `SegmentBatchGenerator.get_batch`)
**Bug**: The patch correctly adds `fee_penalty_bps` to the immediate `step_pnl` (line 636) which supervises the reward head. However, the N-step value targets (`per_step_r`) are derived from `oracle_cum_pnl`, which is precomputed *without* the new fee penalty. This trains the reward head on a high-fee regime and the value head on a low-fee regime. MCTS will combine heavily penalized immediate rewards with unpenalized future values, breaking credit assignment and likely causing the agent to ignore the fee penalty (since the long-horizon value dominates).
**Why it's silent**: Both loss components will converge smoothly, and a smoke test won't crash. The mismatch only surfaces as an unexplained failure of the fee-penalty sweep to actually curb over-trading during multi-seed evaluation.
**Fix**: Extract future selection changes and subtract the extra fee penalty from the N-step `future_pnl` before scaling:
```python
            steps = np.minimum(steps, len(oracle_cum_pnl) - 2)
            # Fetch selections to compute future changes
            future_sel = slot['oracle_selections'][steps]
            future_prev_sel = slot['oracle_selections'][np.maximum(steps - 1, 0)]
            future_changes = np.sum(np.abs(future_sel - future_prev_sel), axis=-1)
            
            future_pnl = oracle_cum_pnl[steps + 1] - oracle_cum_pnl[steps]
            # oracle_cum_pnl already includes transaction_cost; only subtract the new extra penalty
            extra_penalty = future_changes * (self.config.fee_penalty_bps * 1e-4)
            per_step_r = (future_pnl - extra_penalty) * self.config.reward_scale
            per_step_r = np.clip(per_step_r, -10.0, 10.0)
```
**Verification**: Run a 1-cell test with `--fee-penalty-bps 10`. Check that `metrics['td_target_mean']` drops proportionally compared to a baseline run with 0 bps.
**Acceptance**: `nstep_sum` correctly reflects the `fee_penalty_bps` for all future steps.

### Finding F2 — P1: Raw N-step value targets are prematurely clipped
**Severity**: P1
**Location**: `train_phase2.py:1155` and `1142`, `1164` (inside `stage_b_train_step`)
**Bug**: `td_target_raw` is clipped to `[-10.0, 10.0]` *before* PopArt normalization. At `reward_scale=200`, a cumulative PnL of just 5% over the 20-step horizon yields a raw target of 10.0. A 5% return over 20 steps is very common, meaning a significant fraction of highly profitable trajectories are completely censored by this clip. PopArt's purpose is to dynamically scale these unbounded raw targets into the `[-10, 10]` categorical support. Clipping the raw targets destroys PopArt's ability to model the positive/negative tails.
**Why it's silent**: At `reward_scale=100`, the clip hits at 10% returns, which is rare enough not to collapse training. At 200, it censors heavily, artificially capping the value head's ability to differentiate great trades from good trades without throwing any NaN or shape errors.
**Fix**: Remove the `td_target_raw = jnp.clip(td_target_raw, -10.0, 10.0)` lines at 1142, 1155, and 1164. The normalized targets (`td_target_norm`) are correctly clipped to `[-10.0, 10.0]` immediately after PopArt processing, which is the only place clipping is required for the categorical support.
**Verification**: Run a cell with `reward_scale=200`. Observe that `metrics['td_target_mean']` and variance widen appropriately and PopArt `var` scales up to accommodate, rather than artificially capping.
**Acceptance**: `td_target_raw` is unclipped; `td_target_norm` remains clipped.

### Finding F3 — P2: Argparse %-format bug in pre-existing help strings
**Severity**: P2
**Location**: `train_phase2.py:2307-2308`
**Bug**: The help strings for `--mcts-temperature` and `--epsilon-greedy` contain single `%` characters (`mix≈0.7%` and `mix≈4.7%`). `argparse` attempts to interpret these as format specifiers and throws `ValueError: incomplete format key` when `python train_phase2.py --help` is invoked.
**Why it's silent**: The workstation operator only hit it by manually asking for help; automated sweeps bypass `--help`.
**Fix**: Replace the single `%` with `%%` on lines 2307 and 2308 to properly escape them.
**Verification**: `python train_phase2.py --help` prints successfully without crashing.
**Acceptance**: Help text displays `mix≈0.7%` correctly.

---

### RULED OUT Findings
*   **P0 - Reward double-scaling**: RULED OUT. `reward_scale` is applied exactly once to both step rewards and N-step returns. PopArt dynamically tracks the scale variance, and `benchmark_easy.py` extracts it correctly via `popart_state['var']` for MCTS. No double scaling occurs.
*   **P0 - Entropy sign logic**: RULED OUT. `policy_loss = policy_loss - entropy_bonus + floor_penalty` adds the penalty when entropy drops. Adding a penalty means the optimizer will push entropy up to minimize the loss. The direction is correct and encourages exploration.
*   **P0 - Joint floor dead code**: RULED OUT. The target `mcts_visits` max entropy is 3.46 (32 sims), and empirical is ~1.0-1.5, so the policy will reliably drop below the 3.0 floor, meaning the penalty branch is active.
*   **P1 - Checkpoint load mismatch**: RULED OUT. `train_phase2.py` does not resume Phase 2 checkpoints via `--checkpoint` (it only loads Phase 1 bases). If a Phase 2 checkpoint *were* loaded via the `else` branch (no `config.json`), `Phase2Config.__dataclass_fields__` cleanly preserves the new fields.
*   **P2 - Sweep-cell baseline bit-identical**: RULED OUT. The `else: policy_entropy = jnp.array(0.0)` simply logs a zero metric; it doesn't perturb the gradients or loss sum, keeping the baseline bit-identical.
