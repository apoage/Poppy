# The Sign Bit That Ate a Quarter

> *Observation: the instrument on the bench reads in the negative when the harvest is in the positive. The Apothecary weighs every basket of poppy seed as a debt, and the season's logs list deficits where the soil has plainly produced surplus. The garden is not failing. The scale is reversed. I record the contradiction and look for the screw.*

---

It is the late morning of January 31, 2026. `workstation` is on. The desk fan is at its low setting because nothing is in flight on the GPU — I have killed the run and I am reading the log file with the kind of patient anger that the project will end up calling, by May, *the right kind of audit posture*. The line on the screen reads:

```
epoch=27  reward=-30.991934  alpha=-4.624  market_return=2.36  optimal_alpha=3.15
```

Four numbers, in the calm typesetting of an `OUTLINE = epoch_log_fmt(...)` line. The first one says the agent's discounted cumulative reward for the episode was minus thirty-one. The second says the agent underperformed the market by 4.6%. The third says the market itself was up 2.36% — a normal day, nothing special. The fourth, the one that is going to ruin the next twelve hours of work, says that the optimal hindsight oracle — the agent's teacher, the path solver that *knows the future* and computes the best possible trade sequence — captured only 3.15% Alpha.

I read the line three times. The oracle knows the future. The oracle is allowed to be lazy and just hold the market — that floor is +2.36%. The oracle should *never* report less than the market return, because if nothing else worked, it could have done exactly what the market did. Yet the oracle is claiming +3.15%, which is barely above the floor, and the agent is being graded against this allegedly-optimal benchmark with a reward signal that ate −30 units in one episode.

Two things are wrong. The first is that the agent's per-episode reward is roughly fifteen times worse than a typical episode (typical is around −2). The second is that the oracle's optimal Alpha is *suspiciously close* to the market return, when in a healthy run it usually sits several percentage points above. Both of these problems are about *signs* — about which direction is which, about which numbers should be positive and which negative, about whether the optimization target is pointing at the goal or pointing away from it.

I do not know that yet. What I know in this moment is that something has been silently wrong for at least an epoch, possibly for an entire training run, and I need to find it. So I open a new file, `tasks/TASK_126_REWARD_SPIKE.md`, and I start writing down hypotheses.

This is the article about what happened in that file. The short version: I will spend the next thirteen sub-tasks chasing a sign bug that turns out not to be a sign bug, and the real culprit turns out to be one hard-coded default — a 10-basis-point transaction cost that punishes the oracle for *being an oracle*. The longer version is that I will learn something specific about which assumptions deserve periodic re-verification, and the lesson will outlive the bug.

## What the path solver is, and what it is for

To understand why the oracle reading 3.15% on a 2.36% market keeps me up, you have to know what the oracle is.

In Poppy — the MuZero+TRM trader that lives at `mcts_trader/` — the agent is trained against three sources of signal. One is the environment reward (log returns, transformed through `symlog`). One is the predicted-value-head bootstrap target. The third is *the oracle*: a hindsight path solver that looks at the future window of returns the episode actually saw, computes what the *optimal trade sequence over those returns* would have been, and returns both the resulting Alpha and the per-step selection mask. That mask becomes a supervised target: the agent's policy is nudged toward what the perfect hindsight trader would have done. (This is one of the load-bearing ideas of imprinting / expert-iteration training. The oracle is the teacher, the agent is the student, and the loss is the gap between them.)

The path solver lives at `mcts_trader/jax_agent/path_solver.py`. The greedy variant, `greedy_lookahead_path`, is the one in use during this run. It takes a returns matrix of shape `(T, n_assets)` — log returns per step, per tradeable asset — and computes a `(T, n_assets)` selection mask along with a scalar `total_alpha`. The selection mask is the policy target. The total_alpha is the metric.

The math, when you boil it down, is two steps per timestep:

1. **Score** each asset by the cumulative absolute return it would yield over the next `lookahead` steps. Pick the top-k by score.
2. **Compute PnL** for the current step as the dot product of the selection mask with the per-step returns, accounting for a transaction cost on positions that changed since the last step.

The crucial design decision sits in step 2. The oracle is a *long/short* oracle — it can go either direction on any selected asset, because it knows whether the return is positive or negative. In code, this lets you write the per-step PnL one of two equivalent ways:

```python
# Sign-aware form:
signs    = jnp.sign(step_returns)
step_pnl = jnp.sum(signs * step_returns * new_mask)

# Abs form (mathematically equivalent for a perfect oracle):
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask)
```

For a *perfect* oracle that always picks the right direction, these are identical: `sign(r) * r = abs(r)`. The codebase uses the second form. There is a `signs = jnp.sign(step_returns)` line in the function and the variable is then never used. The comment above it says:

> `# For simplicity, assume we go long on positive returns, short on negative`
> — `mcts_trader/jax_agent/path_solver.py`, the line that will become the central exhibit of the next thirteen sub-tasks

This is the setup. A line of code that *says* it tracks direction. A variable named `signs` that gets computed and discarded. A `jnp.abs` that quietly assumes the variable would have done what the comment promised. And a binary selection mask that records *which* assets are held but not *which direction*.

If the sign convention is wrong, the oracle's target points the wrong way and the agent is being trained to do the opposite of the right thing. If the sign convention is *not* wrong, the `abs` is harmless and the bug is elsewhere. From the cold start of the audit, you cannot tell which is true. You have to read the file with the question front of mind.

## The chase — what I tried, what didn't fix it

What follows is the actual chronology, sub-task by sub-task. I will not pretend it was linear. It branched, it doubled back, and at one point I had three competing diagnoses open in three different files of the same task folder. The thirteen sub-tasks are real: `TASK_126a` through `TASK_126m`. They sit in `tasks/` at the time of writing. You can read them. You should — the article is the compressed version.

**126a — Reward spike investigation.** First hypothesis: the −30.99 is the environment's reward computation going off the rails when the portfolio crashes. The reasoning is that `log_return = jnp.log(safe_new / safe_prev) * 100.0` is unbounded before the `symlog` transform — if the portfolio touches the bankruptcy floor (`prev_portfolio_value` is clipped to `1e-8` to prevent division-by-zero, but that clipping makes the log return *enormous* when the floor is hit), the per-step reward explodes, and accumulating 400 explosive steps gives a −31 episode total. Verdict: this could explain part of it, but the *oracle* anomaly (optimal_alpha < market_return) it cannot explain. *"The symlog transformation correctly compresses extreme values, but the issue is that moderately negative per-step rewards (-0.15) compound over 400 steps to produce the -30.99 MC return"* — `TASK_126a_REWARD_SPIKE_INVESTIGATION.md`. The reward spike has multiple sub-causes; clipping `log_return` and guarding `jnp.where` branches against NaN both help, neither is the whole story.

**126b — Reward drop fix.** Second hypothesis, from a different sub-agent: there are three independent scale bugs colliding. The `alpha_relative_reward` logic in `trainer.py` divides percentage-scale rewards by raw-scale `max_alpha` (1.0 / 0.03 = 33× inflation). The `scores` aggregation in `self_play.py` sums *all* steps in the trajectory, including post-termination padding steps where `is_liquidated` is true and the reward is `symlog(-50) ≈ -3.93` per step (phantom rewards after bankruptcy). And the random-burn ordering in the env updates `prev_portfolio_value` *before* applying the burn, so the next step's log return is computed against an artificially high baseline. Three bugs, one symptom. Verdict: all three are real, all three contribute to the magnitude of −30.99, but again, *none of them explain why the oracle is reading 3.15 on a 2.36 market*. The oracle's pathology is independent of the reward-computation pathology.

**126c — Path solver direction bug (DIAGNOSED).** Now the focus shifts to where I suspect it has belonged the whole time. The hypothesis: the `signs` variable in `greedy_lookahead_path` is computed and never applied. *"`signs` is computed but never used"* — the comment claims long-on-positive, short-on-negative; the code does `jnp.abs(...) * binary_mask` and doesn't track direction. Beyond that, the binary mask cannot represent a *direction flip* — if asset A goes from a +5% move to a −3% move, the oracle needs to flip from long to short, but the transaction-cost calculation sees `|1 − 1| = 0` changes and charges nothing for the flip. *"When an asset stays selected but needs to flip direction (long→short or short→long), the current code counts zero transaction cost"* — `TASK_126c_PATH_SOLVER_DIRECTION_BUG.md`. The proposed fix is a directional mask in `{-1, 0, +1}` and PnL computed as `sum(returns * signed_mask)` with the long/short choice baked into the sign.

This *feels* like the bug. The comment promises direction handling; the code doesn't deliver it. The selection criterion is `cumulative_abs_returns`, which contains no information about which direction is profitable. Everything lines up. I am ready to write the fix.

**126d — Metric scale mismatch.** While the path-solver fix is being drafted, a parallel pass surfaces something orthogonal and uncomfortable. The `market_return` metric in `self_play.py` is a *linear* percentage return: `(end - start) / start * 100`. The `optimal_alpha` metric in `trainer.py` is summed from *log* returns: `log(end/start) * 100`. For an 8% move, linear says 8.00 and log says 7.70 — a 0.30pp gap baked into the comparison. For a 50% move the gap is 9.45pp. The two numbers I have been treating as comparable have never been in the same units. Verdict: this alone could explain why `optimal_alpha < market_return` in mild positive episodes, *without any sign bug at all*. Suddenly the central exhibit of 126c is no longer the only suspect. *"Comparing optimal_alpha (log scale) directly to market_return (linear scale) is mathematically unsound for volatile assets, although they converge for small returns"* — `TASK_126l_return_scale_mismatch.md` (which arrives at the same conclusion a day later via a different audit thread).

**126e — Numerical stability audit (first pass).** Parallel-tracked while 126c and 126d are being chewed on. The audit walks `trainer.py`, `path_solver.py`, `jax_trading_env.py`, and a half-dozen exploration utilities looking for divide-by-zero, `log(0)`, `sqrt(negative)`, `exp(overflow)`, and `jnp.where` branches with NaN-producing computations on both sides (the JAX pitfall: `where` evaluates both branches and propagates NaN even from the unselected branch). It finds six real issues — `epsilon_greedy.py:39` unguarded normalization, `dirichlet.py:13` same, `fast_generator.py:317` unclipped `exp`, `observation_encoder.py:224` unguarded `sqrt(var)`, `tree_search.py:457` same, `jax_trading_env.py:1202` unguarded `log_return` feeding a `jnp.where`. Real issues. None of them are *the* issue. They get fixed anyway; they will quietly prevent NaN propagation in some future run that I will never know about. This is what audits do. They make the day you don't have happen.

**126f — Catastrophic reward drop fix.** A return to the reward-spike track, this time with a concrete plan: clip per-step rewards to the `VALUE_SUPPORT` range `[-10, 10]` before they leave the environment; add `isfinite` guards to suppress NaN before it reaches the value head; raise the bankruptcy floor from `1e-8` to `1.0` so a wiped portfolio reads as $1 rather than as ten-to-the-minus-eight dollars (which is the root cause of the unbounded `log_return`). These fixes are clean, they are local, they harden the env. They do not touch the oracle.

**126h — Path solver PnL bug fix.** The implementation pass on 126c. This is the long file. It walks through, with examples, what a "directional mask" would look like; it proposes a signed selection that scores each asset by *direction-aware expected PnL* rather than absolute magnitude; it adds direction-flip accounting to the transaction-cost calculation. *"`signs` is computed but never applied to the PnL calculation. `jnp.abs(step_returns)` is always positive. The mask new_mask selects which assets to trade, but not which direction. This means the oracle assumes it always makes money on selected assets, even when it picks the wrong direction!"* — `TASK_126h_Path_Solver_PnL_Bug_Fix.md`. The fix is drafted. The unit tests are sketched. I am about a `git commit` away from landing it.

**126i — Return scale mismatch fix.** The implementation pass on 126d. Standardize on log returns across the stack, fix `market_return` to compute `np.log(end_prices / start_prices)`, fix `agent_return` to compute `np.log(new_value / initial_value)`, document the convention, retire the linear-percentage version. This is straightforward, the patch is small, the test cases are obvious (round-trip 100 → 110 → 100 should be exactly 0% under log returns and 0.91% under linear, the log version is the one that doesn't lie).

**126j — Numerical stability audit (second pass).** A re-audit by a different sub-agent, mostly confirming 126e, surfacing one or two new edge cases. Belt-and-suspenders. Nothing dramatic.

**126k — Path solver sign bug audit.** And this is the one that breaks the chain.

A late-arriving sub-agent reads the same code as 126c and 126h and writes the dissent. *"`jnp.abs(step_returns)` is equivalent to `jnp.sign(step_returns) * step_returns`. This calculation correctly represents the PnL of a Long/Short oracle that goes Long when returns are positive and Short when returns are negative. The `signs` variable is indeed computed but unused, which is a code smell but **not a logic error** for a Long/Short oracle."* — `TASK_126k_path_solver_sign_bug.md`. The math: for a perfect oracle that always picks the right direction, `abs(r) = sign(r) * r` by construction. The `signs` variable being unused does not change the PnL; it just makes the long/short assumption implicit instead of explicit. The bug 126c and 126h flagged is *not* a bug. It is dead code. The fix they propose would change nothing.

I sit with this for an hour. The agent is right. I write the test that would distinguish "sign bug" from "no sign bug" — feed a tiny synthetic returns matrix with known optimum, run `greedy_lookahead_path`, check whether `total_alpha == sum(|returns|)`. It does. The oracle is mathematically correct. There is no sign bug.

So what *is* the bug? 126k's last paragraph:

> *"The performance drop is likely due to the **default transaction cost of 10bps** punishing the greedy oracle's high turnover."*
> — `TASK_126k_path_solver_sign_bug.md`

The path solver has `transaction_cost: float = 0.001` — ten basis points — as a default function argument. The oracle is a *greedy lookahead* solver; on minute bars with a small lookahead, it churns positions constantly. Each churn costs 10bps. The per-step return on minute bars is often *less than* 10bps. Mathematically: an oracle that pays 10bps on every move, and selects on a per-step basis, will frequently realize less Alpha than a do-nothing buy-and-hold strategy that pays zero transaction cost.

The oracle isn't broken. The oracle is being charged a fee its design assumes it will not have to pay.

**126l — Return scale mismatch (second audit).** Re-confirms 126d / 126i. Recommends log returns across the board for the same reasons (additive over time, symmetric under inversion, the natural reward space for RL). This lands.

**126m — Numerical stability audit (third pass).** Final sweep. Confirms 126e and 126j. Nothing new.

The thirteen sub-tasks compress into a verdict matrix in `TASK_126_COMPILED_FINDINGS.md`. Six findings confirmed by multiple agents (post-done rewards, alpha-relative scale bug, log/linear mix, two edge-case stability fixes, the txn-cost issue). One finding *disputed and overturned*: the sign bug. The compiled verdict on `signs` reads:

> *"126k is correct. For a perfect oracle: Long positive return: profit = +return = abs(return). Short negative return: profit = -(-return) = abs(return). The `signs` variable IS dead code, but the math is correct."*
> — `TASK_126_COMPILED_FINDINGS.md`

The sign bug ate a quarter of the audit's calendar time. It was not, in the end, a bug.

---

> *The Apothecary turns the suspect scale around in the light. There is no defect. The reagent on it has been weighing correctly the entire time. The deficit in the season's log is not from the scale; it is from a fee the herbalist has been paying to the scale's caretaker, a fee the herbalist forgot was on the contract. The Apothecary settles the fee, returns the scale to the bench, and resumes the harvest. The contradiction is recorded with the same care as a discovery.*

---

## The kickback, in three layers

The first layer is the cheap one. *The bug was not where the audit said it was.* Two sub-tasks — 126c and 126h — wrote thousands of words and proposed fully-drafted code changes for a sign convention that turned out to be mathematically equivalent to what was already there. A third sub-task, 126k, reading the same file from a slightly different angle, dispatched the entire diagnosis in one paragraph of algebra. The corrective sub-task is shorter than either of the misdiagnosed ones. This is the normal shape of audits that don't yet know what they're looking for: a lot of motion, much of it confidently in the wrong direction.

The second layer is heavier. *The actual culprit was a default function argument.* Ten basis points of transaction cost, hard-coded as `transaction_cost: float = 0.001` in the signature of `greedy_lookahead_path`. Not a math bug. Not a sign bug. Not a NaN bug. A configuration default, baked in at the time the function was written, that no one had revisited since the oracle's turnover regime changed. The fix is one line: expose `transaction_cost` to the trainer config, set it to `0.0` for "theoretical max Alpha" runs, match the agent's actual fee structure for "realistic regret" runs. Total diff: smaller than the directory listing of the sub-task folder that diagnosed it.

The third layer is the one that takes the longest to write down. *The sign bug was caught not by the tests, not by the metrics, not by the loss curve. It was caught by reading the code with the explicit question "is the sign convention correct, from scratch, ignoring the comment?" front of mind.*

The unit tests for the path solver tested both directions equivalently, because `abs(r)` is symmetric in the sign of `r` — a test that fed positive returns and a test that fed negative returns both got back `total_alpha == sum(|returns|)`, both passed, neither distinguished "correct sign convention" from "no sign convention at all." The training metrics — `loss`, `value_loss`, `policy_loss` — looked normal because the loss curve descends regardless of which direction the supervised signal points, as long as the signal is *consistent*. (A perfectly wrong oracle, training the agent to do exactly the opposite of the right thing, would produce a clean descending loss curve, because the agent would learn to do exactly the opposite of the right thing. The loss is descending. The agent is converging. The agent is converging to the wrong policy.) Only the *deployment Alpha* — what the agent actually does when it trades — would have revealed the issue. And deployment Alpha is precisely the metric you don't get during a 12-hour training run that you're auditing because the *training* metrics look weird.

The sign question was caught because someone (eventually, after three passes) sat down with the file and read each line with the prior assumption that *the comment might be lying*. The comment said long-on-positive, short-on-negative. The code said `abs(...) * binary_mask`. The agent reading line 95 with the question "is the comment correct?" can see in five lines of algebra that *the comment is harmless because abs and signed-product coincide for a perfect oracle* — but the agent reading line 95 with the question "does the code do what the comment says?" sees a `signs` variable computed and discarded and writes a 2,000-word bug report.

Both readings are useful. The first one is the audit posture. The second one is the audit hygiene that surfaces *real* dead code. The mistake in the saga was not in reading the file carefully — it was in not closing the loop with the math after the dead code was flagged. The first 1,800 words of 126c are correct: the code is structured suspiciously, the variable is computed and discarded, the mask is direction-blind. The conclusion of 126c is wrong: the structural suspicion is not a logic error, because the math collapses. *Both halves of the audit need to happen in sequence: spot the structural smell, then close the math. Skip the second step and you get a fix that fixes nothing and a calendar bill of a week.*

## What's left

The defaults change. `transaction_cost` is exposed to the trainer config in the implementation phase that follows 126k's verdict. The `signs` variable gets removed, or alternately the PnL line gets rewritten to `signs * step_returns * new_mask` to make the long/short convention *explicit*. Either is fine — the codebase chooses the second, because the next reader of the file should see the convention, not have to derive it.

The log/linear return scale mismatch is fixed. The metric pipeline standardizes on log returns end-to-end. The hard-coded `transaction_cost = 0.001` does not survive the next quarter — it appears as a config-loadable parameter in TASK_141 (the "Hyperactive Agent" task, which arrives in late February and uses the same path solver to enforce a minimum-alpha-threshold below which the oracle declines to trade at all). The numerical stability fixes from 126e/j/m get committed. The reward-spike protections from 126a/f get committed. The post-done score-masking and the random-burn-ordering fix from 126b get committed.

The path solver, when I open it today (May 22, 2026, on the same `workstation`, same single 3090, the desk fan now at the medium setting because the next training run is queued), still uses `jnp.abs(step_returns)` for PnL computation. The line above it now reads:

```python
# Oracle goes long on positive returns, short on negative
# Using abs() correctly captures profit in both directions:
#   Long +5% = +5% profit, Short -5% = +5% profit
```

That comment is the residue of the sign-bug saga. The `signs` line is gone. The convention is made explicit in the comment instead of the code, because the math doesn't need it but the next reader will. Above that, a TASK_133 fix (the "4x leverage bug" — a separate kickback from a separate audit) divides by `k_count` instead of `k` to handle the case where the threshold-filter reduces the effective number of held assets below `k`. Above *that*, TASK_141's `min_alpha_threshold` lets the oracle decline to trade when no asset's expected return exceeds the configured cost. Above *that*, TASK_128's `max_len` static-shape fix prevents the JIT recompilation cascade that used to stall training at epoch 60. Each layer is the residue of an audit. None of the layers is large. All of them are load-bearing.

The deeper habit survived. Sign conventions get checked from scratch at every audit boundary. The question "what does this variable mean, ignoring the comment?" is the question that pays off, not "is the comment correct?" The audit posture that emerged from TASK_126 is the one that catches the *next* sign-adjacent issue — the log/linear mismatch in `optimal_alpha` vs `market_return`, which is structurally identical to a sign bug (two quantities that should be comparable are measured in incompatible units), gets diagnosed three days later in eleven minutes by an agent that has already learned, from this file, to assume nothing about return-type conventions until it has read the formula.

What didn't survive was the assumption that long-standing code is correct because the loss curve is descending. The loss curve will descend through any consistent supervision, even consistently wrong supervision. The metric that matters is the *deployment metric* — the thing the agent does in the world that the agent has not been trained to do directly — and that metric is structurally not available during the audit. The audit has to reason about the supervision *before* the deployment, has to reason from the math and from the convention, has to be willing to read the file as if it were written by a stranger who might have been confused.

One minus sign did not cost a quarter. The *suspicion* of one minus sign cost a quarter. The actual bug was a default function argument that nobody had thought to revisit. The lesson the suspicion left behind is the one I would not give back: when the metric is anomalous and the math is unclear, *read the file from scratch, do not trust the comments, close the loop with the algebra, and check the defaults*. In that order.

The thirteen sub-tasks compress, after the fact, into a one-paragraph instruction. They did not feel like a paragraph at the time. They felt like an audit that kept finding bugs while missing the one that mattered. By the time `TASK_126_COMPILED_FINDINGS.md` was written and committed, the calendar said the end of January, the training stack had moved on to the next problem (TASK_128, the JIT recompilation hang), and the reward-spike investigation had become a directory of thirteen markdown files that I would not look at again until I sat down to write this article.

I am looking at them now. They are useful. They are useful in the way that audits are useful: not by being right on the first pass but by being *complete enough* on the thirteenth that the trail through the wrong diagnoses is preserved. Future-me, three months later, opens the folder, reads the verdict matrix, sees that the sign bug was disputed and dismissed, sees that the transaction-cost default was the real issue, and does not re-litigate. The cost of the saga was paid once. The cost of *re-running the saga* was paid zero times, because the file exists.

> *The Apothecary closes the season's log. The deficit has been re-attributed to the fee. The harvest stands. The scale is returned to the shelf, calibrated and labeled with the date of the audit, the name of the contradiction, and the resolution. The next instrument is taken down.*

---

## Sources

- `tasks/TASK_126_COMPILED_FINDINGS.md` — the compiled verdict matrix across all 13 sub-tasks
- `tasks/TASK_126a_REWARD_SPIKE_INVESTIGATION.md` — initial reward-spike diagnosis
- `tasks/TASK_126b_reward_drop_fix.md` — alpha-relative scale + phantom rewards + burn ordering
- `tasks/TASK_126c_PATH_SOLVER_DIRECTION_BUG.md` — the suspected sign bug (later overturned)
- `tasks/TASK_126d_METRIC_SCALE_MISMATCH.md` — log/linear return-type mismatch (real, orthogonal)
- `tasks/TASK_126e_NUMERICAL_STABILITY_AUDIT.md` — JAX numerical-pitfall sweep
- `tasks/TASK_126f_Catastrophic_Reward_Drop_Fix.md` — reward clipping + isfinite guards + bankruptcy floor
- `tasks/TASK_126h_Path_Solver_PnL_Bug_Fix.md` — the proposed directional-mask fix (would have fixed nothing)
- `tasks/TASK_126i_Return_Scale_Mismatch_Fix.md` — log-return standardization
- `tasks/TASK_126k_path_solver_sign_bug.md` — *the dissent that closed the case*
- `tasks/TASK_126l_return_scale_mismatch.md` — second-pass audit confirming 126d
- `tasks/TASK_126m_numerical_stability_audit.md` — third-pass stability sweep
- `mcts_trader/jax_agent/path_solver.py` — the file with the dead `signs` variable, the `jnp.abs` PnL, and the residue-comment that survives today
