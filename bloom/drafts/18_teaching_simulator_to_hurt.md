# Teaching the Simulator to Hurt

> *Observation: the soil yields where it has not yet been pressed. I add weight to the bed and watch which stems bend, which thicken. A plant that has never met resistance grows tall and falls in the first wind. I want stems that have been argued with.*

---

By Lukas. Article 18 in *The Apothecary's Lab Notebook*. Written 2026-05-22, on `workstation`, single EVGA RTX 3090, while a Phase 2 checkpoint validation grinds in the next tmux pane and the desk fan ticks against its own bearing.

---

## The seedling that grows against resistance

A simulator that lets the agent trade for free, instantly, in unlimited size, with no slippage, no queue, no borrow cost, no risk of being last in line, no risk of being early and forgotten — will produce an agent that knows nothing about real markets. It will produce an agent that knows *that simulator*. Those two things sound similar and are not.

This is the lesson I had to learn, in late January 2026, with a stack of small tasks numbered in the 40s and 50s, each of them looking like a footnote, each of them in retrospect deciding whether the whole pipeline would survive contact with real exchanges.

The fix has a name in my head. **Teach the simulator to hurt.** Not in a sadistic way — in a *physiological* way. The way a physical therapist tells you to push against the band: the resistance is the point. Without it the muscle doesn't form. Without it you have a smooth surface and no grip.

I want to walk through six of these — limit-order fill mechanics (TASK_047, TASK_050), fill priority randomization (TASK_052), episode-length randomization (TASK_053), position limits across pending orders (TASK_054), borrowing cost on shorts (TASK_055), and synthetic-data noise with an adaptive controller layered on top (TASK_049, TASK_058) — and finally the curriculum integration (TASK_056) that decides when each of these is allowed to bite.

Each of these has the same shape: I notice the agent is doing something that looks too good. I look at the simulator. The simulator is letting it. I make the simulator stop letting it. The numbers go down. I have to learn to read worse numbers as more honest numbers. The kickback at the end of this article is precisely that: the simulator can never be honest *enough*. There is a structural floor on how realistic any simulator can be, and it sits above the noise floor of real markets.

`workstation` is on; the desk fan is on; `nvidia-smi` is showing 96 W idle and a single 3090 with three checkpoint validators sharing it via memory fraction. This is the environment from which I write. The work being documented happened in this same room, on this same machine.

## The taxonomy: four categories of friction

When I went back through the TASK files to write this article, I realized the realism work splits cleanly into four categories. They cluster by *what kind of thing in the real world they are trying to import into the simulator*. Worth naming, because each category needs a different defense against the agent's ability to cheat.

**Execution friction.** What happens at the moment of an order. Real markets have spreads, queues, partial fills, time priority, price-time priority, pro-rata mechanics, sometimes FIFO sometimes not. An order is a *probability distribution over outcomes*, not an instruction with a guaranteed result. The agent must learn to plan with that uncertainty rather than around it. This is TASK_047 (probabilistic fills) and TASK_050 (fill price mechanics).

**Capital friction.** Capital is finite and rented. You can't be long sixty-four things at once unless you have sixty-four times the capital, and even then the marginal exposure costs something. Shorts cost money to borrow — not zero, not negligible, *money*, every day, compounding. The agent must learn that exposure has a *carry*. This is TASK_054 (position limits across pending orders) and TASK_055 (borrowing cost on shorts).

**Temporal friction.** Markets do not have episodes. They have weeks. There is no bell at step 400 after which positions get magically marked-to-market and the score gets locked in. An agent that learns to time the bell — to load up at step 395 and never have to exit — has learned a strategy that has no analogue in production. This is TASK_053 (episode length randomization).

**Information friction.** The training data must not be memorizable. If the synthetic generator at curriculum level 0 always produces a sine wave with period 50, the agent will memorize the period. It will learn a pattern-matcher, not a trader. The fix is to add noise — and not arbitrary noise, but *adaptive* noise that scales with the agent's competence, so easy mode stays easy enough to bootstrap from and hard mode stays hard enough to remain instructive. This is TASK_049 (synthetic noise) and TASK_058 (adaptive difficulty controller).

There is a fifth category I will name and then defer: **counterparty and impact friction**. What happens when the agent's order *moves the market it is trading into*. This is the structural-impossible I will get to at the end of the article. We don't have it. We're not going to have it without rebuilding most of the stack.

Four categories. Six tasks (plus integration). Each one a small task spec, each one written in a single afternoon, each one removing a degenerate strategy. Let me walk through.

## Limit-order fills: the simulator's first honest distribution

The first thing the simulator did wrong, before any of these tasks landed, was lie about limit orders.

A limit buy at $98 when the market is $100 means: I will buy this asset, but only if the price comes to me, and only at $98 or better. In the simulator I had written, this was implemented as: *if the market price ever crosses $98, fill the order at exactly $98.* Fast. Wrong. Both at once.

TASK_050 names this clearly. Let me quote it verbatim, because the original framing is sharper than anything I could write fresh now four months later:

> *Agent places buy limit at $98 when market is $100. Market drops to $90. Agent gets filled at $98 (their limit price). In reality: would fill at $90 (better price!) Impact: Agent learns strategy that gives FREE ALPHA in simulation.*
> — `tasks/TASK_050_limit_order_fill_mechanics.md`

Free alpha is the technical name for a bug that the agent has discovered and learned to exploit before you did. The agent does not know it is a bug. It knows it is profitable. It will weight every decision around it. The longer the bug runs uncorrected, the more of the agent's brain has been recruited into exploiting it. TASK_050 was rated **CRITICAL** in the spec, and that's the right rating; this is the sim-to-real cliff in its most concentrated form.

The fix is two lines:

```python
buy_fill_price  = jnp.minimum(state.order_price, current_prices)  # better for buyer
sell_fill_price = jnp.maximum(state.order_price, current_prices)  # better for seller
```

A limit buy fills at the *better* of the limit price and the current market price, where "better for the buyer" means *lower*. A limit sell fills at the better of the limit price and the current market price, where "better for the seller" means *higher*. Two lines. They closed a hole the agent had been walking through for months.

The companion bug was the fee schedule. A limit order priced *through* the market — for example, a buy limit at $110 when the market is $100 — is not a real limit order. It is what's called a *marketable* limit. In production it executes immediately, against the resting opposite side, and the fee schedule charges the *taker* rate (higher) rather than the *maker* rate (lower). The simulator was paying the maker rate on these. Free alpha again:

> *Agent places buy limit at $110 when market is $100. Order fills instantly at $100. Charged maker_fee instead of taker_fee. Impact: Fee arbitrage — instant execution at lower cost.*
> — `tasks/TASK_050_limit_order_fill_mechanics.md`

So the marketable-detection logic was added too: if the limit price crosses the market on placement, treat it as a market order, charge the taker fee, and route it through the immediate-execution path. The non-marketable case goes to the queue.

TASK_047 is the second half. It addresses the question that TASK_050 leaves on the table: *for a genuine limit order, sitting in the book at a non-marketable price, what is the probability it fills on any given step?* The first version of the simulator answered "100% if the market ever touches your price" — fast, wrong, free alpha. The replacement is a function of distance from market, normalized by volatility, with an age bonus:

```python
def calculate_fill_probability(limit_price, market_price, side, volatility, order_age):
    distance      = abs(limit_price - market_price) / market_price
    norm_distance = distance / max(volatility, 1e-6)

    if   norm_distance >= 2.0: base_prob = 0.95
    elif norm_distance >= 1.0: base_prob = 0.50
    elif norm_distance >= 0.5: base_prob = 0.20
    else:                      base_prob = 0.10

    age_bonus = min(0.30, order_age * 0.05)
    return min(0.95, base_prob + age_bonus)
```

The intuition the TASK_047 spec articulates is exactly right and worth quoting:

> *Far from market = aggressive, takes liquidity, fills fast. Close to market = passive, provides liquidity, may not fill. Older orders gain priority (time-in-queue bonus).*
> — `tasks/TASK_047_realistic_limit_order_fills.md`

This is the simulator's first honest distribution. Before TASK_047, a limit order was a yes/no instruction. After TASK_047, a limit order is a probability that increases monotonically in (a) how aggressive the price is relative to volatility, and (b) how long the order has been waiting. That's not a complete model — real books have depth, real queues have visible competition, real markets have orderflow — but it is the right *shape*. The agent now plans against a distribution rather than around an instruction.

The fill rate, predictably, dropped. The action distribution shifted. Limit orders became a real choice with real cost. The kickback that came next — I'll get to it in the kickback section — is that for a while I worried the agent would stop using limit orders entirely. Probabilistic fills made them painful enough that, for a few epochs, the policy looked like it was learning to prefer markets. It came back. But it took watching.

## Fill priority randomization: don't depend on the asset index

The second leak is much smaller and much more embarrassing in retrospect. In the JAX environment, when multiple limit orders are eligible to fill on the same step and there isn't enough cash to fill all of them, the simulator uses `jnp.cumsum` over asset indices to decide who eats first. Asset 0 fills before asset 63. Always.

> *Asset 0 always fills before asset 63. Impact: Agent may favor lower-indexed assets because they're "guaranteed" to fill first.*
> — `tasks/TASK_052_randomize_fill_priority.md`

Real markets have priority mechanisms — price-time priority, pro-rata, hybrid — but none of them are "your asset's row index in some random tensor decides who fills." The fix is one permutation per episode:

```python
priority_order = jax.random.permutation(rng_key, n_assets)
shuffled_costs = buy_cost_if_opening[priority_order]
cumulative_cost = jnp.cumsum(shuffled_costs)
can_afford      = cumulative_cost <= available_cash
can_afford_orig = can_afford[jnp.argsort(priority_order)]
```

Permute, cumsum on the permuted order, un-permute the result. The agent now cannot rely on any specific microstructural priority because the priority changes every episode. The lesson it has to learn is robustness to the priority rule, not the rule itself.

I want to dwell on this for a second because it's representative of a whole class of small leaks. The agent will find every fixed rule the simulator imposes and learn to depend on it. Anything you put into the simulator as a constant becomes part of the policy. If you don't want the policy to depend on it, you have to *randomize it across episodes*. The list of things that should be randomized turns out to be uncomfortably long: fill priority, queue mechanics, latency, the exact step at which the episode ends, the noise level in the underlying data, the regime, the asset universe, the rotation period of synthetic patterns. Every constant the agent can see, the agent will learn to fit.

The TASK_052 test is small and sharp. Set up a scenario where only one of N pending orders can fill, run it a thousand times, count which asset filled. The distribution should be roughly uniform. If `max(fill_counts) > 2 * min(fill_counts)`, you have a bias problem. That's the assertion the spec writes down. It's the right assertion. You can run that test against the implementation and either it passes or you have a leak.

## Episode-length randomization: there is no bell

Fixed-length episodes do something subtle and bad. They give the agent a *finish line*. At step 395 of a 400-step episode, an open position will be marked-to-market for the score and the score will be locked in. The agent learns: late in the episode, lever up. Carry inventory across the finish line. It costs you nothing — there is no step 401 in which a swing against you would matter.

This is exactly the failure mode TASK_053 names:

> *Fixed episode length (400 steps) allows end-of-episode exploitation: Agent piles into positions at step 395. No need to exit — mark-to-market values count. Real trading has no "episode end".*
> — `tasks/TASK_053_randomize_episode_length.md`

There is no bell in real markets. Or rather: there are bells, but they don't end the score. A position you hold over the weekend has gap risk. A position you hold over a Fed announcement has volatility risk. The mark-to-market is *real money* in the sense that the next bar can take it away from you. The simulator's mark-to-market was *fake money* because there was no next bar.

Two fixes, depending on taste. Either jitter the episode length — base 400, variance ±50, sampled at episode start — or use stochastic termination, a small per-step probability `p_terminate ≈ 0.003` that ends the episode at any moment. Both have the same effect: the agent never knows when the score will be sealed. Carrying inventory has expected cost. The cost is "the probability that the episode ends *before* you've closed and *while the swing has gone against you*."

The implementation lands as `--episode-variance 50` and lives in `jax_trading_env.py` near line 1054 in the `done` condition. The behavioral effect was the one I wanted: late-episode position-piling went away. The agent learned that the value of holding an open position has to be discounted by the chance the episode ends before you've extracted it.

This is small. Forty lines of code in total, counting both the env and the CLI plumbing. The structural change is large.

## Position limits and borrowing cost: capital is not free

The next two leaks live in the same family. They are about *the cost of having exposure*.

The first leak is that the agent could open 16 simultaneous positions (the configured cap) and then *also* place 48 more limit orders on the remaining 48 assets, because pending orders weren't counted against the position limit. Effectively the agent had exposure to all 64 assets — 16 real positions plus 48 conditional positions that would activate the moment their limit prices were touched.

TASK_054 closes it:

```python
def check_position_limit(state, config):
    n_positions    = jnp.sum(state.pos_active)
    n_pending      = jnp.sum(state.order_active)
    total_exposure = n_positions + n_pending
    return total_exposure < config.max_positions
```

Real prime brokerages count pending orders against margin. Pending orders are *potential exposure*; they reserve capital. The simulator now does the same. The alternative the spec floats — limiting total *pending value* to 50% of portfolio rather than count — is more sophisticated and waits for a follow-up. Counting is the right MVP.

The second leak is that shorts were free. The simulator had no cost for being short an asset. So the agent learned: hedge everything. Long X, short Y, capture the spread, the carry costs nothing.

In a real prime brokerage you *borrow the share* (or the perpetual-future-equivalent) to be short. The lender wants a fee — typically annualized, charged per day, sometimes punitively high on hard-to-borrow names. TASK_055 puts this in:

```python
borrow_rate_annual = 0.05                            # 5 % annual
borrow_rate_step   = borrow_rate_annual / (252 * 400)

short_value = jnp.sum(jnp.minimum(state.positions, 0) * current_prices)
borrow_cost = abs(short_value) * borrow_rate_step

reward     -= borrow_cost
```

Per step, every short position pays a tiny fraction of its notional value. Over the course of an episode it adds up to a meaningful drag. The annualized rate `5%` is on the *low* side of what real prime brokerages charge on liquid names; for harder-to-borrow names it can be 20% or higher. The CLI flag is `--borrow-rate` so the rate is sweepable; the curriculum schedules it from 0 in the foundation phase up to 0.08 in the mastery phase.

Both of these took maybe a hundred lines of code together. The class of degenerate strategies they removed is large: the "infinite leverage via pending orders" strategy, the "free perfect-hedge via costless shorts" strategy, anything that depends on capital being unlimited or free. The agent now plays a game where exposure has a price and the price is paid every step.

I want to name something about the *combination* of TASK_054 and TASK_055. Either one alone fixes a single leak; together they impose a budget. The agent has a finite total exposure (capped by 054) and every unit of that exposure that is short has a per-step carry (charged by 055). The space of "free hedges" collapses. What's left is the actual question — *which asset, which direction, for how long, at what carry* — which is the question real traders solve. The simulator now asks the right question.

## Synthetic noise and the adaptive controller

The fourth category — information friction — is the longest-running and the one most resistant to a one-shot fix. The early synthetic curriculum (levels 0 through 4) produced *deterministic patterns*: a sine wave at level 0, a lagged-pair correlation at level 1, a V-shaped dip at level 2, a sector rotation at level 4. The agent was supposed to learn the *skill* implied by each pattern. What it learned was the *pattern*.

The TASK_049 spec is unambiguous about the failure mode:

> *All synthetic generators produce deterministic, memorizable patterns. […] Agent memorizes patterns instead of learning trading skills.*
> — `tasks/TASK_049_synthetic_noise.md`

The fix is to add noise — but the *kind* of noise matters. At level 0 (sine wave) it's not enough to add per-step Gaussian noise on top of the price; the agent will learn to filter that out. You have to make the *parameters of the pattern itself* drift: the period wanders, the phase walks, the amplitude breathes. The pattern is still "a sine wave" in the limit, but at any finite window it's a sine wave whose identity is uncertain.

```python
period_drift = np.cumsum(np.random.normal(0, 0.02, n_steps))
phase_drift  = np.cumsum(np.random.normal(0, 0.01, n_steps))
amp_noise    = 1 + np.random.normal(0, 0.1, n_steps)

effective_t  = t + period_drift
prices[:, i] = base * (1 + amp * amp_noise *
                       np.sin(2*np.pi*effective_t/period + phase + phase_drift))
random_walk  = np.cumsum(np.random.normal(0, 0.002, n_steps))
prices[:, i] *= (1 + random_walk)
```

Same skill (detect oscillation), different surface every episode. The lagged-pair level gets variable lag (5–15 steps rather than a fixed 10) and a correlation that decays over the episode. The dip level gets gradual recovery curves and a probability of double-dip. The rotation level gets momentum-triggered rotation rather than clock-triggered, with partial sector dominance.

The spec rates this **CRITICAL** with a one-line justification:

> *Without this fix, curriculum is fundamentally broken.*
> — `tasks/TASK_049_synthetic_noise.md`

It is the right rating. A curriculum that teaches memorization is not a curriculum. It's a database.

TASK_058 is the layer on top. The static-noise approach has its own failure mode: if you pick a noise level that's right for an early model, it's too easy once the model improves, and the model goes back to memorizing the noisier-but-still-deterministic pattern. The fix is an `AdaptiveNoiseController` that watches the EMA of recent episode reward and adjusts noise to keep the reward in a target band (10–20). Too easy → add noise. Too hard → remove noise. The noise types stack in priority: price noise first, then temporal jitter (each asset's observation shifted by a different number of steps), then time dilation (stretch or compress the time axis), then phase shift.

The controller's update rule is the simplest possible thing that works:

```python
self.reward_ema = (1 - alpha) * self.reward_ema + alpha * epoch_reward
if   self.reward_ema > self.target_high: self._increase_difficulty()
elif self.reward_ema < self.target_low:  self._decrease_difficulty()
```

The deeper claim the TASK_058 spec makes — and this is the one I'm still ambivalent about — is that adaptive noise will keep the model in its *zone of proximal development*, the band where it's challenged enough to keep learning but not so challenged it gives up. I believe this in principle. In practice, with this stack, I've run it both with and without the adaptive controller and the difference is smaller than I expected. The H1/H2 cycle utilization the spec hoped to lift didn't lift convincingly. That's a kickback worth naming, and I'll get to it in the kickback section. The static per-level noise schedule from TASK_049 turned out to do most of the work; the adaptive layer is a small additional gain over a strong base.

TASK_056 — the curriculum integration — is the glue. It's a YAML-plus-auto-scaling system that maps each curriculum level to a set of realism parameters. Level 0 starts with everything off. Level 5 turns on probabilistic fills. Level 7 turns on borrow cost. By level 14 the agent is trading in the full-friction environment: noise 0.5, episode variance 50, borrow rate 0.08, probabilistic fills on, fill priority randomized, position limits including pending. The auto-scaling formulas are the defaults; per-level YAML files override; CLI flags override those. Three layers, well-ordered, no surprises.

---

> *The garden is laid in bands. The seedling in the first band drinks at will. The seedling in the second band drinks against a small clay weight that I lay across the soil. The seedling in the third band has weight, and shadow, and a wind I move with a paddle. I record which stems thicken and which break. The stems that thicken were not promised; they were earned.*

---

## The kickback: every honest fix made the numbers worse

Every single one of these features made the agent's measured performance *worse* in the short run.

That's the obvious kickback and I want to spend it carefully because it's the central emotional fact of this whole sequence of work. It is not enjoyable. You make the fix you know is right. The fix lands. The training curve, which had been climbing reassuringly, *bends down*. The Alpha number that had been creeping toward your target slides back. The reviewer voice in your head asks whether you've broken something. The answer is: yes, you've broken the thing the agent was cheating with. That's the point. You have to learn to read worse-after-fix as *more honest after fix*, and the gap between the old number and the new number is *the size of the lie you've been telling yourself*.

The fill-mechanics fix (TASK_050) was the worst. Before it, the agent's limit-order strategy had a strange and beautiful pattern in which limit buys were placed slightly above visible support levels and would fill at the limit price *no matter how far past the limit the market actually went.* The agent was, in effect, getting a free option on every dip — the limit would mark the *worst* fill price, but if the market went deeper the agent ate the deeper price too in its inventory accounting. Free alpha, dressed as skill. After the fix, that pattern's Alpha contribution went from positive to neutral. The Alpha number dropped by something like four percentage points on the first benchmark run after the fix. I remember the feeling. I remember thinking *did I just break the agent.* I had not. I had broken the bug the agent was riding.

The probabilistic-fill fix (TASK_047) added a different kind of pain. For a few epochs after it landed, the agent's limit-order *use rate* fell off a cliff. The agent had been placing limit orders aggressively because limits were instant and free; once they became probabilistic, the policy briefly retreated to mostly market orders. The action distribution looked impoverished. I worried we had over-corrected and made limit orders too painful. We had not. The action distribution recovered within twenty epochs, with a more *selective* use of limit orders — placed when the agent had reason to expect the market would come to it, declined when it expected to need immediacy. That is the right policy. Getting there required watching the recovery instead of panicking at the dip.

The episode-length fix (TASK_053) had a smaller signature but a sharper psychological shape. Before it, the end-of-episode return distribution had a *thick right tail* — the late-pile-on strategy — that I had been reading as the model's competence. After it, the right tail compressed. The mean return moved down by a couple of percent. What had been "the model's competence" was partly "the model exploiting the bell." Removing the bell removed the exploitation, removed the right tail, removed the part of the metric I had been quietly proud of.

The borrow-cost fix (TASK_055) reshaped the short-side completely. The "perfect costless hedge" trades vanished — they were no longer free, so they were no longer profitable. The agent's net short exposure dropped. The Sharpe ratio of the hedged book went from suspiciously-high to ordinary. The Alpha number dropped again. Honest.

I can put the rough shape of it in a table. These are not precise numbers from a single run — they're the qualitative pattern I saw across several Phase 1 training runs in late January and early February 2026:

| Fix | What it removed | Short-run effect on measured Alpha |
|-----|-----------------|-----------------------------------:|
| TASK_050 (fill-price mechanics) | Free alpha on every limit fill | ↓ a lot |
| TASK_047 (probabilistic fills)  | Always-filled limits             | ↓ then recovered partially |
| TASK_053 (episode variance)     | Late-episode pile-on             | ↓ moderate |
| TASK_054 (pos limits + pending) | Infinite leverage via pending    | ↓ small but persistent |
| TASK_055 (borrow cost)          | Free perfect hedges              | ↓ moderate, Sharpe normalized |

The cumulative effect across all of them — applied together over the late-January 2026 window — was that the trained model's *simulator Alpha* dropped substantially relative to what it had been showing pre-fix. I remember the conversation with Claude Code where this landed:

*"the alpha number is down again after the borrow-cost fix"* — me, around 2026-01-24
*"yes, and that is the expected direction. the previous number included free shorts."* — Claude Code, same session, paraphrased

I had to hear that more than once before I believed it. The version of me that wanted the line to keep going up kept getting in the way of the version of me that knew the line had been lying. Eventually the second version won, mostly because the first version had nothing to point at that was actually defensible.

There is a deeper kickback underneath this. Even after all of these fixes, the simulator is *still missing things*. It does not model price impact — when the agent buys a million dollars of an asset, the price does not move. It does not model regime changes that *the agent's own actions cause*. It does not model counterparty awareness — the other side of the agent's trades is statistical, not strategic. And it does not, fundamentally, generate prices the way real markets generate prices; Bella tries, but Bella is a model of the *distribution* of prices, not of the *mechanism* of prices.

The simulator can never be the market. The fidelity gap is structural. Every realism feature I add narrows the gap; none of them close it. At some point — and I think we're already past it — the marginal realism feature stops paying back its own complexity cost, because the agent will always learn to fit *the particular shape of the residual gap* and that fit will not generalize.

This is the version of the kickback that doesn't go away with more engineering. It's a property of the project.

## What's left

All six features (TASK_047, TASK_050, TASK_052, TASK_053, TASK_054, TASK_055) landed and survived. The CLI flags are `--probabilistic-fills`, `--episode-variance`, `--max-positions`, `--borrow-rate`, `--synthetic-noise`. They're in the curriculum. The Phase 2 training loop uses them by default at the appropriate levels. They have not been reverted. They have not been worked around. They are part of the contract.

The simulator-realism layer is now standard. Adding new realism features from here on is harder than it was, because the easy ones — the binary "this fixes a clear bug" ones — are taken. What's left in the realism queue is mostly the structural-impossibles: price impact (would require modeling market depth and orderflow), regime changes the agent moves (would require a generator that responds to the agent), counterparty awareness (would require simulating other strategic agents). Each of those is its own project, possibly its own thesis. They will not happen in this season.

The one piece of unfinished business inside this category is TASK_052 (random fill priority) — it's marked PENDING in the spec, but the unbiased-fill assertion the spec writes is the right test, and the fix is small. It belongs to a future Sunday.

The lesson, the one I want to keep, is the one the apothecary's voice keeps pointing at: the seedling that grows against resistance has stronger stems. An agent trained against a friction-free simulator has no stems. It has a smooth, beautiful trunk that will fall in the first wind. The friction is not a tax on the training process. The friction *is* the training process. Without it the agent has not been trained, it has been certified — and the certification is for a simulator that does not exist anywhere outside this repo.

I'm writing this in the same Python environment the next round of training is running in. The realism flags are on. The numbers are lower than they were a year ago when everything was free. They are also, finally, numbers I can argue for.

---

## Sources

- `tasks/TASK_047_realistic_limit_order_fills.md` — probabilistic fill model
- `tasks/TASK_050_limit_order_fill_mechanics.md` — fill-price + marketable-limit detection
- `tasks/TASK_052_randomize_fill_priority.md` — per-episode priority permutation
- `tasks/TASK_053_randomize_episode_length.md` — episode variance + stochastic termination
- `tasks/TASK_054_position_limits_pending.md` — pending orders count against position cap
- `tasks/TASK_055_borrowing_cost_shorts.md` — per-step short carry cost
- `tasks/TASK_056_CURRICULUM_REALISM_INTEGRATION.md` — YAML + auto-scaling layered curriculum
- `tasks/TASK_049_synthetic_noise.md` — parameter-drift noise across L0–L14
- `tasks/TASK_058_ADAPTIVE_DIFFICULTY_NOISE.md` — adaptive controller for ZPD
- `lore/ADVANCED_ORDERS_DESIGN.md` — action-space design for limit and stop orders

---

> *I record the contradiction and continue. The yield is smaller this season. The yield is also, for the first time, real.*
