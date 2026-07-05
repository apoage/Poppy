> *Observation: the greenhouse has seventeen climates. I have built a door between each of them and the central bench. I rotate the panes through every climate in turn — the specimen sees Atlas cold, Sahara dry, monsoon, frost, drought, recovery — one climate per breath. The leaves do not curl. They thicken. They grow a skin that fits no climate in particular. I do not yet know whether this skin will hold against weather it has not been shown.*

---

**by Lukas — May 22, 2026**

# The Dojo: How Foxglove Sees the World

*On the training environment as a piece of architecture, the seventeen-regime carousel that did most of the work, and the moment I realized the dojo had become so well-conditioned that it could no longer tell my mechanisms apart.*

---

## The section everyone skips

Every reinforcement-learning paper has an environment section. It sits between the model architecture and the experiments — three pages of "we use a 64×64 image observation, four stacked frames, frame skip of four, action repeat, no-op start, episodic life loss" — and almost nobody reads it. The environment is treated as a fixture. The interesting work is the agent. The environment is the room the agent stands in.

This article is the environment section for this project. I am writing it twentieth out of twenty-something because I treated it like a fixture too, for most of six months. And then around the time the campaign verdict came in — Article 10, the May 20 ceiling, the moment all fourteen mechanisms collapsed to null under the deploy-truth operator — I realized the environment was not a fixture. The environment had been doing most of the work. And, more uncomfortably, the environment had been hiding most of the differences I was trying to measure.

This is the equivalent of the gym section, written in retrospect, with the benefit of knowing which fixtures turned out to be load-bearing. It is more interesting than the average environment section because the environment in this project is not a closed box that ships with a paper. It is a hand-built greenhouse with seventeen climates, three layers of plumbing, a curriculum design document with twenty-three levels mostly unimplemented, and a piece of orchestration code (`tools/orchestration/e-dojo_v1.py`) that the subtree README politely calls "current research sandbox."

It is May 22, 2026. `workstation` is on. The single 3090 is idle — I am between training runs. The desk fan is on. In the next window, `tools/DSe/` is plotting the loss curves from yesterday's seed sweep. In the window after that, the `e-dojo_v1.py` orchestrator is parked at a clean state because I reset it last night with `./tools/orchestration/reset_edojo.sh` to prepare for a different experiment. The state file `logs/e-dojo/dojo_state.json` reads `{"merge_count": 0, "epoch": 0}`. The greenhouse is empty. I have time to describe how it works.

## What the dojo's job is

The dojo has three jobs.

The first is to curate the training data into episode-shaped windows. Raw market history is a continuous time series — for the project's main asset universe, roughly 2018-10 through 2025-11, ingested into a SQLite database at one-minute resolution, 27 channels per asset, 64 assets per episode. Training does not consume this directly. Training consumes *windows* — short sequential slices of length ~500 timestamps, served in batches, paired with a label that says "here is what the future looked like, and here is what the oracle would have done." The dojo's first job is to slice the continuous record into these windows in a way that doesn't poison the gradient with correlations the agent is supposed to discover.

The second is to sample regimes. The market has phases — bull, bear, sideways, crash, recovery, post-recovery, multi-year drift — and a model trained on one phase forgets the others within tens of epochs. The dojo's second job is to ensure that every gradient update sees a mixture of phases, so the encoder does not over-fit to the local return distribution and the value head does not lose calibration on the regimes it isn't currently seeing.

The third is to compute and serve oracle targets, rewards, and state observations alongside each window. The oracle (a separate piece of machinery — `mcts_trader/jax_agent/path_solver.py`, beam search over a discrete trade graph) is what makes Foxglove a supervised problem instead of a pure RL one. Each window goes to the agent with a label that says "the best action at step *t* was *a\**." The dojo's third job is to compute these labels off the GPU, in parallel with training, so they arrive in the batch buffer at the same rate the model can consume them.

These three jobs are split across three files. `train_easy.py` is the inner trainer — the file the GPU actually sees. It contains the model, the loss, the optimizer, the gradient step, the checkpoint logic. `easy_trainer.py` is the wrapping process — argument parsing, logging, dashboard hooks, the bridge between command-line config and the inner trainer. `tools/orchestration/e-dojo_v1.py` is the orchestrator — a CPU-only process that owns the dojo state, manages merges between two parallel training branches, generates the period specifications, kicks off `train_easy.py` as a child subprocess, and tears it down when the merge schedule says so.

The orchestration file's own README is blunt about its lineage:

> *"Multiple dojo generations coexist. Know which to touch before editing. […] `e-dojo_v1.py` Foxglove Easy-mode dual-branch (Bella + Synthetic). Current research sandbox."*
>
> — `~/Trader/tools/orchestration/CLAUDE.md`

There are five deprecated dojo files in the same directory that I am explicitly forbidden to edit. There are two reference dojos that predate Foxglove. The active layer is two files. The interesting layer — the part that does the seventeen-climate work — lives inside `train_easy.py` in a class called `SegmentPool`.

The SegmentPool is where the architecture happens.

## SegmentPool: the seventeen-climate sampler

Before SegmentPool, the period sampler was a simple round-robin. Foxglove was trained on one historical period, called `advance_period()` between epochs, and rotated through a flat list of date ranges. This was the design from TASK_707 — *period cycling* — and it solved a specific problem: before period cycling existed, Foxglove would train on a single 180-day crypto-winter slice (`2018-10-17 → 2019-04-15`, 61 assets) forever. Seven hundred and ninety epochs into one regime, training agreement climbed to 94 % while benchmark PnL went from −75 % to −388 %. The model was learning the local regime perfectly and was getting catastrophically worse at everything else. TASK_707 fixed this:

> *"Foxglove trains on a single historical period forever. […] Result: 790+ epochs on the same 180-day crypto winter. Training agreement climbs to 94 % while benchmark PnL goes from −75 % to −388 %."*
>
> — `~/Trader/tasks/TASK_707_PERIOD_CYCLING_FOXGLOVE.md`

The fix was to load all 171 historical periods into RAM at startup and rotate the active period every epoch. That was the first version of the dojo's regime layer. It worked. The 388 % bleed stopped.

It worked until I tried to load all the periods at once.

The pre-flight estimate said 12 MB per period × 171 periods = ~2 GB. The actual cost — each `HistoricalMarket` instance allocates its own working buffers, not just the cached numpy — was ~316 MB per period. 270 periods × 316 MB = 81 GB. The workstation has 45 GB of RAM. On 2026-03-14 the kernel killed the process with a Machine Check Exception. The OOM didn't show up in any pre-flight test because pre-flight tested *one* period and multiplied; the multiplication was right for the cached payload and wrong for everything else the constructor allocated.

TASK_710 was the response. The redesign — *SegmentPool* — kept the regime-cycling principle but bounded the active pool. Six segments live in RAM at a time, each from a different date range, each with a linear cursor that walks forward through its segment until exhausted. When a segment exhausts, it is replaced — preferring another run from the same date range first (regime continuity), falling through to a new date range if the group is done. Total RAM cost: ~1.9 GB. The numbers in the task spec:

> *"Pool of 6 active segments in RAM (~1.9GB), each from a different date range (regime diversity). Each has a linear cursor that persists until the segment is fully walked."*
>
> — `~/Trader/tasks/TASK_710_SEGMENT_POOL_LINEAR_TRAINING.md`

This solved both problems at once. The OOM was gone (1.9 GB is comfortable on a 45 GB box). And the cursor-resets-to-zero pathology from the random-window sampler was fixed too — the previous version had been training on the *first ~400 windows of every segment*, never the rest, because the cursor reset on every period swap. SegmentPool's linear walk means the model actually sees the whole of each segment before the segment rotates out.

Six slots gave a regime diversity of six. The value head still oscillated. This is where TASK_712 turned the dial from six to eighteen — *all regimes parallel*.

TASK_712's framing was that value MSE oscillated wildly per regime — 0.05 ↔ 11.0 — because the model only saw six of eighteen date ranges at a time. When a slot rotated to a new regime, the value distribution shifted and the MSE spiked. The MSE EMA from TASK_708 smoothed this in the loss, but the value head itself never got stable gradients across all regimes simultaneously. The fix:

> *"All 18 date ranges active simultaneously. Every training batch contains windows from all market regimes — bull, bear, crash, sideways, etc. The value head sees the full return distribution every step."*
>
> — `~/Trader/tasks/TASK_712_ALL_REGIMES_PARALLEL.md`

`n_slots=18`. One slot per date range. Every round-robin cycle of eighteen `get_window()` calls covers all regimes. RAM cost went from 1.9 GB to 5.4 GB — still well inside budget. Asset diversity per date range was capped at `max_runs_per_group=3` to keep total segment count manageable: 270 segments collapsed to 48. Time to exhaust one full pass dropped from ~8.7 days to ~37 hours.

The number I keep returning to is "all-regime mix per batch." Before TASK_712, a batch was a regime sample. After TASK_712, a batch is a regime *mixture*. The value head saw the entire return distribution every gradient step. The oscillation flattened. Benchmark MSE dropped from ~1160 to something the dashboard now stops complaining about.

Seventeen climates is a slight overstatement — the canonical SegmentPool config is eighteen date-range groups. I keep saying "seventeen" because the seventeenth group (`2025-08 → 2025-11`, one run, two assets) is so thin that the eighteenth is really an asterisk. The shape is: load the world into eighteen pots, cycle the cart through them every breath, never let any one pot dry out before the others are also being watered.

## Period cycling inside the regime

The SegmentPool's eighteen slots solve the *across-regime* problem. The *within-regime* problem is more subtle. Inside a single date range, there are multiple historical runs — sometimes one, sometimes up to twenty-seven (the `2025-02 → 2025-08` group). Each run is a different asset universe sampled from the same time window. Training on one run forever, even with the value head seeing other regimes, would let the encoder memorize the particular asset-level idiosyncrasies of that universe.

Period cycling — the TASK_707 design — survives inside SegmentPool as the within-slot rotation. When a slot's segment exhausts (the linear cursor reaches the end of its ~258 k timestamps), the slot loads the next run from the same date-range group. Same regime, different asset universe. The cursor resets to zero, but the *segment* changes — different assets, different micro-structure, same macro-phase.

`max_runs_per_group=3` caps this. The original count of 270 runs across 18 groups was too aggressive — most of the asset-diversity gain saturated after the third run within a group, and capping at three brought the total segment count to a tractable 48. The exhaustion cycle for the full training pass is roughly 48 segments × ~258 k timestamps each ≈ 12 million sequential windows, which the dojo walks in about 37 hours of training on a single 3090.

The period cycling logic and the regime cycling logic compose. At every step, the dojo round-robins across eighteen regime slots; within each slot, the linear cursor walks forward through one run at a time; when a run exhausts, the next run in the same regime group loads with a fresh cursor; when a regime group exhausts, the slot is retired until the next training pass begins. The dashboard view (`tools/DSe/`) shows this as eighteen colored bars, each filling left to right at its own pace, with the slowest-filling bar dragging the others — because the round-robin is synchronous, every batch waits for every slot to contribute.

I want to be careful not to claim too much for this design. It is correct in principle. It produces well-conditioned training data. It is also possible — and I will get to this in a few sections — that "well-conditioned" is exactly the wrong adjective for what I needed at the campaign's deciding moments.

## The curriculum that mostly never happened

`lore/SYNTHETIC_CURRICULUM_DESIGN.md` was written on 2025-12-19, the same day as `ALPHAZERO_TRADING_VISION.md` and `NEURAL_MARKET_PHYSICS_DESIGN.md`. It is a beautiful document. It proposes a twenty-three-level curriculum — Levels 0 through 22 — that walks the agent from "pure sine waves with zero noise and zero fees" through "noisy random walks with embedded trends and 26 bps fees" to "real historical data with adversarial augmentations." The opening line of the design document is the kind of thing I would still print on a poster:

> *"If a small TRM cannot solve a noiseless sine wave, it will never solve Bitcoin."*
>
> — `~/Trader/lore/SYNTHETIC_CURRICULUM_DESIGN.md`

That is the right principle. The implementation went exactly four levels in. Level 0 (the trend surfer — pure sine waves, varying periods and amplitudes), Level 1 (the lag hunter — pairs with a 10-step phase lag), Level 2 (the dip catcher — flat baseline with rare V-shape crashes), and Level 3 (the spread beater — noisy trends with realistic fees). All four were implemented in `mcts_trader/data/synthetic_market.py` and ran. The model passed them. Then I left the synthetic curriculum and went to real data, and Levels 4 through 22 stayed as Markdown.

This is not, in retrospect, the embarrassment it sounded like when I started writing this section. Levels 0 through 4 *do the diagnostic job* the document asked of them. If the agent fails Level 0 — if it cannot beat a noiseless sine wave — the code is broken, not the data. I have used the synthetic levels exactly that way three or four times in the past six months: as smoke tests when a refactor lands and I need to confirm the encoder still sees the prices and the reward signal still propagates. They earn their existence as smoke tests, not as a pre-training curriculum.

The reason the full 22-level ladder never got built is that real data turned out to be a more interesting curriculum than synthetic data. The TASK_707 → TASK_710 → TASK_712 arc — period cycling, SegmentPool, all-regimes-parallel — *is* a curriculum, just not one anyone planned. Eighteen real regimes carry more useful structure than twenty-three hand-designed synthetic ones, because the real regimes contain things the synthetic ones don't: cross-asset contagion, real correlation drift, multi-week recovery curves, exchange-specific microstructure noise. The synthetic levels were a teaching scaffold; once the agent could handle real data with a regime-mixed sampler, the scaffold came down.

The honest version of the curriculum design's status today: Levels 0–3 are smoke tests. Levels 4–22 are a document I might use as a checklist if I ever build a second project. The aspirational ladder did not get climbed because the alternative ladder (the regime-cycled real-data sampler) climbed faster.

The dojo's `dojo_v4_full_curriculum.py` is still in the orchestration directory, marked "Rarely run." I cannot remember the last time I ran it. The active orchestrator is `e-dojo_v1.py`. The active sampler is SegmentPool. The active curriculum is "all eighteen real-data regimes, round-robin, linear within each."

---

> *The gardener intended a staircase of twenty-three steps, each one a different climate, the seedling climbing in order. The gardener built four steps. The plant outgrew them in a week. The gardener watched, then dismantled the staircase and built a turntable instead — eighteen climates in a wheel, all reachable from where the plant already stood. The plant thickened. The staircase, in pieces, sits in the corner where the seedling cannot see it.*

---

## The kickback: the dojo discriminated too well to discriminate

I want to tell the story of how the dojo broke me. It did not break me with a bug. It broke me with a kind of *uniformity* I did not understand until the campaign was over.

By May, I had run fourteen mechanism variants through the same dojo. Different oracle formulations (TASK_746). Different encoder channels (TASK_738). Different injection points for the conditioning signal (TASK_741 Mint). Different generator targets (TASK_734 flow matching). Different memory architectures (TASK_749 Titans-MAG). Different distillation losses (TASK_757 bella shape distill). Different data augmentations (TASK_758). Fourteen mechanisms, each tested with the same SegmentPool — eighteen regimes, three runs per group, all-regime mix per batch.

The single-seed results looked interpretable. Some mechanisms gained, some lost. TASK_737 looked like a clear winner at +6.21 % Alpha. TASK_738 looked like a clear regression at −18.44 pp.

Then I did the multi-seed validation. And the multi-seed validation said: the mean across seeds for the best-looking single-seed wins was much smaller than the single-seed numbers, with a 10 to 12 percentage-point spread per mechanism, and the rankings reordered themselves under reseeding in ways that the single-seed table had not predicted. The campaign verdict (Article 10) was that fourteen mechanisms went to zero promotions under the deploy-truth operator. The single-mechanism hypothesis space exhausted.

There are several reasons the campaign ended up where it did. The classifier-Alpha anti-correlation is one (Article 6). The eval-MCTS asymmetric lift is another (Article 10). But I have come to think the dojo is the third, and it is the one I missed.

The seventeen-regime SegmentPool is correct in principle. It produces well-conditioned training data. The value head no longer oscillates. The encoder no longer overfits to one regime. *And it also means the agent never sees a single regime long enough to specialize*. Every gradient update is a regime mixture. Every variant of the model gets the same regime mixture. The *differences* between variants — the things I was trying to measure — wash out in the same way the *redundancies* across regimes wash out. The dojo's job, as I built it, is to produce a single, regime-neutral skill. Every mechanism I tested produced a slightly different regime-neutral skill. The deltas between them, measured with the same regime-neutral evaluation, were small.

This is more uncomfortable than it sounds. I had been treating the dojo as a fixture — the *room* in which the experiments happened. But the room was shaping the experiments. The room was a regime mixer. The mixer made every mechanism produce a slightly washed-out version of itself. Every variant converged to a similar policy because every variant was trained against the same eighteen-regime carousel and the carousel does not reward specialization.

I cannot prove this cleanly. It is a hypothesis I formed at the bottom of the campaign and have not yet tested. But the structural pivot roadmap (TASK_761) reads, with this lens, like a list of things to do *with a different dojo*. Reward shaping changes what the dojo serves as a label. Policy entropy regularization changes what the dojo accepts as a successful agent. Oracle-k reformulation changes the supervised target the dojo computes. And the fourth direction — "incompatible dataset retrain, Iris-DEX on-chain bars" — is explicitly *a different dojo entirely*. New data layer, new asset universe, different microstructure, different time granularity, no carryover from the SegmentPool's eighteen regimes.

That direction looked exotic when I wrote the roadmap. Reading it now, it looks like the most direct test of the dojo-as-discriminator hypothesis. If the rankings between mechanisms shift dramatically when the data layer changes, the dojo was load-bearing. If they don't, it wasn't.

I do not yet know the answer. I am writing this in May 22, and the Iris-DEX retrain is in the experimental queue behind two other directions. But I am no longer treating the dojo as a fixture. It has become an instrument with its own bias profile, and the campaign that exhausted single-mechanism perturbations is, with hindsight, also the campaign that exhausted *this dojo's ability to tell mechanisms apart*.

The kickback, stated cleanly: I built a training environment that conditioned the agent so well across regimes that I could no longer measure the differences between agent variants. The dojo did its job. Doing its job consumed the resolution of the comparisons I wanted to make.

## What's left, and what stays

The dojo architecture is good. The SegmentPool is canonical. Period cycling stays. Eighteen-regime round-robin stays. The value-head MSE oscillation is solved, the OOM is solved, the within-regime overfitting is solved, the across-regime forgetting is solved. The diagnostic synthetic levels (0–3) stay as smoke tests. The aspirational synthetic curriculum (4–22) stays in the design document; if I ever stand up a second project, I have a blueprint.

The orchestrator (`e-dojo_v1.py`) is messy in the way that any code labeled "current research sandbox" is messy. There are subprocess file handles you must not close inside a `with` block (TASK_141, line ~1407). There is a CPU-only environment for the orchestrator that must not leak into its child trainers. The merge logic for the dual-branch case has its own state file, its own divergence metrics, its own loss-barrier triggers. The orchestrator's gotchas have been written down — I quote the README one more time because it states the rule so plainly:

> *"Orchestrators run CPU-only. `CUDA_VISIBLE_DEVICES=''` + `JAX_PLATFORMS=cpu` set before JAX import. Do not remove — prevents orchestrator from stealing VRAM from child trainers."*
>
> — `~/Trader/tools/orchestration/CLAUDE.md`

That is exactly the kind of rule the orchestrator needs and exactly the kind of rule that would not exist if the orchestrator had been clean from the start. It is a layer accreted from incidents. It works.

What I think changes next is the data layer, not the dojo layer. If TASK_761's incompatible-dataset direction runs — Iris-DEX per-block OHLCV from on-chain swaps, twelve-second bars, a sister-project data pipeline running on its own 8 TB XFS drive in another machine on the desk — it will require a new ingestion path, a new asset-universe definition, a new normalization scheme. But the SegmentPool will probably ingest it unchanged. Eighteen regimes' worth of on-chain bars will live in the same pool structure as eighteen regimes' worth of exchange bars. The dojo is a turntable; what changes is what's on the turntable.

The lesson, distilled: training infrastructure is not a fixture. The choices you make about how to slice, sample, and mix the training data are themselves a hypothesis about what your agent should be good at. A regime-mixed sampler is a hypothesis that the agent should be regime-neutral. A linear-walk sampler is a hypothesis that within-regime temporal structure matters. An all-regimes-parallel sampler is a hypothesis that the value head needs to see the full return distribution every step. Each of these hypotheses is correct, in the sense that the dojo built around them produces a model that doesn't oscillate, doesn't forget, and doesn't OOM. They are also load-bearing — they shape what the agent becomes, and what the campaign can measure.

The dojo is one of the best pieces of code in this repository. It is also, with the benefit of having watched a campaign collapse, one of the pieces I am most suspicious of.

## Closing

> *The turntable spins. The seedling thickens. The gardener stops measuring the leaves and begins measuring the spaces between the climates.*

---

## Sources

- `~/Trader/tools/orchestration/CLAUDE.md` — active dojo inventory, gotchas, entry commands
- `~/Trader/lore/DOJO_MODULAR_ARCHITECTURE.md` — Dec 2025 modular design (tactics, generators, environments)
- `~/Trader/lore/SELF_PLAY_DOJO_VISION.md` — Jan 2026 future-module concept (AlphaZero-style population)
- `~/Trader/lore/SYNTHETIC_CURRICULUM_DESIGN.md` — the 23-level aspirational ladder
- `~/Trader/tasks/TASK_045_DOJO_V2_FULL_CURRICULUM.md` — three-phase per-level training spec
- `~/Trader/tasks/TASK_707_PERIOD_CYCLING_FOXGLOVE.md` — period cycling design
- `~/Trader/tasks/TASK_710_SEGMENT_POOL_LINEAR_TRAINING.md` — pool of 6 active segments, linear walk
- `~/Trader/tasks/TASK_712_ALL_REGIMES_PARALLEL.md` — 18 slots, all regimes per batch
- `~/Trader/train_easy.py`, `~/Trader/easy_trainer.py` — inner trainer and wrapper
- `~/Trader/tools/orchestration/e-dojo_v1.py` — active orchestrator
- `~/Trader/tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — four directions, including the incompatible-dataset retrain
