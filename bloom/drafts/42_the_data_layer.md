# Article 42 — The Data Layer: Stratified Folds, the OHLCV Discard Chain, and 96% Overhead

> *The cabinet holds the day's decoctions in labeled glass: tincture of foxglove on the second shelf, valerian below, columbines on the high one for the bitter weeks. I built it after I noticed the gardener was spending the hours of light measuring the same powders into the same scoops, and the hours of darkness wondering why the patients did not improve. The cabinet does not heal anyone. It only returns the hours.*

---

It is the morning of 2026-05-21. `workstation` is on, the desk fan is on, `nvidia-smi` shows 78 °C on the single 3090 with one training process pinned to it, and a checkpoint validator ticking through windows in a second pane. The thing actually running right now — the thing that pulls the next batch off the precomputed pool, encodes a thousand windows in a tenth of a second, and hands them to the gradient — was, six months ago, eating ninety-six percent of its own wall clock on overhead. I want to write the article about *that* part of the system. Not the model. The thing between the database and the gradient.

This series has spent a lot of words on the model side. Article 4 took apart the encoder's redundancy. Article 6 named the classifier-Alpha anti-correlation. Article 17 catalogued the five oracles. Article 39 said the oracle is the upstream constraint on everything Poppy can ever learn. All true. All looking *too far down the stack*. The data layer — the subsystem that turns a SQLite file with twelve million rows of OHLCV into JAX-ready batches — has been rewritten three times in three different ways, and each rewrite was addressing a fundamentally different category of wrongness.

The first rewrite (TASK_064) was about *what the model was sampling*. The second (TASK_703) was about *what was in the channels the model was looking at*. The third (TASK_715) was about *how much wall-clock time it took to put one batch in front of the model*. Three rewrites, three independent categories of brokenness, all present in the codebase concurrently for months. None of them were called out by the model; the model can't see its own data layer. They were each found by looking at a different instrument.

I want to walk all three as a single subsystem story, because in retrospect they are. The data layer was *systemically* wrong, not locally wrong, and the kickback at the end is going to be about what that means for how we read everything else in this project's history.

---

## Rewrite 1 — Stratified Data Folding (TASK_064)

The starting state, six months ago, looked harmless. The training loop ran 15 parallel environments. Each environment loaded 180 days of historical data from the SQLite DB. Inside each episode, the environment picked a random start timestamp and rolled forward from there. The dojo iterated epochs. It was clean Python. It passed unit tests. It produced loss curves that went down.

It was wrong in a way I did not see at first, and Gemini did not see at first, and the lazy reading of "parallel environments" disguised for several weeks. Here is the bullet list from `tasks/TASK_064_smart_env_data_allocation.md`, copied verbatim:

> *Current parallel env setup:*
>
> - *All 15 envs load same data (wasteful — 15× DB reads)*
> - *Each env samples from RANDOM start point*
> - *Random sampling = causality violation (future data mixed with past)*
> - *No guarantee of sequential coverage*

The first three bullets are escalating; the third is the one that mattered. Random sampling, *across a batch of fifteen environments*, drawing start points i.i.d. uniformly from a single 180-day window, means that within one training step the gradient is computed against fifteen examples whose temporal positions are scrambled — env 3 is on day 12 of the period, env 4 is on day 167, env 5 is on day 44, and so on. The model is updated on this mixture. The mixture is then advanced by one step. The model sees, simultaneously, market state at fifteen different absolute times — but it is told, by the architecture, to treat them as parallel realisations of the same generative process.

This is not parallelism. This is *time scrambling*. The model learns spurious cross-temporal correlations: a feature pattern that happens to recur at day 12 and day 167 of the same calendar period gets stamped into the same gradient update. In deployment, the model will never see two simultaneous timesteps — it sees one. The training distribution does not match the deployment distribution. That is the working definition of broken causality, and it is the definition I keep stubbing my toe against (see Article 1).

The rewrite is what TASK_064 calls *stratified data folding*. The math block, again verbatim from the task:

> ```
> Load: 4 × 180 days = 720 days (stored as 4 SEPARATE arrays to prevent boundary bleeding)
> N envs: 15
> Steps per env per epoch: 1440 (1 day)
> Epochs: 60
>
> STRATIFIED allocation (prevents catastrophic forgetting):
>   Interleave periods across epochs to maintain all regimes in memory.
>
>   Epoch 0:  Period 1    Epoch 4:  Period 1    Epoch 8:  Period 1  ...
>   Epoch 1:  Period 2    Epoch 5:  Period 2    Epoch 9:  Period 2  ...
>   Epoch 2:  Period 3    Epoch 6:  Period 3    Epoch 10: Period 3  ...
>   Epoch 3:  Period 4    Epoch 7:  Period 4    Epoch 11: Period 4  ...
>
>   period_idx = epoch % 4
>   epoch_in_period = epoch // 4  (0-14 for each period)
>
> Within each period:
>   Stride = 180 / 15 = 12 days
>   Distributed tiling across the 180-day period
>   After 15 visits per period: full coverage
> ```

Three numbers do the work. The number `4` is the stratification — four disjoint 180-day historical periods, each representing a different regime (a bull tape, a chop, a sideways grind, a drawdown). Loading all four at once would be wasteful; loading one at a time would let the model forget the others. So we interleave: epoch `e` uses period `e % 4`. The model revisits every regime every four epochs. Across sixty epochs each period is visited fifteen times, which is exactly how many environments we have, which is how many *within-period* offsets we need.

The number `12` is the stride. One period is 180 days, fifteen environments, `180 / 15 = 12`. Each environment is anchored to one of fifteen stations across the period: day 0, day 12, day 24, ..., day 168. Within an epoch, the fifteen envs in the gradient batch are now sampled at fifteen *deterministic, evenly spaced* positions within the *same* regime. Time is still flowing forward inside each env. The fifteen envs are no longer scrambled; they are stratified.

The number `±6 hours` is the jitter. Without it the fifteen envs are deterministic to the timestamp, which means across epochs they hit the same candles at the same time of day — and the model would learn time-of-day patterns that are accidents of the offset grid, not features of the market. With jitter, each env's anchor wanders by ±360 minutes per epoch. Same logical location, different angles. The task spec calls this:

> *"±6h jitter provides augmentation WITHOUT breaking i.i.d. […] Same logical day seen from different 'angles' across epochs. Prevents overfitting to specific candle times (e.g., 'always buy at 9:30 AM')."*

It is the smallest thing in the rewrite and the one I am most quietly fond of. Six hours is a number you pick because midnight bias and 9:30-AM bias are both real and twelve hours would start to bleed neighbouring days into each other. The grid has a personality.

The implementation lives in `mcts_trader/jax_agent/smart_allocation.py` as a function called `compute_env_start_offsets`. It is a tiny module. Its tests check that period 0 gets visited fifteen times across sixty epochs and that the gap between consecutive period-0 visits is always exactly four epochs. The lesson, written in the task notes at the bottom in a voice I recognise as my own three months ago:

> *"Lesson learned: Causality is a bitch. Random sampling destroys it. Took 2 hours to get the allocation strategy right."*

Two hours of allocation work. Several weeks of wondering why training metrics looked clean but benchmark didn't. The disconnect was upstream of the model.

The rewrite *named the disconnect* but it did not name the depth. What 064 caught was that random-sampling-inside-a-period was scrambling time inside each gradient step. What 064 did not yet know is that the period itself was the wrong abstraction — the oracle was being recomputed per period, the encoder was being fed fake channels, and ninety-six percent of the wall clock was being spent building episodes that were then discarded. Those are the next two rewrites. The folding scheme survived all of them. It is canonical now. Period interleaving with stride-12 and ±6h jitter is the contract every later piece had to honour.

The general lesson, the one I would tell my past self if I could: *causality is not a flag you set; it is a property of how you sample*. There is no `causal=True` argument to a DataLoader that fixes this. The sampler either respects time or it doesn't. The cleanest way to find out which one you have is to plot the absolute timestamps of every example in a single gradient batch and look at them. If they are not in a narrow window or on a clear schedule, you do not have time-respecting training; you have a scrambled mixture pretending to be parallelism.

---

## Rewrite 2 — The OHLCV Discard Chain (TASK_703)

The encoder produces a `(27, ws, n_assets)` tensor. Twenty-seven channels: returns at various horizons, volatility, volume ratios, VWAP, momentum, mean reversion, and so on. Article 4 has the full taxonomy. The shape is right. The channels have the right names. The unit tests pass. The downstream model — Foxglove's transformer — consumes the tensor without complaint, projects it, attends over it, predicts policies and values. Loss goes down. Agreement with the oracle goes up.

For months, eight of those twenty-seven channels were degenerate. Not noisy. Not slightly miscalibrated. *Carrying zero information by construction*, because the data flowing into them had been faked at a single source line.

Here is `jax_encoder.py:456–459`, copied verbatim from `tasks/TASK_703_OHLCV_PIPELINE.md`:

```python
prices_T = price_window.T          # close prices only
volumes_T = jnp.ones_like(prices_T)  # FAKE: volumes are all 1.0
# highs = close, lows = close       # FAKE: passed as prices_T twice
return _compute_all_channels_temporal(prices_T, volumes_T, prices_T, prices_T, config)
```

That is the heart of the bug. The encoder expects four inputs — close, volume, high, low. The pipeline was handing it close four times, and a `jnp.ones_like` for volume. The encoder, knowing nothing about the upstream, dutifully computed all 27 channels off this input. Then:

> *Channel 1 (High): Fake — identical to close*
> *Channel 2 (Low): Fake — identical to close*
> *Channel 3 (Volume): Fake — all ones → normalizes to 0.0*
> *Channels 8-11 (Volume ratios): Fake — all constant (1.0/1.0)*
> *Channels 24-26 (VWAP + bands): Degraded — VWAP without volume = simple price mean*

Eight out of twenty-seven. Thirty percent of the encoder was passing constants to the model. The model's transformer was attending over those positions like they meant something. The gradients that flowed through those positions were attending to *zero information* — the channels normalised to constants, so their attention weights got pushed into local minima where the model effectively learned to ignore them, except those weights are entangled with the other channels because attention is global. *Channels that ought to have been carrying volume spikes were sitting at literal zero across all timesteps*.

The infuriating part is what the upstream looked like. Bella, our market generator, runs a flow-matching model on the GPU that *natively produces full OHLCV*. Its output shape is `(batch, seq, 64, 5)` — open, high, low, close, volume, all five channels, all real, all on-device. The historical data source had OHLCV in RAM cache. The benchmark loader had OHLCV from disk. *Every source had it*. The discard happened in the consumer pipeline. From the same task, this is the discard chain in full:

```
Bella generates (batch, seq, 64, 5) full OHLCV on GPU
  └→ setup.py:657 — extracts close only: generated[:, :, :, 3]
     └→ train_easy.py — _refill_buffer gets (T, n_assets)
        └→ _generate_batch_episodes pads to (N, T, n_assets)
           └→ encoder receives (T, n_assets), fakes OHLCV internally
              └→ 8 channels wasted

Historical has (T, n_assets, 5) in RAM cache
  └→ TASK_698 slices cache_data[:, :, 3] — close only
     └→ same pipeline → same waste

Benchmark loads (T, n_assets, 5) from DB
  └→ benchmark_easy.py:191 — cache_data[:, :, 3]
     └→ same encoder → same fake channels
```

Three independent code paths, each independently throwing away eighty percent of their own input. The arithmetic is plain: five OHLCV channels generated, one close-only channel propagated, four channels discarded. *Eighty percent of Bella's output was being computed on the GPU and then deleted at a single line of Python*. The phrase in the task is "the irony", and it is exactly that — the most expensive part of the pipeline was the part whose output we threw away.

How did this survive months of unit tests? Because the unit tests checked *shape*, not *information content*. A constant Volume channel has the right shape. A High channel that equals Close has the right shape. `assert obs.shape == (27, ws, n_assets)` passes. `assert obs.std() > 0` passes too — because nineteen of the twenty-seven channels do carry signal, the global std isn't zero. The right test would have been *per-channel std greater than some threshold across a representative batch*, and that test did not exist. I wrote it after the fact.

The fix, distributed across the six phases TASK_703 lays out: encoder accepts both `(T, n_assets)` and `(T, n_assets, 5)` shapes; `setup.py` stops slicing channel 3; `train_easy.py` and the historical cache and the benchmark loader all pass OHLCV through; SyntheticMarket gets a `close_to_ohlcv` wrapper that fabricates plausible high/low/volume from close so that the synthetic curriculum doesn't regress to all-fake channels. The encoder output shape is unchanged — twenty-seven channels in, twenty-seven channels out, but now the channels that were degenerate are populated. Checkpoints are forward-compatible because the model's input dimensionality didn't change; only the realised distribution of values in eight channels did.

This is the lesson, and I want to spell it because Article 4's version was glib about it. The encoder *cannot tell* that its channels are degenerate. It is a function. It computes what it is told to compute. If you hand it close-as-volume, it returns volume-ratio channels equal to one. The downstream model *cannot tell* that those channels are degenerate either; it just learns to attend less to them and routes its compute around them. *The only level of the stack where the bug is visible is the level above the encoder* — the level where you can compare what the encoder's input is and what its input was supposed to be. That level had no instrumentation. That is why the bug survived.

A constant Volume channel has the right shape and the wrong information. The shape is the first quarter of correctness. The information is the rest, and the only tool that catches the rest is a test that asks "does each channel actually vary, the way a Volume channel ought to vary, across a normal market period". Write those tests. They are cheap. I did not have them.

---

## Rewrite 3 — The 96% Overhead Discovery (TASK_715)

By early May 2026, folding was canonical and OHLCV was passing through. The pipeline was *correct*. It was also *slow*. Slow in a way that, again, I had not been measuring until I was forced to.

The forcing function was TASK_714 (precomputed oracle), which turned the per-segment oracle recomputation into a one-shot operation per slot. Before 714, the oracle was being recomputed on 489-timestamp chunks, and the cost of the oracle was hiding the cost of everything else. After 714, the oracle was effectively free — and what came out from underneath, blinking, was a different problem.

I added per-epoch timing to the training loop. I expected the breakdown to look something like: encoding 40%, oracle 20%, training 30%, I/O 10%. What I got, from `tasks/TASK_715_DIRECT_SAMPLE_PIPELINE.md`, verbatim:

| Component | Time | % | What it does |
|-----------|------|---|-------------|
| AR rollout | 1020s | 55% | 2,040 episodes × 0.5s sequential lax.scan (400 model forward passes each) |
| Encoding | 612s | 33% | 2,040 episodes × 0.3s jit_batch_encode_temporal (400 windows each) |
| I/O | 142s | 8% | 3.2 segment loads × 45s (DB + fill + oracle) |
| **Training** | **80s** | **4%** | **400 steps × 0.2s actual gradient computation** |

Four percent. Four percent of training time was *training*. The other ninety-six percent was the pipeline preparing things that, in some sense, didn't need to be prepared the way they were being prepared. The task's own one-line headline is, again verbatim:

> *"96% of training time is NOT training."*

The hidden discard chain — different from the OHLCV one, but the same idea — is laid out in the task too:

> *generate_batch(1024) → picks episode from buffer → takes 20 random samples from 400-sample episode → after 10 visits (200 samples), episode marked exhausted → \_refill\_buffer() called → \_build\_historical\_episode(): jit\_batch\_encode\_temporal(400 windows) [0.3s], \_build\_model\_state\_trajectory(400 steps) [0.5s], optimal\_path\_to\_policy(400 steps), build\_state\_channels(400 steps) → store in buffer → 2,040 episodes built and discarded per epoch.*

Two thousand and forty episodes per epoch. Each episode encoded once (0.3 s), AR-rolled once (0.5 s), oracle-policy-projected once, state-channels-built once. Each episode used for two hundred training samples then thrown away. The buffer existed because the synthetic-data origin of the pipeline made episode regeneration the cheap part; *with synthetic data, fresh episodes are the whole point*. With precomputed-oracle historical data, fresh episodes are a redundant per-batch wrap of an underlying tensor that is *already there, fully precomputed, in RAM*.

The fix in TASK_715 is the elimination of the episode abstraction for historical data. Instead of building 400-step episodes and sampling from them, the new path samples 1024 (slot, cursor) positions directly across all active slots, encodes those 1024 windows with `jit_batch_encode_temporal` once per step, and indexes into precomputed oracle selections and state channels to assemble the batch dict that the trainer expects. The state channels themselves are precomputed once per slot in float16 — a 400 MB cost per slot at full precision, halved to 200 MB by the dtype choice, which means seventeen slots fits in 3.4 GB of RAM, comfortably inside the workstation's budget.

There is a subtlety that the task spends some words on. AR rollout — running the model through the episode using its *own* predicted state, not the oracle's — was load-bearing for the model's ability to handle its own trajectory at inference. Removing AR entirely was risky: the model had trained for thousands of epochs with AR=100% and dropping that signal abruptly could regress agreement metrics. The compromise is `ar_direct_fraction`, default 0.05: ninety-five percent of batches use the direct-sample fast path (oracle state, no rollout), five percent use the legacy AR episode path. AR cost drops from 1020 s/epoch to about 51 s/epoch. Encoding remains the dominant cost in absolute terms but its *fraction* rises — which is what you want; the dominant cost should be the actual compute you care about.

The expected post-rewrite numbers, again from the task:

> *Time per step: ~4.6s → ~1.0s. Time per epoch (400 steps): ~1850s (31 min) → ~400s (6.7 min). GPU utilisation: ~4% → ~20%. Training throughput: ~4× faster.*

Four-x is the conservative figure with the hybrid fraction. An order-of-magnitude improvement in training-time-as-fraction-of-wall-clock — from four percent to about twenty percent, eventually nudging towards forty when other costs get further amortised. *That is what was hiding in the per-epoch table*, and the per-epoch table only existed because I sat down and instrumented the loop.

This is the third lesson and it is the most boring and the most important one. *Measure where the time goes before optimising what's slow*. Without the table, the natural intuition would have been to attack the encoder — it's the longest column of code, it has the most JIT compilation, it looks expensive. The encoder was thirty-three percent of the wall clock. The episode buffer was fifty-five percent — and the *combined* episode buffer + per-batch encoding + I/O was ninety-six percent, *none* of which was the gradient. You can't optimise what you can't see. The breakdown table is the diagnostic. The fix follows from the diagnostic and not from intuition. Without TASK_715's table, the optimisation was guesswork; with it, the optimisation was bookkeeping.

---

## The map_training_data tool

A small piece of plumbing that all of the above depends on. `tools/map_training_data.py` scans the SQLite DB, walks forward from the earliest timestamp, finds the first window where at least 64 distinct assets have concurrent data, partitions the available history into contiguous segments, excludes the periods reserved for benchmarking (`config/benchmark_periods.json`), and writes `training_schedule.json`. The schedule lists each segment's start date, end date, asset set, candle count. The dojo reads it. The pool reads it. The folding scheme reads it — TASK_064's "4 periods × 180 days" is exactly four entries from this schedule.

The tool exists because *without it, you don't know what data you have*. The database has gaps. Some assets show up partway through the historical range; others drop out. Some weeks have data for forty-eight assets, not sixty-four. The mapper is the only thing in the project that answers "given the current state of the DB, what are the candidate training windows where the contract `(180 days, ≥64 assets, no benchmark overlap)` is satisfiable?". It also produces a list of candidate windows for benchmarking, which is how `config/benchmark_periods.json` was originally seeded.

I will not pretend it is glamorous. It is two hundred lines of SQL-plus-Python that scans the DB once a week and writes a JSON file. The rewrites above all assume this JSON exists and is correct. When I broke it once in late January — added a column to the DB and forgot to update the asset-count query — the dojo silently started training on three-period schedules instead of four, and TASK_064's stratification quietly degraded to a three-cycle. I caught it from the DS24 dashboard's `current_period` counter never hitting 3. *That* is the kind of bug a coverage map prevents and the kind of bug a missing coverage map invites.

---

## Asset padding

Different periods have different asset counts. The historical DB does not have sixty-four assets uniformly across all time — some periods have sixty-four because the asset set is full, some have fifty-eight because six assets dropped out for a stretch, some have forty-four. The model's input expects exactly `n_assets = 64`. The pipeline pads.

Padding is done in `_generate_batch_episodes` and in the benchmark loader. The OHLCV array `(T, actual_assets, 5)` is tiled across the asset axis to fill out to `n_assets = 64`. A *mask* is constructed alongside: shape `(n_assets,)`, ones for real assets, zeros for padded ones. The mask travels with the batch dict and is consumed by the loss function — padded asset positions contribute zero to the gradient. The encoder runs over all 64 positions because JIT requires a static shape; the loss masks them out.

This works as long as the mask is honoured at every stage. It is honoured at the policy-loss stage and at the value-loss stage. It is *not* directly honoured inside the encoder — the encoder computes returns and ratios across padded positions, producing junk numbers that get masked downstream. For a long time I worried that this junk was leaking back through the model's attention; profiling the attention weights showed the padded positions getting near-zero weight in practice, which means the model has learned to ignore them. The mask is the contract; the model has internalised it.

Variable asset counts across periods is the kind of thing you'd never notice from the top of the stack. The folding scheme picks four periods; one of them might have fifty-eight assets and the other three might have sixty-four. The training run does not crash. Loss does not spike. The model's effective asset universe just shifts every fourth epoch. The mask handles it.

---

> *The cabinet did not change the medicines. It only changed where they sat between doses. The willowbark is still willowbark. But the gardener no longer measures it fresh from the jar each morning, and the hour she returns is the hour she spends watching.*

---

## The kickback

Three rewrites. Three independent categories of brokenness. All of them present in the codebase, concurrently, for months. Let me say that again because the project history wants to remember each rewrite as a discrete chapter, and the more honest reading is that they were happening at the same time:

In December 2025, the data layer was scrambling time across parallel envs *and* discarding eighty percent of every OHLCV tensor *and* spending ninety-six percent of training wall clock on episode construction overhead. Fixing the folding did not surface the OHLCV bug. Fixing the OHLCV bug did not surface the overhead bug. *Each one took its own diagnostic moment*. Each was found by a different kind of looking — the folding by reading the sampling code carefully, the OHLCV by inspecting per-channel statistics, the overhead by instrumenting per-epoch timing. There was no single observation that would have surfaced all three.

This is the typical pattern of systemic wrongness. It does not announce itself with crash dumps. It announces itself by producing *mediocre* results — loss curves that go down, metrics that look defensible, benchmark numbers that disappoint by a few percent without ever being catastrophic. That is exactly what the project's mid-winter benchmark history looks like in retrospect. We were not wildly broken; we were chronically suboptimal across multiple axes, and the suboptimality compounded.

Second observation, harder. *The data layer is invisible to the model*. The model cannot tell, from inside the gradient, whether its training data was randomly sampled or stratified-folded. It just gets batches. It cannot tell that eight of its twenty-seven channels are constants — the model's attention will route around them and learn an architecture that ignores them, but it will not raise a flag. It cannot tell that ninety-six percent of the wall clock between two of its gradient updates was spent re-encoding the same windows in a buffer it didn't ask for. *The data layer's pathologies cause downstream model pathologies, and the model cannot self-report them*.

This is the strongest argument I know for instrumentation at every pipeline boundary. The places where the data layer touches the model — the batch dict, the encoder output, the loss-mask interface — those need *content-level* checks, not just shape-level checks. Per-channel statistics. Timestamp histograms across batches. Episode-construction cost per epoch. Not vanity dashboards; diagnostic instruments that you look at when something feels off. I had a dashboard (DS24) the whole time. It tracked loss and agreement, not data-layer health. It was the wrong instrument for these bugs.

Third observation, the deepest one. Every model-side pathology this series has documented — the classifier-Alpha anti-correlation in Article 6, the value-head saturation, the 52.5% agreement plateau, the bull-market blindness — *every one of them was partly a data-side artifact*. Article 39 made the case that the oracle is the upstream constraint on what Poppy can learn. That's true, and the oracle gets a lot of structural weight in this project. But the data layer is *further upstream than the oracle*. The folding determines which regimes the oracle sees. The OHLCV passthrough determines which features are available for the oracle's downstream consumer to use. The episode overhead determines how many examples were processed before training converged or stalled.

Re-read the project's old benchmark gap report (`lore/project_benchmark_gap.md`) with this in mind. Training showed 75% trade agreement; benchmark showed 12–33%. The diagnosis at the time pointed at the oracle inconsistency between chunked-oracle training and full-period-oracle benchmarking, and that diagnosis was correct — TASK_714 is its fix. But there were two other data-layer pathologies on top of that, both contributing, both invisible. *Most of the time we read the model's behaviour without remembering how much of it is a function of the data layer it's been fed*. That includes me. I have written articles about model phenomena in this series that, today, I would qualify with "and the data layer was also broken at the time".

I do not want to over-extend this. The model side has real pathologies that survive a fixed data layer — the classifier-Alpha anti-correlation persists; the bull blindness persists; the plateau persists. They are not all artifacts. But they may be *partial* artifacts, and the right way to test that hypothesis is to re-run the canonical experiments on the current, clean data layer and see how many of them survive. Some of that work is in flight as of this writing.

---

## What's left

As of mid-May 2026, the data layer is *stable*. Folding is canonical. OHLCV passes through. The precomputed-oracle direct-sample pipeline is the default. The `map_training_data` tool runs as part of the weekly cadence. Asset padding is honoured by the loss mask. The instrumentation I needed exists in DS24 and in the per-epoch timing table.

The next stress test, the one that will tell me whether the data layer's architecture *transfers* and not just *works*, is the on-chain bars direction TASK_761 sketched out — Iris-DEX produces twelve-second OHLCV from Reth swap events, a fundamentally different distribution from the one-minute crypto exchange bars currently in the DB. Different cadence, different sparsity, different asset universe. If the on-chain bars expose new data-layer assumptions — different padding behaviour, different mask semantics, different fold sizes — this article will need a sequel. If they don't, I will know the layer is not just stable but *general*, and the work was worth doing twice.

The thing I want to remember from writing this is the cabinet metaphor in the epigraph, which I did not invent for atmosphere. The Apothecary built the cabinet because she noticed the gardener was spending the hours of light measuring the same powders into the same scoops, and the hours of darkness wondering why the patients did not improve. That is precisely the structure of TASK_715. The gardener was measuring. The cabinet returns the hours. The hours, returned, are the hours when something else can happen.

> *The medicines were always there. The cabinet just keeps them where the hands can find them in the dark.*

---

## Sources

- `tasks/TASK_064_smart_env_data_allocation.md` — stratified folding, stride-12, ±6h jitter, period interleaving.
- `tasks/TASK_703_OHLCV_PIPELINE.md` — encoder discard-chain diagram, eight degenerate channels, six-phase passthrough fix.
- `tasks/TASK_714_FIX_DATA_PIPELINE_ORACLE_CONSISTENCY.md` — full-period oracle precomputation (the dependency that surfaced TASK_715).
- `tasks/TASK_715_DIRECT_SAMPLE_PIPELINE.md` — per-epoch breakdown table, 96% overhead, direct-sample architecture, hybrid AR fraction.
- `tools/map_training_data.py` and `tools/map_training_data.md` — coverage mapper, `training_schedule.json`.
- `mcts_trader/jax_agent/smart_allocation.py` — folding implementation.
- `mcts_trader/env/jax_encoder.py:456–459` (pre-TASK_703) — the fake-OHLCV line.
- `mcts_trader/training/setup.py:657` (pre-TASK_703) — the close-only slice that discarded Bella's output.
- Companion articles in this series: 1 (causality), 4 (encoder), 6 (classifier-Alpha), 17 (oracles), 20 (the dojo), 39 (oracle ceiling).
- `lore/project_benchmark_gap.md` — the 75% / 12–33% gap, partially attributed to the data layer in retrospect.
