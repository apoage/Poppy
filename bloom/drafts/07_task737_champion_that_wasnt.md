# TASK_737: The Champion That Wasn't

*On the morning a 4-cell ablation table showed a +6.21% Alpha winner, the post-hoc mechanism I drafted to explain it, and the writeup I should not have written before the second seed had landed.*

> *In the early greenhouse, a single specimen flowers before any other. The Apothecary notes the date in the journal, sketches the petal count, weighs the leaf, records the soil pH. She writes down the dose. She writes down the hour. She does not write down that this is one plant, in one pot, on one shelf, on one morning. The journal does not have a column for that.*

---

## Today is the day, except today is not the day

It is 2026-05-13, just after seven in the morning. The desk lamp is the kind of yellow that says it's been on since before the windows got light. There is a coffee on the left of the keyboard that I cannot honestly say is the first one. `workstation` is on. The desk fan is on. `nvidia-smi` is ticking through nothing in particular in the next window because the workstation 3090 is idle this morning — the overnight runs that produced the table on my screen finished hours ago, and the only thing I am doing with the GPU right now is breathing on it.

The table is open in a terminal:

```
no_aux_solo     rank=ON  aux=OFF   Bal.Acc 0.266   Alpha  +6.21%   6/6
no_rank         rank=OFF aux=ON    Bal.Acc 0.240   Alpha  +3.25%   6/6
a1b1_repro      rank=ON  aux=ON    Bal.Acc 0.249   Alpha  -1.99%   6/6
minimal         rank=OFF aux=OFF   Bal.Acc 0.257   Alpha  -4.99%   6/6
TASK_736 ref    rank=ON  aux=ON    Bal.Acc 0.265   Alpha  +3.87%   6/6
```

I have been staring at it for long enough that the rows have stopped being rows. `no_aux_solo` has been promoted in my head to an adjective. The previous Stage-B baseline — a1b1, both heads on — has reproduced at *minus* 1.99% Alpha this morning, against its original +3.87%, and the variant that turns the auxiliary head off has come in at *plus* 6.21%. That's an 8.2pp gap between two configurations that differ by a single boolean flag.

I open `lore/notes_2026-05-13.md` and start writing §95. I title it *"Full 4-cell TASK_737 result (07:00)"*. I paste the table. Then I type the sentence that, six days later as I am drafting this article, I am still trying to forgive myself for typing without a hedge:

> **The two heads interact NEGATIVELY** (-13 pp combined effect).

I bold the word NEGATIVELY. I capitalize it. I am, in that moment, certain. The right tense for what I felt is not past tense. The right tense is the one I have to translate from: a present-tense feeling that the experiment has *told me something*. We have something.

This article is the morning. It is the half-day that lived inside that certainty. The reseed had not yet been run.

---

## What TASK_737 was meant to measure

To make the rest of this make sense I have to back up to the previous month, because the table on the screen at 07:00 is not a one-shot experiment — it is the cashing-in of a design decision made four weeks earlier.

Through April, the Foxglove encoder had been accumulating auxiliary heads the way a workbench accumulates clamps: one for each problem that came up, never quite tidied away. The two that mattered for this morning were:

- **rank head** (TASK_730). A ranking auxiliary loss that supervises the encoder against the argsort of future cumulative returns across the asset universe. The intuition: a trader does not need to know *what BTC will do*, they need to know *whether BTC will do better than the alternatives over the next horizon*. So you supervise the encoder to be good at the cross-sectional comparison directly, on the assumption that ranking is downstream of representation quality.
- **aux head** (TASK_735). A per-asset, per-horizon log-return regression head. The intuition: if the encoder can predict the actual log-returns at multiple horizons, the value head and the policy head get a richer factorized signal than the ones-and-zeros of the direction classifier. It's the auxiliary loss equivalent of giving the encoder a scale.

By the time TASK_736 landed in late April, the working stack had *both* heads on, plus a value-weight fix, plus cross-sectional channels, plus a 32-channel encoder. TASK_736 produced the first credibly positive Stage-B result of the project — Alpha **+3.87%** averaged across the six-period benchmark, 6/6 beats-random. We had been wandering in flat-or-negative-Alpha territory for so long that +3.87% felt vertiginous.

The problem with +3.87% is that you cannot tell which of the four pivots earned it. TASK_736 was the kind of result that punishes itself: a positive number with three plausible parents. So I wrote TASK_737 — a flag-flip ablation matrix specifically designed to tear the heads off one at a time and see which part of the stack was load-bearing.

The matrix is a 2×2 on the two auxiliary heads, with the same Phase 1 checkpoint, the same value-weight, the same 32-channel encoder, the same six-period benchmark. Naming convention: `a` is rank, `b` is aux. Four cells:

| Tag | rank (a) | aux (b) |
|---|---|---|
| `a0b0` (minimal) | OFF | OFF |
| `a0b1` (no_rank) | OFF | ON |
| `a1b0` (no_aux_solo) | ON | OFF |
| `a1b1_repro` | ON | ON |

The expected null result was, frankly, boring: both-on > each-alone > neither, by margins of a percentage point or two. The two heads supervise different surfaces; they should provide non-overlapping signal; they should compose roughly additively. If `a1b1_repro` came in close to TASK_736's reference (+3.87%) and the single-head variants came in lower, the verdict would be "the stack composes, keep it, move on."

I had budgeted, in my head, about half a day to read the table when it landed and write the move-on memo.

Half a day is approximately what the morning gave me. The memo is what got written. The memo was wrong about almost everything.

---

## The result, verbatim from §95

This is what `lore/notes_2026-05-13.md` §95 says, transcribed exactly. I am quoting myself from the same morning — there is no later editorial layer to thank for the boldness:

> ## 95. Full 4-cell TASK_737 result (07:00)
>
> Pulled overnight checkpoints + ran benchmark eval on both new ones (`a1b1_repro` + `no_aux_solo`) in parallel. Now have:
>
> | Variant | rank | aux | Bal.Acc | Alpha | Beats |
> |---|---|---|---|---|---|
> | **`no_aux_solo`** ⭐ | ON | OFF | 0.266 | **+6.21%** | 6/6 |
> | `no_rank` | OFF | ON | 0.240 | +3.25% | 6/6 |
> | `a1b1_repro` | ON | ON | 0.249 | **-1.99%** | 6/6 |
> | `minimal` | OFF | OFF | 0.257 | -4.99% | 6/6 |
> | TASK_736 ref (a1b1, old) | ON | ON | 0.265 | +3.87% | 6/6 |
>
> **The two heads interact NEGATIVELY** (-13 pp combined effect). The TASK_736 baseline (both heads on) is now known to be sub-optimal — the `no_aux_solo` config with rank-only beats it by ~2.3 pp Alpha.
>
> a1b1_repro (today) at -1.99% vs original a1b1 (+3.87%) — 5.9 pp gap on the same config. **Single-seed variance is large** on this setup. Multi-seed validation (TASK_743) is the highest-priority next step.
>
> — `~/Trader/lore/notes_2026-05-13.md` §95

I want to walk the reader through this slowly, because it is unusually self-incriminating and I should not let the bold typography hide the fact that the same paragraph contains both the conviction *and* its undoing.

The conviction is the first sentence: *The two heads interact NEGATIVELY (-13 pp combined effect)*. That is a mechanistic claim. It says something specific about how the rank and aux losses compose in parameter space.

The undoing is in the third paragraph: *a1b1_repro (today) at -1.99% vs original a1b1 (+3.87%) — 5.9 pp gap on the same config. Single-seed variance is large on this setup.*

I had, in the same writeup, written down the prior that should have killed the claim. The reproduction of the previous baseline configuration disagreed with itself by 5.9pp Alpha, on the same code, same data, same hyperparameters — only the seed (and the wall-clock of the run) had changed. That is a measurement of the *noise floor*. The noise floor was 5.9pp on the variant I was using as a comparison anchor.

The headline claim — that `no_aux_solo` beat the baseline by 2.3pp — fits comfortably *inside* that noise floor. Two-point-three is less than five-point-nine. You could roll the dice and reverse the result. I wrote both numbers down within four lines of each other and I did not see the second one because I was already busy with the first.

The marginal-effects table from `runs/737_eval/result.md` made the conviction worse, because it gave the negative-interaction story a *structure*. Adding the rank head when aux is off improves Alpha by +11.2pp (minimal → no_aux_solo). Adding the aux head when rank is off improves Alpha by +8.2pp (minimal → no_rank). Each head alone is strongly positive. But adding the rank head *when aux is already on* costs −5.2pp, and adding the aux head *when rank is already on* costs −8.2pp. The interaction term is approximately −13pp, hence the bolded phrase.

This is the shape of a result a researcher *wants* to be true. It says something. It is not "we improved a number" — it is "we discovered a mechanism." The shape itself was part of the seduction. I'll come back to that.

---

## The mechanism I invented before lunch

By eight in the morning I was no longer just looking at the table. I was building the explanation. This is the dangerous part, and I am going to walk through it honestly because the article only works if I show the inside of the mistake.

The story I drafted in `runs/737_eval/result.md` reads, slightly trimmed:

> **Strong negative interaction.** Either head alone is a clear positive contribution. Stacking the two cancels out and nets negative (interaction effect ≈ −13 pp).
>
> Mechanism (hypothesis): both heads project from the same per-asset feature stream toward correlated cross-sectional ranking targets — rank head supervises argsort-of-future-cum-return directly; aux_price head supervises per-horizon log-returns, which is essentially the same information one transformation away. Combined they over-regularize the shared encoder and constrain the policy/value heads' usable signal.
>
> — `runs/737_eval/result.md`

This is a real-sounding sentence. Let me state what it would mean if it were true, because that is the part I had quietly skipped.

The encoder is a shared trunk. Its output feeds into the value head, the policy head, the direction head, the rank head, and the aux head. Each downstream head pulls on the trunk via its own loss. If two of those heads happen to pull *in the same direction in parameter space* — that is, if the gradient of the rank loss with respect to the trunk parameters and the gradient of the aux loss with respect to the trunk parameters are highly aligned in expectation — then their sum has reduced effective rank. The trunk gets over-trained on the subspace they both want, at the expense of the subspace that the value and policy heads need.

I sketched this on the back of a piece of paper. The sketch was a Venn diagram of gradient subspaces with two of the auxiliary heads overlapping by maybe 70% and the value head sitting off to one side, getting starved. I labelled the overlap "degenerate basin" and felt clever.

The math, if it had worked, would have been: let $g_r$ be the expected rank-loss gradient on the trunk, $g_a$ the expected aux-loss gradient, $g_v$ the value gradient. The trunk update is approximately $\eta(\lambda_r g_r + \lambda_a g_a + g_v)$. If $\cos(g_r, g_a) > 0.7$, then $\lambda_r g_r + \lambda_a g_a$ approximates $(\lambda_r + \lambda_a) g_r$, doubling the auxiliary pressure on a single direction and leaving less effective capacity for the value head. The encoder converges to a solution that is excellent at the rank-projected subspace and impoverished elsewhere; the policy makes confident calls on small alpha and over-trades; Alpha collapses.

The story is *plausible*. It is not, on the face of it, wrong as a generic mechanism — versions of this have been observed in multi-task learning and have a literature behind them. It is also, in this specific case, a story I built backwards from one number. There was no measurement of gradient alignment. There was no computation of effective rank. There was no held-out replication. There was a 4-cell table, a difference of 8.2pp on the rank=ON axis, an opposite-sign 5.9pp self-disagreement on the comparison anchor, and a researcher who wanted his Saturday morning to mean something.

What I had done, in technical terms, is fit a mechanism to a single data point. The mechanism was post-hoc. The mechanism was specific enough to sound predictive even though it was actually descriptive. And — this is the part that almost always reveals the trick — the mechanism made *no quantitative prediction*. I had not written down what the gradient alignment would be if I measured it. I had not written down what `no_aux_solo` should score on a held-out seed. The story was unfalsifiable as drafted, which is exactly the kind of story that fits noise without resistance.

I want to be careful about a thing here. Post-hoc mechanism-fitting is not always wrong. Sometimes the experiment really does reveal a mechanism, and sometimes the mechanism really does predict the next data point. The risk is that you cannot tell which case you are in until you collect the next data point. Until then, the story and the noise are isomorphic. They feel exactly the same from the inside.

I felt very good about the diagnosis. The Venn diagram is still on my desk somewhere.

---

## What I did next

The morning had a momentum to it. By 07:30 §95 was written. By 08:00 I had pushed the result into the project memory at `~/.claude/projects/-home-lukas-Trader/memory/MEMORY.md` with the line:

> The new best Stage-B config is `no_aux_solo` (rank-only).

That sentence, eight days later, is one of the things I have had to overwrite. At the time it felt like a recording, not a claim.

Then I built the queue. The decision matrix from the TASK_737 spec was specific about what to do if a single-head variant beat the both-heads variant: keep the winning variant as the new baseline, queue multi-seed validation, then build the next campaign on top of the new baseline. I followed the matrix. By mid-morning I had drafted four experiment SPECs that all assumed `no_aux_solo` was the canonical Stage-B starting point:

- `experiments/743_no_aux_multiseed/SPEC.md` — five-seed validation of `no_aux_solo`. Seeds 7, 13, 23, 31, 42. The SPEC's framing was that this was a *formality* — confirming the bound, not testing the claim.
- `experiments/739_dense_sign/SPEC.md` — per-asset directional oracle, to be stacked on `no_aux_solo`.
- `experiments/734_flow_matching/SPEC.md` — flow-matching dynamics loss, also to be stacked on `no_aux_solo`.
- `experiments/740_bella_distill/SPEC.md` — Bella-generator distillation, gated on Bella precompute.

Every SPEC said *use no_aux_solo as the baseline*. I refreshed the master `experiments/QUEUE.md` to reflect the new world. I added a `--seed` CLI flag to `train_phase2.py` — it had been hardcoded to 42 since the project's beginning, an oversight that becomes more interesting in the next article — committed the change as `a0f7dec`, and deployed it.

Then I went to make another coffee. Then I came back to the desk and did the thing that, in retrospect, I am most embarrassed by: I started to compose, in my head, the way I would tell someone about this. Not write it up — *tell* someone. The conviction had moved from the writeup to the bones. I was rehearsing a story about how the rank head had been doing all the work all along, and the aux head had been a polite-looking saboteur, and TASK_736's positive result had been the rank head dragging a passenger.

I want to mark, here, the exact shape of that rehearsal. It is the shape of *premature canonization*. The result has gone from *new data point* to *the way the model now is* before the second seed has landed. The lore entry has been written. The memory has been updated. The downstream experiments have been re-baselined. The story has been refined. The internal monologue has shifted from "this is one observation" to "this is the new state of the world."

There was a five-seed run queued. The first two seeds (7 and 13) were already launched. The flags would drop around 22:30. I would have one more data point — seed 7 — by the end of the evening, and the rest of the answer by the next morning.

I told myself the multi-seed run was a sanity check. I told myself the conviction would survive it. Both of those statements were the same statement.

---

> *The journal is a working document. The Apothecary writes in it while the leaf is still wet, because the wet leaf is the truth of the morning. She has a separate page, in a separate ink, for entries that have survived the night. She does not always remember which page she is on.*

---

## The kickback is structural

This article's kickback is not the seed-13 number. The seed-13 number belongs to the next article. It would be unfair to the structure of the two days to reveal it here; the two-day arc only works if you let the morning's certainty hold its ground through the afternoon and into the evening, the way it did on the actual day.

The kickback in *this* article is structural, and it is the moment of *premature conviction itself*.

I drafted a victory writeup before the second seed had run. I wrote a mechanistic story to fit a single data point. I updated a memory entry that would, the next morning, need to be retracted. I composed four downstream SPECs on top of a baseline I had measured exactly once. None of this is unusual in ML practice — it is, in fact, the default. The default is the kickback.

There is an epistemic structure to the moment that I want to name precisely, because it generalizes well beyond this one experiment. In machine learning, intermediate results live in a permanent superposition between *real* and *variance* until you have collapsed the wave function with replication. The collapse is not optional. The collapse is not "good practice." The collapse is what the result *is*. A number from a single seed is not a *measurement* — it is a *sample*. The measurement is the distribution. Treating a sample as the measurement is the original sin of small-budget research.

We all do it anyway. The reasons are structural. Replication is expensive. Each Stage-B training is two hours on the workstation 3090; a five-seed sweep is ten GPU-hours, plus eval, plus disk space, plus the cost of running it instead of running the *next* experiment you were excited about. The cost of running the sanity check is paid in the currency of the work you don't do; the cost of *not* running it is paid only in the currency of being wrong on a longer time horizon. The asymmetry favours skipping. The asymmetry also favours being wrong.

Two specific symptoms of skipping the collapse have a pattern I can now recognize:

**One: writing the writeup before the reseed.** The act of writing the writeup is itself a form of conditioning. Once the result has been described, with mechanism, with marginal-effects table, with a sentence in the project memory file, the result becomes *load-bearing*. The next data point will be read against the writeup. If the next data point disagrees, the disagreement will be coded as "noise on the new result" rather than "evidence that the new result was noise." This is not because the researcher is dishonest; it is because the writeup has built a frame and the frame is doing the reading.

**Two: building downstream commitments on the new baseline.** Once four SPECs have been refreshed to use `no_aux_solo` as the baseline, the cost of demoting `no_aux_solo` is no longer one writeup. It is four SPEC revisions, plus the QUEUE.md, plus the memory file, plus, eventually, telling someone you were wrong. The cost of demotion compounds with every downstream commitment. The conviction does not just survive — it accretes.

Both of these had happened by lunchtime on 2026-05-13. I had loaded the morning with infrastructure that depended on a single observation being a measurement.

I want to stress that I was not unaware of the multi-seed problem in the general case. In the same paragraph where I wrote *the two heads interact NEGATIVELY (-13pp combined effect)*, I also wrote *Single-seed variance is large on this setup. Multi-seed validation (TASK_743) is the highest-priority next step.* I knew. The knowing did not stop the writing. That is the part that is worth lingering on. The procedural knowledge — "this should be reseeded" — coexists in the same paragraph with the substantive knowledge — "this is a real effect with a real mechanism" — and the procedural knowledge does not get to vote.

I went to bed slightly elated. Two more seeds were running. The conviction was about to be confirmed. I did not check the dashboards once before I closed the laptop.

---

## What I would do differently

The lesson from this morning is short enough to fit on an index card.

**Never write the writeup before the reseed.** The lore can be drafted, in scratch form, as raw observations — *seed 42 came in at +6.21%, baseline reproduced at −1.99%, here is the table* — but the *interpretation*, the *mechanism*, the *promotion to canonical*, must wait. Not as a stylistic preference. As a procedural firewall. Until the wave function collapses, the writeup is fiction. The fiction may turn out to be non-fiction in retrospect. That does not retroactively make it the right thing to have written in the present.

There are a few second-order lessons that are harder to put on the card.

The first is about the seductiveness of *negative interaction* findings specifically. When you see a 2×2 where the diagonals beat the off-diagonals — when the two-on cell does worse than the each-on cells — the shape of the result *feels mechanistic*. It feels like you have found a real interaction. The mechanistic-shape is part of what bypassed my noise filter on the morning of 2026-05-13. I was not just seeing a number; I was seeing a *pattern*. Patterns are harder to dismiss than numbers, because the brain is built to lock onto them, and the brain does not natively distinguish between "pattern in the data" and "pattern in noise." A 4-cell table with a strong interaction term is exactly the shape of finding that begs to be confirmed because it begs to be *understood*. It is, for the same reason, the shape of finding that most often fits noise.

The second is about the loop between *story* and *result* in conviction-building. Conviction in ML is not built on the strength of any single result. It is built on the loop: a result comes in, a story gets drafted, the story makes the result feel coherent, the coherence makes the result feel reliable, the reliability lets the story stiffen, the stiffer story makes the next noisy result easier to fit. The loop runs at the speed of writing. Slowing the writing is the only mechanical defense.

The third — and this one I keep relearning — is that the writing is itself an experimental intervention. The lab notebook is not a passive record of the work; it is a part of the work. What I write down today changes what I am able to *read* tomorrow. The seed-13 number, when it lands, will not be read against a clean prior. It will be read against `lore/notes_2026-05-13.md` §95, and against the four SPEC files I refreshed, and against the memory entry I have already updated. The reading will not be neutral. The reading cannot be neutral.

The only defense is procedural: do not let the writing get ahead of the data. Do not promote a single observation to a measurement until it has been measured. Do not refresh the downstream SPECs to use the new baseline until the new baseline is more than a single seed. Do not type the word NEGATIVELY in bold until there is a second piece of evidence for it.

I did none of these things on the morning of 2026-05-13. The morning ended with me feeling like the project had turned a corner.

---

## Closing

It is now 2026-05-21, eight days later. The canonical Stage-B baseline in the project is back to a1b1, both heads on. The phrase "no_aux_solo is the new best" has been replaced, in the project memory, with a longer phrase that includes the words *upward outlier* and *single-seed variance*. The four SPECs have been edited back to use a1b1. The `MEMORY.md` entry for TASK_737 now begins *Single-seed ablation suggested* and ends *Multi-seed validation OVERTURNED this*. The Venn diagram is still on my desk. I have not thrown it out, because the next time I draw something like it I want to be able to remember what it felt like to be sure.

The journal entry from the morning of 2026-05-13 has not been deleted. The bolded sentence has not been un-bolded. The date is still there. I have learned to leave it.

> *The journal entry is crossed out. The date is kept. The shelf is dusted. The next specimen is brought in.*

---

## Sources

- `~/Trader/lore/notes_2026-05-13.md` §95 — primary, the morning-of writeup quoted verbatim
- `~/Trader/lore/notes_2026-05-13.md` §96–§97 — the queue and the pre-compaction state, source for "what I did next"
- `~/Trader/runs/737_eval/result.md` — full 4-cell ablation table, marginal-effects table, mechanism-hypothesis paragraph
- `~/Trader/tasks/TASK_737_ABLATION_SUITE.md` — the spec, decision matrix, and the "significantly" threshold (≥2pp Alpha) that the −13pp interaction so dramatically cleared
- `~/Trader/runs/737_ablations/` — raw checkpoint output directories per variant
- `~/Trader/.claude/projects/-home-lukas-Trader/memory/MEMORY.md` — the TASK_737 entry, including the since-overturned canonization and the eventual retraction
- `~/Trader/experiments/QUEUE.md`, `experiments/{743,739,734,740}_*/SPEC.md` — the four downstream commitments built on the single-seed result on the same morning
