# Five Seeds and a Funeral

*On the morning the champion turned out to be variance, the row that did the killing, and the operational rule that came out of the wreckage.*

> *The bloom withered overnight. The label on the jar still reads "viable"; the gardener crosses it out, writes "inconclusive," dates it, and waits. The dose, the dose, the date. A specimen mislabeled is worse than a specimen unlabeled — both hands must write, and one of them must be tomorrow's.*

---

## The row

It is the morning of May 14, 2026. The desk fan in `workstation` is on; the 3090 is idling at 38 °C after the overnight evals; the coffee is the second cup, still warm enough to drink but cool enough to forget about. I open `lore/notes_2026-05-13.md` — yes, the file is dated 13, because §99 was appended at 07:00 today and a new lore file did not yet exist when the entry started — and I scroll to the bottom. The overnight runs have filled in the morning's section while I slept. Five rows. Four are fine. The third row is not.

The third row is seed 13: **−4.38 % Alpha**.

I read it twice. I do not yet feel anything; the body has not caught up. What I am feeling is a small administrative discomfort, the way you feel when a return form rejects a digit and you have to start again. I scroll up to confirm I am looking at the right experiment. Yes. TASK_743. The five-seed validation of no_aux_solo — the configuration I had, the morning before, written up as the *new canonical Stage-B baseline*. Yesterday's hero. Today's row 3.

I scroll down to the AVG row. **+3.61 %, std ≈ 4.7 pp.** The previous canonical reference — TASK_736's a1b1 single-seed result — was +3.87 %.

The numbers have not finished arranging themselves yet in my head. I know, abstractly, what the row means. The body has not caught up. I get out of the chair, stand in the kitchen, finish the coffee. Then I come back and read the row again because that is what the work is — the row was there last night while I slept and it is there now while I drink coffee and it will be there tomorrow morning. The texture of conviction I had yesterday and the texture of conviction I have today are indistinguishable. The only thing that changed is that a different number is on the screen.

This is Article 8. Article 7 ended with the sentence "we have something." Article 8 opens with the row that took it back.

---

## Why five seeds, and why these five

A short technical aside, because some readers will not have followed the previous installment and I do not want to assume.

In the Foxglove + Poppy training stack, a "seed" is the integer that initializes the JAX PRNG key. It controls weight initialization, dropout masks, sample order in the SegmentPool, action sampling at rollout time, the lot. Two runs with identical config and different seed are two independent draws from a noisy training process. The noise sources are real and not fully suppressible on this stack — at minimum, sample order through the SegmentPool's 17-regime carousel, dropout, stochastic policy-gradient updates, and value-target Bellman noise compound across 200 epochs into Alpha differences that are not small.

The TASK_737 ablation — the subject of Article 7 — ran with seed 42 only. Four cells of a 2×2 (rank head × aux head), one seed each. The headline was no_aux_solo at **+6.21 %** Alpha versus a1b1 at **−1.99 %**, an 8.2 pp gap that I had explained with a confident mechanism story about heads over-regularizing a shared encoder.

The five seeds were chosen as follows. Seed 42 was the existing TASK_737 run; it stays in the table for free. The four new seeds — 7, 13, 23, 31 — were picked because they were small primes I had not used elsewhere in the project and because nothing about them was supposed to matter. Multi-seed validation is supposed to be boring. You run it because you should, not because you expect it to surprise you.

The internal standard at the time, written into the QUEUE.md spec, was: *if 3 of 5 seeds reproduce the +5 pp gap, the finding is real.* Three out of five was the bar. Two out of five was the trapdoor. Below two was the funeral.

The reseed launched on the evening of May 13. Two waves of two seeds each — a 2-wide concurrency limit learned the hard way after a memorable deadlock that taught me what happens when three of these training jobs share the same machine. The evals ran in parallel between 06:32 and 06:42. By the time I woke up, the table was complete.

---

## The table

From `lore/notes_2026-05-13.md` §99, recorded the morning of 2026-05-14:

| Seed | Bal.Acc | Alpha | +ve periods | Beats random |
|---|---|---|---|---|
| 42 (TASK_737 ref) | 26.6% | **+6.21%** | 6/6 ✅ | 6/6 ✅ |
| 7 | 26.0% | **+2.84%** | 3/6 ❌ | 6/6 ✅ |
| 13 | 24.9% | **−4.38%** | 2/6 ❌ | 6/6 ✅ |
| 23 | 24.0% | **+7.58%** | 4/6 ✅ | 6/6 ✅ |
| 31 | 25.5% | **+7.82%** | 4/6 ✅ | 6/6 ✅ |
| **AVG** | **25.4%** | **+3.61%** | — | 6/6 ✅ |
| **STD** | ~0.9 pp | **~4.7 pp** | — | — |

Range: −4.38 to +7.82 = **12.2 pp spread**. The lore note's exact wording is worth reading slowly:

> "Seed 42 was an upward outlier. The TASK_737 finding 'heads interact negatively, no_aux beats a1b1' is **UNRELIABLE** — driven by single-seed variance."
>
> — `~/Trader/lore/notes_2026-05-13.md` §99

The detail I want to dwell on, because it is the one that makes the funeral honest: **beats-random is 6/6 in every seed**. The model is not broken. It is finding signal. The signal is asset selection (the rank-head story) and it survives every seed. What does *not* survive is the Alpha number — the magnitude of the edge over the same-period buy-and-hold benchmark — which scatters across a 12-pp range. The system is doing something real; the *amount* it is doing is mostly noise on a five-seed sample.

The other detail: Bal.Acc is tight (~0.9 pp std across seeds) while Alpha is loose (~4.7 pp std). The classifier-side metric is stable; the trader-side metric is noisy. This is the *third* time in the project I have seen that shape — Bal.Acc converges, Alpha doesn't — and it is now the texture I should have been expecting. I wasn't.

---

## The moment of seed 13

I want to write this part as honestly as I can, because the moment is the article.

The five rows do not arrive together. Reading is a sequence. The first row is seed 42 — the known +6.21 %. The second row is seed 7, **+2.84 %** — lower than 42 but still positive, and at this point I am already running an internal narrative that says *fine, single-seed variance, the mean will hold*. The third row is seed 13: **−4.38 %**. The fourth is seed 23 at +7.58. The fifth is seed 31 at +7.82.

If the rows had arrived in a different order — say 31, 23, 42, 7, 13 — I would have spent four rows feeling reassured (good, good, good, good) and then one row absorbing a single bad seed at the end. The narrative would have been: *one outlier, four confirmations, finding holds.* The story I would have written would have been different. Possibly wrong, but different.

The rows did not arrive that way. The −4.38 lands in the middle. By the time I am reading rows 4 and 5, I am already inside the standard-deviation calculation, the mean is already dropping into the +3.6 region, and the +7.58 and +7.82 land not as confirmations but as data points in a distribution I am building in my head and do not like the shape of. The two "high" seeds and the one "low" seed and the +2.84 lukewarm seed and the +6.21 reference seed are five samples from something with a 4.7 pp standard deviation, and the +6.21 is now sitting at roughly +0.55σ above the mean. Which is to say: it is an ordinary draw. There is nothing to explain.

I sit with this for a few minutes. Then I do the thing I have learned to do in these moments, which is to write it down before doing anything else. Not to fix it. Not to investigate. Not to launch the next experiment. **Write it down first.** The lore §99 entry exists because I drafted it before I let myself touch anything else. The hardest part of empirical work is not the experiment; the hardest part is letting the result mean what it means before you start negotiating with it.

The Apothecary, if she were standing behind me, would say: *the specimen withered; record the date; do not re-plant it under a different label.*

---

## The math, slowly

It helps to walk through the statistical argument explicitly, because the argument is the article's load-bearing wall.

Five seeds, Alpha values:

$$
\{+6.21,\ +2.84,\ -4.38,\ +7.58,\ +7.82\}
$$

Sample mean:

$$
\bar{x} = \frac{6.21 + 2.84 - 4.38 + 7.58 + 7.82}{5} = \frac{20.07}{5} = +4.01\%
$$

(The lore note records +3.61 % because it uses a slightly different rounding pathway through the AVG row of the raw eval JSONs; the result.md file records +4.02 %. The difference is a handful of basis points downstream of how individual seeds' per-period Alphas were averaged before being averaged across seeds — paired vs unpaired aggregation. I'll use the lore note's +3.61 % for the rest of this article because it is the entry that drove the decision; both numbers tell the same story.)

Sample standard deviation, with the lore note's value:

$$
s \approx 4.7\ \text{pp}
$$

Standard error of the mean:

$$
\mathrm{SE} = \frac{s}{\sqrt{n}} = \frac{4.7}{\sqrt{5}} \approx \frac{4.7}{2.236} \approx 2.10\ \text{pp}
$$

A naive normal-approximation 95 % CI:

$$
\mathrm{CI}_{95} \approx \bar{x} \pm 1.96 \cdot \mathrm{SE} = 3.61 \pm 4.12 = [-0.51,\ +7.73]\ \text{pp}
$$

The CI brackets zero. The one-sample t-statistic against the null hypothesis of zero Alpha effect:

$$
t = \frac{\bar{x} - 0}{\mathrm{SE}} = \frac{3.61}{2.10} \approx 1.72
$$

With four degrees of freedom (n − 1), a two-sided t-test gives roughly p ≈ 0.16; one-sided against the more permissive "positive Alpha" alternative, roughly p ≈ 0.08. Neither clears the conventional 0.05 bar. Even the comparison against the a1b1 reference (+3.87 %) is hopeless: |3.61 − 3.87| = 0.26 pp, sitting at about 0.12 SE.

Where, in this distribution, does the original TASK_737 result live? +6.21 is **(6.21 − 3.61) / 4.7 ≈ +0.55σ** above the mean. That is not an outlier in any honest sense. That is an ordinary upward draw. The thing I had written a mechanism story about — "the two heads interact NEGATIVELY (−13 pp combined effect)" — was fit to a +0.55σ fluctuation.

The operational rule that crystallized that morning, and that I have not relaxed in the week since:

- **Single-seed Alpha gaps below ~5 pp on this stack are untrustworthy.** Any claim of improvement smaller than 5 pp needs at least 3 paired seeds before it is treated as real.
- **Gaps below ~10 pp need 5 paired seeds.**
- **Bal.Acc / loss / dir-acc differences are not substitutes for Alpha differences.** They have a different noise floor and they do not predict deployment behavior.

You can see the math behind why. With σ ≈ 4.7 pp per seed and SE = σ/√n, you need n = 4 to push the SE below 2.5 pp and n = 9 to push it below 1.6 pp. A 5 pp single-seed gap and a 0 pp single-seed gap are about one σ apart, which is roughly the noise of a single coin flip. Three seeds is the minimum number at which a "real" 5 pp effect produces enough separation from the null to be felt over the noise.

This is not a new finding for reinforcement learning as a field. Henderson et al. published "Deep Reinforcement Learning that Matters" in 2018 making approximately this point with approximately this kind of variance plot. I had read that paper. I had read other papers. I had nodded at them. And then, on May 13, with a single +6.21 % seed on the table, I had written a victory note anyway. The lesson is not that I did not know. The lesson is that knowing in the abstract and feeling in the body are not the same skill. The reseed is what closes the gap.

---

## The retraction

The afternoon of May 14 was the retraction.

The mechanics were small and unromantic. I opened `MEMORY.md` — the project's auto-loaded memory index — and re-wrote the TASK_737 entry. The previous version had read, roughly, "no_aux_solo is the new canonical Stage-B best, +6.21 % Alpha vs a1b1 −1.99 %." The replacement, committed and still in place today, reads:

> "**TASK_737 + TASK_743 result (2026-05-13 → 2026-05-14)** — Single-seed ablation suggested `no_aux_solo` (rank ON, aux OFF) was new best at +6.21 % Alpha vs a1b1 at −1.99 %. Multi-seed validation (TASK_743, seeds 7/13/23/31 + 42) **OVERTURNED this**: AVG Alpha **+3.61 %**, range **−4.4 to +7.8 (12.2 pp spread)**. `no_aux_solo` is NOT reliably better than a1b1 (TASK_736 ref +3.87 %). Single-seed +6.21 % was upward outlier. **Canonical Stage-B baseline reverts to a1b1 (rank+aux).** Single-seed Alpha gaps ≤ ~5 pp are within noise — any new claim of improvement needs **≥3-seed validation**."
>
> — `~/.claude/projects/-home-lukas-Trader/memory/MEMORY.md`

I opened `runs/737_eval/result.md` and added a banner near the top noting that the file's headline conclusion was now historical, not canonical — superseded by `runs/743_no_aux_multiseed/result.md`. The mechanism story in §2 of that file — the "strong negative interaction, ≈ −13 pp combined effect" paragraph — is still there, because deleting it would have been dishonest. It happened. I wrote it. I want the record to show that I wrote it, and then I want the record to show, in a banner immediately above it, that I was wrong.

I updated `experiments/QUEUE.md` to mark TASK_743 done and to deprecate "no_aux_solo as new best." I changed the base configuration for the four downstream SPECs — 739 dense_sign, 734 flow_matching, 740 bella_distill, 741 Mint — so that each would stack on **a1b1** going forward, not on no_aux_solo. This last edit took longer than the others, because each SPEC had a paragraph somewhere that justified its choice of base config by reference to the (now-retracted) TASK_737 finding, and each paragraph had to be re-written or marked obsolete.

The peculiar disquiet of having drafted a victory writeup the previous morning and now having to undraft it is hard to describe precisely. It is not regret, exactly — the experiment was the right experiment to run, the writeup was a reasonable first-pass conclusion from a single seed, and the reseed was the correct next step. The discomfort is something else. It is the discovery that the version of me from twenty-four hours ago could not have known the result was wrong, and yet the version of me from twenty-four hours ago was confident anyway. The confidence and the correctness were independent variables. I had not noticed.

> *The Apothecary re-labels the specimen "inconclusive." She does not erase the prior label. She crosses it through, twice, with a single horizontal line in the same ink, and dates the new mark beside it. Tomorrow's reader must be able to read both — the wrong reading and the corrected one — or the journal is not a journal, it is a press release.*

---

## The kickback, in three layers

The single-data-point kickback compounds outward.

**Layer one — every prior single-seed Alpha claim in the project is now suspect.** The TASK_136 era encoder benchmarks, the early Foxglove dir-acc sweeps, the late-April rank-head ablations — most of those were single-seed comparisons evaluated on Alpha gaps in the 1–5 pp range. The new statistical floor invalidates the *form* of those comparisons. It does not necessarily invalidate the conclusions; some of those choices were probably correct by accident, in the same way that a stopped clock is correct twice a day. But "probably correct by accident" is not a state I get to occupy with confidence, and I cannot tell which of the past year's decisions were the real ones and which were the +0.55σ fluctuations I happened to write down. The honest accounting is that **I do not know how many of the project's load-bearing micro-decisions were noise-rationalized**. The pessimistic upper bound is "many." This is not a single experimental failure. It is a retroactive policy that demotes a whole year of micro-results.

**Layer two — the mechanism story from yesterday needed to be retracted along with the result.** Article 7 closes with the mechanism I invented to explain TASK_737: that the rank head and the aux head, projecting from the same per-asset feature stream toward correlated cross-sectional targets, over-regularize the shared encoder and crowd out the policy/value heads. It is a plausible story. It might even be true. But what I learned on May 14 is that it was *not* evidenced; it was *story-fit* to a single +6.21 % data point that turned out to be a +0.55σ draw. The mechanism can still be correct in some deeper sense — encoders can over-regularize, redundant heads can hurt — but that is not what the data said. The data said "this seed was lucky." The asymmetry between story-fit and evidence-fit is the painful one, because both feel the same when you are writing the paragraph. Stories are cheap. Evidence is expensive. Machine learning rewards the production of stories more than is healthy, and I, as a writer, am part of the problem.

**Layer three — and this is the uncomfortable one — I was about to be sure about something untrue.** Not "I made a mistake," because everybody makes mistakes; the mistake is the cheap part. The uncomfortable part is that the *internal texture* of the confidence I had on May 13 morning, with a +6.21 % single-seed result on the table, was indistinguishable from the texture of the confidence I have today about, say, the bull-blindness finding (which I do think is real). I cannot feel the difference between a 1.4σ outlier and a true effect. The body cannot tell me. The intuition cannot tell me. The Slack-message-to-myself voice that says *yes, this is the one* cannot tell me. The only mechanism in the entire stack that can disambiguate between "true effect" and "lucky seed" is the reseed itself. And the reseed is not free — it costs 4× the compute of a single run, plus the eval, plus the manual overhead of paired aggregation. The cost is exactly why I keep wanting to skip it, and the cost is exactly the price of being right.

> *Run the reseed. Always run the reseed. The body lies. The journal does not.*

There is a corollary kickback I want to name even though it is smaller, because it matters operationally: **on this stack, the noise floor on Alpha is ~4.7 pp.** Any single-seed Alpha differential below 5 pp is, by definition, a fluctuation. Most of the per-mechanism improvements I have been chasing — the rank head, the aux head, the Mint cross-sectional injection, the dense_sign target, flow_matching — sit in the 1–7 pp range when they sit anywhere. Which is to say: most of the project's experimental decision-making for the past two months has been operating *at* or *below* the noise floor of the very metric it was optimizing for. That is not a position from which strong claims can be made. It is a position from which only multi-seed paired claims can be made, and only carefully, and only after waiting for the right number of runs to finish.

---

## What I kept

A clean accounting, for the record.

**What survived:**
- The asset-selection signal. Beats-random was 6/6 in every seed. The model is finding *something*. The rank head is doing real work; the magnitude of that work is what is uncertain.
- The 2×2 ablation as a *design*. The four-cell structure (rank × aux) was the right experiment; it is only the single-seed *conclusion* that was unsupported. If I had run the 2×2 at three seeds per cell from the start, I would have learned almost the same thing without the U-turn — except that the multi-seed cost (12 runs) was, at the time, more than the queue could afford. The TASK_737 mistake was not running the 2×2; it was reporting it.
- The mechanism *hypothesis* about redundant heads over-regularizing a shared encoder. Demoted to "open question, not load-bearing." It may resurface in a Tier-2 experiment one day. It is no longer a load-bearing claim.

**What changed:**
- The canonical Stage-B baseline reverted to `a1b1` (rank + aux), the configuration TASK_736 had originally arrived at and which the multi-seed validation showed was statistically indistinguishable from no_aux_solo.
- Every downstream SPEC (739, 734, 740, 741) was re-based on a1b1.
- The operational rule appeared, in writing, in every subsequent task spec: *≥3 paired seeds for any Alpha claim below 5 pp; ≥5 paired seeds for any Alpha claim below 10 pp.* This is now non-negotiable.

**What I learned, in one sentence:**

*The thing I measure changes how I decide, and how I decide changes what I count as a result; if the thing I measure is noisy, then so are my decisions, and so are my results — and the only way to know the difference is to pay the multi-seed cost up front.*

Articles 6 and 8 share a deep lesson, even though they look like different stories. Article 6 said: *the classifier metric is not the trader metric*. Article 8 says: *the single-seed measurement of the trader metric is not the trader metric either*. In both cases, the operational mistake was the same: trusting a measurement that was cheaper than the thing it was supposed to measure.

The seed-13 number is now the symbolic ledger entry — the date stamp on the rule. **−4.38 %**, on the morning of 2026-05-14, around 07:00, in §99 of a file dated 05-13. I do not need to remember the rule when I remember the row.

---

## Closing

It is May 21 as I write this, a week later. The rule is still in force. Every campaign now budgets the multi-seed cost up front. The TASK_758, TASK_759, TASK_760 specs each have a "paired seeds: 3" line in their SPEC YAML and a "verdict requires ≥3 seeds; recommend 5 for borderline" line in their decision criteria. The flow_matching candidate from TASK_734 is the most recent one to clear the bar (paired 3-seed mean Δα +3.38 pp, all 6 periods positive), and I am still resisting the urge to write it up as a winner until the 5-seed extension lands. That resistance is the lesson, working.

The Apothecary's journal page for May 14 has more dates in the margin than the page for May 13. Smaller handwriting. Two ink colors. The dose, the dose, the date. A specimen mislabeled is worse than a specimen unlabeled. The journal admits both readings — yesterday's and today's — and keeps writing.

---

## Sources

- `~/Trader/lore/notes_2026-05-13.md` §99 — primary source. Morning-of-2026-05-14 entry in the file dated 05-13. The 5-seed table, the verdict, and the operational rule live here.
- `~/Trader/lore/notes_2026-05-13.md` §95 — the TASK_737 result that §99 retracted. Read together they are the whole story.
- `~/Trader/runs/743_no_aux_multiseed/result.md` — the full 5-seed per-period detail, alternative aggregation (+4.02 % vs the lore note's +3.61 %), and reference comparisons.
- `~/Trader/runs/737_eval/result.md` — the original 4-cell ablation result, now annotated as historical-not-canonical.
- `~/.claude/projects/-home-lukas-Trader/memory/MEMORY.md` — the post-743 entry that formalized the retraction and the ≥3-seed rule.
- `~/Trader/tasks/TASK_743_NO_AUX_MULTISEED.md` — the SPEC that defined the reseed.
- `~/Trader/experiments/743_no_aux_multiseed/SPEC.md` — the executable form.
- Henderson, Islam, Bachman, Pineau, Precup, Meger. *Deep Reinforcement Learning that Matters.* AAAI 2018. The paper everyone in this corner of the field has read and not internalized; the one I read and did not internalize until seed 13 made me.
- `~/Trader/docs/AI_bleed research/big gemini.txt` — *Coupled Policy-Prediction Collapse in Model-Based RL.* The variance signature in §99 is exactly the "stochastic policy drift" pattern that report describes; relevant for Article 10's structural pivot, but already visible here.
