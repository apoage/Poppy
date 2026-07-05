> *Observation: I have written the dose on the bottle and the dose on the chart. The dose on the bottle reads minutes; the dose on the chart reads days. Both inscriptions are in my hand. Neither inscription was wrong on its own. I did not check the units. The patient is fine, because the dose was never delivered.*

---

**by Lukas — May 21, 2026**

# From 60 Minutes to 93 GPU-Days: The Bella Precompute Disaster

*On a job that the SPEC said would finish over lunch, an arithmetic step I did not write down, a path bug that turned out to be a gift, and the moment a calculator app told me my project was going to finish in 2027.*

---

## "this is moving very slowly"

It is the afternoon of May 14, 2026. The TASK_743 funeral — five seeds, a champion that was an outlier, a 12.2-percentage-point spread — is two paragraphs behind me in the diary; I have just finished writing it up in `lore/notes_2026-05-13.md` and the lesson "anything under five points on a single seed is not signal" is fresh enough that I still feel it in my shoulders. The next thing on the queue is **TASK_740**: feeding the agent forward-rollout signals from **Bella**, the market generator. To make TASK_740 trainable on a sensible wall-clock, the SPEC calls for **precomputing** Bella's forecasts for the whole training corpus once, then having training loops read them from disk. Cheap reads, no live Bella forward passes in the loop. Standard distillation hygiene.

The precompute job has been running for about forty minutes. `workstation` is on. The desk fan is on. `nvidia-smi` in the right pane is showing a single 3090 chewing its way through something at a steady draw — fans spun up, memory pinned, the card doing real work. The left pane has the precompute tail. The tail prints a `tqdm` progress bar. The progress bar is moving. It is moving at **8.6 windows per second**.

I watch the bar for a while. Forty minutes in, the SPEC's "30–60 minutes" estimate is about to expire, and the bar is well short of the first period out of two hundred and seventy. Not 1%. Less. I open the calculator app.

The thing I am about to discover — the thing that this article is the postmortem of — is that the SPEC's estimate is wrong by **three orders of magnitude**. Not 1.5×. Not 5×. Not 20×. **A thousand.** The job, at the rate I am watching it run, is going to take **ninety-three GPU-days**. The SPEC author — me, two weeks earlier, on a different morning — had written *thirty to sixty minutes* with a straight face, because in the moment of writing it, my benchmark had been a single asset, on a single window, and the multiplication out to the full corpus was a step I had skipped. The calculator app is going to tell me this in about sixty seconds.

I am writing this article seven days later. The precompute that *should* have run is not the one we ran. The job I am narrating never finished, was never meant to finish, and the mechanism it was about to feed has since been closed for separate reasons. But the postmortem is the artifact. The lesson — the one I'm about to spend several thousand words on — is small, durable, and the kind of mistake that an experienced ML engineer keeps making until they make the discipline structural.

## Bella, "precompute," and the multiplication I should have written down

Bella is the market generator. The full architecture lives in `diff_teacher/`, and the readable starting point is `docs/UNIFIED_MARKET_GENERATION_ARCHITECTURE.md` if you want the long version. The short version, for this article: Bella is a flow-matching network that, conditioned on a **market skeleton** (regime sequence, vol-regime trajectory, correlation clusters, trend directions), generates a plausible OHLCV trajectory over a target horizon. She is *belladonna* in the apothecary frame — vision of unseen markets, eye medicine for an agent whose own input window is only sixty minutes long. The training pipeline (`diff_teacher/train.py`, `diff_teacher/teacher.py`) is two-phase: a VAE encoder for OHLCV windows, then OT-conditional flow matching on top of a hybrid Mamba/attention backbone. Inference is single-step at the flow-matching level but the conditioning machinery — skeleton encoder, the AdaLN gating, the bi-Mamba scan — is not free. The Bella checkpoint we are using for this job is the converged Phase-1 generator at `checkpoints/diff_teacher/final/generator`.

**TASK_740** — see `tasks/TASK_740_BELLA_DISTILL_HEAD.md` — is the experiment that wants Bella's forecasts as a *distillation target* for the agent's encoder. The agent's encoder sees a 60-step OHLCV window; Bella, trained at Foxglove-scale on the whole corpus, has internalized priors that 60 steps cannot contain. If we add an auxiliary head to the agent that learns to predict what Bella would predict for the next K steps, the encoder gets a richer training signal — without paying Bella's inference cost at training time, because the predictions are precomputed offline and read from disk.

That word — **precomputed** — is the whole shape of the problem.

The SPEC for the precompute looks reasonable on a first read. Walk the **270 periods** listed in `config/foxglove_periods.json` (each period is a roughly six-month minute-bar slice on a 64-asset rotation, see `mcts_trader/CLAUDE.md` on the rotation design). For each period, iterate the **~258,000 minute-bar windows**. For each window, ask Bella for **k=4** stochastic forward rollouts. Save the four predictions per window to an `.npz` shard keyed by the period's date range. The training loop later picks up the shards via a manifest and reads four `(B, n_assets, horizons)` tensors per batch — disk-cheap, MSE-cheap, deployable.

The SPEC's runtime estimate was *30–60 minutes*. I had benchmarked Bella on a single window and seen something on the order of a hundred-and-fifty predictions per second. A hundred-and-fifty per second times sixty seconds times thirty minutes is two hundred and seventy thousand. That number lives in the back of the head; "comfortable headroom for a precompute over a corpus." The two hundred and seventy thousand sat in one place in my brain. The number two hundred and seventy *million* — the actual workload size — sat in a different place. I never put them next to each other on paper.

The shape of the actual workload is this. **Two hundred and seventy** periods. **Two hundred and fifty-eight thousand** windows per period. **Four** stochastic samples per window. The multiplication is one line:

```
270 × 258,000 × 4 = 278,640,000 Bella forward passes.
```

The thirty-minute SPEC estimate implied an effective rate of about **150,000 forwards per second**. At sixty minutes, **75,000 per second**. Either rate would have required Bella to run at roughly a *thousand times* its measured single-window benchmark. I had written the SPEC without doing that division.

## The arithmetic, on the page, where it should have been from the start

The calculator app is open. I do this in the order it occurred:

```
278,640,000 forward passes total.
Observed: 8.6 windows/sec × 4 samples/window = 34.4 passes/sec.

278,640,000 / 34.4 = 8,099,000 seconds (rounded).
                   = 134,990 minutes.
                   = 2,249 hours.
                   = 93.7 GPU-days.
```

The diary entry I write later — `lore/notes_2026-05-13.md` §100, "Bella precompute — TASK_740 Step 1 FAILED + scale issue" — captures the projection in the form that survived:

> *- `config/foxglove_periods.json` has **270 periods**.*
> *- Each ~6-month period has ~258,803 minute-bar windows × k=4 samples.*
> *- Measured: ~8.6 windows/sec → ~8.3 hours per period.*
> *- Naive projection: **270 × 8.3 ≈ 2,241 hours = 93 days of GPU compute.***
> *- Storage: 270 × 250 MB = **~68 GB** of preds.*
> *- SPEC's "30-60 min" estimate is off by ~3 orders of magnitude.*
>
> — `~/Trader/lore/notes_2026-05-13.md` §100

Two thousand two hundred and forty-one hours. Ninety-three days. On the single 3090. With nothing else allowed to touch the card during that window.

I want to give that number its full weight, because the temptation when you discover a mistake of this size is to round it off into a war story and let it stand as a *near miss*. It was not a near miss. The job had already burned **eight and a third hours** of real wall-clock processing the first period before crashing — see the section on the path bug below — and at the moment I am watching the bar at 8.6 w/s on the second attempt, I have already paid most of a day for the first window of a corpus where the next 269 windows are each going to cost the same. If the SPEC's "30–60 minutes" had been true, the entire corpus would have been done before I sat down for lunch. Instead, the corpus would have finished in late August. Three full months of single-GPU compute, on a project that — as of this writing — has *six* months of total wall-clock spent on it.

I write `93` into the diary entry. I underline it. Then I close the precompute job.

## The path bug, which was a gift

I have been quietly misleading you. The version of events I just narrated — "the SPEC said thirty minutes, I watched the bar at 8.6 w/s, I opened the calculator" — is the *tidied* version. The actual first symptom of the failure was not slowness. The actual first symptom was that **the output was not writing to disk**.

The precompute had run for the full first period — eight hours and twenty minutes — produced a 250 MB shard, written it, and then crashed on the manifest-assembly step. The traceback is in `lore/notes_2026-05-13.md` §100:

```
File "tools/precompute_bella_predictions.py", line 210, in _process_period
  return {'path': str(out_path.relative_to(REPO_ROOT)), ...
ValueError: 'data/bella_preds/bella_preds_2018-10-17__2019-04-15.npz'
is not in the subpath of '/home/...
```

A path bug. Line 244 of `tools/precompute_bella_predictions.py` constructs `out_dir = Path(args.output_dir)` — taking the CLI argument as-is, which is **relative**. The `out_path = out_dir / "name.npz"` inherits the relativity. Then at line 210, when the code tries to write the shard's path into the manifest, it asks `out_path.relative_to(REPO_ROOT)` — and `REPO_ROOT` is absolute (line 52: `Path(__file__).resolve().parent.parent`). `Path.relative_to` is strict about this: you cannot compute a relative path between one relative path and one absolute path without resolving them first. So it raises.

The fix is one line: `out_dir = Path(args.output_dir).resolve()`. I did the patch, opened TASK_744 to track the fix, and relaunched the job to verify the shard would land on disk after the next period. **That is when the progress bar started moving at 8.6 windows per second in front of me.** That is when, with the path bug now out of the way, the *actual* scale of the work became visible.

If the path bug had not existed — if the manifest assembly had quietly succeeded on the first period and the job had slipped silently into the second — I would not have been sitting at the terminal watching the bar. I would have hit `tmux detach`, closed the laptop, walked away, and come back two days later to a job that had completed *somewhere between 2.1% and 4.2%* of its work. The path bug crashed the run at exactly the point where the second period was about to begin. It forced me to relaunch with eyes on the screen. It put me in front of the progress bar at 8.6 w/s with forty minutes to kill while I waited for the relaunched first period to finish. That is when I opened the calculator.

I want to be careful with this point, because there is a temptation in the engineering culture to dismiss it. We talk about bugs as monotone bad: every bug is a tax, every bug is debt, every bug is something to prevent. But this one was a *gift*. The path bug was a **synchronization failure** — it crashed the program at the boundary between two units of work, at a moment when I was still attending to the screen. It cost me eight hours of wasted compute on the first period's shard (which we kept; the path fix preserved the file). It saved me *two days* of wasted compute on a job that was structurally never going to finish. The unit economics of the path bug are wildly positive.

Most of the debugging culture I know doesn't have a clean word for this category. We have *graceful degradation* (a failure that produces correct partial output). We have *failing loudly* (a failure that surfaces visibly rather than silently corrupting state). What I'd want is a phrase like **diagnostic crash** — a failure whose only effect is to bring the operator's attention to a *different, larger* failure that would otherwise have proceeded undetected. The path bug at line 244 was a diagnostic crash. It noticed nothing about the scale problem. It had nothing to say about Bella's forward-pass rate. It crashed for an entirely unrelated reason at a structurally fortunate moment, and in doing so it bought me the forty minutes of attention I needed to discover that the SPEC was off by a factor of a thousand.

I do not want to over-romanticize this. The path bug was sloppy code on my part; the absent `.resolve()` would have been caught by any kind of unit test exercising the manifest path. The point is not "yay bugs." The point is that the *interaction* between two independent failures — one trivial, one structural — produced a fail-early property that the structural failure on its own would not have had. I owe the next-eighty-nine GPU-days I did not burn to a missing `.resolve()` call.

## What I should have done on the napkin

Sixty seconds with a calculator. That is what the discipline would have cost me.

The SPEC author — me, on the morning the SPEC for TASK_740's precompute was written — had benchmarked Bella on a single window and got something like 150 predictions per second on the 3090. That number, multiplied out for thirty minutes, is 270,000. That is the number that sat in my head when I wrote *30–60 minutes*. The *actual* number of predictions the job needed to produce was 278,640,000. That is the number I did not put next to 270,000.

Why didn't I?

I would like to tell a story about time pressure or distraction. Those stories would not be wrong, exactly — the SPEC was written while I was preparing TASK_743 and chasing the Mint variant for TASK_741, and the precompute felt like a clerical preliminary to the experiment that mattered. But the deeper reason is that the **small benchmark felt fast**. The Bella checkpoint was loading, the single-window forward was clearing 150 predictions per second, and "the precompute will take maybe half an hour" *resonated* with that observation in the way that estimates do when they're being constructed by gut rather than by arithmetic. The category I had silently put the job in was *"small precompute over a benchmark-confirmed fast operation."* The category I should have put it in was *"a moderate-scale Bella job multiplied by a number — 270 × 258k × 4 — that I have not yet held in my head."* The category error is the whole bug. The arithmetic is an effect.

This is the part where it is tempting to draw a moral about *engineering rigor* and then move on. I want to be more specific than that. The discipline I am about to mandate against this class of mistake is not "be more rigorous." The discipline is structural: **every SPEC with a runtime estimate must show its multiplication on the page**. Not as a comment. Not as a sentence ("the workload is large but manageable"). As an explicit line, in monospace, of the form:

```
total_work = N_periods × windows_per_period × samples_per_window
           = 270 × 258,000 × 4
           = 278,640,000

throughput  = measured_w_per_s × samples_per_window
           = 8.6 × 4
           = 34.4 ops/sec

ETA        = total_work / throughput
           = 278,640,000 / 34.4
           = 8.10e6 s
           = 93.7 GPU-days
```

The benefit of writing this on the SPEC is not that anyone will check it. The benefit is that *I* will see it. The act of typing the multiplication is itself the check. The reason I missed the factor-of-a-thousand is that I never typed the numbers. The fix is the typing.

Every SPEC in this project from May 14 onward includes this block. The template under `experiments/<id>/SPEC.md` has a `## Scale arithmetic` heading that the writer is required to fill in. The cost of the discipline is sixty seconds per SPEC. The cost of *not* having it, on May 13, was a full afternoon of investigation and very nearly a weekend of silently wasted compute.

## The kickback, in three layers

Causality kicks back at the end of every chapter in this series, and this chapter has three kickbacks stacked.

**The first** is the one I just walked through: the SPEC was off by three orders of magnitude and I did not catch it during review. There is no algorithmic fix here — no linter that can catch "did you multiply your benchmark out to the full corpus before writing your runtime estimate." Tooling can lower the cost of doing the multiplication (a `tools/eta_calc.py` helper exists now), but it cannot make the human write the multiplication down. The only fix is the discipline of *forcing the multiplication onto the page* before the job runs. That discipline is now mandatory in the SPEC template. I will keep making this mistake in *new* shapes for as long as I write software; what I can do is make it harder to make it in this *specific* shape.

**The second** is that I had spent the previous evening waiting. The job had been launched in the late hours of May 13 and burned its first eight hours on a single period before crashing on the manifest-assembly path bug. Those eight hours were not the kickback. The kickback was that *I had been planning to launch the same job and walk away.* Forty minutes is the wait the path bug cost me. *Forty hours* is the wait that I would have paid into a job whose progress bar nobody was watching, if the path bug had not crashed the run. The path bug compressed the kickback from "two days later you realize the corpus is going to take a quarter of a year" to "forty minutes in you realize the SPEC is wrong." Compression is a property of fast failure. Fast failure is, again, a *gift* — not because the bug was good but because the timing was kind.

**The third** is structural, and is the one that hurts the most to write. Bella was supposed to be the market generator that unlocked richer training data. The precompute strategy was the bottleneck. After the calculator told me the corpus would take 93 GPU-days, the pivot was to **abandon precompute as a strategy** and **generate Bella forecasts on the fly during training** — slower per-epoch throughput, no 93-day budget, no 68 GB of shards on disk, no manifest. The on-the-fly pivot worked, in the sense that it made TASK_740 trainable on a wall-clock that fit inside a week. It did not work, in the sense that *Bella did not earn her compute.* Two weeks later, the TASK_740 evaluation came in at **Δα −1.92 percentage points**, paired three-seed. Bella's distillation made the agent *worse*. Not by noise — by a confident, signed, repeatable margin in the wrong direction. The same campaign also turned up a second, independent Bella bug — the **column-mismatch** between Bella's database-ID-ordered outputs and the trainer's rotation-specific asset order, documented in `project_bella_column_mismatch.md` — which forced a re-precompute under corrected semantics. After all of that, the Bella-distillation mechanism was closed.

The 93-day estimate, in retrospect, was the universe's first hint that the mechanism wasn't going to earn its keep. We treated it as an engineering problem about precompute design and we routed around it. The thing we routed around was not just the precompute; it was the implicit question *"if this mechanism requires 93 days of compute to evaluate, are we sure it's worth that compute?"* The honest answer, available only in hindsight, would have been "no." But the honest answer in *foresight* should have been "if the cheapest version of this evaluation requires three orders of magnitude more compute than the SPEC author thought, we don't understand the mechanism well enough to evaluate it yet."

I am still working out how to make *that* discipline structural. The arithmetic-on-the-napkin rule is easy: type the multiplication. The category-error rule — "an estimate off by 3 OOMs means you don't understand the work, not that you arithmetic'd wrong" — is harder, because it is asking the SPEC author to notice the *shape* of their misestimate, not its magnitude. I don't have a template field for that yet. I am open to suggestions.

---

> *The Apothecary stops, mid-titration, with the bottle in one hand and the chart in the other. The chart reads days. The bottle reads minutes. Both inscriptions are in the Apothecary's own hand. The Apothecary does not curse. The Apothecary does not amend the bottle. The Apothecary writes, in the margin of the chart, the multiplication that should have been there from the first inscription, and then — and only then — measures the dose again.*

---

## What's left, what I learned

Three things survive from May 14.

**The SPEC template.** Every experimental SPEC under `experiments/<id>/SPEC.md` since May 14 has a `## Scale arithmetic` section. It is required. It contains the multiplication, the per-unit benchmark, and the ETA division. It is short, it is monospace, it lives at the top. I have not yet had to override a SPEC at review time because of that section, which is the point — the section *enforces itself* by making the multiplication visible to the writer before it's visible to a reviewer.

**The on-the-fly Bella pipeline.** The precompute strategy as a category is shelved. For any future Bella-coupled experiment, the design moves to on-the-fly generation in the training loop, even at the cost of per-step throughput. The shelf is not "no precompute ever" — for cheap, deterministic feature extractors a precompute is still right. The shelf is "precompute only when the multiplication checks out, and the multiplication has to check out *on the page.*"

**The category "diagnostic crash."** I am introducing this term into my own engineering vocabulary because I do not have a better one. A diagnostic crash is a failure whose effect is to surface a *different, larger* failure that would otherwise have proceeded silently. The May 14 path bug was a diagnostic crash; the absent `.resolve()` was sloppy on its own merits, but its timing was the entire reason I noticed the scale failure forty minutes in rather than forty hours in. I'd like to keep a few more of these around. The honest version of the engineering principle is: *crash loudly at unit boundaries, even — especially — when the unit completed correctly.* If the first period had crashed silently at the boundary because of some assertion about manifest sanity (rather than crashing for an unrelated reason about path resolution), the same diagnostic property would have held.

And the deeper structural lesson, the one I think I will be working on for years: **when an estimate is off by three orders of magnitude, the error is not arithmetic.** It is a category error about what the job is. The SPEC writer — me — thought the precompute job was a *small precompute over a benchmark-confirmed fast operation*. It was, in fact, a *moderate-scale Bella job multiplied by a four-figure period count and a three-figure window count and a one-figure sample count, against a benchmark whose throughput-per-asset was almost a thousand times slower than the per-asset benchmark had measured*. I had two numbers in my head — 270,000 and 278,000,000 — that I never put next to each other. The category-error rule that follows from this is something like: *if your estimate could be wrong by 3 OOMs in either direction, you do not yet understand the work well enough to estimate it. The next move is not a better estimate; the next move is a smaller, instrumented dry-run that produces a single throughput number you can multiply by hand.* That is the rule I am still drafting in the margins of the SPEC template. I do not have a clean form for it yet. The current crude form is *"if you cannot back-of-the-envelope the runtime to within 2× in five lines of arithmetic, the SPEC is not ready."* That is approximately the right shape. It is not yet the right wording.

---

It is the afternoon of May 21, 2026. `workstation` is on. The desk fan is on. A different progress bar is moving in the left pane — a paired three-seed evaluation under the recalibrated eval-MCTS operator that is, as of yesterday, the canonical scorecard for this project. In the right pane, `vim` is open on a new SPEC under `experiments/`. The first section of that SPEC is `## Hypothesis`, the second is `## Design`, and the third is `## Scale arithmetic`. The third section is filled in. The multiplication is on the page. The ETA divides out to 4.2 hours. I have checked it twice with the calculator app, because the discipline I am drafting is not yet old enough to be muscle memory and I do not want to find out, in some other afternoon, that I have written *4.2 hours* on the bottle and *4.2 days* on the chart.

> *Bella's vision was beautiful. Her dosing chart, this time, was not.*

---

## Sources

- `~/Trader/lore/notes_2026-05-13.md` §100 — the primary diary entry. The scale-projection arithmetic and the crash traceback quoted in this article come from this section verbatim.
- `~/Trader/tasks/TASK_740_BELLA_DISTILL_HEAD.md` — the TASK_740 SPEC that called for the precompute. Pre-computed shard sizing ("~500 MB per period × 17 slots ≈ 8 GB") was its original storage estimate; the actual run produced 250 MB shards and the corpus estimate revised to ~68 GB.
- `~/Trader/tasks/TASK_744_BELLA_PRECOMPUTE_FIX.md` — the path-fix and scale-rethink task opened on May 14 in response to the crash.
- `~/Trader/tools/precompute_bella_predictions.py` — the precompute job. Line 244 is the `Path(args.output_dir)` site; line 210 is the `out_path.relative_to(REPO_ROOT)` site; line 52 is the `REPO_ROOT = Path(__file__).resolve().parent.parent` definition. The one-line fix is `out_dir = Path(args.output_dir).resolve()`.
- `~/Trader/config/foxglove_periods.json` — the 270-period corpus index.
- `~/Trader/diff_teacher/` — Bella's source tree. See `diff_teacher/CLAUDE.md` for the architecture summary, `diff_teacher/teacher.py` for the two-phase trainer, `diff_teacher/generation/generator.py` for `MarketGenerator.from_checkpoint`.
- `~/Trader/docs/UNIFIED_MARKET_GENERATION_ARCHITECTURE.md` and `~/Trader/docs/market_diffusion_architecture.md` — the Bella research-context documents.
- The TASK_740 result (Δα −1.92pp paired three-seed) and the Bella column-mismatch finding referenced in the third kickback are documented in `project_bella_column_mismatch.md` and the wider Bella-class closure note (`project_encoder_bella_class_closed.md`).
