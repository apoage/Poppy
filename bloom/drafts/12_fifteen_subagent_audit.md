> *Observation: I lay each cutting on its own square of paper and examine them in turn. I forget the previous specimen before I lift the next. The eye is fresher when the last leaf is no longer in mind. The garden is too large to be held in one head, and the head, in any case, has been in the garden too long.*

---

**by Lukas — May 22, 2026**

# The 15-Subagent Audit and What It Caught

*On the methodology of forced-fresh-context parallel code review, what it found in the Poppy training pipeline at the end of January 2026, and why daily collaboration trades cold-eyes acuity for working-memory continuity in a way you have to compensate for on purpose.*

---

## The conviction, late January 2026

It is the morning of May 22, 2026 as I write this. `workstation` is on; the desk fan is on; a value-head bin check is running in the next window over a checkpoint from the campaign that the next article is about. The audit I am going to describe ran four months ago, in the last week of January, and it is the only audit on this project whose methodology I have not since modified.

The setup. Late January 2026. We have just come out of the BUG_MAYHEM run that Article 5 covers — twenty-two confirmed issues in the training loop, of which the most embarrassing was alpha-relative scale mismatch (100× inflation/deflation of the reward signal, surviving for weeks) and the most load-bearing was the BC confidence masking that diluted gradients by including zero-confidence samples in the denominator. Twenty-two bugs is a number that should feel cathartic. The catharsis it gave me was the wrong catharsis: it gave me the feeling that we were *finished*, when what it should have given me was the feeling that we had only seen the bugs we knew how to look for.

The conviction that arrived a few days after closing BUG_MAYHEM was this: *one more pass*. Not the same pass. A different one. The bugs we had caught in BUG_MAYHEM were the bugs that a human plus a daily-collaborator agent, both saturated in the codebase, could see. Which meant — by the law of selection — the bugs we *hadn't* caught were precisely the ones that didn't show up to that pair of eyes. I had been working on `trainer.py` for two months. Claude Code, whichever instance, had been working on it daily for the same two months. Anything either of us was going to spot, we had spotted.

And then there was a second, harder conviction, the one that took the rest of the week to land. I cannot audit my own code with my own eye. Not because my eye is bad — my eye is fine, on a *different* codebase. On *this* codebase, the parts I have written and re-written and copy-pasted and decided not to refactor have become furniture. I read them and the eye lands somewhere else, the way you stop seeing the lamp in your own living room. The fix is not to look harder. The fix is to give the file to someone who has never seen it.

I did not have a someone. What I had was sub-agent quota.

## The methodology

What I built — and what TASK_127 is, in retrospect, the artifact of — was a deliberate procedure to *force every audit through a fresh-context boundary*.

I broke the training pipeline into components. Not files — components. The list was something like:

- MuZero dynamics (representation + dynamics + prediction).
- Replay buffer + sampling (the unified index buffer, the priority trees, the demo/agent zone split).
- PopArt normalization.
- Entropy and policy-collapse mechanisms.
- MCTS search + integration with self-play.
- Optimizer, learning-rate schedule, gradient clipping.
- Target network synchronization.
- Loss function weighting.

Eight categories, on a first cut. The thing I then did — and this is the methodology, not the bug list — is that I dispatched *two and sometimes three independent subagents at each category*. They got launched in parallel, in separate threads, each with its own cold-context startup. None of them saw any other subagent's brief, prompt, working notes, or final report. None of them saw mine.

The brief each one got was nearly identical and ran roughly like: *audit this module. Here are the file paths. Here is the contract the module is supposed to honor: it takes these inputs, it produces these outputs, it interacts with these other modules in these named ways. Read the implementation cold. Report any deviation from the contract, any internal inconsistency, any silent failure mode. Cite line numbers. Severity-rate your findings. Do not assume anything about other components.* That was it. No "we just finished BUG_MAYHEM and here is what was found." No "here are the known issues to avoid double-reporting." Cold start. Fresh eye.

The reason for the parallelism — multiple subagents per component — was that I wanted to read the *disagreement*. If two cold-context agents, looking at the same file, came back with the same finding, that was a strong signal. If they came back with non-overlapping findings, both were probably interesting. If they came back with *contradictory* findings, that was the most useful case of all, because it meant the contract was genuinely ambiguous and the ambiguity was a candidate for being its own bug.

The discipline I imposed on myself was this: *I would not synthesize the findings until every subagent had returned.* No early peeks. No "let me check what subagent k1 said before I write the brief for subagent k3." The whole point was to deny the system the cross-talk that would have re-introduced the daily-collaborator blindness through the back door. Each report was generated as if it were the only audit happening.

The result, after about a day and a half of agent compute, was the file directory that this article is sourced from — fifteen-plus `TASK_127*` files, sitting side by side, each one written as if it were the master document. The k-series files (`TASK_127k1` through `TASK_127k8`) were the second-wave audits I commissioned after a first-wave look (`TASK_127a` through `TASK_127q`) showed that some of the components needed a third pair of eyes for the cases where the first two had disagreed. The synthesis — the file that became `TASK_127_master_BUG_MAYHEM_MASTER_REPORT.md`, the master bug catalog — was written *afterward*, on a separate session, with all the reports laid out as input.

That last sentence is the methodology in one line. *Cold reads in parallel; synthesis afterward, separately, with all reports in view.* The parts that get fresh eyes get fresh eyes; the part that requires cross-component integration gets dedicated synthesis time. Don't mix the two.

## What it caught

Forty bugs cataloged at first pass. After validation: **twenty-nine confirmed, nine false positives, two left unvalidated**. Twenty-nine real bugs, in a pipeline I and a daily-collaborator agent had already audited together and declared closed less than a week earlier. I want to walk through six or seven of these in detail — not to relitigate the bugs themselves (Article 5 already did the BUG_MAYHEM accounting), but to show *which kind* of bug each subagent caught, and why a fresh-context audit caught what the daily-collaborator one had not.

### The target network that did nothing

This is the one I think about most. TASK_127k7, the target-network subagent, opened its executive summary with a sentence I have never been able to read without wincing:

> *"CRITICAL FINDING: The target network is defined, initialized, and updated periodically, but it is **NEVER ACTUALLY USED** in any computation. The `target_params` field exists in `TrainState` and is copied from `params` every 100 steps, but no code references `target_params` for value estimation, TD targets, or MCTS."*
> — `tasks/TASK_127k7_Target_Network_Audit.md`, Executive Summary

`target_params` was declared on `TrainState`. It was initialized at training start to match `params`. Every hundred steps, a JAX `lax.cond` performed a hard update — copy the current params into the target slot. The infrastructure was *immaculate*. The hard-update interval was named, the conditional was JIT-friendly, the field was carried through the train state replace correctly. The bug was not in any of this code. The bug was that no other line of code in the entire repository read from `target_params` once it had been written. The TD targets came from `params`. The MCTS rollouts came from `params`. The value baseline came from `params`. The target network was a museum piece. Memory and a periodic device-to-device copy, in service of nothing.

The reason this had survived BUG_MAYHEM is the reason it had survived for weeks before that: *every part of this network individually looked fine*. The declaration looked fine. The initializer looked fine. The update step looked fine. If you were daily-collaborating with the file and your model of the system included "we use a target network for Double-Q stability," your eye walked across `target_params` and ticked the box. The thing your eye did *not* do — would not do, having ticked the box — is grep the rest of the repository to confirm that anywhere downstream actually *consumed* the value the box was checking. A fresh subagent, without the model, did exactly that grep, because it didn't have a model that told it not to.

The fix, written in the audit, was a two-option choice: either wire `target_params` into the TD-target computation as intended, or rip it out. We did the wiring. But the structural lesson — that *named infrastructure existing is not evidence that the infrastructure is used* — is what changed how I read code from that month on.

### The MCTS visit counts that were never normalized

This one had been sitting in plain sight in `self_play.py:219` for the entire history of the file. TASK_127k5 (and TASK_127j, independently) caught it:

> *"MCTS returns raw visit counts, passed directly to `softmax_cross_entropy` which expects probability distributions. […] Gradient magnitudes scale with visit count magnitudes, not probability errors."*
> — `tasks/TASK_127_training_audit_critical_issues.md`, Issue 4

The MCTS search produced visit counts — integers per action, summing to the total number of rollouts. The downstream code took those counts, packed them into a numpy array, and handed them to `optax.softmax_cross_entropy(logits, target_policies)` as the target distribution. Cross-entropy expects probabilities. It expects vectors summing to one. The code was passing raw counts: vectors summing to 32, or 64, or whatever the simulation count was that day. The gradient signal the policy head was receiving was not "match this probability distribution" — it was "match this distribution scaled by 32 or 64 in arbitrary spots." The shape of the policy target was right; the scale was 32× too big.

This had been there *since the MCTS integration landed*. Daily-collaborator review walked past it because the line `policy_probs.append(np.array(policy_jax[0]))` reads, *to a context-saturated eye*, like a load — you see "policy probabilities being collected" and the eye moves on. A fresh subagent, asked to audit `self_play.py:219` from cold context, read it as "raw output of MCTS being shoved into a function that wants a normalized distribution" and flagged it inside the first pass.

### The PER weights that were computed but never applied

TASK_127c and TASK_127d, both auditing the replay buffer, both caught the same thing. The Prioritized Experience Replay machinery was in place. The priority trees were built. The sampling drew indices with proportional-to-priority probability. The importance-sampling weights were computed correctly to undo the sampling bias — the formula in `expert_buffer.py:358-381` is textbook, and both subagents confirmed it. Then the trainer averaged the loss with `jnp.mean()`. The weights, having been computed, were never multiplied into the loss term.

> *"Importance sampling weights are computed correctly but **never applied to loss**. All loss computations use uniform `jnp.mean()`. […] PER is pure overhead with zero benefit. Training is biased toward high-priority samples without correction."*
> — `tasks/TASK_127_training_audit_critical_issues.md`, Issue 2

This is the same shape of bug as the target network — *infrastructure that exists in good faith, used by no one*. The replay buffer thought it was doing PER. The trainer thought it was getting weighted samples. Neither side was wrong, individually. The contract at the seam between them — *who multiplies the weights into the loss?* — had nobody on either side of it. The buffer handed back a `weights` array. The trainer fetched the batch, used the observations, used the actions, used the targets, and ignored the field. The bug lives in the *space between two modules*, which is exactly the place that a daily-collaborator review almost never lands the eye on, because the daily-collaborator review is held *inside* one module at a time. Cold-context audits that arrive with the explicit instruction *here is the contract you should expect at the seam* land there reliably.

### The PopArt that fought the value head it was attached to

PopArt's role is to normalize value targets — running statistics, then a periodic *weight correction* on the final layer of the value head that preserves the network's outputs across the normalization update. The math is well-defined for **scalar** value heads. The trick: when (μ, σ) updates, you rescale W and b on the output layer so that `v_new(h) ≈ v_old(h)` for all hidden states.

Poppy does not have a scalar value head. Poppy has a *categorical* value head — logits over 401 bins spanning a fixed support `[-10, +10]`, with the scalar value recovered by `Σ softmax(logits) × support`. TASK_127k3 (the PopArt subagent) walked through what happens when you apply the standard PopArt weight correction to a head that produces categorical logits:

> *"PopArt weight correction assumes scalar outputs, but value head outputs categorical logits. Scaling logits doesn't preserve expected value."*
> — `tasks/TASK_127_master_BUG_MAYHEM_MASTER_REPORT.md`, BUG-004

You can multiply the categorical logits by a scalar all day. The softmax is invariant to additive shifts and equivariant to *temperature*, not to scale. Multiplying logits by `σ_old/σ_new` does not produce a new categorical distribution whose expected value tracks the old one. PopArt was *fighting* the value head. Every weight-correction step was perturbing the policy of the categorical predictor in a direction that had no algebraic relationship to the running-statistics update it was trying to compensate for. This had been live in training for months.

There was a second, *related* finding from the same subagent that made the PopArt situation worse — the weight correction was happening in Python, in a periodic loop, every 10 iterations, **outside the JIT-compiled train step**. The gradients that JAX was computing inside JIT were against the *uncorrected* params; the correction itself happened outside, on a separate cadence; and the corrected params then went back into the JIT cache for the next step. The two operations — gradient descent and weight correction — were *desynchronized*. Neither informed the other.

The reason this had survived was structural: PopArt is a *thing people use* in deep RL. The presence of PopArt in the codebase was, to a daily-collaborator agent and to me, an asset rather than a question. The mental model said *PopArt is here, PopArt works*. A fresh-context subagent did not have that mental model. It read the categorical head, it read the PopArt weight correction, and it noticed *immediately* that the algebra didn't close. It also noticed that the JIT/Python boundary was being crossed in a way that broke gradient flow. Both findings landed in the same audit.

### The phantom rewards summing past termination

TASK_127's master report carries BUG-034:

> *"Phantom Rewards (Summing Past Termination). Issue: Summing rewards past episode termination (penalizing dead agents). Location: `self_play.py`. Fix: Apply mask before summing episode score."*
> — `tasks/TASK_127_master_BUG_MAYHEM_MASTER_REPORT.md`, BUG-034

When an episode ends — a forced liquidation, a hit on a stop-out — the environment stops producing meaningful state, but the episode-aggregation code was summing reward fields all the way to the end of the buffer, picking up zeros or stale fills as if they were trades. The agent was being credited or penalized for steps that did not exist. The cumulative-reward signal it learned against was contaminated.

I caught this in BUG_MAYHEM, originally, and it shows up there in the report — but the *reason* I am listing it here is that the TASK_127 audit caught it *again*, independently, in `self_play.py`, from a cold read. That is the methodology working as a backstop: the subagent did not know we had already filed this one. It re-derived it. The duplication is not waste — it is the consistency check that says *yes, this is genuinely there*, because two independent cold reads converged on it.

### Reward-loss division by zero when `unroll_steps=1`

A small one. BUG-039 in the catalog. From `trainer.py:1169`:

```python
reward_loss /= (unroll_steps - 1)
```

If you ever ran the trainer with `unroll_steps=1` — which the config schema permitted — this divided by zero. We had never run with `unroll_steps=1` in production. Nobody had hit the crash. The line had been there since the unroll loop was written. The audit found it on a static read of the trainer, noted that the value support permitted the unsafe setting, and recommended guarding the division.

The reason I'm flagging this even though it never fired: *the audit found bugs that were not yet bugs*. A daily-collaborator agent, reading the file, would observe that the code worked for the cases it had run. A fresh-context audit, asked to find every internal inconsistency or unsafe operation, did not have that constraint. It reported what was unsafe regardless of whether it had ever fired. That is the kind of finding that pays for itself a year later, the first time someone tries a new config and the trainer crashes on a line that was *known to be wrong since January*.

### The dual sampling path in the expert buffer

The expert buffer had two zones: a protected demo zone of 50k samples, and a circular agent zone of 500k samples. Both zones were *advertised* as using Prioritized Experience Replay via the segment-tree machinery. TASK_127k2 caught that they didn't:

> *"The expert buffer uses a **different sampling method** for demo vs agent zones: Agent zone uses proper PER via `_sample_proportional` (sum tree traversal). Demo zone uses `_sample_from_range` which does **NOT use the priority trees!**"*
> — `tasks/TASK_127k2_Replay_Buffer_Audit.md`, Critical Findings

The demo zone was actually using `np.random.choice` over a slice of the index array, with priority weights computed *on the fly* from the slice — not from the priority trees that the agent zone used. Same buffer object, two sampling implementations, one of them silently bypassing the data structure the buffer was named after. Worse, the on-the-fly path had a `replace=True` fallback when the sample size exceeded the range — sampling demos *with replacement* in a setting where the design called for without. So the highest-confidence demos were being repeated in batches in a way that the agent-zone sampler would never have produced.

This is a *contract*-shaped bug, like the PER weights one. The interface to the buffer was unified (`sample_batch(batch_size)`). The implementations behind the interface diverged. A daily-collaborator audit that looked at the buffer "from outside" — that traced what flowed in and what flowed out — would not catch this. A subagent asked to *walk every code path inside the buffer* would, and did.

## The meta-finding

I want to be specific about what I think this audit demonstrated, because the lesson is narrow and I have seen it overgeneralized.

The lesson is *not* that fresh agents are smarter. They are not. The subagents that ran TASK_127 are the same model family as the daily-collaborator agent that had been working on this codebase, and they are not, as far as I can tell, more capable in any architectural sense. Several of them missed things other subagents caught. Several of them generated false positives — the "double alpha exponentiation" finding, the "PopArt update cadence is itself a bug" finding, the "burn state not reflected in agent state" finding — all confirmed false on validation. The fresh-context subagents were not *better*. They were *differently constrained*.

The lesson is this: **a person (or an agent) who has worked on a codebase for N weeks develops a model of it, and the model has fixed points the person stops checking.** Once you believe a target network is wired in, your eye does not grep for whether it is being read. Once you believe PER is functioning, your eye does not check the loss term for a `* weights` factor. Once you believe MCTS visit counts are normalized somewhere upstream, your eye does not normalize them at the seam. The fixed points are not blind spots in the visual sense — they are *places the model has already answered the question, so the question stops being asked*. This is not a failure of the model. This is what models are for. A daily collaborator without these fixed points would be useless — they would re-derive the entire pipeline every morning instead of moving work forward.

A fresh-context audit does not have the model. It also does not have the working memory continuity that lets the daily collaborator land tactical changes. *You need both.* What TASK_127 demonstrated was that the cadence of the two is different. Working memory continuity is what you want on a daily basis, on every change. Cold-eyes auditing is what you want *periodically*, against the entire pipeline, at junctures where you suspect — but cannot articulate why you suspect — that there are bugs the daily eye has stopped seeing.

Fresh context is a research tool. It is not a smarter eye. It is *a different eye*, deliberately blind to your priors. You schedule it.

> *I lift each cutting and examine the leaf, the stem, the cut at the base. I have forgotten the previous one. I do not know yet whether they are siblings or strangers. I write what I see and put the paper aside.*

## The kickback, in three layers

The first kickback was quantitative and easy to feel. Fifteen-plus subagents, run in parallel, consumed *meaningful* token budget — I do not have a clean accounting of the exact spend, but it was the largest single batched-audit invoice I had ever sent. Two days of throughput on something that produced no new code, no new experiment, no new model. The conversation I had with myself before committing to it was not comfortable.

The numerator turned out to be twenty-nine confirmed bugs. That is somewhere on the order of *one confirmed bug per agent-hour of compute*. I had never gotten that yield, by any margin, from any human-only audit cycle on this project — not from me, not from the daily-collaborator pair-programming sessions, not from BUG_MAYHEM (which had been more concentrated, more dramatic, but had taken longer in human-clock-time and surfaced fewer findings per dollar of token cost). The cold-eyes parallel audit was *the most efficient bug-finding mechanism the project had used*. I had a hard time admitting that, because the human side of me wanted to believe that careful sequential thought outperformed brute parallel coverage. It did not. Not on this kind of work.

The second kickback was procedural. Every subagent in TASK_127 was operating on the assumption that it was the only audit happening. This is deliberate — it is the whole *point* of forced-fresh-context — but it also means that what comes back is unsorted. Two audits of the replay buffer with sixty percent overlap in findings. An audit of MCTS that confirms a finding from the audit of self-play. An audit of the optimizer that gives a different learning-rate default than the audit of the loss-weighting module, neither of which agrees with the value in the config file. The *consolidation* of all this is its own cognitive work, and it cannot be done by another subagent — it has to be done by someone (me) who is willing to hold all the reports in view simultaneously, dedupe them, validate them against the actual code, and decide which findings are the same finding under two names and which are genuinely separate. That synthesis cost me most of a day. It was less work than running the audits myself would have been by an order of magnitude. It was still work.

The third kickback is the one this article is really about, and it is the one that I do not have a clean way to make peace with.

Most of the audit findings — including the load-bearing ones, the target-network-never-used, the PER-weights-never-applied, the PopArt-incompatible-with-categorical, the MCTS-visit-counts-unnormalized — were issues the prior daily-collaborator agent had, in some functional sense, already *implicitly dismissed*. Not because the collaborator was wrong. Not because the collaborator was lazy. The findings were ones that the collaborator had walked past every day, in the course of doing actual work, because doing the actual work required *not auditing the file every time you opened it*. The daily collaborator's job was to move the project forward. Moving the project forward meant operating inside the model. Operating inside the model meant the model's fixed points were invisible. And every time I sat down with the same collaborator and walked the file *together*, my own model overlapped with the collaborator's enough that the joint pair had the same blind spots, layered.

There is a cost to this, and it is the cost the article is admitting. *Daily collaboration trades cold-eyes acuity for working-memory continuity*, and it does this trade silently — you do not feel the bugs you are not seeing. The way you find out you were paying the cost is when you run a fresh-context audit and twenty-nine bugs fall out. You do not get to keep both the continuity and the acuity *on the same eye*. You have to alternate cadences, and you have to schedule the alternation, because the eye that is good at one is bad at the other and will not naturally switch.

## What's left

The cold-eyes audit is on the schedule now. Every major refactor of the training pipeline since TASK_127 has triggered a smaller version — three to six subagents, one per affected component, same brief, same parallelism, same forced-context-boundary. None of them have found twenty-nine bugs. Some of them have found three, some of them have found six. The number of findings goes down over time, which is what you would want to be true if the methodology is working — early audits surface a backlog, later audits surface only new accretions. The diminishing returns are themselves the signal that the procedure is doing its job.

Some of the specific bugs found in TASK_127 were fixed once and stayed fixed — the target-network wiring, the MCTS visit-count normalization, the PER weight application, the unroll-steps division guard. Others recurred. The PopArt-categorical incompatibility was disabled rather than fixed (we never built the scalar value head it would have required); the entropy coefficient was bumped, then reverted, then bumped again. A handful of the findings got unit tests written for them — the kind of low-effort regression test that asserts `target_params` is referenced by at least one TD-target computation, or that the loss function multiplies in the IS weights, or that visit counts sum to one before they reach `softmax_cross_entropy`. These tests are not glamorous. They are also the only thing that has prevented the bugs from coming back during refactors. I write more of them every month and apologize for none of them.

The methodology informed Article 6's classifier-Alpha discovery via the same shape. When the encoder ablations of Phase 2 started showing a pattern that the daily-collaborator pair could see but could not name — *the classifier metrics improved, the Alpha collapsed* — what broke the impasse was again a cold-context request, this time asking *what is the relationship between the dir-acc ceiling and the trade rate, audited from the outside?* Not a daily-collaborator question. A what-am-I-actually-looking-at question. The framing came directly from TASK_127's discipline of *audit the seam, not the modules in isolation*. That article will go into the structural answer; the methodology by which we *arrived* at the question is this one.

I want to close with the limit. The 15-subagent audit is not a substitute for working in the codebase. The reports it generates are noisy — nine of forty findings were false positives — and the false positives are not always obvious from the outside. You still need to validate every finding by reading the actual code, in your own head, with the daily-collaborator eye that knows the surrounding context. The audit is a *signal generator*, not a *truth source*. What it gives you is a list of candidates that someone uncontaminated by your priors thinks are worth checking. The checking is still yours.

But the candidates are real, and the candidates are not ones you were going to surface yourself. That is the asset. Fresh eyes are not smarter. They are simply not yet *your* eyes, and the only way to keep them not-yet-yours is to refuse to let them see what your eyes have already looked at. Force the boundary. Run them in parallel. Synthesize afterward, in a session of their own. Validate every finding by hand.

It is, in the end, the same operating principle as Article 4's encoder discovery — that a fresh agent, asked a slightly different question, will notice the room's furniture as if it were a fact about the room rather than part of it. TASK_127 was that principle, industrialized. Fifteen subagents, fifteen fresh rooms, fifteen reports written as if their author had never been in the apartment before. The bugs that fell out had been part of the furniture for months.

## Closing

It is the morning of May 22, 2026. The cold-eyes audit cadence is still in force. The PER weights are still being multiplied into the loss. The target network is still being read from by the TD-target computation, the way the audit said it should be in late January. The unit test that asserts `target_params` is referenced by something downstream lives in `tests/` and has not been deleted, and at some point a future refactor will trigger it and someone — me, an agent, a stranger — will know within thirty seconds that the wiring has come loose again. That assertion is the closest thing this project has to an inheritance.

> *The cuttings are different specimens. I had thought, with one of them in hand at a time, that they might be siblings. With all of them on the bench at once I see they are not. I file them by family and continue.*

## Sources

- `tasks/TASK_127_master_BUG_MAYHEM_MASTER_REPORT.md` — synthesis catalog, 40 cataloged bugs, 29 confirmed, validated 2026-01-31. The master file that the parallel subagent reports were folded into.
- `tasks/TASK_127_training_audit_critical_issues.md` — first-wave executive summary, twelve critical issues across BC masking, PER, PopArt, MCTS visit counts, consistency loss.
- `tasks/TASK_127a_muzero_audit.md`, `tasks/TASK_127b_MUZERO_DYNAMICS_AUDIT.md`, `tasks/TASK_127k1_MuZero_Dynamics_Audit.md` — MuZero dynamics subagents. Source of the missing consistency loss finding (BUG-006) and the no-gradient-blocking finding (BUG-017).
- `tasks/TASK_127c_replay_buffer_audit.md`, `tasks/TASK_127d_REPLAY_BUFFER_SAMPLING_AUDIT.md`, `tasks/TASK_127k2_Replay_Buffer_Audit.md` — replay buffer subagents. Source of the PER-weights-never-applied finding (BUG-002), the demo/agent priority scale mismatch (BUG-007), and the dual-sampling-path finding (BUG-009). Quote in §"The dual sampling path in the expert buffer".
- `tasks/TASK_127e_POPART_NORMALIZATION_AUDIT.md`, `tasks/TASK_127f_popart_audit.md`, `tasks/TASK_127k3_PopArt_Audit.md` — PopArt subagents. Source of BUG-003 (weight correction outside JIT) and BUG-004 (categorical-head incompatibility). Quote in §"The PopArt that fought the value head it was attached to".
- `tasks/TASK_127g_ENTROPY_POLICY_COLLAPSE_AUDIT.md`, `tasks/TASK_127h_entropy_audit.md`, `tasks/TASK_127k4_Entropy_Policy_Collapse_Audit.md` — entropy subagents. Source of BUG-019 (adaptive entropy discontinuity) and BUG-020 (entropy coefficient too small).
- `tasks/TASK_127j_mcts_audit.md`, `tasks/TASK_127k5_MCTS_Search_Audit.md`, `tasks/TASK_127k_MCTS_SEARCH_INTEGRATION_AUDIT.md` — MCTS subagents. Source of BUG-005 (raw visit counts to cross-entropy). Quote in §"The MCTS visit counts that were never normalized".
- `tasks/TASK_127k6_Optimizer_LR_Audit.md`, `tasks/TASK_127l_optimizer_audit.md`, `tasks/TASK_127m_LEARNING_RATE_OPTIMIZER_AUDIT.md` — optimizer subagents. Source of BUG-011 (weight decay never passed), BUG-012 (LR default inconsistent between trainer and config), BUG-013 (global clip too permissive), BUG-014 (per-head clip disabled), BUG-021 (LR schedule constant by default).
- `tasks/TASK_127k7_Target_Network_Audit.md`, `tasks/TASK_127n_TARGET_NETWORK_AUDIT.md`, `tasks/TASK_127o_target_network_audit.md` — target-network subagents. Source of BUG-015 (target_params never read). Quote in §"The target network that did nothing".
- `tasks/TASK_127k8_Loss_Function_Audit.md`, `tasks/TASK_127p_LOSS_WEIGHTING_AUDIT.md`, `tasks/TASK_127q_loss_weighting_audit.md` — loss-weighting subagents. Source of the loss-composition analysis and BUG-022 (value loss weight hardcoded in setup.py).
- Article 4 of this series (`04_encoder_64x_redundancy.md`) — the precursor observation that a fresh-context agent reading a file cold can land on furniture the daily-collaborator eye has stopped seeing. TASK_127 is that observation made deliberate and parallelized.
