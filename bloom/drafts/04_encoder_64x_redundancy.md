> *Observation: Foxglove receives water on schedule. The soil is dark and saturated; the surface beads up under the watering can rather than drinking. The leaves yellow at the tips and the new shoots arrive smaller than the last. The dose is correct. The uptake is not.*

---

**by Lukas — May 21, 2026**

# Encoder Output, Repeated 64 Times: The Information-Loss Audit

*On a GPU that was idle most of the time, a 22 GB dataset that should have been 338 MB, a back-and-forth with Gemini that I lost on purpose, and the first time the project asked me whether the runs I had been building intuition from meant anything at all.*

---

## "i mean cli"

It is a Thursday evening in mid-January 2026. `workstation` is on, the desk fan is on, a tail of `train_jax_fast.py` is scrolling in the left pane and `watch -n1 nvidia-smi` is running in the right. The training loop is, by every visible signal, fine. No assertions tripping. No NaN. No silent stall. The loss curve in DS24 — the training dashboard on localhost:8050 that reads from the same checkpoint directory the trainer writes to — is going down, slowly, the way you want it to. Behavioral cloning from the oracle, epoch after epoch, no fireworks.

The thing that is not fine is the right pane. GPU utilization is jumping between 5 % and 17 %. Not for a second between kernel launches — sustained. The card is *idle most of the time*. I had bought this card and tuned the PSU and wired the case to feed it 420 W under load and the card was barely breaking a sweat. The first time I had noticed it I assumed I was misreading the column. By the third epoch I had pulled up `nvidia-smi dmon` and confirmed: SM utilization in the single digits, memory bandwidth low, the kernel was getting starved between calls. Something upstream of the GPU was choking the pipeline.

The Lukas-side prompt I have for the start of all this — the SSD snapshot of the Claude Code conversations from this period preserves it — opens a Jan 13 session with:

> *"are you up to date .. i mean cli"*
> — Lukas, 2026-01-13 22:27, snapshot session `55be4183-bf68-4874-8c07-095eee979767.jsonl`

That is the texture of the work. One-handed prompts at 22:27 on a weeknight, asking the agent if its tooling is current before launching into "the GPU is bored, find out why." There is no formality. There is no preamble. There is a tired person and a fresh agent and a problem that has been sitting on the project for weeks and a feeling — the specific feeling, the one this entire article is about — that *training works but training is wrong*.

The piece of the project that this article concerns was, in hindsight, the first real reckoning. Up to January 2026 every problem in the codebase had been the kind a forward pass crashes on: a shape mismatch, a NaN, a dataloader index off by one, an oracle that returned the wrong action for a corner case. They were *loud*. You ran the trainer, it died, you fixed it, you ran it again. The January audit was about a different category of bug — the kind where nothing dies. The kind where you are training, and the loss is going down, and the dashboard is green, and you are quietly burning four-fifths of your compute on water that is never going to reach a root.

## Behavioral cloning, the oracle, and what the encoder was supposed to do

To make the rest of this legible I need to lay out the pipeline as it stood in January.

The trainer at that point was `train_jax_fast.py`. The plan was the one we had committed to in `ALPHAZERO_TRADING_VISION.md` back in December: a MuZero-style agent with a policy head and a value head, trained against self-play, with an auxiliary **behavioral-cloning** (BC) loss against an *oracle* — a Bellman-style path solver that, given a window of past prices, computes the trajectory of optimal trades over that window in closed form. The BC term gave the policy a target to imitate before self-play had stabilized. The phrase we used internally was *imprint first, plan later*. It is the same idea you find in DQfD and AWAC — pre-train the policy on expert demonstrations so the early self-play episodes start from somewhere useful rather than from random.

The data path looked, roughly, like this. A self-play worker would run an episode (\~400 steps on a 64-asset universe), recording prices and actions. The oracle would label each step in the episode with its optimal action. Those labeled steps got pushed into an **expert buffer** — an indexed structure that the trainer sampled batches from on every iteration to compute the BC loss. To build the *input* for the policy and value heads from a sample, the trainer had to take the price window for that step and run it through the **encoder** — the function that turns 64 assets × 64 lookback bars of OHLCV into the 27-channel feature tensor the network actually consumed.

The encoder lives in `mcts_trader/env/jax_encoder.py`. The relevant function was `_compute_all_channels`. The output shape was `(27, 64, 64)` — 27 channels, 64 assets, 64 something. Even at the time I had a low-grade unease about that last dimension. *What was it?* In some channels it was lookback bars. In others it was a tile of cross-sectional features broadcast across assets. The channels had been added one at a time over weeks. The geometry was no longer something I held in my head; I held the *names* of the channels and trusted the broadcasting.

A BC training step, in pseudo-flow:

```
1. expert_buffer.sample_batch(batch_size=1024)
2.   for each unique episode_id in the batch (typically 10–20):
3.     fetch episode price array from RAM dict
4.     jnp.array(prices)                # host → device
5.     vmap(encoder_fn)(window_starts)  # GPU work
6.     np.array(encoded)                # device → host (blocking)
7. trainer.train_step(batch)
```

I read that loop a dozen times in late December and saw nothing wrong with it. The encoder is JIT-compiled. The vmap is doing the right thing. The work is on the GPU. Why is GPU util sitting at 11 %?

The hypothesis I went into the audit with — the wrong one — was *Python overhead*. The Python loop over episodes was launching tens of small kernels per batch instead of one big one, and the cost was launch latency plus host-device hops. If I batched it into a single call, the gap would close. That was TASK_136b. I wrote the spec for it, handed it to Gemini, and went to bed.

## The collab

This is the section I want to do carefully, because the texture of it is the artifact. There is a thing that happens in the back-and-forth with a fresh agent — Gemini, in this case — where you bring it your bottleneck framed as a question and it brings back, attached to its answer to your question, a *second* answer to a question you didn't ask. The discipline is to read the second answer.

I'll start with my framing. TASK_136 was actually four sub-investigations, run in parallel against Gemini, each on a different candidate fix. 136a was *pre-encode everything once at episode-add time and cache the result*. 136b was *single batched JAX call instead of a Python loop*. 136c was *keep episode prices in VRAM instead of RAM*. 136d was *just run BC less often per epoch*. My priors going in: 136b would be the win, 136c was a marginal optimization, 136d was a hack I'd accept if the others failed, 136a was theoretically nice but I assumed the memory cost would kill it. The encoder output was `(27, 64, 64)` per sample. At 50 000 demos in the buffer:

```
27 × 64 × 64 × 4 bytes × 50 000 = 22.1 GB
```

That was the number I had, sitting at the top of my notes. 22 GB to cache the encoded observations in RAM was a non-starter. The workstation has 64 GB but I was already using \~30 GB for the replay buffer and the dataloader staging area. So 136a was, on my read, dead on arrival.

Gemini's analysis of 136b came back the way I expected: yes, batch the call, here is the gather pattern, here is the speedup estimate. Standard. Useful. The interesting paragraph was in 136b's conclusion, where it punted on the question of *which fix to prioritize* and said the choice depended on whether 136a was viable. Reading that, I went and re-read its 136a analysis, which I had skimmed the first time because of the 22 GB number.

The relevant paragraph was this one:

> *Critical observation: The encoder output is heavily redundant. Every channel uses `jnp.tile` to create `(n, n)` from `(n,)` or `(n, 1)`. […] Each channel is a row-wise tile of a 1D vector: channel shape `(64, 64)` where every row is identical. Storage: `(64,)` per channel instead of `(64, 64)`. Compression ratio: 64×. […] All 27 channels follow this pattern.*
> — `tasks/136a-k.md` (the agent-side investigation file)

I want to be honest about the moment I read that. I sat back in the chair. The desk fan was running. The right pane was still showing the GPU at 13 %. The thing the agent had just told me was that the tensor my model had been training on for the last *month and a half* was, channel by channel, a flat row of 64 numbers stamped 64 times across an axis that had no semantic content. The axis I had assumed was lookback bars was, in fact, *the same row repeated*. I had been feeding the network 64 copies of the same data and patting myself on the back about the channel count.

Let me show what that looks like, because the picture clarifies. Take channel 4, the 1-minute return:

```
Original (n,) per-asset vector:
  returns_1m = [r0, r1, r2, ..., r63]              # shape (64,)

Tile to (n, n):
  jnp.tile(returns_1m[:, -1:], (1, n))             # shape (64, 64)

Result:
  [[r0, r0, r0, ..., r0],
   [r1, r1, r1, ..., r1],
   [r2, r2, r2, ..., r2],
    ...
   [r63, r63, r63, ..., r63]]
```

Every column of the channel is identical. Every value past the first column is *information-free*. And every one of the 27 channels followed this pattern — the close prices, the 1-minute returns, the 5-minute returns, the OHLC normalizations, all of them. The encoder was producing 64 × bigger output than it had information for, and the rest of the pipeline — the BC dataloader, the host-device transfers, the cache decisions, the VRAM budgeting, the fight over 22 GB — was paying the bill.

The math after the observation is the easy part. Per sample, the *informative* content of the encoder output is `27 × 64 × 4 bytes = 6.75 KB`, not `27 × 64 × 64 × 4 bytes = 442 KB`. Across 50 000 demos that is 338 MB, not 22 GB. The compression ratio is exactly the tile factor — 64×. Cache the compressed version in RAM, decompress by re-tiling at sample time (a numpy broadcast that runs in microseconds), and the whole "we can't afford to cache the encoded observations" argument falls apart.

The first time I wrote the next prompt back to the agent I think I typed it twice before I sent it. The Lukas voice in those snapshot sessions does this thing where the volume goes *down* when the discovery is large. There is no exclamation point. There is, instead, the question of whether the discovery is actually a discovery and not a misread.

In the back-and-forth that followed (a five-message thread inside the same session) we walked through the verification together. I asked it to enumerate every channel and check the tile pattern. The agent went through `_compute_all_channels` line by line — channels 0–3 (OHLC), channels 4–7 (returns over various horizons), channels 8–11 (volatility), and so on — and confirmed that every single one terminated in `jnp.tile(..., (1, n))` or an equivalent `jnp.repeat`. There was no channel that used the second 64-axis for anything. The verification ended with the same line, dropped at the bottom of the appendix in the investigation file:

> *"ALL 27 channels follow this pattern."*
> — `tasks/136a-k.md`, "APPENDIX: Verification of Compression Safety"

I want to be careful with one nuance, because it is going to matter for the kickback. **The redundancy was not an encoder *bug* in the sense of "the code does the wrong thing."** The encoder was producing what the encoder had been written to produce. The cross-sectional axis was real in the *design* — the original intent was that the second 64-axis would carry interactions between assets, a matrix where row *i* column *j* held a feature about *asset i in the context of asset j*. The interactions had never been implemented. The placeholder — `jnp.tile` — had been left in. The downstream network learned to *ignore* the redundant axis (it had no signal there to learn from); the dataloader, lacking any way to know the axis was placeholder, dutifully ferried 64× more bytes around than it needed to.

The bug was structural. It was in the *contract* between the encoder and the rest of the pipeline, not in either side in isolation. Each side, audited alone, looked fine. Audited together, the pipeline was running on a dataset 64× larger in memory and 64× heavier in I/O than its information content justified.

There is one more piece of texture from the collab I want to keep. Gemini's 136c response — the one on whether to keep episode prices in VRAM instead of RAM — is the cleanest example I have of an agent giving a *useful negative*. The proposal sounded reasonable on its face: stop the round-trip transfers, keep the data device-side, skip the `jnp.array()` call inside the loop. The agent's verdict:

> *"Do NOT implement VRAM storage as a standalone fix. It yields minimal gain because the loop overhead dominates. […] The bottleneck is the Python control flow, not the memory bandwidth."*
> — `tasks/136c-gemini-solution.md`, §5

That paragraph saved me a day. Without it I would have spent an afternoon writing the VRAM-resident dict, the threading-safe transfer helper, the cache eviction, the JAX-array-in-Python-dict lifetime tests — for a *2 %* speedup against a problem where the actual win was sitting in 136a. The discipline of "another set of eyes whose context is loaded freshly" is, almost more than the discovery itself, what made the collab work.

## What we did about it

The fix was small. The implementation took the next afternoon. The shape of it:

1. In the self-play callback `on_episode_complete_callback` in `train_jax_fast.py`, when an episode finished and the oracle had labeled its steps, batch-encode the entire episode in one JAX call. (340 valid steps × `(27, 64, 64)` → one vmap, one kernel, no Python loop.)
2. Compress immediately: take only the first column of the third axis, giving `(340, 27, 64)`. Drop the rest. (This is the "take a single column because every column is the same" move. It is one line.)
3. Store the compressed tensor in the expert buffer alongside the existing index entry, keyed by demo position.
4. At `sample_batch` time, retrieve the compressed observations by index, decompress with `np.tile(compressed[..., None], (1, 1, 1, n))`, and hand the result to the trainer.

The contract between encoder and trainer did not change. The trainer still consumed `(batch, 27, 64, 64)` tensors. The decompression at sample time produced exactly the same bytes the original encoder would have produced — there was no loss, because the compressed form held all the information. The change was structural: encode once at write time, retile cheaply at read time, instead of encoding every batch.

Memory dropped from a projected 22 GB cached path (which we had been refusing to take) to **338 MB** actually allocated. GPU utilization climbed from the 5–17 % band into the 75–90 % range over the next training run. The epoch time fell roughly in proportion. The training that had felt *slow in a way that didn't add up* started running in a way that did.

It is the morning of May 21, 2026 as I write this. The cache pattern from that afternoon is still in `expert_buffer.py`. The `(27, 64)` compressed form is still what the buffer stores. The decompress-on-read tile is still the thing the trainer indexes into. Every Phase 2 and Phase 3 experiment in this project — every TASK_737 ablation cell, every TASK_743 multi-seed crusher, every TASK_734 flow-matching probe whose results form the spine of articles 6 through 10 — ran on top of this fix. If the fix had not landed in January, the May 2026 campaign would not have been possible at the throughput it ran at. Foxglove's training cycle would still be sipping through a 64× straw.

## The kickback

This is the part of the article I have been avoiding writing for two paragraphs.

The bug had been training for **months**.

Every experiment we had run between October 2025 and January 2026 — every hyperparameter sweep in DS56, every architecture tweak, every reward-shape adjustment, every curriculum-level escalation — had been running on a pipeline that was burning four-fifths of its compute on a placeholder axis. The numbers I had been reading off the dashboard and building intuition from were obtained at roughly one-fifth the effective batch throughput the box was capable of, on a dataset 64× larger in memory than it needed to be, with a GPU bored out of its mind between launches.

Some of the intuitions were probably right anyway. The shape of a learning curve does not lie about whether learning is happening, even if it is happening slower than it should. Loss going down is loss going down. The relative ordering of two configurations under the same broken pipeline is mostly preserved — they were both penalized by the same factor, so the ratio survives.

Some of the intuitions were probably wrong. A learning curve that plateaus after eighteen hours might look like *the network has converged* when it actually means *the network has seen as many gradient updates as it can fit into eighteen hours at one-fifth throughput, and would still be learning at hour ninety*. A configuration that looked "noisy" might have been noisy because every batch contained fewer effective gradient steps than its size suggested. The intuitions I cannot now disentangle are the ones about *time to convergence* and *sample efficiency* — the metrics where the throughput mattered most.

We will never know which intuitions were which. We threw the runs away. We had to — the value-head bins changed underneath the same audit, the encoder contract was about to shift, the checkpoints were stale. The pre-TASK_136 runs are not on disk anywhere I can re-evaluate them. They live as half-memories in old DS56 screenshots and the diary entries I wrote at the time, both of which I now read with a footnote attached: *under-throughput, suspect*.

There is a second observation in this kickback that hurts more than the wasted compute, and it is the one this article is really about.

**The bug was caught by Gemini.** Not by me. Not by Claude Code, which had been working in this codebase daily for weeks. Not by the test suite. Not by any assertion. The bug was caught by a fresh agent reading the encoder file cold, asked a different question, and noticing — in the margin of its answer to that different question — that the output shape did not match its information content.

I want to land this carefully. It is tempting to make the lesson into a Gemini-vs-Claude thing or a we-need-three-agents thing. That is the wrong lesson. Both agents are good. Both had access to the same code. The thing Gemini had that the daily collaborator and I did not have was *context that had not yet adapted to the file*. When you read a piece of code every day, your eye stops landing on the lines that have been there since the beginning. The `jnp.tile` calls in `_compute_all_channels` had been there since November. They were furniture. The mind that reads furniture as furniture stops asking whether the furniture should be in the room.

A fresh agent, loaded with the file from scratch and asked *"could we cache the output of this encoder cheaply,"* read the `tile` line as a *fact about the output shape* rather than a *part of the room*. It noticed, immediately, that the output shape was not what the bytes suggested. The same brain that I had — the same brain Claude had — would have noticed it too, if either of us had been asked the question for the first time, with no priors.

The lesson, then, is not about which model. The lesson is that *another set of eyes whose context is loaded freshly* is a structural asset, not a redundant cost. The pattern of relaying tasks between agents — taking a problem Claude had been working on, handing it sideways to Gemini, taking the answer back to Claude with the framing "by gemini task 120" — became a working method on this project from this point forward. The SSD-snapshot session from Jan 28 preserves the pattern exactly:

> *"ok have rebooted implemented some fixes .. by gemini task 120 .. now need command to star env .. start ds24 start traini[ng]"*
> — Lukas, 2026-01-28, snapshot session `4fe59e46-3e97-4731-8f87-6093498cf158.jsonl`

That sentence — typo-strewn, one-handed, at the start of a session — is the working method. *Implemented some fixes by Gemini, now back here, give me the commands to restart.* Gemini does the cold-read; Claude does the contextual integration; the human relays. The structural insight is not "use both"; it is that the *handoff* — the moment of context-clear at the seam — is where the bugs that have become furniture get caught.

## What's left, what I learned

The cache pattern from TASK_136a survives, untouched, in `mcts_trader/env/expert_buffer.py` and in the self-play callback. The compressed `(27, 64)` form is the canonical storage shape for the buffer. The encoder's `_compute_all_channels` still produces `(27, 64, 64)` — the contract did not change, the *use* of the contract did — and at sample time the decompression re-tiles to the full shape the network consumes. The placeholder cross-sectional axis is still there, still tiled, still semantically empty. A later phase of the project (Phase 2, the encoder saga, articles 6 and onward) attacked the axis directly, with mixed results that are not this article's problem.

The dataloader contract changed forever after this. Every dataloader in this codebase, since January, gets audited the same way: walk the output shape, compare bytes-on-the-wire to bytes-of-information, and if they don't match, find the tile or the broadcast or the duplicated padding and account for it. The internal rule I wrote down at the time and have not had to revise: **trust your own dataloader the way you trust a stranger's, which is to say not at all.** The dataloader is the seam where the upstream and downstream contracts meet, and seams are where the structural bugs live. Audit the seam.

A footnote about the article title. The original tag for this audit, in the lore directory, was *"information loss audit"* — that is the filename, `2026-01-04_information_loss_audit.md`. The label was chosen at the start of the audit, when I thought I was looking for a place in the pipeline where information was being *destroyed* (a value-head bin collapse, a normalization wiping signal, a reward-bin pileup). Those bugs were real and we found them — the value-head support shifted from `[-750, +750]` with 0.5 % bin utilization down to `[-10, +10]` with 31.6 % utilization in the same audit, and that is its own story. But the load-bearing finding of the month, the one this article documents, was not about information *loss*. It was about information *redundancy* — the same row, repeated 64 times, padded into a shape that pretended to be 64× richer than it was. The audit changed shape mid-flight, as audits do. The title stuck. The mechanism it described slid one column over.

The last thing I want to record is what the audit did to my relationship with my own confidence. I had been training for months on a pipeline I had written, that I had reviewed, that I had walked agents through, that had not crashed, that had been producing results I was building a research direction on. The bug that turned out to be load-bearing was a placeholder I had typed myself and forgotten. The fix took an afternoon. The implications took the rest of the project to absorb. From that month on, every "the loss is going down" got a quiet asterisk attached — *under throughput we have measured, on a pipeline whose bytes-to-information ratio we have audited, with seams we have walked.* The asterisk is annoying. It is also the only honest way to read a learning curve.

## Closing

It is now May 21, 2026. The cache pattern is still doing its job. The `(27, 64)` compressed buffer is still 338 MB. The GPU on `workstation` is at 78 % as I write this, running a checkpoint validation in the next window, exactly the kind of saturation that was supposed to be there all along and finally was, four months ago, on the afternoon a fresh pair of eyes told me the second 64-axis was empty.

> *I record the redundancy in the margin and continue. The watering schedule does not change. The pot does.*

## Sources

- `tasks/136a-k.md` — the agent-side investigation file, including the 22 GB → 338 MB memory analysis and the `jnp.tile` channel verification ("ALL 27 channels follow this pattern. Compression ratio: 64×").
- `tasks/136b-gemini-solution.md` — Gemini's batched-JAX-call analysis; the gather pattern, the recommendation that 136a was the superior fix if memory allowed.
- `tasks/136c-k.md` and `tasks/136c-gemini-solution.md` — the VRAM-storage proposal and its negative verdict ("The bottleneck is the Python control flow, not the memory bandwidth.").
- `tasks/136d-k.md` and `tasks/136d-gemini-solution.md` — BC-frequency reduction as a low-effort secondary fix.
- `tasks/137-k.md` — the -102 % alpha investigation, contemporary with the audit (closed as expected behavior).
- `lore/2026-01-04_information_loss_audit.md` — the value-head bin collapse (`[-750,+750]` → `[-10,+10]`), reward categorical binning, confidence-threshold policy disconnect. The audit this article's mechanism slid sideways from.
- `lore/notes_2026-01-20.md` — TASK_105 data-loading hang, same week, adjacent texture.
- SSD-snapshot Claude Code sessions, mounted from a recovered drive:
  - `9774f034-d9c1-489b-a70c-94dfba6b586c.jsonl` — 2026-01-11 21:07, the first prompt of the snapshot era: *"hi take look on this codebase and whole project tell me if its possible oneshoot prompt this ??"*
  - `55be4183-bf68-4874-8c07-095eee979767.jsonl` — 2026-01-13 22:27, opener *"are you up to date .. i mean cli"* (quoted in §1).
  - `79d73903-4ef0-47a9-a778-18e83c65549d.jsonl` — 2026-01-22 22:17, patch-verification grind: *"ok cool .. please check again your patch of im not sure if it was fully sucesfull"*.
  - `4fe59e46-3e97-4731-8f87-6093498cf158.jsonl` — 2026-01-28, the Gemini→Claude relay opener (quoted in the kickback).
