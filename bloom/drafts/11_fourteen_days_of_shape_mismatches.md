> *Observation: the fourteen-day weather front is on us. I do not check the leaves. I do not check the flowers. I check every joint — where pipe meets pipe, where soil meets root, where stem meets leaf, where the trellis bolts to the wall. The plant is one organism only on a sunny day. Under weather, it is a thousand seams.*

---

**by Lukas — May 22, 2026**

# Fourteen Days of Shape Mismatches

*On the bug class that lives in the gaps between modules, the twenty-six-day audit campaign in late January, and the discipline of taking apart a working system to see whether it actually worked.*

---

## What it feels like to crash interestingly

By the time TASK_115 starts I have a pipeline that has stopped being broken in the loud ways and started being broken in the interesting ones. The BUG_MAYHEM audit from a week earlier has cleaned the reward and observation paths — Article 5 is the story of that work — and the model has gone from "1492% Alpha on epoch one" to a clean training run that, on a good day, completes maybe sixty epochs before failing. The failures are no longer about the math being wrong. They are about the *code* being wrong.

A run on `workstation`, the workstation I do all of this on — single EVGA 3090 FTW3 Ultra, 420 W, 24 GB — looks like this. I launch `train_jax_fast.py` with the canonical config. The XLA compile sits for forty seconds. The first training step prints. The second training step prints. Somewhere between step three and step ninety the run dies. Sometimes it dies with an out-of-memory error from inside a Mamba layer. Sometimes it dies with a tracer leak from inside the generation loop. Sometimes it dies with a cryptic Flax error about a broadcast shape that I have to read three times before I understand which tensor it is talking about. Sometimes — and this is the worst kind — it does not die at all. It runs for an hour, prints a loss curve that looks plausible, and produces a checkpoint whose outputs are, on closer inspection, mostly NaN.

I am writing this in late May, four months later, from the same machine. The desk fan is on, `nvidia-smi` is idle, and the `tasks/` directory in this repository contains twenty distinct files whose names start with `TASK_115`. They were not written in one sitting. They are the residue of a campaign that ran from January 25 through early February — fourteen working days, give or take — across which I and a rotating cast of coding agents took the codebase apart joint by joint and put it back together. The master tracker counts forty-four unique bugs across seven categories, distilled from one hundred and seven raw findings.

Article 5 was about *what made the model fail to learn*. This article is about *what made the code fail to run at all*. Different shape of disaster. Different lesson.

## What an integration-seam bug actually is

I want to introduce a category before I go any further, because the campaign only makes sense if you accept it as a category. The bugs in TASK_115 are not the bugs in BUG_MAYHEM. BUG_MAYHEM was about reward signals, future leakage, and algorithm-stack ordering — *what the model was being trained on*. TASK_115 is about a different thing: the bugs that live in the gaps between modules, not inside them.

A module-internal bug is a bug you can find by reading one file. The wrong sign on a reward. The wrong axis on a normalization. The missing log-gamma term in a Student-t loss. You isolate the file, you read it, you understand it, you fix it. The unit test for the file would have caught it, if a unit test had existed and been comprehensive.

An integration-seam bug is different. Every module is correct in isolation. The `MambaBlock` correctly applies FiLM conditioning to its internal representation. The `SkeletonConditioner` correctly returns a `(gamma, beta, h0)` tuple. The `MarketGenerator` correctly accepts a `skeleton` argument and threads it down. But the `MambaBlock`'s internal representation is shape `(batch, seq, d_inner)` where `d_inner = 2 * d_model`, and the `gamma` it receives from the `SkeletonConditioner` is shape `(batch, d_model)`, and JAX silently broadcasts the smaller tensor against the larger one in a way that means the FiLM modulation is applied to half the channels and silently zero on the other half. The bug is not in either module. The bug is in the *handshake*.

These bugs share a property that makes them especially nasty: they are invisible to every unit test that targets the producing or consuming side of the seam alone. If you write a `MambaBlock` test that asserts the output has shape `(batch, seq, d_model)` after the output projection, it passes — the projection works regardless of what FiLM did upstream. If you write a `SkeletonConditioner` test that asserts the returned tuple has three elements of the right shape, it passes — the conditioner is producing what its spec says it produces. The seam itself has no test, because the seam isn't a module. It's the air between modules.

The class includes more than dimension mismatches. The integration-seam category is large. It includes JIT failures at compile boundaries — code that runs fine in eager mode but breaks when the tracer crosses from one component into another. It includes silent failures — where a downstream consumer encounters a NaN or an out-of-bounds index from an upstream producer and *ignores it*, because indexing in JAX silently wraps, and the consumer was written assuming the producer would never produce such a thing. It includes memory explosions — where two reasonable design choices, made in separate modules with separate memory budgets, compose to demand 150 GB of VRAM on a 24 GB card. It includes API drift — where a function signature changes in one file and the caller in another file is updated, but a third caller in a fourth file is not. It includes race conditions — where two threads hit a shared piece of state via paths that each looked single-threaded when read in isolation.

The unifying property: each bug is a function of *how two correct pieces of code talk to each other*, not of what either piece of code does on its own.

## A guided tour through six of the fourteen

Twenty TASK_115 files exist in the tracker. The master report deduplicates them down to forty-four unique bugs, of which twelve are critical. Walking through six is enough.

### Bug one: FiLM conditioning at the wrong dimension

This is the canonical example. `diff_teacher/mamba/mamba_block.py`, lines 117 to 126. The FiLM (Feature-wise Linear Modulation) block is the standard way to inject a conditioning signal into an intermediate representation: you produce a `(gamma, beta)` pair from the conditioning input, and you compute `y * (1 + gamma) + beta` to modulate the activations. The math is clean. The implementation was almost right.

From the shape-mismatch report:

> *"`y` has shape `(batch, seq, d_inner)` where `d_inner = d_model * expand`. But `gamma/beta` is `(batch, 1, d_model)`. When `expand > 1`, shapes mismatch."*
>
> — `tasks/TASK_115_BUGHUNT_SHAPE_MISMATCHES.md`, BUG-001

The default expansion factor in our Mamba blocks is two. So `d_inner` is double `d_model`. The conditioning tensor is `d_model` wide. JAX broadcasts the conditioning tensor across the second half of `d_inner` — which is to say, it doesn't, really; it silently extends the smaller tensor in a way that depends on memory layout and produces a conditioning signal that is correctly applied to the first `d_model` channels and *something* applied to the second `d_model` channels, where *something* is whatever the broadcasting rule resolves to in this particular shape configuration.

This is not the kind of bug that crashes. The shapes are broadcastable, in the technical sense — JAX will not raise. It is the kind of bug where the model trains, the loss descends, and the conditioning signal is half-applied for the entire duration of training. You only see it if you sit down and write the shapes out on paper. The integration-seam report from a different reviewing pass picked up the same bug independently and called it BUG-101:

> *"With `expand=2`, `d_inner = 2 * d_model`, so FiLM is applied to mismatched dimensions. JAX will broadcast gamma/beta, truncating the conditioning to only affect half of d_inner."*
>
> — `tasks/TASK_115p_BUGHUNT_integration_seams.md`, BUG-101

The fix is two lines: move the FiLM application to after the output projection, where `y` has shape `(batch, seq, d_model)` and the conditioning broadcasts cleanly. The interesting part is not the fix. The interesting part is that this bug existed for weeks in code that ran every day, in a module that had three unit tests, in a model architecture I had personally drawn on paper before implementing. The seam was invisible because the seam was *correctness across modules*, and there was no test that crossed the module boundary with the right shape configuration.

### Bug two: the skeleton conditioning that wasn't

The same campaign found a related bug in the generation pipeline that is worse in a specific way: the conditioning input was *correctly produced and entirely ignored*.

`MarketGenerator.generate()` accepts a `skeleton` argument. The argument propagates through to `_generate_impl()`. The implementation function then runs a Euler integration loop calling `self.model.apply(...)` with the latent state, the time index, and the regime — but not the skeleton. From the integration-seam report:

> *"`MarketGenerator.generate()` accepts a `skeleton` parameter (line 146). But `_generate_impl()` never receives or passes skeleton conditioning to the model. `MambaFlowGenerator.__call__` expects `skeleton_cond` parameter (line 154). Skeleton-conditioned generation is documented but non-functional."*
>
> — `tasks/TASK_115p_BUGHUNT_integration_seams.md`, BUG-102

Read that carefully. The API of the model accepts a conditioning input. The API of the generator accepts a conditioning input. The bridge function between them silently drops it. Every test of the generator at the API level passed — you gave it a skeleton, it returned a sample, no error was raised. Every test of the model accepted a `skeleton_cond` and produced different outputs when it varied. *Both sides worked.* The bug was the missing wire. We had spent days wondering why skeleton conditioning seemed to have no effect on generated samples. The reason was that it had no effect on generated samples.

What kills me about this one is that we then found a follow-up. Even when the skeleton conditioning *was* wired through, the receiving code unpacked three components — `gamma_s`, `beta_s`, `h0_s` — and used only two of them:

> *"`MambaFlowGenerator` unpacks beta_s but discards it. Only gamma_s and h0_s are added to the conditioning. Skeleton FiLM shift (translation) is lost. Only scale and initial state affect generation."*
>
> — `tasks/TASK_115p_BUGHUNT_integration_seams.md`, BUG-103

And one further down: `h0_s` was the "initial state" component, threaded all the way to a `MambaBlock` that *did not accept an initial state argument*. The state was initialized to zeros inside the block, and the `h0_s` we had carefully computed and threaded through the entire pipeline was discarded at the last call. Three pieces of skeleton conditioning, three independent failures, all in the same handshake.

### Bug three: the 4D tensor that ate the GPU

`diff_teacher/mamba/ssm.py`, line 96. The selective scan, which is the inner loop of Mamba's state-space model, materializes two 4D tensors of shape `(batch, seq, d_inner, d_state)` as inputs to JAX's `associative_scan`. The math is correct. The implementation is correct. The default hyperparameters are catastrophic.

From the OOM report:

> *"With default config: `B=64, L=512, D=1536 (768*2), N=64`. Size: `64 * 512 * 1536 * 64 * 4 bytes` ≈ **12.8 GB per tensor**. Two tensors (`delta_A`, `delta_B_x`) = **25.6 GB** just for inputs to `associative_scan`. This will immediately OOM on most GPUs (including 24GB consumer cards) for a single layer."*
>
> — `tasks/TASK_115h_BUGHUNT_OOM.md`, BUG-001

Twenty-five and a half gigabytes per Mamba layer. The model has twenty-four layers. Bidirectional. The memory accountant in the memory-explosion report does the multiplication out loud:

> *"24 layers * 2 directions * 12.8 GB = **Impossible**. Even with `d_state=16`, it is `24 * 2 * 3.2 GB` = ~150 GB context."*
>
> — `tasks/TASK_115h_BUGHUNT_OOM.md`, BUG-002

This is a seam bug because every individual choice was reasonable inside its own module. The `MambaBlock` author looked at the Mamba paper, saw `d_state=64` as a not-uncommon setting, and used it as the default. The `BiMambaBlock` author wrapped two Mamba blocks bidirectionally, which is the standard pattern. The `MambaFlowGenerator` author stacked twenty-four of these, which is the standard depth for this class of generative model. *Each choice was paper-supported.* The composition was an instant OOM. None of the modules had a memory budget. None of the layer counts were communicated up the stack. The integration was performed by a stack of `for i in range(self.n_layers)` calls with no instrumentation.

The fix landed in three places at once. Reduce `d_state` default from 64 to 16, which matches the Mamba paper's typical use and cuts memory by 4×. Add `nn.remat` (Flax's gradient checkpointing) to every layer in the stack, which trades compute for memory by recomputing forward activations during backward instead of storing them. Default the parallel scan off entirely — `use_parallel=False` — and use sequential mode unless explicitly opted in. The three fixes compose to make the model fit on a single 3090 with sequence length 512 and batch size 64, which is the configuration I have been running on `workstation` ever since.

### Bug four: JIT failures at the compile boundary

JAX's tracing model has a specific failure mode that does not look like a bug until it bites. A function compiles fine. It runs. The output looks correct. But it recompiles on every call because something about the trace context changes between invocations — most often, a Python loop whose length varies, or a Python `len()` on a JAX array, or a dictionary `get()` with a default that silently changes the trace.

From the JIT audit:

> *"The `sample` method uses a Python `for` loop (`for i in range(num_steps):`) for Euler integration. If `num_steps` is large (e.g., 50-100), this causes the JIT compiler to unroll the loop, creating a massive computational graph, slow compilation, and potentially OOM during compilation."*
>
> — `tasks/TASK_115f_BUGHUNT_JIT_Audit.md`, Issue 1

The Euler integration loop in the flow-matching generator was a Python `for` loop. When JIT-compiled, the loop was unrolled into the XLA graph, producing a computation graph with sixty copies of the velocity network. Compilation time, instead of being a one-time cost of the network forward pass, became proportional to the number of integration steps. On a fresh launch this meant the first call took five minutes to compile. If you then changed `num_steps` from 50 to 60, it recompiled. Five more minutes.

The deeper problem was a different category of JIT failure in the skeleton encoder. The `EventEncoder` iterates over a Python list of `Event` objects, using `event.time` (a Python int) to index into a JAX array:

> *"Python `for event in events` iterates over a Python list. This creates a trace dependency that prevents JIT compilation. `event.time` is Python int used in `at[].set()` which is dynamic indexing. `hash(event.event_type)` is Python operation."*
>
> — `tasks/TASK_115p_BUGHUNT_integration_seams.md`, BUG-106

This pattern looked fine in isolation. The `EventEncoder` was tested with one event, two events, ten events, and produced correct outputs each time. But it had to be called from inside a JIT-compiled training step, where the Python list of events became a trace dependency, which meant the JIT recompiled every time the number of events changed. In production training, the number of events varied per batch. The function recompiled on every batch. The reported training step time was the time of one forward-backward; the *actual* wall-clock time per batch included a JIT recompile that frequently exceeded the gradient step itself.

There is no way for a unit test on `EventEncoder` to catch this. The unit test runs in eager mode. The bug only appears at the JIT boundary, which is in another file, in another module's training loop, and the cost only shows up in the wall-clock — which I was not instrumenting at that level of granularity.

The fix was to refactor `EventEncoder` to take pre-padded JAX arrays (`event_times`, `event_types`, `event_magnitudes`) with a fixed maximum count, and use `jnp.where` for the conditional masking. Pre-padding makes the trace context constant. The function compiles once. The bug had been there for the entire history of the skeleton-conditioning feature.

### Bug five: the silent kind

There is a category of bug in this campaign that I rank above all the others as the worst kind, because it has a specific shape: the upstream module produces a value that is, strictly speaking, *invalid*, but the downstream module treats it as if it were valid, and JAX's indexing semantics ensure that nothing crashes.

The canonical example is regime index out-of-bounds in the skeleton builder:

> *"`vol_map` has 5 elements (indices 0-4). If `regime_seq` contains values >= 5, JAX indexing silently wraps: `5 % 5 = 0`. Generator receives corrupted volatility conditioning without any error."*
>
> — `tasks/TASK_115n_BUGHUNT_silent_failures.md`, BUG-009

A regime index of five — meaning "calm-low-volume," a regime that did not exist when the volatility map was authored — gets silently mapped to regime zero ("crash"). The downstream model receives "this is a crash regime" conditioning for windows that were tagged as calm. The training signal is corrupt and no error is raised at any point in the stack. The only way to find this bug is to know it exists and to grep for unguarded indexing operations.

The same report turns up a sibling bug in the embedding layer:

> *"`nn.Embed(num_embeddings=5)` expects indices [0, 4]. OOB indices wrap: regime 5 → embedding 0. Different regimes produce identical embeddings."*
>
> — `tasks/TASK_115n_BUGHUNT_silent_failures.md`, BUG-010

Same shape of bug. Different module. Found in the same audit pass.

And a third, in the skeleton types file, that I want to quote because it captures the texture of JAX-specific seam failures:

> *"`len()` on JAX Array raises `TypeError` in JIT contexts. Fails at trace time."*
>
> — `tasks/TASK_115n_BUGHUNT_silent_failures.md`, BUG-013

This one is the opposite of silent — it actually crashes — but it only crashes inside JIT. In eager mode, `len(jax_array)` returns a Python int and works. In JIT mode, the trace cannot resolve `len()` on an abstract tracer object and the compilation dies with a `TypeError` whose message points at the wrong file. The fix is to use `.shape[0]` instead of `len()`, which produces a static int in both contexts.

The lesson of the silent-failure category is the one that stayed with me longest. The default behavior of JAX is to compose. The default behavior of Python is to wrap. When a module produces an invalid value and a downstream module reads it, the default outcome is *silent corruption*, not crash. Every numerical operation in the codebase that uses one module's output as another module's input is a potential silent-failure seam unless you have explicitly added a guard. Twenty-two bugs in the numerical-stability category alone. That number is not because the math was wrong. That number is because there were twenty-two places where producer-consumer pairs had no guard at the seam, and the worst of them — division by zero in time-embedding sinusoidals, KL divergence in fidelity evaluation, Hill estimator in stylized facts, flow-matching running std — produce NaN that then propagates through the entire forward pass and silently poisons every gradient downstream.

### Bug six: the integration-seam category itself

I want to close the tour with a meta-bug, because it became the framing for the whole campaign. The two integration-seam reports — `TASK_115p_BUGHUNT_integration_seams.md` and `TASK_115q_BUGHUNT_seams.md` — are not lists of bugs. They are *audits of handshakes*. Each entry names two modules and the contract between them, then describes how the contract is violated.

The OHLCV index inconsistency is the cleanest example:

> *"Stylized facts uses Close price (index 3) for return calculations. Fidelity/controllability use Open price (index 0). Return calculations on Open vs Close give different results. No central OHLCV constant definition."*
>
> — `tasks/TASK_115p_BUGHUNT_integration_seams.md`, BUG-105

Three evaluation modules. Each computes returns. Each indexes into the data tensor. Each makes a *different assumption* about which column is the close price. The data tensor was OHLCV — open at zero, high at one, low at two, close at three, volume at four. One module used three. Two used zero. No module was wrong on its own terms — the one using zero was computing returns on open prices, which is a valid thing to do — but the three modules were producing metrics that could not be compared to each other or to the model's own training signal, which used close.

The fix was to add a single source of truth: a `constants.py` file with `OHLCV_CLOSE = 3` and to update every consumer to import from it. The fix is one constant. The audit to find every consumer that needed updating was its own subtask. The bug existed because nobody had ever sat down and asked the question "where is the contract for which column is which?" The contract lived nowhere. Each module had reconstructed it from local context.

That is the integration-seam category in its purest form: there is no module-internal bug. There is no algorithmic mistake. There is only the absence of a shared contract, and three good-faith implementations that disagree.

> *The gardener kneels at the trellis. She runs a thumbnail along every joint. The wood is sound; the bolts are tight; one fitting is the wrong size and has been the wrong size since planting. The plant has been growing around it all season. She marks the joint with chalk and continues down the row.*

## The kickback, in two layers

There is a kickback in this campaign, but it does not look like the kickbacks in other articles in this series. It is not a number that reversed under more seeds, or a mechanism that failed under a more honest operator. It is structural, and it has two layers.

**The first layer:** every module in this codebase had unit tests. The `MambaBlock` had tests. The `SkeletonConditioner` had tests. The `MarketGenerator` had tests. The selective scan had tests. The flow-matching sampler had tests. The skeleton extractor had tests. The evaluation modules had tests. Each test passed. Each test had been written by someone — me, or a coding agent under my direction — who understood the module they were testing, and the tests covered the module's behavior in isolation. *None of the bugs in TASK_115's master tracker were caught by any of these tests*, because the bugs did not live inside the modules the tests targeted. They lived in the gaps.

This is a humbling property to confront. It is not a property of being a bad test author. It is a property of how testing works as an epistemic technology. A unit test scopes itself to a module by definition. It mocks the module's inputs, exercises the module's outputs, and asserts the module's invariants. The gap between two modules is not a module. It has no surface a unit test can stand on. Integration tests get closer — they exercise multiple modules at once — but for this codebase, the integration test for the full forward pass *also passed*, because the bugs degraded outputs in subtle ways (half-applied FiLM, wrong-column returns, silent regime wrapping) that did not cross the threshold of "the integration test asserts something about the output." The test asserted shapes were correct. They were. The test did not assert that the FiLM modulation reached every channel, because the test author did not think to ask.

**The second layer:** the only way to find these bugs, given the above, is to deliberately set up a different kind of pass over the code. Not a test. An audit. With a specific posture: read the code as if you are trying to break it. Read the seams specifically. Read the handshakes between modules and ask, for each one, *what is the contract here, where is it documented, and is it satisfied?*

This posture has a name in this codebase, by the end of the campaign. It is called *adversarial review*, and TASK_115r is the file where it became explicit. The setup is to take the patches produced by the previous fourteen days of audits, hand them to a different agent with a different prompt frame, and ask it to find what the previous passes missed. The result is itself instructive. Quoting the executive summary verbatim:

> *"**Fixed**: Critical wiring in `MarketGenerator` and `AdaLN` parameter sharing.*
> *— **Partially Fixed**: Skeleton sequence encoding (introduced normalization bug).*
> *— **Not Fixed**: Time-series boundary bleeding in evaluation metrics.*
> *— **New Issues**: 3 new logic/consistency bugs identified in the patched code."*
>
> — `tasks/TASK_115r_ADVERSARIAL_REVIEW.md`, Executive Summary

Three new bugs. In the patches. From the previous audit. Read that twice. We had spent fourteen days finding and fixing forty-four bugs, and the adversarial review of those fixes turned up three more in the patches themselves. One of them was a regression: the patch for the skeleton sequence-embedding bug had introduced a new normalization bug. The patched code summed the per-timestep embeddings correctly but divided by the wrong count — by the number of sequence types present, not by the count of valid contributions at each timestep — so a skeleton with `Trend` and `Events` where the events were short would scale the trend signal by half across the entire sequence. The fix had been subtly worse than the bug it replaced.

The kickback is that there is no fixed point in this process. You do not run the audit, find the bugs, fix them, and arrive at correctness. You run the audit, find the bugs, fix them, run the *adversarial* audit, find the bugs in the fixes, fix *those*, and accept that the fixed version is not the truth — it is the version that has survived two passes of adversarial reading. The third pass would find more. At some point you have to stop. The stopping criterion is not "no bugs remain." The stopping criterion is "the remaining bugs are below the threshold at which they affect the deployed behavior we care about." This is a deeply unsatisfying criterion. It is also the only honest one.

## What's left

The campaign closed on 2026-01-25 with the master tracker showing forty-three of forty-four bugs fixed, the one remaining flagged as an enhancement rather than a bug. The fix rate at the top of the tracker reads **98%**. I want to be careful with that number, because what "fixed" means in this category is "the specific instance is patched and the test that would have caught it now exists." It does not mean the *class* of bug cannot recur. New code can introduce new seams, and new seams can have new mismatches.

What survives, four months later, is not the patches. It is the discipline. Three specific patterns have become standing practice in everything I have written since:

**Shape assertion at every seam.** When a tensor crosses a module boundary, there is an `assert` on its shape at the receiving end. Not a debug print. Not a comment. A runtime assertion that crashes loudly when violated. The `Generator Reshape Without Validation` bug from the shape-mismatch report (BUG-009) is the canonical reason: a reshape from `(n_samples, seq_length * n_assets * n_features)` to `(n_samples, seq_length, n_assets, n_features)` will silently produce garbage if the flattened length is wrong, because reshapes do not check semantic alignment. The cost of the assertion is one line per seam. The benefit is that the next time the architecture changes and somebody updates `n_features` without updating the reshape, the run dies on the first batch rather than the hundredth checkpoint.

**NaN-detect guards at every numerical seam.** Every divison has an epsilon. Every exponential has a clip. Every embedding lookup has a clamp. The `jnp.maximum(x, 1e-10)` pattern that started as a fix for the time-embedding division-by-zero is now the default in every new numerical kernel. The cost is a few extra ops. The benefit is that a single NaN cannot propagate silently from a numerical edge case in one module into a poisoned gradient across the whole training step.

**JIT compilation pre-checks.** Before any new function gets called in the training loop, it gets a small test that compiles it once, calls it twice with different inputs, and asserts that the second call did not trigger a recompile. The check is forty lines of harness. It catches the entire class of Python-loop-in-JIT and `len()`-on-tracer bugs at the time the function is written, not the day after the training run got 5× slower than expected.

And one thing that did not survive in the patches but planted a seed for later: the *adversarial review* idea. The notion of having a second agent, with a different prompt frame, deliberately try to break the work of the first. I would not understand the full implications of that pattern until much later, when it became the seed of `yarrow/` — the multi-model code-review side project named for the wound herb, the one that finds and cleans. Yarrow is parked for now, but the pattern it formalizes — adversarial reading by a second agent — comes directly from TASK_115r.

The last thing I want to land is this: the campaign was, ultimately, infrastructure work. None of these forty-four bugs were research contributions. None of them produced an Alpha improvement on their own. The model after TASK_115 was not measurably better at trading than the model before. *And yet the campaign was the most important fourteen days of work in the project*, because everything that came afterward — every experiment scored against the canonical baseline, every paired-seed comparison, every Tier-2 ablation that I write about in later articles — was scored on a stack that I trusted to be running the code I thought it was running. Before TASK_115, I did not have that. After TASK_115, I did. The trust is the artifact.

## Closing

It is May. The campaign was four months ago. The shape-assertion decorators are everywhere now, the NaN guards are in every kernel, the JIT pre-checks run as part of the test harness on every commit. `workstation` is on, the 3090 is at idle, the next training run will start in a few minutes and it will not die in the gaps. It might die in the math. It might die in the data. It will not die in the seams.

> *The gardener finishes the row. The trellis stands. The joints hold. The plant grows on. She does not believe the plant is healthy because the leaves look green. She believes the trellis holds because she put her hand on every joint and tested each one until it would not move.*

---

## Sources

- `tasks/TASK_115_MASTER_BUG_TRACKER.md` — consolidated 44-bug audit (primary)
- `tasks/TASK_115_BUGHUNT_SHAPE_MISMATCHES.md` — Hunt 1: Mamba + Skeleton modules
- `tasks/TASK_115b_BUGHUNT-SHAPE_MISMATCHES.md`, `TASK_115b_BUGHUNT_SHAPE_MISMATCHES.md` — duplicate-named follow-on shape audits
- `tasks/TASK_115c_BUGHUNT_numerical_stability.md` — logic error audit (Student-t loss, prediction loop, replay buffer)
- `tasks/TASK_115d_BUGHUNT_diff_teacher.md` — diff_teacher-specific numerical bugs
- `tasks/TASK_115f_BUGHUNT_JIT_Audit.md`, `TASK_115g_BUGHUNT_JIT_Audit.md` — JIT and tracing audits
- `tasks/TASK_115h_BUGHUNT_OOM.md` — Mamba selective-scan memory explosion
- `tasks/TASK_115i_BUGHUNT_MEMORY_EXPLOSION.md` — broader memory audit including BiMamba doubling
- `tasks/TASK_115j_BUGHUNT_api.md` — API drift and signature mismatches
- `tasks/TASK_115k_BUGHUNT_skeleton_generation.md` — skeleton-side numerical edges
- `tasks/TASK_115L_BUGHUNT_race_state.md`, `TASK_115m_BUGHUNT_STATE_RACE.md` — race-condition and state-mutation audits
- `tasks/TASK_115n_BUGHUNT_silent_failures.md` — silent failure category (regime OOB, NaN/Inf edges, JIT len())
- `tasks/TASK_115p_BUGHUNT_integration_seams.md`, `TASK_115q_BUGHUNT_seams.md` — integration-seam reports (FiLM dimension, skeleton wiring, OHLCV indices, EventEncoder JIT)
- `tasks/TASK_115r_ADVERSARIAL_REVIEW.md` — adversarial review of the patches; three new bugs in the fixes
