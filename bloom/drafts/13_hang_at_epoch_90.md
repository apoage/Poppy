> *Observation: the gardener's wall-clock stops every afternoon at the same hour. The pendulum holds its position at the same angle, the brass face shows the same minute, the second-hand refuses to move. At first the gardener tries to fix the hour — winds the spring tighter, taps the case, oils the pivot. Nothing. So the gardener learns to plan around the stopped hour: harvests before, transplants after, leaves the broken interval as a known dead zone in the day. Months later, while disassembling the case for cleaning, the gardener finds a tiny coil at the back of the mechanism, seated wrong, catching on itself every time the hands cross a particular tooth. The hour was never the hour. It was a tooth.*

---

**by Lukas — May 22, 2026**

# The Hang at Epoch 90: An RCA

*On a training loop that froze deterministically at the same place every time, the four-day chase that ruled out the obvious, and the lesson that silent failures need a different epistemology than loud ones.*

---

## Oh, it hung again

It is the third time this week. I am at `workstation` with one EVGA 3090 doing what 3090s do, fan curve climbing, the desk fan on. The training log is in the left tmux pane, `nvidia-smi -l 5` in the right. I have been making coffee. I come back, I sit down, and the left pane has not moved.

```
Training:  81%|████████▏ | 65/80 [19:11<04:29, 17.97s/ep]
```

Cursor blinking at the end of that line. No new line under it. I check the right pane: GPU utilization is `0%`. Memory says `17004MiB / 24576MiB`, which is exactly the value it had ten minutes ago, and the value it had twenty minutes ago. The process is not dead — `ps aux | grep train` shows it alive, RSS stable, CPU at zero. It is just *standing still*. No traceback. No error. No log. The kind of failure that looks, the first time you hit it, like the machine has decided to ignore you.

It hangs at epoch 90 on this run. Last run it hung at 78. The run before that it was somewhere around 65. There is a pattern — late epochs, no crash, no signal, GPU idle, memory pinned — and there is a non-pattern — the exact epoch drifts. The shape of the failure is the same; the timing isn't. It is the worst category of bug: deterministic enough to keep happening, non-deterministic enough that you can't anchor a debugger to a known line. Every overnight run dies in this same shape and produces no evidence of why.

I have been chasing this for four days. I want to write it down while the shape of the hunt is still fresh, because the bug itself turns out to be one of those four-line fixes that looks embarrassing in retrospect, and I would like to remember what made it hard. The bug isn't the lesson. The shape of the chase is.

## The symptom, in detail

If you have not lived inside a training loop, the texture of this failure is worth describing precisely, because almost every debugging instinct I had was wrong for it.

A normal crash announces itself. There is a stack trace. The exception type tells you the class of failure. The line number tells you where to look. You read the trace, you find the variable that was `None` when it shouldn't have been, you fix it, you move on. The whole craft of debugging assumes the failure has a *voice*. You build muscle around interpreting the voice.

This thing has no voice. The progress bar stops moving. The GPU goes idle. The memory stays allocated. There is no exception, because nothing has *raised* anything — the program is, in the strict sense, still executing. It is just executing something that doesn't terminate. From outside it looks identical to "the GPU is doing real work and I am being impatient." From the OS point of view the process is a healthy Python interpreter holding some file descriptors and a CUDA context. From the JAX point of view there are queued device kernels that will eventually return. They never do.

The numerical signature, from `TASK_127r_hang_at_epoch_90.md`:

> *"GPU Util: 0% (Idle). GPU Mem: ~0.9GB (Low). CPU Util: ~13% (Idle/System background). Last Action: Finished saving `checkpoint_126000` (Epoch 90). Current Phase: Entering `run_self_play_jax` for Epoch 91."*
>
> — `tasks/TASK_127r_hang_at_epoch_90.md`

(That first audit had it as 0.9 GB because by the time I noticed and ran nvidia-smi, JAX had garbage-collected the previous epoch's tensors but the new epoch hadn't allocated yet. Later audits with live monitoring caught the more common pattern: 17 GB pinned, not released, GPU still at zero.)

The hang is *deterministic in shape* and *non-deterministic in trigger*. Every run of the same config eventually freezes; the epoch at which it freezes varies. That combination is exactly the thing that defeats easy reproduction. If it hung at epoch 90 every single time, I could `pdb` epoch 89 to death. If it hung at random epochs across random configs I could chalk it up to a driver flake. But it hangs reliably-enough to block all long-running training and irregularly-enough that I can't put a breakpoint where it lives.

The reproduction command, for completeness:

```bash
python train_jax_fast.py \
  --name hang_repro \
  --synthetic-level 0 \
  --epochs 100 \
  --steps 200 \
  --parallel-envs 4 \
  --jax-env \
  --batch-size 512 \
  --use-prediction-loss \
  --imprint
```

Roughly thirty to sixty minutes to hit. With `--imprint` it tends to hit earlier. Without it, it eventually hits anyway. Disabling `--use-prediction-loss` shifts the distribution rightward but doesn't eliminate it. There is no single flag I can throw to make the hang go away — only ones that move its expected timing.

It is, in `TASK_128`'s honest framing:

> *"Training hangs indefinitely at random epochs (typically 60-90+). The process doesn't crash - it just stops making progress. GPU utilization drops to 0% while memory stays allocated (17GB+). This makes production training impossible."*
>
> — `tasks/TASK_128_TRAINING_HANG_BUGHUNT.md`

"Makes production training impossible" was not rhetorical. Every overnight 200-epoch run had been dying around the hundredth-epoch mark for a week. The longest successful run in the project to date had been about 78 epochs. There was no path to any ablation result, any seed sweep, any Phase-2 distillation, while this was open. The whole project was sitting on top of a bug that punished long runs.

## The hunt

I want to walk through what got ruled out before what got found, because in retrospect the ruling-out is the actually-instructive part. Every hypothesis below was credible. Most of them were wrong. The one that was right was, of course, the most embarrassing.

### Hypothesis 1: numerical instability — NaN or Inf in the gradient

The most natural first guess. JAX is happy to silently propagate NaN through a graph; once you've got NaN in your gradient and you're using `optax.adam`, the running second moment becomes NaN, and from then on every weight update is NaN, and your model's outputs are NaN, and any downstream `jnp.where` on a NaN comparison goes to whichever branch JAX picks, and you end up in some completely unspecified region of the loss surface. *Sometimes* this produces a clean crash (when something casts to int) and *sometimes* it just produces an infinite loop of NaN updates that, on certain devices, can stall the kernel queue.

I added the obvious instrumentation:

```python
grad_has_nan = jax.tree_util.tree_reduce(
    lambda a, b: a | b,
    jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x) | jnp.isinf(x)), grads)
)
jax.debug.callback(lambda x: print(f"[NAN-CHECK] {x}") if x else None, grad_has_nan)
```

Ran a 200-epoch repro overnight. It hung at epoch 71. No NaN reported in any of the 70 preceding epochs. The last batch before the hang produced gradients that were tame — norms in the usual band of 0.3 to 2.0, no spike, no drift. Numerical instability was not the trigger. Strike one.

### Hypothesis 2: out-of-memory, gradual fragmentation

The intuition: JAX/XLA holds onto allocations aggressively; the orbax checkpoint roundtrip allocates and frees tensors; over enough epochs the GPU heap fragments to the point where the next big allocation can't find a contiguous block, and instead of OOMing cleanly the allocator stalls trying to compact. This was the hypothesis I held longest because it *felt* right — the hang correlated with epoch count, the symptoms looked allocator-shaped.

But the live monitoring killed it cleanly. From `TASK_128b`:

> *"Training runs fine at 17GB / 93% GPU utilization. Memory consumption is stable, not growing unboundedly. Hang is SUDDEN, not gradual degradation."*
>
> — `tasks/TASK_128b_HANG_ROOT_CAUSE_ANALYSIS.md`

I had been reading the 17 GB number as evidence of pressure. It wasn't. 17 GB on a 24 GB card with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` is exactly where steady-state should sit. The card was happy. It went idle in a millisecond. That isn't allocator behaviour. Allocators *thrash* — they go slow, then slower, then they OOM. They don't snap from 93% to 0% utilization in the gap between two log lines. Strike two.

### Hypothesis 3: JAX compilation cache overflow

When you JIT-compile a function with `static_argnames`, every distinct combination of static arguments generates a new compiled artifact. If one of your "dynamic" arguments is actually varying — say, sequence length — XLA will cache one compiled version per length it sees. Over many epochs, with many distinct lengths, the cache can grow to hundreds of compiled programs, eating CPU memory (and eventually GPU buffer pool for kernel arguments).

This one was *promising*. I traced through `compute_optimal_path`:

```python
@partial(jax.jit, static_argnames=['k', 'lookahead'])
def greedy_lookahead_path(returns, k=4, lookahead=5, transaction_cost=0.001):
    T, n_assets = returns.shape
    # ...
    _, (all_masks, all_indices, step_pnls, step_changes) = jax.lax.scan(
        step, init_mask, jnp.arange(T)
    )
```

`T` is the episode length. Episode lengths vary across parallel envs because of truncation and early termination. With `--imprint`, `compute_optimal_path` gets called *twice* per env (once in `run_self_play_jax` for the optimal-alpha metric, once in `on_episode_complete_callback` for the imprint target). Eight calls per epoch in the worst case, each potentially a fresh shape, each a fresh compilation. Over sixty epochs, that's a lot of cached programs.

The `TASK_128b` writeup laid it out:

> *"With 4 parallel envs × 2 call sites = 8 compilations per epoch (worst case). Episode lengths vary (truncation, early termination). Over 60-90 epochs: potentially 500+ unique compilations."*
>
> — `tasks/TASK_128b_HANG_ROOT_CAUSE_ANALYSIS.md`

I padded `compute_optimal_path` to a fixed `MAX_EPISODE_LENGTH = 256` and recompiled. Ran the 200-epoch repro. It hung at epoch 84. Same shape — 0% GPU, 17 GB pinned, no log. The padding had reduced JIT-cache churn (I could see this in the trace logs), but it had not removed the hang. The compilation cache was a contributor at most, not the cause. Strike three.

### Hypothesis 4: device-to-host transfer deadlock

This was the first audit's primary suspect, the one I'd been carrying since `TASK_127r`. The story:

> *"If the GPU kernel fails silently (e.g., due to a rare numerical issue, driver timeout, or internal XLA error) but doesn't crash the process, the CPU thread waits indefinitely for the `fetch` result. [...] The hang appears to be a low-level JAX/XLA synchronization deadlock during the device-to-host transfer of self-play trajectories. It is not a logic bug in the Python code."*
>
> — `tasks/TASK_127r_hang_at_epoch_90.md`

Read that quote carefully. It is wrong in a specific and instructive way. It is *narratively* correct — the symptoms are consistent with a D2H transfer hanging — but it is *attributively* wrong. It attributes the failure to "JAX/XLA," to something below the Python layer, to a region of the stack where the appropriate fix is "restart and hope." That attribution is what stalled the investigation for two days. If you believe the bug is in XLA, you stop looking in your own code.

I wrapped the D2H boundary in `np.array(...)` calls with explicit `jax.block_until_ready()` and SIGALRM-based timeouts. The timeouts never fired before the hang. When the hang occurred, the SIGALRM did eventually trigger — and the traceback it produced showed the thread sitting not in `np.array`, not in `block_until_ready`, but in `threading.Lock.acquire()`. The transfer was finishing. Something on the Python side, after the transfer, was holding a lock.

That was the inflection. Up to that moment I had been chasing a low-level kernel/driver theory. The SIGALRM traceback put the hang squarely in Python.

### Hypothesis 5: GIL contention with the dataloader

JAX is async; PyTorch-style dataloaders aren't. If your dataloader spawns worker threads that hold the GIL while JAX is trying to issue kernels, you can in principle get contention that looks like a stall. We don't *have* a PyTorch dataloader — synthetic data is generated on-device inside `run_self_play_jax` — but I wanted to rule it out, so I ran with `--parallel-envs 1` to eliminate any worker-thread shenanigans. It still hung. Strike four.

### Hypothesis 6: NCCL deadlock

NCCL collective deadlocks are the classic "GPU idle, memory pinned, no traceback" failure mode in multi-GPU code. But this was a single-3090 workstation. There is no NCCL in this build. There is no second device for anyone to deadlock against. I wrote the hypothesis down so I could write it off; that is sometimes the only way to actually let it go. Strike five.

### Hypothesis 7: orbax checkpoint roundtrip

Checkpoints save just before the hang, every time. The pattern looks suspicious. Maybe orbax leaves something open — a file handle, an async writer thread, a mutex — and the *next* epoch's startup deadlocks on the same resource.

I disabled checkpointing entirely and ran 200 epochs. It hung at epoch 92. Checkpoints were not in the path of the bug. They were just adjacent to it in the log. Strike six.

### Hypothesis 8: the expert buffer lock

This was the second-favorite theory, behind the D2H deadlock. The expert buffer uses a `threading.Lock` to coordinate the imprint callback (which runs after an episode completes) with the sampling thread (which the trainer hits to build each minibatch). In `expert_buffer.py`, `sample_batch` looked roughly like:

```python
def sample_batch(self, batch_size=1024, demo_ratio=0.125):
    with self._lock:
        # ... sampling logic ...
        observations = self._encode_observations_vectorized(entries)
        return {...}
```

`_encode_observations_vectorized` calls `jax.vmap` and a JIT-compiled encoder *while holding the lock*. JAX may itself spawn helper threads. If the encoder ever blocks on something that requires the GIL plus this lock, you can get a cycle. The fix proposed in `TASK_128_TRAINING_HANG_FIX.md`:

> *"`_encode_observations_vectorized` function calls `jax.vmap()` and JIT-compiled encoders while holding `self._lock`. JAX may spawn its own threads or interact with the GIL in ways that conflict with the lock."*
>
> — `tasks/TASK_128_TRAINING_HANG_FIX.md`

I refactored to release the lock before the JAX call:

```python
def sample_batch(self, batch_size=1024, demo_ratio=0.125):
    with self._lock:
        entries = self.indices[indices].copy()
        weights = compute_weights()
    observations = self._encode_observations_vectorized(entries)
    return {...}
```

Cleaner. Probably correct. Did not stop the hang. Strike seven.

At this point I had ruled out NaN, OOM, JIT recompilation, D2H deadlock, dataloader contention, NCCL, checkpointing, and the expert-buffer lock. Each had a chapter of investigation behind it. None was the bug. I had been at this for the better part of four days.

The SIGALRM trace was still in my head, though. The hang was on a `threading.Lock.acquire()`. *Some* lock was being held by the thread that was trying to acquire it, or by a thread that had crashed without releasing it. The candidate set of locks in our codebase is small. The expert buffer lock was ruled out. What else holds a `threading.Lock`?

The answer, found about an hour after the SIGALRM revelation, was in `mcts_trader/jax_agent/price_cache.py`. The `PriceCache` is a CPU-side store that accumulates per-episode price tensors so that the imprint callback can compute oracle paths against them. It exposes `add_episode(episode_id, prices)` and an internal `flush()` that writes the accumulated buffer to disk every hundred adds. Both methods are wrapped in `with self._lock:`. The lock is created in `__init__` as:

```python
self._lock = threading.Lock()
```

And `add_episode` calls `flush` directly when the counter hits 100:

```python
def add_episode(self, episode_id, prices):
    with self._lock:
        # ... write to in-memory buffer ...
        self._flush_counter += 1
        if self._flush_counter >= 100:
            self.flush()  # <-- already holding self._lock

def flush(self):
    with self._lock:  # <-- tries to acquire self._lock again
        # ... write buffer to disk ...
```

`threading.Lock` in Python is **not reentrant**. When the same thread tries to acquire it twice, it blocks. Forever. There is no traceback, because nothing has crashed; the thread is healthily waiting for a lock it itself owns. CPU goes to zero, because the thread is in `pthread_cond_wait`. The GPU goes to zero, because nothing is queuing kernels — the imprint callback is wedged before it ever calls back into JAX. Memory stays pinned at 17 GB, because the previous epoch's tensors have not yet been freed.

And the trigger is *exactly every hundred add_episode calls*. With four parallel envs, that's every 25 epochs. With occasional callback early-returns (when alpha was negative the callback would return without adding to the cache, shifting the count), it averages out to "somewhere in the 60–90 range." Late, but not at a fixed epoch. The drift in the failure timing wasn't randomness — it was a counter, mod 100, modulated by callback skip-rate.

From the final report:

> *"`add_episode` acquires `self._lock` to write data, then checks if it's time to flush (`self._flush_counter >= 100`). If true, it calls `self.flush()`. The Deadlock: `self.flush()` also attempts to acquire `self._lock`. The Bug: `self._lock` was initialized as a standard `threading.Lock()`, which is not reentrant in Python."*
>
> — `tasks/TASK_128_FINAL_REPORT.md`

It was a reentrant-lock bug. The whole time. In a file none of the eight previous hypotheses had named.

## The fix

```python
# mcts_trader/jax_agent/price_cache.py
- self._lock = threading.Lock()
+ self._lock = threading.RLock()
```

One character, plus the `R`. `threading.RLock` is the reentrant variant of `threading.Lock`: the owning thread can acquire it any number of times without blocking, provided it releases the same number of times. The `add_episode → flush` recursion goes through cleanly. The hundredth episode triggers a flush; the flush takes the lock a second time, doesn't block, runs to completion, releases twice, returns to `add_episode`, which releases the outer acquire. The callback finishes. The trainer queues the next epoch's self-play. The progress bar moves.

I ran the 200-epoch repro three times. It finished, three times. The fourth time I let it go to 400 epochs out of paranoia. It finished too. Final-report verdict, unedited:

> *"The system is now stable for 100+ epochs. The 'Recompilation Bomb' and 'Thread Contention' hypotheses were ruled out by the definitive trace identifying the reentrant lock failure."*
>
> — `tasks/TASK_128_FINAL_REPORT.md`

That is the entire fix. Four days of investigation. One character of code. The unflattering accounting that comes with this kind of bug is that there is no clever insight to write down — the fix is *literally a known pattern* (use RLock when your locked method may call another locked method) — and the actual work was the elimination tree, not the change.

---

> *The gardener cleans the spring, reseats the coil, closes the case. The clock starts. The afternoon hour is no longer dead. The gardener writes in the ledger: it was the spring. Months of replanning lifted by a quarter-turn of brass. The ledger does not record the relief, only the cause.*

---

## Kickback: deterministic-but-unexplained, and the epistemology of silent failures

Two things kicked back in this episode and I want to name them both.

The first one is about the *category* of failure. There is a folk taxonomy of bugs roughly: (a) loud failures with clear tracebacks, (b) intermittent failures you can't reliably reproduce, (c) deterministic failures with no traceback. (a) is the easy case. (b) you mark "flake" and add a retry. (c) is the worst category, and it is the one I was in. It looks like (a) — it happens reliably, every overnight run — *and* like (b) — the epoch drifts, the cause hides — and it punishes you for both expectations. You can't anchor a debugger on it because the trigger floats. You can't dismiss it as flake because it kills every run. The investigative posture for (a) is "read the trace and fix the named thing." The investigative posture for (b) is "log everything and grep later." For (c) neither works. You need a third posture, and I didn't have one going in.

The third posture is *enumerate-and-eliminate*. You don't read traces; there aren't any. You don't grep for the failure; there's nothing to grep for. You build a hypothesis tree — every credible cause, named separately — and you knock them down one at a time with targeted instrumentation. Each eliminated hypothesis is progress, even though it doesn't fix anything. The tree itself is the artifact. The first time I drew the whole TASK_128 tree out on paper, with eight branches and a column for "ruled out by," was the day before I found the bug. It was the only way to keep track of where I was.

The second kickback is about *what counts as evidence*. Most of the debugging discipline I've internalized — and I suspect most of yours — was built handling failures that announce themselves. The traceback names a file. The exception type names a class of bug. The stderr line names the actor. You learn to read these announcements; you learn to trust them; you learn shortcuts based on them. Roughly: *symptom → search → fix*. You see, then you act.

Silent failures break the first step. There is no symptom to read. The progress bar stops; the GPU goes idle. That isn't a symptom in the diagnostic sense — it's the *absence* of one. You have to *manufacture* the symptom before you can read it. That's a different epistemology. *Instrument first, then reproduce, then look.* Inverting the usual order.

`TASK_127r`, the first writeup, captured this without naming it. It ended on:

> *"Restarting the training (which the user did) often resolves it."*
>
> — `tasks/TASK_127r_hang_at_epoch_90.md`

I read that line, originally, as a counsel of despair — accept the flake, restart. Reading it now, I think it was actually a counsel about evidence: the audit could not see anything, because there was nothing for an audit to see. The trace it inferred — "low-level JAX/XLA synchronization deadlock" — was a *plausible* trace, not an observed one. It was a story the system could tell about the absence. And as long as the story was external to our Python code, the right action was to wait it out. *If you believe the bug is in XLA, you stop looking in your own code.* That sentence was the most expensive belief in the entire chase.

The fix to the epistemology turned out to be SIGALRM. Not the lock change — the SIGALRM. Sticking a 60-second alarm around the suspected-hang region and having it raise a `TimeoutError` if the region didn't return is how you turn a silent failure into a loud one. The SIGALRM forces a traceback. The traceback names a frame. The frame names a file. Suddenly you are debugging (a)-class, not (c)-class. The price-cache file came up in the SIGALRM trace. Nothing else had pointed at it.

The general rule I'd write down for next time: *if a hang is silent, make it loud first*. Don't try to reason your way to the cause through hypothesis ranking. Insert a timeout. Force a traceback. The traceback will name the file. Then debug.

## What's left

The fix is in the code. `price_cache.py:_lock` is an `RLock` and has been since 2026-02-02. No epoch-N hang has been observed since. The 200, 400, and eventually 600-epoch runs that the project's later phases needed went through cleanly. Every result in `lore/notes_2026-05-*` rests, in part, on this change. None of them mention it. That feels right — the fix's job is to disappear into the substrate.

The instrumentation survived, which is the more interesting outcome. The `[HANG-DEBUG]` print-with-flush pattern in `train_jax_fast.py`, the `/tmp/hang_debug.txt` file-write that records the current epoch and phase, the SIGALRM timeout wrapping `run_self_play_jax` — all of it stayed in the code. It costs almost nothing in normal operation, and the day a different silent failure shows up, it will save another four days. The `/tmp/hang_debug.txt` file has been refreshed every epoch of every run for three months. Most of the time I never look at it. The two times since February that I've seen a stall — both turned out to be checkpoint-disk contention from a separate process — I read it first and the file told me which phase had been the last to write. The watchdog is what survived; the bug is incidental.

I have also, more lastingly, internalized the *category* lesson. When something hangs now, my first instinct is no longer "wait a bit longer, maybe it's slow." It is to attach `py-spy dump --pid <PID>` to the live process and read the Python stack of every thread. The stack tells you where each thread is right now, even if no exception has been raised. Half the time that's enough to skip the entire elimination tree. I should have started there on day one of TASK_128. I didn't, because at the time I believed the hang was below the Python layer. The belief was wrong. The belief cost three days.

`workstation` is still on. The desk fan is still on. The 3090 is currently warm from a 600-epoch Phase-2 run that finished overnight with no hang. The progress bar made it from `1/600` to `600/600` without stopping. In the next pane `nvidia-smi -l 5` is showing 31% utilization on a checkpoint validation. The price-cache `RLock` is in there, doing nothing visible.

There is a version of this article that would end on "and so I learned a deep truth about concurrency." There isn't one. The deep truth is: don't use `threading.Lock` in any class where a public method might transitively call another public method that takes the same lock. That's a sentence you can find on the Python docs page for `threading.Lock`. I knew it. I had read it. I had not applied it to my own code. The lesson, if there is one, is that *generic knowledge does not catch your specific bug* — every individual codebase has to be checked, by hand, against the patterns its components imply. The price cache had two locked public methods. One of them called the other. That was the bug. It was hiding in plain sight, sixty epochs deep, where no test ever ran long enough to find it.

The clock works. The afternoon is no longer dead. I do not love the brass coil for being small; I am grateful for it for being found.

---

> *The clock keeps time. The gardener keeps the ledger.*

---

## Sources

- `tasks/TASK_127r_hang_at_epoch_90.md` — first incident report; the D2H-deadlock misattribution.
- `tasks/TASK_128_TRAINING_HANG_BUGHUNT.md` — the full hypothesis space.
- `tasks/TASK_128b_HANG_ROOT_CAUSE_ANALYSIS.md` — JIT recompilation hypothesis, memory-not-the-cause finding.
- `tasks/TASK_128c_hang_debug_plan.md` — debug instrumentation phases.
- `tasks/TASK_128_TRAINING_HANG_FIX.md` — the expert-buffer lock fix that didn't fix it.
- `tasks/TASK_128_BUGHUNT_PROMPT.md` — the subagent prompt that re-framed the hunt.
- `tasks/TASK_128_FINAL_REPORT.md` — RLock root cause, fix, verification.
- `mcts_trader/jax_agent/price_cache.py` — the file containing the four-character fix.

For the broader debugging-discipline arc, see Article 5 (*22 Silent Killers*) on Phase 1's category-of-bug taxonomy, and Article 4 on Gemini-as-cold-eyes auditor — the same epistemic move (force evidence before reasoning) shows up in a different surface there.
