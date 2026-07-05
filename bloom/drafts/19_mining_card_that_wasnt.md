# The Mining Card That Wasn't a Mining Card

> *The mortar was made for grinding bark. I used it, for one winter, to fold honey into resin. It did the job. When I returned it to the bark, the old groove was still there — but the mortar carried, now, a faint sweetness no scrubbing could lift. A tool remembers being misused. It does not always forgive it.*

— Lukas, May 22 2026, `workstation`

---

I had a card sitting in the mining fleet that, on paper, was three quarters of a GeForce.

It was a **CMP 90HX**. The "CMP" stands for Cryptocurrency Mining Processor. NVIDIA built this line during the 2021 GPU shortage as a deliberate hardware fork: take the silicon they were already producing for the GeForce GA102 stack, *cripple it on purpose*, and sell the crippled version to miners so that the un-crippled version could go to gamers and ML people at MSRP. That was the theory. In practice it created a strange object: a card with the same physical die as an RTX 3080, manufactured in the same fab on the same week, that was forbidden by firmware and missing PCB components from doing anything but hash.

The 90HX I owned had spent its first life doing exactly that. Hashing Ethereum until The Merge, then Ravencoin, then Kaspa, sitting in a six-card rig with five siblings, fans at 70 %, breathing warm air into a room I do not heat in winter because the rig heats it. By early 2026 it had paid for itself twice and was, on the spreadsheet, a free GPU.

I had also, by early 2026, started building **Foxglove** — the supervised-imprint phase of the trading agent — and I had started running into the kind of problem you run into when you have one 3090 and several jobs that all want it. Bella, the market generator, wanted VRAM. The trainer wanted VRAM. The periodic benchmark wanted VRAM. They were taking turns. They were not, you might say, polite about it.

So one evening — `workstation` is on as I write this, the desk fan is on, `nvidia-smi` reports one device — one evening in February 2026, I looked at the spec sheet of the CMP 90HX and I started asking a stupid question. The silicon is the same. The drivers are the same. NVIDIA's "no ML" decree is, partly, a firmware promise and a missing-component truth. *Could I make it train?*

The answer, with hindsight, is **yes, badly, for three months, until the costs ate the gains**. This is the story of that arc. The card is gone from `workstation` now. It is back in the mining chassis where it started, hashing whatever it is hashing this week, doing what it was designed for. The morality of the loop does not survive the return trip. The mortar still carries the honey.

---

## What the CMP 90HX actually is

Before I describe the unlock, let me describe the object, because the object is the whole argument.

The CMP 90HX is built on the **GA102** die. That is the same die NVIDIA uses for the RTX 3080, the 3080 Ti, the 3090, and the 3090 Ti — the entire top of the consumer Ampere stack. The 90HX uses a specific binned variant (GA102-100-A1) with 6400 CUDA cores and 200 third-generation tensor cores enabled, where the 3090 has 10496 CUDA cores and the 3080 has 8704. *Fewer cores enabled*, not *fewer cores present*. The fuses are blown on the rest. You can prove this with a microscope; you cannot un-blow them with software.

The board is **PG132**, which is the same physical PCB as the RTX 3080. Same layout. Same mounting holes. Same VRM. Same memory controller layout for **10 GB of GDDR6X on a 320-bit bus** at about 760 GB/s — *identical* to a 3080's memory subsystem. From the heatsink down, the card and a 3080 are siblings.

What's different is what NVIDIA *left off* or *turned off*:

| Feature | RTX 3080 | CMP 90HX |
|---|---|---|
| Display outputs | 3× DP + 1× HDMI | **None.** Pads are unpopulated. |
| PCIe link | Gen 4 ×16 | **Gen 1 ×4.** Both width and speed locked. |
| Driver behavior | Standard GeForce | Driver refuses 3D acceleration; some ML workloads gated. |
| VBIOS | Signed GeForce VBIOS | Signed CMP-specific VBIOS with restricted link caps. |

The PCIe restriction is the cruelest of these. It is *two layers*. The firmware advertises a maximum link speed of 2.5 GT/s (PCIe Gen 1). The PCB is missing the AC coupling capacitors on lanes x4 through x15, so even if you could persuade the firmware to negotiate higher, the lanes physically cannot carry signal. NVIDIA mutilated the *substrate*, not just the *settings*. You would need to solder on 0402-size capacitors to even attempt width recovery, and people who have tried it on the 50HX cousin have reported that the firmware still refuses — the missing caps are belt-and-suspenders.

That gives you, in practice, **about 1 GB/s of bus bandwidth** between the card and the rest of the system. A 3090 in the same slot would have 32 GB/s. The 90HX is, by design, slow at the only thing data-parallel training really cares about: moving gradients across the bus.

That is the object. Now to the unlock.

---

## TASK_688 — what could be unlocked, what could not

I wrote `TASK_688_CMP90HX_UNLOCK_AND_INTEGRATION.md` over two evenings in February 2026. It is unusually long for a task spec because it is half plan, half lab notebook, half hardware-mod survey. *(That is three halves. The task was that kind of task.)* Three phases were sketched:

- **Phase 1**: install the card, get the drivers seeing both GPUs, write a thin device-routing layer so I could pin specific JAX computations to specific devices. **Software only. Zero risk.**
- **Phase 2**: try to unlock the PCIe restriction. *Two methods.* First, `setpci` runtime hacks to renegotiate the link speed — costs nothing, almost certainly fails because the firmware will refuse. Second, the punk move: **cross-flash an RTX 3080 VBIOS** onto the 90HX using `nvflashk`, a tool that bypasses NVIDIA's board-ID check by toggling an internal override they ship inside the flash routine itself. Same PCB, same memory, signed VBIOS — *might* boot, *might* unlock PCIe Gen 3, *might* permanently brick if the GPU variant code (100 vs 200) was load-bearing somewhere I didn't see.
- **Phase 3**: gradient-accumulation and model-parallel training across the two cards, but **only if Phase 2 succeeded**.

I went through Phase 1. I never seriously attempted Phase 2.

The reason I never seriously attempted Phase 2 is the same reason I should not have attempted Phase 1: **the bandwidth math.** Let me show it, because this is the moment the arc was decided and I did not notice at the time.

A typical training step on Foxglove moves roughly 100 MB of gradient state across the device boundary if you run data-parallel. At 1 GB/s (PCIe 1.0 x4), that is **100 milliseconds of pure bus time per step**. A forward+backward on the 3090 takes about 80 ms. The transfer would dominate the step. You'd end up with a setup where the 3090 spent more than half its time waiting for the 90HX to be done sending its gradients home. Data-parallel was *off the table* before I plugged the card in. I had a 27-line section in the task spec listing the bandwidth budget and I had highlighted it, and I went and installed the card anyway, because *the silicon was free*.

This is the lesson of the article and I will earn it back later. For now: I installed the card.

What I told myself the 90HX *could* do, given the bandwidth ceiling, was the **producer / consumer split**:

1. **Bella sideloading.** The market generator (Bella) runs once per training batch to produce a synthetic OHLCV trajectory. The output is small — a `(400, 64, 5)` float32 tensor, about 512 KB per episode, ~16 MB per buffer fill. At 1 GB/s, that's a 16-millisecond transfer; negligible. Bella could live on the 90HX, generate, and ship the result to the 3090 *between* training steps.
2. **The periodic benchmark.** Once every 100 epochs the training loop runs a benchmark on held-out periods. That benchmark wants a GPU, and the 3090 is busy training. Park the benchmark on the 90HX, run it asynchronously, post results back as a CSV. *Independent process, no bus contention with the training step.*
3. **The dual-dojo split.** This was the one I was actually excited about. Run two training branches in parallel — Branch A on the 3090 with batch=512 (exploitation, low noise, fast convergence), Branch B on the 90HX with batch=128 (exploration, more noise, regularization), and periodically *merge their weights*. Two ML training runs, same model, different hyperparameters, no synchronous gradient transfer required because the merging is async.

All three of these have the property that **the bus is touched rarely and in small bursts**. None of them required the PCIe unlock. The 90HX, in these roles, was a perfectly serviceable second device. *In theory.*

---

## The integration — `CUDA_VISIBLE_DEVICES` and the device-placement dance

The card went in on a Saturday. The PSU was the 1600 W brick I'd over-bought a year earlier for exactly this situation (the 3090 draws 350 W, the 90HX 320 W, the rest of the system ~150 W; 1600 W was overkill, which is correct for sustained training loads where you do not want to be anywhere near the rail).

`lspci` saw both. `nvidia-smi` saw both. `jax.devices()` returned a length-2 list. The first time I called `jax.devices()[1]` and put a 4 MB tensor on it, the second card's fan stepped from idle to audible and I felt, briefly, like an electrical engineer.

The first integration was the dual-dojo orchestrator. Two `train_easy.py` processes, each pinned to a different GPU, writing checkpoints to a shared directory, with a thin Python wrapper that merged their weights every N epochs:

```python
# tools/orchestration/dual_dojo.py
branch_A = subprocess.Popen(
    ["python", "train_easy.py",
     "--branch", "A", "--batch-size", "512",
     "--ckpt-dir", "dual_dojo/branch_A"],
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},  # 3090
)
branch_B = subprocess.Popen(
    ["python", "train_easy.py",
     "--branch", "B", "--batch-size", "128",
     "--ckpt-dir", "dual_dojo/branch_B"],
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},  # 90HX
)
```

The orchestrator itself ran CPU-only — `CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu` — so it could not accidentally claim VRAM from either branch. *Zero VRAM theft from the boss process.* This pattern, of forcing the parent to CPU and letting children claim GPUs by index, became one of the genuinely good things to come out of the whole episode and survived the card's removal.

The first dual-dojo run I let go for 414 epochs. The headline number from the lore notes (2026-02-21) was **97.8 % agreement** between the two branches by the end — the highest I had seen on any synthetic data run, by a comfortable margin. Branch A on the 3090 led; Branch B on the 90HX lagged by about three percentage points; the merge step pulled them together each epoch and they walked, hand in hand, toward a shared optimum. I wrote in the notes that day, with some excitement, that **batch size asymmetry was a feature, not a tax** — the noisy 128 branch was acting as a regularizer on the 512 branch, and the merged checkpoint outperformed either parent.

It looked, for about a week, like the CMP 90HX had earned its place in the chassis.

The benchmark sideload landed a few weeks later in TASK_706. Same pattern: `--benchmark-gpu 1` flag in `train_easy.py`, the benchmark process forked to a child that pinned itself to `CUDA_VISIBLE_DEVICES=1`, the parent kept training on `0`. Lore note from 2026-03-11: *"Benchmark runs every 100 epochs on CMP 90HX (CUDA 1), non-blocking. Training continues on RTX 3090 (CUDA 0) uninterrupted."* This worked. It worked for two months.

The Bella sideload, the third leg of the stool, never quite went in. Bella's compile graph kept changing as the encoder did, and pinning her to the 90HX would have required maintaining a second device-placement path through a piece of code that was rewritten three times between February and April. I put it in a queue. The queue is still there. The card came out before the queue was processed.

---

> *Three pots on three burners. The blue flame is steady; the red flame is uneven and the third flame is, the apothecary suspects, not actually burning so much as smoking. She lowers the heat on the smoking one. The kitchen is louder than the recipe required.*

---

## The kickback, in three layers

This is the part of every article in this series where I tell you what reality did to the plan. There are three layers and they are not independent.

### Layer 1: the GPU-index mismatch

In CUDA, devices are numbered. In Linux, PCI devices are also numbered, but the numbering rules are different. The CUDA driver enumerates GPUs roughly by *PCI bus order*, which depends on which slot you plugged into; the Linux PCI bus numbers roughly by *slot priority* on the motherboard, which depends on which slot was advertised first by ACPI; and `nvidia-smi`'s order depends on a third thing, which is the order the kernel module noticed them at boot. **All three of these agreed most of the time.** That was the problem. *Most of the time*.

The conditions under which they disagreed:

- **Cold boot vs warm boot.** Cold boot enumerated the 3090 first; warm boot (after a soft reset) occasionally enumerated the 90HX first because it had been in a deeper power-save state and answered ACPI later.
- **Monitor present at boot vs not.** If the monitor was off when the machine booted, the firmware sometimes decided the 3090 (which has display outputs) was not the primary display device, and the 90HX got CUDA index 0 by default.
- **The 90HX's own self-throttling.** The CMP cards have aggressive power-save behavior because they're built for mining rigs where you want them sleeping when not hashing. If the card was deeply asleep at PCI probe time, the order could shift.

The failure mode of this, when it happened, was that **the training run would start on the wrong card**. `CUDA_VISIBLE_DEVICES=0` would silently mean "the 90HX" and the 3090 would sit idle. You would not notice for half an hour, because the loss curves looked normal — Foxglove fits in 10 GB at batch=128 — and only when you opened `nvidia-smi` to check the throughput would you see that the 3090 was at 13 MB and 0 % utilization and your run was crawling on the wrong silicon.

I wrote, sometime in March, a startup-time assertion in the trainer that read the device name from `jax.devices()[0].device_kind` and crashed loudly if it did not match the expected string. That helped. It did not fix the underlying issue, which was that **`CUDA_VISIBLE_DEVICES=0` was ambiguous in my hardware configuration in a way that could not be made unambiguous from inside the codebase**. The bug never went away. It would surface every other week, cost me an hour of confused stares at `nvidia-smi`, and recede again. It was a system bug, not a code bug. The code could only learn to *detect* it, not *prevent* it.

### Layer 2: the acoustic kickback

This one is mundane and I am almost embarrassed to write it, because it has nothing to do with software, but it is the one that finally drove the decision to remove the card.

Two GPUs under sustained load with stock cooling produce noise that compounds non-linearly. The 3090 alone, at 70 % fan, is a steady whoosh — present but ignorable, the kind of noise you adapt to in an afternoon. The 90HX alone, at 70 % fan, is the same kind of whoosh but pitched higher because the fans are smaller. *Together*, with both running training-load duty cycles, the sound is somewhere between a hairdryer and a small vacuum cleaner. There is no harmonic relationship between their fan curves; they beat against each other.

`workstation` lives under my desk. I work at this desk. By April I had developed a habit of putting on over-ear headphones not to listen to anything but to muffle the workstation. Around the same time I started noticing that when I ran the dual-dojo orchestrator, I would *leave the room* during the run, and *come back* when the run finished, instead of sitting next to the machine and watching the loss curves the way I used to. The noise was, quietly, reshaping how I worked.

I want to flag this as a *real cost* and not a comfort complaint. The reason I sit next to the workstation while it trains is that **the fastest feedback loop in this whole project is me watching a curve and noticing it doing something weird**. If I am not in the room, I am not noticing. The acoustic cost was not just unpleasant; it was *epistemically expensive*. I was getting fewer hours of attention per training day because the room had become hostile to attention.

### Layer 3: the integration-cost-vs-silicon-cost ratio

Both of the above are surface symptoms of the deeper kickback, which is the lesson I will carry forward from this whole arc.

**The silicon was free. The integration was not.**

Let me itemize the integration costs I actually paid, in the order they came up:

- *Two evenings* writing the task spec and the unlock survey.
- *A weekend* installing the card, routing the PCIe slot, sorting out the power cables.
- *About a week* of iterating on the dual-dojo orchestrator until the merge step was correct.
- *Roughly an hour every other week* on average, from late February to mid-May, paid out to the GPU-index mismatch.
- *Several days* on a `setpci` link-renegotiation experiment that, exactly as the spec had predicted, did nothing.
- *Some unknowable but non-zero fraction* of my attention budget, displaced by the noise.
- *Several days* of code complexity in `tools/DSe/server.py` to handle the dual-branch CSV patterns, dual-log-dir discovery, dual-checkpoint resolution — code that *still exists in the repo* and *still has to be maintained* because removing it would mean a chunky cleanup PR I have not yet prioritized.

Set against this, the benefits I actually realized:

- The dual-dojo run that hit 97.8 % agreement. (Real and good. Did not, in retrospect, replicate at the same level on later codebases; the architecture had drifted by April.)
- The periodic benchmark sideload — saved roughly 5–10 % of training time over a few weeks.
- A device-placement discipline in the codebase that *did* turn out to be a permanent improvement.

A hardline accounting: the integration cost was, in hours, *substantially larger* than what I would have paid simply queuing the benchmarks behind the training runs on the 3090. **A second 3090** would have been better, but a second 3090 in February 2026 was around $900 used and I did not want to spend $900 on a problem that, structurally, was about queuing. The CMP 90HX was *cheaper than buying a second 3090* and *more expensive than not having a second GPU at all*. There is a third category I had not given a name to, and I have given it a name now: **negative-value cheap hardware**. The kind that costs less than the obvious alternative and more than the patient alternative, and the difference is taken out of you in hours.

> *— Lukas, after watching the third index-mismatch of the week, on the chat: "the card has to come out. it's an hour a fortnight and I cannot hear myself think."*

---

## The removal — May 17, 2026

I pulled it on a Sunday afternoon. Power off the workstation, monitor unplugged, side panel off, two 8-pin PCIe cables disconnected from the 90HX, two thumbscrews on the back of the case, lift the card out, slot the slot cover back. Put the card on the antistatic mat. The whole removal took eleven minutes. The card sat on my desk for a week. Then it went back into the mining chassis on the other side of the room, where five of its siblings had been waiting.

The PSU stayed. I left the 1600 W brick in the case because the workstation does not consume meaningfully more electricity at idle for being over-spec, and because *if* I ever do find a use case for a second card that survives the integration-cost-vs-silicon-cost test, I will not want to be replacing the PSU at the same time.

`workstation` is now, again, a single-GPU machine. `CUDA_VISIBLE_DEVICES=0` is, in the cosmology of this workstation, exactly equal to "the only GPU." There is no ambiguity. The startup assertion I wrote in March still runs and now passes trivially. The noise is back to one fan curve, one whoosh, present but ignorable. I have started sitting at the desk during training runs again. I have started noticing things in the loss curves I had stopped noticing.

The dual-dojo code still works, in the sense that it would still launch a Branch A and a Branch B if I asked it to — but Branch B has nowhere to go, so the orchestrator now refuses to start with a clear error message. The CSV-discovery code in DSe still handles the two-branch case. The device-placement glue in `train_easy.py` still respects the `--benchmark-gpu` flag. *None of this is being deleted.* It is dormant. If a second GPU ever earns its way back in, the scaffolding is there. Some scars don't heal; some are kept as load-bearing structure, like a buttress on a building whose original wall has been removed.

The 90HX, meanwhile, is back to hashing. It is doing what it was designed for. It is, by the metric it was built against, a competent piece of hardware. NVIDIA's deliberate cuts — the missing display outputs, the PCIe Gen 1 ×4 lockdown, the driver gating — are *features* in the mining role. The fan curve I cursed for three months is, in the mining chassis, drowned out by five identical fans and ignored. The card is happy. The card was always going to be happy here.

---

## What's left

Three things survived the removal, and they are worth naming because they are the actual delta to the project that the whole arc produced.

**Device-placement discipline.** The codebase is more explicit about which tensor lives on which device than it would have been if I'd only ever had one GPU. The pattern of *force the orchestrator to CPU, pin children by `CUDA_VISIBLE_DEVICES`, assert the device kind at startup* is the right pattern even on a single-GPU system — it makes the implicit explicit, and it makes ambiguous configurations crash loudly instead of silently running on the wrong device. This is a free upgrade.

**A clean rule for repurposed hardware.** I now have a check I run before saying yes to integrating any "free" silicon: **what is the integration cost in hours, what is the alternative cost in dollars, and which is bigger?** The CMP 90HX cost about $0 in silicon and about 40 hours in integration and another 30 hours in the recurring GPU-index tax. At my consulting rate, those 70 hours were worth several times what a used second 3090 would have cost. The rule is not "always buy new hardware" — sometimes the integration is cheap and the silicon is genuinely free. The rule is **count both columns before you commit**. *I had not counted both columns.* I had looked at the silicon column and skipped past the integration column. The phrase "negative-value cheap hardware" is in my vocabulary now, and it has earned its space.

**A more honest map of where bandwidth matters.** The 90HX taught me, by being too slow at the wrong thing, exactly which parts of the training loop are bus-bound and which are not. Gradient transfer is bus-bound. Bella-style data generation is not. Benchmark sideloading is not. Checkpointing is not. Even on a single GPU, knowing which operations *would* have been bottlenecked on a dual-GPU setup has changed how I think about scheduling — what to fuse into a single device-resident kernel and what is safe to split across host and device. The 90HX is gone but the bandwidth map it forced me to draw is still in my head.

The mining card is doing what it was designed for. The workstation is quieter. The codebase is more explicit. **Net win**, on a long enough timeline, even after the hour-a-fortnight tax.

But the mortar still carries the honey. There is, in the `tools/orchestration/` directory of this repo, code that *only makes sense* in a two-GPU configuration that no longer exists. There is, in the lore notes, a week in late February when I genuinely believed the CMP 90HX had earned its place. There is, in my head, a slightly more skeptical posture toward "free silicon" arguments that I did not have before. The card came back, the moral did not survive the return trip, and the workstation is, as I write this — fan at low, single 3090 at 619 MB idle, the desk fan louder than the GPU fan — a quieter and more honest place than it has been in three months.

> *The mortar is back at its bark. The honey ghost is still there. The apothecary records the contradiction and continues.*

---

## Sources

- `~/Trader/tasks/TASK_688_CMP90HX_UNLOCK_AND_INTEGRATION.md` — the primary task spec, all three phases, the bandwidth math, the punk VBIOS cross-flash plan that I never executed.
- `~/Trader/lore/notes_2026-02-20.md` — installation planning, PCIe slot configuration, dmesg watch.
- `~/Trader/lore/notes_2026-02-21.md` — dual-dojo first run, 97.8 % agreement, the moment the card seemed to have earned its place.
- `~/Trader/lore/notes_2026-03-11.md` — TASK_706 periodic-benchmark sideload, `--benchmark-gpu 1`.
- `~/Trader/lore/notes_2026-04-23.md` — the last good idle snapshot, both GPUs cool, before the removal decision.
- `~/Trader/bloom/drafts/00_before_alphago_had_a_name.md` — for the hardware-stack frame and the workstation-only opsec rule this article inherits.
- `~/Trader/CLAUDE.md` & MEMORY note: *"CMP 90HX removed 2026-05-17; previously the 10GB second GPU."* The line that prompted the article.
