# Turtle Breaker: Oracle Temperature Sharpening

> *Observation: ten apothecaries are asked, each in their own room, to write down a dose for the same patient. Seven write the same measure; three write a slightly smaller one. The gardener collects the slips and computes the mean. The mean prescribes neither dose — it sits between the two real recommendations, true to no one. The gardener does not throw away the slips. The gardener sharpens the average toward the majority, leaves the rare unanimous fifty-fifty untouched, and prescribes the dose that seven of them actually wrote down.*

---

It is the morning of May 22, 2026. `workstation` is on; the desk fan is at its low setting; the single EVGA 3090 is idling at thirty-nine degrees with `nvidia-smi` ticking through a checkpoint validation in the next window. The editor is open to `mcts_trader/jax_agent/path_solver.py` at line 322 — the line where Oracle 3, the multipass ensemble, averages ten one-hot policies into one soft policy. Right after that line, in the version of the file that lives in TASK_677, there is a small block of code that does something subtle and load-bearing: it sharpens the average back toward a peak, *without* breaking the symmetric case where the average should remain uncertain.

That block is Turtle Breaker. It is six lines long. It does not move deployment Alpha. The math is correct. The name is a pun. The lesson, as with most of what this notebook has produced in the last six months, is that a cleaner training signal does not automatically buy a cleaner trader.

This is Article 31. It is a companion to Article 17, *The Five Oracles*. Article 17 walked the lineage of the supervised target from end to end — five oracle reformulations, none of them moving deployment Alpha, all of them still in the codebase. Turtle Breaker is not a sixth oracle. It is a small surgery on Oracle 3, the multipass ensemble. It operates *after* the averaging — taking the soft distribution that ten passes produced and re-sharpening it before handing it to the loss. The motivation is one paragraph long. The fix is one paragraph long. The math is honest. The kickback is the same kickback the series keeps recording.

## What I was looking at

Oracle 3 — TASK_673, dated mid-February — is the multipass oracle. Article 17 walked it in detail; I will not redo that work here. The relevant fact: at training time, for each (T, n_assets) episode of log returns, the oracle runs ten noise-perturbed solves of `greedy_lookahead_path`, converts each binary path to a per-asset one-hot policy via the state-aware transition table from Oracle 2, and averages the ten one-hot policies into a single soft policy. Shape: `(T, 320)` where 320 = 64 assets × 5 actions per asset.

The averaging is the point. When five different oracle passes pick five different (equally good) assets, the soft target says *any of these five was acceptable* rather than *exactly this one was correct.* That preserves the cross-sectional ambiguity that a single-pass oracle erases. It was the right cut, and it broke the project's 56% agreement plateau by a few percent.

What I was looking at, in February, was a different failure mode of the same target. When seven of the ten passes agreed on BUY for asset 7, and three passes said HOLD for that same asset, the averaged target for asset 7 was `[0, 0.3, 0.7, 0, 0]` — seventy-percent BUY, thirty-percent HOLD, zero on the other actions. The TRM, training against this with softmax cross-entropy, would see a target that *looked ambiguous* and would not commit. The classifier head learned to hedge. The model's BUY logit on asset 7 climbed to roughly the same height as its HOLD logit. At inference, argmax sometimes picked HOLD; sometimes BUY; the action signal was muddy in exactly the cases where the oracle's actual *majority recommendation* was clear.

The mismatch was: *seven out of ten apothecaries said BUY*. The averaged target said: *seventy-percent BUY*. The model read that as: *it is unclear which one of two actions is correct here*. The signal was being lost in the average — not because the average was wrong (it is correct), but because the model's loss function consumes peakedness as a confidence channel, and peakedness is exactly what averaging destroys.

The fix needed to do one thing and not break a second thing. It needed to restore the model's read of "this was a majority recommendation" *without* false-confidence-ing the genuinely-ambiguous case where five passes said BUY and five said HOLD. That case really is uncertain; the soft target should reflect that.

## Why multipass averages in the first place

Before walking the fix, one paragraph on why the average is there at all. Each single oracle pass — even with the state-aware transition table from Oracle 2 — is noisy. The greedy lookahead has a horizon of 20 steps, transaction costs in the threshold, and a top-k tie-breaking rule that can flip on small return perturbations. Two of the project's harder failure modes — bull-regime under-trading, and cross-sectional coin-flips between similar assets — come from the oracle making sharp commitments in regions where the underlying value function is flat. The averaging is variance reduction: by running ten noisy versions of the same oracle and averaging the resulting one-hot policies, the soft policy captures *the distribution of acceptable answers* rather than *one arbitrary acceptable answer*.

But variance reduction has a cost, and the cost is exactly peakedness. The mean of ten near-one-hot distributions is, by construction, blurrier than any single one of them. This is not a quirk of the implementation. It is a property of averages under Jensen's inequality, which is the math I want to do now.

## The math

Let me write down what is actually happening, because the asymmetry between *averaging* and *sharpening* is the whole content of TASK_677.

Each oracle pass `i` produces a per-asset policy. Inside `optimal_path_to_policy`, this is a softmax over per-asset logits with temperature `T_i`:

$$\pi_i(a \mid s) = \mathrm{softmax}(z_i(s) / T_i)$$

Where `z_i(s)` is the per-asset logit vector from the state-aware conversion (about `+5.0` for the chosen action, `−10.0` for the wrong actions, gated by the open/close/maintain transition rules). `T_i` is small inside `optimal_path_to_policy`, so each per-pass policy is *near one-hot* — typically `[0, 0.0001, 0.9999, 0, 0]` or similar. Sharp, by construction.

Now ten of those policies get averaged:

$$\bar{\pi}(a \mid s) = \frac{1}{N}\sum_{i=1}^{N} \pi_i(a \mid s)$$

This is not the same as taking the softmax of the *averaged* logits. If you averaged logits first and then took a softmax — `softmax((1/N) Σ z_i / T)` — you would get a different distribution. Averaging *post-softmax* and averaging *pre-softmax* are different operations. The path solver does the post-softmax version: it runs ten independent softmaxes and averages the resulting probabilities. Each pass commits before the averaging happens.

What does the post-softmax average look like? If seven passes put their mass on action 2 (BUY) and three put their mass on action 1 (HOLD), then:

$$\bar{\pi} \approx \frac{1}{10}\left(7 \cdot [0, 0, 1, 0, 0] + 3 \cdot [0, 1, 0, 0, 0]\right) = [0, 0.3, 0.7, 0, 0]$$

The averaged distribution is *less peaked* than any individual pass. Quantitatively, by Jensen's inequality applied to the concave entropy functional:

$$H(\bar{\pi}) \geq \frac{1}{N}\sum_{i=1}^{N} H(\pi_i)$$

The entropy of the average is at least the average of the entropies. *The mean is blurrier than the average peakedness.* This is not a bug of the multipass scheme; it is a defining property. The whole point of averaging is to surface uncertainty. The cost is that *legitimate uncertainty* and *consensus blur* are encoded in the same channel — the entropy of the soft target — and the model cannot distinguish them.

The fix, the part of TASK_677 that is mine to claim, is: apply a temperature sharpening *after* the averaging, with the right invariants. Take the averaged soft target, run it through one more softmax-with-temperature pass, and emit the sharpened distribution as the actual target the loss sees:

$$\pi^{*}(a \mid s) = \mathrm{softmax}\!\left(\log \bar{\pi}(a \mid s) \,/\, T_{\mathrm{sharpen}}\right)$$

The `log` undoes the original softmax-space → probability-space transformation, putting us back in logit space. Dividing by `T_sharpen < 1` amplifies the differences between logits. The outer softmax renormalizes. The composition acts on a probability vector and returns a sharpened probability vector. It has three properties I needed it to have.

**Property 1: peaked stays peaked.** If `π̄` is already near one-hot — say `[0, 0, 0.9999, 0, 0]` — then `log π̄` has one large finite entry and several large negative entries (`log(ε) ≈ −18` for `ε = 1e-8`). Rescaling by `T_sharpen` and re-softmaxing leaves the result near one-hot. Sharpening is idempotent on already-peaked targets; the case where ten of ten passes agreed sees no change.

**Property 2: symmetric stays symmetric.** If `π̄ = [0, 0.5, 0.5, 0, 0]` — five passes said HOLD, five said BUY — then `log π̄` is `[−∞, −0.69, −0.69, −∞, −∞]`. Two entries equal, the rest at floor. Dividing by any positive `T_sharpen` preserves equality. The softmax of two equal entries with floor elsewhere is still `[0, 0.5, 0.5, 0, 0]`. *The genuinely ambiguous case is left alone.* This is the invariant that makes the surgery surgical. Sharpening cannot manufacture confidence where the passes themselves disagreed evenly. The fifty-fifty stays fifty-fifty by symmetry, not by hyperparameter tuning.

**Property 3: majorities get amplified, with the strength controlled by `T_sharpen`.** At `T_sharpen = 1.0`, the sharpening is the identity transformation (`softmax(log π̄)` is just `π̄`). At `T_sharpen → 0`, the sharpening collapses to argmax — the majority class gets all the mass. In between, there is a continuous knob. At `T_sharpen = 0.1`, a 70/30 split becomes approximately 99.99/0.01. The math: with logits `log(0.7) = −0.357` and `log(0.3) = −1.204`, the difference is `0.847`. Dividing by `0.1` gives a difference of `8.47`. The softmax of a logit-pair with gap `8.47` is `(1 − 0.0002, 0.0002)` — almost a one-hot.

Here is the comparison table from TASK_677, verbatim:

| Target | Before (temp=1.0) | After (temp=0.1) |
|--------|-------------------|-------------------|
| 7/10 BUY, 3/10 HOLD | `[0, 0.3, 0.7, 0, 0]` | `[0, 0.0001, 0.9999, 0, 0]` |
| 5/10 BUY, 5/10 HOLD | `[0, 0.5, 0.5, 0, 0]` | `[0, 0.5, 0.5, 0, 0]` (symmetric, unchanged) |
| 10/10 HOLD | `[0, 1.0, 0, 0, 0]` | `[0, 1.0, 0, 0, 0]` (already peaked) |

— *`tasks/TASK_677_TURTLE_BREAKER.md`*

The asymmetry is the whole feature. The 70/30 collapses; the 50/50 holds; the 100/0 is unchanged. The sharpening operates *only* on the cases where the multipass averaging introduced blur that does not correspond to actual oracle disagreement. The cases where the oracle really did split evenly — where the right answer is "this is ambiguous" — are preserved by the symmetry argument, not by tuning the temperature carefully.

One more piece of math worth saying out loud. The composition `softmax(log(·) / T)` is, in the limit `T → 1`, the identity on the simplex. In the limit `T → 0`, it is the argmax indicator. The intermediate regime is a continuous family of distributions on the simplex that interpolates between identity and one-hot, parameterized by `T`. This family has been used elsewhere in the literature — knowledge distillation, semi-supervised consistency losses, MixMatch's sharpening operation in the 2019 paper, all the way back to Hinton's distillation work. The novelty here is not the operator. The novelty is *applying it after a multipass average*, where the entropy of the average is doing work that does not match what the cross-entropy loss reads it as doing.

I want to be careful to claim this rather than over-claim it. The math is standard. The position in the pipeline is the contribution. Sharpening per-pass would do nothing — each pass is already peaked. Sharpening before the averaging would erase the multipass uncertainty signal entirely. Sharpening *after* the averaging is the only position that preserves the multipass logic while restoring the loss function's read of consensus.

## The implementation

The code that lands in `path_solver.py` is short. Six lines, sitting immediately after the averaging step at line 322. The task spec writes it as:

> *"In multipass_oracle_policy, after line 322:*
> *soft_policy = jnp.mean(all_policies, axis=0)  # (T, 320) — current*
>
> *NEW: Sharpen the averaged soft targets*
> *Reshape to (T, 64, 5) for per-asset sharpening*
> *soft_reshaped = soft_policy.reshape(T, n_assets, n_actions_per_asset)*
> *Log + temperature rescale + softmax (per-asset)*
> *sharpened = jax.nn.softmax(jnp.log(jnp.maximum(soft_reshaped, 1e-8)) / sharpen_temperature, axis=-1)*
> *soft_policy = sharpened.reshape(T, n_assets * n_actions_per_asset)"*

— *`tasks/TASK_677_TURTLE_BREAKER.md`*

A few things worth saying about that block.

The `jnp.maximum(soft_reshaped, 1e-8)` is a numerical floor. Without it, `log(0)` is `−inf`, and `−inf / T_sharpen` is still `−inf`, and propagating `−inf` through the softmax is fine in exact arithmetic but produces NaN gradients under bf16 mixed precision. The floor sits at `1e-8`, which corresponds to a logit of `−18`. After dividing by `T_sharpen = 0.1`, that floor is `−180`, which is comfortably below anything else in the distribution — the softmax assigns it effectively zero mass, the gradient is well-defined, and the bf16 dynamic range is intact.

The reshape to `(T, 64, 5)` is per-asset sharpening. The model treats each asset's five-way action distribution as an independent classification head; the loss is per-asset softmax cross-entropy summed across assets. Sharpening per-asset is the matching operation — sharpening across all 320 entries together would mix asset boundaries, which would be wrong. The reshape costs nothing on a GPU (it's a view in JAX) and keeps the math local to each asset's five actions.

The softmax-of-log structure is the cleanest expression. An equivalent form is `(soft_reshaped ** (1/T_sharpen)) / sum(soft_reshaped ** (1/T_sharpen))`, the standard MixMatch-style sharpening; it has the same fixed points and the same limit behavior. I went with softmax-of-log because it composes more obviously with the rest of the JAX softmax stack, and because `jax.nn.softmax` is the numerically-stabilized version that handles the subtract-the-max trick internally. The two forms are mathematically equivalent on the interior of the simplex; on the boundary, softmax-of-log needs the floor (see above), and the power form needs a `max(·, ε)` for the same reason.

The performance cost is negligible. One softmax over a `(T, 64, 5)` tensor per oracle call. JIT-friendly — `sharpen_temperature` becomes a `static_argname` on the multipass JIT so the compiled kernel is specialized to the configured temperature. On `workstation`, oracle compute is around 15 ms per call for the double-vmap'd batched version; the sharpening adds about 0.3 ms. Against a per-step budget of ~100 ms for the whole episode generation pipeline, the cost is invisible. There are no extra rollouts, no extra noise draws, no extra anything — the sharpening reads the same averaged target the loss was about to consume and rewrites it in-place.

The integration into the training loop is a single config field, `oracle_sharpen_temperature: float = 0.1`. The default is `0.1`, but the operator can dial it to `1.0` to disable, or to `0.5` for partial sharpening, or to `0.05` for aggressive. The flag is plumbed through `batched_multipass_oracle_policy` and added to the JIT's `static_argnames`. No model change. No loss-function change. The encoder, planner, value head, rank head, mask head — everything downstream — sees a marginally sharper soft target and otherwise nothing.

## Apothecary's interlude

> *The Apothecary takes the slips from the ten apothecaries and sets them in two stacks: agreement and disagreement. The agreement stack is thick. The disagreement stack is thin. The gardener marks the mean dose against the patient's chart, then strikes through it and writes the majority dose beside it — not because the mean was wrong, but because the patient cannot drink a number that no one prescribed. The two-stack heap remains beside the ledger. It is the record that there was disagreement, and the record that there was also a majority. The patient receives the majority dose. The patient's response will tell us, eventually, which apothecary was correct.*

---

## The kickback, in three layers

The kickback in this article is the same one in Article 8 and Article 6 and several others before. It comes in three layers, each one a different scale of the same observation.

**The first layer is the name.** Turtle Breaker is a pun. The multipass oracle is *turtle-slow at extracting signal*: it averages ten passes, each of which is already sharp, and produces a conservative, blurrier distribution that hugs the safe centre of the simplex. The model, trained against that conservative distribution, learns to be conservative — it hedges, it under-commits, it under-trades exactly in the cases where the oracle's majority recommendation was clear. The averaging is the turtle. The sharpening *breaks the turtle*. The codename matches the function: the operator is named for the conservatism it dispatches with. I admit, looking back at the commit log, that I named it before I had checked whether it would help deployment Alpha — naming things before they prove their value is a habit I have not yet broken. (Article 7 walked the same habit at larger scale: TASK_737 was given a champion's name before five seeds revealed it as a noise bump.)

**The second layer is that sharpening helps, but the help does not transfer to deployment Alpha.** The ablation that ran on Turtle Breaker showed, predictably, that the BC training loss curves came down faster with `sharpen_temperature = 0.1` than with `sharpen_temperature = 1.0`. The classifier learned the majority action more decisively. Balanced Accuracy ticked up by a small but real amount on the held-out windows. Then I ran the paired-Δα measurement — same seeds, same data, only `oracle_sharpen_temperature` flipped — and the deployment Alpha effect was within noise. By the multi-seed standard the series settled on in Article 8 (mean Δα ≥ +3pp, no single seed below −5pp), Turtle Breaker did not promote. It did not regress, either; the effect was simply small enough that three seeds could not distinguish it from zero. The pattern is by now familiar enough that I have started to predict it before I run the experiment: *if the only thing that changed is the sharpness of the supervisory signal, the deployment Alpha will not move.* Article 6 wrote this finding down as the classifier-Alpha anti-correlation. Turtle Breaker is the gentler version of it — not an anti-correlation but a non-correlation. Sharper labels, identical trader.

**The third layer is the deepest, and the one I have come around to slowly.** Sharpening is the right answer to the wrong question. The multipass oracle averages *because each individual pass is noisy*; the averaging is variance reduction. Sharpening then *increases peakedness post-hoc*. We are first deliberately blurring and then deliberately un-blurring. The composition does preserve the right invariants — the 50/50 stays 50/50, the math is sound — but the question it answers is *"how do I make the averaged target look like the un-averaged target, but without losing the genuine-disagreement signal?"* The answer to that question is Turtle Breaker. The better question is: *why is each oracle pass noisy in the first place?*

The path solver's per-pass noise has a known structure. Most of it comes from k=1: with only one asset selectable per step, the top-1 ranking flips on small return perturbations whenever the second-place asset is close to the first-place asset, which in cross-sectional markets is most of the time. TASK_157 (oracle k and adaptive weight) and TASK_764 (oracle-k reformulation, Direction 3 of TASK_761) both attack this directly. They ask: *what if each pass were less noisy to begin with?* TASK_764's hypothesis is that increasing k to 8 or 12 would both reduce per-pass noise (more selected assets per step → less ranking sensitivity) and increase label density (more BUY/SELL signal per episode). That is the upstream fix. Sharpening is the downstream Band-Aid — it patches the symptom of the per-pass noise rather than the cause.

The Band-Aid is not worthless. It is mathematically correct, it costs nothing, it composes cleanly with the rest of the pipeline, and the symmetric-case preservation property is non-trivial — it is the reason I am keeping the code in the file rather than reverting it. But it is not load-bearing in the way I had hoped when I wrote it. If TASK_764 lands and the new oracle is sharper per-pass, Turtle Breaker will become redundant. If TASK_764 doesn't land, Turtle Breaker won't save the situation either.

This is the second time in this notebook I have had to write down a version of the same lesson: *cleaner training signal does not automatically buy deployment Alpha*. The first version was the classifier-Alpha anti-correlation. The second version is here. I am beginning to suspect there will be a third version, and a fourth, and that the lesson is the central refrain of the entire stack — that the bottleneck of "trader makes money" is not located in any single supervisory-signal channel that imitation learning can sharpen.

## What's left

Turtle Breaker is implemented. The code lives at `mcts_trader/jax_agent/path_solver.py`, six lines after the averaging step in `multipass_oracle_policy`. The config field `oracle_sharpen_temperature` is in `easy_config.py`. The JIT signature is updated. The unit tests pass: peaked-stays-peaked verified, symmetric-stays-symmetric verified to within `1e-6` on a synthetic 50/50 case, monotone-majority-amplification verified across a sweep of temperatures from `0.05` to `1.0`. The implementation is correct; I am confident about that.

It is *not* enabled in the canonical Stage-B config. It sits behind an opt-in flag, defaulting to `1.0` (i.e., off) in the canonical preset and to `0.1` in the experimental presets that wanted to try it. The decision to leave it opt-in rather than promote it to canonical was the kickback's: the paired-Δα measurement did not clear the promotion gate, and the series' multi-seed standard (Article 8) is now binding on all such decisions. A mechanism that does not move Alpha by more than noise is not promoted into the canonical config, no matter how clean the math.

The position in the architecture is right. The math is right. The deployment-Alpha return is unclear. Sharpening will be revisited if TASK_764's richer-oracle direction lands and the new oracle is sharper per-pass — at that point, the multipass averaging itself may no longer need to be a blur-creating operation, and the question of whether to sharpen its output reopens. It is also worth saying that Oracle 6 — TASK_697, the MCTS-shaped target distillation — would obviate Turtle Breaker entirely if it ever runs. MCTS targets are *visit-count distributions*, which are sharp by construction in proportion to how decisively the search converged; the question of "should I sharpen the soft policy after averaging" doesn't arise. So Turtle Breaker is a fix that, if either of the two pending oracle reformulations land, will become unnecessary.

For now, the file is one stanza longer than it was in February. The sharpening operator is in the codebase, gated, tested, and not on the canonical path. It is one more entry in the lineage that Article 17 walked: another mark on the measuring stick that was not the mark that grew the seedling.

> *The Apothecary returns the ten slips to their envelope, labels the envelope with the date and the patient's name, and files it beside the ledger. The dose has been prescribed. The patient is asleep. The Apothecary tends to the next bed.*

---

## Sources

- `tasks/TASK_677_TURTLE_BREAKER.md` — the spec, code block, and comparison table
- `tasks/TASK_673_MULTIPASS_ORACLE.md` — Oracle 3, the averaging step that creates the blur Turtle Breaker un-blurs
- `tasks/TASK_157_ORACLE_K_AND_ADAPTIVE_WEIGHT.md` — the upstream alternative: make each pass less noisy
- `tasks/TASK_764_ORACLE_K_REFORMULATION.md` — the queued richer-oracle direction
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` — Oracle 6 on hold; would obviate Turtle Breaker
- `mcts_trader/jax_agent/path_solver.py` — `multipass_oracle_policy`, line 322 and onward
- `easy_config.py` — `oracle_sharpen_temperature` field
- Article 6 — the classifier-Alpha anti-correlation, the canonical version of this article's kickback
- Article 8 — the multi-seed promotion standard that binds the deployment-Alpha measurement
- Article 17 — *The Five Oracles*, the lineage Turtle Breaker is a footnote to
- Hinton et al. 2015, *Distilling the Knowledge in a Neural Network* — the temperature-sharpening operator's origin
