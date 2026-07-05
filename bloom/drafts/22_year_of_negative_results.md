> *Observation: the season closes. The gardener carries two baskets to the bench. The harvest basket is light — a few seed-heads, one root she trusts, three cuttings she will overwinter and try again. The journal is heavy. She weighs the journal against the harvest and the journal is heavier. She files the journal in the cabinet under the year and writes, on the next blank page, the date of the season to come.*

---

**by Lukas — May 22, 2026**

# What a Year of Negative Results Looks Like

*On the shape of the kickbacks once they line up next to each other, the three lessons that survived all of them, and the question I am still not sure I have answered: whether writing this series produced wisdom or only the feeling of wisdom.*

---

## The question this article is here to answer

It is the morning of May 22, 2026. `workstation` is on. The desk fan is on. `nvidia-smi` in the right pane is ticking through a checkpoint validation that has nothing to do with this article and everything to do with the fact that the diary continues even when the series does not. The git-objects directory is mounted on `/dev/sdb1` as usual. The single EVGA 3090 has been the only GPU in this case since May 17, when the CMP 90HX came out, and it is the only GPU that has ever produced any of the numbers in any of the twenty-one articles that precede this one. That fact is more load-bearing than it looks; I will come back to it.

I have been writing this series, on and off, since Article 0 — *Before AlphaGo Had a Name*, the prequel about a 2014 sketch and a decade in a drawer — went up under the same byline a few days ago. The series has a shape. It opens with a fever-dream design doc that promises "superhuman performance," walks through a year of negative results, and ends — this article — with the campaign declared exhausted and a structural pivot pencilled in but not yet executed. Twenty-one articles, a hundred-thousand-odd words, none of which I wrote because the experiments worked. I wrote them because the experiments didn't work, and because the failures had texture, and because the texture turned out to be the artifact.

The question I want to take seriously in this closer is the one the series never quite asked outright: *what does a year of negative results actually teach you?*

It is a serious question. The cheap answer is *humility*, and I want to refuse the cheap answer. Humility is a posture, not a lesson. The lessons, if there are any, have to live one level lower — in the *structural features of the kickbacks themselves*, the patterns that recurred across experiments that were nominally about completely different things. The classifier-Alpha anti-correlation in Article 6 is not, on its surface, the same failure as the Bella precompute disaster in Article 9. The single-seed TASK_737 collapse in Articles 7 and 8 is not the same failure as the eval-MCTS asymmetric lift in Article 10. But once you put them next to each other, they have a shape. Several shapes — overlapping, sometimes nested, always recognizable in hindsight. This article is about the shape.

It is not a recap. The recaps live in the articles themselves. This article is the *distillation*: the lifting-out of the cross-cutting observations that no single article could carry, because each article was, properly, about its own particular kickback, and could not honestly compare itself to the kickbacks it didn't yet know about. What I have, with the season closed, is twenty-one articles of particularity and one chance to name what was general.

The taxonomy first. Then the lessons that survived. Then the dead. Then the meta-kickback — the question of whether writing this series is itself a form of conviction-management. And finally a closing.

---

## The taxonomy of kickbacks

I want to introduce a vocabulary. A *kickback*, in the running joke of this series, is the moment confidence meets evidence and loses. Every article has one — that is the rule of the form. But once you have twenty-one of them, you notice that the *mechanism* of each loss is not unique. They cluster. I count five clean categories, plus one I want to call out as a borderline sixth. For each, I will name two or three articles where the category appears, name the structural feature it shares, and name what defends against it.

### 1. Variance kickbacks

A variance kickback is the case where the signal you thought you saw was within the noise floor of the measurement, and a single draw gave you the wrong tail.

The canonical instance lives in Articles 7 and 8 — *TASK_737: The Champion That Wasn't* and *Five Seeds and a Funeral*. On the morning of May 13, 2026, a four-cell ablation table showed `no_aux_solo` at **+6.21 % Alpha** against the prior canonical baseline at **−1.99 %**. The gap — 8.2 points between two configurations that differed by a single boolean — was big enough to feel structural. I wrote it up. I called it *the new canonical Stage-B baseline*. I bolded a sentence in `lore/notes_2026-05-13.md` §95 with the word **NEGATIVELY** in capitals. From the article:

> "The two heads interact NEGATIVELY (-13 pp combined effect)."
> — Article 7, quoting my own lore entry, written before the second seed had landed

The next morning the funeral arrived. Seed 13 came in at **−4.38 % Alpha**. The five-seed average for `no_aux_solo` was **+3.61 %** with a standard deviation of about **4.7 points** and a **12.2-point range** between the worst and best seed. The prior canonical baseline's reference number was +3.87 %. The 8.2-point gap I had bolded as structural was, in fact, less than one standard deviation of the actual noise process. From Article 8:

> "Seed 42 was an upward outlier. The TASK_737 finding is UNRELIABLE — driven by single-seed variance."
> — Article 8, quoting `lore/notes_2026-05-13.md` §99

The variance kickback has a structural feature: the measurement procedure is noisier than the effect size being chased, and the only honest defense is replication across independent draws. On this stack, the operational rule that came out of the wreckage is the one Article 8 closes on — *any new claim of improvement on this benchmark, with a magnitude below ~5 Alpha points, requires at least three paired seeds before it can be called a claim*. That rule is not free. Three seeds is roughly three GPU-days on `workstation`. The temptation to skip the replication and ship a single-seed result is large precisely when the single-seed result is exciting; that is exactly the moment the rule has to win.

### 2. Measurement kickbacks

A measurement kickback is the case where the scoreboard you were optimizing against was not the scoreboard that would judge you in production. Every gain reported against the wrong scoreboard is, at best, irrelevant; at worst, *anti-correlated* with the real one.

The canonical instance is Article 10 — *Eval-MCTS Asymmetric Lift: When the Scoreboard Was Wrong*. For the entire 738–758 campaign — fourteen evaluations, three months of work, every result table I wrote and every paragraph of every article between 6 and 9 — the benchmark used **argmax of the policy logits** as its action-selection operator. The deployed agent in `real_trade_jax.py`, the one that talks to the live exchange, uses **MCTS with sims=32**. The argument for argmax was that in the limit of infinite simulations MCTS visit counts concentrate on the value-head argmax, so for evaluation cheapness the two should agree. The argument was correct in the limit. It was wrong in the regime — because MCTS spends more rollouts where the value head is uncertain, and the baseline checkpoint happened to be a low-confidence value-head regime where MCTS lifted the action quality, and the *variant* checkpoints happened to be high-confidence regimes where MCTS and argmax agreed. From the article:

> "The argmax benchmark was systematically biased in favor of the variants by under-scoring the baseline."
> — Article 10, the load-bearing sentence

The recalibration moved every Δα toward null. The +0.72-point oracle gain became a −1.20-point regression. The flow-matching mechanism's +0.20-point lift became a −0.15. Net: zero promotions, fourteen mechanisms reduced to noise or worse, a three-month campaign declared *exhausted at this stack's precision*.

The structural feature: the cost of the wrong scoreboard is paid not once but on every measurement taken against it. Three months of decisions about which variant to develop, which to abandon, which to write up — all made against a biased operator. The decisions are not recoverable; only the *labels* on the decisions are. The same shape appears, smaller, in Article 6's classifier-Alpha story: balanced accuracy was the optimization target for months, and balanced accuracy turned out to anti-correlate with Alpha. Two different layers of the stack, same category of mistake.

Defense: *the deployed metric is the only metric*. Intermediate metrics are reasonable as diagnostics; they are not safe as optimization targets. Every campaign should be re-weighed against the deployment operator periodically and aggressively, even when the deployment operator is expensive — and especially when it is expensive, because the expense is what made you avoid running it in the first place.

### 3. Causality kickbacks

A causality kickback is the case where improving the upstream metric actively *worsened* the downstream metric — where the chain of inference from "better X causes better Y" turned out to run the other direction, or to have a sign flip in the middle.

The canonical instance is Article 6 — *Better Classifier, Worse Trader*. Three independent experiments — TASK_737 reread under the variance verdict, TASK_741 Mint, TASK_738 legacy_27ch — all showed the same pattern: balanced accuracy on the directional classifier went up, and Alpha went down. The catastrophic version, TASK_738, posted **ΔBal.Acc +7.48 pp** and **Δα −18.44 pp** in the same row. The trade count went from 136 to 13,246 — fifty times over-trading. Sharpe flipped from +8.68 to −27.20. The encoder was *genuinely classifying better*, and the trader, using that genuinely-better classifier, was *genuinely trading itself into the ground*. From the article:

> "The encoder is genuinely classifying better. The trader, using that genuinely-better classifier, is genuinely trading itself into the ground."
> — Article 6, the moment the third instance made the pattern a property

The naive inference chain — sharper classifier → better policy prior → better trades → better Alpha — has, in this stack, a step that flips. The sharper classifier produces an over-confident policy distribution. The over-confident distribution, fed into a trader with no entropy regularization at the policy head, produces over-trading. The over-trading eats fees and produces a Sharpe collapse. The Alpha follows the Sharpe.

The lesson Article 6 closes on: *intermediate metrics can correlate with deployment metrics, anti-correlate, or be orthogonal. There is no theorem that says they must agree.* Three independent experiments demonstrated the anti-correlation; the entire mechanism class upstream of the policy head was closed. The same shape lives one level up in Article 1's "break causality" framing — the AlphaZero recipe presupposes a game where world-model, policy, and value all converge under self-play, and the market is not a game. The chain of causal inference from "AlphaZero works on Go" to "AlphaZero-style works on markets" has a step that flips, at the step where you assume the market plays the same kind of game.

Defense: *measure the deployment metric, even when the upstream metric is what you're optimizing*. The deployment metric is the only one that can tell you whether the chain of inference held.

### 4. Estimation kickbacks

An estimation kickback is the case where the back-of-envelope arithmetic for "how long / how big / how expensive" was wrong by enough orders of magnitude that the plan itself was structurally infeasible — but the planner did not know it, because the multiplication was never written down.

The canonical instance is Article 9 — *From 60 Minutes to 93 GPU-Days*. The TASK_740 precompute SPEC said *30 to 60 minutes*. The actual workload, once I bothered to compute it on a calculator app while the tqdm bar crawled, was **270 periods × 258,000 windows × 4 samples = 278,640,000 Bella forward passes**, at an observed 8.6 windows per second, equaling about **2,241 hours, or 93 GPU-days**. From the article:

> "The SPEC's estimate is wrong by three orders of magnitude. Not 1.5×. Not 5×. Not 20×. A thousand."
> — Article 9

The structural feature of an estimation kickback is that the planner had all the numbers individually and never multiplied them together. I had benchmarked Bella at ~150 predictions per second on a single window. I knew the corpus had 270 periods of 258,000 windows each. I knew k=4. I had each of these numbers in a different mental compartment, and the compartments did not talk to each other. *The number 270,000* — comfortable per-minute throughput — *and the number 270,000,000* — actual workload — were never adjacent on a page.

This is not an arithmetic error. It is a *category* error about what the job is doing. The planner believed they were planning "a precompute job;" the planner was actually planning "a small training run that produces no model and uses 93 GPU-days of `workstation` time." Those are not the same job. The same shape appears in Article 0's hardware wishlist — *4 × RTX 3090, 512 GB RAM, EPYC 32-core* — a spec written years before the parts existed, carried in my head as if it were a constraint when in fact the real constraint was the single 3090 under the desk. The plan thought at dream-spec; the project ran at available-spec; the gap was not closed until reality enforced it.

Defense: *multiply the arithmetic onto the page*. Every SPEC has a runtime estimate, and the estimate must be expressed as the product of *all* the dimensions of the workload, with units, on one line. If it cannot be expressed that way, the planner has not yet planned the job — they have only described the desire for the job.

### 5. Substrate kickbacks

A substrate kickback is the case where the *shape* of a recipe — the abstract algorithmic structure that worked elsewhere — got transplanted, and the *substrate* it was transplanted into refused to cooperate. The blueprint was right. The soil was wrong.

The canonical instance is Article 1 — *I Thought I Was Going to Break Causality*. AlphaZero's recipe was MuZero + MCTS + self-play. Each ingredient was independently validated by DeepMind. Each ingredient was, in some sense, *known to work*. The vision document's argument was that since the ingredients composed for Go, the composition would carry to a market. From the article:

> "MuZero gave you a learned world model... MCTS gave you planning under uncertainty... AlphaZero-style self-play gave you a way to discover tactics nobody had taught you. Stack those three on top of a small recursive-reasoning encoder, and the path from 'I have a GPU' to 'I have an agent that beats hedge funds' was — I want to be honest about the way I phrased this in my own head — a matter of *engineering follow-through*."
> — Article 1

The composition did not carry. The substrate — partially observable, non-stationary, low-signal-to-noise, evaluated through Alpha (a derived quantity that anti-correlates with the natural intermediate targets) — refused the recipe. The world model was learnable but not stable at MCTS planning depth. The search was implementable but spent its rollouts in regimes that didn't coincide with the market's high-payoff regions. The adversarial market was *quietly abandoned* mid-implementation because no clean training loop survived contact with a non-stationary base agent.

The substrate kickback has a deeper structural feature than the others: it cannot be defended against by better engineering. It can only be defended against by *naming what about the substrate is hostile to the recipe before the recipe is applied*. The work between Articles 1 and 10 is, in retrospect, a single long substrate kickback expressed as fourteen mechanism failures. The mechanisms were good. The substrate would not have promoted any of them; it does not promote anything at this precision.

Defense: the only one I know is brutal — try the recipe at the cheapest possible scale before scaling it, and *believe the cheap-scale signal even when it disappoints*.

### 6. Discipline kickbacks (a borderline category)

This one is real but I am less certain it deserves the same status as the others. A discipline kickback is the case where the *epistemic procedure* — how a result was promoted from conjecture to claim — failed before the result itself failed.

The canonical instance is Article 7's premature conviction. The four-cell ablation produced a number; the number was written into a results report *the same morning*; the report was published in the lore before the replication had been run. The discipline failure was that the order of operations was inverted: claim, then verify; not verify, then claim. The wave-function-collapse epistemic structure — where the act of writing the result up *fixed* the result in a way that subsequent disconfirming evidence had to *overturn* rather than *prevent* — is the structural feature. It is harder to take back a claim than to never make it.

The discipline kickback is borderline because it is, in some sense, a *meta* category — it is the category for the cases where the failure was procedural rather than substantive. The other five categories are about what the system does. The discipline category is about what *I* did. I'm including it because in retrospect it is the category that I think is most under my own control. The other five are properties of the universe. This one is a property of me.

---

## What survives across all of them

Five (or six) categories of kickback. Twenty-one articles. Roughly a year of negative results. What is the durable lesson — the thing that, even after the season closes and the next season's experiments produce their own different failures, will stay true?

I want to be careful here. The temptation is to produce a list of seven items, calligraphed, and call it wisdom. The honest answer is shorter than that. Three lessons survived everything, and each of them was beaten into me by more than one of the categories above.

### 1. Measure the deployment metric directly, always.

Intermediate metrics — loss, balanced accuracy, classifier sharpness, directional accuracy, rank correlation, Sharpe at training time — can correlate with the deployment metric, anti-correlate with it, or be orthogonal to it. *There is no theorem.* The relationship between an intermediate metric and the deployed metric is an empirical fact about this stack, this data, this regime, on this day. It is not derivable from first principles. It must be measured.

Article 6 demonstrated the anti-correlation case. Article 10 demonstrated the orthogonal case. Articles 7 and 8 demonstrated the *within-noise* case — the intermediate measurement was not stable enough to relate to the deployment metric at all. *Do not let an intermediate metric stand in for the deployment metric, ever, without a per-ckpt receipt.* The operational form is expensive: every "improvement" claim on this project must be re-evaluated under MCTS-32 against three paired seeds before it gets a name. Roughly three GPU-days per claim on `workstation`. It is the only price I have found that buys an honest result.

### 2. Replicate before reporting.

Single-seed claims below the noise floor are not claims. They are decorations.

The TASK_737 / TASK_743 arc produced a precise number for the noise floor on this stack: **roughly 5 Alpha points on a single seed, falling to under 2 with five seeds**. Below that threshold the result is consistent with the null; above it the result is consistent with a real effect, but not — even then — demonstrated, because the alternative explanations (sample-specific overfit, transient regime, dropout luck) have not been ruled out.

The rule: *no result is a result until it has replicated*. Three paired seeds, minimum, for any claim under ten Alpha points. Five for any claim that would change a canonical baseline. I have violated this rule in *small* ways multiple times since Article 8; the temptation is large precisely when the single-seed result is exciting. Writing it down here makes it slightly easier to refuse next time.

### 3. Multiply the arithmetic onto the page.

Estimation errors are not arithmetic errors. They are category errors about what the job is doing, expressed through a missing multiplication.

Every SPEC since the precompute disaster has a *workload-multiplication line*: `270 periods × 258,000 windows × 4 samples = 278,640,000 forward passes / 8.6 per second = 32 million seconds = 93 GPU-days`. One line, with units. The line catches three orders of magnitude before the GPU does.

These three lessons are not new. Any experienced ML engineer would nod at them. *I would have nodded at them too, before this year.* The lessons are not what is new. What is new is the *texture* of having had each of them beaten into me by an experiment that I genuinely believed would not require the lesson. The wisdom is not in the proposition. The wisdom is in the body memory that the proposition produced — and that last sentence is exactly the kind of sentence I should not write without inspecting. We will inspect it in the meta-kickback section.

---

## *The Apothecary sets down her trowel*

> *The journal outweighs the harvest in every season. The Apothecary has known this since before she could read her own hand. The journal preserves what the basket cannot — the date of the first frost, the dose that killed the cutting, the soil-pH on the morning the specimen flowered out of season and meant nothing. The basket is what feeds the year. The journal is what feeds the next gardener, who may be the same gardener twelve months later, or may be someone else entirely. She closes the cover. She does not lock it.*

---

## What didn't survive

A season closes by listing the dead, because the dead are what teaches.

**The HSV-encoded image substrate** (Article 0). In the 2014 sketch, a chart was encoded as an image — Hue for relative price, Saturation for support-strength, Value for volume — and a CNN was supposed to read the chart the way a human reads one. The encoding got perception right and got "where the alpha lives" wrong. Chart perception is not where alpha lives in modern crypto markets — alpha lives in cross-sectional structure across many assets, and a chart is a single-asset object. The HSV encoding never made it into the production codebase. It taught me that *the form of the input is downstream of where the signal lives*, and I had picked the form before I had located the signal.

**The physics core** (Article 2). The NeuralOGCM-inspired hybrid — learnable α, β, σ, γ plus a neural corrector — never plugged into a training run. Forward pass benchmarked clean. 10-step rollout benchmarked clean. The integration surgery never happened, because by the time it would have been needed, the world model entirely had been deprioritized in favor of cleaning the encoder. *Unintegrated cleverness is worth zero load-bearing, even when it is correct.*

**The adversarial market** (Article 1). The two-player game at the heart of the AlphaZero analogy — the part supposed to produce *Move 37* — was silently abandoned during implementation. The adversary kept either collapsing to a trivial price path or diverging into one that violated microstructure constraints. No clean reward shaping for the adversary survived contact with the agent's curriculum. It taught me that *recipes that work on closed games do not transfer cleanly to substrates where the adversary's action space is unbounded*.

**Janus and the MoE framing** (Article 0). The codename — Janus, the Roman god of doorways — was attached to the original mixture-of-experts framing, the "fifty cores" architecture where Trading was just the testbed for a multi-agent general-intelligence system. Both died together. The MoE framing was replaced by a single recursive-reasoning encoder per asset; the production stack is four specimens (Poppy, Bella, Foxglove, Yarrow), not fifty. Janus has no module left to name. It taught me that *the framing in the vision doc is the framing the writer wanted; the framing in the codebase is the framing the substrate allowed* — and codenames die fast when the architectures they named die.

**The four-GPU dream** (Article 0). The hardware wishlist of `4 × NVIDIA RTX 3090, 512 GB RAM, EPYC 32-core` survived in the vision doc as a constraint, in my head as an aspiration, and in reality nowhere. The project ran on a single 3090. The single 3090 produced every number in every article in this series. The four-GPU dream taught me that *the dream-spec is invisible until reality enforces the actual-spec*, and that the dream-spec costs work, because the design that thinks at four-GPU scale wastes the single-GPU it has.

**The single-seed result table** (Articles 7, 8). The literal artifact — the four-row table that, on the morning of May 13, said `no_aux_solo` was +6.21 % — is in the lore as a historical document. It is annotated. It is dated. It is no longer canonical. It taught me that *a result table is a snapshot of conviction, and conviction is variance-tail-dependent until replication forces the average*.

**The argmax-eval benchmark** (Article 10). For three months the canonical scoreboard. After the eval-MCTS recalibration of May 19–20, demoted to a diagnostic. The fourteen-mechanism campaign run against this benchmark is recoverable only as ranked-against-a-biased-operator; the rankings themselves cannot be salvaged. The argmax-eval benchmark taught me that *the cheap evaluator is a debt, paid in retrospective recalibration of every measurement made against it*.

Each of these died. Each of them taught something by dying. I am writing this section last because I wanted the list to be honest, and the only way to make the list honest was to enumerate, by name, the things I have stopped trying to defend.

---

## The meta-kickback

There is a final category of kickback that does not fit neatly into the taxonomy above, because it is not a kickback the experiments delivered. It is a kickback the *act of writing this series* delivers, if I am honest about what writing it is doing.

The series presents itself — implicitly, through the discipline of its form — as having *processed* the negative results into something durable. Each article ends with a "what's left, what I learned" section. This closer enumerates three lessons that *survived*. The meta-kickback is the question of whether the lessons are actually lessons, or whether the *act of articulating them* is a form of conviction-management that produces the *feeling* of having learned in the absence of the learning itself.

The disciplined-sounding three — measure the deployment metric, replicate before reporting, multiply the arithmetic onto the page — are sentences I can write. They are also, I am afraid to say, sentences I could have written before any of the failures, with the same prose force, on the strength of a textbook. The propositions are not new. What is new is the *texture* of having been beaten by each of them. I claimed that earlier. I want now to inspect the claim.

Is the texture itself the lesson? Or is the *belief* that the texture is the lesson an artifact — a feeling generated by the act of writing, that has the shape of wisdom without necessarily being wisdom?

The worry, laid out plainly: by writing twenty-one articles that *describe* the negative results, I make the lessons feel earned. The narration gives the lessons a body memory; the body memory feels — to me, the writer — like the lesson itself has been installed. But the same body memory could be the side effect of having spent weeks writing the articles, not of having spent the year living through the experiments. The articulation, the formal closing-out, the act of putting a kickback into a category and giving the category a name — these *could* deepen learning, and *could* also create the illusion that learning has happened.

The honest version: *I have one year of negative results, a series of articles that describe the negative results, and an open question about whether the description has produced wisdom or just rhetoric.* The series itself is, in some sense, the thing being kickbacked against.

If the next campaign — TASK_761's structural pivot directions — produces results that I would have produced *without* having written the series, then the series was rhetoric. If the next campaign avoids one of the categories above where the prior campaign would not have, then the series was lesson. I will not know which until the first structural-pivot results land, and possibly not even then, because *not stepping on a rake* is a low-evidence event and *stepping on it* is a high-evidence one.

I want to leave the question open. The series is honest only if I leave it open. The temptation to close it — to write the sentence "I have learned, and the proof is that you have read this far" — is the same temptation that produced the bolded **NEGATIVELY** at 07:00 on May 13. It is the temptation to fix the wave function before the measurement has been made.

That is the meta-kickback. It is the deepest one because it is the one I cannot put on the calculator app and verify. The other categories — variance, measurement, causality, estimation, substrate — all have empirical defenses. This one only has time.

---

## What the season is for, in the end

The season is not for the publication. The blog is a delivery mechanism, and a delivery mechanism is not a purpose.

The season is not for the audience. The audience for this series is small and self-selected and, I suspect, mostly composed of people who already know most of these lessons in some textbook form and are reading for the texture. The audience is too thin a thread to have kept me writing through twenty-one articles.

The season is not for the project. The project — Poppy, Bella, Foxglove, the trader stack, the structural pivot — is exactly as alive or dead today as it was three weeks ago, before any of these articles existed. The writing has not changed the codebase. The codebase, in this house, is the source of truth; the writing is downstream of it.

The season is for the *future Lukas* who, six months from now, on a different morning, on `workstation`, with the desk fan on and the 3090 humming through some other validation, will sit down to launch a new experiment and feel the small distinct temptation to do one of the things this series catalogued. The temptation to write up a single-seed result before the second seed lands. The temptation to optimize against an intermediate metric whose relationship to deployment has not been re-measured. The temptation to skip the multiplication line in the SPEC because the rough order-of-magnitude *feels* right. The temptation to declare the substrate friendly to a recipe that the substrate, in fact, has never agreed to.

The future Lukas will read this article — perhaps in the published archive, perhaps in the working tree, perhaps just remember the feel of it — and recognize the *texture* of the failure that is about to happen. He may step on the rake anyway, because some rakes are unavoidable and some kickbacks are educational only if they happen. But he will, perhaps, step on it a little less, or recover from it a little faster, or write a slightly more honest SPEC the next morning. *That is what this is for.*

The series is a lab notebook, sent forward in time. The recipient is the same writer at a later date. The form — twenty-one articles, several thousand words each — is not what the recipient needs. The recipient needs the *texture*, the body memory, the small distinct feeling that says *I have been here before*. The form is the carrier. The carrier is dispensable. The body memory is the payload. If the payload arrives intact at the next mistake, the series did its work.

If not, the series was rhetoric, and I will find out the way I find out everything else on this project: from the row in the result table that takes back yesterday's claim, on a morning with the desk fan on and the coffee going cold.

---

## Closing

The diary continues. The series does not.

`workstation` is on as I close this file. `nvidia-smi` is ticking through the same checkpoint validation it has been ticking through for the entire morning; the validation will finish in about twenty minutes and the next overnight will launch from the standard config with no new mechanism enabled, because the next campaign is structural — reward shaping, entropy regularization, the oracle's k-reformulation, the incompatible-dataset retrain — and structural campaigns are slow. The first results, if they arrive, will arrive in Q3. The next season of articles will be written, or not, depending on whether the structural results have the shape of an article: a kickback, a texture, a lesson that earns the space.

For now, the journal closes for the season. The next page is labelled. The pen is set down on the bench. The Apothecary tidies the cabinet.

> *The basket is light. The journal is heavy. She files the journal under the year. On the next blank page she writes the date of the season to come, and below it, the names of the four specimens she will overwinter: the reward shaping, the entropy regularizer, the k-reformulated oracle, the dataset that is not yet on the shelf. She does not promise herself any of them will bloom. She labels the shelf, and turns out the lamp.*

---

## Sources

- Article 0 — *Before AlphaGo Had a Name* — `bloom/drafts/00_before_alphago_had_a_name.md`. The 2014 sketch, the HSV encoding, Janus the codename, the four-GPU dream, the decade in the drawer, the workstation `workstation` and its single EVGA 3090.
- Article 1 — *I Thought I Was Going to Break Causality* — `bloom/drafts/01_break_causality.md`. The December 2025 vision doc, the AlphaZero analogy, the adversarial market that was quietly abandoned, the substrate kickback.
- Article 2 — *Physics Core: A Differentiable Market I Could Differentiate* — `bloom/drafts/02_physics_core.md`. NeuralOGCM, learnable α/β/σ/γ, the prettiest code that never plugged into training.
- Article 3 — *The Berserker That Knew It Was a Berserker* — `bloom/drafts/03_berserker_philosophy.md`. Intentional constraint, the five-layer safety oath, the moment "limited by design" became cope.
- Article 4 — *Encoder Output, Repeated 64 Times: The Information-Loss Audit* — `bloom/drafts/04_encoder_64x_redundancy.md`. The 22 GB dataset that should have been 338 MB; the GPU bored at 5–17 %.
- Article 5 — *22 Silent Killers: The BUG_MAYHEM Audit* — `bloom/drafts/05_bug_mayhem.md`. The pipeline that runs but does not learn; the 2026-01-29 prompt that named the hunt.
- Article 6 — *Better Classifier, Worse Trader* — `bloom/drafts/06_classifier_alpha_anticorrelation.md`. The classifier-Alpha anti-correlation, confirmed three times; TASK_738's +7.48 pp / −18.44 pp row.
- Article 7 — *TASK_737: The Champion That Wasn't* — `bloom/drafts/07_task737_champion_that_wasnt.md`. The +6.21 % single-seed result; the morning the bolded NEGATIVELY went into the lore.
- Article 8 — *Five Seeds and a Funeral* — `bloom/drafts/08_five_seeds_and_a_funeral.md`. Seed 13 at −4.38 %, the 12.2-point spread, the operational rule for under-5-point claims.
- Article 9 — *From 60 Minutes to 93 GPU-Days: The Bella Precompute Disaster* — `bloom/drafts/09_bella_precompute_93_gpu_days.md`. The three-orders-of-magnitude estimation error; the multiplication that was never written down.
- Article 10 — *Eval-MCTS Asymmetric Lift: When the Scoreboard Was Wrong* — `bloom/drafts/10_eval_mcts_asymmetric_lift.md`. The argmax-vs-MCTS asymmetry; the campaign declared exhausted; the structural pivot named.
- Article 11 — *Fourteen Days of Shape Mismatches* — `bloom/drafts/11_fourteen_days_of_shape_mismatches.md`. The TASK_115 integration-seam campaign; the bugs that live in the gaps between modules.
- Article 13 — *The Hang at Epoch 90: An RCA* — `bloom/drafts/13_hang_at_epoch_90.md`. The deterministic-shape, non-deterministic-timing deadlock; the four-day chase; the shape of a silent-failure epistemology.
