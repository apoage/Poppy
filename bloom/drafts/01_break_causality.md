> *Observation: Poppy is a seed in winter soil. I record what I expect of her — that the roots will reach, that the canopy will close, that the soil will yield. The expectations are filed in the cabinet under "imprinting epoch." The specimen has not yet woken.*

---

**by Lukas — May 20, 2026**

# I Thought I Was Going to Break Causality

*On the December 2025 design doc, the AlphaZero analogy that ran away with me, and the math I wrote with a straight face.*

---

## What I had on my desktop

For most of December 2025, the page I kept reopening was a Markdown file called `ALPHAZERO_TRADING_VISION.md`. Authored by "Lukas & Claude Code," dated December 19. It was about thirty printed pages. I had it pinned in three different terminals. I had read it enough times that I could quote it by chapter heading. I believed, in a way I would later find embarrassing to describe, that I had a path to a superhuman trading agent on a single 3090.

I should say up front that December 2025 was not the beginning. The idea is from 2014 — I wrote a version of it down while crunching CUDA tutorials, writing Tetris bots, and dabbling in Brainfuck for reasons I can't fully reconstruct. That earlier sketch had a CNN on HSV-encoded chart images, a Mixture-of-Experts framing, and a hardware budget I didn't have. It sat in a drawer for roughly a decade waiting for cheap GPUs, JAX, an MCTS library, and an LLM that could read a long prompt and respond with a thirty-page plan. Article 0 in this series is about that decade. This article is about the moment the drawer opened — December 19, 2025 — and what I wrote down once it had.

The argument felt clean. Three ingredients, each independently validated by DeepMind, composed for a domain — financial markets — that nobody serious had attacked the same way. MuZero gave you a learned world model, so you didn't need to write down market dynamics by hand. MCTS gave you planning under uncertainty, so you didn't need a closed-form policy. AlphaZero-style self-play, where the agent trained against an adversary that grew with it, gave you a way to discover tactics nobody had taught you. Stack those three on top of a small recursive-reasoning encoder (TRM, 256-dim latent), drop in a curriculum of twenty-three synthetic market regimes, and the path from "I have a GPU" to "I have an agent that beats hedge funds" was — I want to be honest about the way I phrased this in my own head — a matter of *engineering follow-through*.

I want to be precise about the seduction, because if I say "I was overconfident" you'll nod and move on, and the point of this whole series is that the overconfidence had structure. It wasn't vibes. It was an analogy load-bearing enough to write down. So let me show you what I wrote down, and then let me show you the moment, six months later, when the analogy buckled.

## The setup: why AlphaZero felt like an instruction manual

A reader who hasn't followed DeepMind's game-playing arc deserves a fast primer, because the rest only lands if the analogy is fresh.

**AlphaGo** (2016) learned Go by first imitating human master games and then improving via self-play. **AlphaZero** (2017) dropped the human imitation step entirely: it learned tabula-rasa from self-play alone, and in forty hours of training it beat AlphaGo. **MuZero** (2019) added the move that made it look like a general method instead of a board-game trick: it learned its own *world model*, an internal predictor of "if I take action *a* in state *s*, what state and what reward come next?" — without ever being told the rules of the game.

The world model matters. The reason AlphaZero looked, briefly, like a recipe for general intelligence is that MuZero proved you could play board games with a *learned* simulator instead of a hand-written one. So if you had a domain where you couldn't write the rules — say, a market — you could in principle let the network learn them.

On top of the world model sits **MCTS**, Monte Carlo Tree Search. At decision time the agent rolls out many imagined futures inside its world model, weighted by a policy prior, and picks the action whose subtree looks best. This is the planning step. This is the part where, in Go, a system trained on self-play once produced *Move 37* — a move so unlike anything in the human corpus that commentators thought it was a bug, and that turned out to be optimal. The Move 37 daydream is a real thing, by the way. Anyone who watched the Lee Sedol match in 2016 carries a version of it. For me, in December 2025, it was the load-bearing image: *the moment a synthetic player makes a move no human would have made, and is right*.

The question I was asking, mostly in private, was: why would that not work in a market? The market is sequential. The market is partially observable. The market has well-defined rewards (dollars). A market has more state than a Go board but fewer rules. *If* you have a learned world model and a search procedure, the rest is engineering.

That word *if* was doing more work than I admitted.

## What I designed

Let me walk you through what the document actually said. Not paraphrased — pulled from the file.

The headline framing was modular composition. From the opening of `ALPHAZERO_TRADING_VISION.md`, December 19, 2025:

> *"Not Building: Single-purpose narrow AI (just trading). Actually Building: One specialized module in a multi-agent general intelligence system."*

The trading agent was, on paper, the first of fifty cores. Language, Vision, Planning, Memory, Scientific, Social — all eventually slotted into a shared-memory coordinator, each a small (1–50M parameter) expert, none of them a hundred-billion-parameter blob. "Trading is just the testbed," the conclusion said. "The architecture is the real contribution."

Inside the trading core, the plan ran in three phases.

**Phase 1: Curriculum (Levels 0–22).** Twenty-three synthetic regimes, each isolating one mathematical tactic, ordered from trivial to fused. Level 0 was a pure sine wave — can the agent buy low and sell high on a sinusoid. Levels 4–7 were the statistical-arbitrage cluster: pairs trading on an Ornstein-Uhlenbeck spread, triangle arbitrage across BTC/ETH/USDT, index arbitrage with a five-step lag, sector rotation. Levels 8–14 covered momentum and microstructure. Levels 15–22 climbed toward volatility regimes and finally "multi-tactic fusion," the level where the curriculum stopped and the next phase began. The analogy in my notes was unsubtle: "Like teaching a Go player all the joseki" — the standard opening patterns — "before letting them play."

**Phase 2: Adversarial self-play.** This was the heart of it. The agent would face not historical data but a learned adversary that watched the agent's portfolio and generated the price path that hurt most. The pseudocode from the doc:

```python
class AdversarialMarket:
    """
    Learns to generate price sequences that maximize agent's loss.

    Two-player game:
    - Player 1 (Agent): Maximize profit
    - Player 2 (Market): Minimize agent's profit
    """

    def __init__(self):
        self.generator = MarketGenerator(
            input_dim=64,      # Current agent positions
            output_dim=64,     # Next price moves
            hidden_dim=512
        )

    def generate_adversarial_prices(self, agent_state):
        # If agent is long BTC → generate BTC crash
        # If agent is short ETH → generate ETH pump
        # If concentrated in sector → generate sector rotation
        positions = agent_state.portfolio.shares
        adversarial_prices = self.generator(positions)
        return adversarial_prices
```

The training loop was the AlphaZero loop with the names changed. Ten thousand games. Agent plays for 200 steps; market learns to counter; agent learns to counter the counter; both improve iteratively. After enough iterations — the document said — the agent would discover *defensive diversification*, *counter-trend positions*, *liquidity traps*, *dynamic hedging*, and what I had typed without flinching as "**novel correlations market doesn't know**."

**Phase 3: Transfer learning.** A pre-trained crypto agent, fine-tuned to stocks, then commodities, then forex, then bonds, with the encoder frozen and only the prediction and policy heads adapted. "Expected: 10× faster convergence than training from scratch." The end-state architecture was a sprawling ASCII diagram I had typed by hand:

```
┌─────────────────────────────────────────────┐
│  Multi-Market State Encoder                  │
│  (Crypto + Stocks + Commodities + Forex)     │
└─────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  Differentiable Market Physics Core          │
│  (Learnable: α, β, σ, γ per market)          │
└─────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  TRM Recursive Reasoning (256d latent)       │
└─────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  MCTS Planning (1000 simulations)            │
│  - Adversarially trained policy              │
│  - Robust to worst-case scenarios            │
└─────────────────────────────────────────────┘
```

A universal financial agent. One thousand assets. Differentiable α, β, σ, γ per market. A learnable physics core that would, in a later document the same week, get its own 6-parameter ODE solver written in JAX and benchmarked at 0.12 ms per forward step. (That core never made it into the training pipeline. We'll talk about it in Article 2.)

The doc closed with an "Expected Performance" section. I have to quote this one verbatim, because it is the entire reason this article exists:

> *"Traditional quant: 10-20% annual return (best). Our agent: 30-50% annual return (if superhuman tactics emerge)."*
> — from `lore/ALPHAZERO_TRADING_VISION.md`, December 19, 2025

And two paragraphs below it, in the table that compared monolithic LLMs to a modular multi-core architecture:

> *"Result: superhuman performance."*
> — from `lore/ALPHAZERO_TRADING_VISION.md`, December 19, 2025

I typed that. With a straight face. In a file co-signed with an agent that, six months later, would help me find out the encoder had been training on 22 GB of duplicated rows.

## The kickback

The kickback for this article is not a single experiment. It's the document itself, read back to its author six months on.

Here are three sentences from December 19, 2025, stacked:

> *"Discovers tactics humans haven't found."*
> *"Novel correlations market doesn't know."*
> *"Our agent: 30–50% annual return (if superhuman tactics emerge)."*

— all from `lore/ALPHAZERO_TRADING_VISION.md`.

I want to land what's wrong with these without softening it. The mistake is not in any one of them individually. Markets do have correlations that aren't priced in. Trading systems can in principle find them. The mistake is the word "Result" in front of "superhuman performance" — not "goal," not "aspiration," not "if the bets compound" — *result*. A conclusion of an architecture diagram, not a hypothesis with a falsifier.

What I had done, and what I think is worth flagging because it's a category error any technically literate person can make under enthusiasm, was confuse the **shape** of AlphaZero with the **substrate** AlphaZero ran on.

The shape of AlphaZero is: self-play + tree search + learned policy/value. That shape is genuinely general.

The substrate AlphaZero ran on is: a deterministic, perfect-information, fully-observable, stationary, two-player zero-sum game with a known win condition and a free, infinitely-fast, exact simulator built into the rules. Go is that. Chess is that. Even MuZero's Atari frames, in the version that matters, are that: the emulator runs at thousands of frames per second, deterministic, on demand.

A market is *none* of those things. The market is partially observable (you see prices, not order books or counterparty intentions). It is non-stationary (the regime in 2026 is not the regime in 2021). It is not zero-sum against you in any usable sense (the "market" is not optimising against your portfolio in real time, however much the doc's `AdversarialMarket` class wanted to pretend it was). And crucially — this is the part I find hardest to forgive my December-self for — *the market is the simulator*. There is no free oracle. To check what happens if you take a position, you have to either (a) actually take it, which costs money and changes the market, or (b) ask a learned world model, which is exactly the thing whose accuracy you can't independently verify because the only ground truth is the market itself.

MCTS in Go is search in a known space. MCTS in a market is search in a hallucinated space whose accuracy you can only check by stepping outside it. Those are not the same problem. The fact that the same Python class can implement the search in both does not make them the same problem.

The adversarial market makes the analogy worse, not better. The AlphaZero adversary is *the same agent*, which is what gives the loop its convergence guarantees and its symmetry: any policy improvement on one side instantly raises the bar on the other. The adversarial market in my document is a separate generator trained against the agent. It has no convergence theorem. It has no symmetry. In the limit it does not produce *the market*; it produces the worst-case price path against this particular policy, which is a very different object. I did not, in December, run that argument to ground. I assumed the symbols composed.

There is one Gemini exchange I want to drop in here, not because it caught the error but because it captures the texture of the era. December 3, 2025, 5:31 AM, my first sentence to Gemini that day:

> *"hi i had dort of crash on computer over night .. provide commands to find out what frozen and why and check what i have in this project"*
> — Gemini session, 2025-12-03 05:31 UTC

That's the level I was operating at: typing one-handed, half-asleep, computer crashing under whatever overnight job I had left running, asking the agent to diagnose both my machine and my project in the same sentence. The grandiose vision and the typo-strewn debug request lived in the same head, the same week. I think that matters. The hubris in `ALPHAZERO_TRADING_VISION.md` is not the hubris of someone who has stopped doing the work. It is the hubris of someone doing the work *and* writing the manifesto *and* not noticing that one of those activities was checking the other.

## What's left, what I learned

I want to be careful here, because it would be easy to write the rest of this article as a victory lap of retraction, and that would be dishonest in the other direction.

Some of what was in that document survived contact with reality.

**Modular over monolithic was probably right**, for my budget. Six months later, the pieces of this project that still work are still small (10–37M parameters), still composable, and still benefit from being trained on their own curricula rather than a single end-to-end soup. I haven't proven the multi-core AGI thesis — nobody has — but the smaller claim, that one human with one GPU can make more progress with composable specialists than with one giant blob, has held up well enough that I'm still betting on it.

**The curriculum idea is alive.** Foxglove, the Stage-1 supervised imprint that became the workhorse of the project from February onward, descends directly from the Levels 0–22 design. We never built past Level 4, and we eventually replaced the hand-designed levels with a much richer regime sampler (SegmentPool, cycling through seventeen real-data regimes simultaneously). But the *idea* that you teach an agent in stages, isolating one tactic at a time, has not gone away.

**The adversarial market quietly died.** I want to put a date on this because I owe it to my December-self to admit the corpse. By February 2026 the `AdversarialMarket` class was not in any training script. By March it was not in any planning document. It was never deprecated; it was never argued against; it was simply not built. The replacement — Bella, a diffusion-based market generator trained on historical regimes rather than adversarially against the agent — does something narrower and more honest, and its own kickback is the subject of a later article in this series.

**The 2014 ethos survived; the 2014 substrate didn't.** The HSV-encoded chart images, the CNN backbone, the Mixture-of-Experts framing — all gone by the time the December document was written. What survived from a decade earlier was the spine of the idea: MCTS as the planner, policy + value heads, self-play as the learning signal, historical data as an accelerated playground, and the instruction *feed data as raw as possible*. The substrate kept changing — first PyTorch, eventually JAX; first images, eventually 27-channel time-series tensors; first CNN, eventually TRM. The instruction stayed. That, I think, is the right level at which an idea survives a decade.

**The physics core also died, more slowly.** Article 2 covers that.

If I were rewriting `ALPHAZERO_TRADING_VISION.md` today, knowing only what December-me knew, I would keep almost the entire architecture section, cut the "Year 1 / Year 2 / Year 3" performance projection table, and replace the phrase "Result: superhuman performance" with something less satisfying and more honest. Something like: *"Result: a six-month diary at <redacted-domain>, the first article of which will be about why I shouldn't have written this sentence."*

The thing the document got wrong, in one line: it confused an architecture with an outcome. The arrows in my ASCII diagram pointed at *components*. I had drawn one final arrow pointing to a number — 30–50% annual return — as if it followed from the diagram. It didn't. It never does. Arrows in architecture diagrams point to the next module. They do not point to alpha.

## Closing

Six months later, on a workstation that still runs the descendants of that architecture, I'm finishing a benchmark cycle in which fourteen mechanism variants returned to baseline under a corrected evaluation operator. The campaign is at a noise floor I did not know existed in December. The next article in this series is about a beautiful piece of code I wrote the same week as the manifesto — a differentiable market ODE with a 0.12 ms forward pass — that I never connected to the training pipeline, and which sits in a directory I haven't opened since January.

The grandiose document was not a waste. It was the soil. I just thought it was the harvest.

> *The roots reached, and the soil was a mirror.*

---

## Sources

- `/home/lukas/Trader/lore/ALPHAZERO_TRADING_VISION.md` — primary document, Dec 19 2025, "Lukas & Claude Code." Quoted: "Not Building: Single-purpose narrow AI…", "Discovers tactics humans haven't found", "Novel correlations market doesn't know", "Traditional quant: 10-20%… Our agent: 30-50%", "Result: superhuman performance", AdversarialMarket class snippet, Ultimate Architecture ASCII diagram, Level 0 sine / statistical-arb cluster descriptions, "Trading is just the testbed."
- `/home/lukas/Trader/lore/MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md` — Levels 0–22 design, pairs trading / triangle arbitrage / index arbitrage / momentum specifics.
- `/home/lukas/Trader/lore/MCTS_Trading_Agent_Research_2025_Update.md` — MCTS+MuZero literature context.
- `/home/lukas/Trader/lore/2025-12-31_session_notes.md` — hyperparameter tournament dashboards (DS56/DS78).
- `/home/lukas/.gemini/tmp/trader/chats/session-2025-12-03T05-29-0aca725e.json` — verbatim quote, 2025-12-03 05:31 UTC ("hi i had dort of crash on computer over night…").
- `/home/lukas/Trader/bloom/drafts/_STYLE.md`, `/home/lukas/Trader/bloom/drafts/_OUTLINE.md`, `/home/lukas/Trader/bloom/drafts/_CHRONOLOGY.md` — voice + structure contract for this series.
