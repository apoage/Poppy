> *Observation: a seed in a paper envelope in the back of the cabinet. I weigh it each year against the same brass counterweight. The hull is dry, the embryo intact, the soil outside the window unready. I return the envelope to its drawer. The growing season is not the planting season.*

---

**by Lukas — May 21, 2026**

# Before AlphaGo Had a Name

*On a 2014 sketch, the decade it spent in a drawer, four afternoons in 2025 when I kept trying to crack it open, a Roman god of doorways who didn't make it, and a compost heap.*

---

## "im super lame"

> *"was crunching alot of tutorials on cuda and shaders and well tetrix master bots and brainfuck .. 2014 i guess .. im super lame"*
> — Lukas, in a 2026-05-21 conversation about origins

That is the sentence I have for what I was doing when I first sketched this project. I want to leave it alone, typo by typo, because smoothing it out would lie about who was holding the pen.

It is the morning of May 21, 2026 as I write the rest of this paragraph. `workstation` is on; the desk fan is on; a second monitor shows `nvidia-smi` ticking through a checkpoint validation that has nothing to do with this article and everything to do with the fact that the same box does both jobs now. The workstation that holds the production codebase is the workstation that holds the draft of the prequel to the diary about the workstation. That recursion is the texture of this entire series — there is no separate room where the writing happens. The writing is one more process on the same scheduler as the training. If `train_phase2.py` segfaults in the next twenty minutes I will leave this paragraph mid-sentence and not come back to it for two hours. That is fine. It is, frankly, the conditions under which everything in this series got written.

Let me unpack it anyway, because the texture matters.

**CUDA tutorials.** The era where if you wanted to do anything heavy on a GPU you read NVIDIA's samples directory and Mark Harris's reduction slides and you wrote your own kernels. There was no `pip install jax`. There was a compiler that ate `.cu` files and produced PTX, a debugger that mostly did not work, and a thousand StackOverflow threads about block sizes.

**Shaders.** Same impulse, lower altitude. GLSL fragment shaders, the parlor trick where you give the GPU one triangle that fills the screen and then do all your math per-pixel. ShaderToy was where you went to feel like a wizard for free.

**Tetris master bots.** A programmer's rite of passage. You write an evaluator — board height, line clears, hole count, bumpiness — greedy-search two tetrominoes ahead, and watch the bot stack at impossible speeds. Anyone who has built one remembers the moment the heuristic weights click and the lines start collapsing in fours. It is the first time you feel the shape of a *policy*.

**Brainfuck.** Eight instructions. Turing-complete. Useless. A flex: I will program at the metal even when there is no reason to. Formal constraint as a stand-up routine.

That stack is not the stack of a deep learning researcher in 2014. It is the stack of a hobbyist who likes machines, likes constraints, and likes the moment a search procedure starts to behave like an opinion. The "super lame" is honest. I was not at DeepMind. I had not published anything. I read the papers other people published and tried to figure out which ones I could play with on the box under my desk.

This matters for what comes next, because in 2014 the right papers were already on arXiv. Mnih's DQN had landed in December 2013. Coulom's Monte Carlo tree search work for Go had been around since 2006. Tian and Zhu's CNN-for-Go paper was 2015. The pieces of what would become AlphaGo were on the table, each one a few hundred lines of code if you were brave. Nobody had yet shown you could *stack* them into a system that beat a human at Go. AlphaGo's Nature paper would not appear until January 2016. The name did not exist yet outside DeepMind.

I want to be careful here, because it is tempting to read backward and say *the idea was obvious*. It was not obvious. It was *available*. Anyone reading the literature in 2014 with a finger and a thumb could see the shape: a deep net produces a policy and a value, a tree search consumes them, the system trains on its own games and bootstraps. The shape was visible. The proof that the shape worked at human-master level on a hard game was not.

I saw the shape. I did not see the proof. And I had — let me state this calmly — the wrong job, the wrong hardware, and the wrong tools to chase it.

So I wrote down a sketch and put it in a drawer.

## What I wrote down

The sketch survives. I have it because I pasted it into a Claude.ai conversation on May 10, 2025 — nearly a decade after the first scribble — and the conversation logged it. The vocabulary had picked up sediment between 2014 and 2025; I'll point at the sediment as we go. The version below is what got pasted that morning. It is structured like a project proposal, headed *"Monte Carlo Search Tree Trader with Color-Encoded Market States,"* and it opens like this:

> *"This project aims to develop an autonomous trading agent that learns optimal trading strategies through self-play, inspired by AlphaZero. The core idea is to represent financial market data (e.g., stock or cryptocurrency prices, volume, and derived indicators) as color-encoded images. These images serve as the 'state' input for a deep learning model, likely a Convolutional Neural Network (CNN), which learns to predict future market states or evaluate current market positions."*
> — original prompt, as preserved in `bloom/drafts/_ORIGIN_PROMPT.md`

The phrase *"inspired by AlphaZero"* is anachronistic — the document as it lives in my notes today is a recapitulation; I cleaned the wording later — but the substance is 2014. The substance is: turn a chart into an image, train a CNN, plan with MCTS, train by self-play.

The interesting part — the one I want to defend even though it died — is the encoding:

> *"Utilizing color spaces like HSV for encoding, where:*
>
> *  * **Hue:** Could represent relative price position within a recent range (cyclical, perhaps with modulo for round numbers).*
> *  * **Saturation:** Could represent the strength of identified support and resistance levels.*
> *  * **Value/Lightness:** Could represent normalized trading volume.*
>
> *Emphasis on proper data normalization (e.g., min-max scaling, Z-score) before color encoding to ensure consistency and prevent features with larger magnitudes from dominating. The goal is to feed data as raw as possible, letting the model learn interpretations."*
> — `bloom/drafts/_ORIGIN_PROMPT.md`

I want to give this its due before I describe how it failed.

A human chart reader scans visually. They do not — in the first second — read numbers. They see *shape*, and they see *color* when their charting software paints up-candles green and down-candles red. The cyclical Hue mapping for "where am I in the recent range" mimics the way a trader, glancing at a chart, asks *am I near the top or the bottom of recent action*. The Saturation-for-support-resistance is doing similar work: a level tested many times is more vivid than one tested once. Volume mapped to brightness lights up the bars that matter.

The encoding gets right that perception is a real layer in trading expertise and CNNs are good at perception. What it gets wrong — I'll get to in the closing — is that perception is not the only layer, and chart-reading is not where the alpha lives. But the wrongness took years to be visible. In 2014 the encoding was a clever observation, and my past self deserves to be told that.

The hardware spec is funnier in retrospect.

> *"GPUs: 4 x NVIDIA RTX 3090 (24GB VRAM each). RAM: 512 GB. CPU: AMD EPYC system (please assume a high core count, e.g., 32 or 64 cores). Storage: Sufficient fast NVMe SSD storage."*
> — `bloom/drafts/_ORIGIN_PROMPT.md`

The RTX 3090 was released in September 2020. The 2014 version of this document, in spirit, could not have specified that hardware — the parts did not exist. The version above is the cleaned-up wishlist that crystallized over the years the sketch sat in the drawer, and even at the moment I pasted it back into an LLM in May 2025, the spec was aspirational. A wishlist disguised as a constraint, which is part of how an idea survives a decade: you periodically pretend the budget exists, so the design keeps thinking at the right scale.

Here is what the wishlist actually became, six months into the build, because the hardware story turns out to be funnier and more textured than a single "I have one 3090, not four" sentence.

The workstation is called **workstation**. It is one box, under a desk, with one **EVGA RTX 3090** in it — the FTW3 Ultra variant, factory-tuned at 420 W TDP, so it has roughly 20–30 % more sustained throughput than a stock 3090 of the same generation. The PSU is a 1600 W gold-rated brick that was bought back when I thought the second GPU was going to stay. The second GPU, until very recently, was a **CMP 90HX** — a 10 GB mining card NVIDIA produced for exactly the use case the parallel business in this house was running. It got removed from this machine on May 17 2026, partly because dual-GPU index mismatches were costing me an hour every other week, partly because the case acoustics with two of them under load were closer to a hairdryer than a workstation. The 90HX now lives in a different chassis with the rest of the mining fleet, doing what it was designed for. The single-3090 setup means `CUDA_VISIBLE_DEVICES=0` is, in 2026, exactly equal to "the only GPU." That is a small daily peace.

The storage on `workstation` is two SSDs. The system drive holds the working tree of `~/Trader`. A separate internal drive holds the *git objects*, symlinked into `.git/objects` because the working tree fills the system drive otherwise. There is no automatic mount entry for that second drive. If it isn't mounted on boot, `git status` claims the repository is broken. I find this out about once a quarter, mount it manually, and get on with my day.

The training in this whole series happens on `workstation`. One 3090. JAX, mixed precision, careful batch sizing. Every campaign in the next nine articles — every multi-seed reversal, every value-head saturation finding, every fourteen-mechanism wipeout in May 2026 — was felt on a single consumer GPU in a single box under a single desk. I want that stated clearly because the AlphaZero canon was set on clusters, and the analogy invites cluster-thinking the budget never supported. The interesting constraint in this project was always *one GPU's worth of time*, not *which dozen GPUs do I use*. There is a separate machine I occasionally use for longer runs, but the architectural decisions and the kickbacks all happen here, where I can hear the fans.

The 2014 wishlist asked for sufficient fast NVMe. The 2026 reality is two SSDs, the occasional unmounted drive, and the irritating peace of a smaller-than-planned setup. The wishlist's tacit assumption — that all of this would be in one room, on one power circuit, with one human paying attention — turned out to be the part that *did* survive contact with the budget, just at a smaller scale than budgeted. Most of what the wishlist was budgeting for, it turned out, I did not actually need. A solo developer in 2026 can carry the architectural complexity a small research team carried in 2018 on one GPU. The compute is enough. The hours are the constraint.

The hardware paragraph in the original prompt is therefore not a wishlist that came true. It is a wishlist that came true *sideways* — and *smaller* — which turns out to be fine. Which is, I think, the only way wishlists from a decade ago ever come true.

The MoE framing was the most fashionable wrong note:

> *"Can be viewed as a type of 'Mixture of Experts' (MoE) system, where different analytical components (e.g., a model for price patterns, a model for volume analysis, a potential LLM for sentiment analysis if incorporated) feed into the MCTS (acting as an executive/gating mechanism)."*
> — `bloom/drafts/_ORIGIN_PROMPT.md`

A sentence that wanted to be sophisticated and ended up a hand-wave. MCTS as a *gating mechanism* for experts is not a thing. MCTS is a planner. The MoE language quietly disappeared by the time the project got picked up again. The "potential LLM for sentiment" line is the sweetest tell — in 2014, GPT-2 didn't exist; "an LLM" wasn't a thing you could bolt onto a system. The phrase betrays a later editing pass. I won't pretend the 2014 document and the 2025 paste are bit-identical. The *shape* was 2014. The vocabulary picked up sediment.

The document closed the way every honest proposal closes — with a request for help:

> *"Please provide a structured plan, code examples for key components (especially the state encoding, CNN structure, and MCTS integration), and advice on tackling this complex project incrementally. Discuss potential challenges and how to mitigate them, particularly regarding overfitting, look-ahead bias, and computational demands."*
> — `bloom/drafts/_ORIGIN_PROMPT.md`

In 2014 there was nobody to ask. You posted that paragraph to a forum and got three replies — one sales pitch and two "interesting, follow up if you make progress." Nobody made progress. The drawer closed.

It opened again, briefly, on February 11, 2025. Not with the full document above — that came three months later — but with a stub. The first conversation was a Claude.ai chat I would only rediscover much later, mining a 2025-10-08 account export. The opening sentence is preserved verbatim, typos and all:

> *"How feasible would be alpha go aproach to trading let's say in gpu optimized space .. well let me first introduce battleground .. we have multiple stocks or tokens or what ever that we want to predict saved as color values interpreted from it's actual value .. and we want to predict future color in there ranges or time-frames idea is to realtime monitor and try to predict movements on multiple stocks or channels based on their past interactions and learned relations does that make a sense"*
> — Claude.ai conversation, 2025-02-11T20:30:53Z

*Does that make a sense.* That is what the reactivation actually looked like. Not a triumph, not a grand return — a typo-strewn one-paragraph question with a hedge at the end. Claude's first reply pushed back politely on exactly the right thing: financial markets are not a board game; the state space is effectively infinite; past does not determine future the way a board position determines legal moves. The objection that would, sixteen months later, become this series' central kickback — *the shape of AlphaZero is not the substrate AlphaZero ran on* — was already on the table in the second message of the second conversation. I did not weigh it. I doubled down. By message four I had introduced the self-play-on-historical-data idea, and Claude had said *"a clever approach,"* and the path forward looked clear in the way a path looks clear to someone who has just decided not to think about an objection.

That was the real moment. Not the document — the document came later.

## What the machine said back

Three months after the February stub, I returned with the structured document I quoted above. The reply ran to thirty pages. The model produced a structured plan, code skeletons for the HSV encoder and the CNN backbone, an MCTS implementation sketch, distributed training notes, a visualization module. The summary at the top of the response began:

> *"Now let me create a practical implementation guide and best practices: I've created a comprehensive implementation of the Monte Carlo Tree Search trading agent you described. Here's a summary of what I've provided:*
>
> *## Key Components Created:*
>
> *1. Main Architecture: Complete implementation of market data processor with HSV color encoding; CNN model with dual heads for policy and value prediction; MCTS algorithm implementation; Trading environment simulation*
> *2. Training System: Experience replay buffer; Self-play training loop; Distributed training across multiple GPUs; Checkpoint saving/loading*
> *3. Visualization Tools: State encoding visualization; Training metrics plotting; MCTS tree visualization; Backtest results analysis*
> *4. Implementation Guide: Step-by-step setup; Best practices for avoiding common pitfalls; Advanced extensions (multi-asset, sentiment integration); Production deployment considerations*

> *## Key Features Implemented:*
>
> *1. State Representation: HSV color encoding where Hue = relative price position (cyclical), Saturation = support/resistance strength, Value = normalized volume. Rolling normalization to avoid look-ahead bias.*
> *2. Neural Network Architecture: Shared convolutional backbone; Separate policy and value heads; Optimized for RTX 3090 GPUs.*
> *3. MCTS Implementation: UCB-based node selection; Neural network guided expansion; Efficient backpropagation.*
> *4. Distributed Training: PyTorch DistributedDataParallel support; Multi-GPU optimization; Gradient synchronization."*
> — LLM reply to the pasted prompt, as preserved in `bloom/drafts/_ORIGIN_PROMPT.md`

I want to land what is strange about that reply, because if you skim it it looks like ordinary 2025 LLM output and you miss the temporal point.

In 2014 you could not have gotten that response. The model that produced it did not exist. Half the papers it implicitly cites were not written. The pip-installable MCTS libraries it assumed (`mctx` shipped in 2022) did not exist. PyTorch DistributedDataParallel was not a stable API until 2018. The prompt is roughly timeless — "what would you do if you wanted to build this" — but the *reply* is from a future in which all the parts had already shipped. The reply is a postcard from the decade the sketch spent in the drawer.

That is the moment I want to mark, because it is the moment Article 0 turns on. The project is older than the model that helped me write its first plan.

## The dormant decade

What was happening in the world between 2014 and 2025 that mattered to a sketch like this one? Four bricks, each one of which I now think was load-bearing.

**Cheap GPUs.** The 3090 launched in September 2020. The used market made them affordable for one person by 2022. In 2014, training a non-trivial CNN with self-play required a cluster. In 2022 it required patience and one box.

**JAX matured.** The library the production stack runs on did not exist in 2014. JAX emerged from Google Brain around 2018 and stabilized into something a solo developer could rely on between 2020 and 2022. The reason I can write a recursive transformer plus a Monte Carlo tree search plus a learned dynamics model in one repository and have it all JIT-compile on a single GPU is that one set of decisions by a small team at Google paid off five years later. I did not pay for that work. I am a downstream beneficiary.

**MCTS became installable.** DeepMind released `mctx` — a JAX-native MCTS library — in 2022. Before that, every serious MCTS project meant writing the tree-traversal logic yourself, with all the subtle bugs that come with vectorising over a batch of tree states. After `mctx`, a competent person could compose MCTS into a training loop in an afternoon. The activation energy collapsed.

**LLM-assisted coding.** This is the one I'm most ambivalent about, and the one that mattered most for a project at this scale being doable by one person. Copilot launched in 2021, ChatGPT in late 2022, Claude got serious in 2023. By 2025 the tools were good enough that a careful person could carry a project of moderate architectural complexity without a team. Not because the LLMs write the hard parts — they don't — but because they absorb the tax. Hours that used to go to boilerplate, library APIs, and tensor-shape debugging at 2 AM compress, and the hours that go to *thinking* expand.

In between those bricks landing, ordinary life kept happening, the way it does. There were other projects — Iris-DEX, a DeFi pipeline that became the sister project to Trader, started on its own time in the same window. There was hardware on the desk that needed elsewhere. There were also, I should admit before going further, a few delightfully wrong tangents that *also* used the same GPU: a compost-heap-thermophile electricity generator proposal that ran to a hundred and sixteen messages in my chat history (yes, I really did ask if you could power an Antminer off a hot pile of food waste; the answer turns out to be *no, but interestingly no*), a Crypto Ninja fruit-slicing game pitch for BTC Prague, a self-playing LED Pacman projection, and a perfectly serious thirty-message thread on what kind of casing a Lenovo K880 keyboard backlight needs under Windows 11. The 2025-10-08 export of my Claude account contains 823 conversations. Approximately *six* of them are about MCTS trading. The rest are everything else.

I asked, in a 2026-05-21 conversation that produced this article:

> *"and yes im doing what i have to do .. having crypto and ai job instead of making games and ai models what a lame guy :D"*
> — Lukas, 2026-05-21

That sentence rhymes exactly with the *"im super lame"* from a decade earlier. Same register, same dry self-assessment. It is not blame. It is bookkeeping. The work and the project lived in different folders, and the GPU was on whichever desk needed it that week. The sketch sat in the drawer not because anything was *wrong*, but because the drawer was a drawer and the world had a lot of other drawers in it. Years where the sketch did not get opened. Years where it got opened, glanced at, closed again because nothing on the desktop had changed *enough* to make it actionable. The seed kept. The soil kept failing the touch test. Compost heaps were involved.

The thing I want to say about this decade, without melodrama, is that the idea was not bad in 2014. It was *inactionable*. The gap between "I see the shape" and "I can build the shape" was, in 2014, several years of full-time research-engineer labor plus a cluster. By late 2025 the gap was a weekend of setup followed by a quarter of focused work — no longer a gap, a *project*, and I have spent the six months since living inside it.

> *The Apothecary checks the seed each season. Some years the soil is wrong; some years the hands are wrong; some years the season ends before either is right. The hull holds. The embryo waits. The check is the practice.*

## The afternoons of revival

There was no single afternoon. There were four, spread across nine months in 2025, each one a little more structured than the last.

**February 11, 2025 — the typo-strewn stub.** The Claude.ai conversation I quoted above. *"does that make a sense."* Forty-eight hours of intermittent back-and-forth across three weeks. No code shipped. The objection ("AlphaGo's substrate is not the market's substrate") got raised and dismissed. The conversation closed without building anything.

**May 10, 2025 — the formalization and the first real setup attempt.** Two conversations on the same day. The first was the 8,846-character structured prompt I quoted from `_ORIGIN_PROMPT.md`. The second, fifty-four messages long, titled *"Setting up MCTS Trading Agent Environment,"* opened with the same kind of half-asleep voice as the February stub:

> *"ok need to get the thing up and running .. help me setup enviroment .. this is my initial problem please feel free to ask aditional pquestion .."*
> — Claude.ai, 2025-05-10

That conversation produced a real directory tree: `mkdir data configs checkpoints logs visualizations models`. A `requirements.txt` with version pins. A numpy upgrade that broke half the other deps and got fixed. A note from me, mid-conversation, that *"python should be called python3"* — the kind of detail you only learn when you are actually running the commands. There was a May 2025 codebase. I no longer know where it lives or whether it still runs. But it existed, briefly, between this conversation and the next one.

**July 13, 2025 — the discourse-rage interlude.** A day before the Janus blueprint, I had a small fit. Someone on Twitter had said, more or less, *"just connect Grok to Robinhood with MCP, it's that simple."* I went and asked Claude whether that statement was valid, knowing it wasn't, looking to be told so. Claude obliged. Then I wrote, slightly hot:

> *"Noone asks for robust solution I'm just reacting to someone's reaction on grok 4 the post itself is painfully stupid want it to download an app I want, tell it to invest for me just talking to it. Not having to spend an hour downloading and filling out everything to open a new thing. Example: just say download robinhood. This is my name, my card number. Invest in Tesla with this much money and it do it, take 10 seconds. Actual functions just by talking to it."*
> — Claude.ai, 2025-07-13

The frustration is the data point. There is a recurring tension in 2024–2025 LLM discourse between *people who say LLM trading is trivially achievable from a tweet* and *people who are actually trying to build it.* I was the second kind, having a moment about the first kind. The kind of moment that is healthy if it lasts twenty minutes and bad if it lasts a week.

**July 14, 2025 — Project Janus.** The next day. I returned with an 8,028-character document titled *"Project Blueprint: 'Janus' MCTS Trader."* Janus, the two-faced god of beginnings and transitions, of doorways — apt for a project that was, at that moment, between substrates and between code names. The blueprint still said PyTorch. The blueprint still said CNN. The blueprint described an agent that did not yet exist, with the calm confidence of a document that has not been built. It asked Claude what to add.

Claude added:

- **Position Sizing Module.** Kelly Criterion, fixed fractional, or volatility-based sizing.
- **Risk Controls.** Hard stops on max drawdown, daily loss limits, position concentration limits. These should override MCTS decisions when necessary. *(Reader: these are exactly the layers that became the five-layer safety oath of `BERSERKER_DESIGN_PHILOSOPHY.md` six months later. The seed survived.)*
- **Market Regime Detection.** A module to detect trending vs ranging vs high-volatility conditions and adapt.

Generic-sound advice. Not wrong, not specific to this project. Janus is the codename I gave the agent during a summer when the agent did not have one, and the codename did not stick because the project did not stick. The apothecary names — Poppy for the planner, Bella for the market generator, Foxglove for the supervised imprint, Yarrow for the code-review tool — came months later, after the next reboot, when I had decided the project deserved a *garden* rather than a *Roman god of doorways*. Janus is now a piece of project lore. I keep it the way you keep a band's first demo cassette. Most of the rhythm is in there, the recording is rough, and nobody bought it.

The Janus document also contained a phrase I want to flag because it is the moment the project decided to be longer than I was: *"This document is designed to be a comprehensive blueprint that could be used to guide development or be fed to an advanced AI for code generation."* Read that twice. The blueprint was designed to be *fed to* the next agent. The agent would not just help me code; the agent would *consume* the document and produce the next document. I was already, in July 2025, planning around the fact that the project would survive my own attention span. That turned out to be correct. The agents I work with now are not the agents I worked with then. The documents have to outlast the model that wrote them.

**November 30, 2025 — the Gemini handoff.** Six months after the May setup attempt, six weeks after a series of four short *"Algorithmic trading system design"* prompts on October 6 that I now read as polite failed restarts, I switched tooling entirely. Pasted a 44 KB Markdown document titled `MCTS Trading Agent Full.md` — yet another revision of the same idea, this time targeting Gemini CLI as the agent — and watched a new chat history open. The hardware spec in that document had been honest-sized for the first time: *1× RTX 3090, 48 GB RAM, Ryzen 5800X.* No more four-GPU fantasy. The first version that stuck.

By December 19, 2025, the project had a working repository, a Python environment that didn't break, a JAX-based encoder, and `ALPHAZERO_TRADING_VISION.md` — the document Article 1 is about. None of that came from one afternoon. It came from four of them, plus everything between.

I want to be honest about one thing — the December 2025 document is not where this project started, and the November 30 Gemini handoff is not where it started either. Each of the four moments above was *also* "where it started," for a few weeks, until it stopped. The seed cracked four times. Three of those times the soil was wrong and the embryo went back into the envelope. The fourth time it took.

If you read Article 1 and it sounds like genesis, it is not. It is one specimen in a series of attempts, each of which I called the beginning, until I noticed that *beginnings are cheap* — what matters is which of the beginnings survives a month.

## What survived from 2014, what didn't

I keep a small ledger in `bloom/drafts/_ORIGIN_PROMPT.md` of what the 2014 sketch contributed to the codebase that exists today. Three buckets.

**Survived.** MCTS as the planner — still in `mcts_trader/`, still the namesake. Self-play and tabula-rasa learning — survived into Foxglove's SegmentPool. Historical data as an *accelerated playground*, timestamps as reference points, years crunched in hours — central to how Foxglove and Bella train. The policy-plus-value dual head — still the architecture, on a different encoder. And the ethos — *feed data as raw as possible, let the model learn interpretations* — has held up across every iteration. The encoders changed; the discipline of minimum-interpretation input did not.

**Died.** The HSV color-encoded image. The CNN backbone. The Mixture-of-Experts framing. PyTorch, which I left for JAX. The four-GPU dream. The LLM-for-sentiment hook. The image because chart-reading isn't where the alpha lives, the CNN because non-stationary time series want different inductive biases, the MoE because the framing wasn't load-bearing, PyTorch because JAX won my bake-off, the hardware because money. None deserve elegies. The shape survived; the substrate had to be replaced.

**Reframed.** The sketch contained one phrase more prescient than I deserved. The original prompt said the system should avoid *"hallucinated false positives from overly interpreted inputs."* I meant that as a warning about over-engineered feature pipelines. Six months into the project, the running finding is that the bottleneck is not in the encoder at all — it is *over-confident policy hallucination downstream of the encoder*, the model trading too often because its value head is too sure of itself. The 2014 phrase named the right *kind* of failure mode at the wrong *location* in the stack. The most flattering thing I can say about my younger self: he got the disease right and the organ wrong.

## Closing

Most of the original sketch did not survive the move from 2014 to 2025. The substrate was wrong. The hardware spec was a wishlist. The framing borrowed words it did not earn. The codename was a god who turned out to be a metaphor. What survived was an *ethos*: plan with search, learn with self-play, feed data raw, keep components small and composable. That is not a small thing — it is what the codebase is built on — but it is not a design. It is a posture. The 2014 sketch did not design the 2026 codebase. It taught me a way of standing in the room while the codebase got built.

Three things I want to leave the reader with, since the next nine articles in this series will not be this nostalgic and I want to earn the warmth here while it's appropriate.

*One.* If you have an idea you keep not building, the question is not whether it is good. The question is which of the bricks it is waiting for, and whether any of those bricks just shipped. In my case the bricks shipped on a staggered schedule — September 2020 for the 3090s, around 2022 for JAX, 2022 again for `mctx`, 2023 for the LLM-as-collaborator era — and the project became actionable not when one brick landed but when the last one did. If you have a drawer with a sketch in it, the world is not always failing the touch test. Sometimes it is just March, and the soil thaws in May.

*Two.* The "im super lame" frame is a permission, not an apology. Almost everyone working in machine learning today learned by reading papers other people wrote and trying things on the box under their desk. The number of useful researchers per hobbyist-tinkering-with-shaders is probably not that different from the number of decent novelists per high-school kid writing in a notebook. The tinkering is the cohort. The cohort is the prior on a future research community. Saying "im super lame" while continuing to work is, mathematically, a fine career strategy. The shame is calibration; the work is what counts.

*Three.* The seed and the soil meet on a schedule you do not control. Article 1 is about the document I wrote in December 2025 with the calm and slightly grandiose air of someone who has just learned the soil will accept them. That document believed things that turned out to be wrong, and Article 1 is honest about which things. But the warmth in *this* article — the warmth of finally writing the thing — is also real, and it is still real today even though the next nine articles will be mostly about kickbacks and reversals. Both can be true. The hull cracked in 2025 even though most of what came out of the cracking has had to be rebuilt since.

> *The seed and the soil met. The hull cracked. The check was over. The garden is messier than the catalogue suggested.*

---

## Sources

- `/home/lukas/Trader/bloom/drafts/_ORIGIN_PROMPT.md` — primary source. The verbatim original prompt (HSV encoding section, hardware spec, MoE framing, closing ask), the LLM follow-up plan ("I've created a comprehensive implementation…"), and the survived/died/reframed map.
- `/home/lukas/Trader/bloom/drafts/_CHRONOLOGY.md` — for context on the December 2025 → May 2026 arc that Article 1 picks up.
- `/home/lukas/Trader/bloom/drafts/01_break_causality.md` — the article this prequel hands off to.
- `/home/lukas/Trader/bloom/drafts/_STYLE.md`, `/home/lukas/Trader/bloom/drafts/_OUTLINE.md` — voice and structure contract.
- Verbatim Lukas quote on origins (*"was crunching alot of tutorials on cuda and shaders and well tetrix master bots and brainfuck .. 2014 i guess .. im super lame"*) — 2026-05-21 conversation, bloom-writer session.
- Public chronology referenced: Mnih et al., "Playing Atari with Deep Reinforcement Learning" (Dec 2013); Silver et al., "Mastering the Game of Go with Deep Neural Networks and Tree Search" (Jan 2016); Silver et al., "Mastering Chess and Shogi by Self-Play" / AlphaZero (2017); Schrittwieser et al., MuZero (2019); DeepMind `mctx` library release (2022); NVIDIA RTX 3090 launch (Sept 2020); JAX initial public release (~2018).
