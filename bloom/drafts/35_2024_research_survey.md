> *The Apothecary keeps two journals. The first records observations: what the specimen did this season, where the leaves curled, when the bloom failed. The second records intentions: which cultivars to try next, what soil to amend, which graftings the literature recommends for a garden of this latitude. In a year when nothing is planted, only the second journal is updated. The handwriting in the second journal grows neat. The catalogue of recommended cultivars grows long and confident and, for the year of dormancy, entirely untested.*

---

**by Lukas — May 23, 2026**

# The 2024 Research Survey That Said HSV Was Still Best

*On a 2024-dated Markdown file in the project tree, the literature review the project quietly wrote for itself during a year of inactivity, three encoding alternatives and four CNN backbones it tabulated, an entire phase-by-phase training curriculum nobody ran, and the specific way it was locally correct and globally outdated.*

---

## Present-tense grounding

It is the afternoon of May 23, 2026. `workstation` is on. The desk fan is on. The right-hand monitor is showing `nvidia-smi -l 2` ticking through a Bella precompute that has been running since Tuesday and has nothing to do with this article except that it is using the GPU the article is about. Earlier today I copied a file off the second internal SSD — the one git symlinks `.git/objects` to because the system drive isn't big enough — into the project tree, under `docs/prehistory_2024/`. The file is `MCTS Trading Agent Algorithm_research 2024.md`. Older spelling lived in the parent directory: `MCST/`, the S and the T transposed relative to MCTS the way an old hand-drawn label preserves the order in which the letters first occurred to whoever made the label. The 2024 file is about ten kilobytes of Markdown, dated late in that year, and until this afternoon I could not read it from inside the sandbox the diary-writer runs under.

Now I can. So I am.

I want to talk about this file, because its *existence* corrects a story I told in Article 0. Article 0 said the project sat in a drawer for a decade between the 2014 sketch and the February 2025 reactivation. That story is emotionally accurate. It is intellectually false. The drawer was being opened, at least once, late in the dormant period. Someone — me, at some workstation, in some week of a year I now mostly remember as crypto mining and a hot summer — sat down and wrote a literature review for a project that did not exist as code. Eleven kilobytes of survey notes. Three encoding alternatives weighed against each other. Four CNN backbones tabulated with parameter counts and a five-star rubric. A page on MCTS variants with hyperparameter defaults. A whole *training curriculum*, divided into phases by epoch range, with per-phase exploration constants. A 2024-dated document the project's dormant period wrote *to itself*.

Article 0 was about the emotional shape of the dormant decade — the seed in the envelope, the soil failing the touch test, the bookkeeping rather than blame. This article is about the *intellectual* shape of the same period. Article 0 said *nothing was happening*. The file in the tree says *something was happening, occasionally, in a journal nobody saw*.

The two stories are both true. The journal kept writing field guides. The garden produced nothing.

## What the survey covered

The survey is organized like a competent grad student's first-pass literature review. There is an executive summary that restates the goal — *"the best algorithms and techniques for each module of the MCTS trading system, based on current research (2024)"* — and ten numbered sections, each one a serious one-page treatment of a real design choice: state encoding, neural-network architecture, MCTS algorithm, training algorithm, framework integration, exchange API practices, reward function design, hardware optimization targets, evaluation metrics, and a phase-by-phase training curriculum. It closes with a quick-reference table and a numbered list of seven academic references.

Eleven kilobytes. Ten sections. No code in the repository to apply any of it to.

I find the *absence of code* the most interesting fact about the document. It is a survey written for a project that did not exist. It is, structurally, a wishlist disguised as a literature review — the way the hardware spec in the 2014 sketch was a wishlist disguised as a constraint. The form is *I read what's out there and decided what I'd pick*. The function is *I am keeping the idea alive by performing the rituals associated with it.*

That is what a dormant project looks like from the inside, I think. Not a clean stop. A series of unsent letters to the project's future self.

## The HSV defense

The first numbered section is *"State Encoding - Best Algorithms,"* and its first sub-heading is *"Primary: HSV Color Encoding."* The defense is compact enough to quote whole:

> *"### Primary: HSV Color Encoding*
> *\*\*Why\*\*: Intuitive mapping of market features to color channels, CNN-friendly representation.*
>
> *| Channel | Encoding | Research Basis |*
> *|---------|----------|----------------|*
> *| \*\*Hue\*\* | Relative price position (0-1, cyclical) | Captures price momentum and mean-reversion |*
> *| \*\*Saturation\*\* | Support/Resistance strength | Technical analysis integration |*
> *| \*\*Value\*\* | Normalized volume | Activity/liquidity signal |*"
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, §1

Three lines of table. One phrase per channel for the *what*, one phrase per channel for the *why*. The three pros the table is doing in compressed form are: HSV is intuitive (the chart-reader's perception maps one-to-one onto the three channels), HSV is CNN-friendly (three channels is the standard ImageNet shape, the pre-trained weights are at least dimensionally compatible), and HSV captures cyclical patterns (Hue is a phase, angle mod 2π, so "where am I in the recent range" is naturally circular). All three of those pros are real. The table is doing honest work.

What is striking is the *brevity*. The HSV defense in 2024 is three lines of table. The 2014 prompt that the survey is implicitly defending — preserved in `bloom/drafts/_ORIGIN_PROMPT.md` as the original sketch — spent paragraphs on the same encoding. The survey, ten years later, has internalized the argument to the point where it can compress to a table. That compression is itself an artifact of the dormant period: the encoding had become so familiar to me, after a decade of carrying it around mentally, that the literature review felt no need to relitigate it. *Of course* HSV is the primary. The work to be done is on the *alternatives*.

This is the part of the document I want to give its due before I describe how it failed. The HSV defense is *good*. Each row of the table names a real property of a real encoding. If your problem is "encode a chart as an image and feed it to a CNN," HSV is not a bad answer. It is, given the constraint, possibly the best of the small set of obvious answers.

The con of HSV — the one the 2024 survey does not name, because the kickback hadn't happened yet — is that *chart-reading is not where the alpha lives*. The encoding optimizes for a perceptual surface a human can read off a screen. The thing that produces alpha is not the perceptual surface; it is the cross-asset, multi-horizon, microstructural patterns that no human reads off a screen because they don't fit on a screen. The 2024 survey is locally correct (HSV is a good chart-encoding) and globally outdated (chart-encoding is the wrong substrate). This is not a failure of the 2024 author. It is the substrate question that the project would not even know to ask until Article 6's classifier/alpha anti-correlation finding made the encoder-as-bottleneck hypothesis untenable.

The 2024 survey, in defending HSV, is doing the thing a literature review is supposed to do: deciding well *within the substrate assumed by the problem statement*. The substrate is what didn't survive. The choices made within the substrate were all defensible. The lesson, in retrospect, is that good research within wrong premises is still good research, and it is not the research that fails — it is the premise.

## The GAF detour

The most interesting paragraph in the encoding section is the one on Gramian Angular Field. GAF was, in 2024, the *"cutting edge alternative"* — the encoding that the time-series-classification literature had been converging on as the way to turn a 1D price series into a 2D image while preserving temporal correlation structure.

The survey doesn't waffle. It cites a paper, lists three benefits, and gives the math:

> *"### Alternative: Gramian Angular Field (GAF)*
> *\*\*Paper\*\*: \"Deep Learning and Time Series-to-Image Encoding for Financial Forecasting\" (IEEE 2020)*
> *\*\*Benefits\*\*:*
> *- Captures temporal correlation structure*
> *- Works well with pretrained CNN models (ResNet, EfficientNet)*
> *- Can be used as additional channels alongside HSV*
>
> *```python*
> *# GASF encoding formula*
> *phi = arccos(normalized_price)*
> *GASF[i,j] = cos(phi_i + phi_j)*
> *```"*
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, §1

The math is short enough to walk. You take a price series, normalize it to the closed interval [−1, 1], and read the normalized values as cosines of angular positions. That gives you an angle φ(t) = arccos(x̃(t)) for each timestep t. The Gramian Angular *Summation* Field is then a matrix whose `[i, j]` entry is `cos(φ_i + φ_j)`, equivalent by trig identity to `x̃_i · x̃_j − √(1−x̃_i²) · √(1−x̃_j²)`. The matrix is symmetric, square in the lookback window's length, and ResNet-compatible because it is just an image. Off-diagonal entries encode the angular-sum of pairs of timesteps; the main diagonal encodes the auto-correlation.

The survey's third bullet — *"Can be used as additional channels alongside HSV"* — is the one I want to flag, because it is the only place in the entire encoding section where the document hedges against picking a winner. The hedge says: maybe you don't have to choose. Maybe HSV is three channels, GAF is three more, and the CNN backbone takes six. The hedge never got tested because the substrate question retired the venue. Nobody ever stacked HSV and GAF into a six-channel input on this project. The match never got scheduled.

The encoding section also names a third alternative, *Time2Vec*, with the same compact treatment:

> *"### Cutting Edge: Time2Vec Encoding*
> *\*\*Paper\*\*: \"Time2Vec: Learning a Vector Representation of Time\" (2019)*
> *\*\*Use Case\*\*: Add learnable periodic time features to capture:*
> *- Time-of-day effects*
> *- Day-of-week patterns*
> *- Seasonal cycles*"*
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, §1

Time2Vec was, in 2024, the way you injected sinusoidal positional features into a model that was going to consume time-indexed data. The survey's framing — *"complement rather than replacement"* — is the right framing. Sinusoidal positional encodings are still in the production stack in 2026, just not in the form the survey imagined. The Time2Vec line is the one piece of the encoding section that survived the substrate change, and it survived by being a *positional embedding* rather than a *state encoding* — orthogonal enough to the substrate question to outlast it.

GAF never got built. Not in the 2025 May setup attempt, not in the July 2025 Janus blueprint, not in the November 30, 2025 Gemini handoff (the 44 KB `MCTS Trading Agent Full.md` prompt that became the first real generation pass), and not in the 2026 codebase. The reason is the substrate change. By the time the project had an actual working repository — late November 2025 — the encoder had moved off images entirely. Instead of HSV-or-GAF-as-image fed to a CNN, the design that took was a 27-channel time-series tensor (price diffs, volume normalizations, cross-sectional asset-relative channels, regime indicators) fed to a TRM, a small recursive transformer. The image-substrate question — *which encoding is the best image to feed the CNN* — became moot because the substrate stopped being an image.

GAF was, in retrospect, the *correct alternative to defend* given the substrate the survey assumed, and it was the alternative that got dissolved by the substrate change rather than refuted by a head-to-head bake-off. I find this is the most common way ideas die in machine learning work. Not by losing a head-to-head. By being orthogonal to a substrate change that retired the whole branch of the tree they sat on.

## The backbone benchmark

The neural-network-backbone section is the densest and the most outdated. The 2024 survey lays out a four-way comparison with a five-star rubric, which is genuinely the wording it uses:

> *"### Backbone Comparison (for 64x64 image input)*
>
> *| Architecture | Params | RTX 3090 Speed | Accuracy | Recommendation |*
> *|--------------|--------|----------------|----------|----------------|*
> *| \*\*ResNet-18\*\* | 11M | ★★★★★ | ★★★☆☆ | Development/Testing |*
> *| \*\*EfficientNet-B0\*\* | 5M | ★★★★☆ | ★★★★☆ | Production - Best tradeoff |*
> *| \*\*ConvNeXt-Tiny\*\* | 28M | ★★★☆☆ | ★★★★★ | When accuracy critical |*
> *| \*\*Vision Transformer\*\* | 86M | ★★☆☆☆ | ★★★★☆ | Large dataset only |*"
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, §2

The recommendation lands on EfficientNet-B0, *"Production - Best tradeoff,"* on the argument that it is the Pareto-best of the four on the parameter-count vs. accuracy frontier. ResNet-18 stays as the *"Development/Testing"* baseline because the codebase will be smaller and the gradient is easier to debug. ConvNeXt-Tiny is the upgrade path for *"when accuracy critical."* ViT is included only to be dismissed — *"Large dataset only,"* which on a single 3090 with a self-play replay buffer is a polite way of saying *"not for this project."*

The recommendation comes with a code stub, which is the most concrete thing in the entire survey:

> *"```python*
> *class TradingNet(nn.Module):*
> *    def __init__(self):*
> *        self.backbone = efficientnet_b0(pretrained=False)*
> *        self.backbone.classifier = nn.Identity()  # Remove classifier*
> *        ...*
> *        # Policy head (action probabilities)*
> *        self.policy = nn.Sequential(*
> *            nn.Linear(512, 128),*
> *            nn.ReLU(),*
> *            nn.Linear(128, 5)  # 5 actions*
> *        )*
> *        # Value head (state evaluation)*
> *        self.value = nn.Sequential(*
> *            nn.Linear(512, 128),*
> *            nn.ReLU(),*
> *            nn.Linear(128, 1),*
> *            nn.Tanh()*
> *        )*
> *```"*
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, §2

The `Tanh()` at the value head's output is the AlphaZero-canonical choice, value in [−1, 1]. The five-action policy head matches the five-action `ActionType` enum that the Gemini-bootstrap prompt would later spell out — `STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY`. The dual-head pattern is the AlphaZero pattern. The whole stub is, in spirit, a paste-into-PyTorch-and-it-compiles design. None of it ever got pasted.

What is striking about this section, reading it in May 2026, is how *period* it is. By 2024 the right answer to "what backbone do I use" for a time-series problem was probably *none of these* — it was a transformer of some flavor, or one of the time-series-native architectures (TST, PatchTST, TimesNet) that had been published in the previous two years. The CNN backbones in this survey were the standard answer in 2018–2020 and the inherited-by-default answer in 2024. The survey isn't *wrong* exactly; it is *defending the previous architectural generation* on the implicit assumption that the input is going to be an image, which forces the output of the architecture-choice question into the CNN family.

The eventual choice, in 2026, was none of these four. The encoder in the production stack is a TRM — a Tiny Recursive Model, recursive in the sense that the same small block is applied multiple times with shared parameters — that consumes the 27-channel time-series tensor directly. Roughly 1–4 M parameters in the imprint phase, depending on configuration. Not in the table the 2024 survey drew. Not in the family of architectures the survey was choosing among.

The CNN backbone question got *dissolved by the substrate change* in the same way the HSV-vs-GAF question did. Once the substrate stopped being an image, the CNN-vs-CNN bake-off stopped being a question. The 2024 survey's careful Pareto-frontier reasoning over four backbones reads now like a beautifully-prepared meal in a restaurant that closed before service.

I want to be careful not to read backward and call the 2024 author wrong. Given an image substrate, the four-way comparison is the right comparison, and EfficientNet-B0 is the right pick. The error — if it is an error — is one level up: the survey didn't question the substrate. It questioned the *backbone for the substrate*.

That is, again, the same lesson the encoding section taught. Good research within wrong premises is still good research. The thing that retires the work is the premise revision, not a head-to-head loss.

## The MCTS section, which aged best

Section 3 of the survey is *"MCTS Algorithm - Best Variant,"* and this is the one section that aged well. It opens with the AlphaZero-canonical PUCT formula in plain text:

> *"### AlphaZero-style PUCT (Predictor UCT)*
>
> *\*\*Core Formula\*\*:*
> *```*
> *UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))*
> *```*
>
> *\*\*Key Hyperparameters\*\* (optimized for trading):*
>
> *| Parameter | AlphaZero Default | Trading Optimized | Reason |*
> *|-----------|-------------------|-------------------|--------|*
> *| `c_puct` | 1.0 | 1.5-2.0 | Higher exploration for non-stationary markets |*
> *| `num_simulations` | 800 | 100-200 | Sufficient for 5 actions |*
> *| `root_dirichlet_alpha` | 0.3 | 0.5 | More exploration at root |*
> *| `discount` | 1.0 | 0.99 | Future rewards slightly discounted |*"
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, §3

The *"Reason"* column is doing the most work here. *"Higher exploration for non-stationary markets"* is the right intuition for cranking c_puct above the chess default — financial state distributions drift, so the search has to revisit suboptimal-looking branches more often. *"Sufficient for 5 actions"* is the right reason to drop sims-per-move from chess-scale 800 to trading-scale 100–200 — the branching factor is tiny, you don't need a deep search to disambiguate among `STRONG_SELL/SELL/HOLD/BUY/STRONG_BUY`. *"More exploration at root"* via Dirichlet alpha = 0.5 (versus 0.3 in chess) is the same intuition applied to the noise budget. *"Future rewards slightly discounted"* via γ = 0.99 is the standard RL move for a non-terminating environment.

Every one of those hyperparameters survived into the 2026 production stack in some form. `c_puct` lives in the 1.0–2.5 range depending on the run. `num_simulations` was 16 for most of the Foxglove-era benchmarks, has been swept up to 64 in TASK_760, and the survey's 100–200 is in the same order of magnitude. Dirichlet root noise is in the code. Discount factor is around 0.997 in the canonical Stage-B config — the Gemini-bootstrap prompt would later spell it out exactly that way (`discount: float = 0.997` in the `MCTSConfig` dataclass). The 2024 survey got the *shape* of the hyperparameter space right.

The reason this section aged best is that *MCTS engineering is substrate-invariant*. The same tree-search code applies whether the leaves are evaluated by a CNN-on-image or a TRM-on-tensor. The encoding section is bound to the image substrate; the backbone section is bound to the CNN family; the MCTS section is bound only to *"discrete actions, neural network policy + value, self-play."* That is a much wider envelope, and the wider envelope is what let it survive the substrate change unscathed.

There is a lesson here that the 2024 author did not know they were teaching: *substrate-invariant components age better than substrate-bound ones.* If you are writing a survey and you don't know which way the substrate is going to drift, lean on the substrate-invariant choices. They will still be there when the substrate moves under you.

---

> *The Apothecary notes that the seed catalogue lists three cultivars under "for poor soils," and recommends the second. The second cultivar is not planted. None of the three are planted. The catalogue is closed and shelved. The soil is amended for a different family of crop the catalogue did not cover.*

---

## The training curriculum nobody ran

I want to spend a paragraph on section 10 of the survey, *"Training Curriculum,"* because it is the section that most clearly betrays the document's *I-am-writing-to-my-future-self* register. It is a three-phase plan, divided by epoch range, with per-phase reward shapes and exploration constants:

> *"### Phase 1: Basic Learning (Epochs 1-100)*
> *- Simple reward: raw returns*
> *- Low MCTS simulations: 50*
> *- High exploration: c_puct = 2.0*
>
> *### Phase 2: Risk Awareness (Epochs 100-500)*
> *- Add Sharpe component to reward*
> *- Increase MCTS: 100*
> *- Medium exploration: c_puct = 1.5*
>
> *### Phase 3: Production Tuning (Epochs 500+)*
> *- Full reward with risk penalties*
> *- Full MCTS: 200*
> *- Lower exploration: c_puct = 1.0*
> *- Add transaction cost penalties*"*
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, §10

Read this with 2026 eyes and it is hauntingly close to *and yet structurally different from* what the project actually became. The 2026 stack does have a curriculum — Foxglove (Phase 1 supervised imprint) hands off to Poppy (Phase 2 MuZero+TRM with rollouts). The exploration constant does anneal across phases. The reward shape does grow in complexity from raw return through Sharpe-flavored components. The *shape* of the curriculum the 2024 survey sketched is the *shape* of the curriculum that runs in 2026. But the *content* — supervised imprint from an oracle, paired-Δα multi-seed validation, eval-MCTS recalibration as the canonical scorecard, an explicit segment pool across 17 regimes — is content the 2024 survey could not have written because the failure modes that taught those mechanisms had not yet been experienced.

The 2024 curriculum is *correctly shaped, incorrectly populated*. The shape was carried forward; the population was rewritten by evidence the survey did not have.

Section 9, *"Evaluation Metrics,"* has the same character. The survey lists Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor, and Calmar with target values (*"Sharpe > 1.0, Max Drawdown < 20%, Win Rate > 50%"*) and concludes:

> *"### Benchmark Comparison*
> *Always compare against:*
> *1. Buy & Hold baseline*
> *2. Simple Moving Average crossover*
> *3. Random trading (sanity check)"*
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, §9

The 2026 canonical scorecard is *Alpha* — paired-Δα over multiple seeds, recently recalibrated to eval-MCTS at sims=16. Buy-and-hold is in there as a baseline. SMA crossover is not. Random trading is not. The shape — *"always compare against a baseline that doesn't use a model"* — is right. The specific basket of baselines drifted as the project's understanding of what counted as a meaningful comparator drifted.

## The hardware section, which aged into a different kind of wrong

Section 8 is *"Hardware Optimization Targets,"* and it is the section that hurts the most to re-read in 2026, for opsec-flavored reasons. The survey writes for a single RTX 3090 — *"RTX 3090 Utilization Goals,"* targets of *"GPU Utilization >70%, Memory Usage 18-20GB, Throughput 1000+ states/sec, Training Speed 50+ episodes/hour"* — and breaks down a 24 GB memory budget by component. The companion `MCTS Trader Compact Generation.md` file in the same prehistory directory makes the constraint explicit:

> *"CONSTRAINTS:*
> *- Hardware: 1x RTX 3090 (24GB), 48GB RAM, Ryzen 5800X*
> *- Exchange: Kraken API*
> *- Language: Python 3.10+*
> *- Framework: PyTorch 2.0+"*
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trader Compact Generation.md`

That is the workstation I am writing this on. Single 3090, single 24 GB, Ryzen 5800X, Kraken-flavored crypto data. The 2024 survey wrote for `workstation` *before workstation existed under that name*, on the assumption that the box would have one 3090 in it. By 2025 the box briefly had a CMP 90HX in the second PCIe slot for the parallel mining business; by May 17, 2026, the 90HX was removed and the survey's single-3090 assumption was, retroactively, correct again. The hardware section did not age out. It aged *into* — the 2024 budget is the 2026 budget, modulo two years of CUDA driver updates and a PSU that is now overkill.

The interesting wrong-spot is the very *last* sub-section of the survey, *"Quick Reference: Algorithm Choices,"* which compresses the whole document into a one-page lookup table:

> *"| Module | Recommended Algorithm | Alternative |*
> *|--------|----------------------|-------------|*
> *| \*\*State Encoding\*\* | HSV Color | Gramian Angular Field |*
> *| \*\*Backbone\*\* | EfficientNet-B0 | ResNet-18 |*
> *| \*\*MCTS\*\* | PUCT (AlphaZero) | Entropy-Guided MCTS |*
> *| \*\*Training\*\* | Self-Play + PER | MuZero Unplugged |*
> *| \*\*Reward\*\* | Differential Sharpe | Risk-adjusted log return |*
> *| \*\*Optimizer\*\* | AdamW | RAdam |*
> *| \*\*LR Schedule\*\* | OneCycleLR | CosineAnnealing |*"*
> — `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md`, "Quick Reference"

Score-keeping in 2026: row 1 (encoding) — both options retired by substrate change. Row 2 (backbone) — both options retired by substrate change. Row 3 (MCTS) — PUCT survived, EG-MCTS not used but not refuted. Row 4 (training) — Self-Play + PER is the loose family the production stack lives in, the Reanalyze trick from MuZero is in the code. Row 5 (reward) — risk-adjusted log return is closer to what the production reward shapes look like than Differential Sharpe is. Rows 6–7 (optimizer, LR schedule) — both alive, both standard. Of seven rows, three are retired by substrate change, three are alive, and one is half-alive. That batting average — call it ~50% — is, I think, an honest result for a one-page survey written eighteen months before the codebase existed. The substrate-bound rows died. The substrate-invariant rows lived.

## The kickback

The kickback on this article is in three layers and none of them is the kickback the rest of the series is built around — there is no negative seed, no paired-Δα reversal, no "the oracle is broken" moment in a chat log. The 2024 survey never had its claims tested by code. The kickback is, instead, a retrospective one: the survey is read in 2026 with the benefit of eighteen months of evidence the survey did not have, and the layers of *what it got and what it missed* land in order.

**Layer one: the 2024 survey is locally smart and globally outdated.** Every choice it makes — HSV as primary, GAF as alternative, the four-way CNN backbone bake-off, EfficientNet-B0 as the production pick, PUCT with c_puct = 1.5–2.0, the three-phase training curriculum — is defensible *given the substrate it assumes*. The substrate is what didn't survive. The choices made *within* the substrate were all good. The choice that retired the work was the unstated assumption that the substrate would be an image fed to a CNN. That assumption was inherited from the 2014 prompt and never re-examined inside the survey. The survey's job was to weigh options within the substrate; it did that job well. The substrate-revision job belonged to the next document — `ALPHAZERO_TRADING_VISION.md` in December 2025 — and that document only half-did it (it pivoted to a 27-channel time-series tensor without spelling out *that this was a substrate change*; the change was just *there*, in the data-loader signature, as a fait accompli). The 2024 survey did its job. The next document changed the job.

**Layer two: the dormant decade had intellectual continuity that the chat history doesn't show.** Article 0 said the project sat in a drawer for a decade. The chat history corroborates that — `_GENESIS_CHAT.md` notes that the 2024 Claude.ai history shows roughly zero trading work and the trading idea was dormant. But the *SSD* shows otherwise. The 2024 survey exists on disk. It was written. Someone, in some week of a year that I now mostly remember as crypto mining and a hot summer, sat down and wrote eleven kilobytes of literature review for a project that did not exist. That writing did not show up in any chat conversation because it was not written *with* an LLM — it was written *to* the project's future self, longhand-style, in a Markdown editor. The chat history captures the conversations. It does not capture the journaling. The journaling happened. The seed catalogue was being updated even in years the seed was not planted. The first Feb 2025 reactivation conversation read, in Article 0, like a typo-strewn fresh-from-cold-start prompt. *"How feasible would be alpha go aproach to trading."* What I did not say in Article 0 is that Feb 2025 was a fresh start with a *cached* literature review behind it. The reactivation prompt was not as cold as it looked. The drawer had been opened recently. The catalogue was current.

**Layer three: the most flattering observation about my 2024 self is that he was already wrong about the substrate but right about the posture.** Doing a literature review. Writing a four-way comparison table with parameter counts and a star rubric. Recommending one option and naming the alternative. Quoting the PUCT formula. Citing seven papers by author and year — Silver on AlphaZero, Schrittwieser on MuZero, Ye on EfficientZero, Niu on LightZero, Zheng on EG-MCTS, Bao on the GAF-for-financial-forecasting paper, Moody & Wu on the differential Sharpe ratio. That is the *posture* of a research engineer working on a real problem. The substrate the posture was applied to was wrong, and most of the specific recommendations didn't survive — but the posture survived. By 2026 the project would run a survey before every major mechanism choice: the value-target ablations, the flow-matching detour, the Titans-MAG memory test, the oracle-k sweep. Every one of those was preceded by some version of "let me read what's out there and decide which alternative to defend." The 2024 survey, read this way, is the first instance of that posture being practiced in this project. The first time the project did its homework before writing code. That habit is now load-bearing in the daily work. It did not arrive in May 2026 fully-formed. It was being rehearsed in 2024 on a problem that did not yet exist.

The kickback, if there is one, is that *I underestimated 2024 me* when I wrote Article 0. Article 0 framed the dormant period as drawer-time. The SSD shows it was journal-time. The two are not the same. A drawer is closed; a journal is being updated.

I want to be careful not to overcorrect. Most of the dormant period really was drawer-time. The 2024 survey is one document, one timestamp, eleven kilobytes. It is not a sustained research program. It is one rainy afternoon in a year that was otherwise consumed by ASIC mining and the question of whether you can power an Antminer off a compost heap. (Yes, I really did ask. No, you cannot.) But it is *more than nothing*, and Article 0's framing was *nothing*. The correction here is not "the dormant decade was secretly busy." The correction is *the dormant decade had occasional handwriting in the second journal, and the second journal turns out to be where the posture got rehearsed*.

## What's left

The 2024 survey is now in the project tree, modification timestamp dated to late November 2024, which is a date I want to flag because it is *almost exactly one year before* the November 30, 2025 Gemini CLI bootstrap — the 44 KB `MCTS Trading Agent Full.md` prompt that became the first Gemini handoff. That second prompt opens with the same hardware spec the survey assumed (*"1x NVIDIA RTX 3090 (24GB VRAM), 48GB RAM, AMD Ryzen 5800X"*) and one of its early `MCTSConfig` defaults — `discount: float = 0.997` — is within rounding distance of the survey's recommended γ = 0.99. The bootstrap prompt is, in places, a code-flavored re-statement of choices the 2024 survey had already argued for in prose. The timestamps may be coincidence; they may be that I opened the survey file in November 2025, glanced at it as part of preparing the Gemini bootstrap document, saved it without editing it, and the mtime updated. The latter is consistent with the survey being the *catalogue I checked before placing the order*. The Gemini bootstrap document was written *with* a year-old literature review open in another tab. That is, I think, the cleanest read of the timestamps.

The HSV-vs-GAF-vs-Time2Vec question is closed. The substrate moved off images; the encoding alternatives moved off the table. Time2Vec, of the three, came closest to surviving — sinusoidal positional encodings are still in the production stack, just not in the form the survey imagined. HSV and GAF are not in the codebase in any form. Neither is mentioned in any `lore/` note after February 2025.

The backbone-vs-time-series-substrate question got resolved sideways. None of the four CNNs in the 2024 table is the production encoder. The TRM that runs in 2026 is in a different architectural family. The Pareto frontier the 2024 survey drew is irrelevant to the architecture that ended up shipping; the frontier was the right frontier *for the wrong problem*.

The MCTS-implementation question is the one section of the survey that aged best. PUCT is the selection rule. The hyperparameter band the survey recommended — c_puct in 1.5–2.0, simulations in 100–200, root Dirichlet alpha around 0.5, discount around 0.99 — is the band the production stack lives in. This is partly because the 2024 survey was right about which way the wind was blowing in tree-search engineering, and partly because the MCTS engineering question is *less substrate-sensitive* than the encoding question. The same tree-search engineering applies whether the leaves are evaluated by a CNN-on-image or a TRM-on-tensor. Substrate-invariant components age better than substrate-bound ones. That is a general lesson the 2024 survey illustrates accidentally.

The training-curriculum section aged into right-shape-wrong-content. The phase structure survived. The specific reward shapes and exploration constants drifted under the weight of eighteen months of evidence the survey did not have.

The hardware section aged into a kind of accidental rightness. Single 3090, 24 GB budget, single workstation. The CMP 90HX detour happened and unwound. The 2026 reality matches the 2024 assumption better than the 2025 reality did.

The discipline of writing surveys before writing code survives. It is what I am doing right now, in fact, in a different file in the same repository, on a different mechanism question. The posture is the load-bearing thing. The specific choices die. The posture is durable. That is, I think, the real durable lesson of reading the 2024 survey in 2026 — not "HSV was wrong" or "the backbones aged out," but "doing the survey is a habit that survives every substrate revision the work goes through." The 2024 survey, in this reading, is not a failed predecessor of the production codebase. It is an early instance of a habit that the production codebase is built around.

That is more flattering than I expected to feel about a document I almost forgot existed until I copied it into the project tree this afternoon.

## Closing

Article 0 said the dormant decade was emotional: a seed in a drawer, a hull intact, a soil that kept failing the touch test. This article says the dormant decade was also *intellectual*: a literature review on the SSD, dated late November 2024, eleven kilobytes, recommending an encoding the substrate change retired, four backbones the substrate change retired, an MCTS implementation that survived, a training curriculum whose shape survived and whose content was rewritten, and a single-3090 hardware budget that turned out to be exactly the budget the 2026 work actually runs under. The garden produced nothing in 2024. The journal kept handwriting field guides. The handwriting got neater. None of the cultivars in the catalogue got planted. The next planting season, when it came, used cultivars the catalogue did not cover, and the catalogue's most durable contribution turned out to be the *habit of keeping a catalogue at all*.

> *The Apothecary closes the second journal. The shelf above the bench holds seven other journals, each one full, each one about a season that did not occur. The journals are arranged by date. The first one is dated more than a decade ago. The most recent one is dated this year. The handwriting is neat. The pages are dry. The catalogues are current.*

---

## Sources

- `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Algorithm_research 2024.md` — primary source. Ten-section Markdown literature review, modification timestamp late November 2024, recently copied into the project tree from the workstation's second internal SSD (the `storage disk` drive that `.git/objects` is symlinked to). Survey covers state encoding (HSV primary, GAF alternative, Time2Vec for periodicity), neural-network backbones (ResNet-18 / EfficientNet-B0 / ConvNeXt-Tiny / ViT comparison table with five-star rubric), MCTS algorithm (PUCT with trading-tuned hyperparameters), training algorithm (self-play + PER + Reanalyze), LightZero framework integration, Kraken API practices, reward function design (log return / Sharpe / Moody-Wu differential Sharpe), hardware optimization targets for a single 3090, evaluation metrics, and a three-phase training curriculum. All block quotes in this article are verbatim from this file.
- `/home/lukas/Trader/docs/prehistory_2024/MCTS Trader Compact Generation.md` — companion file in the same prehistory directory. A "quick start" Gemini-CLI prompt for generating a minimal-but-runnable five-file MCTS trader, with the hardware constraint *"1x RTX 3090 (24GB), 48GB RAM, Ryzen 5800X"* spelled out explicitly. Quoted once in this article for the hardware-constraint line.
- `/home/lukas/Trader/docs/prehistory_2024/MCTS Trading Agent Full.md` — the 44 KB Gemini-CLI bootstrap prompt, dated November 30, 2025, that became the first real generation pass. Cited here for the `discount: float = 0.997` `MCTSConfig` default and the matching single-3090 hardware spec — the bootstrap prompt is, in places, a code-flavored re-statement of the 2024 survey's prose recommendations.
- `bloom/drafts/_ORIGIN_PROMPT.md` — preserves the verbatim 2014/2025 HSV-encoding paragraph the 2024 survey implicitly defends.
- `bloom/drafts/_GENESIS_CHAT.md` — chronology of reactivation chats; the in-repo confirmation that the 2024 survey exists as the surviving artifact of an otherwise-undocumented year.
- `bloom/drafts/00_before_alphago_had_a_name.md` — Article 0, the emotional companion to this article.
- `bloom/drafts/01_break_causality.md` — Article 1, which picks up the December 2025 vision document this survey implicitly fed into.
- `bloom/drafts/06_classifier_alpha_anticorrelation.md` — Article 6, where the substrate-revision lesson lands empirically: encoder-side improvements don't transfer to Alpha, which is the 2026 evidence that retired the 2024 survey's substrate assumption.
- Public references the 2024 survey cites by author and year (per its own §References): Silver et al., *"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"* (AlphaZero, 2017); Schrittwieser et al., *"Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"* (MuZero, 2020); Ye et al., *"Mastering Atari Games with Limited Data"* (EfficientZero, 2021); Niu et al., *"LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios"* (NeurIPS 2023); Zheng et al., *"Entropy-Guided Exploration in AlphaZero"* (MLMI 2024); Bao et al., *"Deep Learning and Time Series-to-Image Encoding for Financial Forecasting"* (IEEE 2020); Moody & Wu, *"Optimization of Trading Systems and Portfolios"* (1997).
