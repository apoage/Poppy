# The Book: Poppy & Bella

**Status:** Requires source material
**Type:** Technical documentation with project history
**Scope:** Building an AI trading system with system architecture and decision narrative

---

## Project

A documentation project covering the design and build of an AI trading system. Technical content sufficient to support reproducibility. Historical narrative of design decisions, experiments, and outcomes. Apothecary theme used as organizational structure for component naming.

---

## The Gap

### What Exists
- `lore/` directory: session notes, design docs, evolution
- `tasks/` directory: 116+ feature specs
- The codebase itself: the artifact
- The naming: Poppy, Bella, apothecary theme established

### What's Missing: Ground Zero

**I need you to supply:**

1. **The Origin**
   - When did this start? (Year, context)
   - What was the initial spark?
   - First trading attempt (before Poppy/Bella)
   - What failed and why

2. **The Evolution**
   - Major pivots (when did direction change)
   - Key breakthroughs (moments it clicked)
   - Dead ends (what got abandoned)
   - Timeline of major architectural decisions

3. **The Naming**
   - When did Poppy emerge as a name?
   - When did Bella emerge?
   - The apothecary theme - conscious choice or emergent?
   - Other names considered?

4. **The Person**
   - Background (what you're comfortable sharing)
   - Why trading? Why AI?
   - The obsession driver
   - What "burned out browsing GPU specs at midnight" actually looks like

5. **The Philosophy**
   - Why this approach vs others?
   - What do you believe about markets that others don't?
   - What do you believe about AI that shaped this?
   - The "wife with debit card" philosophy - where does that come from?

---

## Proposed Structure

### PART I: SEEDS
*The problem, the first attempts, why this is hard*

- Chapter 1: The Market is Not Your Friend
  - What markets actually are (adversarial, non-stationary)
  - Why prediction isn't enough
  - The graveyard of trading bots

- Chapter 2: Ground Zero
  - [YOUR ORIGIN STORY]
  - First attempts, first failures
  - The spark that led here

- Chapter 3: The Naming
  - Why Poppy (dreams via MCTS planning)
  - Why Bella (vision of unseen markets)
  - The apothecary philosophy emerges

### PART II: CULTIVATION
*Building the trader - TRM, MuZero, the architecture*

- Chapter 4: The Transformer Reasoning Machine
  - Why transformers for trading
  - The 4 H-cycles, 6 L-cycles
  - Recursive reasoning for markets
  - [TECHNICAL DEEP DIVE]

- Chapter 5: MuZero Dreams
  - Tree search in continuous space
  - 320 actions (5 types x 64 assets)
  - Planning without a model of the world
  - [TECHNICAL DEEP DIVE]

- Chapter 6: The Curriculum
  - L0-L14: basic to arbitrage to microstructure
  - Why progression matters
  - What each level teaches
  - [LORE: the training montage]

### PART III: EXTRACTION
*Building the teacher - Bella, market generation*

- Chapter 7: The Generator Problem
  - Why synthetic data
  - The distribution shift enemy
  - Training on what doesn't exist

- Chapter 8: Bella Sees
  - Flow matching (not diffusion, faster)
  - Mamba and selective state spaces
  - Skeletons: trend, volatility, correlation
  - [TECHNICAL DEEP DIVE]

- Chapter 9: Stylized Facts
  - What makes a market "real"
  - Fat tails, volatility clustering
  - The evaluation problem
  - [LORE: Bella learns to dream real dreams]

### PART IV: DOSAGE
*Reward shaping - the dark arts*

- Chapter 10: The HOLD Problem
  - Why agents learn to do nothing
  - Zero-reward traps
  - The curriculum collapse

- Chapter 11: Burning Money
  - Differential reward (opportunity cost)
  - Market-linked burns
  - "Wife with debit card" mode
  - The 5-stage escalation
  - [LORE: too little does nothing, too much kills]

- Chapter 12: Keeping It Alive
  - PopArt: normalization without forgetting
  - ReDo: resurrecting dead neurons
  - ATC: contrastive learning as life support
  - Cycle balance: forcing diversity
  - [TECHNICAL DEEP DIVE]

### PART V: SIDE EFFECTS
*What went wrong, what was learned*

- Chapter 13: The Graveyard
  - Dead architectures
  - Failed experiments
  - Ideas that seemed good at 3am

- Chapter 14: JAX Pitfalls
  - No cond in vmap
  - Both branches of where execute
  - The sharp edges
  - [TECHNICAL: war stories]

- Chapter 15: The Human Cost
  - Time as the constraint
  - Burnout topology
  - The 3090 screaming at midnight
  - 2km of cable and jerky-making miners
  - [NARRATIVE: what obsession looks like]

### PART VI: SYNTHESIS
*Where it stands, where it goes*

- Chapter 16: The Stack Today
  - Current architecture overview
  - What works, what's fragile
  - The numbers (when you have them)

- Chapter 17: The Research Frontier
  - Hindsight optimization (the 8 prompts)
  - Differentiable DP
  - Self-play with oracle
  - [SPECULATION: where this could go]

- Chapter 18: The Hardware Question
  - The PRO 6000 dream
  - Quad 3090 reality
  - What compute enables
  - [LORE: NVIDIA Time(TM)]

- Epilogue: The Work Continues
  - Open questions
  - What this is really about
  - Why share it

---

## Voice & Style

### Technical Sections
- Don't dumb down
- Show the math when it matters
- Code snippets from actual codebase
- JAX idioms explained but not apologized for

### Narrative Sections
- First person where appropriate
- Honest about failures
- The human behind the code
- Time stamps and session notes as artifacts

### Lore Sections
- Third person, slightly elevated
- Poppy and Bella as entities
- The apothecary metaphor extended
- Can be skipped by pure-technical readers, but add depth

### Fiction Corners
- Short interludes
- Poppy dreaming through tree search
- Bella generating markets that never were
- The poetic version of what the math does

---

## Source Material

### Already Have
- [ ] `lore/DIARY.md` - needs review
- [ ] `lore/notes_*.md` - session notes (Dec 2025 - Jan 2026)
- [ ] `lore/*.md` - design documents
- [ ] `tasks/TASK_*.md` - 116+ feature specifications
- [ ] Codebase comments and structure

### Need From You
- [ ] Ground Zero document (origin story)
- [ ] Timeline of major decisions
- [ ] The failures not documented
- [ ] Philosophy and beliefs
- [ ] What's okay to share vs private

---

## Next Steps

1. **You supply Ground Zero** - the origin, the evolution, the philosophy
2. **Review existing lore** - what's accurate, what's distorted
3. **Outline refinement** - adjust structure based on real history
4. **Chapter drafting** - start with what's clearest
5. **Technical review** - ensure accuracy
6. **Narrative passes** - weave the human in
7. **Lore passes** - add the apothecary layer

---

## Structure

The codebase is the primary artifact. The documentation captures design rationale, experimental results, and lessons from iterations.

---

## Ground Zero: The Archaeology

*Reconstructed from conversation, 2026-01-28. Old notes to be added.*

### The Deep Ancestry

```
ERA 0: Computational Foundations
├── Brainfuck experiments
│   └── Computational minimalism, what can emerge from nothing
├── Game of Life
│   └── Emergence, complex behavior from simple rules
│   └── The intuition that markets might work the same way
└── Early conviction: complex systems are solvable

ERA 1: Graphics & Parallel Thinking
├── Computer graphics
├── Game development
├── Shaders
│   └── GPU-native thinking before ML made it mainstream
│   └── Data as textures, operations as kernels
│   └── This becomes JAX intuition later
├── Photogrammetry
│   └── Extracting structure from noisy observations
│   └── This becomes Bella's market generation
└── Worldbuilding
    └── Naming matters, lore matters
    └── Systems as worlds, not just code
    └── This becomes the apothecary theme

ERA 2: Creative Encoding
├── HSB encoded graphs
│   └── Using color space to represent data
│   └── Creative/non-obvious representations
│   └── This becomes skeleton extraction
├── Pixel-based prediction in shaders
│   └── GPU-native prediction
│   └── Before it was called "deep learning on GPU"
└── GAN networks
    └── First generative model experiments
    └── Precursor to Bella

ERA 3: The Spark
├── Two Minute Papers (YouTube)
│   └── Accessible ML research
│   └── The gateway
├── Monte Carlo AlphaGo report
│   └── Tree search + neural networks
│   └── The MuZero seed planted
├── Atari Gym
│   └── RL becomes tangible
│   └── Agents learning from pixels
└── "Argued with trader guys"
    └── They said it wasn't solvable
    └── You believed it was
    └── Conviction that drives everything

ERA 4: Poppy & Bella Emerge
├── [TIMELINE NEEDED]
├── First trading experiments
├── What failed
├── When the names appeared
├── When the architecture crystallized
└── December 2025: existing lore begins
```

### The Threads That Connect

| Past | Present | Connection |
|------|---------|------------|
| Game of Life | Curriculum L0-L14 | Emergence from simple rules |
| Fractals | Market structure | Self-similarity, patterns at scale |
| Caffeine → Shaders → PyTorch → JAX | Current stack | The GPU compute path |
| HSB encoding | Skeleton extraction | Creative data representation |
| Pixel prediction networks | Bella | Direct lineage |
| Worldbuilding | Apothecary naming | Systems as worlds |
| AlphaGo/MuZero | MCTS trader | Tree search for decisions |
| GANs | Flow matching | Generative evolution |
| "It's solvable" | Everything | The conviction |

### The Timeline Shape

```
├─────────────────────────────────────────┼────┤
│            ~15 YEARS                    │ 2M │
│            sparse                       │dense│
│                                         │    │
│  brainfuck  fractals  caffeine  pytorch │HERE│
│  GoL        shaders   HSB       GANs    │    │
│  graphics   pixels    AlphaGo   Atari   │    │
│                                         │    │
│  experiments, learning, wandering       │FIRE│
└─────────────────────────────────────────┴────┘
                                    Nov 2025 →
```

**15 years of sparse exploration → 2 months of dense crystallization**

This is the book's shape:
- Part I-III: The wandering (compressed, thematic, not chronological)
- Part IV-VI: The fire (detailed, the actual build)

### What's Still Missing

- [ ] Dates and timeline
- [ ] The failed attempts between Era 3 and Era 4
- [ ] When exactly Poppy/Bella names emerged
- [ ] The arguments with trader guys (what they said, what you said)
- [ ] Old notes from shader/GAN era
- [ ] The moment it clicked into current architecture

### Philosophy (Emerging)

**On markets:**
- They're adversarial but not random
- Complex behavior from simple rules (Game of Life intuition)
- Structure exists, can be extracted (photogrammetry intuition)
- Solvable, despite what trader guys say

**On AI:**
- GPU-native thinking (shader background)
- Creative encoding matters (HSB graphs)
- Generation and prediction are linked (GANs → Bella)
- Tree search + learning (AlphaGo spark)

**On building:**
- Naming matters (worldbuilding)
- Systems are worlds
- Lore isn't decoration, it's structure
- The apothecary theme: careful work with powerful tools

---

## Ground Zero Template (For Old Notes)

When you dig out the old notes, capture:

```
### Shader/GAN Era
Date range:
What you built:
What worked:
What failed:
Key code/projects (if any survive):

### First Trading Attempts
Date range:
Architecture:
What it was called (if anything):
Results:
Why it failed:

### The Pivot to Current Approach
When:
What triggered it:
Key insight:

### Naming Emergence
Poppy first appeared:
Bella first appeared:
Apothecary theme conscious or emergent:

### The Arguments
Who were the trader guys:
What they claimed:
What you claimed:
Who was right (so far):
```

---

Ground Zero archaeology in progress. Timeline and old notes pending.
