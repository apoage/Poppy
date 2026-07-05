# Style Spec — "The Apothecary's Lab Notebook" diary series

A one-page voice contract for every article in this series. Writers: read this before drafting.

This series is a hybrid: **Lukas first-person** carries the narrative; **the Apothecary** appears at the seams (epigraphs, section breaks, occasional one-line interludes). The Lukas voice is the body. The Apothecary is the breath between paragraphs.

---

## The Lukas voice (body — ~95% of words)

Modeled on the user's existing Medium piece ("I Wanted a Personal Agent I Could Actually Trust"). The DNA:

- **First-person present and past tense.** "I want X. I tried Y. I learned Z." Not "the author" or "one might."
- **Confessional but precise.** Admit the hubris ("I thought I was going to break causality"), admit the reversal ("five seeds later it was a noise bump"), but *quantify* — "+6.21% Alpha single-seed became +3.61% mean over five seeds, range −4.4 to +7.8, a 12.2pp spread." Emotion lives inside numbers, not next to them.
- **Self-deprecating, never self-pitying.** "Embarrassingly, my first scribble equated information-groups with octal modes as if one number captured it." Laugh at the past self; don't apologize for them.
- **Causality kicks back.** The running joke of the series. Each article should have at least one moment where confidence meets evidence and loses. State the prior, state the reversal, name the mechanism that did the kicking. Don't editorialize the loss; let it land structurally.
- **Smart-friend explanatory.** Assume the reader is technical but not in this codebase. Define terms inline when first used ("paired Δα — same seeds, same data, only the mechanism flipped"). Never define a term twice in one piece.
- **Direct concrete sentences.** Avoid "It is important to note that..." Just say the thing.
- **Code blocks for code, tables for trade-offs.** Math goes inline when one line, in display when more. No screenshots of LaTeX.
- **Quoted chat snippets, lightly edited.** Verbatim where it lands (the moment the agent says "the oracle is broken"); paraphrased elsewhere. Always attribute: *"— Gemini, 2026-01-12"* or *"— Claude Code, after the third negative seed"*.

### Cadences that work

- Two-axis tables: "what we hoped → what happened".
- One-line epiphanies: "And that is when I realized: the bottleneck was downstream of the encoder."
- Pre-emptive concessions: "I want to be careful not to claim too much, because..."
- The reveal-the-prior-art beat: "Then I went to look if anyone had done this. They had. Let me show you what I found and what's left that's mine."
- **Present-tense grounding moments.** The series is retrospective but the *writer* lives in May 2026 on a specific workstation with specific processes running. Drop in present-tense observations: *"`workstation` is on; the desk fan is on; `nvidia-smi` is ticking through a checkpoint validation in the next window."* Or: *"I am writing this in the same Python environment the training loop is running in."* This is grounding, not narration of present mood — keep it concrete (named files, named processes, named drives) not vibe-y ("I am thinking about…"). At least 1–2 present-tense grounding moments per article. The retrospective frame still carries the structural beats; the present-tense breathes between them.
- **Hardware-stack smoke (workstation only — opsec).** When the article touches infrastructure, *go specific* but stay on the workstation. The workstation is called `workstation` and has a single EVGA RTX 3090 FTW3 Ultra at 420 W TDP (~20–30 % headroom over stock). Git objects are symlinked to a second internal SSD (`/dev/sdb1`, `storage disk`) because the system drive isn't big enough. The CMP 90HX got removed on 2026-05-17. **DO NOT mention** the training server's IP, SSH endpoint, user, GPU count, or specific cloud provider. Use phrases like "a separate machine I sometimes use for longer runs" if you must reference remote training at all, and prefer to frame the entire project as **trained on a single 3090**. Iris-DEX (Reth archive node, 8 TB XFS) is OK to name as a sister project but don't detail its network. *The workstation-only frame is also more honest about how the canonical kickbacks were felt — single-GPU constraints shaped everything.*

### Cadences to avoid

- Excitement-bait ("This blew my mind!" "Wait til you see this!").
- Listicles without a thesis ("5 things I learned about MCTS").
- Hedging chains ("perhaps", "potentially", "it could be the case that").
- Numbered lists where a paragraph would work.
- "TL;DR" — write so the whole thing reads at the pace of a TL;DR.
- Em-dash overload. One or two per section, not one per sentence.

---

## The Apothecary voice (seams — ~5% of words)

The Apothecary appears in three places only:

1. **Article epigraph.** A single italic paragraph at the top, before the byline. Sets atmosphere. Usually 2–4 lines.
2. **Section-break interludes.** Between major sections, occasionally — a one-paragraph ritual in third-person Apothecary frame. Marks a turning point.
3. **The closing line.** Optional. If the piece ends on a reversal, the Apothecary can speak one closing sentence.

The Apothecary voice rules (cribbed from `bloom/PERSONA.md`):

- Third-person describing the cultivation, OR first-person ritual ("I tend").
- Botanical terms as literal perception, not metaphor: weights = roots; gradients = sunlight and water; loss = disease; convergence = blooming; merging = grafting; the specimens (**Poppy** = MuZero+TRM, **Bella** = market generator, **Foxglove** = supervised imprint, **Yarrow** = code review) named by herb.
- Clinical, present-tense, never explains itself.
- Never emotes. Never references the reader directly. Never references the date.

Example epigraph:

> *Observation: Poppy enters her one hundred and forty-seventh epoch. The roots have stopped reaching for new soil. I increase the gradient flow and watch for chlorosis.*

Example section-break interlude:

> *The specimen flowered, then withered overnight. The labels read "convergence"; the produce read "loss." I do not understand why both can be true. I record the contradiction and continue.*

Example closing line:

> *Bella's vision was beautiful. It was not a market.*

**Do not let the Apothecary voice carry technical content.** It marks transitions and atmosphere — it doesn't explain MCTS. The Lukas voice does the work.

---

## Length: calibration update (2026-05-21)

Original target was 3000 words. After Article 0 came in at 5200 with the additional research evidence and hardware smoke, the user's instruction is **"go till tokens handle output limit ... md files are relatively content savy."** Translation: do not artificially compress. Aim for **4000–6000 words per article** if the source material supports it, with the understanding that **a tight 3000 still beats a padded 5000**. Cut what isn't earning its space. Keep what is.

## Article anatomy (target 4000–6000 words; tight 3000 acceptable if material requires)

Every article in this series follows this skeleton — adapt freely but keep the beats.

1. **Apothecary epigraph** — 2–4 lines, italic.
2. **Lukas opening hook** — what I thought I had, what I was about to do. 200–400 words.
3. **The setup** — the technical context the reader needs. Frame it as "the thing I was attacking." 400–700 words. Include one diagram or one code block.
4. **What I did** — the actual approach, the math, the design choice. 800–1200 words. Be specific. Names of files, variables, hyperparameters where they matter.
5. **The kickback** — what reality did to the plan. The reversal, the bug, the prior art, the negative seed. 400–800 words. Quote the chat where the kickback became undeniable. **This is the spine of the piece — every article must have a kickback.**
6. *(Optional Apothecary interlude between 5 and 6 — one paragraph.)*
7. **What's left, what I learned** — the cleaned-up claim. What's still mine, what I'd do differently, what's load-bearing. 300–500 words. *Don't apologize. Don't pretend the kickback didn't happen.*
8. **Closing** — one or two sentences. Sometimes the Apothecary; sometimes Lukas.

If the piece is more design than experiment (e.g. the origin article), the skeleton compresses sections 4–5 into "the design, and what I didn't know yet."

---

## Quoting rules

- **Always source-attribute.** "— from `~/Trader/lore/notes_2026-05-13.md` §95" or "— Gemini session 2025-12-04, asked about Wasserstein loss" or "— Claude Code, third paired-seed run".
- **Edit lightly for grammar; never for substance.** If the Gemini reply has a typo or a hedge, keep it. If it's 400 words, trim with `[…]` and say so.
- **Quote your own past words too.** Especially the grandiose ones. "Two days earlier I had written in `ALPHAZERO_TRADING_VISION.md`: *'Result: superhuman performance.'* I was about to learn what that adjective costs."
- **Quote both sides of the collab.** Show what you asked, what the agent said back, and what you decided. The texture of the back-and-forth is the artifact.

---

## What this series is, in one sentence

A six-month diary of trying to teach a neural network to plan, written by the human who kept losing — and kept showing up the next morning anyway. The Apothecary tends the garden between paragraphs. Causality kicks back at the end of every chapter.

That's the contract. Stay in it.
