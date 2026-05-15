# TASK_749 — Titans Pivot (Neural Long-Term Memory for Poppy)

**Status**: STUB — research-and-draft agent in progress 2026-05-15.
**Source**: Titans paper (Behrouz/Zhong/Mirrokni, Google, Jan 2025).
  - Local LaTeX: `/home/lukas/Trader/docs/arXiv-2501.00663v1-titans/`
  - arXiv: https://arxiv.org/abs/2501.00663
**Motivation**: 734_flow_matching is the first positive Tier-2 result on Poppy (mean Δα +3.38pp, all 6 periods positive). It's a planning-side mechanism. Titans is another planning-side family — a small neural memory module whose weights update at inference time. Worth scoping for Poppy.

---

## Goal of this task

Investigate whether Titans-style neural long-term memory (MAC / MAG / MAL variants) can plug into Poppy's TRM+MuZero architecture, and produce a concrete implementation spec that we can deploy as an experiment SPEC.

The spec produced by this task should — when filled in by the research agent — be detailed enough that the implementation work could be handed off to a different agent or done in a single focused session.

---

## What the research agent should produce (sections to fill in)

### 1. Paper summary
- 1-2 paragraphs: what Titans is, the surprise-weighted update rule, the three architectural variants (MAC, MAG, MAL), and what problem it solves that's relevant to us.
- Key equations (memory update, surprise signal, momentum / weight-decay terms).

### 2. Architecture mapping
- Where does the Titans memory module plug into Poppy?
  - Inside the TRM encoder (replaces or augments cross-asset attention)?
  - Between encoder and value/policy heads (as a long-context buffer over many timesteps)?
  - Inside the dynamics MLP (long-context regime memory)?
- Compare the Titans memory module's interface (`q_t, k_t, v_t → updated memory state`) to TRM's existing structures.

### 3. Variant choice
- Pick ONE variant (MAC vs MAG vs MAL) for the first implementation, with rationale.
- Note which other variant would be the natural Tier-2 follow-up if the first works.

### 4. Concrete code-change plan
- Files to modify (with file:line citations for insertion points):
  - `mcts_trader/jax_agent/trm_encoder_v3.py` (or wherever the encoder lives)
  - `train_phase2.py` (CLI flags, loss wiring)
  - `mcts_trader/jax_agent/phase2_config.py` or equivalent (config dataclass)
  - Any new module file (e.g. `mcts_trader/jax_agent/titans_memory.py`)
- New CLI flags (e.g. `--titans-memory`, `--titans-variant {mac,mag,mal}`, `--titans-memory-dim`, `--titans-surprise-momentum`, `--titans-decay`).
- Loss-side changes (if any) — does the memory module train via the existing MuZero losses, or does it need an auxiliary memorization loss?
- Initialization: how to start with the memory module disabled (zero-init γ-style) so existing checkpoints still load.

### 5. JAX/Flax implementation notes
- Pitfalls specific to this stack (see CLAUDE.md JAX Pitfalls section): no `jax.lax.cond` in `vmap`, dynamic shapes, etc.
- How to handle the test-time gradient updates within a JAX-jit-compiled forward pass (this is the hardest part — Titans's memory needs to update during inference, which is unusual for a JAX model). Options: lax.scan with manual gradient computation, custom_vjp, or persistent state outside the jit.
- Memory footprint estimate: how big is the memory module (MLP dims × token count) on a 64-asset × 60-timestep window. Must fit on 24GB RTX 3090 alongside everything else.

### 6. Smoke-test plan
- Smallest possible test: e.g. add `--titans-memory --titans-variant mac --titans-memory-dim 32` to current Stage-B config, run 10 epochs, verify:
  - Memory tensor in `metrics.csv`
  - No NaN
  - Stage-B losses comparable to baseline (not regressed by an order of magnitude)
- Wall: should be < 30 min on workstation.

### 7. Experiment SPEC sketch
- Sketch an `experiments/749_titans_mac/SPEC.md` with paired 3-seed protocol vs current Stage-B baseline (or vs flow_matching baseline if that gets promoted from 734-x5).
- Promotion criteria: same as Tier-2 (mean Δα ≥ +3pp AND all 3 positive).

### 8. Open questions / risks
- What could go wrong? (training instability from test-time updates? memory ablation needed?)
- What's the fallback if it doesn't work? (Titans-style memory is currently a research direction with no public production deployments at this scale — we may be the first to try it on a trader.)

### 9. Estimated effort
- Code work: small / medium / large?
- Smoke-test debugging: hours?
- First experiment SPEC wall time: based on Tier-2 estimates (~3.75h paired 3-seed).

---

## Constraints (load-bearing for the agent)

- **JAX/Flax** required (current stack). No PyTorch port.
- Must fit a single RTX 3090 (24 GB) alongside existing model + MCTS rollouts.
- Don't break existing Phase-1 checkpoint loading (`jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`). Memory module must be optionally toggleable via flag, zero-init defaults so old ckpts load.
- Stay within the **planning-side mechanism class** — don't try to use Titans memory to "improve classification." If the agent finds the natural fit is in classifier/encoder land, flag it and stop (Poppy's bottleneck is downstream of the encoder, per `memory/project_classifier_alpha_anti_correlation.md`).
- Respect the TASK_742 rule (≤2 concurrent train_phase2 per host) and the GPU-0-forbidden rule (server-side).

## Out of scope (don't draft these into the spec)

- Full Bella retrain to use Titans memory inside the generator. That's Tier-D strategic work; the first experiment should keep Bella fixed.
- Multi-mechanism stacks (Titans + flow_matching + oracle). After-individual-results work.
- A full PyTorch reference port. We're JAX-native.

---

## Operator notes

- Paper LaTeX is at `/home/lukas/Trader/docs/arXiv-2501.00663v1-titans/main.tex` plus `MainText/`, `Appendix/`, `Figures/`, `Tables/`.
- Poppy entry: `train_jax_fast.py`, encoder under `mcts_trader/`. Foxglove entry: `train_easy.py`.
- TRM (Tiny Recursive Model): cross-asset attention encoder + MLP reasoning loops + 2-layer MLP dynamics. NOT a transformer — important distinction when mapping Titans (which assumes a transformer host).
- Subtree CLAUDE.md at `mcts_trader/CLAUDE.md` documents Poppy architecture details.
