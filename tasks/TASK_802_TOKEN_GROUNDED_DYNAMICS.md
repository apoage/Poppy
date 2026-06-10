# TASK_802 — Token-grounded dynamics: next-token prediction head on the MuZero transition model

**Status**: 📋 SPEC (2026-06-09). **Blocked on [[TASK_800]]**; pairs with [[TASK_801]] in the
[[TASK_804]] experiment. **This is the planning-side bet of branch C.**

## Why this surface

Poppy's dynamics is a 2-layer MLP over `[latent, action_emb]` trained on **latent-consistency MSE +
reward CE only** (`mcts_trader/jax_agent/muzero_model.py:570-610`; Stage-A loss
`train_phase2.py:322-361`). Latent-consistency self-supervision is the weakest link in the stack —
it can be satisfied by uninformative latents — and at the canonical eval (MCTS sims=16) every
simulation leans on it.

The [[TASK_799]] grammar PASS is, read correctly, a statement about exactly this: *the next token is
learnably predictable in distribution, beyond order-1, transferably across assets* (+0.58 bits/tok).
That is a **world-model target, not a classifier target**. Grounding the latent transition in the
token space gives MCTS a transition model with observable content — the only injection surface that
changes the **planning** side rather than the perception side, which is where every post-mortem
since TASK_761 says the bottleneck lives.

**Not-C0 guard**: C0 predicted next-bar *sign* as a trade signal and was killed as a forecaster trap
(`lore/notes_2026-06-02.md` §175). This head predicts the next *shape-class distribution* as a
grounding loss on the transition model. It never feeds policy, value, or the trade decision —
delete the head at eval time and the agent's actions are unchanged. Judged only on the [[TASK_804]]
scorecard.

## Design

1. **`NextTokenHead`** in `muzero_model.py`, following the aux-head pattern
   (`AuxBellaShapeHead`, `muzero_model.py:110-179`): consumes the dynamics output `next_state
   (B, state_dim)`. **Factorized** to keep params sane:
   `Dense(state_dim → A·E')` → reshape `(B, A, E')` → shared `Dense(E' → V)`, `E'=16`
   (≈ 64·64·16 + 16·V params ≈ 90 K at V=1376 — vs 5.6 M naive). **Zero-init final layer.**
2. **Targets (v1 — DECIDED, keep it single-step)**: per-asset token id at the **first bar after the
   window end** (`window_start + ws`), from the [[TASK_800]] timeline — one CE term on the first
   dynamics step only. No multi-step unroll in v1 (that's a follow-up once arms 4-5 read out).
   The batch already carries win_len = ws + nstep = 84 bars (`train_phase2.py:579-615`), so the
   target bar is in-batch; slice it in `_prepare_numpy` next to the nstep rewards
   (`train_phase2.py:692-717`). CE masked by `valid` (and by `token_id == -1`).
   **Lag note ([[TASK_800]] verdict)**: targets use the **unshifted** full-stream annotation —
   they are hindsight labels, finality is irrelevant. Only *inputs* carry the L1 2-bar serving
   delay ([[TASK_801]] serving-lag convention). Do not shift the targets.
3. **Where it trains**:
   - **Stage A** (dynamics warmup): add head keys to `DYNAMICS_KEYS` (`train_phase2.py:66-70`) so
     the warmup grounds dynamics from the start — this is the natural home.
   - **Stage B**: loss term added at the aux-loss site (`train_phase2.py:1546-1645`), weight knob
     `next_token_weight` (start 0.1, sweep later), flag `use_token_dynamics` default **OFF**.
   - Respect BUG-017 discipline: value/policy gradients through dynamics stay stop-gradiented
     (`muzero_model.py:717-721`); the token CE gradient flows into dynamics + (optionally) encoder
     per the stage's freeze map — no new stop-gradient exceptions.
4. **Composability**: works with or without [[TASK_801]] front-end. In [[TASK_804]] it is evaluated
   **on top of** the front-end arm (arms 4-5) so real-vs-random separation at the dynamics level is
   attributable.
5. **Checkpoint/optimizer**: same backfill (`initialize_missing_aux_heads`) + 'train' label rules
   as [[TASK_801]] items 5-6.

## Acceptance

- **E1/E2**: flag OFF canon-identical; flag ON zero-init byte-identical at step 0.
- **E3**: token CE decreases on a smoke run AND held-out token CE beats the build-bigram codelength
  baseline on ≥1 asset batch (sanity: the head learns at least what [[TASK_799]]'s LM proved
  learnable).
- **E4**: with the head active, latent-consistency MSE does not degrade >10% vs baseline on the
  same smoke (grounding should not fight consistency; if it does, log and report — that itself is
  a finding about latent collapse).
- **E5**: eval path runs with the head present-but-unused (mctx interop untouched —
  `mctx_search.py:109-114` still sees only `(embedding, action)`).

## Diagnostics to log (cheap, per-epoch)

`next_token_ce` (train + a held-out asset slice), `dynamics_consistency_mse`, and their ratio over
epochs. If grounding works, consistency should hold or improve while token CE drops; if token CE
plateaus at the bigram level, the latent isn't carrying shape information forward — report that
verbatim, it changes the [[TASK_804]] interpretation.

## VRAM note

~90 K params + one `(B, A, V)` logits tensor (single step in v1; 256·64·1376 f32 ≈ 90 MB transient at
V=1376 — fine inside 0.7·24 GB, but compute it in bf16 if the envelope is tight; halve V by choosing
a smaller-vocab language if needed). PREALLOCATE default, `MEM_FRACTION=0.7`, single pinned GPU,
server-only ([[feedback_no_gpu_jobs_on_workstation]], [[feedback_sandbox_experiments]]).

## What NOT to do

- Do not feed token predictions (or their embeddings) into policy/value heads — that recreates C0.
- Do not add tokens below the MCTS root; the grounding lives in training losses only.
- No attention in dynamics (`mcts_trader/CLAUDE.md` rule).
- Commit + report; **never `deploy.sh`** ([[feedback_sub_agent_no_deploy]]).
