# TASK_768 — Base Model Refresh (Phase-1 Imprint + Channel Architecture)

**Status**: DRAFT / ROADMAP — operator-authoritative, not executable. Activates **after the Phase-2 structural-pivot queue closes** (762 → 767 → 763), per operator decision 2026-05-22.
**Created**: 2026-05-22.
**Trigger**: operator flagged "we might need a new base model" while preparing future training. This task scopes what that means and gates it correctly.

## What "base model" means here

Operator clarification (2026-05-22): "base model" covers **two coupled things**, not just the weights.

1. **The Foxglove Phase-1 imprint** — `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`. A supervised imitation imprint of the Poppy `MuZeroNetworks` architecture (8.50M params), trained on oracle demonstrations. Every Phase-2 experiment in the campaign stacks on this one frozen checkpoint.
2. **The input channel architecture** — specifically the **cross-sectional channel mixing** (TASK_732). The operator's words: the channel mixing "makes some approaches not practical." This is a structural constraint baked into the *data representation*, upstream of the weights, and it is the more important half of this task.

A "new base model" therefore means re-deciding the channel architecture **and** re-imprinting Phase-1 — not merely retraining the same design.

## Research findings (codebase + lore deep-dive, 2026-05-22)

Two research agents swept the codebase and daily notes. Synthesis:

### The Phase-1 model

- `MuZeroNetworks` (`mcts_trader/jax_agent/muzero_model.py:243`): V3 `StreamingAssetEncoder` (minGRU temporal scan + one cross-asset self-attention layer) → `TRMReasoning` core (MLP H×L cycles, unique weights per cycle) → value/policy heads. Aux heads (direction, rank, aux-price, Bella-distill) are opt-in and **OFF** in the base imprint.
- **8.50M params, and the distribution is the headline**: `trm/reasoning` = **7.88M (93%)**, encoder = **0.29M (3.5%)**, dynamics/action-embed = 0.33M. The encoder compresses `27ch × 64 timesteps × 64 assets` into a 64-dim state — **~1 latent dimension per asset** (TASK_683 documents this 1,728:1 compression).
- `xs32` is **not a size preset** — it is the TASK_732 "32-channel" run label (`xs` = cross-sectional, `32` = channel count). The checkpoint is the `medium` preset with the encoder widened (`--embed-dim 64 --encoder-heads 8`). Presets (small/medium/large) live in `easy_config.py:163-201`.

### VRAM is NOT the constraint

Operator worried "not sure if we fit into VRAM." Research verdict: **scaling Phase-1 fits comfortably.** The 8.5M model trains at batch 256 with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` (~16.8 GB cap of 24 GB). No OOM has ever been tied to Phase-1 / `train_easy.py` / the encoder in the lore — every OOM on record is Phase-2 / Titans-MAG. A 2-4× larger encoder or reasoning core still fits in 24 GB. **VRAM does not gate a scale-up.**

### But naive scale-up is contraindicated by our own evidence

This is the decisive finding. The project has *already tried* "make the base model bigger / sharper":

- TASK_683 (widen encoder state), TASK_729a/b (per-asset routing, wider encoder) — implemented, did **not** break the Alpha ceiling. The encoder bottleneck was diagnosed and attacked five times (726→728c); each fix moved its own metric and hit the same wall.
- `[[classifier-alpha-anticorrelation]]` (3-way confirmed): TASK_737/740/741/757/758 all show **classifier/encoder-side improvements anti-correlate with deployment Alpha** — "loss DOWN, Bal.Acc UP → Alpha collapses via trade-rate explosion." The encoder/Bella mechanism class is **CLOSED**. Verbatim from the campaign: *"the bottleneck is downstream of the encoder."*

**Implication**: a pure scale-up of the existing 8.5M architecture (bigger encoder, bigger reasoning core, same design) contradicts the documented evidence and should be rejected as the base-model refresh. A base-model refresh is only defensible if it changes **structure** — the channel architecture, the oracle, or the planning surface — not just parameter count.

## The channel-mixing problem (the load-bearing half)

`n_channels = 39` (`easy_config.py:27`): **27 per-asset channels + 5 cross-sectional channels (TASK_732) + 7 agent-state channels (TASK_689)**.

The 5 cross-sectional channels are computed by **mixing features across the 64-asset universe** at each timestep (cross-sectional rank / breadth / relative-strength type features). This means **each asset's input vector depends on all 63 other assets**. The 64-asset universe is baked into the *input representation itself*, not just the model.

They are load-bearing: **738_legacy_27ch** (drop the cross-sectional channels, revert to 27) regressed **−18.44pp Δα, all 3 seeds, all 6 periods** (`[[encoder-bella-class-closed]]` history). So they cannot simply be removed.

But the operator's point stands: this universe-coupling "makes some approaches not practical." Approaches it blocks or complicates (**to be enumerated precisely in the deep-dive below**) plausibly include:

- Variable / different asset universe — cross-sectional channels must be recomputed for any universe change.
- Per-asset independent inference or single-asset deployment — an asset cannot be scored in isolation.
- Action-space or routing changes that assume per-asset independence.
- Transfer to a different dataset (e.g. Iris-DEX / TASK_765) — the cross-sectional construction has to be re-derived for the new universe.

**This is the genuine motivation for a new base model**: redesign the channels so the cross-sectional *signal* is preserved (it's worth +18pp) but the *coupling* is restructured — e.g. cross-sectional features computed as a model-internal pooling step rather than baked into the static input, so the universe is a runtime argument rather than a frozen assumption.

### Channel deep-dive — COMPLETE (2026-05-22)

**Where the 5 cross-sectional channels are computed**: `mcts_trader/env/jax_encoder.py:491-535`, inside `encode_observation_temporal` (the 32-channel TASK_732 encoder). They are **input preprocessing** — computed before the model ever sees the data, written into the `(32, window_size, n_assets)` input tensor, not a model component:

| Ch | Feature | Universe operation |
|----|---------|--------------------|
| 27 | rank of current 1-step return | `argsort(argsort(cur_r1)) / (n−1)` — rank across all 64 assets |
| 28 | rank of 20-step cumulative return | rank across all 64 assets |
| 29 | rolling beta to equal-weighted universe | `cov(asset, mean(all assets)) / var(mean)` |
| 30 | idiosyncratic vol | std of residuals after removing `beta · universe` |
| 31 | cross-sectional momentum | `cur_r1 − mean(cur_r1)` over all 64 assets |

Each is computed once (current step) and broadcast across the window via `broadcast_scalar` (`jax_encoder.py:531-535`).

**What this couples**: every channel aggregates over the asset axis — `argsort` over n (27, 28), `mean(returns, axis=0)` = equal-weighted universe (29, 30), `mean(cur_r1)` (31). `n_f = n−1` (`:498`) and the fixed 32-wide output tensor hard-code n=64. So each asset's input vector carries 5 features derived from all 63 others, and the 64-asset universe is baked into the **input tensor's shape and values**. (A second, related cross-asset op lives at `mcts_trader/jax_agent/observation_encoder.py:226` `_cross_asset_rank` — note for completeness, not the primary site.)

**What it makes impractical** (the operator's point, now concrete):
- Variable / different universe size — `argsort` over n, `mean` over n, `n_f`, and the 32-wide channel axis all assume n=64. Any universe change forces a full encoder-input recompute.
- Per-asset / single-asset inference — an asset cannot be encoded without the other 63 present.
- Dataset transfer (Iris-DEX / TASK_765) — the cross-sectional block must be re-derived for the new universe.
- It is *why* 738_legacy_27ch had to **delete** the signal to test it (−18.44pp) — there was no way to keep the signal while removing the coupling.

### Restructure design (768 channel axis)

The 5 features are **deterministic functions of the input prices/returns** — no learnable parameters, no gradient target. They can be computed **inside the encoder forward pass** instead of in preprocessing:

1. **Input tensor becomes 27 per-asset channels** (+7 agent-state) — universe-independent, per-asset.
2. **The encoder computes the 5 cross-sectional channels internally**, as a `(5, window_size, n)` block, from the per-asset channels already present (the 1-step and 20-step return channels), at the *same temporal resolution* — then concatenates to 32 and proceeds through the existing pipeline (per-timestep Dense → minGRU → cross-asset attention).
3. **`n` becomes a runtime axis** — the `mean`/`argsort` aggregations run over a dynamic asset dimension, not a static 64. `n_channels` for the *input* drops 32→27.
4. **Validation**: if step 2 reproduces the same broadcast-constant cross-sectional values at the same `(ws, n)` resolution, the relocated encoder is **bit-identical** to the current one on the 64-asset universe — provable by a spot-check, **no re-imprint needed to confirm correctness**. Re-imprint is required only if the relocation cannot be made exact (e.g. the per-asset channels do not expose the raw quantity a feature needs) — in which case it is a re-imprint with the signal *preserved*, expected Alpha-neutral.

**Reframe — this de-risks 768.** The channel restructure is, by intent, **Alpha-neutral**: it relocates a deterministic computation, it does not change the signal (the +18pp worth stays). Its value is purely **unblocking** — it turns the asset universe from a frozen assumption into a runtime dimension, which is the prerequisite for variable-universe training, dataset transfer (765), and per-asset approaches. So 768's channel axis can ship and be validated by a bit-identity / Alpha-parity spot-check, **decoupled from any Alpha claim**. The Alpha question moves to whatever the unblocking then enables.

## Data preconditions — DO NOT repeat the data mistake

Operator instruction (2026-05-22): "don't repeat the same mistake we already made on data."

**The mistake** (`lore/notes_2026-05-12.md` §83): the `trading_data.db` pushed to the server was corrupt at the SQLite page level. The `zstd → rsync` deploy flow has **no source-side `integrity_check`**, so the corruption propagated silently and was only caught server-side; the DB was restored from the Jan-20 snapshot.

**Current state** (verified 2026-05-22): the workstation `trading_data.db` is **integrity-clean** — `PRAGMA quick_check` = `ok`. It is 24 GB, 247M 1-minute bars, 1,193 assets (64-asset universe is a curated subset), coverage 2018-10 → **2025-12-12**.

**Mandatory gates for any base-model retrain:**
1. **Integrity first** — run `PRAGMA integrity_check` (full, not quick) on `trading_data.db` immediately before Phase-1 training, AND before any deploy/copy of the DB. A retrain on a corrupt DB silently poisons the new base model — exactly the failure to avoid.
2. **Staleness decision** — data ends 2025-12-12, ~5 months stale as of 2026-05-22. Decide explicitly: re-fetch from Kraken to extend coverage before imprinting, or accept the Dec-2025 cutoff. A "new base model" arguably warrants fresh data; not blocking.
3. **Close the deploy hole** — add a source-side `integrity_check` step to the deploy/sync flow (`tools/deploy/`) so a corrupt DB can never again propagate undetected. This is a small, standalone fix worth doing regardless of this task.

Data **quantity** is not a concern — 247M minute bars across 18 regime-diverse date-ranges is ample. Lore consistently attributes the plateau to oracle/planning structure, not data scarcity (`[[single-mechanism-exhausted]]`, `[[bull-blindness]]`).

## Relationship to TASK_764

TASK_764 (oracle k-reformulation) **is itself a base-model retrain** — it re-imprints Phase-1 with a different oracle. It is one specific instance of this task. If TASK_768 activates, **fold TASK_764 into it** as the "oracle axis" of the refresh rather than running them as separate Phase-1 re-imprints (each Phase-1 imprint is ~12h workstation / ~4h server per seed — do not pay that twice).

## Candidate axes for the refresh (operator picks)

A base-model refresh can move on one or more of these axes. They are not all equal — ranked by the research verdict:

| Axis | What changes | Verdict from research |
|---|---|---|
| **Channel architecture** | Restructure cross-sectional mixing so the universe is a runtime dimension, not a baked-in input assumption | **Primary candidate.** Directly addresses the operator's "not practical" constraint. Needs the deep-dive above. |
| **Oracle (k or objective)** | Re-imprint with TASK_764's k=8/12, or a different oracle objective | Secondary. Fold in TASK_764. Attacks `[[bull-blindness]]`. |
| **Pure scale-up** | Bigger encoder / reasoning core, same design | **Rejected** — contradicts `[[classifier-alpha-anticorrelation]]`; already tried via TASK_683/729. VRAM allows it but evidence says it won't help Alpha. |
| **Architecture redesign** | New encoder / reasoning core inductive bias | Only if it targets the planning/value surface, not classifier sharpness. Largest effort — effectively Foxglove v2. |

## Timing & host

- **After the Phase-2 queue** (762 Wave 2 → 767 → 763), per operator decision. Activates if those close weak — consistent with the TASK_761 stop-rule (if the cheap Phase-2 levers exhaust, the base model is the remaining frontier).
- **Host split**: Phase-1 imprint runs on the **workstation** (single RTX 3090, ~12h/seed). This means the base-model retrain can proceed on the workstation in parallel with whatever the server runs next — but per operator decision, sequence it *after* the Phase-2 queue rather than overlapping, to keep the campaign interpretable.

## Open decisions for operator

1. **Which axes** — channel architecture alone, or channel + oracle (fold TASK_764)?
2. **Staleness** — re-fetch Kraken data past 2025-12-12 before imprinting, or accept the cutoff?
3. **Comparability** — a new base model breaks the paired-Δα contract with the entire 738–767 campaign. Accept the reset, or keep one a1b1 run on the old base model as a bridge?
4. **Scope of redesign** — minimal (channel restructure only) vs Foxglove v2 (new reasoning core). The minimal path is far cheaper and directly serves the stated motivation.

## Status

- **DRAFT / ROADMAP** — not executable.
- **Blocking work before executable**: the channel-architecture deep-dive (3 items above); operator answers to the 4 open decisions.
- **Activation**: after 762 → 767 → 763 close.

## References

- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the campaign this escalates from.
- `tasks/TASK_764_ORACLE_K_REFORMULATION.md` — the oracle axis; fold in if this activates.
- `tasks/TASK_683_WIDEN_ENCODER_STATE.md`, `tasks/TASK_729a_PER_ASSET_DIRECTION_ROUTING.md` — prior encoder-scaling attempts (didn't break the ceiling).
- TASK_732 (cross-sectional channels), TASK_689 (agent-state channels) — the current 39-channel composition.
- `easy_config.py:13-201` — `EasyConfig` + size presets. `easy_config.py:27` — `n_channels=39`.
- `mcts_trader/jax_agent/muzero_model.py:243`, `trm_flax.py`, `trm_encoder_v3.py` — the architecture.
- `lore/notes_2026-05-12.md` §83 — the data corruption incident (the mistake to not repeat).
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md`, `project_single_mechanism_exhausted.md`, `project_bull_blindness.md` — why naive scale-up is contraindicated.

## Implementation deep-dive (2026-05-22) — channel restructure

A read-only deep-dive agent traced the encoder. The restructure is **feasible, bit-identical, and needs NO Phase-1 re-imprint** — confirming the §161 de-risk.

**The approach — "Option A" (recommended):** split the preprocessing function `_compute_all_channels_temporal` (`jax_encoder.py:491-535`) into `_compute_per_asset_channels` (27, universe-independent) + `_compute_cross_sectional` (5, takes the raw price/return arrays + a runtime `n`). The model-input tensor drops to 27 per-asset + 7 agent = **34**; a thin wrapper re-derives the 5 and concatenates to 39. The universe `n` becomes a runtime arg of the preprocessing — no `nn.Module` signature surgery. (Option B — computing the 5 literally inside `StreamingAssetEncoder.__call__` — needs raw-returns plumbed through 3 files and cannot reproduce channel 28 exactly; rejected.)

**Validation = spot-check, not re-imprint.** The 5 features are parameter-free deterministic functions; relocating their *computation* (same formulas, same raw inputs) is bit-identical. Confirm by `jnp.array_equal` on the 32-channel tensor old-vs-new + one benchmark period for Alpha-parity. No 12h re-imprint to validate correctness.

**Two hard risks** (put in the implementation brief):
1. **float32 discipline** — the cross-sectional math runs float32 today; if relocated code runs in the encoder's bf16 context it diverges. Compute float32, cast at concat. Highest-likelihood accidental non-identity.
2. **Channel order** — must assemble `[0-26 per-asset][27-31 cross-sectional][32-38 agent]`. The base imprint's `temporal_proj.kernel` (`trm_encoder_v3.py:141`, shape `(39,32)`) binds each column to a channel index. Wrong order → not bit-identical → forces a re-imprint.

**Two computation sites, not one** — the cross-sectional block is duplicated: `_compute_all_channels_temporal` (`:491-535`, active temporal path) AND `_compute_all_channels` (`:308-341`, tiled legacy). Update both in lockstep or formally retire the tiled path (confirm no live caller first).

**n=64 sites for the restructure** (input stays n=64, Alpha-neutral): all in `jax_encoder.py` (`n_f`, output tensor shape, `broadcast_scalar`, the 3 universe means) + the `n_channels` constant 39→34 in `easy_config.py:27`, `train_easy.py` (`encoder_n_channels` `:1462`; the `use_state_channels = n_channels > 32` checks at `:1794, 2038, 2502` → `> 27`), `easy_trainer.py:70`. `trm_encoder_v3.py` needs **no change** — its forward pass is already n-runtime up to the configured max.

**OUT of scope** — the variable-universe follow-on (needs re-imprint): the two genuine n=64 *weight-shape* hard-stops — `asset_identity_embed` and `global_proj` (`trm_encoder_v3.py`). The channel restructure alone keeps n=64 and is Alpha-neutral; variable-universe *training* is what it unblocks, separately.

**Dead code confirmed:** `observation_encoder.py:226` `_cross_asset_rank` is legacy (TASK_125), not on the active encoder-v3 path — ignore.

**Effort:** ~1-1.5 days, no GPU re-imprint.

## Deploy DB-integrity check — chain item (the data-mistake guard)

The 2026-05-12 corruption propagated because the `zstd → rsync` DB flow has no source-side integrity check (`tools/deploy/.training-exclude` excludes `*.db` / `trading_data.db.zst` / `.sha256` from the regular deploy — the DB moves via a separate manual step, and a `.sha256` of the *compressed* file does not catch SQLite page corruption: zstd faithfully compresses a corrupt DB and the checksum still matches).

**Spec**: a small `tools/deploy/sync-db.sh` (~20 lines) — (1) `sqlite3 trading_data.db 'PRAGMA integrity_check'`, proceed only on `ok`; (2) `zstd` → `rsync` to server; (3) server-side, re-run `PRAGMA integrity_check` after decompress. Make this the *only* sanctioned DB-deploy path. Standalone, no dependency on the other chain links — implement anytime; it is a prerequisite for any base-model retrain.
