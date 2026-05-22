# 769_value_clip — PopArt Pre-Clip Reorder (F2 fix)

**Status**: NEEDS CODE. Flag-gated, default-OFF (canon bit-identical) — deployable mid-campaign. Chain link after 767 + 764.
**Owner**: workstation implements + smoke-tests + deploys; server agent executes; workstation reviews.
**Reference**: `lore/notes_2026-05-22.md` §163 (full F2 diagnosis + rs200 evidence), `tasks/TASK_770_GEMINI_REVIEW_762_763.md` (F2 finding), `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` (value-pipeline = planning-side class).

## Goal

Fix and test the **F2 finding**: the Phase-2 reward/value pipeline clips targets to `[−10, 10]` **before** PopArt normalization, censoring the dynamic range PopArt exists to absorb. Confirmed by 762 Wave 1's rs200 regression (−1.33pp, all 3 seeds). Test whether clipping **after** PopArt normalization (a) removes the rs200 regression and (b) lifts Alpha at canon scale.

This is a value-pipeline structural change — squarely in the planning-side class TASK_761 pivots toward, and the structural reason 762's `reward_scale` lever capped out.

## Hypothesis

PopArt (van Hasselt 2016) adaptively normalizes value targets via running `popart_mean`/`popart_std`; the value head learns in normalized space. A hard `[−10, 10]` clip **upstream** of that normalization defeats the mechanism — it censors the very dynamic range PopArt was built to handle. The `[−10, 10]` bound is the categorical value head's support (401 bins over `[−10, +10]`); that representability constraint applies to the **normalized** target the head actually predicts, not to the raw, pre-PopArt target.

`reward_scale` multiplies into the clipped quantity (`clip(step_pnl × reward_scale, −10, 10)`), so high `reward_scale` saturates the clip — the cause of 762 Wave 1's rs200 collapse.

**Falsifiable**: the F2-fix variant at `reward_scale=200` must NOT reproduce Wave 1's rs200 regression. If `769-fix-rs200` still regresses ≈ −1.3pp vs the 760 a1b1 baseline, the clip was not the cause — F2 is mis-diagnosed, STOP.

## Code prerequisites (NOT shipped)

Flag-gated, default-OFF — same discipline as 762/763/767 and the F1 fix (canon path bit-identical, so it deploys safely mid-campaign):

1. **`value_clip_mode: str = "pre_popart"`** in `Phase2Config` — `"pre_popart"` = canon (current behavior). Variant: `"post_popart"`.
2. **nstep path** (`train_phase2.py:1212-1214`) and **oracle path** (`:1197-1201`): in `post_popart` mode, normalize first, then clip the **normalized** target (`td_target_norm`) to the support range; reconstruct `td_target_raw` consistently (`td_target_norm × popart_std + popart_mean`).
3. **Per-step reward clips** (`:637` immediate reward, `:680` n-step per-step reward): both are pre-PopArt `[−10,10]` clips on `step_pnl × reward_scale`. In `post_popart` mode, widen or remove them — PopArt downstream absorbs the scale. The implementing agent reads the pipeline and picks the cleanest option; **document the choice in the SPEC update and `result.md`**.
4. **CLI flag** `--value-clip-mode {pre_popart,post_popart}`, plumbed like other Phase-2 overrides.
5. **Smoke test**: confirm `pre_popart` is bit-identical to current canon; `post_popart` runs clean — no NaN, value loss stable, `popart_std`/`popart_mean` track sanely, value-target histogram no longer pinned at ±10.

Estimated code: ~25-35 lines. Estimated wall: ~3h workstation (read pipeline + wire + smoke + commit + deploy).

## Variants

Paired, 3 seeds {7,13,23}:

| cell | value_clip_mode | reward_scale | paired against | tests |
|---|---|---:|---|---|
| 769-fix-rs100 | post_popart | 100 | 760 a1b1 sims=32 | Alpha lift at canon scale |
| 769-fix-rs200 | post_popart | 200 | 760 a1b1 sims=32 + 762 Wave 1 rs200 | does the fix rescue the cell the clip broke |

`pre_popart` at any scale = existing baselines, no recompute.

## Promotion gate

- **Mechanism gate (F2 confirmation)**: `769-fix-rs200` must beat 762 Wave 1's `rs200` (Δα > 0 vs the regressed cell) — the clip fix rescues the regression. If it does not, F2 is mis-diagnosed → STOP, write `result.md`.
- **Standard gate**: `769-fix-rs100` paired Δα ≥ +3pp vs 760 a1b1 under eval-MCTS sims=16 joint, no seed < −5pp → 5-seed extension.
- **Bonus signal**: if `769-fix-rs200` ≥ `769-fix-rs100`, fixing the clip *un-blocks* the `reward_scale` lever — a result that re-opens 762's reward-shaping direction.

## Wall estimate

~12h server (6 variant cells, 2-wide; baselines reused).

## Position in the chain

Code chain: **767 → 764 → 769 → 768**. Flag-gated, so the code deploys safely while 762 Wave 2 runs (canon bit-identical).
Server run order: 769 is the natural **conclusion of the 762 reward-shaping arc** — recommend running it immediately after 762 Wave 2 closes (operator confirms exact slot). It answers the question Wave 1 raised but could not: was rs200's collapse the `reward_scale` lever failing, or the clip censoring it.

## References

- `lore/notes_2026-05-22.md` §163 — full F2 diagnosis, the pre-clip pipeline trace, the rs200 evidence.
- `tasks/TASK_770_GEMINI_REVIEW_762_763.md` — F2 as originally found by the bughunt (downgraded P1→P2, deferred — now promoted to an experiment).
- `train_phase2.py:637, 680, 1197-1201, 1212-1214` — the clip sites.
- `mcts_trader/CLAUDE.md` — "Value head: PopArt mandatory, 401 bins over [−10, +10]."
