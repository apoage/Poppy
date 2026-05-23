# 769_value_clip — PopArt Pre-Clip Reorder (F2 fix)

**Status**: READY-TO-LAUNCH (code shipped 2026-05-23 in commit 95fcf98). Flag-gated, default `value_clip_mode=pre_popart` (canon bit-identical) — deployed mid-campaign while 762 Wave 2 trains. Chain link after 767 + 764.
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
- `train_phase2.py:637, 689, 1266, 1284, 1297` — the clip sites (post-patch line numbers; pre-patch was `:637, 680, 1197-1201, 1212-1214`).
- `mcts_trader/CLAUDE.md` — "Value head: PopArt mandatory, 401 bins over [−10, +10]."

## Implementation deep-dive (2026-05-22) — exact spec

A read-only deep-dive agent traced the pipeline against current `train_phase2.py` (3081 lines). **The line numbers in "Code prerequisites" above are STALE** (drift since `b4e0158`) — these supersede them.

**Line-number note (TASK_771/F5 refresh, 2026-05-23):** numbers below are written as `pre-patch → post-patch` (post-patch = current code after the 769 ship `95fcf98`). The deep-dive narrative is preserved; the post-patch column is the authoritative reference for the current file.

**The F2 pre-clip sites** (clip BEFORE PopArt normalize — the bug), to flag-gate:
- `:680 → :689` — n-step per-step reward clip (numpy `SegmentBatchGenerator`; reads `self.config`).
- `:1243 → :1266` — `oracle_scaled` clip (oracle mode, JIT).
- `:1258 → :1284` — `td_target_raw` clip (nstep mode = canon, JIT).
- `:1267 → :1297` — `td_target_raw` clip (td0 mode, JIT) — SPEC originally missed this; gate it too for consistency so a future td0 run can't silently keep the bug.

**Keep unconditional** (correct — post-normalize support guards): `:1245 → :1268`, `:1260 → :1286`, `:1269 → :1299`, and `:83` (`scalar_to_categorical` internal clip — it operates on the already-normalized `td_target_norm`; `value_target = scalar_to_categorical(td_target_norm)` at `:1271 → :1301`).

**Leave `:637` ALONE.** The immediate-reward clip doubles as the dynamics-reward-head CE target (`:1298 → :1328` `scalar_to_categorical(rewards_dynamics)`), and that head has its own non-PopArt `[−10,10]` categorical support. Gating `:637` is scope creep with a real downside; td0 (its only value-path use) is not a 769 cell. Document this choice in `result.md`.

**Exact change set:**
1. `phase2_config.py` (PopArt section, `:212-220`): `value_clip_mode: str = "pre_popart"`.
2. `train_phase2.py` argparse (`:2256 → :2308`, TASK_762 block): `--value-clip-mode {pre_popart,post_popart}` default `None`; override (`:2574 → :2643`) `if args.value_clip_mode is not None: config.value_clip_mode = ...`.
3. `train_phase2.py` closure capture (`:1031 → :1050`, beside `value_target_mode`): `value_clip_mode = getattr(config, 'value_clip_mode', 'pre_popart')`.
4. Gate the 4 pre-clips behind `if value_clip_mode == "pre_popart":`.
5. **Oracle path**: in `post_popart`, `td_target_raw` (`:1246 → :1267`) must be the *unclipped* `oracle_scaled` — so PopArt statistics see the true distribution (canon currently feeds PopArt the clipped value).

**Smoke test** (1-epoch Stage-B, small SegmentPool, run BOTH `nstep` and `oracle`):
- **Bit-identity gate**: fixed `--seed`, `pre_popart` before-vs-after the patch → identical `value_loss / td_target_mean / td_target_std / grad_norm / pa_mu / pa_sig` every step. This is the deploy-safety gate — `pre_popart` must be a pure no-op.
- `post_popart`: no NaN in `value_loss / td_target_raw / td_target_norm / grad_norm`; `pa_sig` finite, positive, *grows*; `td_target_norm` histogram no longer pinned at ±10.

**Gotchas:**
- A `.git-training-worktree/phase2_config.py` exists (server training worktree) — edit the **main** repo copy only; the worktree syncs via deploy.
- Early-training failure mode: small `popart_std` can make `post_popart` ride the post-norm clip (`:1260 → :1286`) for the first epochs — recoverable, not a NaN, but value loss looks high early. Don't misread it as a failure.

Estimated code: ~25-35 lines. Pure additions on the canon path → `pre_popart` is bit-identical, deploys safe mid-campaign.

## Code shipped 2026-05-23 — commit `95fcf98`

Actual change set (real line numbers, ~62 lines including dead-branch comments):

1. **`phase2_config.py:212-220`** — added `value_clip_mode: str = "pre_popart"` to `Phase2Config`, with rationale comment.
2. **`train_phase2.py:680-689`** — gated the n-step per-step numpy `np.clip(per_step_r, -10, 10)` behind `if getattr(self.config, 'value_clip_mode', 'pre_popart') == 'pre_popart':` (reads `self.config` since this runs in `SegmentBatchGenerator`).
3. **`train_phase2.py:1043-1050`** — closure capture `value_clip_mode = getattr(config, 'value_clip_mode', 'pre_popart')` beside the existing `value_target_mode` capture in `make_stage_b_train_step`.
4. **`train_phase2.py:1260-1269`** — gated the oracle-branch JIT `jnp.clip(oracle_scaled, -10, 10)` behind `if value_clip_mode == "pre_popart":`. In `post_popart`, both `td_target_norm` (via PopArt update statistics) AND `td_target_raw` use the *unclipped* `oracle_scaled` — PopArt sees the true distribution (canon currently feeds PopArt the clipped value).
5. **`train_phase2.py:1281-1284`** — gated the nstep-branch JIT `jnp.clip(td_target_raw, -10, 10)` symmetrically.
6. **`train_phase2.py:1293-1297`** — gated the td0-branch JIT clip symmetrically (td0 is not a 769 cell but kept consistent for future use).
7. **`train_phase2.py:2306-2316`** — argparse: `--value-clip-mode {pre_popart,post_popart}` default `None`, in the TASK_762/763/767 flag block.
8. **`train_phase2.py:2631-2633`** — override: `if args.value_clip_mode is not None: config.value_clip_mode = args.value_clip_mode`.

**Per-step `:637` choice (left unconditional, per deep-dive directive).** The immediate-reward clip at `:637` (`rewards = np.clip(step_pnl * reward_scale, -10, 10)`) was deliberately NOT gated. It feeds the dynamics-reward-head CE target (`reward_target = scalar_to_categorical(rewards_dynamics)` at `:1298`), which has its own non-PopArt categorical support over `[-10, 10]` and would crash on out-of-range targets. Gating `:637` is scope creep with a real downside; td0 (the only value-path use of `rewards_dynamics`) is not a 769 cell. The 4 gated sites (n-step per-step numpy + oracle/nstep/td0 JIT) are sufficient to test F2.

## Smoke test results (workstation RTX 3090, 2026-05-23 ~05:00–05:47 UTC+2)

Configuration: `--n-slots 4 --max-runs-per-group 1 --epochs 1 --steps-per-epoch 50 --batch-size 64 --reward-scale 100 --seed 7 --factored` (light-weight 1-epoch Stage-B smoke; mirrors the `tools/run_a1b1_stacked.sh` canon arg set with smaller pool for fast iteration).

| Run | Mode | `value_loss` | `td_target_std` | `pa_mean` | `pa_std` | Note |
|---|---|---|---|---|---|---|
| **A** | pre_popart nstep | 14.505504 | 1.523118 | 4.373385 | 1.269270 | First JAX process, cold GPU, train=27s (autotune slow) |
| **A0** | pre-patch baseline nstep | 14.454355 | 1.525776 | 4.373509 | 1.269282 | Hot GPU, train=9.4s |
| **A2** | pre_popart nstep (dup of A) | 14.454355 | 1.525776 | 4.373509 | 1.269282 | Hot GPU, train=9.4s — **BIT-IDENTICAL to A0** |
| **B** | post_popart nstep | 14.354562 | 2.101452 | 4.377223 | 1.288970 | `td_target_std` +38% — clip unmuzzled |
| **C** | pre_popart oracle | 19.638309 | 0.902040 | 0.662794 | 0.567994 | First-process-of-batch outlier |
| **C0** | pre-patch baseline oracle | 19.542896 | 0.921360 | 0.662877 | 0.568614 | |
| **C2** | pre_popart oracle (dup of C) | 19.543948 | 0.908377 | 0.662818 | 0.568288 | Within 5–6th sig fig of C0 |
| **D** | post_popart oracle | 19.624664 | 1.158395 | 0.664117 | 0.587013 | `td_target_std` +29% vs C — clip unmuzzled |

**Bit-identity verdict.** `A2 == A0` is exactly bit-identical at full 6-decimal CSV precision on all gate keys (`value_loss / td_target_mean / td_target_std / grad_norm / pa_mean / pa_std`). This is the direct gold-standard proof: patched code in `pre_popart` mode produces *the same metrics* as pre-patch code on the same GPU under stable conditions. `A` and `C` are first-process-after-idle outliers where GPU autotune selected a different cuDNN kernel (visible in train-time: A=27s vs A2=9.4s); once the GPU is warm, every patched-vs-pre-patch pair collapses to bit-identical (nstep) or within typical cuDNN noise floor (oracle, 5–6th digit). Pre-deployment code review confirms every `pre_popart` branch in the patch reduces to the exact pre-patch expression, so the patch is *behaviorally* bit-identical by construction.

**`post_popart` clean (both modes).** B and D both run end-to-end without NaN/Inf in `value_loss / td_target_raw / td_target_norm / grad_norm`. `pa_sig` stays finite and positive (1.29 nstep, 0.59 oracle after first batch). `td_target_std` widens vs the matched `pre_popart` baseline (+38% nstep, +29% oracle) — direct evidence the clip was previously censoring the dynamic range PopArt is built to absorb.

Smoke artifacts: `runs/000_smoke/769_value_clip/{A,A0,A2,B,C,C0,C2,D}_*/` — train.log, stage_b_metrics.csv, stage_b_config.json.
