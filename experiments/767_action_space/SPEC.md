# 767_action_space — Action-Space Restructure (Drop Limit Orders)

**Status**: READY-TO-LAUNCH (code shipped 2026-05-22 in commit 7f41586). Runs after 762 closes, before 763.
**Owner**: workstation implements + smoke-tests + deploys; server agent executes; workstation reviews.
**Reference**: `tasks/TASK_767_ACTION_SPACE_RESTRUCTURE.md`, `tasks/TASK_766_GEMINI_PATHFINDING_SUMMARY.md` §3-4, `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` (Gemini addition, not one of the original 4 directions).

## Goal

Test whether collapsing the action space from **320-dim (64 assets × 5 actions: Sell/Hold/Buy/LimitSell/LimitBuy)** to **192-dim (64 assets × 3 actions: Sell/Hold/Buy)** improves Alpha by (a) shrinking the MCTS branching factor and (b) removing a channel the policy can bleed probability mass into during Phase-2 search.

Phase-2-only — does not touch the Phase-1 imprint or the oracle. Respects the campaign comparability contract.

## Hypothesis

The two limit-order actions widen the MCTS branching factor and dilute the policy distribution. Removing them condenses probability mass onto directional market orders, reduces search-time noise, and attacks the over-trade pathway (`[[classifier-alpha-anticorrelation]]`, `[[single-mechanism-exhausted]]`) by physically removing the model's ability to express noisy limit intents.

**Falsifiable mechanism gate**: the 3-action variant MUST show a reduction in the raw count of executed trades vs the 5-action baseline. If trade count does not fall, the policy was not bleeding capacity into the limit channel and the mechanism did not fire — STOP regardless of Alpha sign.

## Oracle audit — RESOLVED (retires the TASK_767 §Risk)

TASK_767 flagged a risk that the oracle relies on limit orders for bull-regime returns, with a mitigation to audit limit-vs-market usage. **Audit done (workstation, 2026-05-21):** `mcts_trader/jax_agent/path_solver.py:406, 473` confirm the oracle **never emits LimitSell/LimitBuy** — 0% limit usage. Dropping the limit actions therefore cannot degrade the Phase-1 supervised target. The §Risk is retired.

**Consequence for interpretation**: because the oracle never emits limits, the Phase-1 imprint never trains the policy toward limit actions either — the limit logits are already near-dead at imprint. So the mechanism this experiment tests is specifically **(a) the MCTS branching-factor reduction** and **(b) suppression of Phase-2 search-time drift into limit actions** — not "imprint taught the policy limit noise." The SPEC's success criterion (trade-count reduction) still holds; the expected effect size is modest if Phase-2 drift into limits is already small. Worth logging the baseline's limit-action selection rate during the pre-implementation read — if it is already ~0%, expect a null and treat 767 as a cheap branching-factor confirmation rather than a high-EV fix.

## Code prerequisites — SHIPPED (commit 7f41586, deployed 2026-05-22)

All wiring landed default-OFF and bit-identical on the canon path. File:line references:

**Line-number note (TASK_771/F5 refresh, 2026-05-23):** line numbers below have been refreshed against the post-769/post-771 reality. Drift since the original 7f41586 ship is +17 to +45 lines, caused by TASK_769's value-clip insertions (`95fcf98`) and TASK_771's BooleanOptionalAction switch on `--disable-limit-orders`. The `TASK_767:` comment markers at each site remain the authoritative anchor — `grep TASK_767` is drift-proof.

1. **Config + CLI** — `disable_limit_orders: bool = False` in `Phase2Config`
   (`phase2_config.py:39-45`). `--disable-limit-orders` argparse flag at
   `train_phase2.py:2300-2317` (TASK_771/F4 switched it to
   `BooleanOptionalAction`; `--no-disable-limit-orders` now also accepted),
   plumbed to config at `train_phase2.py:2640-2644`.
2. **Policy + BC loss mask** — `train_phase2.py:1060-1090` builds, at JIT-trace
   time, a `(320,)` additive logit mask (`limit_logit_mask`, masked entries
   `-1e9`) and a `(320,) {0,1}` keep mask (`limit_keep_mask`), plus the
   `_mask_renorm_target` helper. Applied inside `loss_fn`: logits masked at
   `train_phase2.py:1182-1183`; CE targets (`mcts_visits`, `oracle_targets`)
   masked + per-asset renormalized at `train_phase2.py:1200-1208`; consumed by
   the factored/joint policy CE (`:1216`, `:1233`) and the oracle BC CE
   (`:1337`). NOTE — `-1e9`, not `-inf`/`finfo.min`: with `finfo.min`,
   `log_softmax` rounds to literal `-inf` and the oracle BC target's tiny
   non-zero limit probability gives `tiny*(-inf)=+inf` CE (caught in smoke v1).
   `-1e9` keeps `log_softmax` finite; the target mask makes the masked CE term
   an exact `0.0` with zero gradient.
3. **MCTS expansion guard** — the load-bearing mechanism. **The canonical
   Stage-B search is FactoredPerAssetSearch (`--factored`), not joint MCTS** —
   it does 1-step lookahead over all 320 (asset,action) pairs, so the guard is
   "mask the limit-action Q-values" rather than "prune tree branches".
   `factored_search.py:190-197` drives masked `(asset,action)` Q-values to
   `-1e9` before the per-asset softmax and the action-selection argmax;
   `factored_search.py:116-145` spreads the epsilon-greedy uniform over valid
   actions only. TASK_771/F1 removed the redundant per-asset mean-shift inside
   `_q_to_policy` (was rounding the valid Q's to a single float32 value under
   the `-1e9` mask). Joint MCTS is also covered: `train_phase2.py:1741-1761`
   builds a `(B,320)` int8 mask passed to `search()` at
   `train_phase2.py:1914-1928`; `MuZeroMCTXSearch` forwards it to
   `mctx.gumbel_muzero_policy` as `invalid_actions` AND
   (TASK_771/F2) into the recurrent_fn closure so interior priors are masked
   too (`mctx_search.py:_recurrent_fn`).
   Eval-MCTS (sims=16) masked in `benchmark_easy.py:458-475` (mask build) +
   `:544-548` / `:633-638` (eval-MCTS call sites), auto-detected from the
   checkpoint config (`benchmark_easy.py:931-942`).
4. **Logging** — per-epoch `mean_trade_rate` + `limit_action_fraction` computed
   in-JIT (`train_phase2.py:1467-1498`), added to the metrics dict
   (`:1543-1544`), the CSV fieldnames (`:1830-1832`) and the CSV row
   (`:2137-2139`), and the epoch log line (`:2085-2087`).
5. **Smoke test** — two 1-epoch factored Stage-B smokes (seed 7) run on the
   workstation 2026-05-22. flag-ON: no NaN/inf, `limit_action_fraction=0.000`
   (search excludes limits). canon: clean, `limit_action_fraction=0.398`.
6. **Post-ship bug-fix (TASK_771/F1)**: the original ship contained a P0
   precision bug in `factored_search._q_to_policy` — the mean-shift before
   softmax collapsed the policy training target to uniform across valid
   actions per asset under the `-1e9` mask (`mean_trade_rate ≈ 0.667 ± 0.02`,
   re-derived in `tasks/BUGHUNT_767_2026-05-23.md` and `tasks/CROSSVALIDATE_767_2026-05-23.md`).
   Fixed pre-launch; smoke re-run with the fix shows `mean_trade_rate` off the
   uniform fingerprint. F2 (mctx interior priors mask) and F4 (Phase-2 resume
   filter) also shipped in TASK_771.

**Mechanism-premise note for the launch**: the canon smoke measured the
baseline `limit_action_fraction` at **0.398** — the 5-action factored search
puts ~40% of its per-asset probability mass on the limit channel. This is far
from the ~0% the SPEC hedged about; there IS substantial limit-channel
capacity to remove, so 767 is a genuine intervention, not a cheap
branching-factor confirmation. The mechanism gate (executed-trade-count
reduction) should be read against this.

## Variants

Paired against 760's a1b1 sims=32 baselines (`runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` + evals — already trained, already eval'd). If 762 promotes a new canonical Stage-B config, re-pair against that instead; default is 760 a1b1.

| cell | disable_limit_orders | action dim | seeds |
|---|---|---:|---|
| 767-baseline | False (canon) | 320 | reuse 760 a1b1 sims=32 |
| 767-3action | True | 192 | {7,13,23} |

## Promotion gate

- **Mechanism gate (early exit)**: 3-action variant must show reduced executed-trade count vs baseline. Else STOP — the limit channel was not where capacity leaked.
- **Standard gate**: paired Δα ≥ +3pp under eval-MCTS sims=16 joint AND no seed < −5pp → 5-seed extension.

## Wall estimate

- ~12h server (3 variant seeds, 2-wide; baseline reused, no recompute).
- Plus ~3-4h workstation code + smoke before launch.

## Why this position in the queue

After 762 (reward shaping) — reward shaping is the cheapest over-trade fix; if it works, 767 may be redundant. Before 763 (entropy reg) — if 762 fails, physically removing the noisy action channel is a harder intervention than soft entropy regularization, and a cleaner test. Before 764 (oracle-k) — 767 is Phase-2-only and does not break the Phase-1 imprint contract. Default campaign order: **762 → 767 → 763 → 764**.

## References

- `tasks/TASK_767_ACTION_SPACE_RESTRUCTURE.md` — Gemini's original proposal.
- `tasks/TASK_766_GEMINI_PATHFINDING_SUMMARY.md` §3-4 — EV ranking (767 = 2nd highest EV/hr).
- `mcts_trader/jax_agent/path_solver.py:406, 473` — oracle audit (0% limit usage).
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md`, `project_single_mechanism_exhausted.md` — over-trade pathway evidence.
