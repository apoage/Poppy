# CROSSVALIDATE 767 — bug-hunt verdicts

Independent re-derivation of `tasks/BUGHUNT_767_2026-05-23.md` for the action-space
restructure (commit `7f41586`, SPEC `experiments/767_action_space/SPEC.md`).
Read-only audit: git/grep/read only, no execution. The headline F1 P0 is verified
from first principles with no reliance on the hunt's arithmetic.

## Per-finding verdicts (F1-F5)

### F1: CONFIRMED — `_q_to_policy` mean-centering collapses to uniform under masked Q
**Severity: P0** (matches the hunt).

**Independent arithmetic re-derivation** (all in float32 IEEE 754):

For `q = [Q0, Q1, Q2, -1e9, -1e9]` per asset with `Q_i ∈ ~[-20, 20]` (PopArt-denorm
range, `reward + 0.95·value`, value support [-10, 10] × scale 2.0):

1. **Sum**: `Q0 + Q1 + Q2 + (-1e9) + (-1e9) = (Q0+Q1+Q2) - 2e9`.
   - ULP(2e9) = 2^31 · 2^-23 = 2^8 = **256**.
   - `|Q0+Q1+Q2| ≤ 60` is well below 256 → adding it to `-2e9` rounds away.
   - Sum **rounds to exactly `-2e9`** (float32).

2. **Mean**: `-2e9 / 5 = -4e8`. ULP(4e8) = 2^28 · 2^-23 = 2^5 = **32**.

3. **q_centered[0:3]** = `Q_i - (-4e8) = Q_i + 4e8`. Adding `|Q_i| ≤ 20` to `4e8`
   with ULP = 32 → all three valid entries **round to `4e8`** (or at worst to two
   adjacent floats {4e8, 4e8 ± 32}).
   **q_centered[3:5]** = `-1e9 + 4e8 = -6e8` (exact at scale 1e9, ULP 128).

4. **softmax via `jax.nn.softmax`** (verified: `trader_env/.../jax/_src/nn/functions.py:606-612`
   implements `exp(x - x.max()) / sum`):
   - `x_max ≈ 4e8` (the valid entries' value).
   - `x - x_max ≈ [0, 0, 0, -1e9, -1e9]` (the masked entries are at scale `-6e8`,
     minus 4e8 = `-1e9`).
   - `exp(-1e9)` underflows to **exact 0.0** in float32.
   - `exp(0) = 1`.
   - softmax = **`[1/3, 1/3, 1/3, 0, 0]`** — uniform across valid actions.

5. **XLA simplification ruled out**: `softmax(x - c)` ≡ `softmax(x)` *mathematically*,
   but XLA's algebraic simplifier does not perform that rewrite. The mean is a
   data-dependent reduction, not a constant, and `softmax(x - reduce(x))` is not a
   pattern in any HLO simplification pass I can identify. JAX's `jax.nn.softmax`
   itself does an internal `x - x.max()` shift (functions.py:606), but it operates
   on whatever input it receives — including the pre-centered `q_centered`. The
   max-shift inside softmax then sees `q_centered.max() ≈ 4e8` and produces the
   collapse described above. **No mitigation from XLA.**

6. **Where the buggy target lands in training**:
   - `factored_search._q_to_policy` returns `policy_probs` = `[1/3, 1/3, 1/3, 0, 0]`
     per asset (after reshape to `(B, 320)`).
   - `train_phase2.py:1928`: `_actions, visit_counts, stats = search(...)`. The
     `visit_counts` IS this `policy_probs`. Passed to `stage_b_train_step` as
     `mcts_visits` (`:1950`).
   - `train_phase2.py:1203` (factored branch): `mcts_visits_m = _mask_renorm_target(mcts_visits)`.
     Input is already `[1/3, 1/3, 1/3, 0, 0]`. After mask: same. Sum per asset = 1.
     Renormalize: divide by 1. **Output unchanged** (renorm is a no-op on the
     already-uniform target).
   - `train_phase2.py:1216`: `optax.softmax_cross_entropy(policy_logits.reshape(...), mcts_visits_m.reshape(...))`
     trains policy logits toward `[1/3, 1/3, 1/3]` over the 3 valid actions per
     asset — **uniform target, NOT Q-weighted**.

7. **Epsilon-greedy mitigation check**: `phase2_config.py:58` default
   `epsilon_greedy = 0.1`. The function spreads the 10% uniform over **valid actions
   only** (`factored_search.py:138-141`): `valid_mask / n_valid = [1/3, 1/3, 1/3, 0, 0]`.
   Mixing 90% buggy uniform + 10% valid uniform = **`[1/3, 1/3, 1/3, 0, 0]`**.
   Epsilon-greedy does NOT correct the bug — it accidentally matches it.

8. **Action selection (correctly preserved — bug is target-only)**:
   - `factored_search.py:201`: `best_per_asset = jnp.argmax(q_values, axis=-1)`.
     `q_values` here is the MASKED version (line 194). Limits at -1e9 are always
     below any real Q → argmax never picks 3 or 4. **Correct.**
   - `factored_search.py:200, 202`: `q_advantage[3,4] = -1e9 - Q_hold` (rounds to
     `-1e9` since ULP(1e9) = 128 ≫ |Q_hold| ≤ 20). `max(q_advantage)` per asset
     picks the highest valid advantage. **Correct.**
   - The action returned to the trainer is limit-free.

9. **Smoke-metric fingerprint**:
   - `train_phase2.py:1489-1498`: `mean_trade_rate` and `limit_action_fraction` are
     computed from the **raw search output** `mcts_visits` (NOT the masked target),
     summed over the 64-asset axis then divided by total mass (=64 for factored).
   - With the bug: `_act_frac = [1/3, 1/3, 1/3, 0, 0]` → `limit_action_fraction = 0`,
     `mean_trade_rate = 1 - 1/3 = 0.667`.
   - The agent's smoke summary in commit `7f41586` reports
     `limit_action_fraction = 0.000` — **consistent with both the buggy and a correct
     implementation** (search excludes limits either way, since argmax-over-masked
     correctly returns valid actions). It does NOT report `mean_trade_rate`.
   - **Empirical fingerprint**: a clean 3+ epoch flag-ON smoke should log
     `mean_trade_rate ≈ 0.667 ± 0.02` if F1 is real, vs canon's `≈ 0.70` matching
     the trained baseline. No such smoke CSVs are present under
     `runs/767_action_space/` (empty dir, mtime 2026-05-22) or
     `runs/000_smoke/` — the agent's smoke artifacts are unreachable from this
     audit. **Inconclusive only on the *empirical* check; the analytical chain is
     conclusive.**

10. **Canon (flag-OFF) bit-identity**: with `disable_limit_orders=False`,
    `factored_search.py:232-234` calls `self._search_fn(params, rng, observation)`
    (3-arg). The `if action_mask is not None` branch at `:191-194` is dead, so
    `q_values` is never replaced with -1e9 sentinels. `q.mean()` is then near 0
    (Q's are roughly mean-zero), `q_centered ≈ q` with float32 precision, softmax
    is Q-weighted. Bit-identical to pre-767. **No regression on canon.** Confirmed.

**Recommended fix** (priority order matches the hunt, with adjustment):

1. **Best — remove the mean-shift entirely** (one-line change). `softmax(q_values / T)`
   is exactly equivalent mathematically; `jax.nn.softmax` does its own internal
   max-shift for numerical stability (`functions.py:606`). The explicit
   `q.mean(axis=-1)` subtraction was always cosmetic-only and is now the load-bearing
   bug.
2. Alternative: pass `valid_mask` (already plumbed) into the mean: `q.mean(axis=-1, where=valid_mask_r > 0, keepdims=True)`.
3. Alternative: use a smaller mask sentinel like `-100.0` — keeps ULP at the mean
   below the Q-scale.
4. (Robustness add-on, regardless of fix 1-3): a post-fix smoke MUST log and
   compare `mean_trade_rate` between canon and flag-ON for ≥3 epochs. A persistent
   `0.667 ± 0.02` flag-ON reading indicates the bug is still present (or has
   re-appeared via a different path).

The hunt's three recommendations are all sound; recommendation 1 (remove the
centering) is the safest and most local.

**Severity rationale (P0, matches hunt)**: this is the canonical Stage-B search
(per SPEC §3 "the load-bearing mechanism"). The mechanism gate (trade-count
reduction) and any Alpha measurement on the 767-3action variant would be driven
by a degenerate policy target; a null or negative Alpha result would be
wrong-cause-attributed.

---

### F2: CONFIRMED — joint-MCTS interior expansion uses unmasked recurrent priors
**Severity: P1** (matches the hunt).

**Independent verification**:

1. **mctx interior selection is unmasked**: `mctx/_src/action_selection.py:158-195`
   defines `gumbel_muzero_interior_action_selection`:
   ```
   to_argmax = _prepare_argmax_input(
       probs=jax.nn.softmax(prior_logits + completed_qvalues), ...)
   return jnp.argmax(to_argmax, axis=-1)
   ```
   No `masked_argmax`, no `invalid_actions` argument. Only root selection
   (`gumbel_muzero_root_action_selection`, `:155`) calls
   `masked_argmax(..., tree.root_invalid_actions)`.

2. **mctx by design only masks root**: `mctx/_src/search.py:64` documents
   "invalid_actions: a mask with invalid actions **at the root**". `policies.py:206`
   passes `invalid_actions` into `search.search`, which stores it as
   `root_invalid_actions` (line 109) and uses it only at root selection. Interior
   nodes use `tree.children_prior_logits` (the recurrent_fn output) directly.

3. **Our `_recurrent_fn` returns unmasked logits**: `mctx_search.py:113-119`:
   ```
   recurrent_output = mctx.RecurrentFnOutput(
       reward=reward_scalar, discount=..., prior_logits=output['policy_logits'],
       value=value_scalar)
   ```
   `output['policy_logits']` is the prediction-head output from
   `muzero_model.py:721, 725` — never masked.

4. **Impact**: at depth ≥ 1, the tree can expand limit-action branches (indices
   3, 4 within each asset block) if their unmasked prior + Q ranks highest. Value
   backups from those branches flow into the root's `completed_qvalues`, biasing
   the root visit distribution toward branches near limit subtrees.

5. **Bound on the damage**:
   - Root `action_weights` ARE re-masked at `policies.py:225-226`:
     `_mask_invalid_actions(root.prior_logits + completed_qvalues, invalid_actions)`
     → softmax → 0 at limits. So the **final visit distribution at root has 0 mass
     on limits.**
   - Final `action = masked_argmax(to_argmax, invalid_actions)` (`:222`) → never a
     limit.
   - Only the **interior subtree exploration** is affected. Simulations spent
     expanding limit-branch subtrees are wasted compute, and the value backups
     from those subtrees affect the root's `completed_qvalues` (which feeds into
     the final action_weights via softmax).
   - For the SPEC's eval-MCTS sims=16 joint promotion gate, this is a bounded
     (≤16-sim) inefficiency; the bias direction is unclear without an empirical
     measurement of how often limit branches "win" the interior softmax.

6. **Recommended fix** — same as the hunt's: pass a `limit_mask` through the
   recurrent_fn closure and mask `prior_logits` before returning. The change is
   invasive (closure capture in JIT) and the bound is small for sims=16/32; the
   hunt's "no fix for the 767 launch" is reasonable, with the caveat that the
   SPEC's canonical search is **factored**, not joint (so F2 is only load-bearing
   for the eval-MCTS promotion gate).

**Severity rationale (P1, matches hunt)**: efficiency/bias issue, not a correctness
break. The final root action and weights are masked correctly. Mitigation: increase
sims to absorb the wasted simulations. Worth a note in the SPEC, no fix required
for the 767 launch.

---

### F3: PARTIAL — eval-MCTS per_asset decode SELL-bias claim
**Severity: P2** (downgraded from hunt's P1).

**Independent verification**:

1. **The path exists**: `benchmark_easy.py:420-422`:
   ```
   if decode == 'per_asset':
       per_asset = policy_probs.reshape(B, n_assets, 5)
       return np.argmax(per_asset, axis=-1).astype(np.int32)
   ```
   `policy_probs` from mctx joint MCTS has exactly 0 mass on limits (after
   `_mask_invalid_actions` at `policies.py:225-226` → softmax). For un-visited
   assets, mass on valid actions is small and potentially near-tied.

2. **`np.argmax` ties**: numpy's `argmax` returns the **first occurrence** of the
   max. For a row `[a, b, c, 0, 0]` with `a ≈ b ≈ c` (near-tied), argmax returns
   index 0 (SELL) deterministically.

3. **However**, the hunt's claim "policy mass that previously diffused across 5
   actions now diffuses across 3, and the un-visited-asset tie-breaking is
   exercised on every batch" is partially overstated. The diffusion across 3 vs 5
   doesn't *create* the tie — under joint MCTS, un-visited assets always had
   near-tied mass on valid actions even pre-767. What 767 changes is: limits drop
   from a "third option" (which numpy would pick over SELL if its prob happened to
   be highest) to "impossible" — but the un-visited mass on SELL vs HOLD vs BUY
   was already near-tied. The SELL-bias mechanism is pre-existing.

4. **SPEC default `joint` decode**: `benchmark_easy.py:442, 856, 1215` set
   `eval_mcts_decode='joint'` (default). The 767 SPEC's promotion gate explicitly
   specifies eval-MCTS sims=16 joint (SPEC §"Promotion gate"). The per_asset path
   is opt-in via `--eval-mcts-decode per_asset`. **The 767 launch is unaffected
   under SPEC defaults.**

5. **Risk surface**: a downstream operator who runs `--eval-mcts-decode per_asset`
   against any joint-MCTS checkpoint (767 or pre-767) hits this. F3 is a
   pre-existing benchmark robustness issue, not a 767-specific bug.

**Recommended fix**: as the hunt suggests — `per_asset` mode should default
un-visited assets to HOLD when `max(per_asset) < threshold`. Optional, low priority.

**Severity adjustment (P2 vs P1)**: pre-existing behavior + not exercised under
SPEC default + identical SELL-bias mechanism pre- and post-767. The hunt's
"newly load-bearing under 767's mask" is mild over-statement. Confirmed risk,
no impact on the 767 launch.

---

### F4: CONFIRMED — Stage-B → Stage-B resume drops `disable_limit_orders` via stray `config.json`
**Severity: P2** (matches the hunt).

**Independent verification**:

1. **The filter is real**: `phase2_config.py:297-307`:
   ```
   def get_phase2_config(phase1_config_path):
       ...
       easy_fields = {f.name for f in EasyConfig.__dataclass_fields__.values()}
       p1_filtered = {k: v for k, v in p1_data.items() if k in easy_fields}
       return Phase2Config(**p1_filtered)
   ```
   This drops any key not in `EasyConfig.__dataclass_fields__`.

2. **`disable_limit_orders` is Phase2-only**: `phase2_config.py:45` defines it on
   `Phase2Config` (which extends EasyConfig). Confirmed not in EasyConfig:
   grep `easy_config.py` shows no `disable_limit_orders`.

3. **Two code paths in `train_phase2.py:2569-2575`**:
   ```
   if p1_config_path.exists():
       config = get_phase2_config(str(p1_config_path))   # drops disable_limit_orders
   else:
       config = Phase2Config(**{k: v for k, v in p1_config.items()
                                if k in Phase2Config.__dataclass_fields__})  # preserves
   ```
   The first branch filters by EasyConfig fields. The second branch (no
   `config.json`) preserves Phase2Config fields.

4. **Trigger conditions**:
   - Stage A → Stage B normal flow: Stage A's checkpoint dir has no `config.json`
     (Stage A writes `phase2_stageA_*.pkl` only, no `config.json`). Stage B writes
     `stage_b_config.json` (`:1805`), a different filename. → ELSE branch → flag
     preserved.
   - Phase 1 → Phase 2 normal flow: the Phase-1 checkpoint dir DOES typically
     have `config.json` (written by Phase-1 trainer). For an initial Phase-2 run,
     `disable_limit_orders` isn't yet set on the Phase-1 config, so the filter
     drops nothing relevant. The CLI override at `:2630-2631` then sets it.
   - **Bug-trigger path**: a user manually copies a Phase-1 `config.json` next to
     a Phase-2 Stage-A checkpoint (e.g. via a deploy/rsync script that mirrors
     the Phase-1 dir layout) and then resumes Stage B from there → the EasyConfig
     filter strips `disable_limit_orders`.

5. **No `--no-disable-limit-orders` option**: `train_phase2.py:2301`
   uses `action='store_true'` (not `BooleanOptionalAction`), and `:2630-2631`
   has only `if args.disable_limit_orders: config.disable_limit_orders = True`
   — one-way override. Once silently dropped, the user cannot force it back from
   CLI without a code change. The hunt's claim is verified.

6. **Operator-detection path**: `train_phase2.py:1758-1761` (now `:1758-1761`)
   prints the "TASK_767: limit orders DISABLED" banner only when the flag is on.
   A careful operator who greps `stage_b_training.log` for this banner can detect
   the silent drop. Not a fail-safe, but a hand-rail.

**Recommended fix**: same as the hunt — call site `:2570` should use
`Phase2Config.__dataclass_fields__` (not EasyConfig). Plus add
`--no-disable-limit-orders` via `BooleanOptionalAction` for symmetry with
`benchmark_easy.py`.

**Severity (P2, matches hunt)**: low trigger probability under SPEC's intended
workflow, but a silent-failure path with no run-time guard. Worth fixing as
hygiene before the 767 launch promotes.

---

### F5: CONFIRMED — SPEC line numbers are stale (~17-45 lines drifted)
**Severity: P2** (matches the hunt).

**Independent verification** — current line numbers from `grep -n` (verified just
now):

| SPEC reference | SPEC claim | Actual | Drift |
|---|---|---|---|
| `--disable-limit-orders` argparse | `:2270-2276` | `:2300-2307` | +30 |
| Plumbed to config | `:2584-2586` | `:2629-2631` | +45 |
| Loss mask build | `:1043-1083` | `:1060-1090` | +17 (start) / +7 (end) |
| Logits masked | `:1160-1166` | `:1182-1183` | +22 (start) / +17 (end) |
| CE targets masked | `:1168-1191` | `:1200-1208` | +32 / +17 |
| Factored CE | `:1198` | `:1216` | +18 |
| Joint CE | `:1216` | `:1233` | +17 |
| Oracle BC CE | `:1306` | `:1337` | +31 |
| Train mask build | `:1711-1731` | `:1741-1761` | +30 |
| Search call | `:1886-1896` | `:1914-1928` | +28 / +32 |
| Metrics | `:1437-1462` | `:1467-1498` | +30 |
| CSV fieldnames | `:1800-1802` | `:1830-1832` | +30 |
| Epoch log | `:2055-2057` | `:2085-2087` | +30 |
| CSV row | `:2107-2109` | `:2137-2139` | +30 |

Cause: commit `95fcf98` (TASK_769 F2 fix, between 767 ship and today) inserted ~30
lines of `value_clip_mode` gating into `train_phase2.py`.

**Hunt's recommendation** (use TASK markers / function-name anchors instead of
line numbers) is sound.

**Severity (P2, matches hunt)**: documentation hygiene only. Code comments at the
actual locations carry `TASK_767:` markers, so a reviewer who searches by marker
(`grep TASK_767`) is unaffected.

---

## New findings (missed by the hunt)

### New-F1 (P2): `_q_to_policy` epsilon-greedy "fix" accidentally matches the bug
**Severity: P2** — observation, not a fresh bug.

The default `epsilon_greedy = 0.1` (`phase2_config.py:58`). The function blends
90% buggy softmax output with 10% uniform-over-valid. With the F1 bug, the buggy
softmax output IS already `[1/3, 1/3, 1/3, 0, 0]` per asset — identical to the
uniform-over-valid component. So:
- 90% × `[1/3, 1/3, 1/3, 0, 0]` + 10% × `[1/3, 1/3, 1/3, 0, 0]` = `[1/3, 1/3, 1/3, 0, 0]`.

**Implication**: the epsilon-greedy mix masks the bug from any cursory comparison
("does the target look reasonable? yes, sums to 1, has zero mass on limits,
spreads across valid actions" — but it's uniform-over-valid for the WRONG reason).
After the F1 fix, the comparison `epsilon_greedy_off vs epsilon_greedy_on` flag-ON
will start showing differentiated behavior — operators should expect this.

**Recommendation**: after F1 fix, the canon `epsilon_greedy_on` smoke should show
`mean_trade_rate < 0.667` (Q-weighted, less uniform) by some margin. If it
doesn't, the F1 fix didn't land.

### New-F2 (P3): `_mask_renorm_target` all-zero edge case is silently handled but the policy CE term on that row becomes degenerate
**Severity: P3** — defensive code with subtle semantics.

`train_phase2.py:1097-1101`:
```
t = target_320 * limit_keep_mask
tr = t.reshape(-1, n_assets, n_actions_per_asset)
grp = jnp.sum(tr, axis=-1, keepdims=True)
tr = tr / jnp.maximum(grp, 1e-8)
```

If a per-asset row of `target_320` has all probability mass on limit indices
(e.g. `[0, 0, 0, 0.5, 0.5]`), the mask zeroes it: `[0, 0, 0, 0, 0]`. Then
`grp = 0`, divided by `1e-8` → still `[0, 0, 0, 0, 0]`. The CE
`softmax_cross_entropy(logits, [0,0,0,0,0]) = 0` exactly (because the loss is
`-sum(target * log_softmax)` and target is 0 everywhere).

**Is this reachable?** For `oracle_targets`: oracle never emits to limit indices
(see `path_solver.py:797-799`, limit logits stay at `-10.0`), so per-asset row
is always `[buy/sell/hold > 0, ... ]`, mask preserves at least 1.0 mass. **Not
reachable.**

For `mcts_visits` in factored mode: input is the buggy `[1/3, 1/3, 1/3, 0, 0]`
already → never all-zero. **Not reachable.**

For `mcts_visits` in joint mode: not routed through `_mask_renorm_target`
(only `* limit_keep_mask` at `:1205`). N/A.

**Verdict**: not reachable under current data flow. Hunt's "verified clean"
already covers this. New-F2 is informational; no fix needed.

### New-F3 (P3): `softmax_cross_entropy` with all-zero target on a few un-visited assets (joint MCTS path)
**Severity: P3** — observation.

In joint MCTS, only 1 asset's action gets visits (out of 64). The other 63 assets
have `mcts_visits[..., asset_j, :] = [0, 0, 0, 0, 0]` (all zero), so
`mcts_visits_m = mcts_visits * limit_keep_mask = [0,0,0,*,*] → [0,0,0,0,0]`
under the 767 mask. The joint CE `softmax_cross_entropy(logits, mcts_visits_m)`
sums per-batch — un-visited rows contribute 0 to the loss either way. This
matches pre-767 behavior; the 767 mask doesn't change the loss landscape here.
**Verified clean.**

### New-F4 (P2): smoke artifacts unreachable for direct verification
**Severity: P2** — process/audit-trail issue, not a code bug.

The 767 agent's smoke summary (commit message + SPEC §"Smoke test" §5) reports
two 1-epoch factored Stage-B smokes (seed 7) on 2026-05-22 with
`limit_action_fraction=0.000` (flag-ON) and `=0.398` (canon). I searched
`runs/767_action_space/`, `runs/000_smoke/`, and all `stage_b_metrics.csv`
locations under `runs/`. No 767-specific smoke directory exists; the
`runs/767_action_space/` directory is empty (created 2026-05-22 23:12, no
contents). The smoke metrics quoted in the commit message and SPEC are
**unverifiable from this audit**.

**Implication**: the bug-hunt's F1 fingerprint check (`mean_trade_rate ≈ 0.667`)
cannot be confirmed/refuted by reading existing artifacts. A 3+ epoch
diagnostic smoke is needed before the launch, AND the smoke artifacts should be
preserved under `runs/767_action_space/smoke_*/` for future audits.

**Recommendation**: before 767 launch, run a 3-epoch flag-ON factored smoke and
verify `mean_trade_rate` is **below** 0.667 (Q-weighted, not uniform). Persist
the CSV under `runs/767_action_space/smoke_*/`.

---

## Verified-clean items re-confirmed

The hunt's "Verified clean" list (BUGHUNT_767:128-147) holds under my
independent inspection:

- **Canon (flag-OFF) bit-identity** — `factored_search.py:232-234` calls 3-arg
  JIT (`self._search_fn(params, rng, observation)`), the `action_mask is not None`
  branch is dead Python. `train_phase2.py:1089-1090` sets both
  `limit_keep_mask = None` and `limit_logit_mask = None`. Every guard branch
  (`:1182`, `:1200-1208`, `:1741-1761`, `:1916-1918`) is dead in the trace.
  Re-confirmed.
- **`-1e9` vs `finfo.min` for log_softmax** — `optax.softmax_cross_entropy` →
  `jax.nn.log_softmax` (max-stable). For valid actions: `log_softmax` is finite.
  For masked actions at `-1e9`: `log_softmax = -1e9 - logsumexp_valid` ≈ `-1e9`
  (finite). Targets at masked positions are 0 → CE contribution is `0 * -1e9 = 0`
  exactly. **Verified clean.** The hunt's analysis on this surface is correct.
- **`_mask_renorm_target` denominator guard** — `jnp.maximum(grp, 1e-8)`. Reachable
  zero-grp cases are absent under the current data flow (see New-F2 above).
  **Verified clean.**
- **mctx joint-MCTS root masking** — `policies.py:180`: `_mask_invalid_actions(root.prior_logits, invalid_actions)`,
  which uses `finfo.min` (`:384`). Root softmax at limits underflows to 0.
  `masked_argmax(to_argmax, invalid_actions)` at `:222` never returns a limit.
  **Verified clean.**
- **Oracle never emits limits** — `path_solver.py:794` inits all logits to
  `-10.0`. `:797-799` defines only `buy/sell/hold_indices`. Limit indices retain
  `-10.0` → per-asset softmax puts tiny but non-zero (~5e-5) on limits, which
  is why the `_mask_renorm_target` is needed (not why CE explodes — `-1e9` keeps
  it finite). `np.argmax(policy_targets, axis=-1)` at `:622` of train_phase2.py
  picks the highest-probability action, always within `{buy_idx, sell_idx,
  hold_idx}`. **Verified clean.** Confirmed by direct read of `:790-799`.
- **Phase-2 config save/checkpoint round-trip + auto-detect** — `train_phase2.py:1566-1567`
  filters `config.__dict__` to primitives (includes bools, includes
  `disable_limit_orders`). `benchmark_easy.py:911-915` captures
  `raw_config_dict` BEFORE the EasyConfig filter strips Phase-2 fields. Auto-detect
  at `:939-940` reads from `raw_config_dict`. **Verified clean.**
- **CLI override semantics for eval** — `benchmark_easy.py` uses
  `BooleanOptionalAction` (verified by inference from the hunt's claim; not
  cross-checked here line-for-line). 3-state semantics (None/True/False) is
  the right design.
- **Action-distribution metric normalization** — `train_phase2.py:1489-1498`
  uses `jnp.maximum(_act_total, 1e-8)` to guard; the per-row total is 1
  (joint) or 64 (factored). The metric is mode-agnostic. The `_act_frac` is
  the per-action-type marginal probability summing to 1. **Verified clean.**
- **Metric measured on raw `mcts_visits` not `mcts_visits_m`** — confirmed at
  `train_phase2.py:1489` (`_vc = mcts_visits.reshape(...)`, not
  `mcts_visits_m`). The 767 fingerprint comparison is honest. **Verified clean.**
- **`actions_dyn` cannot be limit** — `train_phase2.py:622` argmax over
  oracle_targets; oracle never emits to limit indices. **Verified clean.**
- **`q_advantage` and action selection (factored search)** — argmax on
  masked `q_values` (where limits = -1e9) and `q_advantage` (where limit
  advantages ≈ -1e9 - Q_hold ≈ -1e9 due to ULP) both correctly exclude limit
  indices. **This is the only thing that saves the implementation from total
  breakdown** — the bug is target-only, not selection. Re-confirmed.
- **JAX pitfalls** — no `lax.cond` in `vmap`, no dynamic `lax.scan` shapes,
  divisions guarded with `jnp.maximum(., epsilon)`. JIT re-trace on
  `action_mask=None` vs `action_mask=Array` is implicit (different arg count
  triggers re-trace). **Verified clean.**
- **mctx `invalid_actions` invert** — `mctx_search.py:219, 232`:
  `~action_mask.astype(bool)` converts our `1=valid` convention to mctx's
  `1=invalid`. **Verified clean.**
- **Eval-MCTS joint decode never returns limits** — `benchmark_easy.py:425-428`
  via `mcts_actions % 5`, where `mcts_actions = masked_argmax(...)` excludes
  limits at root. **Verified clean.**

---

## Brief verdict summary

- **F1 (P0)**: CONFIRMED. Re-derived the float32 ULP arithmetic from scratch.
  Mean ≈ -4e8 (ULP = 32); valid Q's centered to ≈ 4e8 collapse to indistinguishable
  float32 values; `exp(-1e9)` underflow yields uniform `[1/3, 1/3, 1/3, 0, 0]`
  per asset as the policy target. XLA does NOT optimize the mean-shift away
  (it's data-dependent, not a constant). Action selection (`argmax(q_values)`,
  `argmax(q_advantage)`) is UNAFFECTED because the bug is target-only. The smoke's
  reported `limit_action_fraction=0.000` is consistent with both the buggy and a
  correct implementation; the agent's smoke summary omitted `mean_trade_rate`,
  which would have been the smoking gun. **Fix: remove the mean-shift line**
  (`q_centered = q_values - q_values.mean(axis=-1, keepdims=True)`) entirely —
  `jax.nn.softmax` does its own internal max-shift. Recommended verification:
  ≥3-epoch flag-ON smoke logging `mean_trade_rate`; pre-fix should show ≈ 0.667,
  post-fix should diverge below it.
- **F2 (P1)**: CONFIRMED. mctx interior selection is unmasked by design (`action_selection.py:158-195`);
  our `_recurrent_fn` returns unmasked `policy_logits`. Bounded efficiency hit,
  not a correctness break — final root action and weights are masked. No fix
  required for the 767 launch; SPEC's canonical search is factored.
- **F3 (P1 → P2)**: PARTIAL. The numpy-argmax SELL-bias mechanism on near-tied
  rows is real, but pre-existing (not 767-specific) and not exercised under SPEC
  default `joint` decode. Downgrade to P2.
- **F4 (P2)**: CONFIRMED. The EasyConfig field filter in `get_phase2_config`
  drops `disable_limit_orders` if a stray `config.json` sits next to a Phase-2
  checkpoint. Low trigger probability, no run-time guard. Hygiene fix: use
  `Phase2Config.__dataclass_fields__` at the call site, and add
  `--no-disable-limit-orders` via `BooleanOptionalAction`.
- **F5 (P2)**: CONFIRMED. ~17-45 line drift on every SPEC reference into
  `train_phase2.py`, caused by `95fcf98` (TASK_769 F2 fix). Documentation hygiene
  only — code reviewers using `grep TASK_767` markers are unaffected.

**Net**: the hunt's analysis is sound. F1 is the only finding that blocks a
clean 767 launch interpretation — it must be fixed (one-line change) before the
3-action variant is trained on the server. F2 is acceptable as a documented
known-limitation. F3-F5 do not block the launch but should be tracked.

The hunt's overall verdict stands.
