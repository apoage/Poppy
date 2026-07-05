# TASK_818 Tier B — `--reward-attributable` implementation spec

**Status**: SPEC (design-only, 2026-06-27). Campaign-shipper-ready. Default OFF → canon bit-identical.
**Parent**: `tasks/TASK_818_MODEL_COLLAPSE_FIX.md` (Tier B, line 33-36: "multi-step action-attributable
reward (env, jax_trading_env.py:1196-1203 — credit realized log-PnL on True→False position flip;
down-weight mark-to-market)"). This spec implements **only** the env reward component. The other two
Tier-B pieces (policy-prior entropy floor at `mcts_search.py:237-247`; reward-head own MLP at
`muzero_model.py:454-457`) are **out of scope** — separate flags.

---

## 0. The mechanism (why this is the load-bearing fix)

At cost=0 the per-step reward (`jax_trading_env.py:1196-1203`) is `symlog(log(V_t/V_{t-1})*100)` — the
mark-to-market (MtM) of the book **already held** entering step *t*. A Buy/Sell/Close at *t* converts
cash↔position 1:1 at `price_t` and does **not** move `V_t` → `r_t` is ~invariant to `action_t` →
`E[r | state, action]` ≈ 0 for the trade actions → MCTS cannot rank Buy/Sell vs Hold → **95% HOLD**.

**Fix:** on a position **CLOSE** (a `pos_active` True→False flip caused by *this* action), add the
**realized log-PnL of the round-trip** to the step reward. This is a non-zero, *predictable*,
action-conditioned target attached to the CLOSE decision — exactly the rankable signal the reward head
needs. Crucially it is the **holding-period (multi-step) return**, collapsed into the close step:
`Σ_t MtM ≈ log(V_close/V_entry) = realized` (telescoping). This is the horizon where signal lives —
the campaign's verified finding that **per-bar is cost/noise-floor-bound** (806→816, per-bar IC→R²~0.01)
while **multi-step trend (TSMOM) is real** (`runs/816_operating_point/`, gross-positive). Tier B moves the
credit off the action-invariant HOLD steps and onto the action-conditioned CLOSE step.

**Where it lands in training (read before shipping):** the env reward is **not** consumed by MCTS
directly. It becomes `batch['rewards']` → (a) the **reward-head** categorical/Huber target
(`trainer.py:935-964`) and (b) the **value** n-step-return target. MCTS plans through the *learned*
reward+dynamics network (`mctx`), so Tier B changes **what the reward head learns to predict**; the
action-conditioned signal reaches the planner only *after* the head fits it. Therefore: (1) requires a
**FRESH run** (a collapsed checkpoint's prior gating is self-sealing — parent task line 37), and (2) is
**necessary-not-sufficient** alone — pair with the policy-prior entropy floor in a later ship.

---

## 1. Files + exact line ranges to change

| # | File | Anchor (current) | Change |
|---|------|------------------|--------|
| 1 | `mcts_trader/env/jax_trading_env.py` | `EnvConfig` dataclass, after `burn_random_limit` (~L125) | add field `reward_attributable: bool = False` |
| 2 | `mcts_trader/env/jax_trading_env.py` | `step()` just before `jax.lax.switch` (~L1154, after `apply_tp_sl_closures` L1149) | snapshot pre-action position state |
| 3 | `mcts_trader/env/jax_trading_env.py` | `step()` just after reward computed (~L1204, after L1203, before `if config.burning_money`) | inject realized-PnL-on-close term |
| 4 | `mcts_trader/training/config.py` | "Reward shaping" block (~L194-197, after `differential_*`) | add `reward_attributable: bool = False` |
| 5 | `mcts_trader/training/cli.py` | after `--reward-hybrid` (~L172-175) | add `--reward-attributable` argparse |
| 6 | `mcts_trader/training/env_factory.py` | 3 fns: `create_jax_env_and_prices` (sig ~L13-33, `EnvConfig(` ~L107-134), `create_jax_env_unique_prices` (sig ~L141, `EnvConfig(` ~L238-257), `create_jax_env_historical` (sig ~L267, `EnvConfig(` ~L350-366) | add param + thread into each `EnvConfig(...)` |
| 7 | `mcts_trader/training/setup.py` | `common_kwargs` dict, after `burn_random_limit=...` (~L285) | add `reward_attributable=getattr(config,'reward_attributable',False)` (one place → covers both `**common_kwargs` call sites L292 + L300) |
| 8 | `train_jax_fast.py` | `create_jax_env_historical(...)` call, after the `burn_*` kwargs (~L231-235) | add `reward_attributable=getattr(config,'reward_attributable',False)` |

No `close_position()` function exists — closes are scattered (`execute_buy` closes short, `execute_sell`
closes long, `apply_tp_sl_closures`, `process_margin_call`, limit fills). The snapshot-diff approach below
is **robust to this** because it detects the `pos_active` flip regardless of which handler caused it.

---

## 2. The JAX-safe change (`step()`)

### 2a. Snapshot — insert immediately **before** `new_state = jax.lax.switch(...)` (~L1154)

```python
# TASK_818 Tier B: snapshot pre-action position state for realized-PnL attribution.
# Pure name-bindings to existing tracers — emit ZERO XLA ops, so they are harmless
# even when the flag is OFF (they only feed the guarded branch below, which is not
# traced when reward_attributable=False → dead, absent from the jaxpr).
prev_active_attr = state.pos_active   # (n_assets,) bool   — post limit/tp_sl/margin, pre-action
prev_side_attr   = state.pos_side     # (n_assets,) int 0/1/2
prev_entry_attr  = state.pos_entry    # (n_assets,) float
```

Snapshot point = **after** `apply_tp_sl_closures` (L1149) and **before** the action switch (L1155), per
parent-task direction. Consequence (intended): closes from pending **limit fills** (L1142) and **TP/SL**
(L1149, `tp_sl_enabled=False` by default anyway) are excluded — they are not conditioned on *this* step's
action. Closes from the **action** (L1155) **and** the **post-action margin call** (L1169, longs
liquidated to cover short losses the action caused) are included.

### 2b. Inject — insert immediately **after** `reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))` (L1203), **before** the `if config.burning_money:` block (L1206)

```python
# TASK_818 Tier B (flag --reward-attributable, default OFF → canon bit-identical):
# credit realized log-PnL on a True->False position flip (a CLOSE) caused by THIS step.
# Long  realized = log(exit/entry);  Short = mirror = -log(exit/entry) = log(entry/exit).
# This is the multi-step holding-period return, attributed to the CLOSE action.
if config.reward_attributable:
    closed     = prev_active_attr & (~new_state.pos_active)          # (n_assets,) bool
    safe_exit  = jnp.maximum(current_prices, 1e-8)                   # close exec ≈ current (==  at cost=0)
    safe_entry = jnp.maximum(prev_entry_attr, 1e-8)
    long_log   = jnp.log(safe_exit / safe_entry)                    # log(exit/entry)
    realized   = jnp.where(prev_side_attr == 1, long_log,
                  jnp.where(prev_side_attr == 2, -long_log, 0.0))    # short = mirror
    realized   = jnp.where(closed, realized, 0.0)                   # zero where no close
    realized_sum = jnp.sum(realized)                                # scalar; ~1 close/step
    # Don't stack the realized spike onto the liquidation sentinel (symlog(-50)).
    reward = reward + jnp.where(is_liquidated, 0.0, symlog(realized_sum * 100.0))
```

**JAX-safety checklist (all satisfied):** only `jnp.where` / `jnp.log` / `jnp.maximum` / `jnp.sum` — no
Python branching on traced values; division guarded by `jnp.maximum(·, 1e-8)`; both `jnp.where` branches
are finite; the `if config.reward_attributable:` is a **compile-time** Python branch on a **static** field
(see §4). Shapes are static `(n_assets,)` → scalar; no dynamic shapes, no `lax.cond`-in-`vmap`.

**vmap correctness:** `step` is `vmap`'d over the leading env axis (`env_factory`/`self_play` →
`jit_batch_step` at `jax_trading_env.py:1641-1642`, `jax.vmap(lambda s,a: step(...))`). Inside one env all
arrays are `(n_assets,)`; `jnp.sum` reduces to a per-env scalar reward. Per-env correct under `vmap`. ✓

**Close-flip is unambiguous (verified from handler logic):**
- `execute_buy` closing a short sets `pos_active=False`; opening a long requires `~has_position` → a single
  call **never** closes-and-reopens the same asset. Same for `execute_sell` (close long XOR open short).
  So `True→False` cleanly means "closed", with no reopen masking the flip.
- Market closes are **full** (`execute_sell` sets `pos_size=0`, `pos_active=False`) — no partial closes →
  `realized` = full round-trip return. Size cancels in `log(exit_value/entry_value)=log(exit/entry)`, so the
  per-asset entry price (`pos_entry`, weighted-averaged across adds in `execute_buy`) is the correct basis.

---

## 3. Plumbing (cli → config → env_factory → EnvConfig), mirroring `burning_money`

`burning_money` is the correct precedent (an **env-side** flag), not `--reward-hybrid` (trainer-side).

1. **`EnvConfig`** (`jax_trading_env.py` ~L125): add `reward_attributable: bool = False`.
2. **`TrainingConfig`** (`config.py` ~L197, Reward-shaping block): add `reward_attributable: bool = False`.
   Because it is a real dataclass field, the cli converter
   `config_dict = {k:v for k,v in vars(args).items() if hasattr(TrainingConfig, k) and v is not None}`
   (`cli.py:386`) picks it up **automatically** → **no explicit `config.x = getattr(args,...)` copy needed**
   (unlike `reward_hybrid`, which is *not* a field and so needs the manual line at `cli.py:398`).
3. **`cli.py`** (after L172-175): 
   ```python
   parser.add_argument('--reward-attributable', action='store_true',
                       help='TASK_818 Tier B: add realized log-PnL on position CLOSE to the step '
                            'reward (action-attributable, multi-step). Default OFF = canon.')
   ```
   (argparse dest = `reward_attributable`; `store_true` default `False`.)
4. **`env_factory.py`** — add `reward_attributable: bool = False` to each of the 3 function signatures and
   `reward_attributable=reward_attributable` to each of the 3 `EnvConfig(...)` constructions.
5. **`setup.py`** `common_kwargs` (~after L285): `reward_attributable=getattr(config,'reward_attributable',False)`
   (single add — both `**common_kwargs` call sites at L292 + L300 inherit it).
6. **`train_jax_fast.py`** `create_jax_env_historical(...)` (~after L231): 
   `reward_attributable=getattr(config,'reward_attributable',False)`.

---

## 4. Canon-OFF guarantee (static guard)

`step` is jitted with **`config` as a static arg** everywhere it is called:
`@partial(jax.jit, static_argnames=['config'])` at `jax_trading_env.py:1543, 1606, 1613, 1620, 1627, 1634,
1641`. Therefore `if config.reward_attributable:` is resolved at **trace time**. With the default
`False`, the branch body is **never traced** → **zero ops added to the XLA graph** → the compiled `step`
is **bit-identical** to canon. This is the **exact** mechanism already proven by the sibling compile-time
branches `if config.borrow_rate_annual > 0` (L1179), `if config.burning_money` (L1206), and
`if config.burn_random` (L1244). The §2a snapshot lines bind names to existing tracers and feed only the
dead guarded branch → they emit no ops and do not appear in the OFF jaxpr.

---

## 5. CPU validation plan (no GPU, no training — runs on the workstation safely)

All checks are CPU-only / import-and-numpy level. Do **not** launch training on the workstation GPU.

1. **AST + import sanity**
   ```bash
   for f in mcts_trader/env/jax_trading_env.py mcts_trader/training/config.py \
            mcts_trader/training/cli.py mcts_trader/training/env_factory.py \
            mcts_trader/training/setup.py train_jax_fast.py; do
     python -c "import ast,sys; ast.parse(open('$f').read()); print('OK', '$f')"
   done
   python -c "from mcts_trader.env.jax_trading_env import step, EnvConfig, reset; \
              assert EnvConfig().reward_attributable is False; print('default OFF OK')"
   ```

2. **Canon-OFF graph identity** (`JAX_PLATFORMS=cpu`): build a tiny `reset` state (n_assets small,
   fixed seed, a synthetic flat-then-up price array), run `step` once with `EnvConfig(reward_attributable=False)`
   vs the **pre-change** code on the same inputs → assert reward arrays equal exactly (or compare
   `jax.make_jaxpr(step)(...)` text OFF-vs-baseline). Confirms bit-identity.

3. **Numeric realized check** (CPU JAX or hand-rolled numpy mirror of §2b):
   - **Long close:** open a long `pos_entry=100`, then SELL-close at `current_price=110`.
     `realized = log(110/100) = 0.09531`; added reward `= symlog(0.09531*100) = symlog(9.531)
     = log1p(9.531) = log(10.531) ≈ 2.3544`. Assert `r(attributable=True) − r(attributable=False) ≈ 2.354` (tol 1e-3).
   - **Short close (mirror):** open a short `pos_entry=100`, BUY-close at `current_price=90`.
     `realized = -log(90/100) = log(100/90) = 0.10536`; added `= symlog(10.536) = log(11.536) ≈ 2.4454`.
   - **Hold / no-close:** a HOLD step (or any step with no `True→False` flip) → `closed` all-False →
     `realized_sum=0` → `symlog(0)=0` → reward delta exactly 0.
   - **Liquidation:** force `is_liquidated=True` on a close step → assert the realized term is suppressed
     (reward == `symlog(-50)`, no spike added).

   (At `fee_bps=slippage_bps=0` the close exec price == `current_price`, so the snapshot `pos_entry` (which
   embeds the entry-side exec price) and `current_prices` are the clean basis — the numbers above hold
   exactly. With costs > 0 the realized term is a small approximation; see Risk 2.)

---

## 6. Risks (+ the single biggest)

1. **Double-counting the holding-period return.** Per-step MtM already telescopes to the holding-period
   return; adding realized-on-close credits it **twice** in the value (Σ-of-rewards) target.
   *Recommendation: keep it ADDITIVE (default), do NOT attempt to down-weight MtM.* Reasoning:
   (a) MtM is computed at **portfolio scalar** granularity (`compute_portfolio_value`, one `log_return`
   for the whole book) — there is **no per-asset MtM term** to subtract for the closing asset, so a true
   "replace MtM while held" would require re-deriving portfolio MtM excluding closing assets = materially
   larger blast radius and structurally awkward. (b) The double-count is on the **value** target, which
   **PopArt rescales** (value head mandatory PopArt, 401 bins over [-10,+10]); a ~2.4 spike per close is
   well within support. (c) The **reward head** — the thing that was collapsing — gains a clean
   action-conditioned per-step target it can actually rank, which is the entire point. If attenuation is
   ever wanted, expose a single global `mtm_weight` (default 1.0) as a follow-up; not in this minimal ship.
2. **symlog scale shift.** `realized` is a few-percent fraction → `*100` → 5-10 → `symlog` → ~1.8-2.4 per
   close. That is **large** vs typical per-step MtM symlog rewards (~0.001-0.05, `trainer.py:25`). Intended
   (the spike must be rankable), but it shifts the reward-target distribution: the reward head's fixed
   `×reward_scale` categorical binning (`trainer.py:949`) / 401-bin support may **clip** the new targets if
   run **without** Tier A. *Mitigation: run Tier B together with `--reward-hybrid` (Tier A)* — its
   `reward_popart` running-standardize absorbs the new scale. Validate the value support range still covers
   the per-step max.
3. **vmap / static-config correctness.** Addressed in §2 (per-env `(n_assets,)`→scalar) and §4 (static
   config, compile-time guard). Low risk.
4. **Train-vs-inference path.** Env reward = **trainer target only**, not a direct MCTS input (§0). So Tier B
   is **necessary-not-sufficient**: it must be validated on **pct_hold / alpha in a FRESH run** (not a
   collapsed checkpoint), and ideally alongside the policy-prior entropy floor. Validating on `reward_loss`
   is misleading (parent task line 26).
5. **Snapshot scope.** Limit-fill / TP-SL closes are excluded by the pre-action snapshot point (intended;
   not action-conditioned). TP/SL is OFF by default so this is moot for the primary regime; document it.

**BIGGEST RISK — scale × double-count interaction can make Tier B *alone* worse, not better.** A ~2.4
per-close reward spike, double-counted on top of MtM, will push the reward-head categorical targets toward
(or past) the edge of the fixed-scale 401-bin support **if shipped without `--reward-hybrid`** — clipping
the very signal it adds and destabilizing the value target. **Mitigation/standing instruction: never run
`--reward-attributable` without `--reward-hybrid` (Tier A standardization), and gate promotion on
pct_hold/alpha from a fresh cost=0 run**, not on `reward_loss`.
