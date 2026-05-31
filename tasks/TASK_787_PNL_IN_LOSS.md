# TASK_787 — PnL in the Loss (experiment A): over-trade penalty in Stage-B

**Branch:** `exp/A-pnl-in-loss`
**Status:** IMPLEMENTED + smoke PASS (flag default OFF). Real experiment NOT yet run.

## Hypothesis

The Phase-2 Stage-B objective (`train_phase2.py` `make_stage_b_train_step`)
trains the net purely to **classify**: policy-CE vs MCTS visit counts, value-MSE
vs oracle remaining-PnL, plus oracle BC. There is **no realized-trading-cost
gradient** anywhere in the loss. The documented consequence
(`project_classifier_alpha_anti_correlation.md`,
`project_single_mechanism_exhausted.md`) is that classifier-side improvements
**anti-correlate with Alpha** via a "trade-rate explosion" at eval: a sharper
classifier trades more, and the over-trading eats the edge.

`mean_trade_rate` is currently a **diagnostic-only** metric
(`train_phase2.py:1622`, `_act_frac[:, 1]` is the HOLD marginal) — it never
enters `total_loss`. This experiment adds a **default-OFF, flag-gated over-trade
penalty** so the objective penalizes excessive trading **directly**.

## The flag

- **Config:** `Phase2Config.trade_rate_penalty_weight: float = 0.0`
  (`phase2_config.py`, declared next to `fee_penalty_bps`). **0.0 = OFF = canon.**
- **CLI:** `--trade-rate-penalty-weight` (type float, default `None` → inherit;
  set → override), in `train_phase2_cli.py`, wired into the config in
  `train_phase2.py` next to the `--min-entropy` override, plus a one-line
  startup log (`Over-trade penalty: weight=… (TASK_787, ON)`) emitted **only**
  when ON (canon logs unchanged).

## Loss math

Per-asset L2 penalty on the network's trade probability. HOLD is **action index
1** within each per-asset 5-block (`[Sell, Hold, Buy, LimitSell, LimitBuy]`;
confirmed by the existing diagnostic `_act_frac[:, 1]`). `policy_logits` is the
`(B, 320)` per-asset-flattened root policy logits in scope inside `loss_fn`:

```
per_asset_logits = policy_logits.reshape(B, n_assets, 5)       # (B, 64, 5)
per_asset_probs  = jax.nn.softmax(per_asset_logits, axis=-1)
hold_prob        = per_asset_probs[..., 1]                     # (B, 64)
overtrade_penalty = ((1.0 - hold_prob) ** 2).mean()           # scalar
total_loss += trade_rate_penalty_weight * overtrade_penalty
```

`overtrade_penalty` is also added to the returned metrics dict.

## Canon-OFF guarantee

The penalty is gated by a **Python compile-time** test
`if trade_rate_penalty_weight > 0.0:` (the weight is read from static config at
JIT-trace time, `train_phase2.py` closure binding). When the weight is `0.0`
the branch is **not traced** into the XLA graph at all — no extra ops, no extra
gradient. The `else` branch sets `overtrade_penalty = jnp.array(0.0)` purely for
a stable metrics-dict shape; it is never added to `total_loss`. This is the same
discipline used by TASK_763 (`entropy_coef > 0.0`) and TASK_767
(`disable_limit_orders`).

**This is NOT a `jnp.where` / `* 0.0` mask** — the term genuinely does not exist
in the compiled OFF graph, so loss and gradients are **bit-identical** to
pre-787 canon.

## Smoke result

`tools/smoke_A_trade_penalty.py` (tiny synthetic model, no ckpt/DB, <1 min GPU):

- **[A1] Canon-OFF bit-identity** — with weight=0.0:
  - `overtrade_penalty` metric is exactly `0.0`;
  - `total_loss` == sum of component losses with NO penalty term
    (`41.44880295` vs reconstruction `41.44880295`, rel_err `0.00e+00`);
  - the OFF-path `grad_norm` and **all 70 updated-param leaves** are
    bit-identical (`jnp.array_equal`) to a reference step built from a config
    with the `trade_rate_penalty_weight` attribute **deleted entirely**
    (grad_norm `36.1977691650` == `36.1977691650`). → the OFF graph == pre-787
    graph.
- **[A2] ON trainability** — with weight=0.1, 5 Stage-B steps:
  - loss finite (`38.808167`), grads finite & non-zero (`grad_norm 24.085787`),
    no NaN, `mean_trade_rate ∈ [0,1]` (`0.8026`);
  - penalty strictly adds: step-0 `total_loss` ON `41.518394` > OFF `41.448803`,
    and `Δ == weight * overtrade_penalty` to <1e-5.

**Smoking gun:** on pre-change (stashed) code the smoke fails at config
construction — `TypeError: Phase2Config.__init__() got an unexpected keyword
argument 'trade_rate_penalty_weight'` — i.e. the mechanism does not exist in
canon. After the change both assertions PASS.

## Proposed real experiment

Paired 3-seed vs the **760 a1b1** baseline under **eval-MCTS sims=16 joint**
(the current canonical scorecard, per `project_benchmark_live_gap.md` /
`project_single_mechanism_exhausted.md`).

- **Sweep:** `trade_rate_penalty_weight ∈ {0.05, 0.1, 0.2}`.
- **Seeds:** the standard set (e.g. 7, 13, 23) paired against the same-seed
  a1b1 reference.
- **Promotion gate:** mean Δα **≥ +3pp** AND **no seed < −5pp** (matches the
  Tier-2 promotion bar used for 734 flow_matching).
- **Primary anti-correlation diagnostic:** report **ΔBal.Acc AND Δtrade-rate**
  per seed. The mechanism's *thesis* is that it should be possible to hold (or
  improve) Alpha while **lowering trade-rate** — if Bal.Acc rises but Alpha
  still falls, the anti-correlation is intact and the penalty did not fix it; if
  trade-rate drops and Alpha holds/rises, that's the existence proof the loss-
  side trade cost was the missing ingredient.
- **Caveat from prior art:** every single-mechanism perturbation 738–758 has
  come back null-or-worse under eval-MCTS recal. Treat a single-seed lift as
  noise; only the paired 3-seed mean is the truth.

## Files changed

- `phase2_config.py` — +1 field `trade_rate_penalty_weight` (+ doc comment).
- `train_phase2.py` — closure binding (1), gated penalty term in `loss_fn`
  (~8 lines), metrics-dict entry (1), CLI→config override (2), startup log (2).
- `train_phase2_cli.py` — `--trade-rate-penalty-weight` arg.
- `tools/smoke_A_trade_penalty.py` — new smoke.
- `tasks/TASK_787_PNL_IN_LOSS.md` — this spec.
