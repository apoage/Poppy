# TASK_790 — Realized-PnL value term (experiment A2)

**Status**: BUILT (default-OFF flag), smoke = s7 E0 on server
**Created**: 2026-06-01
**Parent**: TASK_787 (A — over-trade penalty). Sibling of TASK_789 (C — tokens), TASK_788 (B — ExIt).

## Hypothesis
A's over-trade penalty reduced trade-rate but moved alpha only **+0.35pp** (gate FAIL) — too weak,
but NOT a backfire (no anti-correlation). The real test of "put PnL in the loss" is to ground the
value signal in **executed next-step return**, not oracle hindsight. If the bottleneck is "the loss
trains a classifier, not a trader," a realized-PnL gradient is the lever that should move alpha
without the trade-rate explosion.

## The mechanism (loss-side only — no new param leaf)
In `train_phase2.py` Stage-B `loss_fn`, gated by a default-OFF flag `realized_value_weight`:
- Expected per-asset position from the policy (action order 0=Sell,1=Hold,2=Buy,3=LimitSell,4=LimitBuy):
  `pos = P(Buy)+P(LimitBuy) − P(Sell) − P(LimitSell)` ∈ [−1,1], from per-asset softmax of `policy_logits`.
- `r_next = future_log_returns[:, 0, :]` — the **horizon-1 realized log-return** (already a JIT input;
  built at `train_phase2.py:728-740` / `:861` from `slot['returns']`; horizons [1,5,10,20]).
- `realized_value = mean((pos·r_next − |r_next|)²)` — MSE onto best-achievable per-step PnL, pushing
  `pos → sign(r_next)` scaled by move size → a direct gradient toward profitable next-step trades.
- `total_loss += realized_value_weight * realized_value`.

**Chosen loss-side-only (no `nn.Dense` head)** so the OFF-path param tree is trivially bit-identical
and review surface stays small. Compile-time gate `if realized_value_weight > 0.0:` (static config) →
when OFF the term is **not traced**, XLA graph + grads bit-identical to canon (same proven pattern as
TASK_787; A's smoke verified all 70 param leaves equal OFF).

## Files changed (default-OFF)
- `phase2_config.py`: `realized_value_weight: float = 0.0`.
- `train_phase2_cli.py`: `--realized-value-weight`.
- `train_phase2.py`: closure capture, gated loss term + `realized_value` diagnostic, startup log, override.

Caveat: the ON-path requires `future_log_returns` populated — true for the canonical 760 a1b1 config
(it sets `--aux-price-weight 0.1` → aux horizons present). Sweep cells must use the canonical recipe.

## Experiment
Paired 3-seed (7/13/23) vs **760 a1b1**, eval-MCTS sims=16 joint. Report **ΔAlpha + ΔTradeRate +
ΔBal.Acc** (anti-correlation diagnostic). Sweep `realized_value_weight ∈ {0.1, 0.5}` first (start
mid). Gate +3pp / no seed < −5pp. Each run wrapped in `tools/safe_run.sh --mem 64G --gpu N`.

## Watch
- Does realized-PnL raise Δα WITHOUT a trade-rate explosion (the failure mode of every classifier
  tweak)? If trade-rate explodes, the realized term is over-encouraging position-taking.
- s7 E0 on server is the de-facto smoke: finite total_loss + a non-zero `realized_value` diagnostic
  with the flag ON confirms shapes/data are correct before paying the full 3-seed.
