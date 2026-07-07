# TASK_826 — On-chain signal probe: mine the node's real data, storage-light, before committing terabytes

**Status**: SPEC'D 2026-07-06. The strategic insight (operator): the archive node's value is NOT
the aggregated OHLCV byproduct — it's the **mempool / per-tx flow / MEV / liquidations / LP
shifts**, which are (1) *idiosyncratic* (uncorrelated with crypto beta → genuinely new effective
breadth, the ceiling [[project_planner_closed_clean_cell]] hit) and (2) *per-decision EV* (discrete
profitable events — the edge profile daily aggregate-trend LACKED, which is what closed the planner
in TASK_821). This is the one data direction that attacks both walls at once. But it must be done
within the storage line the operator drew: **30–40GB fine, 4–6TB not.**

## The honest split (which use needs what)

- **Live MEV extraction (Aconite)** — needs a LOCAL full node + mempool visibility + low latency to
  compete. Multi-TB, cannot be rented (hosted RPC is too slow + mempool-blind). **PARKED on storage.
  OUT OF SCOPE here.** Gated on: (a) this probe showing real residual edge AND (b) storage solved.
- **Historical on-chain signal mining (THIS TASK)** — a TARGETED pull of specific events for a
  specific window from a HOSTED archive source (Dune / Allium / subgraph / Alchemy-QuickNode
  archive RPC) into a compact SQLite. No local chain state, no multi-TB. Storage-light, feasible now.

## Discipline: prove the edge cheap before buying terabytes

Do NOT stand up a local node. Do NOT commit to the live-MEV infrastructure on the *hope* of alpha.
Mine a small historical slice, run it through the full 806→819 gauntlet, and let the RESULT decide
whether the expensive live path is ever justified. Null is a fine, money-saving outcome.

## Phase 1 — targeted historical pull (hard ≤40GB cap)

Pick a hosted source (compare cost/rate-limit/coverage first): Dune/Allium exports, The Graph
subgraphs, or an archive RPC free/cheap tier. Pull a DEFINED window (e.g. 1–2 years) of a DEFINED
token set (top-N liquid) — only the event types that carry signal:
- Large DEX swaps (direction + size) and net per-token flow imbalance per interval.
- Liquidation events (cascades = forced flow).
- LP add/remove (liquidity regime shifts).
- Optionally: CEX-listed tokens only, so the signal is tradeable on Kraken.
Land it in a compact SQLite in a schema parallel to `trading_data.db`. **If a pull would exceed
40GB, NARROW the window or token set — never blow the cap.**

## Phase 2 — honest edge probe (806→819 discipline)

Test whether on-chain features PREDICT forward return at a horizon you could actually trade. The
critical honesty: an on-chain swap has ALREADY moved the DEX price — the edge is NOT "the swap
happened," it must be in (a) predicting the NEXT move, or (b) **on-chain flow leading CEX price**
(the tradeable lag: does aggregate DEX flow at time t predict the Kraken price move at t+Δ?).
- Features: net flow imbalance, large-swap direction, liquidation pressure, LP shift — leak-free
  (only data ≤ t).
- Target: forward Kraken return at Δ (align on-chain time → CEX bars).
- Metrics: IC vs shuffled-label noise floor; per-decision cost-honest Sharpe (gas is NOT the cost
  if trading on Kraken — use Kraken taker+slip); cross-regime; beat trivial baselines.
- **Prior (stay skeptical):** the fast, obvious MEV is already extracted by pros with better infra.
  The probe is whether any RESIDUAL, slower-timescale signal survives for a CEX-trading strategy.
  Expect this to be hard; null is likely and acceptable.

## Gate
- **Real** if an on-chain feature clears IC-vs-noise AND per-decision cost-honest Sharpe CI excludes
  0 AND is LOW-CORRELATION with the TASK_820/825 trend book (the whole point = new breadth). Then:
  it's both a breadth-add for the CTA AND the per-decision-EV data a planner could actually trade →
  this is what would justify (a) solving storage for the live/full-node path and (b) re-opening the
  planner question on the RIGHT data. Big if true — verify adversarially before belief.
- **Null** → on-chain (at least the slow/CEX-tradeable slice) has no residual edge; the local-node /
  live-MEV multi-TB investment is NOT justified; park it cleanly. Money + terabytes saved.

## Run location
Workstation (CPU/IO, storage-light) or wherever the hosted-source keys live. No GPU. No local node.
Sub-agents: build + report; the leak-free / cost-honest / "flow already moved the price" honesty is
non-negotiable — the classic way this study lies to itself is treating an already-executed on-chain
print as a predictor.
