# Trader — a pipedream funnelled through JAX into an MCTS fantasy

A MuZero + Tiny-Recursive-Model crypto trading research engine, written in JAX, run on daily and
minute Kraken OHLC data across ~64–279 assets. Built Dec 2025 → Sep 2026 by one person and a
rotating crew of AI coding agents: 668 commits, ~445 task specs, ~80 experiment directories,
57 unattended "dark factory" runs, and ~285 files of lore.

**It is a good engine. It is not a trader.** Nothing in this repo makes money after honest costs.
The interesting part is not the engine — it is the measured record of *how* each idea failed,
which is written down in more detail than most successes ever get.

> This is an archive, not a product. No data, no checkpoints, no credentials are included.
> Do not point `real_trade_jax.py` at real money.

---

## What this actually is

| Piece | What it does | Where |
|---|---|---|
| **Poppy** | MuZero-style planner (Sampled/Gumbel MCTS) over a learned dynamics/reward/value model with a TRM encoder. Rebuilt mid-project from a single scalar action to a per-asset **vector** action (64 weights per step). | `mcts_trader/`, `train_jax_fast.py`, `train_phase2.py` |
| **Bella** | Flow-matching "market generator" (OT-CFM, hybrid Mamba/attention) — synthesises unseen markets for regime stress. Ended as a resident tool, never load-bearing. | `diff_teacher/` |
| **Foxglove** | Supervised imprint (behaviour cloning) stage before self-play. Retired. | `train_easy.py`, `tools/orchestration/` |
| **`trader/`** | Plain numpy CTA/backtest engine. Exists to keep every learned result honest against a dumb linear trend book. | `trader/` |
| **The corridor** | Every result is graded as a position between a linear-book floor and a per-asset oracle ceiling, net of fees+slippage, ≥3 seeds. | `experiments/830_corridor/`, `docs/pathway.md` |
| **Candle language** | Tokenising candles into a vocabulary and training a small LM over them. The language is real (grammar + cross-asset transfer confirmed); as a trading mechanism it is null (TASK_804 five-arm). | `tools/candle_lang/`, `docs/candle_language/` |
| **Dark factory** | Autonomous implementer loop: ranked queue → implementer agent → 1–4 adversarial judge agents with different lenses → hook-enforced release ritual. Ran this repo's queue unattended. | `.dark-factory/`, `tasks/QUEUE.md`, `tasks/DECISIONS.md`, `.claude/` |

---

## The outcome, in one table

Numbers are net of costs unless marked; "net@21" = annualised Sharpe at 21 bp round-trip cost.
Each row names the ledger entry it comes from (`experiments/LEDGER.md`, or `docs/findings/` for pre-780).

| Question | Answer | Source |
|---|---|---|
| Does the scalar-action planner trade? | No. Converges to **100% HOLD** (0 trades in 1670 decisions). | `findings/planner_closed_clean_cell.md` |
| Can search find profit with a *perfect* model? | **Yes.** True-env search: net@21 **+5.29** vs rule +0.75; 4/4 regime windows pass. Up to **+8.02** with the fixed operator. | 831/E2-REAL, 831x/B, 832/E5-inversion3 |
| Can it with the *learned* model? | **No.** The learned value leaf is action-blind (Pearson **+0.03** vs true reward) so search rationally avoids every fee-paying trade. | 849/leaf |
| Does self-play improve a good imprinted policy? | **It destroys it.** +0.839 imprint → **−0.105 by epoch 100**, −0.79 / −1.22 at the end. | 832/E5-exit, 832/E5-erosion |
| Is the HOLD collapse a bug? | **No — it is correct.** Forcing more activity loses money monotonically: hold 99.0% → −2.9%, 95.7% → −9.1%, 91.2% → −13.8%. | 887/collapse-confirm |
| Does the imprint generalise out of sample? | **No.** 1/5 walk-forward folds above floor, and that one only because the floor collapsed. | 875/wf-generalization |
| Do its trades carry edge vs a random book of equal activity? | **Anti-edge**, 3/3 seeds, every CI below zero; lives in the maintenance re-trims. | 892/action-ev |
| Is the linear trend book (the fallback) alive? | **Was.** 2021 +1.78 → 2024 +0.04 → 2026H1 **−1.66**. Spot-legal long-only version: 2026H1 **−5.22**, CI excludes zero. Point-in-time universe (dead coins included): +0.56, CI straddles 0. | 898/trend-decay, `pathway.md` |
| Minute-bar surface? | Frictionless Sharpe **+204** (fantasy). **0/864 cells** survive any nonzero fee. | 897/809-corrected |
| Is there *any* signal in daily OHLC? | **Yes** — 109 cross-sectional IC cells survive a 1-day skip. **None** converts to net Sharpe at the honest 80 bp tier (best +0.037). Cost is the wall, not signal. | 915 → D89 |
| Short new listings? | Gross +1021 bp at h21, separates from placebo; net CI straddles zero; median capacity ≈ **$3.6k** per event. | 923 → D88 |

Where it stopped: four open decisions (D86–D89 in `tasks/DECISIONS.md`), each a wall with
numbers attached — no positive-expectancy strategy exists in the program at the honest cost tier.

---

## What failed, and why it is interesting

The full chain is in `experiments/LEDGER.md` and `lore/`. The short version, roughly in order:

1. **Six months of knob-turning on the scalar planner** (encoder tweaks, classifier sharpening,
   entropy, reward shaping, Bella distillation, flow-matching) — all null or marginal. One
   flow-matching win **reversed at 5 seeds** after looking good at 3. Lesson adopted as a rule:
   every experiment must attack a named *structural* hypothesis; knob-turns are noise with a GPU
   bill. (`findings/single_mechanism_exhausted.md`, `findings/flow_matching_promising.md`)
2. **The action space was the defect** — one scalar action can't express a portfolio, so the EV
   of any single decision was below noise and HOLD was rational. Rebuilt as a vector action.
3. **Then the environment bankrupted breadth.** `step_vector` charged every buying asset
   against one cash snapshot with no aggregate budget; a 13-long/13-short vector bankrupted 2 of 4
   envs at step 0. The "fees punish exploration" diagnosis from the day before was the wrong root.
   (828s3/env-wall, `findings/vector_env_cash_wall.md`)
4. **Search works, learning doesn't.** With the simulator as the model, search beats the rule
   by a wide margin. With the learned model, the value head carries ~zero action information,
   and three separate defects (policy-blind proposals, action-blind leaf, learned prior in
   Gumbel selection) were isolated one by one. (849/leaf, 832/E5-inversion1–3)
5. **Self-play erodes everything to the same attractor** — from scratch, from cost-0, from a
   good imprint; three independent convergences to ~99% hold. Then argmax distillation rounded
   every per-asset marginal to HOLD. (832/E5-exit, 832/E5-exit2)
6. **The collapse was adaptive.** The policy had correctly concluded its own actions have
   negative expected value. That reframed the whole program. (887)
7. **The instruments lied, repeatedly.** Framework epochs E-a…E-f each invalidated earlier
   verdicts (a cash bug, zero-price tails, split asset identities, local-time windows). Two
   harnesses claiming to replay the same checkpoint disagreed (882). A pre-registered
   feasibility gate turned out to be unable to fail (924 → D87). The LEDGER preamble lists
   which verdicts are citable as closed and which are only history.
8. **The benchmark died under it.** The linear trend book everything was measured against
   decayed to zero edge after 2023 (898). There is no shippable fallback.
9. **The final measurements were cost walls.** Signal exists in daily OHLC (915); a listing
   event effect exists gross (923); a cross-sectional reversal arm carries +4.4 same-close and
   −4.3 at honest cost with one-day delay (918 → D86). Every one dies at realistic fees.

Also preserved: the ops side — a workstation that hard-locked from an AMD data-fabric sync
flood (`findings/workstation_syncflood.md`), and an ext4 stall that a diagnosis pinned on a bad
SATA cable until an adversarial judge showed the causality was inverted
(`.claude/rules/30-agent-conduct.md`).

## What did work

- **Leak-free, cost-honest evaluation** — per-decision Sharpe, ≥3 seeds, pre-registered
  PASS/FAIL bands with SPEC hashes recorded before launch, nulls logged as results.
- **The corridor** as a grading frame (floor +0.424 / ceiling +29.70 net@21, 830/E0-REAL).
- **The rebuilt action-conditional reward head** (834/E3-p1b PASS) and the first positive
  learning signal: value net ranks states above momentum, ρ +0.21/+0.30 (836/E4-p1b).
- **The adversarial review loop.** Most errors that matter in this record were caught by a
  judge lens pointed where the author was blind — including the planner's own briefs. The
  rules that fell out of that are in `.claude/rules/`.

---

## How to read the repo

| Start here | For |
|---|---|
| `docs/pathway.md` | The research design: hypotheses H1–H6, the rung ladder, kill/pivot logic |
| `experiments/LEDGER.md` | One line per verdict, PASS or FAIL, with epoch caveats |
| `docs/findings/README.md` | Long-form diagnoses, incl. the pre-ledger history |
| `tasks/DECISIONS.md` | Every operator decision and the open walls (D86–D89) |
| `tasks/QUEUE.md`, `tasks/TASK_*.md` | What was built, specced and halted |
| `lore/notes_*.md`, `lore/handoff_*` | Session-by-session record, reversals included |
| `CHANGELOG.md` | The long version |

`CLAUDE.md`, `AGENTS.md`, `SERVER.md` and `.claude/rules/` are the operating instructions the
agents worked under. They are left in because they are part of the experiment.

## Running it

Not turnkey. The market database (`trading_data.db`, Kraken OHLCV) and every checkpoint are
excluded. To rebuild a database: `tools/fetch_daily_kraken.py` / `tools/import_kraken_csv.py`
(public endpoints, no keys); the verification fixture is derived from it by
`fixtures/make_fixture.py`. Then:

```bash
python3 -m venv trader_env && source trader_env/bin/activate
pip install -r requirements.txt            # JAX + CUDA 12; see requirements.frozen.txt
python train_jax_fast.py --help            # Poppy training router
python tools/benchmark_easy.py --help      # canonical eval scorecard
```

Host names in the docs (`workstation`, `gpu-server`) are placeholders; the infrastructure is
described only in outline (`docs/infrastructure.md`). Deploy scripts carry generic defaults and
will need your own hosts.

## Caveats on the record

- Verdicts on the **scalar architecture** are scoped to it — honest evidence about those
  formulations, not laws of nature (`docs/findings/README.md`, scope warning).
- Verdicts measured **before framework epoch E-c** are provisional (LEDGER preamble).
- Most code and most prose were written by AI agents under the operator's direction; tone in
  older lore varies with the model of the month.

## License

TBD.
