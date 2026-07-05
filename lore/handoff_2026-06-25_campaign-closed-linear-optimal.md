# Handoff — 2026-06-25 (≈seq 9) — Trading-model campaign CLOSED (linear-optimal); apothecary shipped

Previous: `lore/handoff_2026-06-18_804-launched-failed-fixed-rerunning.md`

- **Branch:** master
- **Working dir:** /home/lukas/Trader
- **Session focus:** Closed the trading-model campaign (806→811): the 1-min Kraken cross-sectional signal
  is **REAL** (IC 0.06–0.15), **LINEAR-OPTIMAL** (no model beats the closed-form ridge), and **COST-BOUND**
  (edge < Kraken taker). The model side is **fully closed**; the binding constraint is **cost, not
  modeling**. Side quest: built + shipped the **apothecary** Claude Code skill marketplace.

## Next action
Operator's call on the fork. The model-side is closed; the open lever is a **DATA/PROBLEM decision, NOT
modeling**, and per **Fork-C** that means **no venue change / stay Kraken** (so not a rebate venue). Two
offers pending the operator's choice: **(a)** draft a Bloom retrospective of the real→linear-optimal→
cost-bound→closed arc, or **(b)** inventory the methodology into a reusable skill. Otherwise the campaign
is at rest — nothing running, server idle. **Do NOT propose more modeling on the 1-min signal** (closed).

## Confirmed compaction rubric

```
Compaction rubric (chandoff 2026-06-25):

KEEP IN FULL:
- CAMPAIGN CLOSED (806→811, model-side): 1-min Kraken cross-sectional signal is REAL (IC 0.06–0.15),
  LINEAR-OPTIMAL (811: model loses to ridge at every H), COST-BOUND (808/809/810). Binding constraint
  = cost, not modeling. Fork-C = stay Kraken, no rebate venue. Trading-model goal unreachable under the
  constraints; the methodology + the definitive negative ARE the deliverable. No more modeling proposed.
- OPEN (operator hasn't chosen): I offered (a) a Bloom retrospective of the arc, or (b) inventory the
  methodology into a reusable skill — awaiting their call. The data/problem fork is also theirs to weigh.
- Apothecary marketplace SHIPPED: apoage/apothecary (3 skills: chandoff/feverfew/erythroxy). Done.

COMPRESS + POINT:
- Full 806→811 arc, per-run numbers, the audit that caught 2 of my own artifacts → lore/notes_2026-06-25
  (close), 06-23 (810), 06-22 (808/809), memory/project_modeling_surface_exhausted.md. Don't re-narrate.
- The repeated systemd-oomd fights → one line: "oomd kills on PSI not the cgroup cap; fix = VRAM-resident
  + cache the 2.5GB build_aligned_ohlcv load to a 513MB npz (runs/811_ic_probe/cache_close.py)."

DROP:
- The naming/policy/JAX-skill chat (coca/Anthropic-drugs/car-tuning/"not a weapon dealer") — resolved:
  keep `erythroxy`, ship to own marketplace, no policy surface.
- The hibernation/connectivity troubleshooting (resolved: server-side tmux survives; my "workstation
  hibernating" guesses were wrong and corrected — don't preserve the wrong guesses).
- Blow-by-blow of each smoke/oomd-kill/design-fix iteration (final designs + verdicts are in lore).

VERBATIM ANCHORS:
- 811: model IC 0.093/0.064/0.024 vs linear 0.151/0.141/0.060 (H=1/5/20).
- apoage/apothecary; runs/810_postcost_headroom/ + runs/811_ic_probe/ (ic_model.py, cache_close.py,
  _close_cache.npz). Audit standard: "net-beat without GROSS(cost=0)-beat on holding-period rows = artifact."

UNCERTAIN:
- Is the Bloom-retrospective / methodology-skill offer a live pending task, or just an offer? (surfaced)
- Does the operator intend to pursue the data/problem fork at all, or is the campaign at rest?
- 809's recompute_best.py never ran (moot — 809 ceiling-bound): footnote or drop entirely?
```

## Git state
- Branch `master` @ `7ac5570` (TASK_804 verdict). **The entire 806→811 campaign is uncommitted** — it
  lives in `runs/` (gitignored) + uncommitted `lore/`. Standing pattern: commit only when asked.
- `git status --short`: **1630** changed/untracked (the known de-facto-canon noise — do not `git add -A`).
- Key artifacts this session (uncommitted): `lore/notes_2026-06-22.md`, `notes_2026-06-23.md` (+ ⚠️ 810
  correction banner), `notes_2026-06-25.md`; `~/.claude/.../memory/project_modeling_surface_exhausted.md`;
  `runs/810_postcost_headroom/` (corrected model.py + result.md); `runs/811_ic_probe/`
  (ic_model.py + cache_close.py + _close_cache.npz + result.md + GUARDS implicit).
- **Apothecary** lives in a separate GitHub repo `apoage/apothecary` (pushed; not in this tree).

## git log -5
```
7ac5570 TASK_804 VERDICT: branch C NULL on both surfaces (no promote)
1ec61ad TASK_804: front-end NULL verdict (shift>=real 3/3) + launch arms 4-5 (dynamics)
b28e500 handoff seq 7: 804 launched-failed-fixed (4 blockers), 6/9 evald, retry-runner closing gap, null lean
79edfae lore §208: TRUE root cause was dropped cache_dir dict key (e02953e); relaunch clean, 2852s->26s tokenize
e02953e TASK_804: thread token_frontend_cache_dir into SegmentPool oracle_cfg
```
