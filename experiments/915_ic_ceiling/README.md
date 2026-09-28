# TASK_915 — IC ceiling (G-A/G-B only)


> **Run-57 closing pass (orchestrator, D76-style — removes two unsupported claims, changes no number):**
> (1) the VERDICT status-string rule is POST-HOC (make_verdict.py was last saved ~48 s after
> posthoc_skip1.json existed; no earlier draft is preserved); it labels, it does not decide — the
> pre-committed consequence key comes from the official G-B verdicts only. (2) under skip1 the purge
> margin shrinks from h to h−1 at one boundary row per quarterly refit (O(1) rows; not a driver of
> the F1 result). Comment-only edits to make_verdict.py / posthoc_skip1.py; no result file embeds their shas.
## VERDICT (computed by `make_verdict.py` from `results/`; see `results/VERDICT.json`)

**Status: `GA_PASS__GB_MARGINAL__F1_TIMING_FRAGILE__SKIP1_SURVIVORS_109__SKIP1_GB_FAIL`**
**Pre-committed consequence (task file, D79): "G-A passes / G-B fails -> record the measured
IC/N_eff/turnover triple; breadth/cost are operator decisions with numbers; no trader build."**
Decision owner: **operator**.

- **Pre-registered G-A** (official 160-cell run, gpu-server `/var/tmp/915-run`, frozen code per
  `RUN_MANIFEST.json`, `grid_results.json` sha256 `19eef63a...16e17d9`): **128/160 cells pass**
  (Bonferroni NW-t crit 3.6047). By feature set F1 27 / F2 24 / F3 24 / F4 27 / ALL 26; by h
  h1 36 / h3 33 / h7 31 / h21 28; by target y1 39 / y2 39 / y3 (abs-return vol control) 40 /
  y4 10.
- **Pre-registered G-B** (binding restated_916 = 80bp base-taker on turnover, **borrow NOT
  charged**): 127 FAIL, **1 MARGINAL** — `y4_sign_volscaled__h1__ALL__ridge`, mean IC 0.0668,
  N_eff 303.9, turnover 15.1x/yr, a*=0.05, net Sharpe bound **+0.693** (legacy21: +1.197).
  No PASS. The task's consequence list has no MARGINAL branch; it is mapped to the not-PASS
  branch and flagged for the operator.
- **POST-HOC skip1** (`posthoc_skip1.py`, NOT pre-registered; label starts at close_{t+1}
  instead of close_t, everything else identical): 109/128 official passes survive G-A. By
  feature set F2 24/24, F3 24/24, F4 24/27, ALL 24/26, **F1 13/27**; by target y1 36/39, y2
  36/39, y3 37/40, **y4 0/10**. **The single MARGINAL G-B cell does not survive** (skip1 mean IC
  0.0039, NW-t 0.46). All 109 skip1 survivors FAIL G-B; best skip1 net Sharpe bound +0.037
  (`y1__h21__ALL__gbm`, skip1 IC 0.1201, N_eff 57.2).
- **F1/F2 split.** F1 ridge collapses under skip1 at every tested h (y1 h1 0.1191 -> 0.0123,
  h3 0.0887 -> 0.0034); F1 GBM keeps a smaller IC (y1 h1 0.1342 -> 0.0300). F2 barely moves
  (y1 h1 0.0442 -> 0.0401, h3 0.0603 -> 0.0539). F4/ALL at h1 lose most of their IC too
  (F4 ridge 0.0957 -> 0.0245) — F4's path includes log-close increments ending at close_t.
- **Cause of F1 timing sensitivity: same-close alignment** — F1 features end at close_t and
  labels start at close_t (bid-ask bounce / stale-close consistent, not individually
  established; 3.73% of present cells have close == previous close). The pre-registered null
  family cannot see it (SPEC.md POST-HOC ERRATA E.1).
- MDE: pinned table used t=3.0; at the stated t_crit 3.6047 no POWERED/UNDERPOWERED flag
  flips (h1/h3 powered, h7/h21 underpowered at both; SPEC.md E.2).

### skip1 table — target y1 (full 128-cell table in `results/VERDICT.json` `posthoc_skip1.table`)

| FS | h | est | official mean IC (NW-t) | skip1 mean IC (NW-t) | skip1 G-A |
|---|---|---|---|---|---|
| F1 | 1 | ridge | 0.1191 (35.21) | 0.0123 (3.56) | fail |
| F1 | 1 | gbm | 0.1342 (39.41) | 0.0300 (9.14) | PASS |
| F1 | 3 | ridge | 0.0887 (22.29) | 0.0034 (0.61) | fail |
| F1 | 3 | gbm | 0.1078 (24.20) | 0.0418 (9.14) | PASS |
| F1 | 7 | ridge | 0.0634 (9.84) | 0.0101 (1.23) | fail |
| F1 | 7 | gbm | 0.0855 (13.61) | 0.0541 (8.18) | PASS |
| F1 | 21 | ridge | official G-A fail | not re-scored | - |
| F1 | 21 | gbm | 0.0803 (7.78) | 0.0671 (6.65) | PASS |
| F2 | 1 | ridge | 0.0442 (11.49) | 0.0401 (11.14) | PASS |
| F2 | 1 | gbm | 0.0451 (13.26) | 0.0283 (8.38) | PASS |
| F2 | 3 | ridge | 0.0603 (11.30) | 0.0539 (10.15) | PASS |
| F2 | 3 | gbm | 0.0589 (11.87) | 0.0478 (9.50) | PASS |
| F2 | 7 | ridge | 0.0764 (10.58) | 0.0703 (9.67) | PASS |
| F2 | 7 | gbm | 0.0682 (10.88) | 0.0629 (9.94) | PASS |
| F2 | 21 | ridge | 0.1059 (9.33) | 0.1017 (8.92) | PASS |
| F2 | 21 | gbm | 0.0913 (10.07) | 0.0866 (9.51) | PASS |
| F3 | 1 | ridge | 0.0484 (12.52) | 0.0408 (10.72) | PASS |
| F3 | 1 | gbm | 0.0429 (12.27) | 0.0379 (11.06) | PASS |
| F3 | 3 | ridge | 0.0667 (12.38) | 0.0595 (11.15) | PASS |
| F3 | 3 | gbm | 0.0600 (12.99) | 0.0561 (12.06) | PASS |
| F3 | 21 | ridge | 0.1151 (10.08) | 0.1099 (9.69) | PASS |
| F4 | 1 | ridge | 0.0957 (30.05) | 0.0245 (8.98) | PASS |
| F4 | 1 | gbm | 0.1046 (31.61) | 0.0343 (12.15) | PASS |
| F4 | 3 | ridge | 0.0699 (17.45) | 0.0242 (6.49) | PASS |
| ALL | 1 | ridge | 0.1236 (37.34) | 0.0387 (11.10) | PASS |
| ALL | 1 | gbm | 0.1413 (44.40) | 0.0450 (12.46) | PASS |
| ALL | 3 | gbm | 0.1198 (27.72) | 0.0658 (13.00) | PASS |
| ALL | 21 | gbm | 0.1299 (13.12) | 0.1201 (11.43) | PASS |

(Selected rows; F3 h7, F4 h7/h21, ALL h3-ridge/h7/h21-ridge omitted here — all PASS, see JSON.)

### Provenance (post-hoc pass, 2026-09-28)

- skip1 run: gpu-server tmux `915fix`, `/var/tmp/915-fix/Trader` (copy of the implementer
  scratch + the post-hoc files), `safe_run.sh --cpu --mem 16G`, OMP/OPENBLAS/MKL=8, 2708s of
  cell time for 128 cells; official outputs read only via a copy (`/var/tmp/915-fix/
  official_copy`, byte-identical cells + `grid_results.json`). Harness check: the k=0 re-score
  of all 64 ridge cells through `posthoc_skip1.py` reproduces the official mean IC exactly
  (max abs diff 0.0).
- `results/` = copies of `/var/tmp/915-fix/results/` (posthoc_skip1.json,
  grid_results_annotated.json, VERDICT.json).
- Post-run code change: `gates.py` (N_TESTS derived from `grid.ALL_CELLS`; `gb_result` returns
  `borrow_charged: False`, `provisional_if_pass: True`). Frozen sha (used by the official run)
  `4d9ffad1372ad6de8aaec4ed5a78b48d1ac254f0c2fcd46f5e37e7bea8a3b818`; post-run sha
  `8ecafd13f09b64c6d931197b7d0974ae62d4299fe100740e3232a9b28069f0a6`.
- SPEC.md: pre-registration text untouched; POST-HOC ERRATA appended (sha before
  `d9b7d4050345adee403e41c5918edce0957418f740a3d1739476c0d6b8c2c95e`, after: see report /
  `sha256sum SPEC.md`).

---

## File map (original)

See `SPEC.md` for the pre-registered math (targets, features, estimators, gates, MDE, GP
scoping disclosure). This README is a file map + run pointer.

## Files

- `loader.py` — PIT OHLCV panel: fixture `ohlcv_daily` + 858-seam CSV extension, ALL FIVE
  columns re-verified at load time (not close-only). `load_panel()`, `apply_liquidity_filter()`.
- `yang_zhang.py` — Yang-Zhang/Parkinson/Garman-Klass range vol, HAR-RV terms. Causal.
- `signature.py` — truncated (level<=3) log-signature via Chen's identity, numpy only (no
  iisignature/esig). Rolling window via the prefix-inverse trick.
- `targets.py` — the 4 targets + market-timing arm, leak-free by construction.
- `features.py` — F1 (trend), F2 (range/vol), F3 (cross-sectional), F4 (signature).
- `stats_utils.py` — per-date Spearman IC, Newey-West SE, Ledoit-Wolf, N_eff, Bonferroni, the
  2026-09-05 null family (block-Rademacher/circular-shift/per-date-permutation), tau_IC, SR0
  multiplicity, D85 MDE formula.
- `gp_fundamental_law.py` — N_eff/Sigma (exact) + GP turnover/TC (disclosed simulation
  proxy, not the closed-form matrix Riccati — see SPEC.md Sec.8).
- `grid.py` — the 160-cell grid, expanding-window quarterly refit, purge>=h.
- `gates.py` — G-A / G-B evaluation from a cell's OOS forecast panel.
- `mde_report.py` — D85 MDE table from panel asset counts (no fitting), run before SPEC pin.
- `run_grid.py` — CLI driver: loads panel, builds features/targets once, iterates the grid
  with per-cell checkpointing (`--outdir/cells/<cell_id>.json`), evaluates G-B for G-A-passing
  cells, writes `grid_results.json` + a `DONE` marker.
- `posthoc_skip1.py` — POST-HOC one-day-skip label re-score of every official G-A pass
  (module's own fit/score/gate code; only the target is shifted) + G-B on survivors.
- `annotate_results.py` — POST-HOC annotation of a copy of the official results (borrow flags,
  per-cell underpowered at both t conventions, timing_fragile).
- `make_verdict.py` — writes `results/VERDICT.json` from the result files.
- `tests/` — unit tests per module, plus `test_positive_control.py` (adversarial-pass
  discipline: the G-A guard is proven to FIRE on a planted strong signal and to STAY QUIET on
  a shuffled-target negative control, not just proven quiet on real data).

## Running

```
trader_env/bin/python -m pytest experiments/915_ic_ceiling/tests -q     # server, wrapped
trader_env/bin/python experiments/915_ic_ceiling/run_grid.py --outdir /var/tmp/915-run --cells all
```

`--cells N` (an integer) runs only the first N cells of `grid.ALL_CELLS` — used for the
fixture smoke and the timed dry run before committing to the full 160-cell launch.

G-C (the GP closed-form book, graded on `trader/` + `eval_vector`) is **out of scope for this
module** — named remainder, see SPEC.md Sec.9.
