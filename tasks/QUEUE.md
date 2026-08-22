# QUEUE — the ranked dark-factory work order

The factory's single entry point (planner-maintained). STATUS banners on task files remain the
truth for what's done; this file only orders what's OPEN. The factory takes from the top, skips
gated rows (noting why), and treats FILLER rows as parallel/anytime slices. Last groomed:
**2026-08-22 (planner, post factory run 38 — the seven-ship run).** Run 38 RATIFIED: **TASK_900,
TASK_894, TASK_897 (instrument), TASK_881S, TASK_898 (instrument), TASK_881G, TASK_896 ALL
SHIPPED.** Notables carried: 898's instrument caught a verdict-changing confound pre-launch (the
spot-legal variant silently de-leverages to 43%/27.6% gross exposure — a leverage-matched series
now ships with it; FULL reads +0.321 unmatched vs −1.308 matched, so the variant must be read on
the matched series); 881G's gate (i) failed the DOC (46 cases authored from the doc alone — a
reusable test shape for load-bearing docs); 896's re-pin was verified by an independent
1,218-key diff (old `6bede1e9…` beside new `330484f5…` in the LEDGER). **Two rulings taken
2026-08-22:** D55(a) — 897's interval gets 892's calibration method (raised block cap) before
any cell is read, the band stands; D56(a) — blast radius MEASURED by the planner: **6 MATERIAL
slots** (the 868 identity-split class; exact-spelling reads miss 33–51% of window rows), so
897's runner applies the canon alias union at READ time for universe ranking AND panel build,
809 lineage untouched. **The 892 real leg is graded: ANTI-EDGE (D47)** — the queue's
measurement rows (897, 898) are the program's live edge-structure front.
**Ready (physically ordered): 1 TASK_881D (protection + deleter + corpus; UNBLOCKED by 881G) →
2 TASK_897 calibration adaptation (small instrument step per D55, then its srv leg is
planner-scheduled). FILLER: TASK_899 (data backfill — planner/server ops). GATED: TASK_880
(D34+D35; the D35 gate opens after 881D ships + real score files the new reader resolves).**
**Planner-scheduled, no factory needed: TASK_898's real leg** (instrument shipped; stage the
daily-table inputs to srv per D54 and launch) **and TASK_897's srv leg** (after the calibration
step). **OPEN decisions: D43 only.**
_Next takeable: **TASK_881D** — the last big infrastructure row; its corpus authors against
881G's written grammar, and the D32/D34 `--apply` freeze conversation finally happens with a
live corpus._

| # | Task | Size | Note |
|---|------|------|------|
| 1 | `tasks/TASK_881D_PROTECTION_DELETER.md` | M | **UNBLOCKED (881G shipped run 38).** Scope item 0: author the real `PROTECTED.txt` corpus against 881G's written grammar (the doc is the spec, tested). The `--apply` freeze STANDS until this ships AND 880's D35 gate opens. Deliberately late-ranked, as 881 was. |
| 2 | `tasks/TASK_897_CORRECTED_809_REMEASURE.md` (calibration step) | XS/S | Instrument shipped run 38; the D55 step remains: run 892's calibration METHOD against 897's statistic with the block cap raised, certify coverage, record in the launch note. Then the srv leg is planner-scheduled. D56 measured: canon alias union applied at read time (6 MATERIAL slots disclosed per-cell). |

Superseded grooming history and the full text of shipped rows live in
[`tasks/QUEUE_ARCHIVE.md`](QUEUE_ARCHIVE.md) — moved 2026-08-11, byte-for-byte, nothing summarised.
Ruled decisions D29–D37 likewise live in [`tasks/DECISIONS_ARCHIVE.md`](DECISIONS_ARCHIVE.md).

## Ready — take from the top

_Run 31/32/33 wrap narratives and the 887 report moved to
[`tasks/QUEUE_ARCHIVE.md`](QUEUE_ARCHIVE.md) 2026-08-16, byte-for-byte (token economy — every
factory startup reads this file; the lessons they carried are quoted as gates in the task files)._

| # | Task | Size | Note |
|---|------|------|------|


_Sequencing (2026-08-08, supersedes the 2026-08-06 note below): **874 first** — XS, and it is the artifact 883 makes mandatory. **883 second** — it closes the fail-open that manufactured the 882 crisis, and every measurement the ladder is about to take runs through that harness. **877 third** — it shrinks the sparse-slot set, which is exactly the surface 878 then re-specifies (878's own out-of-scope says do 877 first). **878 fourth and alone** — it rewrites eval-core execution semantics with every parity gate in the repo riding on it. **881 last** — deliberately. It is real work and it is not urgent: unreadable ⇒ UNSCORED is fail-safe, the `--apply` freeze STANDS, srv holds 162 GB free. Four of the last five planner errors were boundary calls inside this one lane; ranking it last is the structural correction, not a demotion of the work._

_(2026-08-06 note, superseded — kept because its 876/873 half is what actually happened:)_ _Sequencing (2026-08-06): **876 first** — emission gates the long base-training program, and it
touches the training save path that no other Ready row touches. **880 second** — it lifts an
operational freeze on a tool that deletes remotely, and the long run will want that tool. **873
third** — it changes what every suite checks at start-up, so everything after it is measured
under the pin. **877 fourth** — it shrinks the sparse-slot set, which is exactly the surface 878
then re-specifies (878's own out-of-scope says do 877 first if it is scheduled). **874 fifth**
(XS, independent). **878 last and alone** — it rewrites eval-core execution semantics with every
parity gate in the repo riding on it; the factory deliberately deferred it in run 26 to finish
one row properly, and that judgement is ratified here._ _Original note:_ 872 first (live path). 873 and 874 touch different surfaces from each other and from
872 — but 873 changes what every suite checks at start-up, so take it BEFORE 874 and re-run 874's
suites under the pin. Serial is safest; all three are XS/S._

_Specced 2026-08-16 as TASK_891 (FILLER row 5): the bare-`python` trap that cost two launches
(861 chain, 879 rung-3). The 2026-08-06 recommendation text is retired into the task file._

_Recommended, not yet specced (run-34 housekeeping): the watchdog suite
(`--stall-timeout 3 --stall-grace 2`) is a **pre-existing load-sensitive flake**, reproduced at
HEAD in a clean worktree (10/18, 15/18, 15/18) — not caused by any candidate; it costs
verification time every run it touches. Worth its own XS row if it keeps firing._
_Run-34 process lessons now standing clauses for future judge briefs: never `git stash`/
`checkout`/`reset` on the live tree (use a worktree — a judge stash briefly stripped a concurrent
lane); "disable this guard and prove the suite goes red at the REAL drop site" fired FOUR times in
one run; verification agents need explicit do-not-write fencing (a default `--out` landing on a
tracked artifact is a write nobody thinks of as an edit)._

_Recommended, not yet specced (both named in shipped banners, neither urgent):_
_(a) `mcts_trader/data/database.py` — the last known unaddressed RW reach at the canon layer
(aiosqlite, real live-write path; needs a dual-connection refactor, beyond XS — 870 remainder).
(b) `_ro_connect`-style helpers still coexist in three places (shared / `837_walkforward` /
`trader/data.py::_connect`) — "one implementation" holds for 870's touched surface, not repo-wide._

_Shipped 2026-08-02/03 factory run 24 (the D31 re-base wave, strictly serial):_
- _**TASK_866** (`task866-weight-on-pin`, `ac8dfe1`) — weight-ON parity pin reads the tracked 832 baseline as its own-config oracle; the 864 incident class closed. See the task banner + CHANGELOG._
- _**TASK_867** (`task867-861-baseline-promote-port`, `d670d57`) — the 864 promote gate ported to both 861 baselines; the `--quick` silent downgrade is a named REFUSAL; leg `vii_baseline_pins` pins the COMMITTED baselines (tamper → FAIL pre-leg). See the task banner + CHANGELOG._
- _**TASK_868** (`task868-asset-identity-repair`, `3388a0c` + 861 promote `a687aff`) — D31 (a1) load-time alias union (NO DB writes, fail-loud agreement gate, disclosed) + (a2) causal frozen-cell open-gate (whole-action force-HOLD; D30 fill still values; PV exact mask ON). Frozen-interior 2,013 → 1,011 cells; honest not-real share 10.67% → 9.73%. Medium panel: 1 MAJOR (fixture-regen blast radius unattributed) + 4 MINORs closed and re-verified. See the task banner + CHANGELOG._
- _**TASK_869** (`task869-rebase-wave-semantics`, `3e0baa9` + `9a5b0b2` + `894ab36`) — D31 (a3): `_to_epoch` → UTC everywhere (TZ-invariance byte-proven) + daily teacher floor masked (measured inert, defence-in-depth). Release round caught + fixed a REAL leg-vi break (stale constants + 150-iter collapse → 250). Final re-verify: 861/863/865 ALL LEGS PASS. **Wave CLOSED — references un-gate planner-side.** See the task banner + CHANGELOG._

_(prior Ready rows, now shipped:)_

_Recommended new planner rows from run 23 (not yet specced, listed so they are not lost):_
_**(D) The canon panel path still opens `trading_data.db` READ-WRITE** — `mcts_trader/env/vectorbt_env.py:515` and `:731` use a plain `sqlite3.connect`, reached via `build_panel` → `create_jax_env_historical` on the DEFAULT unpinned path (and by the equivalence gate's FLOOR leg at `eval_vector.py:3152`). TASK_865 routed all 11 of its OWN sites through `_ro_connect`, and the orchestrator verified the fix (a hot-journal db is left byte-identical; the probe raises `attempt to write a readonly database`; a nonexistent path is refused with nothing created) — but the canon env loader is a wider surface than 865 and was deliberately left alone. The exposure is real and demonstrated in principle: SQLite performs hot-journal recovery on the first READ through a read-write handle, no write SQL involved, on a 24 GB asset whose only backup predates the 858 backfill. **XS/S, fail-closed, no numerics risk** (a `mode=ro` handle cannot change a SELECT's result). Recommend: route both loaders through the same helper, or refuse a protected `--db` on any RW path._
_**(A) Daily-framed teacher floor walks UNMASKED** — same defect class as D30 rider (a), but masking it moves shipped daily-pinned numerics, so it needs its own ruling rather than a factory fix. **(B) `_to_epoch` resolves `--start/--end` in LOCAL time**, so a benchmark window's extent depends on which machine ran it (measured: a 308-row CEST offset on the OOS window; the reason several counts differ by one between parties — 1734 vs 1735 panel rows, 2 vs 3 stale tails, 90 vs 91 daily fabricated rows). A determinism hazard for a read-of-record operator across workstation and `gpu-server`; changing it re-bases every existing window number, so it is a planner call. **(C) fixture-native interior-gap coverage** — under the shipped (bar-window) definition the fixture has no interior gap, so that proof currently rides on a synthetic DB; `fixtures/make_fixture.py` would need a multi-bar hole._

_Shipped 2026-07-31 factory run 23:_
- _**TASK_865** (`tasks/TASK_865_MINUTE_FRAME_PIN.md`) — minute-framed frame pin, D30 opt (i). See the task banner and the CHANGELOG entry. **Remainder → planner: D31 OPEN, gates the 861 real leg.**_

_(prior Ready row, now shipped:)_

_Sequencing note: these rows touch `experiments/832_linear_imprint/` and
`experiments/861_search_teacher_imprint/` — SERIAL queue, take in rank order. All GPU legs remain
planner-scheduled; the amended 861 chain (minute-pinned frame, 3 imprint seeds, median > +0.839 with
≥2/3 above — see the D29 + D30 addenda in TASK_861) launches only after 865 ships verified
(D30 RULED 2026-07-30, planner opt (i); leg-1 hard checks + masked floors + leg-4 frame-source
gate per the D30 addendum)._

_Shipped 2026-07-29 factory run 21:_
- _**TASK_863** (`task863-imprint-frame-pin`, `609d6d3`) — `--panel-frame-from-checkpoint` on
  `imprint.py`, calling `eval_vector`'s proven 853 validator/framed loader rather than re-deriving
  them (`eval_vector.py` byte-unmodified); 861 provenance gate now fail-closed BOTH directions,
  same-membership reorder refused. **OFF bit-identical proven five independent ways** (2 orchestrator
  re-proofs at matched `_REPO` with identical `--out`/`--ckpt-out` → stdout byte-identical, 170
  leaves/0 differing; fixer's mirrored tree 0-of-759 + 0-of-545; regression judge's **30-leg** legacy
  battery, zero rc/artifact divergence; fresh judge's mirrored-tree control). Smoke **8 → 11 → 13
  legs**; selftest 12/12; 861 e2e 9/9. 3-member panel + a **FRESH fail-closed judge** over the fix
  pass: **1 MAJOR + 6 MINOR + 3 NIT closed over 2 fix rounds**, every new guard demonstrated to FAIL
  when disabled. MAJOR (pre-existing, fixed pin-gated): a degenerate pinned run exited **0** with a
  green-looking artifact (null Sharpes beside `all_losses_finite: true`, no marker) whenever
  `--smoke-assert` was off — the default for any non-repo-fixture db, **i.e. the real leg**. Also
  closed: masked-cell dilution of every reported scalar (0.8929 vs 0.8571 active-only); a
  false-AND-vacuous `panel_source` pin (2 judges independently); a FALSE "flat-zero features"
  disclosure (896/6048 cells at a constant 0.2) now **asserted** by the smoke; three malformed-frame
  paths; and `panel_frame: {}` as a real **fail-open**. The force-HOLD mask had **no regression guard
  at all** (inert on the tv path + zero partial slots in the smoke) — now boundary-asserted.
  **Spec correction (binding): D28 + TASK_861 line 179 wrongly state `imprint.py:620` "only WARNs" —
  it is a hard `RuntimeError`.** **Remainder → planner, BLOCKED on D29.**_

_Shipped 2026-07-28 factory run 20 (861 + 862 parallel lanes, disjoint surfaces as planned):_
- _**TASK_861** (`task861-search-teacher-imprint`) — `--collect-chosen-vectors` (eval_vector) +
  `--teacher-vectors` (imprint); both flags OFF **bit-identical** (orchestrator-reproduced vs
  `HEAD`: 0 differing leaves on both modules). 3-member panel, **2 fix rounds, 5 MAJOR closed** —
  the pinned +8.024 operator was ungated (3 verifiers imprinted garbage teachers with green
  gates, and the falsification metric scored garbage HIGHER than the real teacher); no payload
  integrity (`x−0.5` codes swapped a +13.893 teacher for −0.649); empty-string flags failed OPEN
  (one clobbered the tracked parity baseline); the two pre-registered legs were mutually
  un-runnable; and — found by brute force — the new replay guard is a TRAJECTORY not a TARGET
  check: **315/400 cells (78.8%) admit trajectory-inert substitutions**, so 78.8% of BC targets
  could be rewritten with net@21 bit-identical and every gate green (closed via a payload-vs-header
  action-composition check). **Remainder → planner, BLOCKED on D28.**_
- _**TASK_862** (`task862-imprint-scorecard`) — `experiments/835_stage4_eval/scorecard.py` + the
  canonical 7-window manifest + smoke; NEW FILES ONLY, zero diffs to existing tracked files
  (re-confirmed). 2-member panel, **2 fix rounds, 3 MAJOR + 1 latent-MAJOR closed** — any
  non-`WindowFailure` exception destroyed the whole artifact (a non-numeric floor value killed a
  completed battery, no json/no md, contradicting its own "never silently skipped" contract); the
  operator block was asserted-not-observed (~25 knobs unpinned — a top-m battery would emit a
  status-OK card headed "848 read of record"); the recovery path re-ran the call that had failed;
  and the write sat outside every guard (truncated unparseable JSON). Also closed: ambient
  `JAX_PLATFORMS` silently selecting the backend while the header said `null`. **Decision-changing
  finding → D27. Remainder → planner: canonical manifest on `ckpt_imprint_full` (workstation).**_

_(prior Ready rows, now shipped:)_
| 1 | `tasks/TASK_861_SEARCH_TEACHER_IMPRINT.md` | M | **D26 RULED (operator, 2026-07-27): "prepare dark factory tasks" — opt (a).** Marry the two proven pieces: the +8.024 true-env-leaf search operator (PINNED verbatim per `runs/832_e5/results_e5inv854_db_pw0_*.json`) as an OFFLINE teacher × the exact 832 weight-native imprint recipe (+0.839/+0.898-validated). Two flag-gated extensions: `--collect-chosen-vectors` on the search walk (`eval_vector.py`, pure recording — walk numbers identical ON/OFF) + `--teacher-vectors` on `imprint.py` (hard chosen-vector targets, fail-loud provenance gate; NOT visit-distribution CE — the ExIt2 collapse mode). Falsification gate must be high-SNR (the 860 lesson — no small-n held-out quantities). OFF = bit-identical both modules. **Real bars = planner post-ship (pre-registered in the spec): teacher pass on the pinned `--end 2025-10-04` panel (1673 dec, flagged) → imprint → sanity replay > +0.839 → 858-extended OOS + WF quarters.** |
| 2 | `tasks/TASK_862_IMPRINT_SCORECARD.md` | S | FILLER (parallel-safe). Codify the imprint's D17 battery into one command + one fail-loud scorecard artifact: DEC-2 regime windows (floors from `runs/831_ext/*.json`) + OOS window + WF quarters via the 848 policy-argmax read-of-record configuration. **NEW FILES ONLY — must not edit `eval_vector.py`/`imprint.py` (861's surface); merge-safe by construction.** No silent window skips; schema documented (the eventual gate-ladder submission cites it). Real leg = planner (workstation, DB source of record). |

_Independence note: 861 edits `eval_vector.py` + `imprint.py`; 862 adds
`experiments/835_stage4_eval/scorecard.py` + a windows manifest and is FORBIDDEN from editing
861's files — zero file overlap, parallel lanes safe. Neither row trains anything; all GPU legs
are planner-scheduled. Search-leaf lane (857/860 distill line) is PARKED per D26 — no rows._

_Shipped 2026-07-26 factory run 19:_
- _**TASK_860** (`task860-reward-head-codistill`, `736fdc6`) — `--codistill-reward` (OFF =
  shipped-857 bit-identical, 615/615 leaves diff 0): reward head trained on true-r0 counterfactual
  targets alongside the value imprint. Panel BLOCKER: the shuffle-falsification gate was fixture-noise
  (deployed-Q at n=12) AND the newly-trainable reward head defeated the frozen-reward assumption —
  replaced with a high-SNR train-set fit gate (~n=972, floor +0.45, 0.38 cushion), 8/8 seed sweep
  re-verified. **Remainder → planner: real 3-seed co-distill + corridor (RUNNING).**_

_(prior Ready row, superseded:)_
_| 1 | `tasks/TASK_860_REWARD_HEAD_CODISTILL.md` | S/M | **D24 RULED (operator, 2026-07-26): "seed co-distill."** Extend the shipped 857 recipe: emit true-r0 targets, `--codistill-reward` unfreezes the `dynamics_dense2` reward projection (λ-weighted reward CE alongside the value CE); r_model calibration column added to the instrument. OFF = shipped-857 bit-identical. Real bars = planner post-ship: deployed-Q ≥ +0.5 → the same corridor (floor +0.424; value-only best-seed reference −4.23). |

_Context: 858 real ingest DONE (8.28M rows, 62/64 → 2026-03-31); first binding OOS reads landed —
imprint +0.898 net@21 OOS (Δ+2.33 over the on-window floor, LEDGER 832/E5-oos-real); 856 top-m NOT
confirmed OOS. The imprint is the deployable-artifact frontrunner; its walk-forward battery is
planner lane._

_Shipped 2026-07-24 factory run 18:_
- _**TASK_858** (`task858-minute-backfill`, `19f955c`) — `tools/backfill_minute.py`, the single
  sanctioned append-only write path to `trading_data.db`. Panel closed 2 BLOCKER + 2 MAJOR in the
  guard architecture (commit-before-guards → atomic; false-firing append-only proof → subset proof;
  fail-open backup gate → hardened; 11/64 unreachable dump spellings → `_canon_key`). Real ingest
  correctly STOPPED (no dumps on disk). **Remainder → operator/planner: acquire Kraken OHLCVT
  quarterly dumps (Q4-2025…Q3-2026), fresh DB backup, run on the WORKSTATION.**_
- _**TASK_859** (`task859-eval-activity-metadata`, `d32c64a`) — additive `activity` block in eval
  JSONs; no panel findings. Caveat: `n_long`/`n_short` = action-code intent, not net-position
  direction (a Sell closing a long buckets as "short")._

_Independence note: 858 touches the data pipeline (`tools/`), 859 touches `eval_vector.py` result
metadata — zero file overlap with each other AND with the in-flight 857 server runs (which only
read the DB and write to `runs/`). 858's real-ingest leg (iii) runs on the WORKSTATION (the DB
source of record lives here); do not run it on the server DB._

_Context: the 857 binding reads (3 distill+corridor chains) are running on gpu-server overnight;
their verdicts + the D24 fork are planner lane, not queue rows._

_Shipped 2026-07-24 factory run 17 (D23 queue cleared):_
- _**TASK_856** (`task856-decision-rule-probe`, `861ea63`) — `--decision-rule {argmax,threshold,top-m}`
  (+ `--decision-tau`/`--decision-m`) on the 848 replay operator + the `net_dependent` grading-hygiene
  fix (fail-loud on zero-learned-component operators). Panel MAJOR: top-m originally vacuous in the
  collapse regime (assigned full argmax incl. HOLD) — fixed to best NON-HOLD on selected assets,
  mutation-tested. Default bit-identical. **Remainder → planner: the real rule-grid sweep (scheduled).**_
- _**TASK_857** (`task857-vhat-leaf-distill`, `15a80f5`) — `experiments/857_vhat_distill/distill.py`
  (counterfactual dataset via the 851 true-env leaf → value-head imprint, dynamics+value only) +
  `--leaf-eval vhat-distilled` (deployable model-leaf on the distilled ckpt; `net_dependent: true`).
  3 fix rounds; calibration instrument hardened (held-out per-state Pearson; deployed-Q leg;
  `--shuffle-targets` falsification hook). **Decision-changing finding → D24 (OPEN): v̂-alone improves
  (+0.233→+0.342 fixture) but deployed-Q (r_model+γ·v̂) does NOT lift — frozen 832 reward head is a
  poor r0 estimator (corr≈0.27). Remainder → planner: real distill + corridor read (scheduled).**_
- _**TASK_855** (`task855-flag-hygiene-minors`, `a19c528`) — `value_shift` inert-combo fail-loud
  (sibling to 854's `value_scale`), ST11 pins both; 835 smoke baseline refreshed (additive keys only,
  tiers-SHA unchanged)._

_Shipped 2026-07-22 factory run 15:_
- _**TASK_853** (`task853-inactive-mask-oos-panel`, `c8ee701`) — `--panel-frame-from-checkpoint`:
  fixed 64-slot daily panel in checkpoint order, zero-coverage + pre-listing slots INACTIVE and
  dual-disclosed, fail-loud on identity scrambling; OFF bit-identical (SHAs), selftest 10/10.
  Verification caught a REAL LEAK (partial slots tradeable in a fabricated pre-listing region —
  3 verifiers independently) — fixed to flat-zero + price-validity force-HOLD. Plus the D20 audit
  (`5bbbd26`, read-only) independently matching the planner's concurrent ruling. **Remainder →
  planner: 853 numeric leg (+0.839 ±0.06 round-trip, needs the imprint's 64-symbol provenance
  JSON) then the binding OOS benchmark.**_

_Shipped 2026-07-21 factory run 14:_
- _**TASK_851** (`task851-true-env-leaf`, `41561d4`) — `--leaf-eval {model,true-env}` on the
  Sampled-MuZero search (D19 opt (a), the E1 +5.29 operator); default `model` bit-identical.
  Fixture inversion (i) PASS (+6.801 vs −6.778 net@21) + E1-consistency anchor (ii) PASS
  (fresh-init +2.198 true-env vs −10.287 model, floor −3.437). Medium panel, 2 rounds: **2 MAJOR
  found + fixed + independently re-verified** (unique-per-env-prices dispatch crash; unreproducible
  headline numbers — root-caused to a concurrent TASK_852 edit transiently mutating the shared
  `eval_vector.py` file mid-review, now pinned via LEDGER + committed artifacts). Zero
  regressions, selftest 10/10. Real binding leg + Amendment-6 ExIt re-try = planner, post-ship._
- _**TASK_852** (`c105038`, finding-style commit — not tagged, this is a STOP not a ship) —
  diagnosed + partially fixed the daily/minute panel divergence (LEDGER 850/eqgate). Lookahead
  hypothesis REFUTED; root cause = densification of zero-coverage assets (78% of the gap).
  Fix (active-mask filter + fail-loud pad guard) verified — but FLOOR leg still fails its own
  tolerance (residual Δ0.0555 vs 0.02) ⇒ **STOP, DECISIONS D20 seeded OPEN**. 4-member panel (the
  task's own mandated fail-closed lens + correctness/regression/flow-verifier) caught a real
  **BLOCKER**: the fix's active-mask filter reactivated a dormant duplicate-cycling bug in
  `create_daily_panel`, reproduced on TASK_852's own scheduled command (28/64 dropped → would
  have double-exposed 28 symbols) — 3 of 4 panel members found it independently; fixed to fail
  loud instead. All numbers independently re-verified by the orchestrator against the real DB._

_Shipped 2026-07-18 factory run 12 (both D17 instrument builds):_
- _**TASK_847** (`task847-policy-prior-proposals`, `873effe`) — `policy-prior` scheme; canon pin bit-identical, selftest 8/8, mechanism leg 40/40. **Real-panel numeric leg FAILED post-ship (−7.839 vs ≥+0.74, LEDGER 832/E5-inversion) — the fault moved to leaf-eval selection (⇒ 849); the scheme stays shipped/inert (default unchanged).**_
- _**TASK_848** (`task848-policy-replay-instrument`, `371e7c5`) — policy-argmax operator + OOS window + on-window teacher floor; 1 BLOCKER (date-coercion in-sample leak) + 2 MAJOR fixed. **Its fail-loud guard immediately caught the STALE SERVER DB (daily ends ~2025-10-04 vs local 2026-07-06) on the first OOS attempt — the OOS read moved to the workstation.**_

_Shipped 2026-07-17 factory run 11:_
- _**TASK_846** (`task846-weight-fee-headroom`, `c2f1ba5`) — weight-budget fee headroom (D14 opt (a)):
  `budget × (1 − 2·fee_rate)`, closes the all-short fee16 full-wipe; fee0 BIT-IDENTICAL (3 pins MATCH
  incl. `6293668f…`); 5 D14 regression tests, selftest 7/7. Acceptance (iv) went through **D16**
  (fixture teacher-gate Δ1.576 → held → planner's binding 64-asset real-panel read Δ=0.0008 ≪ 0.02 →
  RULED ship-as-is; (a′) rejected unsound). Un-gated the ExIt continuation + all phase-2 fee16._

_Shipped 2026-07-16 factory run 10 (843 → 845 in sequence, the D12-cleanup → E5-compat chain):_
- _**TASK_843** (`task843-weight-incremental-netting`, `ed27849`) — weight-path incremental netting + snap-to-flat
  (D12): delta-only re-trim (entry preserved) + snap-to-flat kills the short over-tax (185.2→≈canon) and
  slippage dust. Canon-OFF `48e1f467…` + budget `4f8b692e…` pins UNCHANGED; weight-ON/fee0 RE-BASED
  `1c50d58c…`→`6293668f…`. Escalated 4-member panel: **1 BLOCKER (winning-short survivor under-reservation
  → all-short margin wipe at fee0 from a valid state; orchestrator-crossval-CONFIRMED vs pre-edit) FIXED**
  (footprint-fraction reservation `(2E−P)/P` clamped ≥1; re-run fail-closed judge confirmed the fee0 hole
  closed, 3,840-case fuzz + sweep 0 hits). Acceptance (i)–(iv) PASS; E1 fee16 edge +2.1667 (from 842's −0.134).
  **Remainder → DECISIONS D14** (PRE-EXISTING fee16 all-short overdraft wipe, stash-confirmed 842-identical)._
- _**TASK_845** (`task845-imprint-weight-compat`, `4ac7721`) — 832 imprint runner threaded through the weight
  env + teacher-expressibility gate; ZERO canon diffs. Medium panel + fresh fix-judge: **1 MAJOR (weight-native
  remediation mapping was DIAGNOSTIC-ONLY — runner always imprinted the order-code teacher) FIXED** via
  `--imprint-mapping {order-code,weight-native}` (default order-code = byte-identical) + **2 MINOR guards**
  (fail-closed flag-coupling assert; None-guard the fee16 bankrupt-teacher WARN crash). Expressibility gate
  REPORTED, fixture PASS/FAIL INCONCLUSIVE (losing slice) → **DECISIONS D15** (structural sizing divergence
  under default w_unit/w_max; binding read = real-panel run per DEC-2). Un-blocks the scheduled E5 runs._

_Shipped 2026-07-14 factory run 9 (solo, read-only diagnostic):_
- _**TASK_844** (`task844-short-bias-probe`, `b53ecfe`) — short-bias postmortem instrument (mirror-book
  value / flat-book marginals / leg decomposition + fixture smoke), zero canon diffs, D13-independent.
  Medium panel: regression clean; **2 MAJOR FIXED** — probe-1 mirror-Δ was uncalibrated (independently-
  init BUY/SELL embedding columns ⇒ a FRESH model trips |t|~4; now gated on a matched fresh-init null
  z-stat, smoke robust across seeds 0/7/8 z +0.10/−0.85/−1.12) + probe-3 mislabeled an all-close SUPERSET
  as the EXACT reward_attributable term (relabeled + action-attributed subset now gates credit_misassign).
  MINOR (smoke tol false-alarm) fixed; NIT (open_count counts limit-fill opens) → planner. Real numbers =
  the SCHEDULED planner run (p1c s0/s1(+s2) + p1b s1 control, weight-env flags)._

_Shipped 2026-07-12 factory run 8 (solo, canon-adjacent fee fix):_
- _**TASK_842** (`task842-weight-close-fee`, `30013a4`) — canon-equivalent explicit close/open-short fee
  on the TASK_840 weight re-trim (was undercharging ~16bps/close, D9). Every term ×fee_rate ⇒ **fee0
  bit-identical** (THIRD pin weight-ON/fee0 `1c50d58c…` captured pre-edit at ba53294; canon `48e1f467…`
  + budget `4f8b692e…` also MATCH) ⇒ **RUNNING phase-1c cost0 untouched**. Medium panel: correctness+
  regression session-limit-truncated (trending clean, orchestrator-substituted); flow-verifier re-run
  CONFIRMED **no undercharge** (weight ≥ canon per re-trim — the short RT is an OVER-charge from 840
  close+reopen/residual geometry, fee-independent, → **D12**) + canon-faithful arithmetic. NIT: within-1e-6
  is float32-summation fragile (float64-exact). **Un-gates phase-2 fee16.** 840 (iii) E1 fee16 edge honestly
  moved +0.235→−0.134 (the +0.235 was flattered by the fixed undercharge)._

_Shipped 2026-07-11 factory run 7 (840 + 841 parallel lanes, disjoint surfaces):_
- _**TASK_840** (`task840-weight-rebalance-step`, `31a545f`) — target-weight rebalance step (H6 env,
  D8 opt (c)): flag `--vector-weight-rebalance` default OFF (canon + 839-budget both bit-identical,
  TWO pins `48e1f467…`/`4f8b692e…`). Full high-tier panel: **1 BLOCKER (held-winning-short budget
  under-reservation, crossval-CONFIRMED) + 1 flow-verifier-CONFIRMED held-leg drift bleed → BOTH FIXED**
  (whole-book posval/PV reservation + cap-clip re-trims every drifted leg → cash≥0, Σ|w|≤1, margin no-op).
  Acceptance (i) survives full 1697-step walk (invariant gate cash≥0/Σ|w|≤1) / (ii) sticky-short PASS;
  **(iii) E1 edge +0.235 (flipped from −0.23, below +0.436) = FINDING, non-gating**. Residuals → **D9**
  (fee undercharge, gates fee16 phase-2) + **D10** (expressibility). Unblocks phase-1c re-train._
- _**TASK_841** (`task841-checkpoint-wedge`, `52b155f`) — `run_queue.sh --stall-timeout` wedge watchdog,
  **default OFF (opt-in)** per invariant #1 (unarmed = byte-identical to HEAD; CSV-declaring jobs only when
  armed). Medium panel: **1 MAJOR (default-armed false-kills healthy CSV-less/`--name=` jobs at ~55min)
  double-CONFIRMED → FIXED** + 3 MINOR. Root cause characterized (orbax post-save GC vs RLock), STOPPED at
  out-of-scope trainer surgery; `--debug-stall-dump` wired for the next real wedge. Default → **D11**._

_Shipped 2026-07-09 factory run 6 (solo, canon-adjacent env fix):_
- _**TASK_839** (`task839-budget-vector-step`) — the env-wall fix: flag-gated (`--vector-budget-step`,
  default OFF = canon bit-identical, pin `48e1f467…` MATCH pre-edit 39ae0bf) budget-constrained
  equal-split sizing in `step_vector` (RATIFIED D7 opt (a)). Full high-tier panel: **1 MAJOR
  crossval-CONFIRMED → RESOLVED** ("≤cash by construction" holds only at PV≈cash → honest wording +
  PV>cash test + **DECISIONS D8** seeded; ratified formula untouched) + security MINOR (symlink
  write-guard realpath) & fail-closed MINOR (probe exit code) FIXED. Acceptance: OFF bankrupt step 0 /
  **ON 4/4 survive ≥64 steps zero-overdraft**; E1 byte-identical (narrow-regime cap non-binding).
  **Unblocks the line**: phase-1 RE-RUN under the fixed step → phase-2 fee16 + 832 imprint-sanity +
  834/836 re-runs. (NB: phase-2 gate row below still lists TASK_839 as a gate — now SATISFIED; D8 is
  the residual planner call on PV-relative vs cash-relative before flipping the flag ON.)_

_Shipped 2026-07-09 factory run 5 (836+837 parallel lanes, then 838 canon-adjacent solo):_
- _**TASK_836** (`task836-e4-rank-probe`) — E4 value-vs-momentum rank probe; 0 MAJOR; 3 MINORs fixed
  (honest baseline labels + true walk-PV mirror, overlap p-caveat, degenerate-value reading)._
- _**TASK_837** (`task837-walkforward`) — R4 walk-forward rails; `build_pit_universe` verified NON-PIT
  → module-side PIT; 3 MAJOR gate-coverage roots fixed (POLICY-PIN to pre-registered constants,
  eval-file identity, de-confounded 5-negative battery + no-op control)._
- _**TASK_838** (`task838-pernode-resample`) — flag-gated per-node resampling, OFF bit-identical
  (pin 22b0c7f5 MATCH, provenance independently reproduced); 1 MAJOR fixed (835 threads the flag —
  train/eval operator symmetry); **acceptance FINDING: OFF > ON under an uninformed policy
  (Δnet@21 −2.79) — any ON run must re-measure with a trained policy first; default stays OFF**._

_Shipped 2026-07-08 factory run 4 (parallel-lane batch, all zero-canon-diff instrument modules,
each medium-panel verified with adversarial crossval + fix re-verify):_
- _**TASK_834** (`task834-e3-runner`) — checkpoint E3/DEC-1 runner; 2 MAJORs found+fixed (legality
  mask on both proposal legs; fail-loud vs flax silent fresh-init restore — a reproduced FALSE PASS)._
- _**TASK_835** (`task835-stage4-eval`) — faithful Stage-4 vector eval; 0 MAJOR; 3 MINORs fixed
  (LEDGER pointer, partial-rollout guard, provenance)._
- _**TASK_832** (`task832-linear-imprint`) — E5 buffer-injection BC runner; 1 MAJOR found+fixed
  (imprint entropy_coef leaking into the ExIt half via checkpoint metadata)._

_Shipped 2026-07-08 factory run 3: **TASK_828 Stage 3** (`task828-stage3` — vector-action training
loop + D2 action-conditional reward head + factored-marginal targets; parity MATCH==PINNED, E3 probe
green, full high-tier panel 0 BLOCKER/MAJOR + 1 MINOR fixed). Stage-3 TRAINING itself is the scheduled
gpu-server GPU run below, not a factory-loop job._
_Shipped 2026-07-08 factory run 2: TASK_833 (`task833-run-queue-repair` — repaired scheduler
redeployed to srv same day) and TASK_829 (Foxglove retired, `40afa14`)._

## Scheduled runs — owner: planner/main session (NOT factory-loop jobs)

**Server status (updated 2026-07-09)**: gpu-server db verified at FULL workstation parity —
daily tables synced 2026-07-08, and the minute `ohlcv` table measured at max ts 1765558500
(2025-12-12, == workstation; the "Jan-2026 vintage" concern was file-mtime, not data). All
daily AND minute jobs cleared. Server code current; the 7 drifted server files remain STASHED
(`git stash list`).
Stage-3 training destination = gpu-server GPUs 1/2/3 — run_queue.sh REPAIRED (TASK_833) and
redeployed 2026-07-08; the scheduler is cleared for use.

| Run | What | Where | Status |
|-----|------|-------|--------|
| 830 real corridor | full-panel floors+ceiling+expressibility gap, `trading_data.db` read-only | workstation CPU (screen+safe_run) | **DONE 2026-07-08** — corridor official [+0.424, +29.698] net@21, ref +0.751; H6 armed; LEDGER 830/E0-REAL |
| 831 real E1/E2 sweep | 9 scheme×K combos, full panel, seeds 0,1,2 | workstation CPU (screen+safe_run, ~7 min) | **DONE 2026-07-08** — E1 PASS 9/9 ⇒ Stage-3 training gate OPEN; carried scheme dithered-book\|K16; LEDGER 831/E1-REAL + E2-REAL |
| 828 Stage-3 training **phase 1** | cost0 ×3 seeds from scratch, dithered-book\|K16 **hb0.0/d1.0** (3-arm diagnostic winner; fee16-from-scratch REJECTED — collapse finding, LEDGER 828s3/preflight), snapshots every 100; `jobs_phase1.manifest` | gpu-server GPUs 1/2/3 via run_queue.sh (tmux `q_828s3`) | **DONE 3/3 2026-07-10 05:18** — s1/s2 ep2000 first pass; s0 wedged ep1581 (log-freeze post-checkpoint-save) → killed, archived `*_wedged_ep1581`, solo retry completed clean (rc 0, 20 snapshots; wedge did NOT reproduce at ep1581 — non-deterministic, not data-positional). Ops note: fresh tmux needs `source trader_env/bin/activate` (bare `python` = system Py2) |
| 828 phase-1 **instrument passes** | 835 eval + 834 E3/DEC-1 + 836 E4 on s1/s2 `snapshot_ep2000`, phase-1 flags EXPLICIT (`--vector-hold-bias 0.0` — 835's CLI default is 3.0!), fee0/slip0 run configs `runs/828s3p1_instruments/cfg_*.json` (server) | gpu-server GPUs 2/3 via run_queue.sh (tmux `q_828s3_inst`) | **DONE 2026-07-09 20:06** — 835 FAIL both seeds (bankrupt, below floor; LEDGER 835/S4-p1); 834 fail-loud walk-bankrupt (834/E3-p1); 836 n=0 (836/E4-p1). Root = **828s3/env-wall**. 834/836 re-run AFTER phase-1b. s0's 835 DONE 2026-07-10: same FAIL (bankrupt 501/676, hold 98.2) — verdict 3/3, LEDGER row folded |
| 828 Stage-3 training **phase 1b** | RE-RUN under the fixed step: phase-1 command + `--vector-budget-step` (D7 opt (a), **D8 ratified ship-and-measure**), cost0 ×3 seeds, `jobs_phase1b.manifest`; pre-registration = SPEC amendment 3 | gpu-server GPUs 1/2/3 via run_queue.sh | **s1/s2 DONE 2026-07-10 ~16:15** (ep2000, 20 snapshots, 9.7h). **s0 WEDGED at ep162** — SAME signature as phase-1's wedge: frozen immediately after "Finished synchronous save" (orbax post-checkpoint deadlock, now 2/6 seed-runs — pattern, candidate factory row) → killed 2026-07-11 21:08, archived `*_wedged_ep162`, solo retry RUNNING (tmux `q_828s3p1b_s0`, GPU 1) |
| 828 phase-1b **instrument passes** | 835 (D8 primary evidence) + 834 DEC-1 + 836 E4 on s1/s2 `snapshot_ep2000`, phase-1b flags (budget ON) | gpu-server GPUs 2/3 | **DONE 2026-07-11 21:35, 6/6 rc0 — the decisive mixed verdict**: 835 still bankrupt (step 132/181, s1 book = 10–11 shorts → D8 escalation FIRED ⇒ TASK_840); **834 DEC-1 PASS ×2** (R 1.30/1.12 vs bars 0.33/0.24 — D2 rebuild works); **836 value>momentum ×2** (ρ +0.21/+0.30 vs −0.03 — first positive learning signal). LEDGER 835/S4-p1b · 834/E3-p1b · 836/E4-p1b |
| 828 Stage-3 **phase 1c** | re-train under the TASK_840 weight env: phase-1 recipe + `--vector-weight-rebalance` (NO budget flag — one variable), cost0 ×3 seeds, `jobs_phase1c.manifest`; pre-registration = SPEC amendment 4; **D10 ratified PROCEED**; watchdog ARMED per D11 (`--stall-timeout 2700`) | gpu-server via run_queue (tmux `q_828s3p1c`, `--gpus 1,2,3 --acquire-timeout 0`) | **s0/s1 DONE 2026-07-12 07:32 + instruments graded same morning (LEDGER 835/S4-p1c · 834/E3-p1c · 836/E4-p1c): ENV FIXED (zero bankruptcies, first-ever complete rollouts) / POLICY BELOW FLOOR 2/2 (short-biased book) ⇒ D13 OPEN**; s2 DONE 14:16 + record pass graded (−1.385, complete rollout, 7L/8.4S — verdict 3/3). **Battery closed CLEAN: zero failures, zero wedges, watchdog armed throughout.** Earlier: s2 start was delayed by the **p1b-s0 retry wedge #3 at ep982** → py-spy stack CAPTURED (841 root-cause: XLA/CUDA dispatch hang post-save, orbax-lock hypothesis REFUTED — see TASK_841 evidence note + `lore/notes_2026-07-12.md`), killed, NO re-retry (p1b verdict stands on s1/s2). Graded read: corridor vs floor +0.424 on ≥2/3 seeds; any bankruptcy = TASK_840 defect |
| 844 **short-bias postmortem** | `bias_probe.py` on p1c s0/s1/s2 + p1b s1 control (`jobs_844.manifest`, weight/order run flags from `cfg_*.json`) | gpu-server GPUs 2/3 via run_queue.sh (tmux `q_844`) | **DONE 2026-07-14 20:40, 4/4 rc0 — NO LOCUS TRIPPED (LEDGER 844/postmortem)**: value clean (calibrated \|z\|≤0.9, the raw-t would have false-tripped ×4), prior BUY-leaning (0.65–0.84), credit-misassignment unsupported. Planner post-hoc: probe leg tracker is FLIP-BLIND in weight semantics — shorts exit ONLY by BUY-flip to long (exact arithmetic 3/3); longs culled by non-action re-trim closes (margin ruled out — fee0 no-op invariant). Folded into D13 |
| 832 **imprint-sanity** (E5 leg 1) | teacher replay floor (weight env, real panel — the binding ±0.10-absolute expressibility read) + BC imprint + deterministic sanity replay vs teacher ±0.10 (DEC-2 tolerance); cost0, weight flags = phase-1c set; run BOTH mappings via `--imprint-mapping {order-code,weight-native}` (845); ≤1 GPU-day | gpu-server single verified-idle GPU, run_queue + D11 watchdog | **DONE 2026-07-17 00:06 — BOTH LEGS (LEDGER 832/E5-gate + 832/E5-imprint).** Gate-read: order-code FAIL (rel 0.290) / weight-native PASS (rel 0.041) ⇒ D15 branch 2, no retune. Full imprint (weight-native, iters 2000): BC converged (CE ÷100, agreement 97.5%), sanity replay net@21 **+0.839 vs floor +0.542** — ABOVE floor both tiers, zero bankruptcies, first learned policy above floor in the program. Strict ±0.10 window missed on the UPSIDE (+0.297) ⇒ ruled PASS-with-caveat (planner; H5 kill clause targets BC failure, which did not occur). Ckpt `runs/832_e5/ckpt_imprint_full` (srv) ready for the ExIt restore |
| 832 **ExIt** (E5 leg 2, = the phase-2 continuation under redirect) | short ExIt from the imprint ckpt (`--load-checkpoint --reset-optimizer --clear-buffer`; verify `Restored entropy_coef: 0.01` at launch — SPEC PIN c), ≥3 seeds, fee16 per DEC-2; PASS = floor+0.10 net@21 ≥3 seeds (per-seed above bare floor) + ≥3/4 regime windows at/above their window floor | gpu-server, run_queue + D11 watchdog | **DONE — FAIL 2/2 (LEDGER 832/E5-exit, graded 2026-07-18).** s1 −1.223 / s2 −0.791 net@21 vs bar +0.524 from a +0.839 warm start — self-play eroded the prior into the from-scratch attractor (99% hold, sparse book); s0 wedge-failed ×2 (watchdog fail-loud), moot. Zero bankruptcies at fee16 (846 validated in production). H1 improvement operator implicated; H6 NOT triggered (imprint holds the floor). Pivot = **D17** |
| **882 decisive — Leg A (identity check)** | eval-only, 3 seeds: `ckpt_ref_s{0,1,2}` graded over the CONTINUOUS 2021-01-01..2025-10-04 — the same window their `imprint.py` sanity replay used — CORRECTLY PINNED to `frame_provenance_ref_s{0,1,2}.json`, `--operator policy-argmax --vector-weight-rebalance --teacher-floor`. Manifest `experiments/882_instrument_reconciliation/manifest_882_decisive.txt`. **Bands PRE-REGISTERED before launch** (TASK_882 §6): |Δ| vs `+0.9914` ≤ 0.1 ⇒ the instruments AGREE once pinned · ≤ 0.5 ⇒ named residual · > 0.5 or a SIGN FLIP ⇒ a second, pin-independent defect and the corroboration is WITHDRAWN. Framework-epoch diff run BEFORE launch per the 879 doctrine and recorded in the manifest header: srv redeployed b695487 → `60f77f7`; the eval-path delta is TASK_870's `_ro_connect` move plus a `--smoke`-only `overwrite=True`, neither semantic; `--vector-weight-rebalance` and `--panel-frame-from-checkpoint` PRESENT on every line. | gpu-server GPUs 1/2/3, tmux `d882` | **DONE 2026-08-08 — PASS, TOP BAND.** LEDGER `882/decisive-legA`. The instruments AGREE to **Δ 0.000214** net@21 (+0.9915919 pinned eval_vector vs +0.9913783 imprint.py), PV within 0.060%, residual localised to exactly 3 of 107,072 decision cells (float32 argmax tie-breaks). 3/3 seeds +0.99159 / +0.99355 / +0.95212, 1673/1673 decisions, no bankruptcy, 64/64 slots active. **`+0.9914` RESTORED, instrument-independent and 3-seed.** Does NOT close `861/ref-oos-rebase` — the OLD `ckpt_imprint_full` anomaly is an EPOCH question, untested here. |
| **884 rung 1 — step-0 smoke** | 40 epochs, seed 0, 1 GPU, phase-1c recipe under the FIXED env, window truncated to 2021-01-01..2025-10-04 so the held-out span matches the imprint lane's OOS window. Pre-registered `experiments/884_rung1/SPEC.md` (pre-launch sha256 `2b265701…`); universe preflighted on the actual periods file (`preflight_frame.py`: 64/64 order-exact, pad does not fire, 0 zero-price cells). | gpu-server GPU 1, tmux `r884smoke` | **DONE 2026-08-08 — S2 FAILED, BATTERY NOT LAUNCHED.** LEDGER `884/rung1-smoke`. `pct_hold` 82.3% → 95.6% by **epoch 3** → 99.0% plateau; `policy_entropy` 0.691 → 0.070. **14.17 s/epoch measured** ⇒ the battery would have been ~7.9 h × 3 seeds; the gate cost 9 GPU-min and saved ~24 GPU-h. **No bankruptcy anywhere** (min `agent_return` −22.6%) — E-a fixed the cash cascade and the hold collapse happens anyway, so they were separate defects. `manifest_884_battery.txt` staged, NOT launched. |
| **885 hold-collapse screen** | 3 arms × 40 epochs, seed 0: **B** `--sims 8` · **C** `--sims 128` · **D** `--sims 32 --epsilon-greedy 0.25` (the TASK_812 target-mix that severs MCTS-sharpening → policy-sharpening). Arm A = the completed 884 smoke. Tests whether the collapse is SEARCH-driven — the training-lane twin of `879/rung3`'s subtractive-search control. **Bands pre-registered** in `experiments/885_collapse_screen/SPEC.md` (sha256 `fc2edd4d…`): (P1) `pct_hold` ordered B<A<C with ≥2.0pp B-to-C spread AND `mcts_visit_entropy` ordered B>A>C · (P2) arm D reaches `pct_hold` < 95% · (NULL) all arms within 1.0pp. DIAGNOSTIC SCREEN, 1 seed/arm — the winner earns a 3-seed battery, nothing promotes off this. | gpu-server GPUs 1/2/3, tmux `r885` | **DONE 2026-08-08 — P1 FAILS, P2 PASSES.** LEDGER `885/hold-collapse-screen`. `pct_hold@40`: A 99.016 · B 98.854 · C 98.987 · **D 90.971**. A 16× sims range moves hold **0.13pp** and is non-monotone ⇒ **search depth is irrelevant**; the epsilon target-mix holds it at a plateau (82.3 → 92.2 by ep5, then flat to ep40) and keeps the RAW pre-mix `mcts_visit_entropy` at its epoch-1 value (0.1554 vs A's 0.0246, 6.3×). **The collapse is the distillation loop, not the search depth.** 1 seed/arm, 40 epochs — a screen; the winner owes a 3-seed battery. |
| ~~886 rung-1 re-spec~~ — **STRUCK 2026-08-16 by 887/collapse-confirm** | 887's C3 finding rules this out in its own words: a rung-1 re-spec around epsilon mixing "trains a policy to trade more and lose more" — at zero costs, activity is monotone in loss (A −2.86% · E −9.11% · D −13.84%). The collapse was ADAPTIVE: the policy correctly refuses negative-EV actions. **No training battery launches until the action-EV question is answered.** The replacement scheduled run is the **892 action-EV audit real leg** (eval-only, ~84 lightweight evals on existing checkpoints, workstation; EDGE/ANTI-EDGE/NULL bands pre-registered in `experiments/892_action_ev/SPEC.md`), which gates any new structural hypothesis (D42). `manifest_884_battery.txt` stays unlaunched regardless. | workstation (planner-scheduled) | **REPLACED BY TASK_892** — spec landed 2026-08-16; real leg awaits the factory build + operator go |

## Gated — do NOT start (named gate must open first)

| Task | Gate |
|------|------|
| `tasks/TASK_880_CKPT_ARCHIVE_DISCOVERY_FIX.md` | **GATE REWRITTEN BY D35 (2026-08-08): TASK_881's path-first reader must be SHIPPED, and a deployed run must have written `scores.jsonl` that THAT READER RESOLVES.** D34's original gate — 876 shipped AND deployed AND real score files on the host — is now known to be necessary but NOT sufficient, and half of it is already literally met (deployed at `60f77f7`, 2026-08-08) while the signal stays unreadable for `checkpoint_<step>` and `checkpoint_epoch_<N>`. `--unscored archive` (pull, never delete) remains authorized under D34 and is unaffected. The `--apply` freeze on `tools/ckpt_archive.py` STANDS until both halves hold. _(D34 gate text, superseded, kept verbatim for the record:)_ **TASK_876 must be SHIPPED *and DEPLOYED to gpu-server*, with a run having written real `scores.jsonl` files there** (D34). Not "876 merged" — the host must actually carry the artifact, because 100% of this tool's score signal is otherwise inference over `results*.json`. Run 27's real-host dry-run measured `ARCHIVE 1 ckpt / 0.09 GB · DROP-WITHOUT-ARCHIVE 27 ckpts / 2.51 GB` — one operator command from deleting 27 of 28 checkpoints including `runs/861_ref/ckpt_ref_s0/checkpoint_2000` at net@21 **0.9914**, the clean-recipe anchor. The `--apply` freeze STANDS; `--unscored archive` (pull, never delete) is the only authorized mode meanwhile. |
| `tasks/TASK_828_VECTOR_ACTION_MUZERO.md` (Stage 4 eval) | Stages 1–3 shipped ✓ (2026-07-08) + Stage-3 training run's rung verdicts green; SCHEDULED server run, not a factory-loop job |

_TASK_832 (E5 linear imprint) moved to Ready 2026-07-08 — its gate (828 Stage 3 banner) is now satisfied._

## Parked (D1/D4, 2026-07-07 — non-binding "interesting outview"; not factory work)

- `tasks/TASK_824_SERVER_HONEST_REMEASURE.md`, `tasks/TASK_825_CROSSASSET_TREND_CTA.md`,
  `tasks/TASK_826_ONCHAIN_SIGNAL_PROBE.md` (D4: OHLC-only, live within the 24GB DB),
  `tasks/TASK_823_VPS_MICROLIVE_CTA.md` (D0: rejected).

## Grooming rules
Planner re-ranks after each factory run + banners; a row leaves this file only by shipping
(banner) or being explicitly parked. If the factory finds the top row blocked in practice, it
takes the next and records why in its wrap-up.


## Recently shipped (full rows in QUEUE_ARCHIVE.md)
- **TASK_878** — failed-fill semantics: no-print bar = attempted order, data-derived seeded coinflip, intent preserved, PV exact; OFF-parity 248/248 · 266/266 · 953/953 (`27ddef5`, run 35, 3 cycles). Remainder: the 3-arm ablation real leg (authorized, planner-scheduled).
- **TASK_895** — item 1 (the disproven docstring) shipped (`bfa8f97`, run 35, 1 cycle); item 2 escalated to D44 → RULED document-and-leave → TASK_896.
- **TASK_895** — item 1 only: the `_ffill_bfill_cols` docstring stops claiming "no cross-time leak" (`bfa8f97`, run 35, 1 cycle). **Item 2 (the 819 rider) NOT shipped — gate (iii) is internally unsatisfiable → `D44` OPEN.** Corrects the D41 record: 893's Δ=0 held because the fill is VALUE-NEUTRAL, not because row 0 is unreached — row 0 IS read by every lookback spanning it.
- **TASK_886** — universe-pin certification contract; `verified: true` reachable + unforgeable, honest PROVEN/NOT-PROVEN split (`a0da06b`, run 34, 3 cycles).
- **TASK_890** — distill CLI repaired + 868 open gate applied; exposed D43 (`2804eb7`, run 34, 2 cycles).
- **TASK_893** — D41 measurement: prefixes are 7.38% of FULL, masking moves the bar 0.000000, docstring claim disproven (`c1e556d`, run 34, 2 cycles).
- **TASK_888R** — one checkpoint resolver, a loader that refuses, a row that cannot write (`7cc50fb`, run 33, 2 cycles).
- **TASK_889** — save-time architecture records + refusing loader + per-leaf evidence (`f9d8ee4`, run 33, 3 cycles).

- TASK_876_SCORE_BASED_CHECKPOINT_RETENTION — 2026-08-08 `task876-score-emission`
- TASK_882_INSTRUMENT_RECONCILIATION — 2026-08-08 `task882-instrument-reconciliation`
- TASK_873_FIXTURE_IDENTITY_PIN — 2026-08-08 `task873-fixture-identity-pin`
- TASK_874_IMPRINT_EMITS_FRAME_PROVENANCE — 2026-08-08 `task874-imprint-frame-provenance`
- TASK_883_EVAL_VECTOR_UNIVERSE_IDENTITY_FAIL_LOUD — 2026-08-09 `task883r-universe-identity-fail-loud`
- TASK_877_XBT_ALIAS_FOLD — 2026-08-11 `task877-xbt-alias-fold`
- TASK_865_MINUTE_FRAME_PIN — shipped
- TASK_866_WEIGHT_ON_PIN — shipped
- TASK_867_861_BASELINE_PROMOTE_PORT — shipped
- ? — shipped
- TASK_843_WEIGHT_INCREMENTAL_NETTING — shipped
