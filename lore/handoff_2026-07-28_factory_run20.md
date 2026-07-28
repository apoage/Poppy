# Dark-factory run 20 — handoff (2026-07-28)

Implementer lane. Queue on entry: Ready = TASK_861 (M) + TASK_862 (S, filler), factory GREEN,
no OPEN decisions. Both rows shipped. Two new OPEN decisions seeded (**D27**, **D28**), and D28
**blocks the 861 real legs** — do not launch them as pre-registered.

## Shipped

| Task | What | Verification |
|---|---|---|
| **TASK_861** | `--collect-chosen-vectors` (eval_vector) + `--teacher-vectors` (imprint): the +8.024 search as an OFFLINE teacher for the proven 832 BC recipe. Both flags OFF bit-identical. | 3-member panel (correctness · regression · flow), **2 fix rounds**, 5 MAJOR closed |
| **TASK_862** | `scorecard.py` + canonical 7-window manifest + smoke: the D17 imprint battery as one command and one fail-loud artifact. New files only. | 2-member panel (correctness/fail-closed · flow), **2 fix rounds**, 3 MAJOR + 1 latent-MAJOR closed |

## What the panels actually caught (the part worth reading)

Both tasks arrived plausible and both were wrong in ways the implementers could not see.

**861 — every gate was green on garbage, three separate ways.**
1. The pinned +8.024 operator was never checked. Three verifiers independently collected under
   wrong operators (−7.08, −19.89, +0.81) and imprinted clean. Worse, the falsification metric
   scored the *garbage* teacher **higher** (+0.431) than the real one (+0.36) — it only measures
   whether BC memorised whatever labels it was handed, so it is structurally incapable of
   noticing a wrong teacher.
2. No payload integrity. Every action code written as `x−0.5` truncated toward zero and swapped
   a +13.893 teacher for −0.649, with two green gates and `SMOKE PASS`.
3. **The subtlest one, found by brute force in round 2**: the new replay-integrity guard is a
   *trajectory* check, not a *target* check. Under weight-rebalance the codes
   {Hold, LimitSell, LimitBuy} are mutually trajectory-inert at most cells — **315 of 400 fixture
   cells (78.8%) admit an inert substitution**. Rewriting all of them jointly left the PV path,
   `realized_book` and net@21 **bit-identical** while replacing 78.8% of the BC targets. Closed
   by cross-checking the payload's own action composition against the `walk_pct_hold_agg` the
   walk already recorded and nothing was reading (76.75% → 11.50%).

**862 — the honesty machinery had holes in exactly the places it advertised.**
1. Any exception that was not `WindowFailure` destroyed the entire artifact. A floor artifact
   with a non-numeric value — no stub, no injection — killed a completed battery with no json
   and no md, discarding every finished window. The module docstring said "a window is NEVER
   silently skipped".
2. The operator block was asserted, never observed: the three checked fields were echoes of the
   harness's own argv, while ~25 real knobs went unpinned and unrecorded. A `top-m` battery would
   have emitted a status-OK card headed "848 read of record".
3. The recovery path re-ran the same call that had failed (`_sha256(manifest_path)`), so the
   last-resort still lost everything; and the artifact write sat outside every guard, so a
   serialization failure left truncated unparseable JSON — worse than nothing.

## Two things that bite the NEXT run

**D28 — the 861 real legs are blocked. This is the important one.**
Neither pre-registered leg pins the TASK_853 positional frame, and `imprint.py` cannot (it only
WARNs). Both derive the universe from `ORDER BY COUNT(*) DESC LIMIT 64` over their window.
Measured read-only today: vs `runs/832_e5/frame_provenance_imprint.json` the leg-1 window's
top-64 differs in **59 of 64 positional slots** and in membership; the frame no longer reproduces
even from its own recorded `train_window` — the 858 backfill moved the coverage ranking under it.

Why every in-band check misses it: the teacher walk is genuinely unaffected (the pinned operator
is `prior_weight 0.0` + `dithered-book` + true-env leaf + no v̂ tail ⇒ `net_dependent: False`, so
the checkpoint is never read), and legs 1↔2 derive the *same* universe as each other, so the
provenance gate passes legitimately. The damage is to the **grading**: leg 3's ">+0.839" would
compare a net trained on universe A against a number measured on universe B, confounding "better
teacher" — the only question this task asks — with "different assets". Leg 4 also cannot reuse
the old frame provenance. D28 recommends teaching `imprint.py` the shipped 853 flag and pinning
both legs; that also keeps the whole 858-OOS + walk-forward battery reusable.

**D27 — the 862 battery's headline consistency read is cross-configuration.**
TASK_862 scope 1 mandates the `runs/831_ext` per-window floors. Those were measured at n_assets
95/204/214/273, `position_size_pct` 0.0018–0.0053, warm 31, server db, 10-seed dithered-book
search. The battery grades 64 slots (32 active), 0.1, warm 60, local db, 1-seed policy-argmax.
The factory shipped the *disclosure* and promoted the on-window teacher floor to PRIMARY — which
is the sounder basis but deviates from scope 1 and from the ratified DEC-2 row, so it needs a
ruling rather than a filler task's judgement.

## Scope limits recorded, deliberately NOT closed

- **861 acceptance (iv) certifies the fixture pipeline only.** The falsification gate is an
  entropy ratio calibrated on a 76.75%-HOLD fixture; the real teacher is 94.2% HOLD (H≈0.30)
  where both arms compress toward zero, and `smoke_assert` defaults to `is_fixture` so **the gate
  never fires on the real panel**. Obs-derivable degenerates reach +0.106/+0.193 on that metric,
  so it falsifies target↔observation misalignment, not "learned the teacher".
- **The digest+replay pair is accident-detection, not tamper-detection.** `payload_sha256` covers
  `decisions` only; the header the operator pin judges is unprotected, and the digest function
  ships in the repo. A consistent forgery passes. Fine for the threat model (drift, corruption,
  wrong file), stated so nobody over-reads it.
- **Sanity-replay seed noise spans −8.93 … +12.74 across 10 seeds** against a pinned +13.89
  teacher floor — an order of magnitude wider than leg 3's own bar. Three seeds cannot decide it.
- **861's `floor_inenv`/`corridor_position` now reference the SEARCH teacher** (the 819-book walks
  are skipped on this path), so "above floor both tiers" is unreachable by construction; leg 3's
  wording was corrected and a separate corridor referent named.
- **862**: `MEM_FRACTION` is not a cap (eval_vector sets `PREALLOCATE=false`) — a GPU battery
  still needs `safe_run`; `max_positions`/`min_cash_threshold` are argv-pinned only because
  eval_vector does not report them.

## Pre-existing, handed over (not caused by this run)

`experiments/832_linear_imprint/results_fixture_smoke.json` is a **tracked parity baseline that
the module's own default fixture invocation rewrites in place**, and it is stale by ~74 commits —
drift attributable to `c2f1ba5` (TASK_846 fee headroom: gross byte-identical, net moved). 861
strictly reduced the exposure (suffixed teacher-vector outputs are gitignored, the blank-path
clobber is closed, the harness byte-checks the baseline) but the canon path still self-clobbers.
Risk is a future agent running the bare smoke, seeing the file dirty, and committing the
"refresh" without realising the values moved for a reason. Refresh it in a commit that *names*
TASK_846, or move the canon artifact out of the default write path.

## Orchestrator-run objective checks (not delegated)

- `python -m trader selftest` → **12/12 PASS** (re-run after every fix round)
- Flag-OFF bit-identity independently reproduced against `git show HEAD:` copies staged in-place:
  `eval_vector --smoke` **0 differing leaves**; `imprint.py` default **0 differing leaves**, no
  `teacher_vectors` key
- Fixture db md5 unchanged across every run; `trading_data.db` read-only (mode=ro) throughout
- All 25 CLI flags in the pre-registered real-leg argv verified to exist
- 861 composition fix proven end-to-end against the flow-verifier's own `attack_A_inert.json`
  (refused, both problems reported, no artifact); 862 smoke PASS after each fix round

## State

Clean tree, two tags, nothing pushed. Queue Ready is empty; both remainders are planner-owned and
**D28 gates the 861 one**.
