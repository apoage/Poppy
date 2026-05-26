# TASK_780 — Smoke Protocol (operator-facing)

**Use this protocol after a real Phase-A bridge training (~165 min/seed server wall)
to decide whether to authorize Phase-B Stage-B.** It encodes the Phase-A sanity
gate per `AUTHORIZATION_2026-05-26.md` §"Phase-A sanity gate" and the load-
contract gates per `tasks/TASK_780_IMPLEMENTATION_PLAN.md` §"Phase 3".

## When to run

After **each** Phase-A bridge training seed completes
(`runs/780_oracle_as_mcts/780A-default_s{7,13,23}/phase1_best.pkl`). Before
authorizing the matched Phase-B Stage-B retrain. The smoke runs on workstation
GPU 0 in <30 minutes per seed (3 seeds × ~30 min = ~1.5 h total).

If the smoke fails on **≥ 2 of 3 seeds**, write `result.md` with the negative-
but-informative verdict per AUTHORIZATION §"Phase-A sanity gate" and STOP.

## Prerequisites

- `runs/780_oracle_as_mcts/canon_imprint_baseline.json` exists, with per-regime
  Bal.Acc + epoch from `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
  per AUTHORIZATION §"Pre-implementation checklist" item 4.
- Workstation has `trader_env/` activated.
- All three Phase-A seeds (`780A-default_s7`, `s13`, `s23`) have written
  `phase1_best.pkl` + `config.json` to their respective `runs/` subdirectories.

## Step 1 — Phase-B bootstrap integration smoke (per seed)

This verifies the bridge ckpt loads cleanly into the Phase-2 pipeline (Gates
i–v + Phase-4 encoder-lr wiring per `TASK_780_IMPLEMENTATION_PLAN.md` Phase 3).

```bash
# Run from repo root with trader_env activated.
for s in 7 13 23; do
  echo "=== seed $s ==="
  python tools/smoke_780_phase_b_bootstrap.py \
    --bridge-ckpt runs/780_oracle_as_mcts/780A-default_s$s/phase1_best.pkl \
    --canon-ckpt  jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl
done
```

Each invocation prints 6 gates with PASS/FAIL. Acceptance:

- Gates **i, ii, iii** must PASS on **all 3 seeds**. These are load-contract
  invariants — a FAIL signals a structural bug in the bridge trainer's pickle
  format (not a training-time issue).
- Gate **iv** (consistency_loss bridge ≤ 2× canon) must PASS on **≥ 2/3 seeds**.
  Per `TASK_780_IMPLEMENTATION_PLAN.md` §"Phase 3" the *strong* form is
  `consistency_loss < 0.01`; the smoke uses a relative comparator because
  absolute numbers depend on the encoder distribution. If gate iv fails on
  ≥ 2 seeds → likely the bridge dynamics has drifted; halt and triage.
- Gate **v** (latent_std parity) must PASS on **≥ 2/3 seeds**. A FAIL on
  ≥ 2 seeds signals the bridge training pushed the encoder distribution far
  from canon, which Stage-B will struggle to recover from. Halt and triage.
- Gate **vi** (Phase 4 CLI wiring) is a *static* CLI check; PASS on all 3
  seeds is required. A FAIL would mean the `--encoder-unfreeze-epoch` flag
  regressed since this protocol was authored.

## Step 2 — Phase-A sanity gate (per seed)

This is the canonical Phase-A sanity gate from AUTHORIZATION §"Phase-A sanity
gate": **bridge per-asset Bal.Acc ≥ canon − 3pp** on ≥ 2/3 seeds.

```bash
for s in 7 13 23; do
  python tools/smoke_780_phase_a.py \
    --bridge-ckpt runs/780_oracle_as_mcts/780A-default_s$s/phase1_best.pkl \
    --canon-ckpt  jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
    --output      runs/780_oracle_as_mcts/sanity_gate_s$s.csv \
    --levels 0 3 \
    --n-steps 4096 \
    --n-subset 1024 \
    --threshold-pp 3.0
done
```

Notes on parameters:

- `--levels 0 3` evaluates on Level-0 (sine) and Level-3 (noisy trends).
  These cover both the trivial momentum regime and a noisy-but-tractable
  regime. The full 6-DB-period benchmark is the *Phase-B* gate, not this
  one — Phase A is a *classifier-side* check.
- `--n-steps 4096` + `--n-subset 1024` keeps each seed under 10 minutes on a
  single 3090. Raise to 8192 / 2048 for a slower but more reliable check.
- `--threshold-pp 3.0` is the AUTHORIZATION-specified gate.
- The script exits 0 on PASS, 1 on FAIL (bash `$?` captures it for scripting).

Each invocation writes a CSV: `asset_id, regime, canon_balacc, bridge_balacc,
delta_pp` for every (asset × regime) pair, plus a console summary table.

## Step 3 — Aggregate verdict (3-seed combine)

After all 3 seeds have written `sanity_gate_s*.csv`:

```bash
python -c "
import csv, glob, sys
from pathlib import Path
ALL = []
for path in sorted(glob.glob('runs/780_oracle_as_mcts/sanity_gate_s*.csv')):
    seed = path.rsplit('_s', 1)[1].split('.')[0]
    with open(path) as f:
        for row in csv.DictReader(f):
            row['seed'] = seed
            ALL.append(row)
# Per-seed mean delta_pp
by_seed = {}
for r in ALL:
    by_seed.setdefault(r['seed'], []).append(float(r['delta_pp']))
n_pass = 0
print(f'{'seed':<6s} {'mean_delta_pp':>14s} {'min_delta_pp':>14s} {'verdict':>10s}')
for seed, vals in by_seed.items():
    m = sum(vals)/len(vals)
    mn = min(vals)
    verdict = 'PASS' if m >= -3.0 else 'FAIL'
    if verdict == 'PASS': n_pass += 1
    print(f'{seed:<6s} {m:>+13.2f} {mn:>+13.2f} {verdict:>10s}')
print()
print(f'Overall: {n_pass}/3 seeds PASS')
print(f'AUTHORIZATION gate: ≥ 2/3 seeds with mean delta_pp ≥ -3pp')
sys.exit(0 if n_pass >= 2 else 1)
"
```

PASS on ≥ 2/3 seeds → proceed to Phase-B Stage-B (canon a1b1 retrain off the
bridge ckpts; see AUTHORIZATION §"Once-authorized variant matrix" Phase-B).

FAIL on ≥ 2/3 seeds → write `runs/780_oracle_as_mcts/result.md` with the
negative-but-informative verdict (which gate, which seeds, decomposed Bal.Acc
per regime), STOP, and trigger the cancellation trigger #1 from AUTHORIZATION
§"Cancellation triggers during build". Per `project_classifier_alpha_anti_correlation.md`,
this is the **leading anti-correlation indicator** — diagnose loss shape
(likely a class-balance issue in the visit distribution) before paying
Phase-B cost.

## Step 4 — Document the verdict

Append to `lore/notes_YYYY-MM-DD.md`:

- 3-seed verdict (PASS/FAIL/MIXED).
- Per-seed mean delta_pp + min delta_pp.
- Decision: proceed to Phase-B / halt / re-train.
- If halted, list which gate(s) failed and the diagnosis hypothesis.

The CSVs in `runs/780_oracle_as_mcts/sanity_gate_s*.csv` are durable artifacts
for follow-up analysis (per-asset breakdown lets us identify whether failure
is global vs concentrated on a small subset of assets).

## Tool reference

- `tools/smoke_780_phase_b_bootstrap.py` — Phase 3 + 4 bootstrap smoke (6 gates).
- `tools/smoke_780_phase_a.py` — Phase 5 Phase-A sanity gate helper.
  Synthetic prices via `mcts_trader.data.synthetic_market.SyntheticMarket`;
  optional `--use-bella` reserved for a future dispatch (TASK_780 Phase 5
  ships the synthetic path only — DB-period eval is the heavier Phase-B
  benchmark via `tools/benchmark_easy.py`).

## See also

- `runs/780_oracle_as_mcts/AUTHORIZATION_2026-05-26.md` §"Phase-A sanity gate".
- `tasks/TASK_780_IMPLEMENTATION_PLAN.md` §"Phase 3" / §"Phase 4" / §"Phase 5".
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` —
  leading anti-correlation indicator; the Phase-A sanity gate is the cheap
  exit if 780 reproduces the 737/738/741 pattern.
