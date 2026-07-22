> **STATUS (2026-07-22): SHIPPED as `task853-inactive-mask-oos-panel`** (factory run 15). Flag
> `--panel-frame-from-checkpoint` default OFF = bit-identical; fixed 64-slot frame in checkpoint
> order, zero-coverage + partial(pre-listing) slots INACTIVE, all spans dual-disclosed (JSON +
> stdout), fail-loud on every identity-scrambling condition. Panel: correctness + fail-closed +
> regression + flow-verifier (2 rounds); no BLOCKER/MAJOR; fix pass closed 2 MINOR (duplicate-symbol
> guard; partial-slot pre-listing leak — now flat-zero + price-validity force-HOLD, both operators)
> + 1 NIT (empty-flag guard), all re-verified; selftest 10/10; OFF bit-identical (SHAs on both trees).
> **Remainder (planner-owned, scheduled real-DB run):** acceptance (i) numeric (+0.839 ±0.06 on
> `runs/832_e5/ckpt_imprint_full`) + the binding ~240-decision OOS benchmark (LEDGER 832/E5-oos,
> BENCHMARK only per the D17 operator correction) — needs a provenance JSON with the imprint's 64
> training symbols in recorded order; the fixture carries neither the trained ckpt nor the window.

# TASK_853 — Fixed-frame daily panel with inactive masking (the last OOS-read mechanic)

**Status**: READY 2026-07-22 (planner seed; D20-scoped daily source, D17 Track 2).
**Size: S. Risk: low-medium** (eval panel loader only; zero training-path code; the danger this
task exists to avoid is SILENT positional identity scrambling).

## Why (tonight's fail-loud chain, LEDGER 832/E5-oos → D20)
The imprint checkpoint's encoder FLATTENS assets (kernel 1728 = 27 feats × 64 slots): the model
is hard-locked to the 64-slot training universe, POSITIONALLY — slot k IS asset k. In the OOS
window only 36/64 assets have real daily coverage; the 852-fixed loader correctly refuses to
densify (28 fake columns) AND a 36-asset sub-panel both mis-shapes the net and would scramble
slot identities (two fail-louds did their job tonight). The correct mechanic — same as the
TRAINING panel's handling of mid-window listings (SOL et al.) — is: keep the checkpoint's
64-slot frame in its EXACT training order, and mark zero-coverage assets INACTIVE.

## Scope
1. `--panel-frame-from-checkpoint` (or equivalent) on the 848 replay path: build the daily
   panel on the checkpoint's full symbol list in its RECORDED training order (the cfg/symbols
   provenance saved with the ckpt — fail-loud if absent), one column per slot.
2. Zero-coverage (or partially-covered) slots: inactive per the env's EXISTING active-mask
   semantics (no fills, no fees, excluded from n_present renormalization) — reuse, do not
   reinvent; assert at build time that every inactive span is reported (symbol + span) in the
   JSON and stdout, so a benchmark on 36/64 live slots is IMPOSSIBLE to mistake for one on 64.
3. Teacher floor (`--teacher-floor`): same panel, same masks — the D20 within-panel scoping.
4. Fail-loud stays primary: unknown symbols, order mismatch vs checkpoint provenance, all-slots
   -inactive, non-1440 bars.

## Acceptance (gates)
- (i) Round-trip: on the TRAINING window (minute-covered), the fixed-frame daily panel replay of
  `ckpt_imprint_full` reproduces the known policy-argmax behavior class (dense book, net@21
  within the D20 ±0.06 discretization envelope of +0.839) — slot identities proven intact.
- (ii) OOS smoke: 2025-07-03..2026-06-30 builds 64 slots / 36 active, inactive spans printed,
  rollout completes, floor computed on same panel.
- (iii) Defaults bit-identical (no flag = today's behavior); selftest green; small panel
  (correctness + fail-closed lenses).

## After ship (planner-owned)
The binding OOS benchmark, third attempt: fixed-frame daily panel, ~240 decisions, 36 active
slots documented → LEDGER 832/E5-oos (binding; replaces the n=31 preliminary; BENCHMARK only
per the operator's D17 correction).

## Out of scope
Universe redefinition / retraining on the 36; minute backfill; anything in the search/leaf path
(that is the 851 chain, running independently).
