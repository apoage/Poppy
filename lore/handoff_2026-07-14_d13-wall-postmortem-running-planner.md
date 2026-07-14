# Handoff 2026-07-14 — D13 wall, 844 postmortem running (planner #8)

**Previous:** `lore/handoff_2026-07-14_factory-run9-short-bias-probe.md` (factory #9; planner
chain: `lore/handoff_2026-07-09_phase1-battery-planner.md`)
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session role:** PLANNER/orchestrator — queue/decisions/specs + scheduled server runs; the
factory (separate session) executes queue rows; lane guard + git gate enforce while armed.
**Session focus:** the env-fix arc completed and measured — phase-1 (env-wall FAIL) → 839
budget step (p1b: nets learn/env kills) → 840 weight env (p1c: env FIXED, policy below floor,
short bias) → D13 fork seeded; 844 postmortem instrument shipped (factory run 9) and its real
run launched on all checkpoints.

## Compaction rubric

```
Compaction rubric (chandoff 2026-07-14):

KEEP IN FULL:
- Role contract: this session = PLANNER (queue/DECISIONS/SPECs/scheduled server runs); factory
  = separate session executing queue rows; lane guard + git gate block planner code-lane writes
  and commits while ARMED (factory currently DISARMED).
- LIVE STATE: TASK_844 short-bias postmortem probes RUNNING on gpu-server (tmux q_844, GPUs
  2/3): 4 jobs — p1c s0/s1/s2 + p1b s1 cross-env control; outputs
  runs/828s3p1_instruments/844_bias_p1c_s{0,1,2}.json + 844_bias_p1b_s1.json (server-side;
  pull after). Watcher bp0mh3d0q (5-min cadence). ON COMPLETION: read locus verdicts — probe-1
  value-locus uses the CALIBRATED z (fresh-init null), NOT raw t; probe-2 flat-book marginals
  (sell/buy ratio >1.5 = policy-prior locus); probe-3 credit_misassign gated on the
  action-attributed subset only — then fold into the D13 row and report; the ruling stays with
  the operator.
- D13 OPEN — THE ONLY WALL (operator call, overrides a pre-registration): "redirect" (E5/832
  imprint→ExIt path, planner recommendation; DEC-2 bar ε=+0.10 over floor) vs "follow D10"
  (H6 action-ladder widening). Everything downstream waits: 832-weight-compat task (not yet
  written), H6-widening task (not yet written), TASK_843 netting cleanup (gate = phase-1c done
  — satisfiable, but grooming deferred until D13).
- Phase-1c SPLIT VERDICT (3/3 seeds): env FIXED — first-ever complete trained-policy rollouts,
  1670/1670, ZERO bankruptcies, battery closed clean under the armed D11 watchdog; policy FAIL
  — below floor 3/3 (net@21 −0.827/−1.115/−1.385 vs floor +0.424), persistent ~6-7L/~8-10S
  short book, profits ONLY in bear22w. Same short bias as p1b ⇒ systematic to the learning
  path (objective checked: absolute return; reward-hybrid = loss recipe, NOT market-relative).
- Instrument caveats that gate future reads: 834/DEC-1 is PASS-but-VACUOUS in weight semantics
  (R_true≈1e-10 — deadbanded re-trims make per-step reward action-blind; calibrated gate
  uninformative there; informative in order-semantics envs). 836 weak-mixed on p1c (1/5, 2/5,
  1/5 vs the ≥3/5 bar).
- Wedge root cause (3 occurrences): py-spy stack on live wedge #3 = MainThread stuck in
  jax pxla.__call__ executing the self-play lax.scan (self_play.py:742) — CUDA/XLA dispatch
  hang post-checkpoint-save; orbax-GC-vs-RLock hypothesis REFUTED; D11 watchdog
  (--stall-timeout 2700, opt-in) armed on all training launches — p1c ran clean under it.
- Decisions ledger: D7-D12 all DECIDED (D12 = accept short over-tax, cleanup = TASK_843
  gated); D13 OPEN. Env-fix chain shipped: 839 budget → 840 weight rebalance (Σ|w|≤1,
  bankruptcy impossible) → 842 canon close-fee (fee0 bit-identical; un-gates phase-2 fee16;
  honest E1 fee16 edge −0.134).
- Ops: fresh server tmux needs `source trader_env/bin/activate` (bare python = Py2); py-spy
  into the venv via `~/.local/bin/uv pip install --python ~/Trader/trader_env/bin/python
  py-spy`, run under sudo (ptrace_scope=1).

COMPRESS + POINT:
- p1/p1b/p1c battery + instrument-pass mechanics → tasks/QUEUE.md scheduled-runs table; all 28
  verdict JSONs pulled LOCAL to runs/828s3p1_instruments/.
- Factory runs 6-9 ship details → QUEUE.md shipped-log + task banners (839/840/841/842/844).
- Wedge saga → TASK_841 evidence note + lore/notes_2026-07-12.md.
- All experiment verdicts → experiments/LEDGER.md rows (828s3/env-wall, 835/S4-p1{,b,c},
  834/E3-p1{b,c}, 836/E4-p1{b,c}).

DROP:
- ssh/tmux quoting mechanics, manifest/config-json authoring detail, watcher shell
  construction, deploy push logs, /usage-/effort noise, py-spy install fumbling.

VERBATIM ANCHORS:
- Corridor floor +0.424 / ref +0.751 / ceiling +29.698 net@21; p1c −0.827/−1.115/−1.385.
- tmux q_844 + watcher bp0mh3d0q; outputs 844_bias_p1c_s{0,1,2}.json, 844_bias_p1b_s1.json.
- Ship tags task839-budget-vector-step … task844-short-bias-probe; HEAD 744fa47; server deploy
  fc13509.
- DEC-2 ε=+0.10 (the E5 bar if "redirect"); D10 trigger text "≥2/3 below floor ⇒ H6 widening".

UNCERTAIN (operator: correct me):
- Sequencing: fold the 844 locus verdicts into D13 context BEFORE the operator rules
  (recommended), vs rule D13 immediately on return — confirm preference.
- p1b-s0: no further retry planned (verdict stands on s1/s2) — confirm.
- The full runs/+logs pull (background job b3ai8wtyg) completion unconfirmed — verdict JSONs
  were pulled directly and are local; re-run tools/deploy/pull-results.sh if gaps found.
```

## Git state
```
744fa47 QUEUE + handoff #11: TASK_844 shipped (factory run 9) → Ready empty behind D13 wall
b53ecfe TASK_844: short-bias postmortem probe (read-only diagnostic instrument)
800f015 p1c s2 record folded: split verdict confirmed 3/3; battery closed clean under watchdog
f3e9b3b p1c verdicts: env FIXED / policy below floor with systematic short bias; D13 OPEN
ef8f913 Wedge #3 stack captured (XLA dispatch hang); D12 ratified; TASK_843 seeded gated
```
Status: clean except untracked `experiments/844_short_bias/` artifacts if the factory left any
(check `git status` on resume). Nothing pushed (owner pushes after review).

## Next action
Watcher bp0mh3d0q fires → pull + read the four 844_bias_*.json locus verdicts (calibrated-z /
marginals-ratio / gated credit_misassign) → append LEDGER row(s) + fold the locus into the D13
OPEN row → report to operator for the D13 ruling ("redirect" vs "follow D10") → on ruling,
seed the chosen track's task row(s) (832-weight-compat + imprint-sanity schedule, or
H6-widening) + un-gate TASK_843 grooming. Factory resumes when new Ready rows land.
