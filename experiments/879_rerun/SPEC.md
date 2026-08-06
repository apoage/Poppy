# 879/rung3-distill-remeasure — SPEC (PRE-REGISTERED 2026-08-06, before launch)

Rung 3 of the TASK_879 re-run ladder, executing the operator's standing directive:

> "if there is experiment on known broken framework we shall rerun experiment on fixed
> framework .. because experiment might get fixed"

Planner-scheduled (not a factory row). D17 applies: benchmark read, **no gate status**.

## Question

Do the two search-leaf distillation FAILs reproduce on the fixed framework?

| verdict | date | number | epoch measured on |
|---|---|---|---|
| `857/vhat-corridor` | 2026-07-26 | best **−4.23** vs floor +0.424 | pre-E-c |
| `860/codistill-corridor` | 2026-08-01 | best **−4.02**; deployed-Q +0.36 (bar ≥ +0.5) | pre-E-c |

Both concluded "residual structural in the dynamics embedding". Both were measured before the
identity split was repaired (E-c), before UTC windows + the masked daily floor (E-d), and before
the zero-price→fill change (E-b). Under the epoch rule those verdicts are **PROVISIONAL**.

## Design — EVAL-ONLY, same checkpoints, same command lines, new code

The re-measure changes exactly ONE thing: the framework. No re-distillation, no re-training, no
new hyper-parameters, no window change.

- **Checkpoints (unchanged, already on srv):** `runs/857_vhat/ckpt_distilled_s{0,1,2}` ·
  `runs/860_codistill/ckpt_codistill_s{0,1,2}`.
- **Command lines:** taken VERBATIM from the recorded manifests
  (`runs/857_vhat/manifest_857_vhat.txt`) — `eval_vector.py --operator search --leaf-eval
  vhat-distilled --prior-weight 0.0 --vector-proposal-scheme dithered-book --vector-k 16
  --vector-hold-bias 3.0 --vector-dither-scale 1.0 --sims 32`, window
  `2021-01-01 .. 2025-12-01`.
- **Attribution baseline:** `vd_pre` — the SAME operator on the UN-distilled
  `runs/832_e5/ckpt_imprint_full`. Without it, "the distill still fails" is inseparable from
  "the framework moved every number".
- **Code:** srv `training` @ **b695487** (2026-08-03 deploy) — carries E-a (cash wall), E-b
  (zero-price→fill), E-c (alias union + frozen-cell), E-d (UTC + masked daily floor). NOT E-e
  (877) or E-f (878), which are still queued; that limitation is part of the read, not a defect.
- 7 jobs, GPU pool 1,2,3 via `tools/deploy/run_queue.sh` (per-GPU advisory locks, live idle
  check, RAM cgroup cap, `--gpu-frac 0.7`), inside tmux.

Because seed, checkpoint, operator, window and command are all held fixed, **the per-seed delta
against the original number IS the framework's contribution** to that FAIL. That is the whole
point of the design and the reason this is worth GPU hours rather than an argument.

## Bar — pre-registered, stated BEFORE any result is seen

Primary metric: per-lane MEDIAN of the 3 seeds' `net@21`, compared against the floor **reported
by these same runs** (the floor may itself move under E-d's masked walk — report both the old
+0.424 and the newly measured one, and never silently substitute one for the other).

Three graded outcomes, decided in advance:

- **(i) FRAMEWORK ARTIFACT** — either lane's median clears its own re-measured floor. Then the
  pre-E-c FAIL was an artifact, the parked distill lane (D26) un-parks, and the old LEDGER rows
  are superseded with an explicit "measured on a since-fixed framework" note.
- **(ii) FRAMEWORK-SENSITIVE, STILL FAIL** — medians stay below floor but move by **> 1.0**
  net@21 at the median. The VERDICT stands; its MAGNITUDE does not, and no argument may quote
  −4.23/−4.02 again.
- **(iii) STRUCTURAL, CONFIRMED** — every seed moves by **≤ 1.0** and stays below floor. The FAIL
  is framework-independent; the lane stays parked, now with a fixed-framework stamp. This is the
  outcome that would let the program stop re-litigating the distill lane.

Secondary (reported, not gating): the deployed-Q calibration re-read (857 +0.03 → 860 +0.36,
original bar ≥ +0.5).

## Hard checks — a STOP, not a grade

Zero bankruptcies · stale tails exactly `[14,41]` · not-real cell share ≈ 9.74% · alias/frame
disclosure present in every artifact. **Materially different ⇒ STOP and investigate; do not
grade a run whose panel disclosure does not match the 861-chain baseline.**

Known deviation risk, disclosed in advance: the recorded command carries **no frame pin**, so it
reads the canon unpinned loader — exactly as the original did, which is what preserves
comparability. If a post-868/869 refusal now blocks that command, **that refusal is itself the
finding** (the old experiment is no longer runnable as specified); the re-run is then re-issued
WITH `--panel-frame-from-checkpoint runs/832_e5/frame_provenance_imprint.json` and the deviation
recorded in the LEDGER row rather than quietly absorbed.

## Launch note (recorded so it stops recurring)

The first launch failed all 7 jobs in ~30 s with `SyntaxError: Non-ASCII character '\xe2'` —
`run_queue.sh` does not activate the venv, so a bare `python` in a manifest resolves to
**python2** on gpu-server. Manifests must name the interpreter explicitly:
`trader_env/bin/python` (3.10.20). This has now cost two launches (the 861 chain and this one);
the real fix is for `run_queue.sh` to resolve the venv itself, which is queued as a recommended
row rather than patched here. Failing fast and loudly was the harness working correctly.

## Not in this leg

Rung 1 (828s3 phase-1 self-play — the long base-training block) stays gated on TASK_876's score
emission, because a multi-day run whose checkpoints carry no scores produces exactly the garbage
weights the operator ruled against keeping. Rung 2 (ExIt) stays gated on TASK_878. Rung 4's
fixture-scale 851 legs are cheap and independent and can fill the pool afterwards.
