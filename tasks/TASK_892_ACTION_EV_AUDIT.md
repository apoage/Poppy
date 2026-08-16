# TASK_892 — action-EV audit: does any policy on this recipe have positive-EV actions?

**Status**: READY 2026-08-16 (planner/orchestrator; the research row the queue was missing).
**Size: S. Risk: LOW to build (new instrument module + one flag-gated operator), HIGH in
consequence** — its verdict seeds D42, the pivot ruling for the whole post-887 line.
**Sequenced AFTER TASK_886, TASK_890 AND TASK_878** — 886/890 touch `eval_vector.py` and the
distill surface (this row adds one flag to `eval_vector.py` and must not race them), and 878
re-specifies the eval-core execution semantics this audit measures under, so the audit must be
built and read on post-878 semantics. **The pre-registered
experiment design is `experiments/892_action_ev/SPEC.md` — it is the contract; this file is the
build row.**

## Thesis

887 proved the hold collapse is adaptive: at zero costs, more activity = monotonically more loss.
What 887 could not distinguish: whether the policy's non-HOLD actions carry **no edge** (the
collapse is a correct refusal of worthless actions) or carry edge the policy fails to **find**
(the collapse is a search/selection failure). The two readings demand opposite pivots — reward/
data/edge structure vs policy machinery — and the distinguishing measurement is cheap: replay the
policies, and compare realized forward returns after their non-HOLD decisions against a
**matched-activity random null** that sees the same states at the same costs.

## Scope

1. **`--decision-rule random-matched`** on the 835/848 eval operator (`eval_vector.py`,
   flag-gated, default OFF = byte-identical, proven against HEAD): per decision, each asset's
   action drawn from the SUBJECT policy's own marginal per-asset action distribution (the
   marginals recorded from a prior policy walk, loaded from a sidecar JSON; fail-loud if the
   sidecar's universe/frame does not match the walk's). Seeded and path-independent — the same
   seed reproduces the same null walk bit-for-bit.
2. **The trace instrument** (`experiments/892_action_ev/`, new module, 849-style): per-decision
   record of taken action, position delta, and realized forward H=21 net return (env accounting;
   PV conservation asserted exact per walk). Emits one JSON per walk with the universe_pin
   record inline (886's mechanism-typed record — first consumer).
3. **The comparison report**: per-regime-bucket matched means, block-bootstrap CI (block 21,
   10k resamples), the Δ quantity and its band verdict per the SPEC, plus the instrument-fail
   checks (null activity mismatch > 0.5pp, < 30 non-HOLD decisions per seed, PV violation).
4. Fixture smoke: a synthetic fixture walk where the null's marginals are constructed known —
   the report must recover a pre-registered sign (e.g. a fixture policy hard-coded to buy before
   up-moves must read EDGE on the fixture).

## Acceptance (gates)

- (i) Default OFF: every scored block byte-identical vs HEAD (orchestrator-reproduced).
- (ii) The null is matched by construction: activity-rate and long/short-mix deltas vs the
  subject walk ≤ 0.5pp, reported in the artifact.
- (iii) Path-independence proven: same seed → bit-identical null walk; different seed → different
  walk, same marginals.
- (iv) The fixture smoke recovers the planted sign; a shuffled-trace falsification (targets
  permuted) reads non-EDGE.
- (v) No test monkeypatches the function under test (run-33 rule); the null operator's documented
  CLI invocation is run, not stubbed.
- (vi) Regression tests for (i)–(v), mutations independently rebuilt.

## Out of scope

The real leg (scheduled workstation run — planner lane; ~84 evals, no training). Any verdict
past INSTRUMENT GREEN: the EDGE/ANTI-EDGE/NULL ruling is the planner's, seeded as D42 with the
LEDGER row. The secondary search-operator subject and the imprint_full contrast (both are
planner add-ons on the shipped instrument). Training anything.
