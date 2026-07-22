# TASK_854 — Prior-weight hook on Gumbel selection (D21; the last delta vs E1)

**Status**: READY 2026-07-22 (D21 ruled — operator, factory run-15 wrap).
**Size: S. Risk: medium** (search SELECTION scoring, flag-gated; default bit-identical; wired
into eval + ExIt target generation — the same path trains the policy if the re-try re-arms).

## Why (LEDGER 832/E5-inversion2 — the isolation ladder)
851's real binding leg missed the bar on all 6 operator variants (best tier21 −1.44 vs ≥ +0.74)
while tier0 search now BEATS the raw policy (+3.85 vs +1.45). The monotone ladder leaves exactly
one learned component between the shipped operator and the E1 anchor (+5.29): **the prior term
in Gumbel candidate scoring**. E1 ranked candidates by pure true-rollout Q with no prior; the
imprint prior is cloned from a ~2%-hold ladder teacher and votes for trading, driving
fee-domain trajectory churn (hold 55–68%). The knob to test this (`--value-scale`) is
CLI-accepted but INERT in the 835 search path — the probe came back byte-identical.

## Scope
1. `--prior-weight <float>` on the Gumbel scoring path (score = prior_weight·logits + g +
   σ(Q)): default **1.0 = today, bit-identical**; 0.0 = pure-Q = the E1 selection. Wire into
   the 835 eval operator AND training self-play target generation (one implementation, both
   consumers — mismatched scoring between target-gen and eval is the cross-env class of bug
   the gates exist to prevent).
2. The inert `--value-scale`: either wire it truthfully into σ(Q) scaling or REMOVE it and
   fail-loud on use — a CLI flag that silently does nothing is a verification hazard
   (tonight's pure-Q probe burned a GPU run on it).
3. Selftest: scoring property tests (prior-weight 0 ⇒ ranking invariant to logits; 1.0 ⇒
   byte-identical scores to today; monotone interpolation between).

## Acceptance (gates)
- (i) **The inversion re-run, fixture leg**: frozen fixture ckpt, true-env leaf, no tail,
  hb3.0, `--prior-weight 0.0` — search-eval preserves-or-beats the policy replay on the same
  rows (directional). The REAL binding leg (frozen `runs/832_e5/ckpt_imprint_full`, bar
  ≥ +0.74 tier21; honest expectation E1-class, and a MISS falsifies the isolation ⇒ back to
  diagnosis, no retune-and-retry) = planner's scheduled run.
- (ii) Default bit-identical (canon + existing 835 result JSONs reproduce; SHAs).
- (iii) Selftest green incl. the new property tests; medium panel (correctness + regression;
  flag-wiring lens for the value-scale cleanup).

## After ship (planner-owned, pre-registered)
1. Real inversion at prior-weight 0.0 → LEDGER.
2. PASS ⇒ small eval sweep (prior-weight {0, 0.25, 0.5} × R {3, 6}, frozen imprint) picks the
   ExIt operator → the Amendment-6 ExIt re-try RE-ARMS with it (3 seeds × 300 ep,
   prior-survival read, D11 watchdog).
3. MISS ⇒ isolation falsified ⇒ new DECISIONS row with the residual-delta candidates
   (visit-vs-Q final selection, trajectory path-dependence of the dithered book).

## Out of scope
Changing K/sims/gumbel_scale defaults; the OOS/853 track (independent); any training spend
before the real inversion read passes.
