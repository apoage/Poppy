# Adversarial critique of the Stage-1 design (workflow `w6fwtir2h`)

> The critique pass that the revised design (`01_*`) answers. Kept because it documents *why* the
> design is shaped the way it is — most importantly, it caught the first synthesis silently overriding
> the operator's resolved F1 (vol-unit) decision, and proposing a circular eval gate. Verbatim.

## Verdict in one line

The spine (relative encoding → numerosity reduction → BPE → compression/transfer gates) is sound and
spec-legal. But the first draft **contradicted the one fork the operator already RESOLVED** (F1
amplitude → vol-unit, not log-return), and its headline structural gate (A3 Persist-vs-shuffle) is
**trivially passable by construction**, so the eval did NOT prove "the language is right." Two
load-bearing fixes, plus several smaller ones.

## (0) HARD CONTRADICTION with the committed spec — ignored a resolved fork

`TASK_791` records **F1 RESOLVED → (A) vol-unit magnitude**. The first draft instead encoded raw
`log(c_t/c_{t-1})` and argued invariance "for free" from log-returns, never dividing by local vol.
That makes `log(1.25)` on a placid and a frantic asset the **same** token; the operator decided they
should be **different** (magnitude relative to local regime). Worse, the draft's $225/$0.45 demo works
*either way* (a proportional rescale leaves vol-units unchanged), so it dropped a resolved decision for
no gain. **Fix:** encode `r_t / σ_local(t)`, causal local-vol. Second-order: with causal `σ`, A7
jitter-stability gets harder (σ reacts to a jittered bar), so the boundary-F1 > 0.9 threshold was
optimistic → relaxed to 0.85.

## (1) Smuggled prediction / circular eval — the structural gate is trivially passable

The designated replacement for the sign-probe was **A3: Persist self-transition KL, real vs
time-shuffled**. It is circular: numerosity reduction + BPE manufacture local self-similarity, and
minute returns/vol are autocorrelated, so a shuffled control beating the real stream is near-impossible
for **any** tokenizer — even a bad one. A3 tests "does the data have temporal structure" (known), not
"is this vocabulary good." Same error class as the sign-probe. **Fix:** replace the *time* control with
a *vocabulary* control — **random-merge BPE** and a frequency-matched random codebook; the real vocab
must beat *matched-complexity alternatives* on MDL/coverage/transfer. Demote A3 to a control input only.

## (2) Weakest link — A1's bit budget is underspecified and gameable

A1 (MDL bits-per-candle) named the headline, but a token spans a variable number of candles and 30–40%
of bars are null/fallback — if those aren't coded, A1 is rigged in the tokenizer's favor; if coded at
raw cost, A1 may never beat the baseline. **This ambiguity alone flips the go/no-go.** **Fix:** define
the bit budget over the **entire stream** — covered tokens + null bars at base-symbol entropy +
**boundary/segmentation bits** + codebook. Pick vocab size by the **transfer-stable region** (A8), not
an eyeballed MDL knee (soft/absent on fat-tailed financial data; a `p_min` floor keeps a self-paying
Zipf tail that fakes a knee).

## (3) Minute-granularity / transferability leaks

- **Gap:** `log(c/c₋₁)` is close-to-close and throws away the **open-gap**, which is large on minute
  data (illiquid alts especially). A gap-up-then-flat bar ≠ a grind-up bar. **Fix:** add
  `gap_t = log(o_t/c_{t-1})` as its own feature.
- **24/7:** "per-session re-anchoring" (imported from equities/ABBA) is undefined for crypto/FX with no
  session. **Fix:** fixed minute-cadence (UTC-day) re-anchor.
- **A8's "exclude top-k mega-tokens before measuring overlap" is the single best idea in the eval** —
  the flat/doji mega-token inflating transfer is exactly how a bad tokenizer fakes the central claim.

## (4) Cross-asset coupling — clean in Stage 1, one subtle Stage-2 smuggle

Stage 1 is coupling-free (shared codebook applied per-asset). But a bidirectional masked-token encoder
on the **pooled** stream can let context straddle asset boundaries if concatenated naively. **Fix
(footnote for 793):** masking/attention must not cross asset-sequence seams; no asset-ID conditioning.
A shared *word-level* embedding space across assets is fine; cross-asset *attention over price* is not.

## (5) What the research missed

- **Null-token semantics** were under-defined (drop-for-trader vs code-for-MDL vs placeholder-for-
  embedding conflict). → reconciled in `01_*` §2.4.
- **No token length-distribution gate** — a tokenizer can pass MDL+transfer while degenerating to
  all-singletons or one mega-token. → added A9.
- **Greedy longest-match ≠ BPE-merge-order** at serving → silent train/serve segmentation mismatch
  breaks A0. → serve with exact merge order + A0-at-serving.

## (6) Does the eval prove "the language is right"? — only after re-weighting

The honest scope (these gates prove *well-formed*, not *tradeable* — meaning is Stage 2/3) is correct.
The problem was weighting: the verdict leaned on A3 (circular) and an underspecified A1. Strip those and
what actually survives as proof is **A8 cross-asset transfer (mega-tokens excluded)** + **A2 fidelity**
+ **corrected A1 vs random-merge** + **A9 non-degeneracy**. That is the verdict rule in `01_*`.

## Fix list (priority order)

1. Encode in **local-vol units**, not raw log-returns (resolved F1). 2. Kill/fix A3 (random-merge vocab
control, not time-shuffle). 3. Fix A1's bit budget (charge null + boundary bits, full stream; baseline =
random-merge). 4. Add `gap_t`. 5. Fix drift for 24/7 (fixed cadence). 6. Define null token end-to-end.
7. Add A9 length-non-degeneracy. 8. Serve with BPE merge-order. 9. Stage-2 no-cross-asset-seam footnote.
10. Re-weight verdict around A8 + A2 + corrected-A1, drop A3.
