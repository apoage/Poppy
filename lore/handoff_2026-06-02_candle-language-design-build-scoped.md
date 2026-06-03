# Handoff — 2026-06-02 — Candle-language: loss-side closed, the market gets a language (design + build scoped + audited)

**Sequence**: #9
**Previous**: `lore/handoff_2026-06-01_A2-C0-live-saferun-deploy.md` (#8)
**Branch**: `master` (tip after this handoff; artifacts commit `93d143f`)
**Working dir**: `/home/lukas/Trader`

## Session focus
A2 and C0 verdicts landed (loss-side closed null; C0 KILL withdrawn as a strawman), and the session
pivoted hard into branch C: a non-predictive "candle language" tokenizer. Two workflows produced a
locked design + a leak/OOM-audited JAX build plan; everything was written to `docs/candle_language/`,
tasks 789/791–795, lore §174–175, a Bloom draft, and two memory entries; all committed as `93d143f`.

## Confirmed rubric
```
KEEP IN FULL:
- Branch C is the active structural bet (with B — Poppy ExIt). Loss-side CLOSED null: A +0.35pp, A2
  +0.06pp (BOTH clean, no trade-rate explosion → PnL-in-loss exhausted). C0 token KILL WITHDRAWN as a
  strawman (it tested hourly / 5-asset / next-bar-SIGN-probe — none of which the real design uses).
- Design LOCKED: MINUTE, NON-predictive vocabulary, VOL-UNIT relative moves (F1 = vol-unit magnitude),
  SINGLE-asset, transfer-by-construction (one shared codebook, NO cross-asset attention in tokenizer).
  Pipeline: ① tokenizer → ② masked-token embedding (non-predictive) → ③ existing trader on tokens.
- PANEL decision: L1 BPE-delta / L2 ABBA / L3 SAX-grammar / L4 learned VQ-FSQ / L5 Matrix-Profile —
  isolated languages over ONE shared vol-unit substrate; L5 also the validator. MULTILINGUAL single
  model speaks all of them (LLM-style shared embedding, cross-language alignment with FREE supervision —
  the languages are parallel re-encodings of the same bars; shared time axis = the parallel corpus).
- Non-predictive eval battery: A0 rescale-invariance / A8 cross-asset transfer (HEADLINE; beat BOTH a
  returns-shuffle AND a random-merge same-size vocab, mega-tokens excluded) / A1 corrected MDL (charge
  null + boundary bits) / A2 reconstruction / A9 length non-degeneracy. Naive time-shuffle control is
  CIRCULAR → demoted to a control input only. Judge a language like a codebook, not a forecaster.
- JAX build: single-asset ⇒ (asset,window) is an UNBOUNDED flat sample axis (64 is NOT a ceiling);
  stream DB → bounded RAM ring → VRAM (never full-load); OOM/leak safety paramount. safe_run cgroup is
  RAM-ONLY, NO VRAM ENVELOPE → NEVER blind `--gpu 1,2,3` (per-device MEM_FRACTION + PREALLOCATE-on
  collides with the 2-wide-train + GPU-3-eval contract → OOMs the co-tenant). Posture: SINGLE-GPU on a
  verified-idle GPU; multi-GPU only when the pool is confirmed empty.
- Build is SPECCED + AUDITED but NOT WRITTEN. Target package: tools/candle_lang/ (donor =
  tools/tokenizer_c0/). 6 BLOCKING corrections (TASK_795): FSQ bound() even-level bug (only 7/8 codes
  reachable) + reachability assert; online-bincount eval (no host token-stream full-load); tail-batch
  pad+mask + n_dev divisibility; donate xb ONLY (not params/opt_state); verify history.py citations; DB
  path param + remove PREALLOCATE=false + PRAGMA quick_check.

COMPRESS + POINT:
- Full design / research / eval / build detail → docs/candle_language/ (README + 01-05 + raw JSON). Point, don't re-summarize.
- The day's narrative arc → lore/notes_2026-06-02.md §175 (+ §174 for the A2 backstory).

DROP:
- The Bloom-article false-positive saga (editor called +0.06 "fabricated"; it was real, lore was stale) beyond "article written, lore fixed".
- Workflow run IDs (w6fwtir2h / wzx1vegea / wnhmmi4a0) — completed, outputs saved into docs/candle_language/.
- Per-file git-status churn; scp/deploy retry history.

VERBATIM ANCHORS:
- commit 93d143f (today's artifacts); prior 5df6537 (#8), 936ce31 (A2).
- A2: runs/790_realized_pnl/result.md — mean +0.06pp (s7/13/23 = +0.15/+0.40/−0.36), FAIL.
- C0: runs/789_token_c0/RESULT.md — edge +0.0003 (KILL, withdrawn as strawman).
- Tasks 789/791/792/793/794/795. Bloom: bloom/drafts/48_language_of_markets.md.
- Memory: project_candle_language, feedback_server_vram_envelope.
- Pairing comparator unchanged: runs/760_train_mcts_sims/eval_baseline_paired_seed_{7,13,23}; eval-MCTS sims=16 joint.

UNCERTAIN:
- Build tools/candle_lang/ next (single-GPU-safe, 6 corrections) vs hold? (the open question to the user).
- F2/F3 = panel, F4 = transfer-driven k, F5 = coarse run-length — user may still revisit.
- B's pkl→Orbax warm-start adapter still unbuilt (the other structural bet).
- TASK_785 deploy-hygiene debt (broken git-push deploy) still open; panel code may need direct scp.
```

## Git state
```
93d143f Branch C (candle language): design, research, JAX build plan + leak audit, tasks 789/791-795, lore §174-175, Bloom draft 48
5df6537 Handoff #8: A2 + C0 live under safe_run (64G cap); A null, 780/775 closed
936ce31 TASK_790 (exp A2): realized-PnL value term in Stage-B loss (default-OFF flag)
8351f7a lore §172-173: cycle closes (780/775 null), A/B/C branches, workstation crash + safe_run
47689ee tools/safe_run.sh: hard RAM-cap (cgroup) job envelope + CPU/GPU placement
```
(Working tree still carries the pre-existing TASK_785 debt — untracked `_old/` deletions etc. — left untouched; today's commit added only the 16 candle-language artifacts.)

## Next action
Decide with the operator: **build `tools/candle_lang/`** (single-GPU-safe, the 6 TASK_795 corrections
baked in) vs hold. If building: start with `volunit.py` + `db_stream.py` + the fixed `fsq_core.py`,
smoke CPU-first on the workstation, then L4 single-GPU on a verified-idle server GPU. B's pkl→Orbax
adapter and TASK_785 deploy hygiene remain open.
