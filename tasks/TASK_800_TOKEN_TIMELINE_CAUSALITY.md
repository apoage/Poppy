# TASK_800 — Trader-facing token timeline + serving-path causality audit

**Status**: ✅ **DONE — SIGNED OFF 2026-06-10.** Gate open; unblocks [[TASK_801]]/[[TASK_802]]/[[TASK_803]]/[[TASK_804]].
Follows the [[TASK_799]] grammar PASS. CPU-only, numpy. Result: `runs/800_timeline/RESULT.{md,json}`.

## VERDICT (strengthened audit, 20/20 effective, all 7 sign-off criteria hold)
| Language | Verdict | evidence |
|---|---|---|
| **L8 Haar** (window-rate) | **SERVABLE** | churn ≡ 0 over 1000 cuts; A0 byte-identical; leak-free at t |
| **L1 BPE** (variable-rate) | **SERVABLE-WITH-LAG-1** | 1000 cuts, n_churned=266, lag_hist {0:987, 1:13}, **max lag 1** (not censored vs 128); 968 stress cuts / 17,203 eligible; A0 byte-identical |

Canonical 150/150 split; single benign `asset_split_source` deviation. **The strengthened audit
overturned the thin-run's provisional LAG-0 → LAG-1** via the targeted deep-merge stress cuts (the thin
5-asset run was right in direction, wrong in bound).

### Serving-lag convention (DECIDED — feeds [[TASK_801]]/[[TASK_803]])
- **L8 inputs**: unshifted (`token_input[t] = token_id[t]`).
- **L1 inputs**: full-stream annotation **delayed 2 bars** (`token_input[t] = token_id[t−2]`). LAG-1 means
  bar t−1 is still *provisional* at decision time t, so t−2 is the last guaranteed-final bar. **"up to
  t−1" is WRONG (off-by-one).** Shift lives in the timeline-slicing layer only.
- **Targets stay UNSHIFTED** everywhere (hindsight labels — only *inputs* carry the delay; see [[TASK_802]]).

Harness lineage: `6068ef5` (build) → `17cb581` (strengthen) → `2ee1ad3` (L8 floor) → `d95db39`
(replace-not-abort + honest eligibility). Open cleanups (next commit, non-blocking): F1 dead
`--l1-stress-cuts` flag, F2 lag-hist conflates no-churn with k=0, F3 dead `n_total` param.

## The problem

`tokenizer_api.encode_asset` (`tools/candle_lang/tokenizer_api.py:272-304`) yields one token chunk
**per source window at stride=1** — consecutive chunks re-encode 63 of the same 64 bars. That is the
right object for corpus statistics (the A8 gate, `token_lm.py`) and the **wrong** object for a trader:
Poppy needs **one causal token annotation per (bar, asset)**, sliceable by the same `window_start`
indexing the encoder uses (`mcts_trader/env/jax_encoder.py:722`).

There is also a **causality hazard nobody has tested**: BPE merges and ABBA segments are
boundary-sensitive — the token identity covering bar *t* can change once bar *t+1* arrives. The gate
battery validated *transfer*; it never validated *serving-path causality at bar resolution*. This is
branch C's lookahead-leak surface and gets the same measurement-first paranoia that caught the v1
gate artifact ([[TASK_796]] §reversal, `lore/notes_2026-06-06.md` §179-180).

## Deliverable 1 — `tools/candle_lang/token_timeline.py`

```python
annotate_asset(db, asset_id, tokenizer, cfg) -> TimelineResult
# TimelineResult: ts (T,) i64 · token_id (T,) i32 · codelength (T,) f32 · valid (T,) bool
```

- **One entry per minute bar.** `token_id[t]` = id of the token covering bar *t*, computed from bars
  **≤ t only** (prefix-causal). **Convention (DECIDED, do not revisit): emit the last *completed*
  token at or before *t*.** No `<PENDING>` vocab entry; bars before the first completed token get
  `token_id = -1` with `valid=False`. Downstream ([[TASK_801]]) maps `-1`/invalid to a zero-masked
  embedding row.
- **`codelength[t]`** = bits of the current token under the BUILD bigram
  (reuse `eval_battery._bigram_codelength` machinery — do not reimplement). This is the
  "informativeness" float that [[TASK_803]] consumes.
- **Gaps**: dt≠60 → `valid=False`. No forward-fill, matching `config.py:36-37`. Day-boundary σ
  re-anchor behaves exactly as in `volunit.py:30-49` (causal, fine).
- **Languages (DECIDED — build exactly these two, in this order)**:
  1. **L1 BPE** — `numerosity_reduce` already returns the `bars` coverage map
     (`symbolic/l1_bpe.py:67-88`); token→bar alignment is half-built. Grammar PASS already in hand.
  2. **L8 Haar** (window-rate) — one token per *trailing* 64-bar window is trivially a per-bar
     causal channel. **Verify the token is timestamped at window END (`end_ts`), not start**
     (`db_stream.py:250-257`).
  - **L2 ABBA / L5 matrix-profile: do NOT build.** Write a ≤1-page feasibility memo in the RESULT.md
    (what alignment would require, expected churn behaviour) and stop. The champions-on-transfer
    question is the main thread's Step-1 scope, not this task's.
- Pre-encode contract: one pass per asset → RAM int arrays (~2 MB/asset-year). Downstream training
  loops index from RAM; **zero tokenization and zero DB I/O in any training loop** (CLAUDE.md
  anti-patterns).

## Deliverable 2 — prefix-stability (causality) audit

For ≥20 sampled assets (price-stratified, both split halves): annotate the full stream, then
re-annotate prefixes cut at sampled bars `t`, and compare:

```
churn(t, k) = [token_id at bar t-k computed from prefix ≤ t] != [same, from full stream]
```

- **PASS**: window-rate members exact (churn ≡ 0); variable-length members have a **bounded,
  measured firm-up lag** (report the churn-lag distribution; p99 lag in bars is the headline).
- **FAIL**: unbounded or large churn → that language is not servable per-bar; fall back to
  window-rate members for the trader and say so in the result.
- Audit must also assert **A0 still holds through the timeline path** (rescale a sampled asset's
  prices ×λ → byte-identical `token_id` array).

## Output

`runs/800_timeline/RESULT.md` + `.json`: per-language verdict (SERVABLE / SERVABLE-WITH-LAG-k /
NOT-SERVABLE), churn-lag distributions, A0 check, and the chosen `<PENDING>`/last-completed
convention. This decision **selects the trader language(s)** for [[TASK_801]]+.

## Implementation order (follow literally)

1. Read `tokenizer_api.py`, `symbolic/l1_bpe.py` (`numerosity_reduce` returns `bars`),
   `db_stream.py` (`iter_windows`, `end_ts`), `volunit.py`. No code yet.
2. L8 timeline first (it is the trivial case): stride-1 windows → one token per window-end bar →
   assert one id per bar over a contiguous run; gaps → `valid=False`.
3. L1 timeline: encode each contiguous dt==60 run ONCE (not per-window); walk tokens with their
   `bars` coverage; emit last-completed-token per bar.
4. Audit harness: 20 assets from `runs/796_scaled/asset_split.json` (10 build / 10 holdout,
   price-stratified); ≥50 random cut points per asset; churn matrix + p50/p99 lag; A0 rescale
   check (×1000 on one asset → byte-identical `token_id`).
5. `runs/800_timeline/RESULT.md` + `.json` via the existing `result_writer`.

Smoke (must pass before the full audit): one asset, 10k bars, both languages — timeline length ==
bar count, all ids < V, `valid` false exactly at gaps and pre-first-token bars, runtime < 60 s.

## Discipline

- Pure numpy/CPU. Workstation OK ([[feedback_no_gpu_jobs_on_workstation]] not triggered).
- Reuse `tokenizer_api`/`eval_battery`/`db_stream` — no parallel implementations.
- Commit + report; **do not run `deploy.sh`** ([[feedback_sub_agent_no_deploy]]).

## References

- `tools/candle_lang/tokenizer_api.py:272-304` (the overlap behaviour being fixed)
- `tools/candle_lang/symbolic/l1_bpe.py:67-88` (`bars` coverage), `db_stream.py:250-257` (`end_ts`)
- `mcts_trader/env/jax_encoder.py:699-752` (`window_start` slicing contract the timeline must match)
- Review that scoped this: session 2026-06-09 (tokenizer × trader architecture review)
