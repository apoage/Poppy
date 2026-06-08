# TASK_799 — Tokenised-model experiment: is there grammar beyond the bigram?

**Status**: BUILT 2026-06-08, CPU-smoke PASS (runs clean). GPU sweep PENDING on gpu-server.
First model trained *on* the candle tokens. Extends [[TASK_796]]/[[TASK_797]]/[[TASK_798]].
Code: `tools/candle_lang/token_lm.py` (commit `707bbb2`).

## The question
The whole panel so far measures the tokenizer (A0/A1/A2/A8 + the bigram axis). This is the first
experiment that puts a **model on the tokens**. The candle language is non-predictive by design
(vol-unit candle *shapes*, not returns), so this LM does **not** predict price. It asks one honest
question in the panel's own currency (bits/token = codelength = compression = transfer):

> **Is there sequential GRAMMAR in the candle language beyond the order-1 bigram?**

A small decoder-only causal transformer is trained on BUILD-asset token streams, then scored on
**held-out** cross-asset streams it never saw, against two same-data nulls plus an order control.

## Verdict rule (PASS)
```
held-out LM bits/token   <  bigram bits/token              (higher-order grammar exists)
AND
shuffled-holdout LM b/t  ≈  unigram bits/token             (the LM uses ORDER, not just marginals)
```
- **unigram** (order-0): holdout under the BUILD Laplace unigram → `eval_battery._transfer_codelength`.
- **bigram** (order-1): holdout transitions under BUILD Laplace p(curr|prev) → `_fold_with_bigram` + `_bigram_codelength`.
- **shuffled control**: the *same trained LM* evaluated on per-asset token-shuffled holdout — order
  destroyed, so a grammar-using LM must rise back toward the unigram.
- **Headline number**: the gap `bigram_bits − lm_bits` in bits/token.

Split discipline is identical to A8: vocab fit on BUILD, LM trained on BUILD streams only, every metric
measured on HELD-OUT assets. Cross-asset, no leakage. (CPU smoke at 50 steps = FAIL, as expected — it
only proves the harness runs and emits all four numbers + the verdict.)

## Which tokenizer feeds it
v1 uses the frozen **L1-BPE** tokenizer (`tokenizer_api.fit_and_save` → `CandleTokenizer.encode_asset`),
because it already produces a portable, byte-stable vocab file. **Follow-up**: re-run over the
tournament champions — **L2 ABBA (149/150)** and **L5 matrix-profile (135/150)** — by exposing their
per-asset id streams through the same loader. The grammar question is most interesting on the members
that already transfer hardest.

## Run plan (gpu-server, sandboxed, single idle GPU)
**Memory discipline is mandatory (the OOM guard):**
- **VRAM** — rely on JAX default preallocation; **never** set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, never
  the platform allocator. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` (via `--gpu-frac`, `setdefault` so the
  launcher wins). **Single device only** — pin one verified-idle GPU with `CUDA_VISIBLE_DEVICES`; the
  shared server's 2-wide-train + GPU-3-eval contract means no blind multi-GPU ([[feedback_server_vram_envelope]]).
  The script **prints the VRAM envelope before the train loop** (param count, param+optimizer bytes,
  configured fraction) — read it and confirm it fits before walking away.
- **RAM** — `safe_run.sh --cpu? no: --mem` cgroup cap; token streams load once into a RAM int array,
  minibatches indexed from RAM, zero DB/disk I/O in the loop.
- **VRAM scaling knobs** (header-documented): `context_len` is the quadratic term (attention
  `batch·n_head·context²`); `batch` and `d_model` are linear. Halve `batch`/`d_model` → ~half activation
  VRAM; halve `context_len` → ~quarter the attention term. Size the launch from the printed envelope.
- Launch in **tmux** so it survives workstation-off ([[feedback_server_launches_in_tmux]]); never bare-local
  ([[feedback_sandbox_experiments]]).

### Suggested v1 config (single 3090, well inside 0.7·24GB)
`--n-layer 4 --d-model 128 --n-head 4 --context-len 128 --batch 64 --steps 4000`
on the 150/150 split (`runs/796_scaled/asset_split.json`). Then a width/context sweep
(`d_model ∈ {128,256}`, `context_len ∈ {128,256}`) to see whether more capacity widens the
`bigram − lm` gap (real grammar) or saturates (the language is mostly order-1).

## Deliverable
A PASS/REVIEW/KILL on "grammar beyond the bigram" with the `bigram − lm` gap as the headline, the
shuffled-control gap as the order-proof, on ≥1 tokenizer (L1) and ideally the L2/L5 champions. If PASS:
the candle language has multi-token structure a sequence model can exploit — the first green light to
build a tokenised *trader* (the TASK_796 item-11 downstream), still real-vs-random gated, ≥3 seeds.

## References
- `tools/candle_lang/token_lm.py` (header documents the full design + memory model + smoke).
- `eval_battery.py` (`_transfer_codelength`, `_fold_with_bigram`, `_bigram_codelength` — reused, not reimplemented).
- `tokenizer_api.py` (frozen-encode lib), `asset_select.select_split`, `result_writer` (RESULT.md + .json).
- Frame + tournament context: `lore/notes_2026-06-06.md` §190, [[TASK_798]].
