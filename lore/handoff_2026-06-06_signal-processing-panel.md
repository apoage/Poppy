# Handoff — 2026-06-06 — Signal-processing tokenizer panel + the false-null reversal

**Sequence**: #11
**Previous**: `lore/handoff_2026-06-03_candle-lang-panel-live.md` (#10)
**Branch**: `master` · **Working dir**: `/home/lukas/Trader`

## Session focus
Caught the candle-language panel's "branch-C null" as a broken metric, rebuilt the eval battery around
codelength (TASK_796), gave the tokens a model-unbound use (TASK_797 fingerprints), reframed the panel as a
signal-processing tournament and added the compression (L6 LZW) + transform (L7 SFA) columns (TASK_798),
fixed L4's codebook collapse (entropy + strided pool, 11%→85%), survived a workstation hard-crash (→ a
sandbox rule), and wrote a Bloom article. Full arc: `lore/notes_2026-06-06.md` §179–189.

## Confirmed rubric
```
Compaction rubric (chandoff 2026-06-06):

KEEP IN FULL:
- THREE experiments RUNNING on gpu-server, sandboxed (safe_run+tmux), NOT harness-tracked -> must SSH-PULL,
  no auto-notify: (a) L4 entropy-fix 150/150 sweep, tmux l4s_{strided,attn,w50} on GPU 1/2/3 ->
  runs/l4_entfix/scaled_*_RESULT.md (does the de-collapsed codebook lift L4 transfer past 23/150);
  (b) L6 LZW + L7 SFA 150/150 scorecard, tmux 798_l6 / 798_l7 (CPU) -> runs/798_scorecard/{L6,L7}_RESULT.md
  (does a compressor / Fourier coder transfer at scale, vs L1-L5).
- THE §186 VERDICT RULE: gate on the FAIR freq-matched random-codebook null on BOTH unigram AND bigram;
  the substrate controls (shuffle / random_merge) are ENTROPY-CONFOUNDED -> REPORTED, not gated. Re-derive
  the 150/150 member verdicts from the JSON's beats_random_codebook[_bigram], NOT the old verdict string.
  (This flipped "null" -> transfers: L2 149/150, L5 135/150, L1 70/150, L3 36/150.)
- L4 cure (validated, 9-asset): entropy loss on the MEAN pool makes collapse WORSE (34/256); entropy +
  STRIDED pool fixes it (218/256 = 85%). Pool-fix AND usage-pressure together, neither alone.
- Sandbox rule (standing): experiments -> server in safe_run + tmux, NEVER bare-local on the display
  workstation (it hard-crashed mid-session). See [[feedback_sandbox_experiments]].

COMPRESS + POINT:
- The whole v2-eval-repair -> token-fingerprint -> SP-panel arc -> lore/notes_2026-06-06.md §179-189.
- Workstation crash diagnosis (no OOM, hard lockup, failing nvme0; unsandboxed jobs) -> §188 + the feedback memory.
- The Bloom article -> bloom/drafts/49_the_wrong_medicine.md (drafted, fact-corrected, 1.5D-fractal aside added).
- The two research dossiers (compression-as-tokenizer / transform-domain) -> recommendations baked into L6/L7 + TASK_798.

DROP:
- The 1610-file dirty tree (pre-existing TASK_785 _old/ deletions + untracked bloom/) — NEVER sweep; commit
  ONLY tools/candle_lang + tasks/TASK_79x + lore/notes_2026-06-06 + the one article file.
- The full dmesg/journalctl dump; per-poll server status snapshots; truncated/duplicated agent reports;
  MCP disconnect/reconnect + deferred-tool noise; workflow-orchestration mechanics.

VERBATIM ANCHORS:
- gpu-server = operator@<redacted-ip>:22 ; 150/150 split = runs/796_scaled/asset_split.json.
- Running tmux: l4s_strided / l4s_attn / l4s_w50 ; 798_l6 ; 798_l7.
- eval_battery: a8_verdict (§186 gate), _fold_with_bigram, _BIGRAM_MAX_K=2048 (L6 needs --max-dict <=2048
  to keep the bigram axis available).
- Commits: dae2369 (article 1.5D) / 1cb4bff (lore§189+fix) / e09599f (L7 SFA) / 7f83d84 (L6 LZW) /
  148d3f0 (L4 entropy fix) / 5c80111 (debt close) / 4c4393c (v2 eval-battery repair).

UNCERTAIN:
- Whether 85% L4 usage HOLDS at 150-asset scale (verification was 9-asset); whether L6/L7 transfer at
  150/150 (smoke-positive only so far).
- Deferred, NOT built: zstd-dict + NCD scorers; wavelet L8 (the "fractal-native" / scale-native coder);
  the per-asset base-symbol corpus dump.
- TASK_785 deploy hygiene (git-push deploy broken -> rsync) still open.
```

## Git state
```
dae2369 article: add the 1.5D honesty (price path self-affine, D=2-H~1.5; tokenize 1-D trace of 1.5-D object)
1cb4bff lore §189 + article fix: SP tournament (L6 LZW + L7 SFA live); correct 49_the_wrong_medicine
e09599f TASK_798 L7: SFA transform-tokenizer — A0=1.0 byte-exact (no z-norm)
7f83d84 TASK_798 L6: LZW compression-tokenizer — A0=1.0, transfers (beats both fair nulls)
1a0826a lore §188: 150/150 verdict (L2 149/150, L5 135/150) + L4 collapse fix validated (11%->85%)
```
`git status --short`: ~1610 entries — ALL pre-existing TASK_785 debt (`_old/` deletions) + untracked `bloom/`.
Untouched this session. Every commit added only `tools/candle_lang/`, `tasks/TASK_79x`, `lore/notes_2026-06-06.md`,
and `bloom/drafts/49_*`. Do NOT sweep the tree.

## Next action
SSH `operator@<redacted-ip> -p 22` and pull the two running sets: `runs/l4_entfix/scaled_*_RESULT.md`
(L4 — read `used_codes` + `holdout_transfer_rate`) and `runs/798_scorecard/{L6,L7}_RESULT.md` (read
`beats_random_codebook`/`_bigram` + `holdout_transfer_rate`). Lay out the full L1–L7 tournament scorecard,
fold into TASK_796/798 + lore. Then the natural next build is the **wavelet L8** column (the fractal-native
coder the article now motivates). Everything sandboxed on the server; nothing on the workstation.
