# Factory run 33 — 2026-08-13/14 — TASK_888R and TASK_889 SHIPPED

Two rows taken and shipped; the D39 sequence's first two steps are done.
Tree clean, nothing pushed. No OPEN DECISIONS rows at start, none seeded — neither row needed a ruling.

| Row | Commit | Tag | Cycles |
|---|---|---|---|
| TASK_888R | `7cc50fb` (+ handoff `0960f60`) | `task888r-one-checkpoint-resolver` | 2 |
| TASK_889 | `f9d8ee4` | `task889-checkpoint-architecture-binding` | 3 |

**Sequence status: 888R → 889 done. Next per D39: 886 → 878 → 881.**

## What shipped

**888R** — one checkpoint resolution function feeding BOTH guard and loader at five sites, plus a loader
that refuses via object identity instead of silently starting fresh. Detail in its CHANGELOG entry.

**889** — save-time `vector_action` + `use_prediction_head` + `param_fingerprint`, a loader that REFUSES on
recorded-vs-loaded disagreement, per-leaf `trainer_metadata_<step>.json` records, a retro-stamp that infers
rather than guesses, a provenance emitter that refuses to silently rewrite tracked documents, and the config
fossil flipped. Detail in its CHANGELOG entry.

**`stash@{0}` is LEFT INTACT** — both rows took their half from it and neither dropped it.

## The lesson this run bought, and it is not the one I expected

**Both rows' BLOCKERs were outside the module under review, and both were found by running the real thing
rather than the test.**

- 888R: five lenses correctly cleared every escape hatch inside `checkpoint_io`. The hole was one line
  upstream — `train_jax_fast.py` gated entry to it on TRUTHINESS, so `--load-checkpoint ""` collapsed to
  "flag not given" and training ran on fresh weights in silence.
- 889: the retro-stamp's own test monkeypatched `infer_vector_action`, the function that fails. Only the
  tool's real bare invocation revealed it crashed on the one GPU-saved checkpoint.

**A test that stubs the thing under test cannot fail.** For the rows still ahead on this lane, the cheap
high-yield move is to run each tool's own documented invocation and each site's real CLI, not only its
suite.

**Two orchestrator directions were wrong and were caught because subagents were told to argue rather than
comply.** (1) The proposed per-leaf filename `checkpoint_<step>.trainer_metadata.json` would have been
returned AS the checkpoint — flax's `latest_checkpoint` globs `checkpoint_*` without checking the entry is
a directory. (2) "Infer `vector_action` from the payload" deserializes arrays and so depends on saved
sharding. Both were measured and corrected. Keep briefing subagents to disagree with evidence.

## Operator decisions pending — three, and the third is newly unblocked

1. **`861/smoke_e2e.py` needs ONE authorization to go fully green.** 11 legs PASS; `vii_baseline_pins`
   fails on the single leaf `/provenance/alias_union_rule`, `results` pin IDENTICAL. TASK_877's deliberate
   provenance-DESCRIPTION rewrite; no numeric leaf moved. Verified bit-identical across four full battery
   runs this session, before and after both rows. Command:
   `python experiments/861_search_teacher_imprint/smoke_e2e.py --promote-baselines`. Re-installs a tracked
   oracle, so it is not a factory action.
2. **`distill.py`'s CLI has been DEAD since TASK_874 (2026-08-08)** — `run()` unpacks 3 values from
   `imprint.load_training_panel`, which returns 5. Reproduced by three lenses; pre-existing at HEAD.
   **The 857/860 LEDGER numbers predate the break and are unaffected.** Repairing it decides whether distill
   consumes 868's `open_gate` and the alias union, which changes what a distill run measures — a ruling,
   not a patch. This is why 888R shipped gate (ii) proven at 4 sites and by direct import at the fifth,
   rather than quietly fixing one line to make its own gate go green.
3. **The `--apply` freeze on `runs/` can now lift, if you want it to.** D39 froze it "until architecture is
   recorded at SAVE time" — 889 does that, so the stated precondition is met. Nothing was applied this run;
   the 7 `trainer_metadata.json` manifest md5 `151d8524b87a96f9be771c5eab6dc244` is unchanged across 3
   cycles, 4 judges and 4 batteries, and the 4 tracked provenance documents are zero-diff. **This gates
   TASK_886's positive claim**, whose `verified: true` reads `param_fingerprint` from the frame-provenance
   document — and all 4 tracked documents still lack that field. Note the blast radius the 889 task file
   names: `eval_vector.py:819`'s `provenance_meta_block` copies every non-`frame_symbols` key into every
   artifact reading that frame, and `runs/861_ref/frame_provenance_ref.sha256` would be invalidated.

## Evidence carried forward

- **The 7 orphaned `param_fingerprint` values are CORROBORATED.** D39 recorded them as untrusted. Three
  independent agents re-derived all 7 this run — on a HEAD worktree and on the candidate — and they match
  their recorded values (`ckpt_ref_s0` → `83b8c96126fac…`; `ckpt_imprint_full` → `b93e767738c0370d…` under
  the newly-inferred architecture). They still must be recomputed, never trusted, but the evidence says
  they are correct.
- **Byte-identity is proven at unusual depth for 889:** a HEAD worktree compared all 7 real checkpoints on
  fingerprint, per-leaf raw sha256 across all 204 leaves, PopArt state, and a full `initial_inference`
  forward pass — all 7 byte-for-byte identical.
- All 117 `checkpoint_*` leaves in the repo carry `_METADATA`; 0 carry the legacy `checkpoint` file.

## Why I stopped here rather than taking TASK_878

886's contract is buildable but its positive claim is gated on decision 3 above. 878 is Ready and needs no
ruling — but the planner ratified exactly this call in run 31: *"Stopping rather than starting 878 is
RATIFIED — it rewrites eval-core execution semantics with every parity gate riding on it and carries a
3-arm ablation; starting it at the end of a run to leave it half-built would have been the worse call."*
That reasoning holds identically at the end of this run. 878 and 881 remain takeable by the next run.

## Queue state

Ready: **1 TASK_886** (contract only; positive claim gated on the `--apply` decision) → 2 TASK_878 →
3 TASK_881. GATED: TASK_880 (D34+D35). No OPEN DECISIONS rows.
