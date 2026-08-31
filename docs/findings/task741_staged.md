# task741 staged

> TASK_741 per-asset post-pool injection (FiLM) implemented in working tree on 2026-05-13 by a sub-agent. 10 files modified. Default OFF (byte-identical at init). NOT yet committed, NOT yet smoke-tested, NOT yet deployed to server. Smoke test + deploy is the next step before any TASK_741 experiments.

*Migrated from local agent memory 2026-08-31 (`project_task741_staged.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
The Mint (per-asset post-pool injection) implementation lives in the
`~/Trader/` working tree as of 2026-05-13 20:26-20:36. Done by a subagent
that returned GREEN verdict, with full implementation + 3 new passing
tests.

**Why:** TASK_733 found Stage A→B Val Agree caps at ~87%. Hypothesis is
that per-asset scalar features get smeared by the cross-asset temporal
pool before becoming asset-specific again. Mint moves them to a
post-pool FiLM injection (zero-init γ=0, β=0 ⇒ byte-identical at start).

**How to apply:**
- If the next session is doing TASK_741-related work, the implementation
  is already done — verify with `ls -la ~/Trader/mcts_trader/jax_agent/trm_encoder_v3.py`
  (mtime should be 2026-05-13 20:26 or later).
- 10 modified files: `trm_encoder_v3.py`, `trm_flax.py`, `muzero_model.py`,
  `easy_config.py`, `phase2_config.py`, `easy_trainer.py`, `train_easy.py`,
  `train_phase2.py`, `tools/benchmark_easy.py`, `tools/tests/test_trm_encoder_v3.py`.
- CLI flag: `--per-asset-injection {none,cheat,xs,both}` on both `train_easy.py` and `train_phase2.py`. Default OFF.
- Smoke-test BEFORE committing: `python tools/tests/test_trm_encoder_v3.py`
  should show `FiLM zero-init max diff: 0.00e+00`.
- Then Gate 1 hypothesis test on workstation (~10 min):
  ```
  python train_easy.py --preset medium --encoder-version 3 \
    --cheat-channel --per-asset-injection cheat \
    --epochs 5 --steps-per-epoch 50 --batch-size 32 \
    --levels 0 --output-dir /tmp/mint_smoke --no-checkpoint
  ```
- Only THEN commit + deploy to server.
- Server still on git HEAD `a0f7dec` (does NOT include TASK_741). Server's
  running TASK_743 multi-seed validation as of 2026-05-13 20:02 → ~22:30.

**Risk if forgotten:** A future agent could try to re-implement (waste),
or push without smoke-test (could break Phase 1 ckpt loading despite the
zero-init design). The agent's report claims byte-identical at init and
all tests passing, but the operator wants Gate 1 verified locally first.

See `lore/notes_2026-05-13.md` §98 for full file list, recovery procedure,
and pre-shutdown state capture.
